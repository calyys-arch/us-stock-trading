"""
IB tick-by-tick + Level 2 depth capture -> local JSONL archive
(data/ticks/ + data/depth/), feeding the tick/order-book-based signal-trap
heuristics in python/signals/trap_detector.py (stop-hunt/marking-the-close
need trade prints; order-book churn/spoofing-layering proxies need depth
snapshots — daily bars can't see either).

Report-only scope (user decision, 2026-07-28): this module ONLY records
market microstructure to disk for later offline analysis. It never touches
the trading engine, never gates a signal, and runs on its own
`tick_capture_client_id` (configs/broker.yaml) so it can't collide with the
live feed/broker sessions.

What gets captured, per subscribed symbol:
  - tick-by-tick trades ("AllLast": every print, with exchange + conditions)
  - tick-by-tick best bid/ask ("BidAsk")
  - Level 2 depth updates (reqMktDepth: insert/update/delete per book level)

File layout (two separate roots, per plan): trades/bidask go to
data/ticks/<SYMBOL>/<YYYYMMDD>.jsonl; L2 depth goes to
data/depth/<SYMBOL>/<YYYYMMDD>.jsonl. One JSON object per event,
`recorded_at` wall-clock added at write time. Files rotate at the ET date
boundary. Both roots are gitignored (regenerable-only-forward runtime data —
ticks not captured today are gone forever, which is exactly why this
archiver should run continuously during RTH).

RTH discipline: by default events outside regular trading hours are
dropped (python/core/calendar.py) — pre/post-market books are thin and
would dominate churn statistics with noise. `--include-extended-hours` on
the capture script disables the filter.

IB constraint worth knowing: Level 2 depth needs market-data subscriptions
on the account, and IB caps concurrent depth subscriptions (typically 3 for
a basic account). The capture script therefore defaults depth to the FIRST
`max_depth_symbols` symbols only, logging which ones.

Resilience (2026-07-29 incident): a real overnight run showed IB Gateway can
report "Connectivity between IBKR and Trader Workstation has been lost"
(error 1100) for hours while the local API socket stays technically open,
then eventually get closed by the peer. ib_async's default `RequestTimeout`
is 0 (wait forever), so a `qualifyContracts()` call issued while the
sec-def data farm is down would hang indefinitely — the process sat idle
for ~2h before the socket finally dropped and crashed with an unhandled
`ConnectionError`, capturing zero data all night. `run()` now: (1) sets a
bounded `RequestTimeout` so a stuck request raises instead of hanging,
(2) wraps each connect+subscribe+stream session in an outer retry loop with
backoff, reconnecting on any disconnect/exception rather than exiting the
process. `stop()` still ends the process for good.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ..core.calendar import is_regular_trading_hours

log = logging.getLogger(__name__)

TICKS_DIR = Path("data/ticks")     # trades + bidask (per plan's file layout)
DEPTH_DIR = Path("data/depth")     # Level 2 depth snapshots
DEFAULT_DEPTH_ROWS = 10

# Which root each event `kind` rotates into.
_KIND_TO_ROOT = {"trades": TICKS_DIR, "bidask": TICKS_DIR, "depth": DEPTH_DIR}


class TickCaptureWriter:
    """Date-rotating JSONL writer: data/ticks/<SYMBOL>/<date>.jsonl for
    trades/bidask, data/depth/<SYMBOL>/<date>.jsonl for L2 depth — one file
    handle per (symbol, kind, ET date)."""

    def __init__(self, ticks_dir: str | Path = TICKS_DIR, depth_dir: str | Path = DEPTH_DIR) -> None:
        self._roots = {"trades": Path(ticks_dir), "bidask": Path(ticks_dir), "depth": Path(depth_dir)}
        self._handles: dict[tuple[str, str, str], object] = {}

    def write(self, symbol: str, kind: str, payload: dict) -> None:
        now = datetime.now(timezone.utc)
        day_key = now.strftime("%Y%m%d")
        key = (symbol, kind, day_key)
        handle = self._handles.get(key)
        if handle is None:
            # Rotate: close any previous-day handle for this (symbol, kind).
            for stale in [k for k in self._handles if k[0] == symbol and k[1] == kind]:
                try:
                    self._handles.pop(stale).close()
                except Exception:
                    pass
            symbol_dir = self._roots[kind] / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            path = symbol_dir / f"{day_key}.jsonl"
            handle = open(path, "a", encoding="utf-8")
            self._handles[key] = handle
        payload = {"recorded_at": now.isoformat(), **payload}
        handle.write(json.dumps(payload) + "\n")

    def flush(self) -> None:
        for handle in self._handles.values():
            try:
                handle.flush()
            except Exception:
                pass

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


class IbkrTickCapture:
    """Subscribe to tick-by-tick + depth for `symbols` and archive every
    event. Synchronous ib_async style: call run() and it blocks until
    stop() (or KeyboardInterrupt in the capture script)."""

    def __init__(
        self,
        symbols: list[str],
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 41,
        ticks_dir: str | Path = TICKS_DIR,
        depth_dir: str | Path = DEPTH_DIR,
        max_depth_symbols: int = 3,
        rth_only: bool = True,
        request_timeout: float = 30.0,
        reconnect_delay: float = 15.0,
        max_reconnect_delay: float = 120.0,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._host = host
        self._port = port
        self._client_id = client_id
        self._writer = TickCaptureWriter(ticks_dir, depth_dir)
        self._max_depth_symbols = max_depth_symbols
        self._rth_only = rth_only
        # See module docstring "Resilience" note: bound how long any single
        # IB request can hang, and how the reconnect loop backs off.
        self._request_timeout = request_timeout
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._ib = None
        self._running = False
        self.event_counts: dict[str, int] = {"trades": 0, "bidask": 0, "depth": 0}

    # ── event handlers ───────────────────────────────────────────────────────

    def _in_scope_now(self) -> bool:
        if not self._rth_only:
            return True
        return is_regular_trading_hours(datetime.now(timezone.utc))

    def _on_pending_tickers(self, tickers) -> None:
        if not self._in_scope_now():
            return
        for ticker in tickers:
            symbol = ticker.contract.symbol
            for t in ticker.tickByTicks:
                # TickByTickAllLast (price/size/exchange) vs TickByTickBidAsk
                if hasattr(t, "price"):
                    self._writer.write(symbol, "trades", {
                        "time": t.time.isoformat() if getattr(t, "time", None) else "",
                        "price": float(t.price),
                        "size": float(t.size),
                        "exchange": getattr(t, "exchange", ""),
                        "special_conditions": getattr(t, "specialConditions", ""),
                    })
                    self.event_counts["trades"] += 1
                elif hasattr(t, "bidPrice"):
                    self._writer.write(symbol, "bidask", {
                        "time": t.time.isoformat() if getattr(t, "time", None) else "",
                        "bid_price": float(t.bidPrice),
                        "ask_price": float(t.askPrice),
                        "bid_size": float(t.bidSize),
                        "ask_size": float(t.askSize),
                    })
                    self.event_counts["bidask"] += 1
            for d in ticker.domTicks:
                self._writer.write(symbol, "depth", {
                    "time": d.time.isoformat() if getattr(d, "time", None) else "",
                    "position": int(d.position),
                    "market_maker": getattr(d, "marketMaker", ""),
                    "operation": int(d.operation),   # 0=insert 1=update 2=delete
                    "side": int(d.side),             # 0=ask 1=bid
                    "price": float(d.price),
                    "size": float(d.size),
                })
                self.event_counts["depth"] += 1
        self._writer.flush()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocks until stop() is called. Internally reconnects (with
        backoff) on any disconnect or request failure instead of raising —
        see the module docstring's "Resilience" note. A KeyboardInterrupt
        still propagates out so the CLI script's Ctrl-C handling keeps
        working unchanged."""
        from ib_async import IB, Stock  # type: ignore[import]

        self._running = True
        attempt = 0
        try:
            while self._running:
                attempt += 1
                try:
                    self._run_session(IB, Stock)
                except Exception:
                    log.exception(
                        "tick_capture: session error on attempt %d — will reconnect", attempt,
                    )
                finally:
                    self._disconnect_quietly()

                if not self._running:
                    break
                delay = min(self._reconnect_delay * attempt, self._max_reconnect_delay)
                log.warning("tick_capture: reconnecting in %.0fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
        finally:
            self._writer.close()
            log.info("tick_capture: stopped — event counts: %s", self.event_counts)

    def _run_session(self, IB, Stock) -> None:  # type: ignore[no-untyped-def]
        """One connect -> qualify -> subscribe -> stream session. Raises on
        any failure so run()'s outer loop can reconnect."""
        ib = IB()
        ib.RequestTimeout = self._request_timeout
        ib.connect(self._host, self._port, clientId=self._client_id, timeout=10)
        self._ib = ib
        log.info("tick_capture: connected to %s:%d (clientId=%d)",
                 self._host, self._port, self._client_id)

        contracts = []
        for symbol in self._symbols:
            contract = Stock(symbol, "SMART", "USD")
            try:
                qualified = ib.qualifyContracts(contract)
            except Exception:
                log.warning("tick_capture: qualify %s timed out/failed — skipping this session",
                            symbol, exc_info=True)
                continue
            if not qualified:
                log.warning("tick_capture: could not qualify %s — skipping", symbol)
                continue
            contracts.append(contract)

        if not contracts:
            # Most likely the sec-def data farm is still down — bail out and
            # let the outer loop retry with a fresh connection/backoff
            # rather than looping here with nothing subscribed.
            raise ConnectionError("tick_capture: no contracts could be qualified this session")

        depth_contracts = contracts[: self._max_depth_symbols]
        for contract in contracts:
            ib.reqTickByTickData(contract, "AllLast", 0, False)
            ib.reqTickByTickData(contract, "BidAsk", 0, False)
        for contract in depth_contracts:
            ib.reqMktDepth(contract, numRows=DEFAULT_DEPTH_ROWS, isSmartDepth=True)
        log.info("tick_capture: tick-by-tick on %s; L2 depth on %s (IB depth-subscription cap)",
                 [c.symbol for c in contracts], [c.symbol for c in depth_contracts])

        ib.pendingTickersEvent += self._on_pending_tickers
        while self._running and ib.isConnected():
            ib.sleep(1.0)

    def _disconnect_quietly(self) -> None:
        ib = self._ib
        self._ib = None
        if ib is not None:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        self._disconnect_quietly()
