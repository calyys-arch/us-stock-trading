"""
Futu/Moomoo tick-by-tick + Level-2 order-book capture -> local JSONL archive
(data/ticks/ + data/depth/), as an alternative to
python/interfaces/ibkr_tick_capture.py.

Why this exists (2026-08-04): the IB Gateway session in use turned out to be
a free-standing IBKR **Demo** account (Client Portal showed "This is not a
brokerage account" + Customer Type "Individual (Demo)"), not a Paper Trading
account linked to a real live account. IBKR only lets a Paper account
inherit real-time market-data subscriptions from a funded live account —
a bare Demo account has no live account to subscribe against, so it is
permanently stuck on 15-20min delayed data (explains Error 10189/10190 and
Warning 2152 seen from python/interfaces/ibkr_tick_capture.py). The user's
Futu/Moomoo account IS a real, funded account with LV3 quote permission, so
it can genuinely stream real-time tick-by-tick trades and Level-2 depth for
US equities — through **OpenD**, Futu's local gateway process (the
equivalent of IB Gateway; must already be running and logged in before this
module can connect on port 11111).

Report-only scope: exactly like ibkr_tick_capture.py, this module ONLY
records market microstructure to disk for offline analysis by
python/signals/trap_detector.py. It never touches the trading engine, never
gates a signal.

What gets captured, per subscribed symbol:
  - tick-by-tick trades (SubType.TICKER): every print, including Futu's own
    `ticker_direction` (BUY/SELL/NEUTRAL straight from the tape) — a bonus
    column trap_detector.py doesn't use yet, but strictly better than the
    tick-rule inference order_flow_imbalance_score falls back to for IB data.
  - Level-2 order book (SubType.ORDER_BOOK): Futu pushes a FULL ordered
    snapshot of the top-N bid/ask price levels on every update, unlike IB's
    reqMktDepth which streams incremental insert/update/delete events. To
    let trap_detector.py's spoofing/layering heuristic (which expects IB's
    {operation, side, position, size} per-level event schema) run
    unmodified against Futu data, this module diffs each new snapshot
    against the previous one *by book position* and synthesizes
    insert(0)/update(1)/delete(2) events.
    ⚠️ APPROXIMATION, not a verified equivalent of IB's native diff stream:
    comparing by position index means a level shifting up/down one slot
    when the best price changes shows up as several "update" events (every
    shifted position) rather than IB's single clean insert/shift. Treat
    Futu-sourced depth churn counts as a coarser proxy, same honesty
    contract as the rest of this module's report-only scores.

Schema notes — both sources write into the SAME data/ticks/ + data/depth/
directories (python/interfaces/ibkr_tick_capture.TickCaptureWriter), each
row tagged `"source": "futu"` / `"source": "ibkr"` so a reader can tell them
apart or just treat the archive as multi-source:
  - Futu trades rows have NO `exchange` or `special_conditions` field (the
    US-equity Ticker feed exposes no venue/condition codes) — this
    correctly makes trap_detector.dark_pool_internalization_score and
    .print_lag_score report "unavailable" for Futu-sourced days (both
    explicitly check for those columns first) rather than silently implying
    a verified 0%.

Connection notes: futu-api's OpenQuoteContext reconnects to OpenD
automatically at the socket layer — it does not share ib_async's
"RequestTimeout=0 hangs forever" failure mode that caused the 2026-07-29 IB
incident. This module still wraps the whole session in an outer retry loop
with backoff (same lesson, applied defensively) purely as a safety net for
OpenD itself restarting out from under a live connection, and polls
`get_global_state().qot_logined` once a second as a liveness check.

Protocol encryption (2026-08-04): if OpenD has an `rsa_private_key` file
configured (OpenD's "Encrypted Private Key" setting), InitConnect responses
are RSA/AES-encrypted and a client that doesn't present the SAME private
key fails with `proto_id:1001 ... check sha error!` on every connection
attempt — this is the standard futu-api handshake behaviour, not a network
problem, and looks identical to a real outage from the outer retry loop's
perspective (indistinguishable "session error, reconnecting" logs forever).
Pass `rsa_key_path` (mirrors `configs/broker.yaml: futu.rsa_key_path`) to
have this module call `SysConfig.enable_proto_encrypt(True)` +
`SysConfig.set_init_rsa_file(...)` before connecting; leave it unset if
OpenD's "Encrypted Private Key" field is empty (protocol encryption off).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..core.calendar import is_regular_trading_hours
from .ibkr_tick_capture import DEFAULT_DEPTH_ROWS, DEPTH_DIR, TICKS_DIR, TickCaptureWriter

log = logging.getLogger(__name__)


def _iso(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _symbol_from_futu_code(code: str) -> str:
    return code.split(".", 1)[-1] if "." in code else code


class FutuTickCapture:
    """Subscribe to tick-by-tick trades + Level-2 order book for `symbols`
    via a local Futu/Moomoo OpenD gateway and archive every event. Same
    run()/stop()/event_counts contract as IbkrTickCapture so
    scripts/capture_market_microstructure.py can pick either source."""

    def __init__(
        self,
        symbols: list[str],
        host: str = "127.0.0.1",
        port: int = 11111,
        market_prefix: str = "US",
        ticks_dir: str | Path = TICKS_DIR,
        depth_dir: str | Path = DEPTH_DIR,
        max_depth_symbols: int = 20,
        depth_rows: int = DEFAULT_DEPTH_ROWS,
        rth_only: bool = True,
        reconnect_delay: float = 15.0,
        max_reconnect_delay: float = 120.0,
        rsa_key_path: str | None = None,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._host = host
        self._port = port
        self._market_prefix = market_prefix
        self._rsa_key_path = rsa_key_path
        self._writer = TickCaptureWriter(ticks_dir, depth_dir)
        # Futu's default subscription quota (~1000, own_used shared across
        # QUOTE/TICKER/ORDER_BOOK) dwarfs IB's ~3-concurrent-depth cap, so
        # the default here covers the whole 20-symbol universe rather than
        # truncating to the first few symbols like ibkr_tick_capture.py must.
        self._max_depth_symbols = max_depth_symbols
        self._depth_rows = depth_rows
        self._rth_only = rth_only
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._quote_ctx = None
        self._running = False
        self.event_counts: dict[str, int] = {"trades": 0, "depth": 0}
        # Previous order-book snapshot per futu code (list of (price, size)
        # tuples per side), used to diff into synthetic IB-shaped events.
        self._prev_book: dict[str, dict[str, list[tuple[float, float]]]] = {}

    def _futu_code(self, symbol: str) -> str:
        return f"{self._market_prefix}.{symbol}"

    def _in_scope_now(self) -> bool:
        if not self._rth_only:
            return True
        return is_regular_trading_hours(datetime.now(timezone.utc))

    # ── event handlers (called from the futu SDK's push callbacks) ─────────

    def _handle_ticker(self, data) -> None:
        if not self._in_scope_now() or data is None or len(data) == 0:
            return
        for _, row in data.iterrows():
            symbol = _symbol_from_futu_code(str(row.get("code", "")))
            self._writer.write(symbol, "trades", {
                "time": _iso(row.get("time")),
                "price": float(row.get("price", 0.0) or 0.0),
                "size": float(row.get("volume", 0.0) or 0.0),
                "ticker_direction": str(row.get("ticker_direction", "")),
                "tick_type": str(row.get("type", "")),
                "sequence": str(row.get("sequence", "")),
                "source": "futu",
            })
            self.event_counts["trades"] += 1
        self._writer.flush()

    def _handle_order_book(self, data) -> None:
        if not self._in_scope_now() or not isinstance(data, dict):
            return
        code = str(data.get("code", ""))
        symbol = _symbol_from_futu_code(code)
        now_iso = datetime.now(timezone.utc).isoformat()
        book = self._prev_book.setdefault(code, {"Bid": [], "Ask": []})

        for side_name, side_code in (("Bid", 1), ("Ask", 0)):
            levels = (data.get(side_name) or [])[: self._depth_rows]
            prev_levels = book[side_name]
            for position, level in enumerate(levels):
                price, size = float(level[0]), float(level[1])
                order_count = int(level[2]) if len(level) > 2 else 0
                prev = prev_levels[position] if position < len(prev_levels) else None
                if prev is None:
                    operation = 0  # insert
                elif prev[0] != price or prev[1] != size:
                    operation = 1  # update
                else:
                    continue  # unchanged level — IB only streams diffs, mirror that here
                self._writer.write(symbol, "depth", {
                    "time": now_iso,
                    "position": position,
                    "market_maker": str(order_count),
                    "operation": operation,
                    "side": side_code,
                    "price": price,
                    "size": size,
                    "source": "futu",
                })
                self.event_counts["depth"] += 1
            for position in range(len(levels), len(prev_levels)):
                prev_price, prev_size = prev_levels[position]
                self._writer.write(symbol, "depth", {
                    "time": now_iso,
                    "position": position,
                    "market_maker": "",
                    "operation": 2,  # delete — level fell off the book
                    "side": side_code,
                    "price": prev_price,
                    "size": prev_size,
                    "source": "futu",
                })
                self.event_counts["depth"] += 1
            book[side_name] = [(float(level[0]), float(level[1])) for level in levels]
        self._writer.flush()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocks until stop() is called. Internally reconnects (with
        backoff) on any disconnect or subscribe failure instead of raising —
        see the module docstring's "Connection notes". A KeyboardInterrupt
        still propagates out so the CLI script's Ctrl-C handling works."""
        self._running = True
        attempt = 0
        try:
            while self._running:
                attempt += 1
                try:
                    self._run_session()
                except Exception:
                    log.exception(
                        "futu_tick_capture: session error on attempt %d — will reconnect", attempt,
                    )
                finally:
                    self._close_quietly()

                if not self._running:
                    break
                delay = min(self._reconnect_delay * attempt, self._max_reconnect_delay)
                log.warning("futu_tick_capture: reconnecting in %.0fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
        finally:
            self._writer.close()
            log.info("futu_tick_capture: stopped — event counts: %s", self.event_counts)

    def _run_session(self) -> None:
        from futu import (
            OpenQuoteContext,
            OrderBookHandlerBase,
            RET_OK,
            Session,
            SubType,
            SysConfig,
            TickerHandlerBase,
        )

        if self._rsa_key_path:
            # Must match OpenD's "Encrypted Private Key" setting exactly —
            # see module docstring's "Protocol encryption" note. Set on
            # every attempt (cheap, idempotent) since SysConfig is process-global
            # and this method may run again after a reconnect.
            SysConfig.enable_proto_encrypt(True)
            SysConfig.set_init_rsa_file(self._rsa_key_path)

        owner = self

        class _TickerRelay(TickerHandlerBase):
            def on_recv_rsp(self, rsp_pb):
                ret_code, data = super(_TickerRelay, self).on_recv_rsp(rsp_pb)
                if ret_code != RET_OK:
                    log.warning("futu_tick_capture: ticker push error: %s", data)
                    return ret_code, data
                owner._handle_ticker(data)
                return RET_OK, data

        class _OrderBookRelay(OrderBookHandlerBase):
            def on_recv_rsp(self, rsp_pb):
                ret_code, data = super(_OrderBookRelay, self).on_recv_rsp(rsp_pb)
                if ret_code != RET_OK:
                    log.warning("futu_tick_capture: order book push error: %s", data)
                    return ret_code, data
                owner._handle_order_book(data)
                return RET_OK, data

        ctx = OpenQuoteContext(host=self._host, port=self._port)
        self._quote_ctx = ctx
        ctx.set_handler(_TickerRelay())
        ctx.set_handler(_OrderBookRelay())
        log.info("futu_tick_capture: connected to OpenD at %s:%d", self._host, self._port)

        codes = [self._futu_code(s) for s in self._symbols]
        ret, err = ctx.subscribe(codes, [SubType.TICKER], subscribe_push=True, session=Session.ALL)
        if ret != RET_OK:
            raise ConnectionError(f"futu_tick_capture: TICKER subscribe failed: {err}")

        depth_codes = codes[: self._max_depth_symbols]
        if depth_codes:
            ret, err = ctx.subscribe(depth_codes, [SubType.ORDER_BOOK], subscribe_push=True)
            if ret != RET_OK:
                log.warning("futu_tick_capture: ORDER_BOOK subscribe failed: %s — continuing with trades only", err)
                depth_codes = []

        log.info("futu_tick_capture: tick-by-tick on %s; L2 depth on %s",
                 self._symbols, [_symbol_from_futu_code(c) for c in depth_codes])

        while self._running:
            time.sleep(1.0)
            ret, state = ctx.get_global_state()
            if ret != RET_OK or not (isinstance(state, dict) and state.get("qot_logined")):
                raise ConnectionError(f"futu_tick_capture: OpenD quote session not logged in: {state}")

    def _close_quietly(self) -> None:
        ctx = self._quote_ctx
        self._quote_ctx = None
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        self._close_quietly()
