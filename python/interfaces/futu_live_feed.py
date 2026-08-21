"""
File-tailing MarketDataFeed that reconstructs live Tick objects for the
dashboard's live engine (dashboard/engine_bridge.py, `data_source:
futu_live`) by POLLING the same data/ticks/<SYMBOL>/<date>.jsonl and
data/depth/<SYMBOL>/<date>.jsonl files scripts/capture_market_microstructure.py
--source futu (python/interfaces/futu_tick_capture.py) is already writing —
see docs/microstructure_pivot_plan.md §7's "把訊號迴路真正接上 dashboard".

Why file-tailing instead of a second Futu OpenD connection: opening a
second `OpenQuoteContext` subscription here would risk interfering with
(or duplicating quota against) the capture script's own stable, days-old
subscriptions covering the full 20-symbol universe. Reading its
already-written JSONL output instead needs ZERO new Futu connections, at
the cost of feed latency bounded by `poll_interval` (default well under a
second) — an explicit, acceptable trade-off given the system's own
Observe-only-first principle (plan §0): this phase never submits a market
order, so sub-second staleness in a diagnostic-only feed has no execution
consequence. If the capture script (or OpenD) isn't running, this feed
simply emits no ticks for the affected symbols — see "Degradation" below —
it never tries to reconnect anything itself.

Schema this module tails (see python/interfaces/futu_tick_capture.py's own
docstring for the authoritative version):
  trades  data/ticks/<SYMBOL>/<YYYYMMDD>.jsonl (UTC date in the filename,
          matching TickCaptureWriter.write's own `day_key`) — one JSON
          object per print: {"time": <naive-ET ISO>, "price": float,
          "size": float, ...}.
  depth   data/depth/<SYMBOL>/<YYYYMMDD>.jsonl — one JSON object per
          order-book LEVEL CHANGE, already diffed against the previous
          Futu snapshot by FutuTickCapture (NOT a full snapshot on every
          line): {"position": int, "operation": 0|1|2 (insert/update/
          delete), "side": 0|1 (ask/bid), "price": float, "size": float,
          ...}.

Best bid/ask reconstruction: rather than replaying the full top-N book
(effectively python/backtest/depth_replay.py's Phase-3 scope), this feed
tracks ONLY position == 0 (top-of-book) per side, updating on insert/
update events at position 0 and — a deliberate, documented simplification
rather than a silently-assumed equivalence — IGNORING delete events at
position 0: it keeps the last known best price rather than trying to infer
the new best from a lower position shifting up, which position-indexed
diff data alone cannot reliably do (the same "APPROXIMATION, not a
verified equivalent" caveat futu_tick_capture.py's own docstring makes one
layer up, applied again here). This is acceptable for an observe-only
signal journal; it would NOT be acceptable as an execution-quality input,
which this system never uses it for — SimBroker/ExecutionGateway fill
against signal/limit prices computed upstream, never against this feed's
bid/ask directly.

Degradation, by design, never a crash:
  - A symbol's trades file doesn't exist yet (before market open, or the
    capture script isn't running for that symbol) -> this feed emits NO
    ticks for that symbol; logged ONCE (not once per poll) until the file
    appears.
  - A symbol's depth file is missing/not-yet-existing while its trades
    file DOES exist -> ticks are still emitted from trades alone, with
    bid=ask=price and quote_ready=False (python/core/data_engine.py's own
    turnover math already knows to fall back to trade price when
    quote_ready is False — exactly how a not-yet-populated IBKR quote
    already degrades in python/interfaces/ibkr_feed.py).
  - UTC day rollover mid-session: this feed re-derives today's expected
    filenames every poll from `clock()` (defaults to real UTC now,
    matching TickCaptureWriter's own `day_key = now.strftime("%Y%m%d")`
    exactly) — so a long-running dashboard session automatically picks up
    the new day's files without a restart, the same way the capture
    script itself rotates writers at UTC midnight.

One-time catch-up cost: a symbol's file offset starts at 0 on the FIRST
poll after this feed is constructed, so if the dashboard is started
mid-session (an already-large file on disk, e.g. a full RTH day of ticks
is well over 100MB), that first poll for that symbol reads/parses the
WHOLE existing file rather than only "new since Start" — verified against
real multi-day capture output at ~9s for one ~140MB trades file. This is
one-time (every later poll only reads the incremental tail) and runs off
the asyncio event loop (`stream()` wraps `poll_once()` in
`run_in_executor`), so it delays this feed's first few ticks per symbol by
that much without freezing the rest of the dashboard — an accepted cost
for an observe-only feed with no execution consequence, not something this
pass optimizes further (e.g. by seeking to end-of-file on first open,
which would then also silently DROP any real ticks between session-open
and Start being clicked — a tail-only feed's honest choice, not a hidden
trade-off).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from ..core.types import Tick
from .ibkr_tick_capture import DEPTH_DIR, TICKS_DIR
from .market_data import MarketDataFeed

log = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SEC = 0.75


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _SymbolTailState:
    # ── trades side ──────────────────────────────────────────────────────
    trades_path: Optional[Path] = None
    trades_pos: int = 0
    trades_missing_warned: bool = False
    latest_price: Optional[float] = None
    latest_size: Optional[float] = None
    latest_trade_time: Optional[str] = None
    has_new_trade: bool = False

    # ── depth side (top-of-book only — see module docstring) ────────────
    depth_path: Optional[Path] = None
    depth_pos: int = 0
    depth_missing_warned: bool = False
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None


class FutuLiveFeed(MarketDataFeed):
    """Tails the Futu capture script's own JSONL output — see module
    docstring for the full design rationale. `codes` should be a subset of
    (or equal to) the universe scripts/capture_market_microstructure.py
    --source futu is running against; a symbol outside that set simply
    never has a file to tail and degrades per the "missing trades file"
    case above."""

    def __init__(
        self,
        codes: list[str],
        ticks_dir: str | Path = TICKS_DIR,
        depth_dir: str | Path = DEPTH_DIR,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SEC,
        clock: Callable[[], datetime] = _default_clock,
    ) -> None:
        self.codes = [c.upper() for c in codes]
        self._ticks_dir = Path(ticks_dir)
        self._depth_dir = Path(depth_dir)
        self._poll_interval = max(poll_interval, 0.05)
        self._clock = clock
        self._states: dict[str, _SymbolTailState] = {c: _SymbolTailState() for c in self.codes}
        log.info(
            "FutuLiveFeed initialised (file-tailing, no new Futu/OpenD connection): "
            "codes=%s ticks_dir=%s depth_dir=%s poll_interval=%.2fs",
            self.codes, self._ticks_dir, self._depth_dir, self._poll_interval,
        )

    # ── polling core (synchronous — safe to call directly in tests) ─────

    def poll_once(self) -> list[Tick]:
        """One tail pass over every symbol's trades + depth files. Returns
        one Tick per symbol that had at least one NEW trade line since the
        previous call (never more than one per symbol per call — multiple
        new trade lines within one poll interval collapse to the latest,
        an explicit trade-off for this observe-only, non-latency-critical
        feed)."""
        day_key = self._clock().strftime("%Y%m%d")
        ticks: list[Tick] = []
        for code in self.codes:
            st = self._states[code]
            self._tail_trades(code, st, day_key)
            self._tail_depth(code, st, day_key)
            if st.has_new_trade:
                st.has_new_trade = False
                ticks.append(self._build_tick(code, st))
        return ticks

    def _tail_trades(self, code: str, st: _SymbolTailState, day_key: str) -> None:
        path = self._ticks_dir / code / f"{day_key}.jsonl"
        if path != st.trades_path:
            st.trades_path = path
            st.trades_pos = 0
            st.trades_missing_warned = False
        if not path.exists():
            if not st.trades_missing_warned:
                log.info(
                    "FutuLiveFeed: no trades file yet for %s (%s) — emitting no ticks for it "
                    "until scripts/capture_market_microstructure.py writes one", code, path,
                )
                st.trades_missing_warned = True
            return
        lines = self._read_new_lines(path, st, is_depth=False)
        for line in lines:
            row = self._parse_json_line(line, path)
            if row is None:
                continue
            try:
                price = float(row["price"])
            except (KeyError, TypeError, ValueError):
                log.warning("FutuLiveFeed: skipping trades row missing/invalid 'price' in %s", path)
                continue
            size = row.get("size", 0.0)
            try:
                size = float(size or 0.0)
            except (TypeError, ValueError):
                size = 0.0
            st.latest_price = price
            st.latest_size = size
            st.latest_trade_time = row.get("time") or None
            st.has_new_trade = True

    def _tail_depth(self, code: str, st: _SymbolTailState, day_key: str) -> None:
        path = self._depth_dir / code / f"{day_key}.jsonl"
        if path != st.depth_path:
            st.depth_path = path
            st.depth_pos = 0
            st.depth_missing_warned = False
            # A new day means a fresh order book — yesterday's top-of-book
            # is not a safe carry-over guess for today's open.
            st.best_bid = None
            st.best_ask = None
        if not path.exists():
            if not st.depth_missing_warned:
                log.info(
                    "FutuLiveFeed: no depth file yet for %s (%s) — its ticks will carry "
                    "bid=ask=trade_price (quote_ready=False) until one appears", code, path,
                )
                st.depth_missing_warned = True
            return
        lines = self._read_new_lines(path, st, is_depth=True)
        for line in lines:
            row = self._parse_json_line(line, path)
            if row is None:
                continue
            if row.get("position") != 0:
                continue  # only top-of-book tracked — see module docstring
            operation = row.get("operation")
            if operation not in (0, 1):
                continue  # ignore delete-at-top; keep last known best (documented simplification)
            try:
                price = float(row["price"])
            except (KeyError, TypeError, ValueError):
                continue
            side = row.get("side")
            if side == 1:
                st.best_bid = price
            elif side == 0:
                st.best_ask = price

    @staticmethod
    def _read_new_lines(path: Path, st: _SymbolTailState, is_depth: bool) -> list[str]:
        """Reads new bytes since the last stored file offset and returns
        only FULLY newline-terminated lines, decoded as UTF-8. Binary mode
        (byte offsets throughout) deliberately avoids the ambiguity of
        doing arithmetic on a text-mode file's opaque `tell()` value — the
        capture script may be mid-`write()` on the very last line at the
        instant we read it, so a trailing partial line (no `\\n` yet) is
        left unconsumed (file offset rewound to just before it) and picked
        up whole on the NEXT poll once TickCaptureWriter's next flush()
        lands the newline."""
        try:
            with path.open("rb") as f:
                start_pos = st.depth_pos if is_depth else st.trades_pos
                f.seek(start_pos)
                new_bytes = f.read()
        except OSError:
            log.exception("FutuLiveFeed: failed reading %s", path)
            return []

        if new_bytes.endswith(b"\n"):
            complete = new_bytes
            consumed = len(new_bytes)
        else:
            split_at = new_bytes.rfind(b"\n")
            complete = new_bytes[: split_at + 1] if split_at >= 0 else b""
            consumed = len(complete)

        new_pos = start_pos + consumed
        if is_depth:
            st.depth_pos = new_pos
        else:
            st.trades_pos = new_pos
        return [ln.decode("utf-8", errors="replace") for ln in complete.split(b"\n") if ln]

    @staticmethod
    def _parse_json_line(line: str, path: Path) -> Optional[dict]:
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            log.warning("FutuLiveFeed: skipping malformed JSON line in %s", path)
            return None

    def _build_tick(self, code: str, st: _SymbolTailState) -> Tick:
        price = st.latest_price or 0.0
        volume = max(int(st.latest_size or 0), 0)
        has_quote = (
            st.best_bid is not None and st.best_ask is not None
            and st.best_bid > 0 and st.best_ask > 0
        )
        bid = st.best_bid if has_quote else price
        ask = st.best_ask if has_quote else price
        return Tick(
            code=code,
            price=round(price, 4),
            volume=volume,
            bid=round(bid, 4),
            ask=round(ask, 4),
            timestamp=self._parse_trade_time(st.latest_trade_time),
            quote_ready=has_quote,
            source="futu_live",
        )

    def _parse_trade_time(self, raw: Optional[str]) -> datetime:
        if raw:
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                log.warning("FutuLiveFeed: unparseable trade timestamp %r — using clock() instead", raw)
        return self._clock()

    # ── MarketDataFeed contract ──────────────────────────────────────────

    async def stream(self) -> AsyncIterator[Tick]:
        loop = asyncio.get_event_loop()
        try:
            while True:
                ticks = await loop.run_in_executor(None, self.poll_once)
                for tick in ticks:
                    yield tick
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            log.info("FutuLiveFeed: stream cancelled")
            raise
