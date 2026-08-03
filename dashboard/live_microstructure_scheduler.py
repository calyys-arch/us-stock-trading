"""
Live glue between DataEngine's bus("snapshot") and the intraday
microstructure signal + risk-qualification pipeline
(python/microstructure/signals/{sweep_reclaim,fvg_retest,orb_vwap}.py,
python/core/risk_engine.py:RiskEngine.qualify_microstructure_order).

Design principle: this module owns bar-buffering, session bookkeeping, and
account/position state ONLY. It deliberately reuses
python/backtest/intraday_engine.py's own `_evaluate_signal` dispatch
(context construction: liquidity levels, session VWAP, opening range) so
live signal behavior can never silently drift from whatever was actually
WFO-validated in the backtester — see that module's module docstring on
its no-lookahead causality contract, which this scheduler inherits for
free by construction (it only ever hands `_evaluate_signal` bars up to and
including the just-closed bar, never anything from the future).

l2_absorption is DELIBERATELY excluded from LIVE_SIGNALS (see
python/microstructure/signals/l2_absorption.py's module docstring: a
bar-only proxy for its full spec, not validated against real L2 data, and
deliberately excluded from scripts/run_intraday_backtest.py's WFO/
promotion gate for the same reason — it must not reach real orders here
either).

Bar-close detection: DataEngine's CandleBuilder.last(n) always returns
[...closed history..., current in-progress candle] as long as any tick has
been processed for that (code, timeframe) — see python/core/data_engine.py.
So `MarketSnapshot.candles_1m[:-1]` is always "every closed 1-minute bar we
know about", and the LAST closed bar's timestamp is what this module tracks
per symbol to detect "a new bar just closed" without needing its own timer/
scheduler loop — bus("snapshot") already fires (throttled to ~0.2s) on
every live tick, which is far more often than once a minute.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from python.backtest.intraday_engine import IntradayBacktestConfig, SIGNAL_PARAM_KEYS, _evaluate_signal
from python.core.bus import MessageBus
from python.core.calendar import is_regular_trading_hours
from python.core.event_blackout import is_event_blackout
from python.core.risk_engine import RiskEngine
from python.core.types import MarketSnapshot

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# The three signals actually validated end-to-end through
# python/backtest/intraday_engine.py's fill/P&L simulation and
# scripts/run_intraday_backtest.py's WFO gate. l2_absorption is NOT in
# this tuple on purpose — see module docstring.
LIVE_SIGNALS: tuple[str, ...] = ("sweep_reclaim", "fvg_retest", "orb_vwap")

_MAX_SESSION_BARS = 500  # generous cap; a full RTH session is ~390 one-minute bars
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _to_et_naive(ts) -> pd.Timestamp:
    """python/microstructure/context.py's bar-index convention is a
    tz-naive US/Eastern DatetimeIndex (see that module's docstring).
    Timezone-AWARE timestamps (every real live tick — python/interfaces/
    ibkr_feed.py always stamps `datetime.now(timezone.utc)`) are converted
    to ET and stripped of tzinfo; already-naive timestamps
    (python/interfaces/market_data.py's SimulatedFeed, whose virtual clock
    is constructed starting at 09:30 to represent ET wall time) are
    assumed to already BE ET, matching how the rest of this live pipeline
    (e.g. python/core/data_engine.py feeding tick.timestamp straight into
    is_regular_trading_hours) already treats them."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_ET).tz_localize(None)
    return ts


@dataclass
class _SymbolBuffer:
    session_date: Optional[date] = None
    rows: list[dict] = field(default_factory=list)          # today's CLOSED 1m bars, chronological
    last_closed_ts: Optional[pd.Timestamp] = None
    prior_day_bars: Optional[pd.DataFrame] = None            # for YDH/YDL + orb_vwap's gap-trap rule
    prior_close: Optional[float] = None

    def bars_today(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        return pd.DataFrame(self.rows).set_index("ts")[_OHLCV_COLUMNS]


class MicrostructureScheduler:
    """Owns, per symbol: a rolling buffer of today's closed 1-minute bars,
    prior-session context (optional — see `seed_prior_session`), and a
    simple in-memory count of currently-open microstructure positions
    (RiskEngine.qualify_microstructure_order's `open_micro_positions`
    input — see `_on_execution_report`'s docstring for exactly how/when
    this is incremented/decremented, and its known limitations).

    `on_snapshot` is the single public entry point, meant to be called
    from EngineRuntime's existing bus("snapshot") subscriber once per
    published MarketSnapshot — this class does not subscribe to the bus
    itself for that topic (EngineRuntime already owns that subscription
    and mirrors snapshots into DashboardState; this keeps one obvious
    place, not two, that reacts to "snapshot")."""

    def __init__(
        self,
        bus: MessageBus,
        risk_engine: RiskEngine,
        strategy_params: dict[str, dict],
        get_account_equity: Callable[[], float],
        signals: tuple[str, ...] = LIVE_SIGNALS,
    ) -> None:
        self._bus = bus
        self._risk = risk_engine
        self._params = strategy_params
        self._get_equity = get_account_equity
        self._signals = signals
        self._buffers: dict[str, _SymbolBuffer] = {}
        self._open_positions: set[tuple[str, str]] = set()  # {(symbol, strategy)}
        self._cfg = IntradayBacktestConfig()
        bus.subscribe("execution_report", self._on_execution_report)

    # ── open-position bookkeeping ────────────────────────────────────────

    async def _on_execution_report(self, report: dict) -> None:
        """Best-effort live counter for
        RiskEngine.qualify_microstructure_order's `open_micro_positions`
        gate. KNOWN LIMITATION (documented, not silently assumed away):
        this only observes ORDER SUBMISSION acceptance
        (ExecutionGateway._on_microstructure_order's own execution_report,
        published at submit time, not at actual fill time) and broker-wide
        FLATTEN acceptance — it has no fill-level feed to detect a
        protective stop/target getting hit naturally without a flatten
        call, and it collapses repeat entries into the SAME (symbol,
        strategy) into one set member rather than counting each one, so it
        is a reasonable but imperfect proxy, not an exact broker
        reconciliation. See the module's final design-notes for the
        defensible alternative (a real fill-level position ledger) that
        was out of scope to build tonight."""
        try:
            rtype = report.get("type")
            if rtype == "micro_order":
                symbol = report.get("symbol")
                strategy = report.get("strategy")
                entry = report.get("entry") or {}
                if symbol and strategy and entry.get("accepted"):
                    self._open_positions.add((symbol, strategy))
            elif rtype in ("eod_flatten", "manual_flatten", "emergency_flatten"):
                code = report.get("code")
                result = report.get("result") or {}
                if code and result.get("accepted"):
                    self._open_positions = {p for p in self._open_positions if p[0] != code}
        except Exception:
            log.exception("MicrostructureScheduler: failed updating open-position bookkeeping from execution_report=%r", report)

    def open_micro_position_count(self) -> int:
        return len(self._open_positions)

    # ── optional prior-session context seeding (best-effort) ────────────

    def seed_prior_session(self, symbol: str, prior_day_bars: pd.DataFrame, prior_close: float) -> None:
        """Best-effort YDH/YDL + gap-trap context from the most recent
        CACHED session (data/history_1m/, via
        python/data/intraday_cache.py) for a symbol, called once by
        EngineRuntime.start() before the live feed begins. Symbols with no
        cached prior session simply never get this called — sweep_reclaim
        still works off same-session EQH/EQL and round levels, and
        orb_vwap still works, just without the gap-trap rule (an honest,
        documented degradation, not a crash)."""
        buf = self._buffers.setdefault(symbol, _SymbolBuffer())
        buf.prior_day_bars = prior_day_bars
        buf.prior_close = prior_close

    # ── main hook: one call per DataEngine snapshot ──────────────────────

    async def on_snapshot(self, snap: MarketSnapshot) -> None:
        now = snap.timestamp
        if not is_regular_trading_hours(now):
            return

        try:
            buf = self._roll_session_if_needed(snap)
            new_candles = self._new_closed_candles(snap, buf)
        except Exception:
            log.exception("MicrostructureScheduler: failed extracting closed bars for %s — skipping this snapshot", snap.code)
            return

        # Appended and evaluated ONE bar at a time — NOT "append every newly
        # closed candle, then evaluate N times against the fully-populated
        # buffer" — the latter would let a signal see bars from its own
        # future relative to earlier bars in the same batch (e.g. if the
        # scheduler ever falls behind and a snapshot carries several newly
        # closed bars at once). This loop reproduces
        # python/backtest/intraday_engine.py's own `for i in
        # range(1, len(bars))` one-bar-at-a-time causality exactly, even
        # when multiple bars close between two "snapshot" events.
        for ts, c in new_candles:
            buf.rows.append({"ts": ts, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume})
            buf.last_closed_ts = ts
            if len(buf.rows) > _MAX_SESSION_BARS:
                del buf.rows[: len(buf.rows) - _MAX_SESSION_BARS]
            try:
                await self._on_bar_close(snap)
            except Exception:
                log.exception(
                    "MicrostructureScheduler: signal evaluation failed for %s at bar %s — "
                    "skipping this bar only, other symbols/bars are unaffected",
                    snap.code, ts,
                )

    # ── bar buffering ─────────────────────────────────────────────────────

    def _roll_session_if_needed(self, snap: MarketSnapshot) -> _SymbolBuffer:
        buf = self._buffers.setdefault(snap.code, _SymbolBuffer())
        today = _to_et_naive(snap.timestamp).normalize().date()
        if buf.session_date != today:
            if buf.rows:
                prior_df = buf.bars_today()
                buf.prior_day_bars = prior_df
                buf.prior_close = float(prior_df["close"].iloc[-1])
            buf.rows = []
            buf.last_closed_ts = None
            buf.session_date = today
        return buf

    def _new_closed_candles(self, snap: MarketSnapshot, buf: _SymbolBuffer) -> list[tuple[pd.Timestamp, object]]:
        candles = list(snap.candles_1m)
        if len(candles) < 2:
            return []  # CandleBuilder.last() always appends the in-progress bar last; need >=1 closed + open
        out: list[tuple[pd.Timestamp, object]] = []
        for c in candles[:-1]:
            ts = _to_et_naive(c.timestamp)
            if buf.last_closed_ts is not None and ts <= buf.last_closed_ts:
                continue
            out.append((ts, c))
        return out

    # ── signal evaluation + qualification ────────────────────────────────

    async def _on_bar_close(self, snap: MarketSnapshot) -> None:
        symbol = snap.code
        buf = self._buffers[symbol]
        bars_so_far = buf.bars_today()
        if len(bars_so_far) < 2:
            return

        for signal_name in self._signals:
            # Don't re-evaluate a strategy for a symbol it already has an
            # accepted, still-open entry for (see _on_execution_report) —
            # mirrors python/backtest/intraday_engine.py's own
            # "position is None and pending is None" gate before evaluating
            # a fresh signal. Without this, a signal condition that stays
            # true for several consecutive bars (e.g. price hovering just
            # above a swept round-number level) would submit a brand new
            # entry attempt on EVERY such bar instead of once per trade.
            if (symbol, signal_name) in self._open_positions:
                continue
            try:
                sig = _evaluate_signal(
                    signal_name, bars_so_far, symbol, self._params.get(signal_name, {}), self._cfg,
                    buf.prior_day_bars, buf.prior_close,
                )
            except Exception:
                log.exception(
                    "MicrostructureScheduler: %s evaluation raised for %s — skipping this signal only",
                    signal_name, symbol,
                )
                continue
            if sig is not None:
                await self._qualify_and_publish(sig, snap)

    async def _qualify_and_publish(self, sig, snap: MarketSnapshot) -> None:
        try:
            equity = float(self._get_equity() or 0.0)
        except Exception:
            log.exception(
                "MicrostructureScheduler: get_account_equity() failed — treating equity as 0.0 "
                "(fail-safe: RiskEngine will size the order to 0 shares and reject it)"
            )
            equity = 0.0

        try:
            blackout = is_event_blackout(sig.symbol, sig.signal_time, window_minutes=self._risk.cfg.event_blackout_minutes)
        except Exception:
            log.exception(
                "MicrostructureScheduler: is_event_blackout failed for %s — treating as a blackout "
                "(fail-safe: reject rather than trade through unknown event risk)", sig.symbol,
            )
            blackout = True

        order = self._risk.qualify_microstructure_order(
            sig, snap, account_equity=equity,
            open_micro_positions=self.open_micro_position_count(),
            event_blackout=blackout, now=sig.signal_time,
        )
        # Always publish, approved or not — ExecutionGateway._on_microstructure_order
        # handles both cases (submits when its own two-key AND gate is armed,
        # otherwise reports the rejection) so the observe-mode report/log
        # path stays populated for visibility, exactly as the task requires;
        # this scheduler never re-implements or bypasses that gate itself.
        await self._bus.publish("qualified_micro_order", order)
