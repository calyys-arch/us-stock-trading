"""
Live glue between DataEngine's bus("snapshot") and the intraday
microstructure signal + risk-qualification pipeline.

2026-08-15 paper-forward experiment: LIVE_SIGNALS is ONLY
`absorption_breakout` (frozen TIGHT6 + breakout_atr_mult=0.5 + the
round-3 macro beta gate). Retired / confirmed-losing names
(sweep_reclaim, fvg_retest, orb_vwap, orb_vwap_regime, vwap_band_fade,
vp_breakout, l2_absorption) are a live footgun and must never appear
here. This is NOT a WFO GO promotion — see
backtests/reports/absorption_breakout_paper_protocol.md.

Design principle: this module owns bar-buffering, session bookkeeping, and
account/position state ONLY. It deliberately reuses
python/backtest/intraday_engine.py's own `_evaluate_signal` dispatch
so live signal behavior can never silently drift from whatever was
actually WFO-validated in the backtester.

Decision chart is 5 minutes (IntradayBacktestConfig.chart_minutes, default
5). 1-minute is the raw feed only: every closed 1m bar is ingested, but
signals are evaluated only when a new 09:30-ET-aligned 5m bin closes
(09:34, 09:39, …), and `_evaluate_signal` is handed the resampled closed
5m OHLCV. Signals that resample internally (vsa/obv/auction) still receive
the 1m prefix so they are not double-resampled.

The macro beta gate is a hardcoded structural rule on the live path
(not a leftover monkeypatch): long only if QQQ/SPY/XLK composite 1m AND
5m momentum align; short mirrored; missing index bars fail CLOSED.

Bar-close detection: DataEngine's CandleBuilder.last(n) always returns
[...closed history..., current in-progress candle] as long as any tick has
been processed for that (code, timeframe) — see python/core/data_engine.py.
So `MarketSnapshot.candles_1m[:-1]` is always "every closed 1-minute bar we
know about".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from python.analytics.macro_beta_gate import LiveMacroGate
from python.backtest.intraday_engine import IntradayBacktestConfig, SIGNAL_PARAM_KEYS, _evaluate_signal
from python.core.bus import MessageBus
from python.core.calendar import is_regular_trading_hours
from python.core.event_blackout import is_event_blackout
from python.core.paper_forward import ABSORPTION_BREAKOUT_UNIVERSE
from python.core.risk_engine import RiskEngine
from python.core.types import MarketSnapshot, QualifiedMicroOrder
from python.microstructure import context as ctx
from python.microstructure.signal_journal import SignalJournal

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Paper-forward experiment: the ONLY live microstructure signal.
# Retired names must never be added back here.
LIVE_SIGNALS: tuple[str, ...] = ("absorption_breakout",)

# 1m ingest buffer. RTH is ~390 one-minute bars / ~78 five-minute
# decision bars; 500 keeps a full 1m session for resample + later
# fill/stop resolution without trimming the open.
_MAX_SESSION_BARS = 500
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Signals that resample 1m → Nx themselves. Passing already-resampled
# decision bars would double-resample. LIVE_SIGNALS is absorption-only;
# this set is defensive if a test overrides `signals`.
_SELF_RESAMPLE_SIGNALS = frozenset({
    "vsa_no_demand", "obv_divergence", "auction_reclaim", "vsa_effort",
})


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
        signal_journal: Optional[SignalJournal] = None,
        live_universe: Optional[frozenset[str]] = None,
        macro_gate: Optional[LiveMacroGate] = None,
        apply_macro_gate: bool = True,
    ) -> None:
        self._bus = bus
        self._risk = risk_engine
        self._params = strategy_params
        self._get_equity = get_account_equity
        self._signals = signals
        # TIGHT6 for absorption_breakout; tests may override. Symbols
        # outside this set are never evaluated on the live path.
        self._live_universe: Optional[frozenset[str]] = (
            live_universe if live_universe is not None
            else frozenset(ABSORPTION_BREAKOUT_UNIVERSE)
        )
        # Fail-closed macro beta gate. Tests inject a pre-loaded
        # LiveMacroGate (or set apply_macro_gate=False to exercise the
        # raw evaluate/qualify path without index bars).
        self._apply_macro_gate = apply_macro_gate
        self._macro_gate = macro_gate if macro_gate is not None else LiveMacroGate()
        # Deliberately opt-in (default None => no journaling, no disk
        # writes) rather than always-constructing a SignalJournal here —
        # this class is built directly (with no journal arg) by several
        # existing unit tests that must stay side-effect-free; only
        # dashboard/engine_bridge.py's live EngineRuntime passes a real one.
        self._journal = signal_journal
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
        if self._live_universe is not None and snap.code not in self._live_universe:
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
        bars_1m = buf.bars_today()
        if len(bars_1m) < 2:
            return

        minutes = int(self._cfg.chart_minutes)
        if minutes > 1:
            if not ctx.session_bin_just_closed(bars_1m, minutes):
                return
            decision_bars = ctx.closed_session_bars(bars_1m, minutes)
            if decision_bars.empty:
                return
        else:
            decision_bars = bars_1m

        if self._apply_macro_gate:
            try:
                self._macro_gate.refresh_for(decision_bars.index[-1])
            except Exception:
                log.exception(
                    "MicrostructureScheduler: macro refresh_for failed for %s — "
                    "continuing with last seed (fail-closed if still empty)",
                    symbol,
                )

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
                bars_for_sig = (
                    bars_1m if signal_name in _SELF_RESAMPLE_SIGNALS else decision_bars
                )
                sig = _evaluate_signal(
                    signal_name, bars_for_sig, symbol, self._params.get(signal_name, {}), self._cfg,
                    buf.prior_day_bars, buf.prior_close,
                )
            except Exception:
                log.exception(
                    "MicrostructureScheduler: %s evaluation raised for %s — skipping this signal only",
                    signal_name, symbol,
                )
                continue
            if sig is not None:
                if signal_name == "absorption_breakout" and self._apply_macro_gate:
                    if not self._macro_gate.ok(sig.direction, sig.signal_time):
                        await self._publish_macro_gate_closed(sig, snap)
                        continue
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
        # Journal EVERY signal that fires, approved or RiskEngine-rejected
        # (python/microstructure/signal_journal.py's `risk_passed` flag) —
        # this is the "紙上實測 vs 回測" evidence layer (plan §7), so it must
        # never depend on whether the order goes on to actually submit.
        # Best-effort: a journal failure must never block the live signal
        # pipeline that ALSO needs to reach the bus below.
        if self._journal is not None:
            try:
                self._journal.record(sig, order)
            except Exception:
                log.exception(
                    "MicrostructureScheduler: signal journal write failed for %s/%s — "
                    "continuing without journaling this signal", sig.symbol, sig.strategy,
                )
        # Always publish, approved or not — ExecutionGateway._on_microstructure_order
        # handles both cases (submits when its own two-key AND gate is armed,
        # otherwise reports the rejection) so the observe-mode report/log
        # path stays populated for visibility, exactly as the task requires;
        # this scheduler never re-implements or bypasses that gate itself.
        await self._bus.publish("qualified_micro_order", order)

    async def _publish_macro_gate_closed(self, sig, snap: MarketSnapshot) -> None:
        """Journal + publish a rejected order when the macro beta gate
        blocks (missing index bars, NaN momentum, or direction mismatch).
        Fail-closed: never silently drop the gate and trade ungated."""
        order = QualifiedMicroOrder(
            raw=sig, qty=0, entry_limit_price=0.0,
            stop_price=sig.stop_price, stop_limit_price=0.0,
            target_price=sig.target_price, gross_notional=0.0,
            approved=False,
            rejection_reason="macro_beta_gate_closed_or_missing_index",
        )
        if self._journal is not None:
            try:
                self._journal.record(sig, order)
            except Exception:
                log.exception(
                    "MicrostructureScheduler: signal journal write failed for %s/%s (macro gate) — "
                    "continuing without journaling this signal", sig.symbol, sig.strategy,
                )
        await self._bus.publish("qualified_micro_order", order)
