"""
Intraday event backtester — 1-minute bar event loop for the microstructure
signals (sweep_reclaim / fvg_retest / orb_vwap).

Causality contract (the whole point of an "event" backtester instead of
vectorizing over the full day at once): at loop iteration i, only
`bars.iloc[:i+1]` (bar i and everything before it) is ever visible to
signal detection or exit-condition checks. A signal detected AT bar i can
only be FILLED starting at bar i+1 — enforced structurally by the loop
order (exit-check and fill-check for existing position/pending order
happen BEFORE a new signal is evaluated in the same iteration), never by a
flag that could be forgotten.

Cost model (deliberately stricter than the daily engines — intraday costs
are the primary way a "profitable on paper" signal dies in practice):
  - entry AND exit each pay half-spread + a volume-participation impact
    term, scaled by `stress_slippage_multiplier` (set to 2.0 for the
    mandatory 2x slippage stress test, configs/goal.yaml intraday gates).
  - IB-tiered-style flat per-share commission with a minimum, paid on
    both legs.
  - position size = 1% account risk / stop distance (docs/
    microstructure_pivot_plan.md §6), not a fixed notional.
  - exits: stop, target, time-stop (no favorable move within
    `time_stop_minutes`), or forced EOD flatten
    `flatten_buffer_minutes` before the session's last bar (mirrors
    python/core/calendar.py's _INTRADAY_FLATTEN_BUFFER convention).

Output contract: `metrics_from_report` returns EXACTLY the same dict shape
as python/backtest/optimize.py's `_metrics_from_returns` (sharpe_ratio,
max_drawdown, n_trades, total_net_pnl, n_days, daily_returns) so
walk_forward.py / monte_carlo.py / promotion.py consume it with ZERO
changes. The Sharpe/drawdown formula is intentionally duplicated here
(not imported) to avoid a cross-module coupling on a private helper — if
one changes, check the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analytics import regime as regime_mod
from ..microstructure import context as ctx
from ..microstructure.signals import MicroSignal
from ..microstructure.signals.absorption_breakout import evaluate_absorption_breakout
from ..microstructure.signals.auction_reclaim import evaluate_auction_reclaim, preopen_1h_environment
from ..microstructure.signals.vsa_effort import evaluate_vsa_effort
from ..microstructure.signals.vsa_no_demand import evaluate_vsa_no_demand
from ..microstructure.signals.obv_divergence import evaluate_obv_divergence
from ..microstructure.signals.fvg_retest import evaluate_fvg_retest
from ..microstructure.signals.l2_absorption import evaluate_l2_absorption
from ..microstructure.signals.orb_vwap import evaluate_orb_vwap
from ..microstructure.signals.orb_vwap_regime import evaluate_orb_vwap_regime
from ..microstructure.signals.sweep_reclaim import evaluate_sweep_reclaim
from ..microstructure.signals.vp_breakout import evaluate_vp_breakout
from ..microstructure.signals.vwap_band_fade import evaluate_vwap_band_fade

SIGNAL_PARAM_KEYS = {
    "sweep_reclaim": ["sweep_min_atr", "reclaim_bars", "stop_atr_mult"],
    "fvg_retest": ["vol_mult", "entry_pct", "expiry_bars"],
    # max_entries_per_session / stop_atr_buffer_mult / target_r_multiple were
    # added by the 2026-08-13 cost-to-edge rescue investigation
    # (backtests/reports/orb_vwap_rescue_report.md). All three default to
    # "off" in configs/strategy.yaml, so a caller that does not set them
    # observes byte-for-byte the pre-existing behavior.
    "orb_vwap": ["or_minutes", "vwap_side_filter", "max_entries_per_session",
                 "stop_atr_buffer_mult", "target_r_multiple"],
    # New signal hypotheses (docs/microstructure_pivot_plan.md's discipline
    # applied to genuinely new ideas, not re-tuning the three above after
    # their NO-GO — see backtests/reports/new_signals_report.md).
    "orb_vwap_regime": ["or_minutes", "vwap_side_filter"],           # 0 NEW params vs orb_vwap — the regime gate is a hardcoded rule
    "vwap_band_fade": ["band_sigma_mult", "stall_bars", "stop_atr_mult"],
    "vp_breakout": ["vol_mult", "confirm_bars", "stop_atr_mult"],
    # l2_absorption (S4) — a BAR-ONLY proxy for real L2 confirmation (see
    # l2_absorption.py's module docstring), but it IS dispatched through
    # run_symbol_day's full fill/P&L simulation and
    # scripts/run_intraday_backtest.py's WFO gate as of 2026-08-14 (see
    # backtests/reports/l2_absorption_validation_report.md) — the dispatch
    # below was always shared/generic (this module never special-cased
    # l2_absorption out of run_symbol_day), only the CLI's signal lists and
    # the live scheduler's LIVE_SIGNALS ever excluded it. `target_r_multiple`
    # is the same lever `orb_vwap`'s rescue investigation added.
    "l2_absorption": ["volume_mult", "touch_atr_mult", "stop_atr_mult", "target_r_multiple"],
    # absorption_breakout (2026-08-14) — the "Option A" continuation variant
    # investigated in backtests/reports/absorption_breakout_investigation_report.md.
    # Same level/volume definitions as l2_absorption (see that module vs.
    # this one's docstring for the polarity difference). `micro_stop_cents`
    # (round 2, same date) is a 5th, ALTERNATIVE-not-additive stop-distance
    # lever tested against `stop_atr_mult` — see absorption_breakout.py's
    # docstring and the report's round-2 addendum.
    "absorption_breakout": ["volume_mult", "breakout_atr_mult", "stop_atr_mult", "target_r_multiple",
                            "micro_stop_cents"],
    # auction_reclaim (2026-08-18) — Creamer-style 5-minute auction reclaim
    # on prior-session value area + fib discount/premium. Optional GEX /
    # footprint are loaded as session context (not free parameters).
    "auction_reclaim": ["min_rel_volume", "min_wick_frac", "stop_atr_mult", "target_r_multiple"],
    # vsa_effort (2026-08-18) — Wyckoff/VSA effort-without-result on 5m.
    # GEX is session context, not a free parameter.
    "vsa_effort": ["effort_vol_mult", "test_vol_mult", "stop_atr_mult", "target_r_multiple"],
    # vsa_no_demand (2026-08-18) — Williams/Coulling no-demand /
    # no-selling-pressure on a narrow 5m bar, confirmed by the next bar.
    # GEX is session context, not a free parameter.
    "vsa_no_demand": ["spread_atr_max", "vol_lookback", "stop_atr_mult", "target_r_multiple"],
    # obv_divergence (2026-08-18) — Granville B-2 / S-2 on session 5m OBV.
    # GEX is session context, not a free parameter.
    "obv_divergence": ["lookback_bars", "obv_lag_frac", "stop_atr_mult", "target_r_multiple"],
}

# Keys that appear in SIGNAL_PARAM_KEYS (so the WFO/param-grid plumbing in
# python/backtest/optimize.py carries them end-to-end) but are consumed by
# THIS module's per-session event loop rather than by the signal's own
# `evaluate_*` function — session-scoped state a stateless per-bar signal
# deliberately does not carry. `_evaluate_signal` strips them before
# building the signal's kwargs; `run_symbol_day` reads them directly.
_SESSION_LEVEL_PARAM_KEYS = frozenset({"max_entries_per_session"})

# Research-only ablation keys forwarded from IntradayBacktestConfig
# .signal_filter_overrides. Not in SIGNAL_PARAM_KEYS.
_VSA_FILTER_KEYS = ("require_location", "require_confirm", "require_volume")
_OBV_FILTER_KEYS = ("require_location", "require_obv_lag")


def _volume_book_filter_kwargs(cfg: IntradayBacktestConfig, keys: tuple[str, ...]) -> dict:
    src = cfg.signal_filter_overrides or {}
    return {k: src[k] for k in keys if k in src}

# orb_vwap_regime's regime gate (docs/microstructure_pivot_plan.md-style
# hardcoded structural rule, not a tunable parameter — see
# orb_vwap_regime.py's module docstring): python/analytics/regime.py's OWN
# module defaults, applied here rather than exposed as new free parameters.
_REGIME_WINDOW = 20
_REGIME_THRESHOLD = 0.02


def _daily_trending_flags(bars: pd.DataFrame) -> dict[pd.Timestamp, bool]:
    """Per-symbol "is trading day `d` a trending (Bull/Bear, not Sideways)
    day" lookup for orb_vwap_regime, computed ONCE per symbol from that
    symbol's own daily closes (last close of each session already present
    in `bars`). No lookahead: day `d`'s flag is the regime label AS OF
    THE PRIOR trading day's close (`.shift(1)` on the label series, which
    is itself computed from trailing-only data — see regime.label_regimes)
    — never `d`'s own close, which is not knowable before `d`'s session
    opens. Days without enough trailing history for a label (the first
    `_REGIME_WINDOW` trading days of whatever range `bars` covers) default
    to False (no trade) rather than guessing — an honest, conservative
    edge condition, same spirit as run_intraday_backtest's "first day has
    no prior_day_bars" note."""
    if bars.empty:
        return {}
    daily_close = bars.groupby(bars.index.normalize())["close"].last()
    if len(daily_close) <= _REGIME_WINDOW:
        return {}
    labels = regime_mod.label_regimes(daily_close, window=_REGIME_WINDOW, threshold=_REGIME_THRESHOLD)
    prior_labels = labels.shift(1)
    sideways_idx = regime_mod.STATES.index("Sideways")
    return {date: bool(val != sideways_idx) for date, val in prior_labels.dropna().items()}


@dataclass
class IntradayBacktestConfig:
    capital: float = 1_000_000.0
    risk_per_trade_pct: float = 0.01
    max_notional_pct: float = 0.20  # position sizing ceiling — see _position_size
    half_spread_bps: float = 2.0
    # Optional per-symbol override for half_spread_bps, calibrated from real
    # captured L2 depth data (scripts/calibrate_slippage_spreads.py ->
    # backtests/reports/calibrated_spreads.json) instead of the flat
    # constant above — docs/microstructure_pivot_plan.md §4a's honest
    # caveat ("half_spread 用該股近期平均買賣價差...沒有就用保守常數") now
    # has real data to fill in for symbols we have captured depth for.
    # None (the default) or a symbol missing from the dict both fall back
    # to `half_spread_bps` unchanged — this is a strictly additive,
    # backward-compatible knob: no existing caller/test that never sets
    # this field observes any behavior change.
    half_spread_bps_by_symbol: dict[str, float] | None = None
    # Market impact, in basis points of price, AT 100% participation (order
    # size == that bar's entire volume) — see _slippage_price. NOT a raw
    # fraction: impact_bps_per_participation=20 means "trading 100% of a
    # bar's volume costs ~0.20% of price in extra slippage", which is
    # already a stress-y assumption for a 1-minute bar.
    impact_bps_per_participation: float = 20.0
    max_participation: float = 2.0  # hard cap on (shares / bar_volume) fed into the impact term
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    time_stop_minutes: int = 10
    flatten_buffer_minutes: int = 5
    stress_slippage_multiplier: float = 1.0
    eq_lookback: int = 20
    eq_atr_mult: float = 0.1
    # Structural decision-chart size (not a WFO free parameter — not in
    # SIGNAL_PARAM_KEYS / yaml grids). Live scheduler + absorption_breakout
    # use this so 1-minute is the raw feed only; vsa_no_demand /
    # obv_divergence also consume it. Default 5.
    chart_minutes: int = 5
    # Research-only leave-one-out switches for vsa_no_demand /
    # obv_divergence (scripts/ablate_volume_book_filters.py).
    # Not in SIGNAL_PARAM_KEYS / yaml grids / Chan free-param count.
    # Empty dict = all require_* defaults (True) = current behavior.
    signal_filter_overrides: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    target_price: float | None
    entry_commission: float
    signal_time: pd.Timestamp


@dataclass
class PendingOrder:
    signal: MicroSignal
    created_at: pd.Timestamp


@dataclass
class IntradayTrade:
    symbol: str
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    shares: int
    gross_pnl: float
    costs: float
    net_pnl: float
    signal_time: pd.Timestamp | None = None


@dataclass
class IntradayBacktestReport:
    trades: list[IntradayTrade] = field(default_factory=list)
    signals_emitted: int = 0
    signals_filled: int = 0

    def daily_pnl(self) -> dict[pd.Timestamp, float]:
        out: dict[pd.Timestamp, float] = {}
        for t in self.trades:
            d = t.exit_time.normalize()
            out[d] = out.get(d, 0.0) + t.net_pnl
        return out


# ── cost model ───────────────────────────────────────────────────────────────

def _slippage_price(
    price: float, direction: str, shares: int, bar_volume: float,
    cfg: IntradayBacktestConfig, is_entry: bool, symbol: str | None = None,
) -> float:
    """half-spread + a participation-scaled impact term, both in basis
    points of price (never a raw price multiplier — a position that
    happens to be large relative to a thin bar's volume must cost more
    basis points, not become a different-magnitude number entirely).
    `max_participation` caps the (shares / bar_volume) ratio fed into the
    impact term as a second line of defense against pathological cases
    (e.g. a very tight stop producing a position that is still large
    relative to an unusually thin fill bar).

    `symbol` looks up `cfg.half_spread_bps_by_symbol` for a calibrated
    override (see that field's docstring); `symbol=None` or a dict miss
    (or the dict being entirely None) falls back to the flat
    `cfg.half_spread_bps` — the ONLY path that existed before this field
    was added, so every pre-existing call site that doesn't pass `symbol`
    is byte-for-byte unaffected."""
    half_spread_bps = cfg.half_spread_bps
    if symbol is not None and cfg.half_spread_bps_by_symbol:
        half_spread_bps = cfg.half_spread_bps_by_symbol.get(symbol, half_spread_bps)
    participation = min(shares / max(bar_volume, 1.0), cfg.max_participation)
    impact_bps = cfg.impact_bps_per_participation * participation
    total = price * ((half_spread_bps + impact_bps) / 10_000.0) * cfg.stress_slippage_multiplier
    adverse_up = (direction == "long") == is_entry
    return price + total if adverse_up else price - total


def _commission(shares: int, cfg: IntradayBacktestConfig) -> float:
    return max(cfg.commission_per_share * shares, cfg.min_commission)


def _position_size(entry_price: float, stop_price: float, cfg: IntradayBacktestConfig) -> int:
    """1% account risk / stop distance (docs/microstructure_pivot_plan.md
    §6), capped at `max_notional_pct` of capital. The cap matters a lot in
    practice: a signal whose stop happens to sit very close to entry (a
    tight round-number level, a low-ATR name) would otherwise imply an
    enormous, unrealistic share count under pure risk-based sizing —
    nothing in a live broker would let that order through, and letting it
    through in a backtest silently turns a small per-share cost edge into
    a nonsensical P&L number."""
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0 or entry_price <= 0:
        return 0
    risk_dollars = cfg.capital * cfg.risk_per_trade_pct
    shares_by_risk = risk_dollars / stop_dist
    shares_by_notional = (cfg.capital * cfg.max_notional_pct) / entry_price
    return max(int(min(shares_by_risk, shares_by_notional)), 0)


# ── signal dispatch ──────────────────────────────────────────────────────────

# sweep_reclaim's context (ATR period=14, eq_lookback default 20 — neither
# is a gridded parameter) only ever depends on a bounded trailing window.
# Slicing to this tail BEFORE calling into context.py turns what would
# otherwise be an O(bars-so-far) recompute on every single bar (i.e.
# O(session_length^2) per session) into O(1) per bar, with IDENTICAL
# results — ctx.atr's rolling(14) and _equal_extrema's tail(eq_lookback)
# already only look at their own trailing window; we are just not handing
# them a much bigger array to re-scan for no reason.
_SWEEP_CONTEXT_TAIL_BARS = 80


def _inferred_bar_minutes(bars: pd.DataFrame) -> int:
    """Best-effort bar size from the last two timestamps. A single bar
    (or a non-positive delta) is treated as 1-minute — the engine's
    native feed — so we never skip a needed resample."""
    if bars is None or len(bars) < 2:
        return 1
    delta = (bars.index[-1] - bars.index[-2]).total_seconds() / 60.0
    if delta <= 0:
        return 1
    return max(1, int(round(delta)))


def _decision_bars_for_absorption(bars_so_far: pd.DataFrame, minutes: int) -> pd.DataFrame | None:
    """OHLCV at the absorption decision chart, or None if this 1m prefix
    has not yet closed an N-minute bin. `minutes <= 1` is the raw 1m
    feed. If `bars_so_far` is already at decision frequency (live
    scheduler path) it is returned unchanged — do not resample twice."""
    if minutes <= 1:
        return bars_so_far
    if bars_so_far is None or bars_so_far.empty:
        return None
    if _inferred_bar_minutes(bars_so_far) >= minutes:
        return bars_so_far
    if not ctx.session_bin_just_closed(bars_so_far, minutes):
        return None
    decision = ctx.closed_session_bars(bars_so_far, minutes)
    return None if decision.empty else decision


def _evaluate_signal(
    signal_name: str,
    bars_so_far: pd.DataFrame,
    symbol: str,
    params: dict,
    cfg: IntradayBacktestConfig,
    prior_day_bars: pd.DataFrame | None,
    prior_close: float | None,
    session_vwap_series: pd.Series | None = None,
    session_opening_range=None,
    session_vwap_bands: pd.DataFrame | None = None,
    session_atr: pd.Series | None = None,
    is_trending_day: bool = True,
    gex_snapshot=None,
    ticks_so_far: pd.DataFrame | None = None,
    prior_sessions: list | None = None,
    preopen_env=None,
) -> MicroSignal | None:
    keys = SIGNAL_PARAM_KEYS[signal_name]
    kwargs = {k: params[k] for k in keys if k in params and k not in _SESSION_LEVEL_PARAM_KEYS}

    if signal_name == "sweep_reclaim":
        lookback_bars = bars_so_far.tail(_SWEEP_CONTEXT_TAIL_BARS)
        levels = ctx.compute_liquidity_levels(
            lookback_bars, prior_day_bars, eq_lookback=cfg.eq_lookback, eq_atr_mult=cfg.eq_atr_mult,
        )
        atr_series = ctx.atr(lookback_bars)
        return evaluate_sweep_reclaim(lookback_bars, levels, atr_series, symbol=symbol, **kwargs)

    if signal_name == "fvg_retest":
        return evaluate_fvg_retest(bars_so_far, symbol=symbol, **kwargs)

    if signal_name == "orb_vwap":
        or_minutes = int(kwargs.get("or_minutes", 15))
        # VWAP is a pure running (cumulative) statistic: its value at
        # position i depends only on bars[0..i], so a full-session VWAP
        # series computed ONCE and then sliced to a prefix is mathematically
        # identical to recomputing session_vwap on that prefix every bar —
        # this is a pure performance optimization, not a lookahead
        # shortcut. Same reasoning for the opening range once we're past
        # its window (see run_symbol_day's precomputation comment).
        if session_vwap_series is not None:
            vwap_series = session_vwap_series.iloc[: len(bars_so_far)]
        else:
            vwap_series = ctx.session_vwap(bars_so_far)
        orange = session_opening_range if session_opening_range is not None else ctx.opening_range(bars_so_far, minutes=or_minutes)
        # ATR is only needed when the stop-buffer lever is switched on; when
        # it is off (the default) nothing precomputes session_atr for this
        # signal and `atr_series=None` keeps the original raw-OR-extreme stop.
        atr_series = session_atr.iloc[: len(bars_so_far)] if session_atr is not None else None
        return evaluate_orb_vwap(bars_so_far, orange, vwap_series, symbol=symbol, prior_close=prior_close,
                                 atr_series=atr_series, **kwargs)

    if signal_name == "orb_vwap_regime":
        or_minutes = int(kwargs.get("or_minutes", 15))
        if session_vwap_series is not None:
            vwap_series = session_vwap_series.iloc[: len(bars_so_far)]
        else:
            vwap_series = ctx.session_vwap(bars_so_far)
        orange = session_opening_range if session_opening_range is not None else ctx.opening_range(bars_so_far, minutes=or_minutes)
        return evaluate_orb_vwap_regime(
            bars_so_far, orange, vwap_series, is_trending_day, symbol=symbol, prior_close=prior_close, **kwargs,
        )

    if signal_name == "vwap_band_fade":
        # vwap_bands (like session_vwap above) is a purely trailing/
        # cumulative statistic — slicing a full-session precomputation to
        # a prefix is identical to recomputing it fresh on that prefix.
        if session_vwap_bands is not None:
            bands = session_vwap_bands.iloc[: len(bars_so_far)]
        else:
            bands = ctx.vwap_bands(bars_so_far)
        atr_series = session_atr.iloc[: len(bars_so_far)] if session_atr is not None else ctx.atr(bars_so_far)
        return evaluate_vwap_band_fade(bars_so_far, bands, atr_series, symbol=symbol, **kwargs)

    if signal_name == "vp_breakout":
        lookback_bars = bars_so_far.tail(_SWEEP_CONTEXT_TAIL_BARS)
        # include_eq_levels=False: vp_breakout's target_resistance_levels/
        # target_support_levels never read eq_highs/eq_lows (see those
        # functions' docstring in context.py), so skip the ATR +
        # extrema-clustering work that field would otherwise cost every
        # bar — this was profiled as ~2/3 of this signal's total per-bar
        # runtime for zero effect on its actual decisions.
        levels = ctx.compute_liquidity_levels(
            lookback_bars, prior_day_bars, eq_lookback=cfg.eq_lookback, eq_atr_mult=cfg.eq_atr_mult,
            include_eq_levels=False,
        )
        # Unlike sweep_reclaim's ATR (matched to the tail-truncated
        # `lookback_bars` it also uses for level detection), vp_breakout's
        # volume-profile computation needs the FULL bars_so_far (a session-
        # cumulative value area, not a trailing-window one — see
        # vp_breakout.py's module docstring), so its ATR must be aligned to
        # that same full-session frame, not the truncated one.
        atr_series = session_atr.iloc[: len(bars_so_far)] if session_atr is not None else ctx.atr(bars_so_far)
        return evaluate_vp_breakout(bars_so_far, levels, atr_series, symbol=symbol, **kwargs)

    if signal_name == "l2_absorption":
        return evaluate_l2_absorption(bars_so_far, symbol=symbol, **kwargs)

    if signal_name == "absorption_breakout":
        decision = _decision_bars_for_absorption(bars_so_far, int(cfg.chart_minutes))
        if decision is None:
            return None
        return evaluate_absorption_breakout(decision, symbol=symbol, **kwargs)

    if signal_name == "auction_reclaim":
        return evaluate_auction_reclaim(
            bars_so_far, prior_day_bars=prior_day_bars, symbol=symbol,
            gex_snapshot=gex_snapshot, ticks_so_far=ticks_so_far,
            prior_sessions=prior_sessions, preopen_env=preopen_env,
            chart_minutes=int(cfg.chart_minutes), **kwargs,
        )

    if signal_name == "vsa_effort":
        return evaluate_vsa_effort(
            bars_so_far, prior_day_bars=prior_day_bars, symbol=symbol,
            gex_snapshot=gex_snapshot, chart_minutes=int(cfg.chart_minutes), **kwargs,
        )

    if signal_name == "vsa_no_demand":
        kwargs = {**kwargs, **_volume_book_filter_kwargs(cfg, _VSA_FILTER_KEYS)}
        return evaluate_vsa_no_demand(
            bars_so_far, prior_day_bars=prior_day_bars, symbol=symbol,
            gex_snapshot=gex_snapshot, chart_minutes=int(cfg.chart_minutes), **kwargs,
        )

    if signal_name == "obv_divergence":
        kwargs = {**kwargs, **_volume_book_filter_kwargs(cfg, _OBV_FILTER_KEYS)}
        return evaluate_obv_divergence(
            bars_so_far, prior_day_bars=prior_day_bars, symbol=symbol,
            gex_snapshot=gex_snapshot, chart_minutes=int(cfg.chart_minutes), **kwargs,
        )

    raise ValueError(f"intraday_engine: unknown signal_name {signal_name!r}")


def _session_gex_and_ticks(
    signal_name: str,
    symbol: str,
    session_ts: pd.Timestamp | None,
    gex_snapshot=None,
    day_ticks: pd.DataFrame | None = None,
):
    """Optional Creamer-context loaders. Missing files stay None — never
    invented. auction_reclaim loads GEX + ticks; vsa_effort /
    vsa_no_demand / obv_divergence load GEX only (their tape proxy is
    5-minute bar volume, not footprint)."""
    if signal_name not in ("auction_reclaim", "vsa_effort", "vsa_no_demand", "obv_divergence"):
        return gex_snapshot, day_ticks
    if gex_snapshot is None and symbol and session_ts is not None:
        from ..data.gex_cache import load_gex_env

        gex_snapshot = load_gex_env(symbol, session_ts)
    if signal_name == "auction_reclaim" and day_ticks is None and symbol and session_ts is not None:
        from ..data.tick_cache import load_trade_ticks

        day_ticks = load_trade_ticks(symbol, session_ts)
    return gex_snapshot, day_ticks


# ── fill / exit checks ───────────────────────────────────────────────────────

def _check_fill(
    order: PendingOrder, bar: pd.Series, bar_time: pd.Timestamp, cfg: IntradayBacktestConfig,
) -> tuple[str, Position | None]:
    sig = order.signal
    if sig.order_type == "next_open":
        fill_ref = float(bar["open"])
    else:
        if sig.expiry_time is not None and bar_time > sig.expiry_time:
            return "expired", None
        lo, hi = float(bar["low"]), float(bar["high"])
        if not (lo <= sig.entry_price <= hi):
            return "pending", None
        fill_ref = sig.entry_price

    shares = _position_size(fill_ref, sig.stop_price, cfg)
    if shares <= 0:
        return "invalid", None
    fill_price = _slippage_price(fill_ref, sig.direction, shares, float(bar["volume"]), cfg, is_entry=True, symbol=sig.symbol)
    commission = _commission(shares, cfg)
    return "filled", Position(
        symbol=sig.symbol, strategy=sig.strategy, direction=sig.direction,
        entry_time=bar_time, entry_price=fill_price, shares=shares,
        stop_price=sig.stop_price, target_price=sig.target_price, entry_commission=commission,
        signal_time=sig.signal_time,
    )


def _check_exit(
    position: Position, bar: pd.Series, elapsed_minutes: float, cfg: IntradayBacktestConfig,
) -> tuple[str, float] | None:
    lo, hi, close = float(bar["low"]), float(bar["high"]), float(bar["close"])
    if position.direction == "long":
        if lo <= position.stop_price:
            return "stop", position.stop_price
        if position.target_price is not None and hi >= position.target_price:
            return "target", position.target_price
        unrealized = close - position.entry_price
    else:
        if hi >= position.stop_price:
            return "stop", position.stop_price
        if position.target_price is not None and lo <= position.target_price:
            return "target", position.target_price
        unrealized = position.entry_price - close

    if elapsed_minutes >= cfg.time_stop_minutes and unrealized <= 0:
        return "time_stop", close
    return None


def _close_position(
    position: Position, exit_ref_price: float, exit_time: pd.Timestamp, reason: str,
    bar_volume: float, cfg: IntradayBacktestConfig,
) -> IntradayTrade:
    exit_price = _slippage_price(exit_ref_price, position.direction, position.shares, bar_volume, cfg, is_entry=False, symbol=position.symbol)
    exit_commission = _commission(position.shares, cfg)
    if position.direction == "long":
        gross = (exit_price - position.entry_price) * position.shares
    else:
        gross = (position.entry_price - exit_price) * position.shares
    costs = position.entry_commission + exit_commission
    return IntradayTrade(
        symbol=position.symbol, strategy=position.strategy, direction=position.direction,
        entry_time=position.entry_time, entry_price=position.entry_price,
        exit_time=exit_time, exit_price=exit_price, exit_reason=reason,
        shares=position.shares, gross_pnl=gross, costs=costs, net_pnl=gross - costs,
        signal_time=position.signal_time,
    )


# ── per-symbol, per-day event loop ───────────────────────────────────────────

def scan_signals_for_session(
    signal_name: str,
    bars_today: pd.DataFrame,
    params: dict,
    cfg: IntradayBacktestConfig | None = None,
    prior_day_bars: pd.DataFrame | None = None,
    symbol: str = "",
    gex_snapshot=None,
    day_ticks: pd.DataFrame | None = None,
    prior_sessions: list | None = None,
    preopen_env=None,
) -> list[MicroSignal]:
    """Every signal `_evaluate_signal` would fire on `bars_today`, evaluated
    bar-by-bar with the SAME causality contract as run_symbol_day — a
    read-only diagnostic scan (no fill simulation, no position/pending-order
    state machine), used to answer "where would this pattern have fired"
    for chart overlays (dashboard/app.py's /api/chart/{symbol}/context)
    without needing to run a full backtest. Unlike run_symbol_day this does
    NOT skip bars while "in a position" — every bar is checked independently,
    so a busy session can legitimately report signals that a real backtest
    would never have filled (it was already in a trade). That's intentional
    for a "where did the pattern occur" diagnostic; it is NOT a trade list.
    For the same reason it does NOT apply orb_vwap's
    `max_entries_per_session` budget (a fill-path rule, enforced in
    run_symbol_day) — every qualifying break of the session is reported."""
    cfg = cfg or IntradayBacktestConfig()
    if len(bars_today) < 2:
        return []

    session_vwap_series: pd.Series | None = None
    session_opening_range = None
    if signal_name in ("orb_vwap", "orb_vwap_regime"):
        or_minutes = int(params.get("or_minutes", 15))
        session_vwap_series = ctx.session_vwap(bars_today)
        session_opening_range = ctx.opening_range(bars_today, minutes=or_minutes)
    session_vwap_bands: pd.DataFrame | None = None
    session_atr: pd.Series | None = None
    if signal_name == "orb_vwap" and float(params.get("stop_atr_buffer_mult") or 0.0) > 0:
        session_atr = ctx.atr(bars_today)
    if signal_name == "vwap_band_fade":
        session_vwap_bands = ctx.vwap_bands(bars_today)
        session_atr = ctx.atr(bars_today)
    if signal_name == "vp_breakout":
        session_atr = ctx.atr(bars_today)

    prior_close = float(prior_day_bars["close"].iloc[-1]) if prior_day_bars is not None and len(prior_day_bars) else None
    session_ts = bars_today.index[0] if len(bars_today) else None
    gex_snapshot, day_ticks = _session_gex_and_ticks(
        signal_name, symbol, session_ts, gex_snapshot=gex_snapshot, day_ticks=day_ticks,
    )
    if signal_name == "auction_reclaim" and preopen_env is None:
        sessions = list(prior_sessions) if prior_sessions else []
        if prior_day_bars is not None and not prior_day_bars.empty and not sessions:
            sessions = [prior_day_bars]
        preopen_env = preopen_1h_environment(sessions)

    # orb_vwap_regime's regime gate needs multi-day trailing daily closes
    # (see _daily_trending_flags) that a single day's `bars_today` cannot
    # provide — this diagnostic scan is single-session only (dashboard
    # chart overlays), so it intentionally does NOT apply the regime gate
    # (is_trending_day defaults to True, i.e. unfiltered "where would the
    # underlying orb_vwap pattern have fired" view). The real WFO/promotion
    # pipeline (run_intraday_backtest below) DOES apply the gate.
    signals: list[MicroSignal] = []
    for i in range(1, len(bars_today)):
        bars_so_far = bars_today.iloc[: i + 1]
        sig = _evaluate_signal(
            signal_name, bars_so_far, symbol, params, cfg, prior_day_bars, prior_close,
            session_vwap_series=session_vwap_series, session_opening_range=session_opening_range,
            session_vwap_bands=session_vwap_bands, session_atr=session_atr,
            gex_snapshot=gex_snapshot, ticks_so_far=day_ticks,
            prior_sessions=prior_sessions, preopen_env=preopen_env,
        )
        if sig is not None:
            signals.append(sig)
    return signals


def run_symbol_day(
    symbol: str,
    day_bars: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None,
    signal_name: str,
    params: dict,
    cfg: IntradayBacktestConfig,
    prior_close: float | None = None,
    is_trending_day: bool = True,
    gex_snapshot=None,
    day_ticks: pd.DataFrame | None = None,
    prior_sessions: list | None = None,
    preopen_env=None,
) -> tuple[list[IntradayTrade], int, int]:
    """One session's worth of the event loop for one symbol. Returns
    (trades, signals_emitted, signals_filled). `is_trending_day` only
    matters for signal_name == "orb_vwap_regime" (see that module's
    docstring) — every other signal ignores it; the default of True keeps
    existing callers/tests for the other signals unaffected."""
    trades: list[IntradayTrade] = []
    signals_emitted = 0
    signals_filled = 0

    if day_bars.empty:
        return trades, signals_emitted, signals_filled

    flatten_cutoff = day_bars.index[-1] - pd.Timedelta(minutes=cfg.flatten_buffer_minutes)
    trading_bars = day_bars.loc[day_bars.index <= flatten_cutoff]
    if len(trading_bars) < 2:
        return trades, signals_emitted, signals_filled

    # Precomputed ONCE per session (see _evaluate_signal's docstring on why
    # this is safe, not a lookahead shortcut) instead of once per bar —
    # this is what keeps a full trading day's evaluate_orb_vwap calls at
    # O(session_length) total instead of O(session_length^2).
    session_vwap_series: pd.Series | None = None
    session_opening_range = None
    if signal_name in ("orb_vwap", "orb_vwap_regime"):
        or_minutes = int(params.get("or_minutes", 15))
        session_vwap_series = ctx.session_vwap(trading_bars)
        session_opening_range = ctx.opening_range(trading_bars, minutes=or_minutes)
    session_vwap_bands: pd.DataFrame | None = None
    session_atr: pd.Series | None = None
    if signal_name == "orb_vwap" and float(params.get("stop_atr_buffer_mult") or 0.0) > 0:
        session_atr = ctx.atr(trading_bars)
    if signal_name == "vwap_band_fade":
        session_vwap_bands = ctx.vwap_bands(trading_bars)
        session_atr = ctx.atr(trading_bars)
    if signal_name == "vp_breakout":
        session_atr = ctx.atr(trading_bars)

    session_ts = trading_bars.index[0]
    gex_snapshot, day_ticks = _session_gex_and_ticks(
        signal_name, symbol, session_ts, gex_snapshot=gex_snapshot, day_ticks=day_ticks,
    )
    if signal_name == "auction_reclaim" and preopen_env is None:
        sessions = list(prior_sessions) if prior_sessions else []
        if prior_day_bars is not None and not prior_day_bars.empty and not sessions:
            sessions = [prior_day_bars]
        preopen_env = preopen_1h_environment(sessions)
        # Sideways / unknown 1h chart: stand aside the whole session.
        if preopen_env.bias is None:
            return trades, signals_emitted, signals_filled

    # Session-scoped entry budget (see _SESSION_LEVEL_PARAM_KEYS). None (the
    # default everywhere) = unlimited, i.e. the pre-existing behavior where a
    # signal may re-fire for the rest of the session every time the state
    # machine is free. The budget counts SIGNALS EMITTED, not fills: "the
    # first qualifying break of the session" is a statement about the
    # pattern, so a qualifying break that then fails to size into a position
    # still consumes the budget rather than silently granting a retry.
    max_entries = params.get("max_entries_per_session")
    max_entries = int(max_entries) if max_entries is not None else None
    entries_taken = 0

    pending: PendingOrder | None = None
    position: Position | None = None

    for i in range(1, len(trading_bars)):
        bar_time = trading_bars.index[i]
        bar = trading_bars.iloc[i]

        if position is not None:
            elapsed = (bar_time - position.entry_time).total_seconds() / 60.0
            exit_info = _check_exit(position, bar, elapsed, cfg)
            if exit_info is not None:
                reason, exit_ref_price = exit_info
                trades.append(_close_position(position, exit_ref_price, bar_time, reason, float(bar["volume"]), cfg))
                position = None

        if position is None and pending is not None:
            status, new_position = _check_fill(pending, bar, bar_time, cfg)
            if status == "filled":
                position = new_position
                signals_filled += 1
                pending = None
                # The fill happened at this bar's open (or, for a limit
                # order, somewhere inside this bar's range); the REST of
                # this same bar's high/low still chronologically follows
                # that fill point, so a same-bar stop/target check here is
                # NOT lookahead — it is the realistic "you could have been
                # stopped out in the same minute you got filled" case.
                exit_info = _check_exit(position, bar, 0.0, cfg)
                if exit_info is not None:
                    reason, exit_ref_price = exit_info
                    trades.append(_close_position(position, exit_ref_price, bar_time, reason,
                                                    float(bar["volume"]), cfg))
                    position = None
            elif status in ("expired", "invalid"):
                pending = None
            # "pending": keep waiting for a future bar to touch the limit price.

        if position is None and pending is None and (max_entries is None or entries_taken < max_entries):
            bars_so_far = trading_bars.iloc[: i + 1]
            sig = _evaluate_signal(
                signal_name, bars_so_far, symbol, params, cfg, prior_day_bars, prior_close,
                session_vwap_series=session_vwap_series, session_opening_range=session_opening_range,
                session_vwap_bands=session_vwap_bands, session_atr=session_atr,
                is_trending_day=is_trending_day,
                gex_snapshot=gex_snapshot, ticks_so_far=day_ticks,
                prior_sessions=prior_sessions, preopen_env=preopen_env,
            )
            if sig is not None:
                signals_emitted += 1
                entries_taken += 1
                pending = PendingOrder(signal=sig, created_at=bar_time)

    if position is not None:
        last_bar = trading_bars.iloc[-1]
        last_time = trading_bars.index[-1]
        trades.append(_close_position(position, float(last_bar["close"]), last_time, "eod_flatten",
                                       float(last_bar["volume"]), cfg))

    return trades, signals_emitted, signals_filled


# ── multi-day, multi-symbol orchestrator ─────────────────────────────────────

def run_intraday_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    signal_name: str,
    params: dict,
    cfg: IntradayBacktestConfig | None = None,
) -> IntradayBacktestReport:
    """`bars_by_symbol[symbol]` must already be restricted to the desired
    [start, end) window and RTH-only 1-minute bars, tz-naive ET index (the
    same contract as python/data/intraday_cache.get_cached_intraday_panel).
    Each symbol's sessions are walked independently and chronologically;
    `prior_day_bars`/`prior_close` for YDH/YDL and the gap-trap rule come
    from that SAME symbol's immediately preceding session in this window
    (not looked up externally), so the very first day of a window has no
    prior-day context — an honest, unavoidable edge condition, not a bug."""
    cfg = cfg or IntradayBacktestConfig()
    report = IntradayBacktestReport()

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        session_dates = sorted(set(bars.index.normalize()))
        prior_day_bars: pd.DataFrame | None = None
        prior_close: float | None = None
        recent_sessions: list[pd.DataFrame] = []
        # Computed ONCE per symbol from ALL of this call's bars (see
        # _daily_trending_flags docstring) — every session date's flag is
        # looked up below, defaulting to False (no trade) for dates without
        # enough trailing daily-close history for a regime label yet.
        trending_by_date: dict[pd.Timestamp, bool] = (
            _daily_trending_flags(bars) if signal_name == "orb_vwap_regime" else {}
        )

        for d in session_dates:
            day_bars = bars.loc[bars.index.normalize() == d]
            if len(day_bars) < 5:
                if not day_bars.empty:
                    prior_day_bars = day_bars
                    prior_close = float(day_bars["close"].iloc[-1])
                    recent_sessions.append(day_bars)
                continue

            env_sessions = recent_sessions[-2:] if recent_sessions else None
            trades, emitted, filled = run_symbol_day(
                symbol, day_bars, prior_day_bars, signal_name, params, cfg, prior_close=prior_close,
                is_trending_day=trending_by_date.get(d, False),
                prior_sessions=env_sessions,
            )
            report.trades.extend(trades)
            report.signals_emitted += emitted
            report.signals_filled += filled
            prior_day_bars = day_bars
            prior_close = float(day_bars["close"].iloc[-1])
            recent_sessions.append(day_bars)

    return report


# ── metrics (same output contract as optimize.py's _metrics_from_returns) ───

def metrics_from_report(report: IntradayBacktestReport, capital: float) -> dict:
    daily = report.daily_pnl()
    if not daily:
        returns = pd.Series(dtype=float)
    else:
        dates = sorted(daily.keys())
        returns = pd.Series([daily[d] / capital for d in dates], index=pd.DatetimeIndex(dates))

    sharpe = 0.0
    if len(returns) >= 2 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    max_dd = 0.0
    if len(returns):
        equity = (1.0 + returns).cumprod()
        max_dd = float((equity / equity.cummax() - 1.0).min())

    gross_profit = float(sum(t.net_pnl for t in report.trades if t.net_pnl > 0))
    gross_loss = float(sum(-t.net_pnl for t in report.trades if t.net_pnl < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    # PRE-COMMISSION profit factor. NOT pre-cost: `IntradayTrade.gross_pnl`
    # is computed in `_close_position` from `position.entry_price` and
    # `exit_price`, and BOTH of those are `_slippage_price` outputs — so
    # half-spread and participation impact are already inside `gross_pnl`.
    # The only thing this differs from `profit_factor` by is
    # `IntradayTrade.costs`, which `_close_position` sets to
    # `entry_commission + exit_commission` alone.
    #
    # Slippage dominates commission for this engine's position sizes (1%
    # risk / a tight ATR stop routinely sizes into the `max_notional_pct`
    # cap, so a round trip is a few bps of a six-figure notional against a
    # per-share commission of half a cent). Measured on vsa_no_demand 5m
    # over 2026-07: pre-cost net $2,104, of which slippage took $1,950 and
    # commission $116 — i.e. this "gross" figure (PF 1.026) had already
    # absorbed 93% of the total cost drag, while the genuinely pre-cost PF
    # was 1.471.
    #
    # So the pair (profit_factor_gross, profit_factor) does NOT separate
    # "no edge at all" from "edge exists but costs eat it" — both members
    # are post-slippage and they differ only by commission, which is why
    # they track each other within a few percent on every signal measured
    # so far. Answering that question requires a separate run with
    # `half_spread_bps=0` and `impact_bps_per_participation=0`; any
    # conclusion of the form "costs are only N% of the loss" that was
    # derived from `total_costs / gross_pnl` is measuring commission only
    # and understates the true drag (this includes
    # backtests/reports/l2_absorption_validation_report.md, whose
    # failure-mode diagnosis this diagnostic was originally added for and
    # which should be re-derived against a zero-slippage baseline).
    trade_gross_profit = float(sum(t.gross_pnl for t in report.trades if t.gross_pnl > 0))
    trade_gross_loss = float(sum(-t.gross_pnl for t in report.trades if t.gross_pnl < 0))
    if trade_gross_loss > 0:
        profit_factor_gross = trade_gross_profit / trade_gross_loss
    else:
        profit_factor_gross = float("inf") if trade_gross_profit > 0 else 0.0

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trades": len(report.trades),
        "total_net_pnl": float(sum(t.net_pnl for t in report.trades)),
        "n_days": int(len(returns)),
        "daily_returns": [float(r) for r in returns.tolist()],
        "signals_emitted": report.signals_emitted,
        "signals_filled": report.signals_filled,
        # cost-adjusted (net_pnl already includes slippage + commission) —
        # configs/goal.yaml's intraday.min_cost_adjusted_profit_factor gate.
        "profit_factor": profit_factor,
        # diagnostic only, NOT gated on — see comment above.
        "profit_factor_gross": profit_factor_gross,
        "gross_pnl": float(sum(t.gross_pnl for t in report.trades)),
        "total_costs": float(sum(t.costs for t in report.trades)),
    }
