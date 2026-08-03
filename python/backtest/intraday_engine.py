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

from ..microstructure import context as ctx
from ..microstructure.signals import MicroSignal
from ..microstructure.signals.fvg_retest import evaluate_fvg_retest
from ..microstructure.signals.l2_absorption import evaluate_l2_absorption
from ..microstructure.signals.orb_vwap import evaluate_orb_vwap
from ..microstructure.signals.sweep_reclaim import evaluate_sweep_reclaim

SIGNAL_PARAM_KEYS = {
    "sweep_reclaim": ["sweep_min_atr", "reclaim_bars", "stop_atr_mult"],
    "fvg_retest": ["vol_mult", "entry_pct", "expiry_bars"],
    "orb_vwap": ["or_minutes", "vwap_side_filter"],
    # l2_absorption (S4) — deliberately NOT run through run_symbol_day's
    # fill/P&L simulation or scripts/run_intraday_backtest.py's WFO gate
    # (see l2_absorption.py's module docstring: bar-only proxy, no L2
    # confirmation yet). Only scan_signals_for_session below dispatches it,
    # for observe-only detection/logging.
    "l2_absorption": ["volume_mult", "touch_atr_mult", "stop_atr_mult"],
}


@dataclass
class IntradayBacktestConfig:
    capital: float = 1_000_000.0
    risk_per_trade_pct: float = 0.01
    max_notional_pct: float = 0.20  # position sizing ceiling — see _position_size
    half_spread_bps: float = 2.0
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
    cfg: IntradayBacktestConfig, is_entry: bool,
) -> float:
    """half-spread + a participation-scaled impact term, both in basis
    points of price (never a raw price multiplier — a position that
    happens to be large relative to a thin bar's volume must cost more
    basis points, not become a different-magnitude number entirely).
    `max_participation` caps the (shares / bar_volume) ratio fed into the
    impact term as a second line of defense against pathological cases
    (e.g. a very tight stop producing a position that is still large
    relative to an unusually thin fill bar)."""
    half_spread_bps = cfg.half_spread_bps
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
) -> MicroSignal | None:
    keys = SIGNAL_PARAM_KEYS[signal_name]
    kwargs = {k: params[k] for k in keys if k in params}

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
        return evaluate_orb_vwap(bars_so_far, orange, vwap_series, symbol=symbol, prior_close=prior_close, **kwargs)

    if signal_name == "l2_absorption":
        return evaluate_l2_absorption(bars_so_far, symbol=symbol, **kwargs)

    raise ValueError(f"intraday_engine: unknown signal_name {signal_name!r}")


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
    fill_price = _slippage_price(fill_ref, sig.direction, shares, float(bar["volume"]), cfg, is_entry=True)
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
    exit_price = _slippage_price(exit_ref_price, position.direction, position.shares, bar_volume, cfg, is_entry=False)
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
    for a "where did the pattern occur" diagnostic; it is NOT a trade list."""
    cfg = cfg or IntradayBacktestConfig()
    if len(bars_today) < 2:
        return []

    session_vwap_series: pd.Series | None = None
    session_opening_range = None
    if signal_name == "orb_vwap":
        or_minutes = int(params.get("or_minutes", 15))
        session_vwap_series = ctx.session_vwap(bars_today)
        session_opening_range = ctx.opening_range(bars_today, minutes=or_minutes)

    prior_close = float(prior_day_bars["close"].iloc[-1]) if prior_day_bars is not None and len(prior_day_bars) else None

    signals: list[MicroSignal] = []
    for i in range(1, len(bars_today)):
        bars_so_far = bars_today.iloc[: i + 1]
        sig = _evaluate_signal(
            signal_name, bars_so_far, "", params, cfg, prior_day_bars, prior_close,
            session_vwap_series=session_vwap_series, session_opening_range=session_opening_range,
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
) -> tuple[list[IntradayTrade], int, int]:
    """One session's worth of the event loop for one symbol. Returns
    (trades, signals_emitted, signals_filled)."""
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
    if signal_name == "orb_vwap":
        or_minutes = int(params.get("or_minutes", 15))
        session_vwap_series = ctx.session_vwap(trading_bars)
        session_opening_range = ctx.opening_range(trading_bars, minutes=or_minutes)

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

        if position is None and pending is None:
            bars_so_far = trading_bars.iloc[: i + 1]
            sig = _evaluate_signal(
                signal_name, bars_so_far, symbol, params, cfg, prior_day_bars, prior_close,
                session_vwap_series=session_vwap_series, session_opening_range=session_opening_range,
            )
            if sig is not None:
                signals_emitted += 1
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

        for d in session_dates:
            day_bars = bars.loc[bars.index.normalize() == d]
            if len(day_bars) < 5:
                if not day_bars.empty:
                    prior_day_bars = day_bars
                    prior_close = float(day_bars["close"].iloc[-1])
                continue

            trades, emitted, filled = run_symbol_day(
                symbol, day_bars, prior_day_bars, signal_name, params, cfg, prior_close=prior_close,
            )
            report.trades.extend(trades)
            report.signals_emitted += emitted
            report.signals_filled += filled
            prior_day_bars = day_bars
            prior_close = float(day_bars["close"].iloc[-1])

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
    }
