"""
RETIRED (2026-08-13) — verdict NO-GO on configs/universe.yaml's 20-symbol
mega-cap universe at the time. cost_adjusted_profit_factor 0.195
(calibrated, full window) vs the 1.3 gate required by configs/goal.yaml —
the worst profit factor of all six microstructure signals reviewed at that
time. Also failed wfo_go (0% pass ratio), monte_carlo_p5_sharpe (-20.499),
and the mandatory 2x-slippage stress test (-$4.35M net). Full evidence:
backtests/reports/strategy_review_summary.md §3.2 and
backtests/reports/slippage_calibration_report.md.

2026-09-16 cost-to-edge rescue investigation (docs/fvg_retest_rescue_report.md,
mirroring orb_vwap_rescue_report.md's method): a fast zero-cost/real-cost
diagnostic found the SHIPPED params (vol_mult=2.0, entry_pct=0.5,
expiry_bars=10) have a genuinely positive GROSS profit factor (1.307, 48.7%
win rate, +72% of capital over 11 months/20 symbols) — unlike sweep_reclaim
(gross PF < 1 even at zero cost, a falsified thesis), this pattern's
directional edge is real. The gap to the 0.195 net figure above is almost
entirely trading-cost drag from firing ~21,000 times (4.7 trades/symbol/
session) with the target mirrored at a strict 1:1 R:R (see the
`target_r_multiple` docstring below for why 1:1 was called out as the
literal cause in the original retirement note). `max_entries_per_session`
and `target_r_multiple` were added below to test whether the SAME kind of
frequency/reward levers that partially rescued orb_vwap (0.573 -> 1.003,
still short of 1.3) can close a similar gap here, starting from a stronger
gross base. See docs/fvg_retest_rescue_report.md for the outcome — this
signal remains excluded from scripts/run_intraday_backtest.py's default run
(RETIRED_SIGNALS) regardless of that outcome; a GO there would be evidence
to review, not an automatic re-promotion.

S2 — Fair Value Gap (FVG) Retest.

Thesis (docs/microstructure_pivot_plan.md §1, S2): a 3-bar sequence where
the middle bar is a large, high-volume directional move that leaves a
price gap between bar1's extreme and bar3's opposite extreme. Price
statistically tends to retrace partway into that gap before continuing —
so a limit order is placed at `entry_pct` of the gap, in the direction of
the impulse, with a time-based expiry if the retest never comes.

Free parameters (5, Chan discipline): vol_mult, entry_pct, expiry_bars,
max_entries_per_session, target_r_multiple. The last two default to the
pre-2026-09-16 behavior exactly (unlimited entries; a target mirrored at
1:1 R:R) and were added by the rescue investigation above;
`max_entries_per_session` is consumed by the ENGINE
(python/backtest/intraday_engine.py's run_symbol_day), not here, same as
orb_vwap's identically-named lever.
"""
from __future__ import annotations

import pandas as pd

from . import MicroSignal


def evaluate_fvg_retest(
    bars: pd.DataFrame,
    symbol: str = "",
    vol_mult: float = 2.0,
    entry_pct: float = 0.5,
    expiry_bars: int = 10,
    volume_lookback: int = 20,
    target_r_multiple: float = 1.0,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff bars[-3], bars[-2], bars[-1]
    form a fresh FVG (bar2 = bars[-2] is the impulse bar). Only evaluates
    the LAST 3 bars of the given window plus `volume_lookback` bars of
    history for the volume-average baseline — never anything after
    bars.index[-1], so each call can only ever detect a gap that just
    completed, not one already several bars old (no re-firing on a stale
    gap on subsequent calls).

    `target_r_multiple` (default 1.0 = the ORIGINAL behavior, byte-for-byte:
    a target mirrored at exactly 1x the stop distance) sets the profit
    target at that many multiples of the stop distance away from entry,
    same lever/semantics as orb_vwap.evaluate_orb_vwap's parameter of the
    same name. Still guaranteed correct-side by construction (mirrors the
    stop, does not reference the impulse bar's close — see the 2026-07-30
    incident note this replaced), just at a configurable R instead of
    hardcoded 1R."""
    if len(bars) < 3:
        return None

    bar1, bar2, bar3 = bars.iloc[-3], bars.iloc[-2], bars.iloc[-1]
    now_time = bars.index[-1]

    baseline_window = bars["volume"].iloc[:-1].tail(volume_lookback)
    if baseline_window.empty:
        return None
    baseline_vol = float(baseline_window.mean())
    if baseline_vol <= 0 or float(bar2["volume"]) < vol_mult * baseline_vol:
        return None

    if len(bars) >= 2:
        bar_interval = bars.index[-1] - bars.index[-2]
    else:
        bar_interval = pd.Timedelta(minutes=1)

    if bar1["high"] < bar3["low"]:
        gap_low, gap_high = float(bar1["high"]), float(bar3["low"])
        direction = "long"
        stop = gap_low - (gap_high - gap_low)
    elif bar1["low"] > bar3["high"]:
        gap_high, gap_low = float(bar1["low"]), float(bar3["high"])
        direction = "short"
        stop = gap_high + (gap_high - gap_low)
    else:
        return None

    entry_price = gap_low + entry_pct * (gap_high - gap_low)
    # Target: mirror the stop distance onto the favorable side of entry, at
    # `target_r_multiple` x the stop distance — NOT the impulse bar's raw
    # close. Investigated after the 2026-07-30 backtest report's
    # catastrophic fvg_retest results: bar2 (the impulse bar) can close
    # anywhere within its own range — nothing guarantees bar2["close"] sits
    # on the profitable side of `entry_price`, let alone far enough past it
    # to justify calling it a "target". Empirically this produced
    # "target"-labeled exits with negative net P&L (entry above exit on a
    # long, or vice versa) because the reference price itself was on the
    # wrong side before slippage was even applied. A stop-mirrored target is
    # guaranteed correct-side by construction regardless of the multiple.
    risk = abs(entry_price - stop)
    target_price = (entry_price + target_r_multiple * risk if direction == "long"
                     else entry_price - target_r_multiple * risk)
    return MicroSignal(
        symbol=symbol, strategy="fvg_retest", direction=direction,
        signal_time=now_time, entry_price=entry_price, stop_price=stop,
        target_price=target_price, order_type="limit",
        expiry_time=now_time + expiry_bars * bar_interval,
        context={"gap_low": gap_low, "gap_high": gap_high, "impulse_volume": float(bar2["volume"]),
                 "baseline_volume": baseline_vol, "impulse_close": float(bar2["close"])},
    )
