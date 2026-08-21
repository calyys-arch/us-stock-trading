"""
RETIRED (2026-08-13) — verdict NO-GO, no further work planned.
cost_adjusted_profit_factor 0.195 (calibrated, full window) vs the 1.3 gate
required by configs/goal.yaml — the worst profit factor of all six
microstructure signals reviewed. Also fails wfo_go (0% pass ratio),
monte_carlo_p5_sharpe (-20.499), and the mandatory 2x-slippage stress test
(-$4.35M net). Root cause: a strict 1:1 risk:reward target on this
retracement-into-continuation pattern needs a win rate far above what the
pattern actually has on this universe/timeframe. Full evidence:
backtests/reports/strategy_review_summary.md §3.2 and
backtests/reports/slippage_calibration_report.md. Code and tests are kept
and still correct; this signal is excluded from the default run of
scripts/run_intraday_backtest.py (see its RETIRED_SIGNALS) but remains
importable and explicitly runnable/testable — the logic below is
unchanged by this retirement.

S2 — Fair Value Gap (FVG) Retest.

Thesis (docs/microstructure_pivot_plan.md §1, S2): a 3-bar sequence where
the middle bar is a large, high-volume directional move that leaves a
price gap between bar1's extreme and bar3's opposite extreme. Price
statistically tends to retrace partway into that gap before continuing —
so a limit order is placed at `entry_pct` of the gap, in the direction of
the impulse, with a time-based expiry if the retest never comes.

Free parameters (3, Chan discipline): vol_mult, entry_pct, expiry_bars.
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
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff bars[-3], bars[-2], bars[-1]
    form a fresh FVG (bar2 = bars[-2] is the impulse bar). Only evaluates
    the LAST 3 bars of the given window plus `volume_lookback` bars of
    history for the volume-average baseline — never anything after
    bars.index[-1], so each call can only ever detect a gap that just
    completed, not one already several bars old (no re-firing on a stale
    gap on subsequent calls)."""
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
    # Target: mirror the stop distance onto the favorable side of entry (1R),
    # NOT the impulse bar's raw close. Investigated after the 2026-07-30
    # backtest report's catastrophic fvg_retest results: bar2 (the impulse
    # bar) can close anywhere within its own range — nothing guarantees
    # bar2["close"] sits on the profitable side of `entry_price`, let alone
    # far enough past it to justify calling it a "target". Empirically this
    # produced "target"-labeled exits with negative net P&L (entry above
    # exit on a long, or vice versa) because the reference price itself was
    # on the wrong side before slippage was even applied. A stop-mirrored
    # target is guaranteed correct-side by construction and keeps the
    # trade's designed risk:reward at 1:1 without adding a free parameter.
    risk = abs(entry_price - stop)
    target_price = entry_price + risk if direction == "long" else entry_price - risk
    return MicroSignal(
        symbol=symbol, strategy="fvg_retest", direction=direction,
        signal_time=now_time, entry_price=entry_price, stop_price=stop,
        target_price=target_price, order_type="limit",
        expiry_time=now_time + expiry_bars * bar_interval,
        context={"gap_low": gap_low, "gap_high": gap_high, "impulse_volume": float(bar2["volume"]),
                 "baseline_volume": baseline_vol, "impulse_close": float(bar2["close"])},
    )
