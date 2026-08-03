"""
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
    return MicroSignal(
        symbol=symbol, strategy="fvg_retest", direction=direction,
        signal_time=now_time, entry_price=entry_price, stop_price=stop,
        target_price=float(bar2["close"]), order_type="limit",
        expiry_time=now_time + expiry_bars * bar_interval,
        context={"gap_low": gap_low, "gap_high": gap_high, "impulse_volume": float(bar2["volume"]),
                 "baseline_volume": baseline_vol},
    )
