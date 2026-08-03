"""
S3 — Opening Range Breakout + VWAP side filter.

Thesis (docs/microstructure_pivot_plan.md §1, S3): after the opening range
(first `or_minutes`) is established, a fresh breakout of the OR high/low
that is on the same side as session VWAP is traded with the breakout.
Hard rule (not a parameter — an operating rule, per the plan): no entries
during the opening range itself.

Gap-trap rule (also a hard rule, not a parameter): if the pre-market gap
direction is OPPOSITE the OR breakout direction, that combination reads as
a distribution/accumulation trap (a gap that fails to follow through) —
the trade is faded (taken in the OPPOSITE direction of the raw breakout)
instead of followed.

Free parameters (2, Chan discipline): or_minutes, vwap_side_filter.
"""
from __future__ import annotations

import pandas as pd

from ..context import OpeningRange
from . import MicroSignal


def evaluate_orb_vwap(
    bars: pd.DataFrame,
    opening_range: OpeningRange,
    vwap_series: pd.Series,
    symbol: str = "",
    or_minutes: int = 15,
    vwap_side_filter: bool = True,
    prior_close: float | None = None,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff "now" is the bar where price
    FIRST crosses the opening-range high/low (a single-bar crossing check
    against the PREVIOUS bar's close — not "still above", which would
    re-fire every bar). Returns None unconditionally while `now` falls
    inside the opening range window (hard no-entry rule)."""
    if len(bars) < 2 or opening_range.high is None or opening_range.low is None:
        return None

    now_time = bars.index[-1]
    if opening_range.end is not None and now_time < opening_range.end:
        return None

    prev_close = float(bars["close"].iloc[-2])
    now_close = float(bars["close"].iloc[-1])

    breakout_dir: str | None = None
    if prev_close <= opening_range.high < now_close:
        breakout_dir = "long"
    elif prev_close >= opening_range.low > now_close:
        breakout_dir = "short"
    if breakout_dir is None:
        return None

    if vwap_side_filter:
        if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
            return None
        now_vwap = float(vwap_series.iloc[-1])
        if breakout_dir == "long" and now_close <= now_vwap:
            return None
        if breakout_dir == "short" and now_close >= now_vwap:
            return None

    direction = breakout_dir
    trap_flag = False
    if prior_close is not None:
        today_open = float(bars["open"].iloc[0])
        gap_dir = "long" if today_open > prior_close else ("short" if today_open < prior_close else None)
        if gap_dir is not None and gap_dir != breakout_dir:
            direction = "short" if breakout_dir == "long" else "long"
            trap_flag = True

    stop = opening_range.low if direction == "long" else opening_range.high
    return MicroSignal(
        symbol=symbol, strategy="orb_vwap", direction=direction,
        signal_time=now_time, entry_price=now_close, stop_price=stop,
        target_price=None, order_type="next_open",
        context={"breakout_dir": breakout_dir, "trap_flag": trap_flag, "or_minutes": or_minutes,
                 "or_high": opening_range.high, "or_low": opening_range.low},
    )
