"""
RETIRED (2026-08-13) — verdict NO-GO, no further rescue work planned. This
was the closest of the six microstructure signals to passing, and it was
given a dedicated rescue investigation (four cost-to-edge levers) before
being retired. Honest bug-fixed baseline mean OOS Sharpe -6.734 (the
previously-reported +1.407 was substantially a stop-inversion bug — see
the "CORRECTNESS FIX" note below). Best rescue configuration reached only
cost-adjusted profit factor 1.003 in-sample against the 1.3 gate required
by configs/goal.yaml, and a 2-month untouched holdout scored WORSE (PF
0.713, net -$93,107), contradicting the in-sample gain. Full evidence:
backtests/reports/strategy_review_summary.md §0/§3.3 and
backtests/reports/orb_vwap_rescue_report.md (the authoritative report for
this signal). Code and tests are kept and still correct; this signal is
excluded from the default run of scripts/run_intraday_backtest.py (see its
RETIRED_SIGNALS) but remains importable and explicitly runnable/testable.
The four rescue levers (max_entries_per_session, stop_atr_buffer_mult,
target_r_multiple, and a universe restriction applied by the runner) stay
in the code, defaulted off, exactly as the rescue report left them — this
retirement banner and the correctness fix below are the only changes made
after that investigation; no entry/exit/scoring logic changed.

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

CORRECTNESS FIX (2026-08-13, backtests/reports/orb_vwap_rescue_report.md
§"Lever 0"): the gap-trap rule flips `direction` but the original stop
assignment was `opening_range.low if direction == "long" else
opening_range.high` — computed from the FLIPPED direction while the entry
price sits on the other side of the range. For every trap-faded trade that
put the stop on the FAVORABLE side of the entry (a "stop" above entry for a
long), which python/backtest/intraday_engine.py's `_check_exit` then
triggers almost immediately as a *profitable* exit labelled "stop".
Measured on 2025-08-01..2025-11-01, 20 symbols, or_minutes=5: 3,859 of
7,591 raw signals (50.8%) had an inverted stop, and 1,442 "stop"-labelled
exits booked +$342k of profit. This is the same class of defect
fvg_retest.py documents and already fixed for its target price. The fix
here is structural, not a new knob: the stop is placed at the more ADVERSE
of {the opening-range extreme, the breakout bar's own extreme}, which is a
no-op for an ordinary breakout (the bar's low sits inside the range for a
long break) and puts a trap fade's stop beyond the failed-breakout extreme
— the level that actually invalidates the fade.

Free parameters (5, exactly at the Chan Ch.3 ceiling enforced by
python/backtest/param_guard.py): or_minutes, vwap_side_filter,
max_entries_per_session, stop_atr_buffer_mult, target_r_multiple. The last
three all default to "off" (the pre-2026-08-13 behavior) and were added by
the rescue investigation above; `max_entries_per_session` is consumed by
the ENGINE (python/backtest/intraday_engine.py's run_symbol_day), not here,
because "how many entries has this session already taken" is state the
per-bar signal function deliberately does not carry.
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
    atr_series: pd.Series | None = None,
    stop_atr_buffer_mult: float = 0.0,
    target_r_multiple: float | None = None,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff "now" is the bar where price
    FIRST crosses the opening-range high/low (a single-bar crossing check
    against the PREVIOUS bar's close — not "still above", which would
    re-fire every bar). Returns None unconditionally while `now` falls
    inside the opening range window (hard no-entry rule).

    `stop_atr_buffer_mult` (default 0.0 = the original raw-OR-extreme stop)
    widens the stop by that many ATRs beyond the structural level, to stop
    ordinary 1-minute noise right after the break from chopping out
    otherwise-correct trades. Requires `atr_series` aligned to `bars`; when
    it is missing or NaN the buffer is skipped rather than guessed.

    `target_r_multiple` (default None = the original "no target at all",
    winners cut only by the engine's time-stop/EOD flatten) sets a profit
    target that many multiples of the stop distance away from the entry.
    """
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

    now_low = float(bars["low"].iloc[-1])
    now_high = float(bars["high"].iloc[-1])
    if direction == "long":
        stop = min(opening_range.low, now_low)
    else:
        stop = max(opening_range.high, now_high)

    atr_now = _atr_now(atr_series)
    if stop_atr_buffer_mult and atr_now is not None:
        buffer = stop_atr_buffer_mult * atr_now
        stop = stop - buffer if direction == "long" else stop + buffer

    target = None
    if target_r_multiple is not None:
        risk = abs(now_close - stop)
        if risk > 0:
            target = (now_close + target_r_multiple * risk if direction == "long"
                      else now_close - target_r_multiple * risk)

    return MicroSignal(
        symbol=symbol, strategy="orb_vwap", direction=direction,
        signal_time=now_time, entry_price=now_close, stop_price=stop,
        target_price=target, order_type="next_open",
        context={"breakout_dir": breakout_dir, "trap_flag": trap_flag, "or_minutes": or_minutes,
                 "or_high": opening_range.high, "or_low": opening_range.low,
                 "stop_atr_buffer_mult": stop_atr_buffer_mult,
                 "target_r_multiple": target_r_multiple},
    )


def _atr_now(atr_series: pd.Series | None) -> float | None:
    if atr_series is None or len(atr_series) == 0:
        return None
    value = atr_series.iloc[-1]
    if pd.isna(value) or float(value) <= 0:
        return None
    return float(value)
