"""
Track 2 (backtests/reports/alt_universe_frequency_exploration.md) — a DAILY-
bar analog of `orb_vwap`'s continuation thesis, testing whether going SLOWER
(daily bars, multi-day holds) fixes the cost/frequency mismatch that retired
`orb_vwap` and three other 1-minute microstructure signals (see
backtests/reports/strategy_review_summary.md and
backtests/reports/orb_vwap_rescue_report.md).

This is deliberately a NEW signal, not a reuse of `orb_vwap`'s 1-minute code
(the opening-range concept does not carry over to daily bars — there is no
"opening range" on a daily bar). What carries over is the THESIS: a fresh
breakout of an established range, in the direction of the break, is more
often followed through than not. Here "established range" is the prior
`range_days` trading days' high/low (instead of the first `or_minutes` of
one session), and the breakout is a daily CLOSE outside that range (instead
of a 1-minute bar crossing the opening range).

Two lessons carried over directly from `orb_vwap_rescue_report.md`'s four
levers, applied from the start rather than discovered by a later rescue:
  1. Lever 3 (ATR stop buffer) — the stop here is ALWAYS `stop_atr_mult *
     ATR(atr_days)` away from entry, never the bare structural extreme,
     because the rescue found a raw structural stop is "inside the noise"
     and a buffered one recovers real profit factor.
  2. Lever 4 (profit target) — `target_r_multiple` is a first-class
     parameter (not `None`-by-default the way orb_vwap originally shipped),
     because the rescue found `target_price=None` caps reward without
     capping risk.
Lever 1 (tight-spread universe) and Lever 2 (entries-per-session cap) do not
apply here: this signal only ever takes ONE entry at a time per symbol (no
re-firing is possible — a new signal cannot be evaluated while a position is
open), so the "4.8 fills per session, only the first profitable" pathology
that motivated Lever 2 is structurally absent by construction, not fixed by
a knob.

No-lookahead / fill discipline: `evaluate_daily_breakout` is called with
`bars` truncated to `bars.iloc[:i+1]` (bars strictly through TODAY's close,
inclusive) and decides the trade using ONLY that data — the prior-N-day
range window explicitly excludes today's own bar. The caller
(`python/backtest/daily_breakout_engine.py`) fills the resulting signal at
TOMORROW's open, not today's close — correcting the "structural optimism"
`backtests/reports/pairs_scan_report.md` §9 flagged in the existing
same-bar-close-fill pairs engines (a genuine improvement made possible by
this being new code, not a retrofit).

Free parameters (Chan Ch.3 ceiling: 5), all consumed by the engine/signal
together: `range_days`, `hold_days` (engine), `stop_atr_mult`,
`target_r_multiple`. Four total, one under budget.
"""
from __future__ import annotations

import pandas as pd


def _wilder_atr(bars: pd.DataFrame, atr_days: int) -> float | None:
    """Simple (non-Wilder-smoothed) rolling-mean True Range over `atr_days`,
    using ONLY bars already in `bars` (caller is responsible for truncating
    to "no later than today" — see module docstring). Returns None if there
    is not enough history or the result is non-positive."""
    if len(bars) < atr_days + 1:
        return None
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.tail(atr_days).mean()
    if pd.isna(atr) or atr <= 0:
        return None
    return float(atr)


def evaluate_daily_breakout(
    bars: pd.DataFrame,
    range_days: int = 20,
    stop_atr_mult: float = 2.0,
    target_r_multiple: float | None = 3.0,
    atr_days: int = 14,
) -> dict | None:
    """Evaluate ONE candidate signal using `bars` truncated through TODAY's
    close (`bars.index[-1]` is "today"; every earlier row is available
    history). Returns None if no breakout, or a dict describing the
    candidate trade (direction/entry reference price/stop/target) for the
    engine to fill at TOMORROW's open.

    Breakout rule: today's CLOSE is strictly outside the [low, high] of the
    PRIOR `range_days` bars (today's own bar is excluded from the range
    window — this is a break of an already-established range, not a
    self-referential one). Long if close > prior range high, short if close
    < prior range low. No signal if inside the range, and no signal at all
    until there are at least `range_days + atr_days + 1` bars of history.
    """
    if len(bars) < range_days + atr_days + 1:
        return None

    prior = bars.iloc[:-1]
    range_window = prior.iloc[-range_days:]
    if len(range_window) < range_days:
        return None
    range_high = float(range_window["high"].max())
    range_low = float(range_window["low"].min())
    today_close = float(bars["close"].iloc[-1])

    if today_close > range_high:
        direction = "long"
    elif today_close < range_low:
        direction = "short"
    else:
        return None

    atr_now = _wilder_atr(bars, atr_days)
    if atr_now is None:
        return None

    stop = today_close - stop_atr_mult * atr_now if direction == "long" else today_close + stop_atr_mult * atr_now
    target = None
    if target_r_multiple is not None:
        risk = abs(today_close - stop)
        if risk > 0:
            target = (today_close + target_r_multiple * risk if direction == "long"
                      else today_close - target_r_multiple * risk)

    return {
        "direction": direction,
        "signal_close": today_close,
        "stop": stop,
        "target": target,
        "range_high": range_high,
        "range_low": range_low,
        "atr": atr_now,
    }
