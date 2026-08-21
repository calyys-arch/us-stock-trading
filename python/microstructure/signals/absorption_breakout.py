"""
NEW signal (2026-08-14) — Absorption Failure / Breakout Continuation.

This is the genuine "Option A" investigation from `l2_absorption`'s
retirement pivot list (`backtests/reports/l2_absorption_validation_report.md`,
`backtests/reports/signal_status.md`): `l2_absorption`'s thesis is that
heavy volume touching a level and closing back on the DEFENDED side marks a
resting order absorbing aggressive flow (a fade/bounce AT the level, see
`l2_absorption.py`'s module docstring). This module tests the OPPOSITE
hypothesis on the SAME level definition and SAME volume-spike confirmation:
when heavy volume touches a level and the level's defense FAILS — price
closes THROUGH it rather than holding — trade in the direction of the
break (momentum continuation) instead of betting on a bounce back.

This is a distinct entry condition implemented as its OWN module/function,
NOT a flag added to `l2_absorption.py` (which stays untouched, still
RETIRED — this repo's convention is to never resurrect a retired signal
file in place; see that module's RETIRED banner). Concretely: fires when a
bar (a) trades >= `volume_mult` x the trailing `volume_lookback`-bar
average volume, AND (b) CLOSES beyond (at least `breakout_atr_mult` x ATR
clear of) the rolling `level_lookback`-bar extreme immediately preceding
it — `now_close < level_low - breakout_atr_mult*ATR` fires a SHORT
(bearish breakdown, in the direction of the break) and `now_close >
level_high + breakout_atr_mult*ATR` fires a LONG (bullish breakout).
`breakout_atr_mult` defaults to 0.0, the literal "closed beyond the level,
no extra clearance required" reading of the hypothesis stated in the
investigation brief; a positive value is a minimum-breakout-magnitude noise
filter (a candidate rescue lever, not part of the baseline entry rule —
see `backtests/reports/absorption_breakout_investigation_report.md`).

Genuinely new vs. the two EXISTING breakout/continuation-family signals in
this repo: `orb_vwap.py` trades the opening-range high/low (a single fixed
intraday reference, only relevant around the session open) and
`vp_breakout.py` trades a volume-profile value-area edge (a volume-WEIGHTED
reference, evaluated at any point in the session). This module's
`level_low`/`level_high` (a rolling `level_lookback`-bar OHLC extreme,
computation identical to `l2_absorption.py`'s) is neither of those — this
combination (this specific level definition + continuation polarity) had
never been backtested in this repo before this investigation.

Stop / target: because this is a CONTINUATION trade, the stop belongs back
INSIDE the level that just broke (a failed breakout re-entering the prior
range invalidates the thesis) — `stop_atr_mult` x ATR inside `level_high`
for a long breakout, inside `level_low` for a short breakdown. This mirrors
`vp_breakout.py`'s own stop placement (`vah - stop_atr_mult*ATR` /
`val + stop_atr_mult*ATR`), not `l2_absorption.py`'s (which places the stop
just PAST the level, on the far side, since that thesis is a fade). Optional
`target_r_multiple` (default `None` = no target, winner cut only by the
engine's time-stop/EOD flatten) is the same lever/convention every other
signal in this package uses under the identical name.

`micro_stop_cents` (added 2026-08-14, round 2 of this investigation — see
`backtests/reports/absorption_breakout_investigation_report.md`'s dated
addendum) is an ALTERNATIVE stop-distance rule, mutually exclusive with
`stop_atr_mult`: when set (not `None`), the stop is placed exactly
`micro_stop_cents` dollars past the broken level (`level_high -
micro_stop_cents` for a long, `level_low + micro_stop_cents` for a short)
instead of `stop_atr_mult * ATR` past it. Rationale: if the level's failure
is the ENTIRE thesis, a failed breakout re-entering the prior range
invalidates the trade instantly, at a fixed, small distance — not at a
volatility-scaled (and on a high-ATR name/day, potentially much wider)
distance. Default `None` preserves the original ATR-based behavior
byte-for-byte (backward compatible, same convention as every optional
lever in this package). **1-minute-bar modeling caveat, stated explicitly
because this lever is unusually sensitive to it**: this backtest engine can
only detect a stop-out when some 1-minute bar's `low`/`high` crosses the
stop price — it has no sub-minute intrabar path, so a stop 1-2 cents past
the level is resolved by the SAME coarse "did this whole minute's range
touch the stop" logic as any wider ATR-based stop, not a faithful
tick-by-tick simulation of what a real resting stop order would have
actually done inside that minute (which could easily have touched and
recovered multiple times, or never touched despite the minute's range
crossing it on a print the stop wouldn't have queued ahead of). Treat any
`micro_stop_cents` result as a coarser approximation than this signal's
other (ATR-based) results, not a more precise one, despite the tighter
dollar distance suggesting otherwise.

No lookahead: identical contract to every module in this package (see
`python/microstructure/signals/__init__.py`) — only ever looks at `bars`
up to and including "now" (`bars.index[-1]`); no argument or computation
here ever reads beyond that index.

Free parameters (4 in the ORIGINAL 2026-08-14 version — `volume_mult`,
`breakout_atr_mult`, `stop_atr_mult`, `target_r_multiple` — at parity with
`l2_absorption.py`, 1 under `python/backtest/param_guard.py`'s
`MAX_FREE_PARAMETERS=5` ceiling; `micro_stop_cents` added the same day in
round 2 is a 5th, putting this signal AT the ceiling with no further
headroom — see the round-2 addendum in
`backtests/reports/absorption_breakout_investigation_report.md` for why it
was still added rather than swapped in place of `stop_atr_mult`: the two
are tested against each other, not stacked, so only one is ever active per
run, but Chan's discipline counts declared tunable knobs, not
simultaneously-active ones). `level_lookback`/`volume_lookback`/
`atr_period` are fixed structural constants (identical VALUES to
`l2_absorption.py`'s, for direct comparability across the two signals'
shared level/volume definition), not exposed as tunable grid axes — the
same convention `vp_breakout.py`'s
`_VOLUME_LOOKBACK`/`_MIN_BARS_FOR_PROFILE`/`_N_BINS` already use in this
package.

See `backtests/reports/absorption_breakout_investigation_report.md` for
the full validation (diagnostic + this signal's own WFO/Monte
Carlo/cost-adjusted-PF/2x-slippage-stress gate run, honest holdout, and
the round-2 addendum covering `micro_stop_cents`) and
`backtests/reports/signal_status.md` for the current status.
"""
from __future__ import annotations

import pandas as pd

from .. import context as ctx
from . import MicroSignal


def evaluate_absorption_breakout(
    bars: pd.DataFrame,
    symbol: str = "",
    volume_mult: float = 3.0,
    breakout_atr_mult: float = 0.0,
    stop_atr_mult: float = 0.5,
    level_lookback: int = 20,
    volume_lookback: int = 20,
    atr_period: int = 14,
    target_r_multiple: float | None = None,
    micro_stop_cents: float | None = None,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff "now"'s bar (a) traded on
    >= `volume_mult` x the trailing `volume_lookback`-bar average volume,
    AND (b) CLOSED beyond (at least `breakout_atr_mult` x ATR clear of) the
    rolling `level_lookback`-bar extreme immediately preceding it — i.e.
    the level FAILS rather than holds. Trades in the direction of the
    break (continuation) — the opposite polarity of
    `l2_absorption.evaluate_l2_absorption`'s fade/bounce entry, on the
    IDENTICAL trigger's level/volume definitions (same `bars` input
    contract, same `level_lookback`/`volume_lookback`/`atr_period`
    defaults). Only ever looks at `bars` up to and including "now" — no
    lookahead, same contract as every other signal in this package.

    `micro_stop_cents` (default `None`): when set, OVERRIDES the
    `stop_atr_mult`-based stop with a fixed `micro_stop_cents`-dollar
    distance past the broken level instead — see module docstring for the
    full rationale and the 1-minute-bar approximation caveat this
    parameter carries."""
    min_bars = max(level_lookback, volume_lookback, atr_period) + 2
    if len(bars) < min_bars:
        return None

    now = bars.iloc[-1]
    now_time = bars.index[-1]

    baseline_window = bars["volume"].iloc[-(volume_lookback + 1):-1]
    baseline_vol = float(baseline_window.mean())
    if baseline_vol <= 0 or float(now["volume"]) < volume_mult * baseline_vol:
        return None

    atr_series = ctx.atr(bars, period=atr_period)
    now_atr = float(atr_series.iloc[-1])
    if pd.isna(now_atr) or now_atr <= 0:
        return None

    prior = bars.iloc[-(level_lookback + 1):-1]
    level_low = float(prior["low"].min())
    level_high = float(prior["high"].max())

    now_close = float(now["close"])
    clearance = breakout_atr_mult * now_atr

    direction: str | None = None
    if now_close > level_high + clearance:
        direction = "long"
        stop = (level_high - micro_stop_cents if micro_stop_cents is not None
                else level_high - stop_atr_mult * now_atr)
    elif now_close < level_low - clearance:
        direction = "short"
        stop = (level_low + micro_stop_cents if micro_stop_cents is not None
                else level_low + stop_atr_mult * now_atr)
    else:
        return None

    target = None
    if target_r_multiple is not None:
        risk = abs(now_close - stop)
        if risk > 0:
            target = (now_close + target_r_multiple * risk if direction == "long"
                      else now_close - target_r_multiple * risk)

    return MicroSignal(
        symbol=symbol, strategy="absorption_breakout", direction=direction,
        signal_time=now_time, entry_price=now_close, stop_price=stop, target_price=target,
        order_type="next_open",
        context={
            "level_low": level_low, "level_high": level_high,
            "bar_volume": float(now["volume"]), "baseline_volume": baseline_vol,
            "tier": "bar_only_proxy_no_l2_confirmation",
            "target_r_multiple": target_r_multiple,
            "micro_stop_cents": micro_stop_cents,
        },
    )
