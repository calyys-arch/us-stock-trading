"""
RETIRED (2026-08-13) — verdict NO-GO, no further work planned.
wfo_go: 0% pass ratio across every fold (8 of 8), under BOTH the flat and
calibrated cost models — not a marginal cost problem. cost_adjusted_profit_
factor 0.575 (calibrated) vs the 1.3 gate required by configs/goal.yaml;
also fails monte_carlo_p5_sharpe (-13.113) and the mandatory 2x-slippage
stress test (-$1.90M net). Root cause: this universe is dominated by
momentum-driven mega-cap/semis names where a 2+ sigma VWAP-band extension
is empirically more often the START of a bigger move than exhaustion —
i.e. this signal fades real momentum, not noise. Full evidence:
backtests/reports/strategy_review_summary.md §3.5 and
backtests/reports/slippage_calibration_report.md /
backtests/reports/new_signals_report.md. Code and tests are kept and
still correct; this signal is excluded from the default run of
scripts/run_intraday_backtest.py (see its RETIRED_SIGNALS) but remains
importable and explicitly runnable/testable — the logic below is
unchanged by this retirement.

New signal hypothesis — VWAP band mean-reversion fade.

Economic rationale: a deliberately DIFFERENT (mean-reversion, not
breakout-continuation) hypothesis from every other signal in this
package. python/microstructure/context.py's `vwap_bands` already computes
+/-1sigma / +/-2sigma bands around the session VWAP. When price extends
beyond a band multiple and then shows early signs of failing to extend
FURTHER (a run of `stall_bars` consecutive bars making no new extreme
beyond the extension), that reads as exhaustion of the move rather than
genuine continuation — so the trade fades back toward VWAP itself, which
is the definitionally natural "fair value" target for this thesis (not a
free parameter: there is no other level a "fade to VWAP" trade could
sensibly target).

Detection ("stall" definition, exact and testable): let
`window = bars.iloc[-(stall_bars+1):]` — the FIRST bar of this window is
the candidate extension extreme; the signal fires iff that first bar's
high (low) is simultaneously (a) beyond the current upper (lower) band
by `band_sigma_mult` and (b) still the MAXIMUM (MINIMUM) high (low) of
the whole window, i.e. none of the following `stall_bars` bars made a
NEW extreme past it. This is a single, mechanical, causal check over
already-closed bars — no swing-detection heuristics, no re-fitting.

No lookahead: only ever reads `bars`/`vwap_bands_df`/`atr_series` up to
and including "now" (`bars.index[-1]`); `vwap_bands_df` is expected to be
a session-VWAP-bands frame computed on a prefix ending at or before "now"
(the same "cumulative stat computed once, sliced per bar" pattern
python/backtest/intraday_engine.py already uses for orb_vwap's VWAP
series — recomputing on a growing prefix gives IDENTICAL values at each
position because vwap_bands is a purely trailing/cumulative statistic).

Free parameters (3, Chan discipline): band_sigma_mult, stall_bars,
stop_atr_mult. Target (session VWAP) is intentionally NOT a free
parameter — see rationale above.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import MicroSignal


def evaluate_vwap_band_fade(
    bars: pd.DataFrame,
    vwap_bands_df: pd.DataFrame,
    atr_series: pd.Series,
    symbol: str = "",
    band_sigma_mult: float = 2.0,
    stall_bars: int = 3,
    stop_atr_mult: float = 0.5,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") — see module docstring for the
    exact stall-detection window definition. `vwap_bands_df` must be
    index-aligned with `bars` (same convention as orb_vwap's
    `vwap_series` argument) and carry at least the `vwap`/`upper_1`
    columns (used to derive sigma = upper_1 - vwap at "now").

    Performance note: like vp_breakout.py, this is re-evaluated on a
    growing bars-so-far prefix every unheld bar of every session, so the
    hot path reads raw numpy scalars/arrays (`.to_numpy()`/`.values`) once
    up front instead of repeated `.iloc[...]` pandas indexing — same
    arithmetic, ~3-4x faster per call at this call volume."""
    n = len(bars)
    if stall_bars < 1 or n < stall_bars + 1:
        return None
    if vwap_bands_df.empty or len(vwap_bands_df) != n:
        return None
    atr_arr = atr_series.to_numpy(dtype=float)
    if atr_arr.size == 0 or math.isnan(atr_arr[-1]) or atr_arr[-1] <= 0:
        return None

    vwap_now = float(vwap_bands_df["vwap"].to_numpy(dtype=float)[-1])
    upper_1_now = float(vwap_bands_df["upper_1"].to_numpy(dtype=float)[-1])
    if math.isnan(vwap_now) or math.isnan(upper_1_now):
        return None
    sigma_now = upper_1_now - vwap_now
    if sigma_now <= 0:
        return None

    band_up = vwap_now + band_sigma_mult * sigma_now
    band_down = vwap_now - band_sigma_mult * sigma_now
    atr_now = float(atr_arr[-1])

    now_time = bars.index[-1]
    close_np = bars["close"].to_numpy(dtype=float)
    high_np = bars["high"].to_numpy(dtype=float)
    low_np = bars["low"].to_numpy(dtype=float)
    now_close = float(close_np[-1])

    window_start = n - (stall_bars + 1)
    window_high = high_np[window_start:]
    window_low = low_np[window_start:]

    extreme_high = float(window_high[0])
    if extreme_high >= band_up and extreme_high == float(window_high.max()) and now_close > vwap_now:
        stop = extreme_high + stop_atr_mult * atr_now
        return MicroSignal(
            symbol=symbol, strategy="vwap_band_fade", direction="short",
            signal_time=now_time, entry_price=now_close, stop_price=stop,
            target_price=vwap_now, order_type="next_open",
            context={"vwap": vwap_now, "band_level": band_up, "extension_extreme": extreme_high,
                     "sigma": sigma_now},
        )

    extreme_low = float(window_low[0])
    if extreme_low <= band_down and extreme_low == float(window_low.min()) and now_close < vwap_now:
        stop = extreme_low - stop_atr_mult * atr_now
        return MicroSignal(
            symbol=symbol, strategy="vwap_band_fade", direction="long",
            signal_time=now_time, entry_price=now_close, stop_price=stop,
            target_price=vwap_now, order_type="next_open",
            context={"vwap": vwap_now, "band_level": band_down, "extension_extreme": extreme_low,
                     "sigma": sigma_now},
        )

    return None
