"""
RETIRED (2026-08-13) — verdict NO-GO, no further work planned.
wfo_go: 0% pass ratio (both cost models). cost_adjusted_profit_factor
0.456 (calibrated, full window) vs the 1.3 gate required by
configs/goal.yaml — tied for the worst full-window PF of the six signals
alongside fvg_retest, despite the lowest trade count (1,816). Also fails
monte_carlo_p5_sharpe (-11.563) and the mandatory 2x-slippage stress test
(-$952K net). Root cause: a value-area break on 3x volume is at least as
consistent with a volume climax/exhaustion print as with a genuine
continuation breakout, and the signal has no way to distinguish the two
from bars alone — extra entry selectivity did not buy quality. Full
evidence: backtests/reports/strategy_review_summary.md §3.6 and
backtests/reports/slippage_calibration_report.md /
backtests/reports/new_signals_report.md. Code and tests are kept and
still correct; this signal is excluded from the default run of
scripts/run_intraday_backtest.py (see its RETIRED_SIGNALS) but remains
importable and explicitly runnable/testable — the logic below is
unchanged by this retirement.

New signal hypothesis — Volume Profile value-area breakout.

Economic rationale: a third, structurally distinct hypothesis from every
other signal in this package — not a reversal-to-VWAP trade (unlike
vwap_band_fade.py) and not an opening-range-specific setup (unlike
orb_vwap*.py); this can trigger at ANY point in the session. Thesis:
price breaking decisively out of the CURRENT session's Volume Profile
value area (above VAH or below VAL — python/microstructure/context.py's
`volume_profile`) on above-average volume signals a shift in where the
market is willing to transact, favoring continuation AWAY from the value
area rather than a snap-back.

Value-area reference, and why it is computed EXCLUDING the breakout
window: `context.volume_profile` is self-referential — call it on bars
that already include the breakout print, and that print's own (large)
volume can pull VAH/VAL to already encompass it, since a genuine
Volume-Weighted value area updates with every bar's volume by
definition. To keep "price breaks OUT of an ESTABLISHED value area" a
coherent, non-circular idea, the value area used for the breakout/hold
check is always computed on bars STRICTLY BEFORE the `confirm_bars`
window being evaluated — i.e. the value area as it stood right before
the breakout began, not a value area that already absorbed the breakout
bar's own volume.

Confirmation / entry: the breakout must be on above-average volume (the
FIRST bar after the established value area, i.e. `window.iloc[0]` in the
confirm window, must trade >= `vol_mult` x the trailing rolling average
volume) AND price must hold outside the value area for the full
`confirm_bars` window (a "fresh cross" check against the bar immediately
before the window, mirroring orb_vwap.py's single-bar-crossing pattern,
generalized to a multi-bar hold).

Stop / target: stop is placed `stop_atr_mult` x ATR back INSIDE the
value area (beyond VAH for a long breakout, beyond VAL for a short one)
— a failed breakout that reverts back into the value area invalidates
the thesis. Target is the NEXT liquidity pool level in the breakout
direction (YDH/YDL/PMH/PML/round number) via context.py's
`target_resistance_levels`/`target_support_levels`/
`nearest_liquidity_target` — the SAME fixed selection logic sweep_reclaim.py
uses post-bugfix (see context.py's docstring for that incident), reused
here rather than re-implemented so this signal cannot reintroduce the
eq_highs/eq_lows-as-target bug that logic was fixed to avoid (this
signal never even has access to eq_highs/eq_lows as targets, by
construction, since it calls the same narrow `target_*` functions).

No lookahead: every quantity here is computed from `bars_today` up to
and including `bars_today.index[-1]` ("now") — the value-area reference
window is a strict PREFIX of that (see above), never anything beyond
"now".

Free parameters (3, Chan discipline): vol_mult, confirm_bars,
stop_atr_mult. `volume_lookback` (rolling average window) and
`min_bars_for_profile`/`n_bins` (value-area computation knobs) are fixed
structural constants, matching fvg_retest.py's `volume_lookback`
convention — not exposed as tunable grid axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import context as ctx
from ..context import LiquidityLevels
from . import MicroSignal

_VOLUME_LOOKBACK = 20
_MIN_BARS_FOR_PROFILE = 30
_N_BINS = 30


def evaluate_vp_breakout(
    bars_today: pd.DataFrame,
    levels: LiquidityLevels,
    atr_series: pd.Series,
    symbol: str = "",
    vol_mult: float = 2.0,
    confirm_bars: int = 2,
    stop_atr_mult: float = 0.5,
) -> MicroSignal | None:
    """Fires AT `bars_today.index[-1]` ("now") — see module docstring for
    the full detection contract. `bars_today` must be sliced to ONE
    session up to and including "now" (the same convention as
    `context.opening_range`/`context.compute_liquidity_levels`).

    Performance note: this signal is re-evaluated on a GROWING bars-so-far
    prefix every unheld bar of every session (intraday_engine.py's event
    loop), so it works on numpy arrays extracted once up front rather than
    repeated pandas `.iloc[...]` slicing + Series arithmetic — profiled at
    ~5x faster per call over a full session at negligible risk, since the
    arithmetic itself (see `ctx.volume_profile_from_arrays`) is unchanged."""
    n = len(bars_today)
    if confirm_bars < 1 or n < _MIN_BARS_FOR_PROFILE + confirm_bars:
        return None
    if atr_series.empty or pd.isna(atr_series.iloc[-1]) or atr_series.iloc[-1] <= 0:
        return None
    atr_now = float(atr_series.iloc[-1])

    now_time = bars_today.index[-1]
    close_np = bars_today["close"].to_numpy(dtype=float)
    high_np = bars_today["high"].to_numpy(dtype=float)
    low_np = bars_today["low"].to_numpy(dtype=float)
    volume_np = bars_today["volume"].to_numpy(dtype=float)
    now_close = float(close_np[-1])

    # Value area as it stood BEFORE the confirm window began — see module
    # docstring on why the breakout window itself must be excluded.
    vp_end = n - confirm_bars
    if vp_end < _MIN_BARS_FOR_PROFILE:
        return None
    tp_vp = (high_np[:vp_end] + low_np[:vp_end] + close_np[:vp_end]) / 3.0
    vp = ctx.volume_profile_from_arrays(tp_vp, volume_np[:vp_end], n_bins=_N_BINS)
    if vp.vah is None or vp.val is None:
        return None

    baseline_window = volume_np[max(0, vp_end - _VOLUME_LOOKBACK):vp_end]
    if baseline_window.size == 0:
        return None
    baseline_vol = float(baseline_window.mean())
    if baseline_vol <= 0:
        return None

    window_close = close_np[-confirm_bars:]
    breakout_volume = float(volume_np[vp_end])
    before_close = float(close_np[vp_end - 1])

    if bool(np.all(window_close > vp.vah)) and before_close <= vp.vah:
        if breakout_volume >= vol_mult * baseline_vol:
            stop = vp.vah - stop_atr_mult * atr_now
            target = ctx.nearest_liquidity_target(
                ctx.target_resistance_levels(levels, now_close), now_close, "long",
            )
            return MicroSignal(
                symbol=symbol, strategy="vp_breakout", direction="long",
                signal_time=now_time, entry_price=now_close, stop_price=stop, target_price=target,
                order_type="next_open",
                context={"vah": vp.vah, "val": vp.val, "breakout_volume": breakout_volume,
                         "baseline_volume": baseline_vol},
            )

    if bool(np.all(window_close < vp.val)) and before_close >= vp.val:
        if breakout_volume >= vol_mult * baseline_vol:
            stop = vp.val + stop_atr_mult * atr_now
            target = ctx.nearest_liquidity_target(
                ctx.target_support_levels(levels, now_close), now_close, "short",
            )
            return MicroSignal(
                symbol=symbol, strategy="vp_breakout", direction="short",
                signal_time=now_time, entry_price=now_close, stop_price=stop, target_price=target,
                order_type="next_open",
                context={"vah": vp.vah, "val": vp.val, "breakout_volume": breakout_volume,
                         "baseline_volume": baseline_vol},
            )

    return None
