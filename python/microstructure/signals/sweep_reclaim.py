"""
S1 — Liquidity Sweep & Reclaim.

Thesis (docs/microstructure_pivot_plan.md §1, S1): price pierces a known
liquidity pool (YDH/YDL, PMH/PML, EQH/EQL, round number) by at least
`sweep_min_atr` x ATR, then reclaims back inside the level within
`reclaim_bars` bars — read as the sweep having run stops/liquidity rather
than starting a genuine breakout, so the reversal is traded.

Free parameters (3, Chan discipline): sweep_min_atr, reclaim_bars,
stop_atr_mult.
"""
from __future__ import annotations

import pandas as pd

from ..context import LiquidityLevels
from . import MicroSignal


def _resistance_levels(levels: LiquidityLevels, current_price: float) -> list[float]:
    out = [lvl for lvl in (levels.ydh, levels.pmh) if lvl is not None]
    out.extend(levels.eq_highs)
    return out


def _support_levels(levels: LiquidityLevels, current_price: float) -> list[float]:
    out = [lvl for lvl in (levels.ydl, levels.pml) if lvl is not None]
    out.extend(levels.eq_lows)
    return out


def _round_resistance_levels(levels: LiquidityLevels, current_price: float) -> list[float]:
    return [r for r in levels.round_levels if r > current_price]


def _round_support_levels(levels: LiquidityLevels, current_price: float) -> list[float]:
    return [r for r in levels.round_levels if r < current_price]


def _nearest_opposite_level(levels: list[float], price: float, direction: str) -> float | None:
    """Nearest level on the side a `direction` trade is heading toward
    (used as a simple profit target — the opposite liquidity pool)."""
    if not levels:
        return None
    if direction == "long":
        candidates = [lvl for lvl in levels if lvl > price]
        return min(candidates) if candidates else None
    candidates = [lvl for lvl in levels if lvl < price]
    return max(candidates) if candidates else None


def evaluate_sweep_reclaim(
    bars: pd.DataFrame,
    levels: LiquidityLevels,
    atr_series: pd.Series,
    symbol: str = "",
    sweep_min_atr: float = 0.15,
    reclaim_bars: int = 3,
    stop_atr_mult: float = 0.25,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff a sweep of some liquidity
    level occurred within the last `reclaim_bars` bars (including "now"
    itself, for a same-bar sweep+reclaim) and "now" has closed back inside
    that level. Only ever looks at `bars` as given — no access to anything
    beyond bars.index[-1], so this is safe to call every bar in an event
    loop without leaking future information."""
    if len(bars) < 2 or reclaim_bars < 1:
        return None
    if atr_series.empty or pd.isna(atr_series.iloc[-1]) or atr_series.iloc[-1] <= 0:
        return None

    current_atr = float(atr_series.iloc[-1])
    now_time = bars.index[-1]
    now_close = float(bars["close"].iloc[-1])
    window = bars.iloc[-min(reclaim_bars, len(bars)):]
    # Round-number levels are recomputed EVERY call from the current close
    # (context._round_levels_near), so they are not a stable multi-bar
    # reference the way YDH/YDL/PMH/PML/EQH/EQL are — an older bar in
    # `window` may have printed a price that only LOOKS like it "swept" a
    # round level defined using a much later price context. Round levels
    # are therefore only evaluated same-bar (pierce and reclaim within
    # `now`'s own OHLC), never against the multi-bar trailing window.
    now_bar_only = bars.iloc[-1:]

    stable_resistance = [(level, window) for level in _resistance_levels(levels, now_close)]
    round_resistance = [(level, now_bar_only) for level in _round_resistance_levels(levels, now_close)]
    for level, scan_window in stable_resistance + round_resistance:
        swept = scan_window[scan_window["high"] >= level + sweep_min_atr * current_atr]
        if swept.empty or now_close > level:
            continue
        sweep_extreme = float(swept["high"].max())
        stop = sweep_extreme + stop_atr_mult * current_atr
        target = _nearest_opposite_level(_support_levels(levels, now_close), now_close, "short")
        return MicroSignal(
            symbol=symbol, strategy="sweep_reclaim", direction="short",
            signal_time=now_time, entry_price=now_close, stop_price=stop,
            target_price=target, order_type="next_open",
            context={"swept_level": level, "sweep_extreme": sweep_extreme, "level_kind": "resistance"},
        )

    stable_support = [(level, window) for level in _support_levels(levels, now_close)]
    round_support = [(level, now_bar_only) for level in _round_support_levels(levels, now_close)]
    for level, scan_window in stable_support + round_support:
        swept = scan_window[scan_window["low"] <= level - sweep_min_atr * current_atr]
        if swept.empty or now_close < level:
            continue
        sweep_extreme = float(swept["low"].min())
        stop = sweep_extreme - stop_atr_mult * current_atr
        target = _nearest_opposite_level(_resistance_levels(levels, now_close), now_close, "long")
        return MicroSignal(
            symbol=symbol, strategy="sweep_reclaim", direction="long",
            signal_time=now_time, entry_price=now_close, stop_price=stop,
            target_price=target, order_type="next_open",
            context={"swept_level": level, "sweep_extreme": sweep_extreme, "level_kind": "support"},
        )

    return None
