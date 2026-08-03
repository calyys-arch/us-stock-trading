"""
S4 — L2 Absorption / Iceberg (docs/microstructure_pivot_plan.md §1, S4).

Thesis: a recent support/resistance level ("level_low"/"level_high" — the
rolling extreme over `level_lookback` prior bars) gets TOUCHED on an
unusually large volume bar, but price fails to meaningfully violate it and
closes back on the defended side — the observable footprint of a resting
institutional order absorbing aggressive flow at that price without being
displaced. Direction is a bounce/fade AT the level, not a breakout — this
is what distinguishes it from S1 (sweep_reclaim), which specifically
requires price to clear the level by `sweep_min_atr` first: absorption
never really lets the level get violated at all.

TIER / HONESTY NOTE (2026-07-29): this is a BAR-ONLY proxy. The full plan
(§3b/§4b) calls for confirming absorption with actual L2 depth (the level
being refreshed rather than depleted) and a LOW `order_book_churn_score`
(ruling out spoofing/layering as the reason price didn't move) — that
needs `python/backtest/depth_replay.py` (Phase 3, not yet built) plus
weeks of `scripts/capture_market_microstructure.py` depth archive, neither
of which exist yet (see that script's docstring: "ticks not captured are
gone forever"). Until then, every MicroSignal this module emits carries
`context["tier"] = "bar_only_proxy_no_l2_confirmation"` so nothing
downstream can mistake this for the L2-confirmed version. Per the plan,
this signal is deliberately NOT added to scripts/run_intraday_backtest.py's
SIGNALS list — it is wired only into intraday_engine.py's signal-scan path
(python/backtest/intraday_engine.py:scan_signals_for_session) for
observe-only detection/logging, never into the WFO/promotion pipeline.

Free parameters (3, Chan discipline): volume_mult, touch_atr_mult, stop_atr_mult.
"""
from __future__ import annotations

import pandas as pd

from .. import context as ctx
from . import MicroSignal


def evaluate_l2_absorption(
    bars: pd.DataFrame,
    symbol: str = "",
    volume_mult: float = 3.0,
    touch_atr_mult: float = 0.25,
    stop_atr_mult: float = 0.5,
    level_lookback: int = 20,
    volume_lookback: int = 20,
    atr_period: int = 14,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff "now"'s bar (a) traded on
    >= `volume_mult` x the trailing `volume_lookback`-bar average volume,
    AND (b) touched (within `touch_atr_mult` x ATR) the rolling
    `level_lookback`-bar extreme immediately preceding it WITHOUT closing
    through that level. Only ever looks at `bars` up to and including
    "now" — no lookahead, same contract as every other signal in this
    package."""
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

    now_low, now_high, now_close = float(now["low"]), float(now["high"]), float(now["close"])
    tolerance = touch_atr_mult * now_atr

    direction: str | None = None
    if abs(now_low - level_low) <= tolerance and now_close > level_low:
        direction = "long"
        stop = level_low - stop_atr_mult * now_atr
    elif abs(now_high - level_high) <= tolerance and now_close < level_high:
        direction = "short"
        stop = level_high + stop_atr_mult * now_atr
    else:
        return None

    return MicroSignal(
        symbol=symbol, strategy="l2_absorption", direction=direction,
        signal_time=now_time, entry_price=now_close, stop_price=stop, target_price=None,
        order_type="next_open",
        context={
            "level_low": level_low, "level_high": level_high,
            "bar_volume": float(now["volume"]), "baseline_volume": baseline_vol,
            "tier": "bar_only_proxy_no_l2_confirmation",
        },
    )
