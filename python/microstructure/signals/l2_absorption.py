"""
RETIRED (2026-08-14) — verdict NO-GO, no further work planned. Official
pipeline run (full 20-symbol universe, full `configs/param_grids.yaml`
grid, calibrated per-symbol costs): wfo_go 0% pass ratio (0/7 folds), mean
OOS Sharpe -14.975, cost_adjusted_profit_factor 0.381 vs the 1.3 gate,
GROSS (pre-cost) profit factor 0.392 — this signal has NO detectable gross
edge in any configuration tested, unlike orb_vwap (which had gross edge
killed by costs). Four rescue levers (tight-spread universe, target_r_
multiple 1-3R, trend-efficiency mean-reversion regime gate) were tested;
the best (tight-spread top-6, unmodified baseline params) still only
reaches cost-adjusted PF 0.636 in-sample / 0.960 on the untouched final
holdout, both NO-GO. Full evidence:
backtests/reports/l2_absorption_validation_report.md (authoritative) and
backtests/reports/signal_status.md. Code, `configs/param_grids.yaml`'s
grid, and tests are kept and still correct; this signal is excluded from
the default run of scripts/run_intraday_backtest.py (see that script's
RETIRED_SIGNALS) but remains importable and explicitly runnable/testable
(`--signal l2_absorption` always works) — the logic below is unchanged by
this retirement. With this signal settled, EVERY microstructure signal in
this repo (sweep_reclaim, fvg_retest, orb_vwap, orb_vwap_regime,
vwap_band_fade, vp_breakout, l2_absorption) is now confirmed NO-GO.

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
weeks of `scripts/capture_market_microstructure.py` depth archive. As of
2026-08-14, `data/ticks/`/`data/depth/` only cover 2026-08-04..2026-08-14 —
entirely AFTER the cached 1-minute OHLCV history this signal is validated
against ends (2026-07-31) — so an L2-confirmed version genuinely cannot be
backtested yet; every MicroSignal this module emits still carries
`context["tier"] = "bar_only_proxy_no_l2_confirmation"` so nothing
downstream can mistake this for that (not yet buildable) L2-confirmed
version.

VALIDATION STATUS (2026-08-14): this signal HAS now been run end-to-end
through `python/backtest/intraday_engine.py`'s fill/P&L simulation and
`scripts/run_intraday_backtest.py`'s WFO/Monte Carlo/cost-adjusted-PF gate
(`--signal l2_absorption`) — see the RETIRED banner at the top of this
docstring for the verdict, and
`backtests/reports/l2_absorption_validation_report.md` for the full
methodology. It remains wired into the dashboard's observe-only
chart-overlay path AND excluded from `dashboard/live_microstructure_scheduler.py`'s
`LIVE_SIGNALS` — both independent of the WFO verdict above, since the
bar-only-proxy caveat is a permanent limitation regardless of whether any
future re-run of this signal's WFO gate ever passes.

Free parameters (4, Chan discipline): volume_mult, touch_atr_mult,
stop_atr_mult, target_r_multiple. `target_r_multiple` (added 2026-08-14,
same lever as `orb_vwap`'s rescue investigation —
`backtests/reports/orb_vwap_rescue_report.md`) defaults to `None`, i.e. the
original "no target at all, winner is cut only by the engine's time-stop/
EOD flatten" behavior — strictly backward-compatible for every existing
caller that does not pass it.
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
    target_r_multiple: float | None = None,
) -> MicroSignal | None:
    """Fires AT `bars.index[-1]` ("now") iff "now"'s bar (a) traded on
    >= `volume_mult` x the trailing `volume_lookback`-bar average volume,
    AND (b) touched (within `touch_atr_mult` x ATR) the rolling
    `level_lookback`-bar extreme immediately preceding it WITHOUT closing
    through that level. Only ever looks at `bars` up to and including
    "now" — no lookahead, same contract as every other signal in this
    package.

    `target_r_multiple` (default None = no target, winner cut only by the
    engine's time-stop/EOD flatten — the original behavior) sets a profit
    target that many multiples of the stop distance away from `now_close`,
    identical convention to `orb_vwap.evaluate_orb_vwap`'s own lever of the
    same name (`backtests/reports/orb_vwap_rescue_report.md`)."""
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

    target = None
    if target_r_multiple is not None:
        risk = abs(now_close - stop)
        if risk > 0:
            target = (now_close + target_r_multiple * risk if direction == "long"
                      else now_close - target_r_multiple * risk)

    return MicroSignal(
        symbol=symbol, strategy="l2_absorption", direction=direction,
        signal_time=now_time, entry_price=now_close, stop_price=stop, target_price=target,
        order_type="next_open",
        context={
            "level_low": level_low, "level_high": level_high,
            "bar_volume": float(now["volume"]), "baseline_volume": baseline_vol,
            "tier": "bar_only_proxy_no_l2_confirmation",
            "target_r_multiple": target_r_multiple,
        },
    )
