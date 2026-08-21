"""
RETIRED (2026-08-13) — verdict NO-GO, no further work planned.
cost_adjusted_profit_factor 0.891 (calibrated, full window) vs the 1.3 gate
required by configs/goal.yaml; also fails wfo_go (pass ratio 50%->38% under
calibrated costs) and monte_carlo_p5_sharpe (-3.916), and mean OOS Sharpe
sign-flips to -0.580 under calibrated costs. Root cause: the underlying
hypothesis under test here — that gating on a trailing 20-day/2% regime
label reduces intraday whipsaw losses — is FALSIFIED, not just cost-crushed:
the regime filter made both pass ratio and mean OOS Sharpe WORSE than plain
orb_vwap, before calibration was even applied. It also inherits
orb_vwap.py's since-fixed gap-trap stop-inversion bug (the numbers below
were never re-measured post-fix). Full evidence:
backtests/reports/strategy_review_summary.md §3.4 and
backtests/reports/slippage_calibration_report.md. Code and tests are kept
and still correct; this signal is excluded from the default run of
scripts/run_intraday_backtest.py (see its RETIRED_SIGNALS) but remains
importable and explicitly runnable/testable — the logic below is
unchanged by this retirement.

New signal hypothesis — Regime-filtered ORB+VWAP continuation.

Economic rationale (see backtests/reports/intraday_backtest_report.md,
orb_vwap section): orb_vwap's raw OOS Sharpe was actually positive
(+1.41 mean, WFO pass ratio 62%) — the only one of the three original
signals with a genuinely positive pattern — but it still failed
cost-adjusted profit factor / Monte Carlo p5 / the 2x-slippage stress
test. Plausible explanation: a pure opening-range breakout-continuation
trade gets chopped up by whipsaw on SIDEWAYS/mean-reverting days, and
those chop losses eat the edge that shows up on genuinely trending days.

Hypothesis under test: gating orb_vwap.evaluate_orb_vwap's UNCHANGED
signal logic to only fire on days classified as trending (Bull or Bear,
i.e. NOT Sideways) by python/analytics/regime.py's Markov regime
classifier will materially improve cost-adjusted profit factor and
reduce whipsaw losses.

Implementation discipline:
  - orb_vwap's actual entry/stop/gap-trap logic is imported and called
    UNCHANGED (`evaluate_orb_vwap`) — this module adds exactly ONE new
    gate on top, it does not reimplement or tweak the underlying signal.
  - The regime gate is a HARD STRUCTURAL RULE, not a tunable parameter
    ("any non-sideways regime may trade, sideways may not") — same
    discipline the plan doc (docs/microstructure_pivot_plan.md §1, S3)
    already uses for the "no entries in the first 15 minutes" rule.
    regime.py's own `window`/`threshold` stay at their module defaults
    (20 trading days / 2%) rather than being exposed as a knob here, so
    this signal adds ZERO new free parameters versus plain orb_vwap
    (still 2: or_minutes, vwap_side_filter).
  - No lookahead: "is today trending" is decided from the regime LABEL
    AS OF THE PRIOR trading day's close (python/backtest/intraday_engine.py
    computes this once per symbol from daily closes, shifted by one day
    before being handed to this module) — never from today's own price
    action, which would not be known before today's session starts.

Free parameters (2, same as orb_vwap): or_minutes, vwap_side_filter.
"""
from __future__ import annotations

import pandas as pd

from ..context import OpeningRange
from . import MicroSignal
from .orb_vwap import evaluate_orb_vwap


def evaluate_orb_vwap_regime(
    bars: pd.DataFrame,
    opening_range: OpeningRange,
    vwap_series: pd.Series,
    is_trending_day: bool,
    symbol: str = "",
    or_minutes: int = 15,
    vwap_side_filter: bool = True,
    prior_close: float | None = None,
) -> MicroSignal | None:
    """Identical to `orb_vwap.evaluate_orb_vwap` except it returns None
    unconditionally when `is_trending_day` is False — the ONLY new
    behavior this module adds. `is_trending_day` must be computed by the
    caller from data available BEFORE today's session (see module
    docstring) — this function does not compute or validate that itself,
    to keep the no-lookahead contract visibly enforced at the call site
    (python/backtest/intraday_engine.py), the same place every other
    signal's causality is enforced."""
    if not is_trending_day:
        return None

    sig = evaluate_orb_vwap(
        bars, opening_range, vwap_series, symbol=symbol,
        or_minutes=or_minutes, vwap_side_filter=vwap_side_filter, prior_close=prior_close,
    )
    if sig is None:
        return None

    sig.strategy = "orb_vwap_regime"
    sig.context["regime_gated"] = True
    return sig
