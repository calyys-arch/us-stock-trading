"""
Current market-regime label, for tagging optimization runs.

Reuses python/analytics/regime.py's `label_regimes` rather than inventing a
second regime definition — one heuristic, one place, so a regime tag on a
promotion record means the same thing as the regime panel a human reads.

Scope, and why this does not spend Chan's parameter budget: the label here
is METADATA on a completed optimization run (recorded so a later run can ask
"what did this parameter set score the last time the market looked like
today"). It is NOT wired into any strategy's entry/exit condition, so
`window` and `threshold` remain diagnostic settings, not strategy free
parameters — the same contract regime.py's own docstring sets out. If a
regime label ever gates a live signal, its parameters count against
python/backtest/param_guard.py's budget and must earn a GO through WFO like
anything else.

No look-ahead: `label_regimes` reads a trailing rolling return only, and
`current_regime` takes the label of the LAST bar of the window it is given.
Callers tagging a backtest window pass that window's own prices, so the tag
describes the period that was tested, not anything after it.
"""
from __future__ import annotations

import logging

import pandas as pd

from python.analytics.regime import STATES, label_regimes

log = logging.getLogger(__name__)

UNKNOWN = "unknown"


def current_regime(
    close: pd.Series,
    *,
    window: int = 20,
    threshold: float = 0.02,
) -> str:
    """Regime label ("Bull" / "Bear" / "Sideways") of the last bar.

    Returns "unknown" when there is not enough history to form a trailing
    return, so callers get a value they can store rather than an exception.
    """
    close = pd.Series(close).dropna()
    if len(close) <= window:
        return UNKNOWN

    labels = label_regimes(close, window=window, threshold=threshold).dropna()
    if labels.empty:
        return UNKNOWN

    return STATES[int(labels.iloc[-1])]


def dominant_regime(
    close: pd.Series,
    *,
    window: int = 20,
    threshold: float = 0.02,
) -> str:
    """The most common regime across a window — a better descriptor of a
    multi-year backtest period than its final day, which is what a
    per-fold/per-window tag wants."""
    close = pd.Series(close).dropna()
    if len(close) <= window:
        return UNKNOWN

    labels = label_regimes(close, window=window, threshold=threshold).dropna()
    if labels.empty:
        return UNKNOWN

    return STATES[int(labels.value_counts().idxmax())]
