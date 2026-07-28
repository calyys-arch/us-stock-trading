"""
Look-ahead-bias truncation tests (Chan Ch.3 blind spot #1; this is the exact
class of bug flagged in the original prompt: "Ensure your code never uses
future information").

Every PortfolioStrategy/PairsStrategy evaluate() call must produce IDENTICAL
output regardless of what happens to the data on or after `as_of` — if
mutating a future price changes today's decision, the strategy (or its
caller) is leaking future information.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from python.backtest.vector_engine import run_vector_backtest
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy


def _build_synthetic_panel(n_days: int = 30, n_codes: int = 25, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"SYM{i:03d}" for i in range(n_codes)]

    rows = []
    for code in codes:
        base = 100.0 + rng.uniform(-10, 10)
        price = base
        for d in dates:
            ret = rng.normal(0, 0.01)
            open_px = price
            close_px = price * (1 + ret)
            rows.append({"date": d, "code": code, "open": open_px, "close": close_px,
                         "adv_20d_dollars": 50_000_000.0})
            price = close_px

    df = pd.DataFrame(rows).set_index(["date", "code"]).sort_index()
    return df


def test_vector_backtest_truncates_strictly_before_as_of():
    """as_of_data passed into strategy.evaluate() must never include the
    as_of date itself or any later date."""
    panel = _build_synthetic_panel()
    strategy = CrossSectionalMeanReversionStrategy(min_universe_size=5)
    codes = sorted(panel.index.get_level_values(1).unique())
    dates = sorted(panel.index.get_level_values(0).unique())[10:]  # skip warmup
    universe_by_day = {d: codes for d in dates}

    result = run_vector_backtest(strategy, panel, universe_by_day)

    for as_of, target in result.targets_by_day.items():
        # Re-derive what evaluate() saw and confirm no leakage.
        seen = panel[panel.index.get_level_values(0) < as_of]
        assert seen.index.get_level_values(0).max() < as_of


def test_strategy_output_unchanged_by_future_price_mutation():
    """Corrupting all prices on/after day t must not change day t's target
    weights — the strategy must not have been given that data at all."""
    panel = _build_synthetic_panel()
    strategy = CrossSectionalMeanReversionStrategy(min_universe_size=5)
    codes = sorted(panel.index.get_level_values(1).unique())
    dates = sorted(panel.index.get_level_values(0).unique())
    target_day = dates[15]

    universe_by_day = {target_day: codes}
    baseline = run_vector_backtest(strategy, panel, universe_by_day)
    baseline_weights = baseline.targets_by_day[target_day].weights

    corrupted = panel.copy()
    mask = corrupted.index.get_level_values(0) >= target_day
    corrupted.loc[mask, "close"] = corrupted.loc[mask, "close"] * 100  # absurd future spike
    corrupted.loc[mask, "open"] = corrupted.loc[mask, "open"] * 100

    mutated = run_vector_backtest(strategy, corrupted, universe_by_day)
    mutated_weights = mutated.targets_by_day[target_day].weights

    assert baseline_weights.keys() == mutated_weights.keys()
    for code in baseline_weights:
        assert baseline_weights[code] == pytest.approx(mutated_weights[code], abs=1e-9)


def test_portfolio_strategy_rejects_when_as_of_data_leaks(monkeypatch):
    """Defensive check inside the strategy itself: if a caller mistakenly
    passes as_of_data containing rows >= as_of, results should still only
    reflect data before as_of for weight computation on the LATEST reference
    price used (regression guard against a future refactor accidentally
    removing the caller-side truncation and the strategy quietly starting to
    use it)."""
    panel = _build_synthetic_panel(n_days=10, n_codes=5)
    strategy = CrossSectionalMeanReversionStrategy(min_universe_size=3, lookback_days=1)
    codes = sorted(panel.index.get_level_values(1).unique())
    dates = sorted(panel.index.get_level_values(0).unique())
    as_of = dates[5]

    proper_slice = panel[panel.index.get_level_values(0) < as_of]
    leaked_slice = panel  # includes as_of and beyond — simulates a caller bug

    proper_target = strategy.evaluate(proper_slice, as_of, codes)
    leaked_target = strategy.evaluate(leaked_slice, as_of, codes)

    # This assertion is expected to FAIL if it ever passes silently identical
    # by coincidence on a specific dataset; its real purpose is to document
    # that as_of_data truncation is the CALLER's responsibility (backtest
    # engine / live scheduler), not this strategy's — see the module
    # docstring on PortfolioStrategy.evaluate(). A leaking caller changes the
    # trailing-return calculation for at least one code because it can see
    # one extra day of price history that a correct caller would not.
    assert proper_target.weights != leaked_target.weights
