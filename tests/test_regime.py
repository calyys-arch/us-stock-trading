"""
python/analytics/regime.py — report-only Markov regime diagnostic. Covers
labeling correctness (no-lookahead, leading-NaN handling), transition-matrix
MLE counting (including the zero-row uniform-fallback), stationary
distribution, and the end-to-end RegimeReport contract. The naive backtest
is checked only for its "honest degrade" contract (None when data is thin,
a clearly-labeled dict otherwise) — its actual Sharpe number is explicitly
NOT a thing this test suite should assert is "good", since the module's own
docstring says it is illustrative, not validated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.analytics import regime as reg


def _prices(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, name="close")


def test_label_regimes_bull_bear_sideways_and_leading_nan_dropped():
    # window=5: flat for 5 days, then a clear +10% run, then flat, then a clear -10% run.
    values = [100.0] * 6 + [110.0] * 6 + [110.0] * 6 + [99.0] * 6
    close = _prices(values)
    labels = reg.label_regimes(close, window=5, threshold=0.02)

    # first `window` days have no trailing return -> dropped, not defaulted to Sideways.
    assert close.index[0] not in labels.index
    assert close.index[4] not in labels.index

    # day 10 (index 10): close=110 vs close[5]=100 -> +10% -> Bull.
    assert labels.loc[close.index[10]] == reg._BULL
    # day 22: close=99 vs close[17]=110 -> -10% -> Bear.
    assert labels.loc[close.index[22]] == reg._BEAR


def test_label_regimes_flat_series_is_all_sideways():
    close = _prices([100.0] * 40)
    labels = reg.label_regimes(close, window=10, threshold=0.02)
    assert (labels == reg._SIDEWAYS).all()
    assert len(labels) == 30  # 40 - window(10)


def test_build_transition_matrix_counts_correctly():
    # Sequence: Bear, Bear, Sideways, Bull, Bull, Bull, Sideways, Bear
    labels = pd.Series([0, 0, 1, 2, 2, 2, 1, 0], index=pd.RangeIndex(8))
    P = reg.build_transition_matrix(labels)

    # Bear->Bear once, Bear->Sideways once out of 2 Bear->X transitions (indices 0,1,7 are Bear but
    # only transitions FROM Bear that have a next state count: (0->0), (0->1), (7 has no next)).
    assert P[0, 0] == pytest.approx(0.5)
    assert P[0, 1] == pytest.approx(0.5)
    assert P[0, 2] == pytest.approx(0.0)
    # rows sum to 1
    assert np.allclose(P.sum(axis=1), 1.0)


def test_build_transition_matrix_never_visited_state_gets_uniform_row():
    # Only Bull and Sideways ever appear — Bear (state 0) row must fall back to uniform,
    # not an all-zero (non-stochastic) row.
    labels = pd.Series([1, 2, 1, 2, 1, 2], index=pd.RangeIndex(6))
    P = reg.build_transition_matrix(labels)
    assert np.allclose(P[0], [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(P.sum(axis=1), 1.0)


def test_stationary_distribution_sums_to_one_and_matches_known_matrix():
    # A matrix where every row is identical is already its own stationary distribution.
    P = np.array([
        [0.2, 0.5, 0.3],
        [0.2, 0.5, 0.3],
        [0.2, 0.5, 0.3],
    ])
    pi = reg.stationary_distribution(P)
    assert pi.sum() == pytest.approx(1.0)
    assert np.allclose(pi, [0.2, 0.5, 0.3], atol=1e-6)


def test_n_step_forecast_matches_matrix_power():
    P = reg.build_transition_matrix(pd.Series([0, 1, 2, 1, 0, 1, 2], index=pd.RangeIndex(7)))
    assert np.allclose(reg.n_step_forecast(P, 1), P)
    assert np.allclose(reg.n_step_forecast(P, 2), P @ P)


def test_illustrative_naive_backtest_none_when_insufficient_data():
    close = _prices(list(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, 60))))
    labels = reg.label_regimes(close, window=20, threshold=0.02)
    assert reg.illustrative_naive_backtest(close, labels, min_train=252) is None


def test_illustrative_naive_backtest_returns_labeled_dict_with_enough_data():
    rng = np.random.default_rng(42)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 500)))
    close = _prices(list(values))
    labels = reg.label_regimes(close, window=20, threshold=0.02)
    result = reg.illustrative_naive_backtest(close, labels, min_train=252)

    assert result is not None
    assert set(result) == {"sharpe_naive_no_cost", "max_drawdown_naive_no_cost", "n_days", "note"}
    assert result["n_days"] > 0
    assert "Illustrative only" in result["note"]
    assert "NOT a validated strategy" in result["note"]


def test_compute_regime_report_end_to_end_contract():
    rng = np.random.default_rng(7)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 400)))
    close = _prices(list(values))

    report = reg.compute_regime_report(close, symbol="TEST", window=20, threshold=0.02, min_train=252, recent_days=30)

    assert report.symbol == "TEST"
    assert report.current_state in reg.STATES
    assert set(report.transition_matrix) == set(reg.STATES)
    for row in report.transition_matrix.values():
        assert set(row) == set(reg.STATES)
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(report.stationary_distribution.values()) == pytest.approx(1.0, abs=1e-4)
    assert set(report.stationary_distribution) == set(reg.STATES)
    assert len(report.recent_history) == 30
    assert all(entry["state"] in reg.STATES for entry in report.recent_history)
    assert report.as_of == close.index[-1].strftime("%Y-%m-%d")

    as_dict = report.to_dict()
    assert as_dict["symbol"] == "TEST"
    assert as_dict["naive_backtest"] is not None  # 400 days > min_train(252) + 30


def test_compute_regime_report_raises_on_insufficient_history():
    close = _prices([100.0] * 5)
    with pytest.raises(ValueError):
        reg.compute_regime_report(close, symbol="TOO_SHORT", window=20)


def test_no_lookahead_current_state_matches_last_label_only():
    """Regression guard: compute_regime_report's current_state must come
    from the LAST labeled day, and the transition matrix must be built
    from the full label history INCLUDING that day (matching
    build_transition_matrix's own no-lookahead contract of only ever
    counting transitions that have already happened)."""
    rng = np.random.default_rng(1)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))
    close = _prices(list(values))
    labels = reg.label_regimes(close, window=20, threshold=0.02)

    report = reg.compute_regime_report(close, symbol="X", window=20, threshold=0.02, min_train=252)
    assert report.current_state == reg.STATES[int(labels.iloc[-1])]
