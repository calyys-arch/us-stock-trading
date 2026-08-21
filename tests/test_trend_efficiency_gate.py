"""
python/analytics/trend_efficiency_gate.py — Phase 2 regime-GATING classifier
(backtests/reports/regime_gate_report.md). Covers: efficiency-ratio
correctness on known synthetic shapes (pure trend -> ER near 1, a round-trip
-> ER near 0, flat price -> undefined not fabricated), the trailing-median
gate rule, and — the most important guarantee, matching
tests/test_regime.py's own no-lookahead pattern and
tests/test_lookahead_bias.py's project-wide convention — that mutating
FUTURE prices never changes a PAST label, and that `shifted_entry_gate`'s
entry decision for day t depends only on data available through day t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.analytics import trend_efficiency_gate as teg


def _prices(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, name="close")


def test_efficiency_ratio_pure_trend_is_near_one():
    # Monotonically increasing by a constant step every day: net drift ==
    # path length exactly -> ER == 1.0 for every fully-windowed row.
    values = [100.0 + i for i in range(60)]
    close = _prices(values)
    er = teg.efficiency_ratio(close, window=20)
    assert er.dropna().apply(lambda x: x == pytest.approx(1.0)).all()


def test_efficiency_ratio_round_trip_is_near_zero():
    # Up N/2 days, back down N/2 days to (approximately) the start: net
    # drift ~ 0 while path length is large -> ER near 0.
    up = [100.0 + i for i in range(1, 11)]
    down = [110.0 - i for i in range(1, 11)]
    values = [100.0] + up + down  # 21 points, back to ~100
    close = _prices(values)
    er = teg.efficiency_ratio(close, window=20)
    last = er.iloc[-1]
    assert last == pytest.approx(0.0, abs=1e-9)


def test_efficiency_ratio_flat_price_is_nan_not_fabricated():
    close = _prices([100.0] * 40)
    er = teg.efficiency_ratio(close, window=20)
    # Zero path length (nothing moved) is UNDEFINED, not "perfectly trending"
    # or "perfectly choppy" — must stay NaN, never silently become 0.0 or 1.0.
    assert er.dropna().empty


def test_efficiency_ratio_leading_window_is_nan():
    close = _prices(list(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, 30))))
    er = teg.efficiency_ratio(close, window=20)
    assert er.iloc[:20].isna().all()
    assert er.iloc[20:].notna().all()


def test_efficiency_ratio_bounded_zero_to_one():
    rng = np.random.default_rng(3)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 400)))
    close = _prices(list(values))
    er = teg.efficiency_ratio(close, window=20).dropna()
    assert (er >= 0.0).all() and (er <= 1.0).all()


def test_compute_gate_labels_undecided_before_reference_window():
    rng = np.random.default_rng(5)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 100)))
    close = _prices(list(values))
    labels = teg.compute_gate_labels(close, window=20, reference_window=252)
    # Fewer than `reference_window` days of ER history anywhere in a
    # 100-day series -> every row must be undecided (pd.NA), never a
    # fabricated True/False.
    assert labels.isna().all()


def test_compute_gate_labels_on_when_er_below_trailing_median():
    # Construct a long, mildly noisy trend (moderate, fairly stable ER) with
    # one sharp round-trip spliced in near the end: the round-trip's ER
    # should sit BELOW the trailing median built mostly from the trend
    # portion, i.e. the gate should read True (mean-reversion allowed)
    # exactly on the choppy portion.
    rng = np.random.default_rng(11)
    trend = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.003, 320)))
    last = trend[-1]
    up = last * np.exp(np.cumsum(np.full(10, 0.01)))
    down = up[-1] * np.exp(np.cumsum(np.full(10, -0.01)))
    values = list(trend) + list(up) + list(down)
    close = _prices(values)

    labels = teg.compute_gate_labels(close, window=20, reference_window=252)
    decided = labels.dropna()
    assert len(decided) > 0
    # The very last day (deep in the round-trip, ER should be low) must be
    # decidable and gate-ON.
    assert bool(labels.iloc[-1]) is True


def test_shifted_entry_gate_defaults_closed_when_undecided():
    rng = np.random.default_rng(6)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 50)))
    close = _prices(list(values))
    gate = teg.shifted_entry_gate(close, window=20, reference_window=252)
    assert gate.dtype == bool
    assert not gate.any()  # far too little history for reference_window=252 anywhere


def test_shifted_entry_gate_is_boolean_never_na():
    rng = np.random.default_rng(9)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 400)))
    close = _prices(list(values))
    gate = teg.shifted_entry_gate(close, window=20, reference_window=252)
    assert gate.isna().sum() == 0
    assert set(gate.unique().tolist()) <= {True, False}


# ── No-lookahead (the guarantee that actually matters) ─────────────────────

def test_no_lookahead_mutating_future_prices_never_changes_past_labels():
    """The project-wide anti-lookahead pattern (tests/test_lookahead_bias.py,
    tests/test_regime.py::test_no_lookahead_current_state_matches_last_
    label_only): corrupt every price AFTER a cutoff day and confirm every
    label ON OR BEFORE the cutoff is bit-identical."""
    rng = np.random.default_rng(13)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 500)))
    close = _prices(list(values))

    cutoff = 350
    corrupted = close.copy()
    corrupted.iloc[cutoff + 1:] = rng.normal(1.0, 500.0, len(close) - cutoff - 1)

    er_original = teg.efficiency_ratio(close, window=20)
    er_corrupted = teg.efficiency_ratio(corrupted, window=20)
    pd.testing.assert_series_equal(
        er_original.iloc[: cutoff + 1], er_corrupted.iloc[: cutoff + 1]
    )

    labels_original = teg.compute_gate_labels(close, window=20, reference_window=252)
    labels_corrupted = teg.compute_gate_labels(corrupted, window=20, reference_window=252)
    pd.testing.assert_series_equal(
        labels_original.iloc[: cutoff + 1], labels_corrupted.iloc[: cutoff + 1]
    )

    gate_original = teg.shifted_entry_gate(close, window=20, reference_window=252)
    gate_corrupted = teg.shifted_entry_gate(corrupted, window=20, reference_window=252)
    # shifted_entry_gate(t) reads compute_gate_labels(t-1), so it is safe
    # through `cutoff` inclusive too (label(cutoff) is unaffected, and
    # gate(cutoff) = label(cutoff - 1), also unaffected).
    pd.testing.assert_series_equal(
        gate_original.iloc[: cutoff + 1], gate_corrupted.iloc[: cutoff + 1]
    )


def test_shifted_entry_gate_day_t_never_uses_day_t_close():
    """Direct causality pin: changing ONLY today's close must never change
    TODAY's entry-gate decision (which must be a function of yesterday's
    label, i.e. of data strictly before today)."""
    rng = np.random.default_rng(21)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 400)))
    close = _prices(list(values))
    probe_day = 300

    perturbed = close.copy()
    perturbed.iloc[probe_day] = perturbed.iloc[probe_day] * 5.0  # a huge, obvious one-day shock

    gate_original = teg.shifted_entry_gate(close, window=20, reference_window=252)
    gate_perturbed = teg.shifted_entry_gate(perturbed, window=20, reference_window=252)
    assert gate_original.iloc[probe_day] == gate_perturbed.iloc[probe_day]


def test_load_regime_proxy_close_uses_local_cache(tmp_path):
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=300)
    s = pd.Series(100.0 + np.arange(300), index=idx, name="close")
    path = tmp_path / "SPY.csv"
    s.to_csv(path, header=True)
    out = teg.load_regime_proxy_close("SPY", lookback_days=800, cache_path=path)
    assert out is not None
    assert len(out) >= 272
    assert out.iloc[-1] == pytest.approx(100.0 + 299)
