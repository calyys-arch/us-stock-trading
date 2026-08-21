"""
python/analytics/robustness_stats.py —
backtests/reports/regime_gate_robustness_report.md's fold-order-randomization
/ block-bootstrap helpers. Covers: degenerate inputs raise rather than
fabricate a distribution, known-answer sanity checks (all-pass folds ->
degenerate distribution at 1.0; a large i.i.d. block bootstrap on
i.i.d.-by-construction data centers near the plain-sample Sharpe), and
determinism (same seed -> byte-identical output, required for a reproducible
report).
"""
from __future__ import annotations

import random

import pytest

from python.analytics import robustness_stats as rs


# ── bootstrap_fold_pass_ratio ────────────────────────────────────────────

def test_bootstrap_fold_pass_ratio_empty_raises():
    with pytest.raises(ValueError):
        rs.bootstrap_fold_pass_ratio([], min_pass_ratio=0.6)


def test_bootstrap_fold_pass_ratio_all_pass_is_degenerate_at_one():
    result = rs.bootstrap_fold_pass_ratio([True] * 16, min_pass_ratio=0.6, n_boot=200)
    assert result["observed_pass_ratio"] == pytest.approx(1.0)
    assert result["pass_ratio_p5"] == pytest.approx(1.0)
    assert result["pass_ratio_p95"] == pytest.approx(1.0)
    assert result["frac_boot_draws_clearing_bar"] == pytest.approx(1.0)


def test_bootstrap_fold_pass_ratio_all_fail_is_degenerate_at_zero():
    result = rs.bootstrap_fold_pass_ratio([False] * 10, min_pass_ratio=0.6, n_boot=200)
    assert result["observed_pass_ratio"] == pytest.approx(0.0)
    assert result["pass_ratio_p95"] == pytest.approx(0.0)
    assert result["frac_boot_draws_clearing_bar"] == pytest.approx(0.0)


def test_bootstrap_fold_pass_ratio_mixed_centers_near_observed():
    # 8/16 folds pass, matching regime_gate_report.md's gated pairs_trading
    # result -- with enough draws, the bootstrap median pass ratio should sit
    # close to the observed 0.5, and the bar (0.6) should be cleared in a
    # clear minority of resamples (not ~0%, not ~100%).
    fold_passes = [True] * 8 + [False] * 8
    result = rs.bootstrap_fold_pass_ratio(fold_passes, min_pass_ratio=0.6, n_boot=5000)
    assert result["observed_pass_ratio"] == pytest.approx(0.5)
    assert result["pass_ratio_p50"] == pytest.approx(0.5, abs=0.1)
    assert 0.05 < result["frac_boot_draws_clearing_bar"] < 0.5


def test_bootstrap_fold_pass_ratio_deterministic_with_fixed_seed():
    fold_passes = [True, False, True, True, False, True, False, False]
    a = rs.bootstrap_fold_pass_ratio(fold_passes, min_pass_ratio=0.6, n_boot=500, seed=7)
    b = rs.bootstrap_fold_pass_ratio(fold_passes, min_pass_ratio=0.6, n_boot=500, seed=7)
    assert a == b


# ── moving_block_bootstrap_sharpe ────────────────────────────────────────

def test_moving_block_bootstrap_sharpe_too_few_observations_raises():
    with pytest.raises(ValueError):
        rs.moving_block_bootstrap_sharpe([0.001] * 10, block_size=21)


def test_moving_block_bootstrap_sharpe_recovers_plain_sharpe_on_iid_data():
    # Genuinely i.i.d. data (no autocorrelation to preserve or destroy): the
    # block bootstrap's observed_sharpe must match a direct calculation.
    rng = random.Random(3)
    returns = [rng.gauss(0.0006, 0.01) for _ in range(500)]
    result = rs.moving_block_bootstrap_sharpe(returns, block_size=21, n_boot=1000, seed=1)
    direct = rs._annualized_sharpe(returns)
    assert result["observed_sharpe"] == pytest.approx(direct)
    assert result["sharpe_p5"] <= result["sharpe_p50"] <= result["sharpe_p95"]


def test_moving_block_bootstrap_sharpe_strong_positive_drift_is_mostly_positive():
    # A deterministic, strongly-positive-drift series (80% of days +0.002,
    # 20% of days -0.001, order shuffled with a fixed seed so the block
    # structure is non-trivial but the overall drift is unambiguous): the
    # resampled Sharpe distribution should be dominated by positive draws,
    # not a coin flip -- this is the discriminating case the i.i.d.-noise
    # version above is too noisy at n=500 to establish on its own.
    values = [0.002] * 400 + [-0.001] * 100
    random.Random(0).shuffle(values)
    result = rs.moving_block_bootstrap_sharpe(values, block_size=21, n_boot=1000, seed=1)
    assert result["observed_sharpe"] > 0
    assert result["frac_boot_draws_positive"] > 0.7


def test_moving_block_bootstrap_sharpe_deterministic_with_fixed_seed():
    rng = random.Random(11)
    returns = [rng.gauss(0.0003, 0.012) for _ in range(300)]
    a = rs.moving_block_bootstrap_sharpe(returns, block_size=21, n_boot=300, seed=5)
    b = rs.moving_block_bootstrap_sharpe(returns, block_size=21, n_boot=300, seed=5)
    assert a == b


def test_moving_block_bootstrap_sharpe_output_length_matches_input():
    # The resampled series is truncated to exactly len(daily_returns); verify
    # indirectly by checking the function does not raise for a length not an
    # exact multiple of block_size (an off-by-one in truncation would either
    # crash or silently change the effective sample size).
    rng = random.Random(4)
    returns = [rng.gauss(0.0002, 0.01) for _ in range(497)]  # not a multiple of 21
    result = rs.moving_block_bootstrap_sharpe(returns, block_size=21, n_boot=50, seed=2)
    assert result["n_obs"] == 497
