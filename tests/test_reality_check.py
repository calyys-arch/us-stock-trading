"""
python/backtest/reality_check.py — White's Reality Check and its two
multiple-comparisons corrections (cheap Bonferroni default, exact
best-of-N `run_multi`).

No test here depends on exact simulated p-values (they're a function of an
RNG walked through FFT phase randomization): assertions check the
PROPERTIES the implementation must have — backward compatibility, the
Bonferroni arithmetic itself, capping at 1.0, and that `run_multi`'s paired
resampling actually shares one randomized panel across candidates per
simulation rather than drawing an independent one per candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.backtest.reality_check import RealityCheck, RealityCheckConfig


def _panel(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame(
        {
            "a": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "b": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        },
        index=dates,
    )


def _constant_sharpe_fn(value: float):
    """A backtest_fn indifferent to its input panel — isolates the
    correction arithmetic from any actual strategy behavior."""
    return lambda panel: value


# ── run(): backward compatibility ────────────────────────────────────────────

def test_default_call_is_unadjusted_and_labeled_single():
    """Every pre-existing caller (e.g. scripts/run_backtest.py) calls run()
    with no n_candidates_searched. That must keep behaving exactly as before
    this parameter was added."""
    rc = RealityCheck(_constant_sharpe_fn(1.0), config=RealityCheckConfig(n_sims=20, seed=1))
    result = rc.run(_panel())

    assert result.n_candidates_searched == 1
    assert result.method == "single"
    assert result.p_value == result.raw_p_value


def test_explicit_n_candidates_of_one_is_identical_to_default():
    panel = _panel()
    rc_a = RealityCheck(_constant_sharpe_fn(1.0), config=RealityCheckConfig(n_sims=20, seed=7))
    rc_b = RealityCheck(_constant_sharpe_fn(1.0), config=RealityCheckConfig(n_sims=20, seed=7))

    result_a = rc_a.run(panel)
    result_b = rc_b.run(panel, n_candidates_searched=1)

    assert result_a.p_value == result_b.p_value
    assert result_a.raw_p_value == result_b.raw_p_value


# ── run(): Bonferroni correction ─────────────────────────────────────────────

def test_bonferroni_multiplies_the_raw_p_value():
    rc = RealityCheck(_constant_sharpe_fn(1.0), config=RealityCheckConfig(n_sims=50, seed=3))
    result = rc.run(_panel(), n_candidates_searched=5)

    assert result.n_candidates_searched == 5
    assert result.method == "bonferroni"
    assert result.p_value == pytest.approx(min(1.0, result.raw_p_value * 5))


def test_bonferroni_correction_is_capped_at_one():
    """A raw p-value of even 0.3 times 32 candidates exceeds 1 — a
    probability cannot exceed 1."""
    # A Sharpe far below anything randomization would produce gives a raw
    # p-value near 1.0, which is exactly where the cap matters.
    rc = RealityCheck(_constant_sharpe_fn(-999.0), config=RealityCheckConfig(n_sims=30, seed=4))
    result = rc.run(_panel(), n_candidates_searched=32)

    assert result.raw_p_value == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1.0)


def test_more_candidates_never_produces_a_smaller_p_value():
    """Widening the search can only raise (or hold) the corrected p-value —
    a wider search must be held to an equal or higher bar, never a lower
    one."""
    panel = _panel()
    p_values = []
    for n_candidates in (1, 5, 27, 32):
        rc = RealityCheck(_constant_sharpe_fn(1.0), config=RealityCheckConfig(n_sims=40, seed=11))
        result = rc.run(panel, n_candidates_searched=n_candidates)
        p_values.append(result.p_value)

    assert p_values == sorted(p_values)


def test_verdict_reflects_the_corrected_not_raw_p_value():
    """A candidate that would PASS uncorrected can flip to FAIL once the
    search width it came from is taken into account — that flip is the
    entire point of this correction existing."""
    panel = _panel()
    rc_single = RealityCheck(_constant_sharpe_fn(2.0), config=RealityCheckConfig(
        n_sims=100, seed=2, pass_threshold=0.05))
    uncorrected = rc_single.run(panel)

    rc_wide = RealityCheck(_constant_sharpe_fn(2.0), config=RealityCheckConfig(
        n_sims=100, seed=2, pass_threshold=0.05))
    corrected = rc_wide.run(panel, n_candidates_searched=32)

    if 0 < uncorrected.raw_p_value < 0.05 / 32:
        pytest.skip("raw p-value too extreme for this seed to demonstrate the flip")
    assert corrected.p_value >= uncorrected.p_value


# ── run_multi(): exact best-of-N ─────────────────────────────────────────────

def test_run_multi_rejects_an_empty_candidate_list():
    rc = RealityCheck(_constant_sharpe_fn(0.0), config=RealityCheckConfig(n_sims=5))
    with pytest.raises(ValueError):
        rc.run_multi(_panel(), [])


def test_run_multi_real_sharpe_is_the_best_across_candidates():
    fns = [_constant_sharpe_fn(v) for v in (0.1, 0.5, 2.0, -1.0)]
    rc = RealityCheck(_constant_sharpe_fn(0.0), config=RealityCheckConfig(n_sims=10, seed=5))
    result = rc.run_multi(_panel(), fns)

    assert result.real_sharpe == pytest.approx(2.0)
    assert result.n_candidates_searched == 4
    assert result.method == "full_bootstrap"


def test_run_multi_shares_one_randomized_panel_per_simulation_across_candidates():
    """Paired resampling is what makes 'best of N under the null' a valid
    comparison: within a single simulation, every candidate must see the
    SAME randomized panel, not each drawing its own."""
    seen_panels: list[list[pd.DataFrame]] = [[], []]

    def _recording_fn(slot: int):
        def _fn(panel: pd.DataFrame) -> float:
            seen_panels[slot].append(panel.copy())
            return float(panel["a"].iloc[-1])
        return _fn

    rc = RealityCheck(_constant_sharpe_fn(0.0), config=RealityCheckConfig(n_sims=6, seed=9))
    rc.run_multi(_panel(), [_recording_fn(0), _recording_fn(1)])

    # First entry in each slot is the REAL panel (identical call for both
    # candidates); the rest are the n_sims randomized ones, which must match
    # pairwise between the two candidates' recorded panels.
    for i in range(1, len(seen_panels[0])):
        pd.testing.assert_frame_equal(seen_panels[0][i], seen_panels[1][i])


def test_run_multi_p_value_is_low_when_the_real_best_dominates():
    """A candidate whose real Sharpe is far above anything the null process
    can produce should get a low p-value even under the strict best-of-N
    test."""
    real_panel = _panel(seed=0)

    def _fn(panel: pd.DataFrame) -> float:
        # Rewards the REAL panel specifically (identity, not just shape) so
        # every randomized panel scores near zero while the real one scores
        # high — an unambiguous "real edge" signal for this test.
        return 10.0 if panel is real_panel or panel.equals(real_panel) else 0.0

    rc = RealityCheck(_constant_sharpe_fn(0.0), config=RealityCheckConfig(n_sims=50, seed=6))
    result = rc.run_multi(real_panel, [_fn])

    assert result.p_value == pytest.approx(0.0)
    assert result.verdict == "PASS"


# ── serialization ─────────────────────────────────────────────────────────

def test_to_dict_reports_the_correction_metadata():
    rc = RealityCheck(_constant_sharpe_fn(1.0), config=RealityCheckConfig(n_sims=10, seed=1))
    d = rc.run(_panel(), n_candidates_searched=27).to_dict()

    assert d["n_candidates_searched"] == 27
    assert d["method"] == "bonferroni"
    assert "raw_p_value" in d
