"""
python/backtest/walk_forward.py — the `WFOConfig.anchored` fold-structure
convention added for `backtests/reports/regime_gate_robustness_report.md`'s
WFO-fold-convention comparison.

`anchored=False` is the DEFAULT and must stay byte-identical to the
pre-existing rolling-fixed-window behavior already exercised by
`tests/test_backtest_engine_audit.py` — this file does not re-litigate that
(see that file for the fold-boundary/no-lookahead audit); it pins ONLY the
NEW `anchored=True` behavior and the fact that turning it on changes nothing
about `anchored=False`'s output.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig


def _flat_backtest_fn(start, end, params):
    return {"sharpe_ratio": 0.2}


def test_anchored_defaults_to_false():
    assert WFOConfig().anchored is False


def test_anchored_false_is_byte_identical_to_omitting_it():
    cfg_explicit = WFOConfig(is_days=200, oos_days=60, step_days=60, anchored=False)
    cfg_default = WFOConfig(is_days=200, oos_days=60, step_days=60)
    start, end = datetime(2020, 1, 1), datetime(2022, 6, 1)
    a = WalkForwardOptimizer(_flat_backtest_fn, cfg_explicit, [{}]).run(start, end)
    b = WalkForwardOptimizer(_flat_backtest_fn, cfg_default, [{}]).run(start, end)
    a_dict, b_dict = a.to_dict(), b.to_dict()
    a_dict.pop("run_at"), b_dict.pop("run_at")   # wall-clock timestamp, not behavior
    assert a_dict == b_dict


def test_anchored_is_start_stays_fixed_across_folds():
    cfg = WFOConfig(is_days=200, oos_days=60, step_days=60, anchored=True,
                     min_pass_folds_ratio=0.0)
    start, end = datetime(2020, 1, 1), datetime(2022, 6, 1)
    wfo = WalkForwardOptimizer(_flat_backtest_fn, cfg, [{}]).run(start, end)
    assert len(wfo.folds) >= 3
    for fold in wfo.folds:
        assert datetime.fromisoformat(fold.is_start) == start


def test_anchored_is_end_grows_by_step_days_each_fold():
    cfg = WFOConfig(is_days=200, oos_days=60, step_days=60, anchored=True,
                     min_pass_folds_ratio=0.0)
    start, end = datetime(2020, 1, 1), datetime(2023, 1, 1)
    wfo = WalkForwardOptimizer(_flat_backtest_fn, cfg, [{}]).run(start, end)
    assert len(wfo.folds) >= 3
    is_ends = [datetime.fromisoformat(f.is_end) for f in wfo.folds]
    for k, is_end in enumerate(is_ends):
        assert is_end == start + timedelta(days=cfg.is_days + k * cfg.step_days)
    # Strictly growing, unlike the rolling case where IS length is constant.
    is_lengths = [
        (datetime.fromisoformat(f.is_end) - datetime.fromisoformat(f.is_start)).days
        for f in wfo.folds
    ]
    assert is_lengths == sorted(is_lengths)
    assert is_lengths[-1] > is_lengths[0]


def test_anchored_fold_0_matches_rolling_fold_0_exactly():
    # Both conventions must agree on the very first fold (same IS window,
    # since the anchored IS window starts at is_days length just like the
    # rolling window's constant length) -- they only diverge from fold 1
    # onward.
    start, end = datetime(2020, 1, 1), datetime(2021, 6, 1)
    rolling = WalkForwardOptimizer(
        _flat_backtest_fn, WFOConfig(is_days=200, oos_days=60, step_days=60, anchored=False), [{}]
    ).run(start, end)
    anchored = WalkForwardOptimizer(
        _flat_backtest_fn, WFOConfig(is_days=200, oos_days=60, step_days=60, anchored=True), [{}]
    ).run(start, end)
    assert rolling.folds[0].is_start == anchored.folds[0].is_start
    assert rolling.folds[0].is_end == anchored.folds[0].is_end
    assert rolling.folds[0].oos_start == anchored.folds[0].oos_start
    assert rolling.folds[0].oos_end == anchored.folds[0].oos_end


def test_anchored_oos_windows_stay_gapless_and_non_overlapping_with_is():
    cfg = WFOConfig(is_days=100, oos_days=40, step_days=40, anchored=True,
                     min_pass_folds_ratio=0.0)
    wfo = WalkForwardOptimizer(_flat_backtest_fn, cfg, [{}]).run(
        datetime(2020, 1, 1), datetime(2022, 1, 1))
    assert len(wfo.folds) >= 3
    for fold in wfo.folds:
        is_start = datetime.fromisoformat(fold.is_start)
        is_end = datetime.fromisoformat(fold.is_end)
        oos_start = datetime.fromisoformat(fold.oos_start)
        oos_end = datetime.fromisoformat(fold.oos_end)
        assert is_start < is_end == oos_start < oos_end
        assert (oos_end - oos_start).days == cfg.oos_days


def test_anchored_reflected_in_config_dict():
    cfg = WFOConfig(anchored=True)
    wfo = WalkForwardOptimizer(_flat_backtest_fn, cfg, [{}]).run(
        datetime(2020, 1, 1), datetime(2022, 1, 1))
    assert wfo.to_dict()["config"]["anchored"] is True
