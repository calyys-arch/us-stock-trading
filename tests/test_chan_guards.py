"""
Chan Ch.3 methodology guards: parameter-count discipline, sample-size
sufficiency, train/test split structure, and 4-sigma data-quality checks.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from python.backtest.param_guard import (
    MAX_FREE_PARAMETERS,
    check_max_parameters,
    required_days,
    sufficient_sample_size,
)
from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig
from python.core.data_quality import flag_extreme_moves, quality_report

STRATEGY_CONFIG_PATH = "configs/strategy.yaml"


def _load_strategy_configs() -> dict:
    with open(STRATEGY_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_every_strategy_has_at_most_five_free_parameters():
    configs = _load_strategy_configs()
    assert configs, "configs/strategy.yaml must not be empty"
    for name, cfg in configs.items():
        ok, n = check_max_parameters(cfg)
        assert ok, f"strategy '{name}' has {n} free parameters (> {MAX_FREE_PARAMETERS} allowed)"


def test_pairs_trading_parameter_count_is_pinned():
    """Regression pin: pairs_trading currently uses exactly 5 free
    parameters (entry_z, exit_z, half_life_multiplier_max_hold,
    min_half_life_days, max_half_life_days) — exactly at the Chan-discipline
    ceiling. If this grows further, it must be a deliberate, reviewed
    decision (with an accompanying MAX_FREE_PARAMETERS discussion), not
    silent config creep."""
    configs = _load_strategy_configs()
    cfg = configs["pairs_trading"]
    ok, n = check_max_parameters(cfg)
    assert n == MAX_FREE_PARAMETERS
    assert ok


def test_sufficient_sample_size_rule_of_thumb():
    # 3 parameters -> need >= 756 trading days (3 years) of tested history.
    assert required_days(3) == 756
    assert sufficient_sample_size(800, 3) is True
    assert sufficient_sample_size(400, 3) is False


def test_walk_forward_enforces_disjoint_train_test_windows():
    """Every WFO fold's OOS window must start exactly where the IS window
    ends (no overlap = no leakage of the 'test' period into 'train')."""
    def fake_backtest_fn(start, end, params):
        # Deterministic function of the date range so we can distinguish folds.
        return {"sharpe_ratio": (end - start).days / 100.0}

    cfg = WFOConfig(is_days=100, oos_days=50, step_days=50, min_pass_folds_ratio=0.0,
                     min_oos_sharpe_abs=-999, max_sharpe_decay=1.0)
    optimizer = WalkForwardOptimizer(fake_backtest_fn, config=cfg)
    result = optimizer.run(datetime(2015, 1, 1), datetime(2020, 1, 1))

    assert result.total_folds > 0
    for fold in result.folds:
        is_end = datetime.fromisoformat(fold.is_end)
        oos_start = datetime.fromisoformat(fold.oos_start)
        is_start = datetime.fromisoformat(fold.is_start)
        oos_end = datetime.fromisoformat(fold.oos_end)
        assert is_end == oos_start, "OOS window must start exactly at IS window end (no overlap, no gap)"
        assert is_start < is_end < oos_end


def test_four_sigma_extreme_move_flagging():
    rng = np.random.default_rng(1)
    normal_returns = pd.Series(rng.normal(0, 0.01, 200))
    # Inject an obvious bad print (e.g. an un-adjusted stock split) at index 150.
    normal_returns.iloc[150] = 0.80

    flagged = flag_extreme_moves(normal_returns, sigma_threshold=4.0, rolling_window=60)
    assert flagged.iloc[150] == True  # noqa: E712
    # Should not flag the vast majority of ordinary observations.
    assert flagged.sum() < 5


def test_quality_report_detects_zero_and_duplicate_prices():
    prices = pd.Series([100.0, 100.0, 100.0, 101.0, 0.0, 102.0, 102.0])
    report = quality_report(prices)
    assert report["n_zero_or_negative_prices"] == 1
    assert report["n_duplicated_consecutive_prices"] >= 2
