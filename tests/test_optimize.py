"""
optimize.py tests — the WFO<->engine bridge layer.

The critical case is pairs warmup: run_pairs_backtest consumes the first
coint_lookback_days rows as pure warmup, so a short OOS window MUST be
handed extra history before `start` — and metrics must still come ONLY
from inside [start, end). These tests verify both halves of that contract,
plus grid expansion/loading, parameter-discipline pre-flight, per-strategy
WFO config overrides, and the drawdown / has-trades gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.backtest.intraday_engine import IntradayBacktestConfig
from python.backtest.optimize import (
    build_intraday_backtest_fn,
    build_pairs_backtest_fn,
    build_xsection_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    check_min_trades_gate,
    check_profit_factor_gate,
    expand_param_grid,
    load_param_grid,
    load_wfo_config,
    preflight_check,
    run_intraday_stress_test,
)
from python.backtest.walk_forward import FoldResult, WFOResult


def _cointegrated_pair(n_days: int = 900, seed: int = 5):
    """Same construction as run_backtest._synthetic_pair (genuinely
    cointegrated, multi-day half-life) — duplicated minimally here so the
    test doesn't import from scripts/."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    common = np.cumsum(rng.normal(0, 1, n_days))
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = 0.92 * spread[t - 1] + rng.normal(0, 0.15)
    log_a = 4.0 + 0.1 * common + 0.5 * spread + rng.normal(0, 0.02, n_days)
    log_b = 3.8 + 0.1 * common - 0.5 * spread + rng.normal(0, 0.02, n_days)
    return pd.Series(np.exp(log_a), index=dates), pd.Series(np.exp(log_b), index=dates)


_PAIRS_BASE_CFG = {
    "entry_z": 2.0, "exit_z": 0.5, "coint_lookback_days": 252,
    "revalidate_every_days": 21, "notional_per_leg": 50_000.0,
    "half_life_multiplier_max_hold": 3.0, "min_half_life_days": 1.0,
    "max_half_life_days": 60.0,
}


# ── grid expansion / loading ─────────────────────────────────────────────────

def test_expand_param_grid_cartesian_product():
    combos = expand_param_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(combos) == 6
    assert {"a": 1, "b": 20} in combos
    assert expand_param_grid({}) == [{}]


def test_load_param_grid_from_repo_config():
    combos = load_param_grid("pairs_trading")
    assert len(combos) >= 4
    assert all("entry_z" in c for c in combos)
    with pytest.raises(KeyError):
        load_param_grid("no_such_strategy")


# ── pre-flight discipline ────────────────────────────────────────────────────

def test_preflight_rejects_unknown_keys():
    with pytest.raises(ValueError, match="not in configs/strategy.yaml"):
        preflight_check("pairs_trading", _PAIRS_BASE_CFG, [{"brand_new_knob": 1.0}])


def test_preflight_accepts_repo_grids_against_repo_configs():
    import yaml

    with open("configs/strategy.yaml", encoding="utf-8") as f:
        strategy_cfg = yaml.safe_load(f)
    for name in ("pairs_trading", "xsection_mean_reversion"):
        preflight_check(name, strategy_cfg[name], load_param_grid(name))


def test_preflight_warns_but_does_not_reject_short_sample_window():
    # 3 free params -> Chan's rule of thumb wants >= 756 trading days.
    # 300 is short: preflight must WARN (return sample_size_sufficient=False)
    # without raising — this is a soft gate, unlike check_max_parameters.
    result = preflight_check(
        "pairs_trading", _PAIRS_BASE_CFG, [{}], total_trading_days=300)
    assert result["sample_size_sufficient"] is False
    assert result["required_days"] == 252 * result["n_free_parameters"]


def test_preflight_reports_sufficient_sample_when_window_long_enough():
    result = preflight_check(
        "pairs_trading", _PAIRS_BASE_CFG, [{}], total_trading_days=252 * 10)
    assert result["sample_size_sufficient"] is True


def test_preflight_omits_sample_size_fields_when_days_not_given():
    result = preflight_check("pairs_trading", _PAIRS_BASE_CFG, [{}])
    assert "sample_size_sufficient" not in result
    assert "n_free_parameters" in result


# ── WFO config overrides ─────────────────────────────────────────────────────

def test_load_wfo_config_applies_per_strategy_override():
    xsection = load_wfo_config("xsection_mean_reversion")
    pairs = load_wfo_config("pairs_trading")
    assert xsection.is_days == 504
    assert pairs.is_days == 1008          # goal.yaml wfo.pairs_trading.is_days
    assert pairs.oos_days == xsection.oos_days == 126


# ── pairs warmup contract ────────────────────────────────────────────────────

def test_pairs_short_window_trades_thanks_to_warmup():
    """A 6-month window fed WITHOUT warmup would be entirely consumed by the
    252-day cointegration lookback (zero trades, sharpe 0). With the
    warmup slice, the same window must produce trades."""
    prices_a, prices_b = _cointegrated_pair()
    fn = build_pairs_backtest_fn("A", "B", prices_a, prices_b, _PAIRS_BASE_CFG)
    start = prices_a.index[400]
    end = prices_a.index[520]  # ~120 trading days << 252
    metrics = fn(start, end, {})
    assert metrics["n_trades"] > 0
    assert metrics["n_days"] > 0


def test_pairs_metrics_only_from_inside_window():
    """Same data, two disjoint windows: their daily-return day counts must
    match the window sizes, proving warmup rows never leak into metrics."""
    prices_a, prices_b = _cointegrated_pair()
    fn = build_pairs_backtest_fn("A", "B", prices_a, prices_b, _PAIRS_BASE_CFG)
    start, end = prices_a.index[500], prices_a.index[600]
    metrics = fn(start, end, {})
    # engine books daily_pnl only for dates it iterates INSIDE the window
    assert metrics["n_days"] <= 100
    assert len(metrics["daily_returns"]) == metrics["n_days"]


def test_pairs_insufficient_history_returns_empty_metrics():
    prices_a, prices_b = _cointegrated_pair(n_days=300)
    fn = build_pairs_backtest_fn("A", "B", prices_a, prices_b, _PAIRS_BASE_CFG)
    # window at the very start of the data: only ~50 warmup rows exist
    metrics = fn(prices_a.index[10], prices_a.index[60], {})
    assert metrics["n_trades"] == 0
    assert metrics["sharpe_ratio"] == 0.0


# ── xsection window restriction ──────────────────────────────────────────────

def _xsection_panel(n_days: int = 300, n_codes: int = 20, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rows = []
    for i in range(n_codes):
        price = 100.0
        for d in dates:
            open_px = price
            close_px = price * (1 + rng.normal(0, 0.01))
            rows.append({"date": d, "code": f"S{i:02d}", "open": open_px,
                         "close": close_px, "adv_20d_dollars": 3e7})
            price = close_px
    return pd.DataFrame(rows).set_index(["date", "code"]).sort_index()


_XSECTION_BASE_CFG = {"lookback_days": 1, "gross_leverage_target": 1.0, "min_universe_size": 15}


def test_xsection_backtest_restricted_to_window():
    panel = _xsection_panel()
    codes = sorted(panel.index.get_level_values(1).unique())
    fn = build_xsection_backtest_fn(panel, codes, _XSECTION_BASE_CFG)
    all_dates = sorted(panel.index.get_level_values(0).unique())
    start, end = all_dates[100], all_dates[150]
    metrics = fn(start, end, {})
    assert 0 < metrics["n_days"] <= 50


def test_xsection_empty_window_returns_zero_metrics():
    panel = _xsection_panel(n_days=100)
    codes = sorted(panel.index.get_level_values(1).unique())
    fn = build_xsection_backtest_fn(panel, codes, _XSECTION_BASE_CFG)
    metrics = fn(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-06-01"), {})
    assert metrics["n_days"] == 0
    assert metrics["sharpe_ratio"] == 0.0


# ── gates ────────────────────────────────────────────────────────────────────

def _wfo_result_with_folds(oos_metrics_list: list[dict]) -> WFOResult:
    folds = [
        FoldResult(fold_idx=i, is_start="", is_end="", oos_start="", oos_end="",
                   is_sharpe=1.0, oos_sharpe=1.0, oos_pass=True, oos_metrics=m)
        for i, m in enumerate(oos_metrics_list)
    ]
    return WFOResult(folds=folds, total_folds=len(folds), passing_folds=len(folds),
                     pass_ratio=1.0, decision="GO", config={})


def test_drawdown_gate():
    ok = _wfo_result_with_folds([{"max_drawdown": -0.10}, {"max_drawdown": -0.20}])
    bad = _wfo_result_with_folds([{"max_drawdown": -0.10}, {"max_drawdown": -0.40}])
    assert check_drawdown_gate(ok, 0.25) is True
    assert check_drawdown_gate(bad, 0.25) is False


def test_has_trades_gate():
    assert check_has_trades_gate(_wfo_result_with_folds([{"n_trades": 0}, {"n_trades": 3}])) is True
    assert check_has_trades_gate(_wfo_result_with_folds([{"n_trades": 0}, {"n_trades": 0}])) is False


def test_min_trades_gate():
    assert check_min_trades_gate(_wfo_result_with_folds([{"n_trades": 120}, {"n_trades": 150}]), 100) is True
    assert check_min_trades_gate(_wfo_result_with_folds([{"n_trades": 120}, {"n_trades": 50}]), 100) is False
    assert check_min_trades_gate(_wfo_result_with_folds([]), 100) is False


def test_profit_factor_gate():
    assert check_profit_factor_gate(_wfo_result_with_folds([{"profit_factor": 1.5}, {"profit_factor": 2.0}]), 1.3) is True
    assert check_profit_factor_gate(_wfo_result_with_folds([{"profit_factor": 1.5}, {"profit_factor": 0.9}]), 1.3) is False


# ── intraday (microstructure signals) window restriction + stress test ──────

_ORB_BASE_CFG = {"or_minutes": 15, "vwap_side_filter": False}


def _orb_day(start_str: str, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start_str, periods=30, freq="1min")
    bars = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 100_000.0}, index=idx)
    bars.loc[bars.index[5], "high"] = price + 1.0
    bars.loc[bars.index[6], "low"] = price - 1.0
    bars.loc[bars.index[20], "close"] = price
    bars.loc[bars.index[21], "close"] = price + 2.0  # guaranteed OR breakout -> exactly 1 trade/day (eod_flatten)
    return bars


def _multi_day_orb_bars(n_days: int = 5, start_price: float = 100.0) -> pd.DataFrame:
    days = pd.bdate_range("2024-06-03", periods=n_days)
    return pd.concat([_orb_day(f"{d.date()} 09:30", price=start_price) for d in days])


def test_intraday_backtest_restricted_to_window():
    bars = _multi_day_orb_bars(n_days=5)
    fn = build_intraday_backtest_fn({"AAA": bars}, "orb_vwap", _ORB_BASE_CFG)
    days = pd.bdate_range("2024-06-03", periods=5)
    metrics = fn(days[1], days[3], {})  # window covers exactly 2 sessions
    assert metrics["n_trades"] == 2
    assert metrics["n_days"] == 2


def test_intraday_backtest_uses_prior_day_as_warmup_without_leaking_trades():
    bars = _multi_day_orb_bars(n_days=5)
    fn = build_intraday_backtest_fn({"AAA": bars}, "orb_vwap", _ORB_BASE_CFG, warmup_days=1)
    days = pd.bdate_range("2024-06-03", periods=5)
    # Window starts exactly on day 2 (index 2) — day 1 is only warmup context,
    # its trade must NOT be counted even though it's inside the sliced data.
    metrics = fn(days[2], days[4], {})
    assert metrics["n_trades"] == 2


def test_intraday_backtest_empty_window_returns_zero_metrics():
    bars = _multi_day_orb_bars(n_days=3)
    fn = build_intraday_backtest_fn({"AAA": bars}, "orb_vwap", _ORB_BASE_CFG)
    metrics = fn(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-06-01"), {})
    assert metrics["n_trades"] == 0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["daily_returns"] == []


def test_run_intraday_stress_test_increases_cost_vs_normal_run():
    bars = _multi_day_orb_bars(n_days=5)
    days = pd.bdate_range("2024-06-03", periods=5)
    fn = build_intraday_backtest_fn({"AAA": bars}, "orb_vwap", _ORB_BASE_CFG)
    normal = fn(days[0], days[5] if len(days) > 5 else days[-1] + pd.Timedelta(days=1), {})
    stress = run_intraday_stress_test(
        {"AAA": bars}, "orb_vwap", _ORB_BASE_CFG, {},
        days[0], days[-1] + pd.Timedelta(days=1), stress_slippage_multiplier=2.0,
    )
    assert stress["total_net_pnl"] < normal["total_net_pnl"]
