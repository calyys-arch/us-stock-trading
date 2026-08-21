"""
Bridges between the strategies' backtest engines and the generic
WalkForwardOptimizer (walk_forward.py), for the self-improve loop
(scripts/self_improve_loop.py).

walk_forward.py deliberately knows nothing about either engine — it just
calls `backtest_fn(start, end, params) -> {"sharpe_ratio": ...}`. This
module builds those callables:

  - build_pairs_backtest_fn: wraps engine.run_pairs_backtest. THE critical
    subtlety is warmup: run_pairs_backtest consumes the first
    `coint_lookback_days` rows of whatever series it receives as pure
    cointegration-estimation warmup before the first trade can trigger. A
    126-calendar-day OOS window (~86 trading days) fed in naively would
    therefore produce ZERO trades on every OOS fold — the WFO gate would be
    vacuously failing. Fix: slice the input series to include
    `coint_lookback_days` extra rows BEFORE `start`, then compute metrics
    ONLY from trades exited and P&L booked inside [start, end). The warmup
    rows are strictly-historical data at `start`, so this adds no
    look-ahead.

  - build_xsection_backtest_fn: wraps vector_engine.run_vector_backtest.
    The vector engine already evaluates each day from strictly-prior data,
    so "warmup" just means handing it the full panel while restricting
    universe_by_day to dates inside [start, end).

Also here: param-grid expansion/loading (configs/param_grids.yaml),
WFOConfig loading with per-strategy overrides (configs/goal.yaml), the
OOS-drawdown gate, and pre-flight parameter-discipline checks.
"""
from __future__ import annotations

import itertools
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy
from .engine import PairsBacktestConfig, run_pairs_backtest
from .intraday_engine import (
    SIGNAL_PARAM_KEYS,
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)
from .param_guard import (
    MAX_FREE_PARAMETERS,
    check_max_parameters,
    count_free_parameters,
    required_days,
    sufficient_sample_size,
)
from .vector_engine import run_vector_backtest
from .walk_forward import WFOConfig, WFOResult

log = logging.getLogger(__name__)

PARAM_GRIDS_PATH = Path("configs/param_grids.yaml")
GOAL_PATH = Path("configs/goal.yaml")

_CAPITAL = 1_000_000.0  # matches PairsBacktestReport.to_dict's default

# Keys in goal.yaml's wfo block that are WFO knobs (everything else at that
# level is assumed to be a per-strategy override sub-dict).
_WFO_SCALAR_KEYS = {
    "is_days", "oos_days", "step_days",
    "min_pass_folds_ratio", "min_oos_sharpe", "max_oos_drawdown", "max_sharpe_decay",
}


# ── Param grids ──────────────────────────────────────────────────────────────

def expand_param_grid(grid: dict[str, list]) -> list[dict]:
    """{'a': [1, 2], 'b': [x]} -> [{'a': 1, 'b': x}, {'a': 2, 'b': x}]"""
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def load_param_grid(strategy_name: str, path: str | Path = PARAM_GRIDS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        grids = yaml.safe_load(f) or {}
    grid = grids.get(strategy_name)
    if not grid:
        raise KeyError(f"{path} has no grid for strategy '{strategy_name}'")
    return expand_param_grid(grid)


def preflight_check(
    strategy_name: str,
    base_cfg: dict,
    param_grid: list[dict],
    total_trading_days: int | None = None,
) -> dict:
    """Mirrors run_backtest.py's pre-flight checks before a WFO run:

    - check_max_parameters: HARD gate. Refuse to optimize when the grid
      would violate parameter discipline — every gridded key must already
      exist in the strategy's config (grids override, never introduce), and
      the merged config must stay within Chan's MAX_FREE_PARAMETERS ceiling.
      Raises ValueError (this one rejects the run outright).

    - sufficient_sample_size: SOFT gate. When `total_trading_days` is given,
      warns (does not raise) if the window is too short for the number of
      free parameters per Chan's 252-days-per-parameter rule of thumb.
      Returned so the caller can record it in the run's report — a WFO GO
      decision on a too-short window is still evidence, just weaker evidence.
    """
    for candidate in param_grid:
        unknown = set(candidate) - set(base_cfg)
        if unknown:
            raise ValueError(
                f"{strategy_name}: grid introduces keys not in configs/strategy.yaml: {sorted(unknown)}"
            )
        merged = {**base_cfg, **candidate}
        ok, n = check_max_parameters(merged)
        if not ok:
            raise ValueError(
                f"{strategy_name}: merged candidate has {n} free parameters "
                f"(> {MAX_FREE_PARAMETERS} allowed): {candidate}"
            )

    n_free = count_free_parameters(base_cfg)
    result = {"n_free_parameters": n_free, "required_days": required_days(n_free)}
    if total_trading_days is not None:
        sample_ok = sufficient_sample_size(total_trading_days, n_free)
        result["total_trading_days"] = total_trading_days
        result["sample_size_sufficient"] = sample_ok
        if not sample_ok:
            log.warning(
                "%s: only %d trading days in this WFO window for %d free parameters "
                "(Chan's rule of thumb wants >= %d) — treat this run's WFO decision "
                "with extra skepticism.",
                strategy_name, total_trading_days, n_free, result["required_days"],
            )
    return result


# ── WFO config ───────────────────────────────────────────────────────────────

def load_wfo_config(strategy_name: str, goal_path: str | Path = GOAL_PATH) -> WFOConfig:
    """goal.yaml `wfo` block -> WFOConfig, applying the per-strategy override
    sub-dict (e.g. wfo.pairs_trading.is_days) over the shared defaults."""
    with open(goal_path, encoding="utf-8") as f:
        goal = yaml.safe_load(f) or {}
    wfo = dict(goal.get("wfo", {}) or {})
    override = wfo.get(strategy_name)
    settings = {k: v for k, v in wfo.items() if k in _WFO_SCALAR_KEYS}
    if isinstance(override, dict):
        settings.update({k: v for k, v in override.items() if k in _WFO_SCALAR_KEYS})

    cfg = WFOConfig()
    return WFOConfig(
        is_days=int(settings.get("is_days", cfg.is_days)),
        oos_days=int(settings.get("oos_days", cfg.oos_days)),
        step_days=int(settings.get("step_days", cfg.step_days)),
        min_pass_folds_ratio=float(settings.get("min_pass_folds_ratio", cfg.min_pass_folds_ratio)),
        min_oos_sharpe_abs=float(settings.get("min_oos_sharpe", cfg.min_oos_sharpe_abs)),
        max_sharpe_decay=float(settings.get("max_sharpe_decay", cfg.max_sharpe_decay)),
    )


def max_oos_drawdown_threshold(goal_path: str | Path = GOAL_PATH) -> float:
    with open(goal_path, encoding="utf-8") as f:
        goal = yaml.safe_load(f) or {}
    return float((goal.get("wfo", {}) or {}).get("max_oos_drawdown", 0.25))


# ── Metric helpers ───────────────────────────────────────────────────────────

def _metrics_from_returns(returns: pd.Series, n_trades: int, total_net_pnl: float) -> dict:
    sharpe = 0.0
    if len(returns) >= 2 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    max_dd = 0.0
    if len(returns):
        equity = (1.0 + returns).cumprod()
        max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "total_net_pnl": total_net_pnl,
        "n_days": int(len(returns)),
        # Raw daily returns so the self-improve loop can bootstrap a Monte
        # Carlo distribution from a full-window run without re-plumbing the
        # engines. walk_forward.py ignores unknown keys.
        "daily_returns": [float(r) for r in returns.tolist()],
    }


# ── Pairs ────────────────────────────────────────────────────────────────────

def build_pairs_backtest_fn(
    code_a: str,
    code_b: str,
    prices_a: pd.Series,
    prices_b: pd.Series,
    base_cfg: dict,
):
    """Returns backtest_fn(start, end, params) for WalkForwardOptimizer.

    Warmup handling: the slice handed to run_pairs_backtest starts
    `coint_lookback_days` TRADING rows before `start` (when available), but
    all reported metrics come exclusively from daily P&L / trade exits
    inside [start, end) — warmup activity (there should be none anyway,
    since the engine's own loop skips the first coint_lookback_days rows)
    can never leak into fold metrics."""
    df = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna().sort_index()

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        merged = {**base_cfg, **params}
        cfg = PairsBacktestConfig(
            entry_z=merged["entry_z"],
            exit_z=merged["exit_z"],
            coint_lookback_days=merged["coint_lookback_days"],
            revalidate_every_days=merged["revalidate_every_days"],
            notional_per_leg=merged["notional_per_leg"],
            half_life_multiplier_max_hold=merged["half_life_multiplier_max_hold"],
            min_half_life_days=merged["min_half_life_days"],
            max_half_life_days=merged["max_half_life_days"],
        )
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        start_pos = int(np.searchsorted(df.index.to_numpy(), np.datetime64(start_ts), side="left"))
        end_pos = int(np.searchsorted(df.index.to_numpy(), np.datetime64(end_ts), side="left"))
        warmup_pos = max(0, start_pos - cfg.coint_lookback_days)
        window = df.iloc[warmup_pos:end_pos]
        if len(window) <= cfg.coint_lookback_days:
            return _metrics_from_returns(pd.Series(dtype=float), 0, 0.0)

        report = run_pairs_backtest(code_a, code_b, window["a"], window["b"], cfg)

        in_window_pnl = {d: p for d, p in report.daily_pnl.items() if start_ts <= pd.Timestamp(d) < end_ts}
        idx = pd.DatetimeIndex(sorted(in_window_pnl.keys()))
        returns = pd.Series([in_window_pnl[d] / _CAPITAL for d in idx], index=idx)
        trades = [t for t in report.trades if start_ts <= pd.Timestamp(t.exit_date) < end_ts]
        return _metrics_from_returns(returns, len(trades), float(sum(t.net_pnl for t in trades)))

    return backtest_fn


# ── Pairs, scanned universe ─────────────────────────────────────────────────

def build_pairs_scan_backtest_fn(
    close_panel: pd.DataFrame,
    adv_panel: pd.DataFrame,
    scan_schedule: dict,
    base_cfg: dict,
    half_spread_bps: float | None = None,
    max_concurrent_pairs: int | None = None,
    exit_rules: dict | None = None,
    entry_gate: pd.Series | None = None,
):
    """Returns backtest_fn(start, end, params) for WalkForwardOptimizer, over
    the SCANNED pair universe (python/backtest/pairs_scan_engine.py).

    `entry_gate` (optional, default None = unchanged behavior): forwarded
    unmodified to `pairs_scan_engine.run_scan_backtest` — see that function's
    docstring. Used by `backtests/reports/regime_gate_report.md` Phase 2 to
    apply `python/analytics/trend_efficiency_gate.shifted_entry_gate` as a
    point-in-time entry filter; every existing caller that omits this
    argument gets byte-identical behavior to before this parameter existed.

    Warmup handling is identical to `build_pairs_backtest_fn`: the slice
    handed to the engine starts `coint_lookback_days` trading rows before
    `start`, but reported metrics come exclusively from P&L booked and trades
    exited inside [start, end). Those warmup rows are strictly historical at
    `start`, so this adds no look-ahead.

    `scan_schedule` is precomputed ONCE over the full price history by
    `pairs_scan_engine.build_scan_schedule` and shared by every fold. That is
    both the performance story (one CADF fit per pair per scan date for the
    whole study, not per fold per candidate) and a correctness one: the scan
    cadence is anchored to the full index, so no fold can shift it.

    `exit_rules` switches on the opt-in exit ablations (`dynamic_half_life`,
    `coint_breakdown_exit`, `stop_z_multiple`, `stale_timeout_enabled` — see
    `PairsScanConfig`). It is applied identically to every fold and every
    parameter candidate, i.e. it defines the STRATEGY VARIANT under test and
    is never something the walk-forward search can select. Omit it and the
    behavior is unchanged.
    """
    from .pairs_scan_engine import PairsScanConfig, run_scan_backtest

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        merged = {**base_cfg, **params}
        cfg = PairsScanConfig(
            entry_z=merged["entry_z"],
            exit_z=merged["exit_z"],
            coint_lookback_days=merged["coint_lookback_days"],
            revalidate_every_days=merged["revalidate_every_days"],
            notional_per_leg=merged["notional_per_leg"],
            half_life_multiplier_max_hold=merged["half_life_multiplier_max_hold"],
            min_half_life_days=merged["min_half_life_days"],
            max_half_life_days=merged["max_half_life_days"],
        )
        if half_spread_bps is not None:
            cfg.half_spread_bps = float(half_spread_bps)
        if max_concurrent_pairs is not None:
            cfg.max_concurrent_pairs = int(max_concurrent_pairs)
        for key, value in (exit_rules or {}).items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown exit rule {key!r}")
            setattr(cfg, key, value)

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        idx = close_panel.index.to_numpy()
        start_pos = int(np.searchsorted(idx, np.datetime64(start_ts), side="left"))
        end_pos = int(np.searchsorted(idx, np.datetime64(end_ts), side="left"))
        warmup_pos = max(0, start_pos - cfg.coint_lookback_days)
        window_close = close_panel.iloc[warmup_pos:end_pos]
        if len(window_close) <= cfg.coint_lookback_days:
            return _pairs_scan_metrics(pd.Series(dtype=float), [], 0)
        window_adv = adv_panel.iloc[warmup_pos:end_pos]

        report = run_scan_backtest(window_close, window_adv, scan_schedule, cfg, entry_gate=entry_gate)

        in_window_pnl = {d: p for d, p in report.daily_pnl.items()
                         if start_ts <= pd.Timestamp(d) < end_ts}
        pnl_idx = pd.DatetimeIndex(sorted(in_window_pnl.keys()))
        returns = pd.Series([in_window_pnl[d] / _CAPITAL for d in pnl_idx], index=pnl_idx)
        trades = [t for t in report.trades if start_ts <= pd.Timestamp(t.exit_date) < end_ts]
        return _pairs_scan_metrics(returns, trades, report.open_at_end)

    return backtest_fn


def _profit_factor(values) -> float:
    wins = sum(v for v in values if v > 0)
    losses = abs(sum(v for v in values if v < 0))
    if losses > 0:
        return float(wins / losses)
    return float("inf") if wins > 0 else 0.0


def _pairs_scan_metrics(returns: pd.Series, trades: list, open_at_end: int = 0) -> dict:
    """`_metrics_from_returns` plus the pairs-scan-specific diagnostics the
    report needs PER FOLD rather than only in aggregate.

    `profit_factor` is cost-adjusted (net_pnl is already after commission,
    fees, borrow, impact and half-spread); `profit_factor_gross` is the same
    quantity before any cost, which is what separates "the edge is real but
    too small to pay for itself" from "there is no edge". `exit_reasons` is
    here because the exit-rule ablations are judged on it directly, and
    `open_at_end` because a rule that simply holds losers longer would
    otherwise look better by never booking them.
    """
    metrics = _metrics_from_returns(
        returns, len(trades), float(sum(t.net_pnl for t in trades)))
    metrics["profit_factor"] = _profit_factor([t.net_pnl for t in trades])
    metrics["profit_factor_gross"] = _profit_factor([t.gross_pnl for t in trades])
    metrics["gross_pnl"] = float(sum(t.gross_pnl for t in trades))
    metrics["total_cost"] = float(sum(t.cost for t in trades))
    metrics["n_distinct_pairs"] = len({(t.code_a, t.code_b) for t in trades})
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    metrics["exit_reasons"] = dict(sorted(reasons.items()))
    metrics["open_at_end"] = int(open_at_end)
    return metrics


# ── Cross-sectional ──────────────────────────────────────────────────────────

def build_xsection_backtest_fn(
    panel: pd.DataFrame,
    universe_symbols: list[str],
    base_cfg: dict,
    skip_first_days: int = 30,
):
    """Returns backtest_fn(start, end, params) for WalkForwardOptimizer.
    `panel` is the FULL-range (date, code) OHLCV panel; each call restricts
    universe_by_day to dates inside [start, end) while the engine still sees
    the full panel for its strictly-before-as_of evaluation (the fixed
    universe list itself never varies by day — configs/universe.yaml).
    `skip_first_days` drops the panel's earliest dates from any window,
    matching run_backtest.py's convention of never trading before the
    strategy has a minimal lookback history."""
    all_dates = sorted(panel.index.get_level_values(0).unique())
    tradeable_dates = all_dates[skip_first_days:]

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        merged = {**base_cfg, **params}
        strategy = CrossSectionalMeanReversionStrategy(
            lookback_days=merged["lookback_days"],
            gross_leverage_target=merged["gross_leverage_target"],
            min_universe_size=merged["min_universe_size"],
        )
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window_dates = [d for d in tradeable_dates if start_ts <= pd.Timestamp(d) < end_ts]
        if not window_dates:
            return _metrics_from_returns(pd.Series(dtype=float), 0, 0.0)

        universe_by_day = {d: list(universe_symbols) for d in window_dates}
        result = run_vector_backtest(strategy, panel, universe_by_day, capital=_CAPITAL)
        returns = result.daily_returns
        n_active = int((returns != 0).sum())
        return _metrics_from_returns(returns, n_active, float(returns.sum() * _CAPITAL))

    return backtest_fn


# ── Intraday microstructure signals ─────────────────────────────────────────

def build_intraday_backtest_fn(
    bars_by_symbol: dict[str, pd.DataFrame],
    signal_name: str,
    base_cfg: dict,
    engine_cfg: IntradayBacktestConfig | None = None,
    warmup_days: int = 1,
):
    """Returns backtest_fn(start, end, params) for WalkForwardOptimizer,
    wrapping python/backtest/intraday_engine.run_intraday_backtest.

    Warmup handling: each symbol's slice starts `warmup_days` calendar days
    before `start` so the FIRST session inside [start, end) still gets a
    real `prior_day_bars` for YDH/YDL (sweep_reclaim) instead of None —
    same principle as build_pairs_backtest_fn's coint_lookback warmup, just
    a single day instead of a whole lookback window since intraday
    liquidity levels only look back one session. Trades are then filtered
    to those that EXITED inside [start, end) before computing metrics, so
    warmup-day activity (there should be very little — the warmup day
    itself is a normal tradeable session, just outside the reported
    window) never contributes to the fold's numbers."""
    engine_cfg = engine_cfg or IntradayBacktestConfig()
    sig_keys = SIGNAL_PARAM_KEYS[signal_name]

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        merged = {**base_cfg, **params}
        sig_params = {k: merged[k] for k in sig_keys if k in merged}

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        warmup_start = start_ts - pd.Timedelta(days=warmup_days)
        sliced = {}
        for symbol, bars in bars_by_symbol.items():
            window = bars.loc[(bars.index >= warmup_start) & (bars.index < end_ts)]
            if not window.empty:
                sliced[symbol] = window

        if not sliced:
            return _metrics_from_returns(pd.Series(dtype=float), 0, 0.0)

        report = run_intraday_backtest(sliced, signal_name, sig_params, engine_cfg)
        in_window = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
        filtered = IntradayBacktestReport(
            trades=in_window, signals_emitted=report.signals_emitted, signals_filled=report.signals_filled,
        )
        return metrics_from_report(filtered, engine_cfg.capital)

    return backtest_fn


def run_intraday_stress_test(
    bars_by_symbol: dict[str, pd.DataFrame],
    signal_name: str,
    base_cfg: dict,
    params: dict,
    start: datetime,
    end: datetime,
    stress_slippage_multiplier: float = 2.0,
    warmup_days: int = 1,
    half_spread_bps_by_symbol: dict[str, float] | None = None,
    base_engine_cfg: IntradayBacktestConfig | None = None,
) -> dict:
    """Re-runs the SAME window/params at `stress_slippage_multiplier`x the
    normal slippage cost (configs/goal.yaml intraday.stress_slippage_multiplier)
    — docs/microstructure_pivot_plan.md §4c's mandatory 2x-slippage stress
    test. A signal that only survives at 1x cost is not survivable; this is
    a hard, separate re-run rather than a parameter in param_grids.yaml so
    it can never be accidentally "optimized away" by the grid search.
    `warmup_days` must match whatever was used to build the signal's main
    backtest_fn (see build_intraday_backtest_fn's docstring) — passing a
    mismatched value would not be incorrect for causality (warmup is always
    >= the minimum needed), but WOULD make a signal that structurally needs
    more warmup silently perform worse under stress than under normal
    conditions for a data-availability reason having nothing to do with
    slippage, which would misattribute the stress test's result.
    `half_spread_bps_by_symbol` (default None -> unchanged flat-cost
    behavior) is the SAME calibrated-spread override as
    build_intraday_backtest_fn/IntradayBacktestConfig — the 2x multiplier
    still applies on top of whichever half-spread (flat or per-symbol) is
    in effect, since `_slippage_price` applies `stress_slippage_multiplier`
    to the resolved half_spread_bps+impact total, not to the flat constant
    specifically.
    `base_engine_cfg` copies structural knobs (chart_minutes,
    time_stop_minutes, signal_filter_overrides) so a 1m/15m WFO is
    stressed on the same decision chart, not silently reset to 5m."""
    base = base_engine_cfg or IntradayBacktestConfig()
    spreads = (
        half_spread_bps_by_symbol
        if half_spread_bps_by_symbol is not None
        else base.half_spread_bps_by_symbol
    )
    stress_cfg = IntradayBacktestConfig(
        stress_slippage_multiplier=stress_slippage_multiplier,
        half_spread_bps_by_symbol=spreads,
        chart_minutes=base.chart_minutes,
        time_stop_minutes=base.time_stop_minutes,
        signal_filter_overrides=dict(base.signal_filter_overrides),
    )
    fn = build_intraday_backtest_fn(bars_by_symbol, signal_name, base_cfg, engine_cfg=stress_cfg, warmup_days=warmup_days)
    return fn(start, end, params)


# ── Gates ────────────────────────────────────────────────────────────────────

def check_min_trades_gate(wfo_result: WFOResult, min_trades: int) -> bool:
    """Every OOS fold must clear the minimum fill count — a fold with too
    few trades is not statistically meaningful even if its Sharpe happens
    to look good (configs/goal.yaml intraday.min_trades_per_oos_fold)."""
    if not wfo_result.folds:
        return False
    return all(int(fold.oos_metrics.get("n_trades", 0)) >= min_trades for fold in wfo_result.folds)


def check_pooled_trades_gate(wfo_result: WFOResult, min_total: int) -> bool:
    """Pooled OOS fills across folds. Sparse signals (Creamer/VSA) cannot
    honestly clear 100 fills in every 30-day fold; the sample that matters
    is the walk-forward OOS total."""
    if not wfo_result.folds:
        return False
    total = sum(int(fold.oos_metrics.get("n_trades", 0)) for fold in wfo_result.folds)
    return total >= min_total


def check_profit_factor_gate(wfo_result: WFOResult, min_profit_factor: float) -> bool:
    """Every OOS fold's cost-adjusted profit factor (gross profit / gross
    loss, already net of slippage + commission) must clear the ceiling
    (configs/goal.yaml intraday.min_cost_adjusted_profit_factor)."""
    if not wfo_result.folds:
        return False
    return all(float(fold.oos_metrics.get("profit_factor", 0.0)) >= min_profit_factor for fold in wfo_result.folds)


def check_pooled_profit_factor_gate(wfo_result: WFOResult, min_profit_factor: float) -> bool:
    """Trade-weighted mean of OOS fold profit factors. One quiet losing
    month must not veto a sparse signal whose pooled OOS still covers
    costs."""
    if not wfo_result.folds:
        return False
    weighted = 0.0
    n = 0
    for fold in wfo_result.folds:
        trades = int(fold.oos_metrics.get("n_trades", 0))
        if trades <= 0:
            continue
        weighted += float(fold.oos_metrics.get("profit_factor", 0.0)) * trades
        n += trades
    if n <= 0:
        return False
    return (weighted / n) >= min_profit_factor


def check_drawdown_gate(wfo_result: WFOResult, max_oos_drawdown: float) -> bool:
    """Every OOS fold's |max_drawdown| must stay within the goal.yaml
    ceiling. Uses fold oos_metrics recorded by walk_forward.py."""
    for fold in wfo_result.folds:
        dd = abs(float(fold.oos_metrics.get("max_drawdown", 0.0)))
        if dd > max_oos_drawdown:
            return False
    return True


def check_has_trades_gate(wfo_result: WFOResult) -> bool:
    """At least one OOS fold must have actually traded — a WFO 'pass' made
    of all-zero return series is vacuous, not evidence."""
    return any(int(fold.oos_metrics.get("n_trades", 0)) > 0 for fold in wfo_result.folds)
