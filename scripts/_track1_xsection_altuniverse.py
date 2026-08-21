"""
Track 1 (backtests/reports/alt_universe_frequency_exploration.md): does
`xsection_mean_reversion` (Chan's 1-day cross-sectional reversal, UNMODIFIED
strategy code) retain more edge than it costs on a deliberately
LESS-liquid, wider-spread universe than `configs/universe.yaml`'s fixed
mega-cap top-20?

Universe: `configs/alt_universe_midcap.yaml` (built by
scripts/_altuni_build_universe.py — point-in-time S&P 500 membership as of
2016-11-01 via a historical Wikipedia revision, liquidity BAND ranks
150-220 by trailing 60-day dollar volume, i.e. one tier below the mega-cap
names this repo already tested and found the reversal edge arbitraged
away in). 71 symbols; this script fetches their full daily OHLCV history
and drops any symbol without a sufficiently long/complete series (mergers,
delistings since 2016 — real point-in-time attrition, disclosed rather
than backfilled).

Cost model — the wide-spread assumption, derived rather than guessed:
`backtests/reports/slippage_calibration_report.md` calibrated real
half-spreads for the CURRENT 20-symbol mega-cap universe: 0.32bps (AAPL,
~$17B/day 2026 ADV) up to 6.59bps (STX, ~$4.1B/day 2026 ADV) — a decade of
well-documented spread-vs-liquidity literature says spread widens as ADV
shrinks, and this codebase's own 20 calibrated points already show that
relationship (log-log OLS slope -0.67, i.e. roughly an inverse-square-root
relationship, correlation -0.56 — noisy but directionally unambiguous).
The alt universe's ADV band ($109M-$159M/day, priced at ITS OWN 2016
as-of date so this is not an apples-to-apples current-dollar comparison,
but it does not need to be: the band was chosen SPECIFICALLY to sit
one-to-two orders of magnitude below every calibrated point, so any
reasonable extrapolation lands far wider than 6.59bps) extrapolates that
fit to ~28-36bps — and per this task's explicit "if in doubt, assume costs
are WORSE" instruction, and because a log-log fit with correlation 0.56 is
NOT precise enough to trust its point estimate at 25-1000x outside its
calibrated range, the number actually used is rounded UP and padded to a
clean **40bps** base assumption (not fitted, chosen to be conservative
relative to the fit), stress-tested at **80bps** (2x, matching this
repo's standard stress convention). For comparison: this is 6-130x wider
than every calibrated mega-cap spread and >13x the 3.0bps ETF assumption
`pairs_scan_report.md` used — deliberately erring toward "too expensive"
rather than risk repeating the flat-cost optimism `strategy_review_summary.md`
diagnosed as part of the original mega-cap-universe failures.

Order-size sanity check (the "would this order move the market" half of
the task's tradeability requirement): `xsection_mean_reversion`'s Chan
eq. 3.7 weights are dollar-neutral and roughly evenly split across the
eligible universe each day — on $1M capital across ~50-70 eligible names
that is single-digit-thousands of dollars per name, versus $109M+/day ADV
— under 0.1% of ADV per name, well inside "does not move the market."

Uses `python/core/strategies/xsection_mean_reversion.py` and
`python/backtest/vector_engine.py` COMPLETELY UNMODIFIED in their
strategy logic — only vector_engine.py gained the new OPTIONAL
`half_spread_bps` parameter (see that file's own docstring; `None` default
reproduces every prior caller byte-for-byte, pinned by the pre-existing
test suite passing unchanged).

Usage:
    python scripts/_track1_xsection_altuniverse.py --phase fetch    # resumable, slow (yfinance)
    python scripts/_track1_xsection_altuniverse.py --phase dev
    python scripts/_track1_xsection_altuniverse.py --phase holdout  # refuses to re-run
    python scripts/_track1_xsection_altuniverse.py --phase report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
log = logging.getLogger("track1_xsection_altuniverse")

import numpy as np
import pandas as pd
import yaml

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import expand_param_grid, load_wfo_config, max_oos_drawdown_threshold
from python.backtest.vector_engine import run_vector_backtest
from python.backtest.walk_forward import WalkForwardOptimizer
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy
from python.data.price_cache import get_cached_price_panel

UNIVERSE_YAML = Path("configs/alt_universe_midcap.yaml")
CACHE_DIR = "data/history_altuni_midcap_full"
BASE_HALF_SPREAD_BPS = 40.0
STRESS_HALF_SPREAD_MULTIPLIER = 2.0
FETCH_START = "2016-11-01"     # the universe's own as_of date; no data before it is used
FETCH_END = "2026-08-01"
DEV_START = "2016-11-01"
DEV_END = "2024-01-01"
HOLDOUT_END = "2026-08-01"
CAPITAL = 1_000_000.0
SKIP_FIRST_DAYS = 30

CACHE_JSON = Path("backtests/reports/_track1_xsection_cache")
DEV_CHECKPOINT = CACHE_JSON / "_checkpoint_dev.json"
HOLDOUT_CHECKPOINT = CACHE_JSON / "_checkpoint_holdout.json"
REPORT_JSON = Path("backtests/reports/track1_xsection_altuniverse_report.json")

BASE_CFG = {"lookback_days": 1, "gross_leverage_target": 1.0, "min_universe_size": 15}
PARAM_GRID = expand_param_grid({"lookback_days": [1, 3, 5], "gross_leverage_target": [0.5, 1.0]})


def _load_universe() -> list[str]:
    doc = yaml.safe_load(UNIVERSE_YAML.read_text(encoding="utf-8"))
    return sorted(doc["alt_universe_midcap"]["symbols"])


def phase_fetch(args) -> None:
    symbols = _load_universe()
    log.info("fetching %d symbols [%s, %s) into %s (resumable, slow) ...",
             len(symbols), FETCH_START, FETCH_END, CACHE_DIR)
    panel, quality_flags, meta = get_cached_price_panel(
        symbols, FETCH_START, FETCH_END, cache_dir=CACHE_DIR, refresh=args.refresh_data)
    fetched = sorted(panel.index.get_level_values(1).unique())
    missing = sorted(set(symbols) - set(fetched))
    log.info("fetched %d/%d symbols (source=%s); missing: %s",
             len(fetched), len(symbols), meta.get("fetched_source"), missing)
    log.info("data quality flags on %d symbols", len(quality_flags))


def _load_panel_and_universe() -> tuple[pd.DataFrame, list[str], dict]:
    symbols = _load_universe()
    panel, quality_flags, meta = get_cached_price_panel(
        symbols, FETCH_START, FETCH_END, cache_dir=CACHE_DIR)
    fetched = sorted(panel.index.get_level_values(1).unique())
    too_short = []
    warmup_ts = pd.Timestamp(FETCH_START) + pd.Timedelta(days=15)
    for c in fetched:
        sub = panel.xs(c, level=1)
        if sub.index.min() > warmup_ts:
            too_short.append(c)
    usable = sorted(set(fetched) - set(too_short))
    meta_out = {
        "n_universe_symbols": len(symbols),
        "n_fetched": len(fetched),
        "n_missing": len(symbols) - len(fetched),
        "missing": sorted(set(symbols) - set(fetched)),
        "n_dropped_short_history": len(too_short),
        "dropped_short_history": too_short,
        "n_usable": len(usable),
        "source": meta.get("fetched_source"),
        "n_symbols_with_data_quality_flags": len(quality_flags),
        "first_date": str(panel.index.get_level_values(0).min().date()),
        "last_date": str(panel.index.get_level_values(0).max().date()),
    }
    return panel, usable, meta_out


def _build_backtest_fn(panel: pd.DataFrame, universe_symbols: list[str], half_spread_bps: float):
    all_dates = sorted(panel.index.get_level_values(0).unique())
    tradeable_dates = all_dates[SKIP_FIRST_DAYS:]

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        merged = {**BASE_CFG, **params}
        strategy = CrossSectionalMeanReversionStrategy(
            lookback_days=merged["lookback_days"],
            gross_leverage_target=merged["gross_leverage_target"],
            min_universe_size=merged["min_universe_size"],
        )
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window_dates = [d for d in tradeable_dates if start_ts <= pd.Timestamp(d) < end_ts]
        if not window_dates:
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "n_trades": 0, "total_net_pnl": 0.0,
                    "n_days": 0, "daily_returns": []}
        universe_by_day = {d: list(universe_symbols) for d in window_dates}
        result = run_vector_backtest(strategy, panel, universe_by_day, capital=CAPITAL,
                                      half_spread_bps=half_spread_bps)
        returns = result.daily_returns
        n_active = int((returns != 0).sum())
        gross = float(result.daily_gross_returns.sum() * CAPITAL)
        cost = float(result.daily_costs.sum() * CAPITAL)
        sharpe = 0.0
        if len(returns) >= 2 and returns.std(ddof=1) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        equity = (1.0 + returns).cumprod()
        max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
        return {
            "sharpe_ratio": sharpe, "max_drawdown": max_dd, "n_trades": n_active,
            "total_net_pnl": float(returns.sum() * CAPITAL), "n_days": int(len(returns)),
            "gross_pnl": gross, "total_cost": cost,
            "daily_returns": [float(r) for r in returns.tolist()],
        }

    return backtest_fn


def _strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "daily_returns"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def phase_dev(args) -> dict:
    panel, universe_symbols, meta = _load_panel_and_universe()
    log.info("dev data: %s", meta)
    wfo_cfg = load_wfo_config("xsection_mean_reversion")
    fn = _build_backtest_fn(panel, universe_symbols, BASE_HALF_SPREAD_BPS)

    start_ts, end_ts = pd.Timestamp(DEV_START), pd.Timestamp(DEV_END)
    n_days = int(((panel.index.get_level_values(0).unique() >= start_ts) &
                   (panel.index.get_level_values(0).unique() < end_ts)).sum())
    log.info("dev: WFO over [%s, %s), %d candidates, %d trading days, half_spread=%.1fbps",
             start_ts.date(), end_ts.date(), len(PARAM_GRID), n_days, BASE_HALF_SPREAD_BPS)

    candidate_wfo = WalkForwardOptimizer(fn, wfo_cfg, PARAM_GRID).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())
    candidate_wfo.print_summary()

    if not candidate_wfo.folds:
        raise SystemExit("dev window too short for a single WFO fold")

    final_params = dict(candidate_wfo.folds[-1].best_params)
    full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)
    mc = MonteCarloValidator(n_sims=1000, seed=42).run(full.get("daily_returns", []))

    stress_fn = _build_backtest_fn(panel, universe_symbols, BASE_HALF_SPREAD_BPS * STRESS_HALF_SPREAD_MULTIPLIER)
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)

    zero_cost_fn = _build_backtest_fn(panel, universe_symbols, 0.0)
    zero_cost = zero_cost_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd = max_oos_drawdown_threshold()
    cost_ratio = (full["gross_pnl"] / full["total_cost"]) if full.get("total_cost", 0) > 0 else float("inf")
    gates = {
        "wfo_go": candidate_wfo.decision == "GO",
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        "full_window_drawdown_within_limit": abs(full["max_drawdown"]) <= max_dd,
        "cost_gate_gross_to_cost_ratio_ge_2": cost_ratio >= 2.0,
    }

    result = {
        "phase": "dev", "strategy": "xsection_mean_reversion", "universe": "alt_universe_midcap",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "data": meta,
        "half_spread_bps": BASE_HALF_SPREAD_BPS,
        "param_grid": PARAM_GRID,
        "final_params": final_params,
        "candidate_wfo": candidate_wfo.to_dict(),
        "full_window": _strip(full),
        "full_window_cost_gate_ratio": cost_ratio,
        "stress_2x_spread": _strip(stress),
        "zero_cost_control": _strip(zero_cost),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    _write_json(DEV_CHECKPOINT, result)
    return result


def phase_holdout(args) -> dict:
    if not DEV_CHECKPOINT.exists():
        raise SystemExit(f"missing {DEV_CHECKPOINT} — run --phase dev first")
    if HOLDOUT_CHECKPOINT.exists():
        raise SystemExit(f"{HOLDOUT_CHECKPOINT} exists — holdout evaluated exactly once already")
    dev = json.loads(DEV_CHECKPOINT.read_text(encoding="utf-8"))
    final_params = dev["final_params"]

    panel, universe_symbols, meta = _load_panel_and_universe()
    fn = _build_backtest_fn(panel, universe_symbols, BASE_HALF_SPREAD_BPS)
    start_ts, end_ts = pd.Timestamp(DEV_END), pd.Timestamp(HOLDOUT_END)
    log.info("holdout: single run of %s over [%s, %s)", final_params, start_ts.date(), end_ts.date())
    metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)
    mc = MonteCarloValidator(n_sims=1000, seed=42).run(metrics.get("daily_returns", []))

    stress_fn = _build_backtest_fn(panel, universe_symbols, BASE_HALF_SPREAD_BPS * STRESS_HALF_SPREAD_MULTIPLIER)
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd = max_oos_drawdown_threshold()
    cost_ratio = (metrics["gross_pnl"] / metrics["total_cost"]) if metrics.get("total_cost", 0) > 0 else float("inf")
    gates = {
        "has_trades": metrics["n_trades"] > 0,
        "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= min_oos_sharpe,
        "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_dd,
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        "cost_gate_gross_to_cost_ratio_ge_2": cost_ratio >= 2.0,
    }

    result = {
        "phase": "holdout", "strategy": "xsection_mean_reversion", "universe": "alt_universe_midcap",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "params": final_params,
        "half_spread_bps": BASE_HALF_SPREAD_BPS,
        "metrics": _strip(metrics),
        "cost_gate_ratio": cost_ratio,
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    _write_json(HOLDOUT_CHECKPOINT, result)
    return result


def phase_report(args) -> dict:
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "strategy": "xsection_mean_reversion", "universe": "alt_universe_midcap",
        "dev": json.loads(DEV_CHECKPOINT.read_text(encoding="utf-8")) if DEV_CHECKPOINT.exists() else None,
        "holdout": json.loads(HOLDOUT_CHECKPOINT.read_text(encoding="utf-8")) if HOLDOUT_CHECKPOINT.exists() else None,
    }
    _write_json(REPORT_JSON, out)
    log.info("wrote %s", REPORT_JSON)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["fetch", "dev", "holdout", "report", "all"], default="all")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    phases = ["fetch", "dev", "holdout", "report"] if args.phase == "all" else [args.phase]
    for phase in phases:
        log.info("=== phase: %s ===", phase)
        {"fetch": phase_fetch, "dev": phase_dev, "holdout": phase_holdout, "report": phase_report}[phase](args)
    log.info("done")


if __name__ == "__main__":
    main()
