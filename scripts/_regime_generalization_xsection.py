"""
Regime-generalization test — `xsection_mean_reversion`'s CURRENT PRODUCTION
CONFIG (`configs/strategy.yaml`), run UNMODIFIED against 2020 (COVID crash)
and 2022 (rate-hike bear), plus an optional calmer 2018-2019 contrast window.

THIS IS NOT A NEW TUNING ROUND. The strategy code
(`python/core/strategies/xsection_mean_reversion.py`), the engine
(`python/backtest/vector_engine.py`), and the config
(`lookback_days: 1, gross_leverage_target: 1.0, min_universe_size: 15`) are
untouched. The universe is the SAME fixed top-20 mega-cap snapshot
(`configs/universe.yaml`, computed_at 2026-07-28) already used, unmodified,
for every prior 2018-2025/2024-2026 evaluation of this strategy
(`self_improvement_log.md`, `strategy_review_summary.md` §2.2) — i.e. the
SAME survivorship-flavored "one snapshot applied backward" approximation
this repo has always used for this strategy, not a new point-in-time
reconstruction. See the report for exactly which symbols in that snapshot
did not exist yet (or existed under a different ticker) during 2020/2022 and
how that is handled (the engine simply has no bars for a not-yet-listed
name; `min_universe_size: 15` already tolerates missing names).

Primary cost model, kept IDENTICAL to the one that produced the existing
-0.492 baseline OOS Sharpe (`strategy_review_summary.md` §2.2) and every
other number this strategy has ever been judged against: commission +
square-root market impact only, `half_spread_bps=None` (vector_engine's
pre-existing default, unchanged). A SEPARATE, clearly-labeled stress variant
additionally charges this same universe's Aug-2026-calibrated half-spreads
(`slippage_calibration_report.md`) — included per this task's cost-assumption
instruction, with the explicit caveat (restated in the report) that 2020's
realized volatility means true intraday spreads were almost certainly WIDER
than an Aug-2026 calm-period calibration.

Usage:
    python scripts/_regime_generalization_xsection.py
    python scripts/_regime_generalization_xsection.py --window 2022_bear
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
log = logging.getLogger("regime_generalization_xsection")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import max_oos_drawdown_threshold
from python.backtest.vector_engine import run_vector_backtest
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy
from python.data.price_cache import get_cached_price_panel

CACHE_DIR = Path("backtests/reports/_regime_generalization_cache")
REPORT_JSON = Path("backtests/reports/regime_generalization_xsection.json")
STRATEGY = "xsection_mean_reversion"
_CAPITAL = 1_000_000.0
SKIP_FIRST_DAYS = 30  # matches build_xsection_backtest_fn's convention

# Calibrated Aug-2026 half-spreads, `median_bps` verbatim from
# `backtests/reports/slippage_calibration_report.json`'s `calibrated_spreads`
# (2 captured trading days, 20260804/20260806), for the stress variant ONLY.
# Names in this universe not present in the calibration table fall back to
# 0.0 in run_vector_backtest's dict-lookup convention (none are missing here
# — all 20 fixed_universe symbols were captured).
CALIBRATED_HALF_SPREAD_BPS = {
    "AAPL": 0.3232, "AMAT": 3.7419, "AMD": 1.8302, "AVGO": 1.3071,
    "GOOGL": 0.4188, "INTC": 0.9953, "LITE": 5.7478, "LRCX": 4.1930,
    "META": 1.2888, "MRVL": 3.0477, "MSFT": 0.7074, "MU": 2.0632,
    "NBIS": 4.6270, "NVDA": 0.4551, "ORCL": 2.0764, "PLTR": 0.9388,
    "QCOM": 1.8894, "SNDK": 3.3442, "STX": 6.5875, "WDC": 4.4261,
}

WINDOWS: dict[str, tuple[str, str]] = {
    "2018_2019_calmer": ("2018-01-01", "2020-01-01"),
    "2020_covid": ("2020-01-01", "2021-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
}

FETCH_START = "2017-11-01"
FETCH_END = "2022-12-31"


def _strategy_cfg() -> dict:
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)[STRATEGY]


def _universe_symbols() -> list[str]:
    with open("configs/universe.yaml", encoding="utf-8") as f:
        return list(yaml.safe_load(f)["fixed_universe"]["symbols"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("checkpoint written: %s", path)


# Empirically confirmed (this task, 2026-08-14) via a direct yfinance fetch
# attempt: these two `fixed_universe` symbols have ZERO price history before
# their real-world listing events and yfinance errors out ("possibly
# delisted; no price data found") rather than returning an empty series, so
# they must be excluded from the fetch call itself (not just tolerated as
# missing bars) or `build_price_panel` raises for the whole batch. NBIS
# (Nebius Group) only began trading under this ticker in 2024; SNDK
# (SanDisk) was spun off from WDC in 2025. Neither existed as a tradeable
# equity at any point in 2017-2022. This is disclosed in the report as an
# approximation inherent to applying today's fixed_universe.yaml snapshot
# backward — the SAME caveat this strategy's universe has always carried
# (fixed_universe.py's own module docstring), just concretely enumerated
# here for the first time against a pre-2023 window.
_SYMBOLS_NOT_YET_LISTED_IN_TEST_RANGE = {"NBIS", "SNDK"}


def _load_panel() -> tuple[pd.DataFrame, dict]:
    symbols = _universe_symbols()
    fetchable = [s for s in symbols if s.upper() not in _SYMBOLS_NOT_YET_LISTED_IN_TEST_RANGE]
    panel, quality_flags, meta = get_cached_price_panel(fetchable, FETCH_START, FETCH_END)
    present = sorted(panel.index.get_level_values(1).unique())
    missing = sorted(set(s.upper() for s in symbols) - set(present))
    coverage = {}
    for s in symbols:
        try:
            sub = panel.xs(s.upper(), level=1)
            coverage[s] = {"first": str(sub.index.min().date()), "last": str(sub.index.max().date()),
                          "n_bars": int(len(sub))}
        except KeyError:
            coverage[s] = None
    return panel, {
        "universe_symbols": symbols,
        "n_universe_symbols": len(symbols),
        "symbols_missing_from_fetch_entirely": missing,
        "per_symbol_coverage": coverage,
        "sources": meta["sources"],
        "fetch_range": [FETCH_START, FETCH_END],
    }


def _metrics(result, capital=_CAPITAL) -> dict:
    net = result.daily_returns
    gross = result.daily_gross_returns
    cost = result.daily_costs
    sharpe = 0.0
    if len(net) >= 2 and net.std(ddof=1) > 0:
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
    equity = (1.0 + net).cumprod() if len(net) else pd.Series(dtype=float)
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    n_active = int((net != 0).sum())
    gross_pnl_dollars = (gross * capital)
    cost_dollars = (cost * capital)
    net_pnl_dollars = (net * capital)

    def _pf(series: pd.Series) -> float:
        wins = float(series[series > 0].sum())
        losses = float(-series[series < 0].sum())
        if losses > 0:
            return wins / losses
        return float("inf") if wins > 0 else 0.0

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_active_days": n_active,
        "n_days": int(len(net)),
        "total_net_pnl": float(net_pnl_dollars.sum()),
        "total_gross_pnl": float(gross_pnl_dollars.sum()),
        "total_cost": float(cost_dollars.sum()),
        "gross_to_cost_ratio": (float(gross_pnl_dollars.sum()) / float(cost_dollars.sum())
                                if cost_dollars.sum() != 0 else float("inf")),
        "profit_factor_net_daily": _pf(net_pnl_dollars),
        "profit_factor_gross_daily": _pf(gross_pnl_dollars),
        "daily_returns": net.tolist(),
    }


def _run_one(window_name: str, start: str, end: str, panel: pd.DataFrame,
             symbols: list[str], base_cfg: dict) -> dict:
    strategy = CrossSectionalMeanReversionStrategy(
        lookback_days=base_cfg["lookback_days"],
        gross_leverage_target=base_cfg["gross_leverage_target"],
        min_universe_size=base_cfg["min_universe_size"],
    )
    all_dates = sorted(panel.index.get_level_values(0).unique())
    tradeable_dates = all_dates[SKIP_FIRST_DAYS:]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    window_dates = [d for d in tradeable_dates if start_ts <= pd.Timestamp(d) < end_ts]
    universe_by_day = {d: list(symbols) for d in window_dates}

    result_primary = run_vector_backtest(strategy, panel, universe_by_day, capital=_CAPITAL,
                                          half_spread_bps=None)
    result_stress = run_vector_backtest(strategy, panel, universe_by_day, capital=_CAPITAL,
                                         half_spread_bps=CALIBRATED_HALF_SPREAD_BPS)

    metrics = _metrics(result_primary)
    metrics_stress = _metrics(result_stress)
    daily_returns = metrics.pop("daily_returns")
    metrics_stress.pop("daily_returns")
    mc = MonteCarloValidator(n_sims=500).run(daily_returns)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd = max_oos_drawdown_threshold()
    min_cost_ratio = float(goal["cost_gate"]["min_gross_to_cost_ratio"])
    gates = {
        "has_active_days": metrics["n_active_days"] > 0,
        "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= min_oos_sharpe,
        "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_dd,
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        "cost_gate_gross_to_cost_ratio": metrics["gross_to_cost_ratio"] >= min_cost_ratio,
    }
    return {
        "config": "xsection_mean_reversion_production",
        "config_source": "configs/strategy.yaml (unmodified) + configs/universe.yaml (unmodified)",
        "params": base_cfg,
        "window_name": window_name,
        "window": [str(start_ts.date()), str(end_ts.date())],
        "n_window_dates_available": len(window_dates),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "metrics_primary_no_spread": metrics,
        "metrics_stress_calibrated_spread": metrics_stress,
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", choices=list(WINDOWS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    panel, panel_meta = _load_panel()
    base_cfg = _strategy_cfg()
    symbols = _universe_symbols()
    log.info("panel loaded: %s", {k: v for k, v in panel_meta.items() if k != "per_symbol_coverage"})

    windows = {args.window: WINDOWS[args.window]} if args.window else WINDOWS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for window_name, (start, end) in windows.items():
        ckpt = CACHE_DIR / f"xsection_production_{window_name}.json"
        if ckpt.exists() and not args.force:
            log.info("resuming [%s] from %s", window_name, ckpt)
            all_results[window_name] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue
        result = _run_one(window_name, start, end, panel, symbols, base_cfg)
        _write_json(ckpt, result)
        all_results[window_name] = result
        log.info("[%s] verdict=%s sharpe=%.3f mc_p5=%.3f",
                  window_name, result["verdict"],
                  result["metrics_primary_no_spread"]["sharpe_ratio"],
                  result["monte_carlo"]["sharpe"]["p5"])

    _write_json(REPORT_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Regime-generalization test. Production config re-run UNMODIFIED "
            "against new calendar windows; no parameter here was selected or "
            "tuned using any result in this file."
        ),
        "data": panel_meta,
        "results": all_results,
    })
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
