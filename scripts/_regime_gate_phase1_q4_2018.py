"""
Phase 1 replication test (regime_gate_report.md) — SECONDARY/WEAK window.

PRE-DECLARED (before this script was run, before any result was seen):
Q4 2018 (2018-10-01..2019-01-01, the Oct-Dec 2018 selloff: S&P -19.8%
peak-to-trough, VIX above 25 for weeks, a real but short volatility-spike/
partial-mean-reversion episode) is used as a SECONDARY, WEAK check only. It
is explicitly declared too short (~63 trading days) for a meaningful trade
count on its own — this is stated per the task's instruction, not softened
after seeing results. It is a useful check only because it needs zero new
data fetching (already inside the 2016-06-01..2026-07-31 pairs panel and the
2017-11-01..2022-12-31 xsection panel both used, unmodified, by
`scripts/_regime_generalization_pairs.py` / `_xsection.py`) and it is a
genuinely different, non-overlapping sub-window from anything previously
reported at this granularity (the existing report only tested the full
2018-2019 two-year span combined, never isolated the Q4 2018 spike).

All THREE frozen configs are copied VERBATIM from
`scripts/_regime_generalization_pairs.py` / `_regime_generalization_xsection.py`
— not one parameter is re-picked here. Same point-in-time scan schedule,
same universe files, same engines, same cost model, same gates
(`configs/goal.yaml`, unmodified).

Usage:
    python scripts/_regime_gate_phase1_q4_2018.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("regime_gate_phase1_q4_2018")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import build_pairs_scan_backtest_fn, max_oos_drawdown_threshold
from python.backtest.pairs_scan_engine import (
    DEFAULT_HALF_SPREAD_BPS,
    MAX_CONCURRENT_PAIRS,
    STRESS_HALF_SPREAD_MULTIPLIER,
)
from python.backtest.vector_engine import run_vector_backtest
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy
from python.data.price_cache import get_cached_price_panel
from run_pairs_scan_backtest import _load_schedule, _strategy_cfg as _pairs_strategy_cfg, _strip, load_panels

CACHE_DIR = Path("backtests/reports/_regime_gate_phase1_cache")
REPORT_JSON = Path("backtests/reports/regime_gate_phase1_q4_2018.json")

WINDOW_NAME = "2018_q4_selloff"
WINDOW = ("2018-10-01", "2019-01-01")

PAIRS_FROZEN_CONFIGS: dict[str, dict] = {
    "pairs_dynamic_halflife_exit": {
        "final_params": {"entry_z": 2.5, "exit_z": 1.0, "half_life_multiplier_max_hold": 4.0},
        "exit_rules": {"dynamic_half_life": True},
    },
    "pairs_lowfreq_entry_z_4": {
        "final_params": {"entry_z": 4.0, "exit_z": 0.5, "half_life_multiplier_max_hold": 3.0},
        "exit_rules": {},
    },
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("checkpoint written: %s", path)


def _panel_args():
    import argparse
    return argparse.Namespace(
        warmup_start="2016-06-01", start="2018-01-01",
        dev_end="2024-01-01", holdout_end="2026-08-01", refresh_data=False,
    )


def run_pairs() -> dict:
    close, adv, _universe, candidate_pairs, meta = load_panels(_panel_args())
    base_cfg = _pairs_strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)
    start_ts, end_ts = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd = max_oos_drawdown_threshold()

    results = {}
    for name, cfg in PAIRS_FROZEN_CONFIGS.items():
        fn = build_pairs_scan_backtest_fn(
            close, adv, schedule, base_cfg,
            half_spread_bps=DEFAULT_HALF_SPREAD_BPS,
            max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
            exit_rules=cfg["exit_rules"],
        )
        metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg["final_params"])
        mc = MonteCarloValidator(n_sims=500).run(metrics.get("daily_returns", []))
        stress_fn = build_pairs_scan_backtest_fn(
            close, adv, schedule, base_cfg,
            half_spread_bps=DEFAULT_HALF_SPREAD_BPS * STRESS_HALF_SPREAD_MULTIPLIER,
            max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
            exit_rules=cfg["exit_rules"],
        )
        stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg["final_params"])
        gates = {
            "has_trades": metrics["n_trades"] > 0,
            "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= min_oos_sharpe,
            "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_dd,
            "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        }
        results[name] = {
            "config": name, "final_params": cfg["final_params"], "exit_rules": cfg["exit_rules"],
            "window_name": WINDOW_NAME, "window": [str(start_ts.date()), str(end_ts.date())],
            "metrics": _strip(metrics), "stress_2x_spread": _strip(stress),
            "monte_carlo": mc.to_dict(), "gates": gates,
            "verdict": "GO" if all(gates.values()) else "NO-GO",
        }
        log.info("[pairs/%s] verdict=%s sharpe=%.3f trades=%d",
                  name, results[name]["verdict"], metrics["sharpe_ratio"], metrics["n_trades"])
    return results


_SYMBOLS_NOT_YET_LISTED = {"NBIS", "SNDK"}  # same exclusion as _regime_generalization_xsection.py


def run_xsection() -> dict:
    with open("configs/universe.yaml", encoding="utf-8") as f:
        symbols = list(yaml.safe_load(f)["fixed_universe"]["symbols"])
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)["xsection_mean_reversion"]

    fetchable = [s for s in symbols if s.upper() not in _SYMBOLS_NOT_YET_LISTED]
    panel, _quality, _meta = get_cached_price_panel(fetchable, "2017-11-01", "2022-12-31")

    strategy = CrossSectionalMeanReversionStrategy(
        lookback_days=base_cfg["lookback_days"],
        gross_leverage_target=base_cfg["gross_leverage_target"],
        min_universe_size=base_cfg["min_universe_size"],
    )
    all_dates = sorted(panel.index.get_level_values(0).unique())
    tradeable_dates = all_dates[30:]
    start_ts, end_ts = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    window_dates = [d for d in tradeable_dates if start_ts <= pd.Timestamp(d) < end_ts]
    universe_by_day = {d: list(symbols) for d in window_dates}

    result = run_vector_backtest(strategy, panel, universe_by_day, capital=1_000_000.0, half_spread_bps=None)
    net = result.daily_returns
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252)) if len(net) >= 2 and net.std(ddof=1) > 0 else 0.0
    equity = (1.0 + net).cumprod() if len(net) else pd.Series(dtype=float)
    max_dd_val = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    gross = result.daily_gross_returns * 1_000_000.0
    cost = result.daily_costs * 1_000_000.0
    netd = net * 1_000_000.0

    def _pf(s):
        wins = float(s[s > 0].sum())
        losses = float(-s[s < 0].sum())
        return wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)

    metrics = {
        "sharpe_ratio": sharpe, "max_drawdown": max_dd_val,
        "n_active_days": int((net != 0).sum()), "n_days": int(len(net)),
        "total_net_pnl": float(netd.sum()), "total_gross_pnl": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "gross_to_cost_ratio": float(gross.sum()) / float(cost.sum()) if cost.sum() != 0 else float("inf"),
        "profit_factor_net_daily": _pf(netd),
    }
    mc = MonteCarloValidator(n_sims=500).run(net.tolist())
    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd_gate = max_oos_drawdown_threshold()
    min_cost_ratio = float(goal["cost_gate"]["min_gross_to_cost_ratio"])
    gates = {
        "has_active_days": metrics["n_active_days"] > 0,
        "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= min_oos_sharpe,
        "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_dd_gate,
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        "cost_gate_gross_to_cost_ratio": metrics["gross_to_cost_ratio"] >= min_cost_ratio,
    }
    result_out = {
        "config": "xsection_mean_reversion_production",
        "window_name": WINDOW_NAME, "window": [str(start_ts.date()), str(end_ts.date())],
        "n_window_dates_available": len(window_dates),
        "n_universe_symbols_used": len(symbols) - len(_SYMBOLS_NOT_YET_LISTED),
        "metrics": metrics, "monte_carlo": mc.to_dict(), "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    log.info("[xsection] verdict=%s sharpe=%.3f n_active_days=%d",
              result_out["verdict"], sharpe, metrics["n_active_days"])
    return result_out


def main() -> None:
    pairs_results = run_pairs()
    xsection_result = run_xsection()
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": "Phase 1 SECONDARY/WEAK replication window (Q4 2018 selloff, ~63 trading "
                "days, pre-declared as too short for a meaningful trade count on its own). "
                "All configs frozen verbatim; window declared BEFORE this script was run.",
        "window": WINDOW,
        "pairs_results": pairs_results,
        "xsection_result": xsection_result,
    }
    _write_json(REPORT_JSON, out)
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
