"""
Phase 2 gated-strategy validation (regime_gate_report.md) — `pairs_trading`,
low-frequency `entry_z=4.0` config, gated by
`python/analytics/trend_efficiency_gate.shifted_entry_gate`.

METHODOLOGY — the thing Phase 1's gate earns the right to attempt, and the
thing `orb_vwap_regime` got wrong: this script does NOT check whether the
classifier retroactively labels 2008, 2020, 2022, 2018-2019 or 2024-2026
"correctly" (that would be curve-fitting to the exact known-answer windows
this whole task exists to avoid). It runs the GATED strategy through the
standard walk-forward structure over the FULL 2018-2026 history (all
regimes blended, exactly like every other round in this campaign) and asks
one question only: does gating improve the cost-adjusted PF / WFO pass
ratio / Monte Carlo p5 Sharpe of the full-history backtest, versus the
IDENTICAL ungated frozen config on the IDENTICAL folds?

Classifier input: SPY's close (already inside the pairs universe panel —
`us_equity_broad` bucket — so this needs no new fetch), `window=20`,
`reference_window=252`, BOTH pinned at the values declared in
`trend_efficiency_gate.py`'s module docstring before this script was ever
run. `shifted_entry_gate` (not the raw label) is passed to
`build_pairs_scan_backtest_fn(entry_gate=...)`, so day t's entry decision
depends only on prices through day t-1's close.

Free-parameter accounting for this GATED PIPELINE (configs/goal.yaml /
python/backtest/param_guard.py's 5-parameter ceiling, task's explicit
instruction to fix/remove other parameters to make room): the pairs config
under test (`entry_z=4.0, exit_z=0.5, half_life_multiplier_max_hold=3.0`) is
copied VERBATIM from a prior round — none of its 3 signal parameters are
re-tuned or gridded anywhere in this file (param_grid=[frozen_params], a
single fixed candidate, at every WFO fold). `min_half_life_days` /
`max_half_life_days` stay at their `configs/strategy.yaml` defaults and were
NEVER gridded in ANY round to date (`configs/param_grids.yaml`'s own
comment: "optimizing eligibility bounds against returns is a classic
data-snooping trap") — for this composite accounting they are treated as
the same kind of fixed structural bound `coint_lookback_days` /
`revalidate_every_days` already are, freeing headroom for the classifier's
2 parameters. Total genuinely free (chosen-before-seeing-results)
parameters actually being validated by this script: the classifier's
`window` and `reference_window` — 2, comfortably under the 5 ceiling for
the combined gated pipeline.

Usage:
    python scripts/_regime_gate_phase2_pairs.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import timezone, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("regime_gate_phase2_pairs")

from python.analytics.trend_efficiency_gate import (
    DEFAULT_REFERENCE_WINDOW,
    DEFAULT_WINDOW,
    shifted_entry_gate,
)
from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import build_pairs_scan_backtest_fn, load_wfo_config, max_oos_drawdown_threshold
from python.backtest.pairs_scan_engine import DEFAULT_HALF_SPREAD_BPS, MAX_CONCURRENT_PAIRS, STRESS_HALF_SPREAD_MULTIPLIER
from python.backtest.walk_forward import WalkForwardOptimizer
from run_pairs_scan_backtest import _load_schedule, _strategy_cfg, _strip, load_panels

CACHE_DIR = Path("backtests/reports/_regime_gate_phase2_cache")
REPORT_JSON = Path("backtests/reports/regime_gate_phase2_pairs.json")
FULL_START, FULL_END = "2018-01-01", "2026-08-01"

FROZEN_PARAMS = {"entry_z": 4.0, "exit_z": 0.5, "half_life_multiplier_max_hold": 3.0}
CONFIG_NAME = "pairs_lowfreq_entry_z_4"


def _panel_args():
    import argparse
    return argparse.Namespace(
        warmup_start="2016-06-01", start="2018-01-01",
        dev_end="2024-01-01", holdout_end="2026-08-01", refresh_data=False,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("checkpoint written: %s", path)


def _full_window_result(fn, params: dict) -> dict:
    start_ts, end_ts = pd.Timestamp(FULL_START), pd.Timestamp(FULL_END)
    metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), params)
    mc = MonteCarloValidator(n_sims=500).run(metrics.get("daily_returns", []))
    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd = max_oos_drawdown_threshold()
    gates = {
        "has_trades": metrics["n_trades"] > 0,
        "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= min_oos_sharpe,
        "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_dd,
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
    }
    return {
        "window": [FULL_START, FULL_END],
        "metrics": _strip(metrics),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }


def main() -> None:
    close, adv, _universe, candidate_pairs, meta = load_panels(_panel_args())
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)
    log.info("panel/schedule loaded: %s", meta)

    spy_close = close["SPY"].dropna()
    entry_gate = shifted_entry_gate(spy_close, window=DEFAULT_WINDOW, reference_window=DEFAULT_REFERENCE_WINDOW)
    log.info("entry_gate: %d/%d days ON (window=%d, reference_window=%d)",
              int(entry_gate.sum()), len(entry_gate), DEFAULT_WINDOW, DEFAULT_REFERENCE_WINDOW)

    fn_ungated = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=DEFAULT_HALF_SPREAD_BPS, max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
    )
    fn_gated = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=DEFAULT_HALF_SPREAD_BPS, max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
        entry_gate=entry_gate,
    )

    wfo_cfg = load_wfo_config("pairs_trading")
    log.info("WFO config: %s", wfo_cfg)

    results = {}
    for label, fn in (("ungated", fn_ungated), ("gated", fn_gated)):
        ckpt = CACHE_DIR / f"pairs_{label}.json"
        if ckpt.exists():
            log.info("resuming [%s] from %s", label, ckpt)
            results[label] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue
        log.info("=== %s WFO over full history [%s, %s) ===", label, FULL_START, FULL_END)
        wfo = WalkForwardOptimizer(fn, wfo_cfg, [FROZEN_PARAMS]).run(
            pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime())
        wfo.print_summary()

        # Concatenate every fold's OOS daily returns for one aggregate,
        # full-history-blended Monte Carlo / Sharpe (distinct from, and
        # complementary to, the fold-by-fold pass ratio above).
        oos_returns: list[float] = []
        for f in wfo.folds:
            oos_returns.extend(f.oos_metrics.get("daily_returns", []))
        mc_aggregate = MonteCarloValidator(n_sims=500).run(oos_returns)

        full_window = _full_window_result(fn, FROZEN_PARAMS)

        results[label] = {
            "config": CONFIG_NAME,
            "final_params": FROZEN_PARAMS,
            "wfo": wfo.to_dict(),
            "aggregate_oos_monte_carlo": mc_aggregate.to_dict(),
            "aggregate_oos_n_days": len(oos_returns),
            "full_window_2018_2026": full_window,
        }
        _write_json(ckpt, results[label])

    # 2x-spread stress on the full window, gated vs ungated, matching every
    # other round's mandatory stress re-run.
    stress = {}
    for label, entry_gate_arg in (("ungated", None), ("gated", entry_gate)):
        fn_stress = build_pairs_scan_backtest_fn(
            close, adv, schedule, base_cfg,
            half_spread_bps=DEFAULT_HALF_SPREAD_BPS * STRESS_HALF_SPREAD_MULTIPLIER,
            max_concurrent_pairs=MAX_CONCURRENT_PAIRS, entry_gate=entry_gate_arg,
        )
        stress[label] = _strip(fn_stress(
            pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime(), FROZEN_PARAMS))

    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Phase 2 gated-vs-ungated full-history (2018-2026, all regimes "
            "blended) WFO comparison for pairs_lowfreq_entry_z_4. Classifier "
            "params (window=20, reference_window=252) fixed BEFORE this "
            "script was run; frozen strategy params never re-tuned here."
        ),
        "classifier": {"window": DEFAULT_WINDOW, "reference_window": DEFAULT_REFERENCE_WINDOW,
                        "n_days_gate_on": int(entry_gate.sum()), "n_days_total": len(entry_gate)},
        "results": results,
        "stress_2x_spread_full_window": stress,
    }
    _write_json(REPORT_JSON, out)
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
