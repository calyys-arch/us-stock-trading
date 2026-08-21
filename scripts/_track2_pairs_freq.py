"""
Track 2 (backtests/reports/alt_universe_frequency_exploration.md), second
half: does a MATERIALLY LOWER trade frequency (much higher `entry_z`) fix
the fixed-cost-floor problem `pairs_scan_report.md` diagnosed?

That report's own closing recommendation (its final paragraph) is the
premise here, quoted directly: "a rule that trades the same spreads far
less often but at much larger deviations (materially higher `entry_z`,
correspondingly fewer trades) would be attacking the $53.83/trade floor by
amortizing it over a bigger move". This script is exactly that follow-up,
nothing else — reuses round 1's ALREADY-COMPUTED point-in-time scan
schedule (`backtests/reports/_pairs_scan_cache/scan_schedule.jsonl`, never
recomputed here) and the identical `PairsTradingStrategy` /
`PairPositionManager` code (round 1's baseline exit rule, A0 — not any of
round 2's ablations, to isolate the ONE variable this test is about:
entry threshold, not exit logic).

Grid: `entry_z` in {3.0, 3.5, 4.0} (round 1/2's grid topped out at 2.5 —
this starts strictly above that) x `exit_z` in {0.0, 0.5} (round 2's A2
holdout pick and round 1's incumbent, the two exit_z values already shown
to matter). `half_life_multiplier_max_hold` fixed at the incumbent 3.0
(not gridded here — bounding scope to the ONE new axis this test is about,
per the exploration's "1-2 candidates per track, go deep not wide" rule).
6 candidates, same 3-parameter family as round 1 (no new free parameter).

Honest holdout-reuse disclosure (stated once, prominently, in the
exploration report): the 2024-01-01..2026-08-01 window has now been read by
THREE separate evaluations across this repo's history (round 1's config,
round 2's A2, and this run) — round 1 and round 2 already spent some of
this window's "surprise value" on different configurations. This run's
holdout number should be read with that in mind: it is a real out-of-sample
check for THIS configuration (never before evaluated on this window), but
the window itself is no longer pristine in the way a first-ever holdout
would be.

Usage:
    python scripts/_track2_pairs_freq.py --phase dev
    python scripts/_track2_pairs_freq.py --phase holdout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("track2_pairs_freq")

import pandas as pd
import yaml

import run_pairs_scan_backtest as base  # noqa: E402
from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.optimize import (  # noqa: E402
    build_pairs_scan_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    load_wfo_config,
    max_oos_drawdown_threshold,
)
from python.backtest.pairs_scan_engine import (  # noqa: E402
    DEFAULT_HALF_SPREAD_BPS,
    MAX_CONCURRENT_PAIRS,
    STRESS_HALF_SPREAD_MULTIPLIER,
)
from python.backtest.walk_forward import WalkForwardOptimizer  # noqa: E402

LOW_FREQ_GRID = [
    {"entry_z": ez, "exit_z": xz, "half_life_multiplier_max_hold": 3.0}
    for ez, xz in product([3.0, 3.5, 4.0], [0.0, 0.5])
]

DEV_CHECKPOINT = base.CACHE_DIR / "_checkpoint_track2_lowfreq_dev.json"
HOLDOUT_CHECKPOINT = base.CACHE_DIR / "_checkpoint_track2_lowfreq_holdout.json"
REPORT_JSON = Path("backtests/reports/track2_pairs_lowfreq_report.json")


def _strip(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "daily_returns"}


def phase_dev(args) -> dict:
    close, adv, _universe, candidate_pairs, meta = base.load_panels(args)
    base_cfg = base._strategy_cfg()
    schedule = base._load_schedule(close, candidate_pairs, base_cfg)

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.dev_end)
    wfo_cfg = load_wfo_config(base.STRATEGY)
    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps,
        max_concurrent_pairs=args.max_concurrent_pairs)

    log.info("dev (low-freq): WFO over [%s, %s), %d low-freq candidates",
             start_ts.date(), end_ts.date(), len(LOW_FREQ_GRID))
    candidate_wfo = WalkForwardOptimizer(fn, wfo_cfg, LOW_FREQ_GRID).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())
    candidate_wfo.print_summary()

    if not candidate_wfo.folds:
        raise SystemExit("dev window too short for a single WFO fold")

    # Selection rule, fixed BEFORE looking at results (same convention round
    # 1 used): the last fold's IS-winning candidate.
    final_params = dict(candidate_wfo.folds[-1].best_params)
    full = base._full_window_run(fn, start_ts, end_ts, final_params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=args.max_concurrent_pairs)
    stress = base._full_window_run(stress_fn, start_ts, end_ts, final_params)

    # Also report the incumbent-grid best (round 1's A0, entry_z<=2.5) on the
    # SAME window/schedule for a direct side-by-side comparison.
    incumbent_wfo = WalkForwardOptimizer(fn, wfo_cfg, [
        {"entry_z": ez, "exit_z": xz, "half_life_multiplier_max_hold": hh}
        for ez in [1.5, 2.0, 2.5] for xz in [0.0, 0.5, 1.0] for hh in [2.0, 3.0, 4.0]
    ]).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    gates = {
        "wfo_go": candidate_wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(candidate_wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(candidate_wfo),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
    }

    result = {
        "phase": "dev", "variant": "low_frequency_entry_z", "hypothesis_source":
            "pairs_scan_report.md closing recommendation",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "data": meta,
        "half_spread_bps": args.half_spread_bps,
        "grid": LOW_FREQ_GRID,
        "final_params": final_params,
        "candidate_wfo": candidate_wfo.to_dict(),
        "incumbent_grid_wfo_same_window": incumbent_wfo.to_dict(),
        "full_window": _strip(full),
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    base._write_json(DEV_CHECKPOINT, result)
    return result


def phase_holdout(args) -> dict:
    if not DEV_CHECKPOINT.exists():
        raise SystemExit(f"missing {DEV_CHECKPOINT} — run --phase dev first")
    if HOLDOUT_CHECKPOINT.exists():
        raise SystemExit(f"{HOLDOUT_CHECKPOINT} already exists — holdout is evaluated exactly once; "
                          "delete it manually if you really intend to redo this (you should not).")
    dev = json.loads(DEV_CHECKPOINT.read_text(encoding="utf-8"))
    final_params = dev["final_params"]

    close, adv, _universe, candidate_pairs, meta = base.load_panels(args)
    base_cfg = base._strategy_cfg()
    schedule = base._load_schedule(close, candidate_pairs, base_cfg)

    start_ts, end_ts = pd.Timestamp(args.dev_end), pd.Timestamp(args.holdout_end)
    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps,
        max_concurrent_pairs=args.max_concurrent_pairs)
    log.info("holdout (low-freq): single run of %s over [%s, %s)", final_params, start_ts.date(), end_ts.date())
    metrics = base._full_window_run(fn, start_ts, end_ts, final_params)
    mc = MonteCarloValidator(n_sims=500).run(metrics.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=args.max_concurrent_pairs)
    stress = base._full_window_run(stress_fn, start_ts, end_ts, final_params)

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

    result = {
        "phase": "holdout", "variant": "low_frequency_entry_z",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "holdout_reuse_disclosure": (
            "This window (2024-01-01..2026-08-01) was already read by round 1's "
            "config and round 2's A2 config in pairs_scan_report.md, both NO-GO. "
            "This is a 3rd distinct configuration evaluated on it — a real "
            "out-of-sample check for THIS config, but the window is not a "
            "pristine first-ever holdout at the study level; see this repo's "
            "own multiple-comparisons discussion in pairs_scan_report.md section 9."
        ),
        "params": final_params,
        "half_spread_bps": args.half_spread_bps,
        "metrics": _strip(metrics),
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    base._write_json(HOLDOUT_CHECKPOINT, result)
    return result


def phase_report(args) -> dict:
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "strategy": base.STRATEGY, "mode": "low_frequency_entry_z",
        "dev": json.loads(DEV_CHECKPOINT.read_text(encoding="utf-8")) if DEV_CHECKPOINT.exists() else None,
        "holdout": json.loads(HOLDOUT_CHECKPOINT.read_text(encoding="utf-8")) if HOLDOUT_CHECKPOINT.exists() else None,
    }
    base._write_json(REPORT_JSON, out)
    log.info("wrote %s", REPORT_JSON)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["dev", "holdout", "report", "all"], default="all")
    parser.add_argument("--warmup-start", default="2016-06-01")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--dev-end", default="2024-01-01")
    parser.add_argument("--holdout-end", default="2026-08-01")
    parser.add_argument("--half-spread-bps", type=float, default=DEFAULT_HALF_SPREAD_BPS)
    parser.add_argument("--max-concurrent-pairs", type=int, default=MAX_CONCURRENT_PAIRS)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    phases = ["dev", "holdout", "report"] if args.phase == "all" else [args.phase]
    for phase in phases:
        log.info("=== phase: %s ===", phase)
        {"dev": phase_dev, "holdout": phase_holdout, "report": phase_report}[phase](args)
    log.info("done")


if __name__ == "__main__":
    main()
