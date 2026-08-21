"""
Exit-rule ablations for scanned-universe pairs trading (round two).

WHAT THIS TESTS
---------------
Round one (`scripts/run_pairs_scan_backtest.py`,
`backtests/reports/pairs_scan_report.md`) fixed the sample-size problem — 8
trades became 1,592 — and still concluded NO-GO. It also localized the
failure precisely: **91-96% of positions exited on the STALE TIMEOUT rather
than on z-reversion**. The half-life-derived max-hold was closing trades at
an arbitrary point rather than a considered one, so the natural question is
whether a better exit rule rescues trades that are currently being timed out
at an unfavorable moment.

Costs are NOT the hypothesis here: round one measured a pre-cost profit
factor of 0.850 (dev) and 0.991 (holdout), so the raw signal was already at
or below breakeven before a dollar of cost. This round asks only whether the
EXIT is what is throwing the edge away.

Four variants, run in the brief's order, on the same development window and
the same precomputed point-in-time scan schedule as round one:

  A0_baseline               round one, unchanged (the control)
  A1_dynamic_half_life      re-derive max-hold from the freshest point-in-time
                            half-life estimate while the position is open
  A2_coint_breakdown_exit   exit as soon as the pair stops passing the
                            existing `is_tradeable` screen
  A3a_stop_replaces_timeout z-widening stop at `entry_z * 1.5`, with the stale
                            timeout DISABLED so the stop replaces it and the
                            free-parameter count stays at 5
  A3b_stop_plus_timeout     both, which is 6 parameters — DIAGNOSTIC ONLY,
                            explicitly not promotable, run to separate "the
                            stop helps" from "removing the timeout hurts"

**A3a/A3b are a documented POLICY change.** Chan argues against price stops
for mean reversion (a spread that moved against you is cheaper, not worse),
and `python/core/pair_position_manager.py` says so in its module docstring.
Enabling a stop needs explicit human sign-off regardless of what the numbers
say; this script only measures.

SELECTION RULE — FIXED BEFORE ANY RESULT WAS SEEN
--------------------------------------------------
The single configuration that goes to the holdout is chosen by: highest WFO
pass ratio, tie-broken by higher mean OOS Sharpe, tie-broken by variant order
above. A3b is excluded from selection because it is over the parameter
budget. If A0 wins, the answer is "no change" and NO new holdout run happens
— round one already spent that one look, and re-running it would be a second
peek at the reserved window.

DURABILITY
----------
Every ablation checkpoints to its own JSON under
`backtests/reports/_pairs_scan_cache/`; a rerun skips what is already on
disk, so an interruption costs at most one ablation. The expensive scan
schedule is reused from round one and never recomputed.

Usage:
    python scripts/run_pairs_exit_ablations.py --phase dev
    python scripts/run_pairs_exit_ablations.py --phase holdout   # once, ever
    python scripts/run_pairs_exit_ablations.py --phase report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("run_pairs_exit_ablations")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import (
    build_pairs_scan_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    load_param_grid,
    load_wfo_config,
    max_oos_drawdown_threshold,
)
from python.backtest.pairs_scan_engine import STOP_Z_MULTIPLE, STRESS_HALF_SPREAD_MULTIPLIER
from python.backtest.walk_forward import WalkForwardOptimizer

# Reuse round one's data loading, scan-schedule loading and checkpoint
# helpers verbatim — the ablations must differ from the control in the exit
# rule and NOTHING else.
from run_pairs_scan_backtest import (  # noqa: E402
    CACHE_DIR,
    DEV_CHECKPOINT as ROUND_ONE_DEV_CHECKPOINT,
    HOLDOUT_CHECKPOINT as ROUND_ONE_HOLDOUT_CHECKPOINT,
    STRATEGY,
    _load_schedule,
    _strategy_cfg,
    _strip,
    _write_json,
    load_panels,
)

ABLATIONS: dict[str, dict] = {
    "A0_baseline": {},
    "A1_dynamic_half_life": {"dynamic_half_life": True},
    "A2_coint_breakdown_exit": {"coint_breakdown_exit": True},
    "A3a_stop_replaces_timeout": {"stop_z_multiple": STOP_Z_MULTIPLE,
                                  "stale_timeout_enabled": False},
    "A3b_stop_plus_timeout": {"stop_z_multiple": STOP_Z_MULTIPLE},
    # Added only AFTER A1 and A2 each cleared the "enough promise to justify
    # combining" bar the brief sets — A1 produced the first pre-cost profit
    # factor above 1.0 seen in this study, A2 the first WFO pass ratio above
    # the 0.60 gate. Still 5 free parameters and no policy change: both
    # components are mechanical. Added BEFORE the holdout was touched, and it
    # widens the selection pool from 4 promotable variants to 5, which is
    # disclosed as multiple-comparisons exposure in the report.
    "A4_dynamic_plus_breakdown": {"dynamic_half_life": True,
                                  "coint_breakdown_exit": True},
}

# Over the 5-parameter budget (stop AND timeout), so it can be measured but
# never selected. See the module docstring.
NOT_PROMOTABLE = {"A3b_stop_plus_timeout"}

# Enabling a price stop contradicts the documented design and needs a human.
POLICY_CHANGE = {"A3a_stop_replaces_timeout", "A3b_stop_plus_timeout"}

ABLATION_HOLDOUT_CHECKPOINT = CACHE_DIR / "_checkpoint_ablation_holdout.json"
REPORT_JSON = Path("backtests/reports/pairs_exit_ablations.json")


def _dev_checkpoint(name: str) -> Path:
    return CACHE_DIR / f"_checkpoint_ablation_dev_{name}.json"


# ── dev ─────────────────────────────────────────────────────────────────────

def _run_one_ablation(name: str, rules: dict, close, adv, schedule, base_cfg,
                      args) -> dict:
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.dev_end)
    param_grid = load_param_grid(STRATEGY)
    wfo_cfg = load_wfo_config(STRATEGY)

    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps,
        max_concurrent_pairs=args.max_concurrent_pairs,
        exit_rules=rules)

    log.info("[%s] WFO over [%s, %s), %d candidates, exit_rules=%s",
             name, start_ts.date(), end_ts.date(), len(param_grid), rules or "{} (control)")
    wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())
    wfo.print_summary()
    if not wfo.folds:
        raise SystemExit("dev window too short for a single WFO fold")

    final_params = dict(wfo.folds[-1].best_params)
    full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=args.max_concurrent_pairs,
        exit_rules=rules)
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    gates = {
        "wfo_go": wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(wfo),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
    }
    result = {
        "ablation": name,
        "exit_rules": rules,
        "promotable": name not in NOT_PROMOTABLE,
        "policy_change_requires_human_signoff": name in POLICY_CHANGE,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "final_params": final_params,
        "wfo": wfo.to_dict(),
        "full_window": _strip(full),
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    _write_json(_dev_checkpoint(name), result)
    return result


def phase_dev(args) -> dict:
    close, adv, _universe, candidate_pairs, meta = load_panels(args)
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)

    results = {}
    for name, rules in ABLATIONS.items():
        ckpt = _dev_checkpoint(name)
        if ckpt.exists() and not args.force:
            log.info("[%s] resuming from %s", name, ckpt)
            results[name] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue
        results[name] = _run_one_ablation(name, rules, close, adv, schedule, base_cfg, args)

    log.info("dev ablations complete: %s", _leaderboard(results))
    return results


def _leaderboard(results: dict) -> list:
    """Fixed selection rule (declared in the module docstring, before any
    result existed): pass ratio, then mean OOS Sharpe, then variant order."""
    order = list(ABLATIONS)
    ranked = sorted(
        (r for r in results.values() if r["promotable"]),
        key=lambda r: (-r["wfo"]["pass_ratio"], -r["wfo"]["oos_sharpe_mean"],
                       order.index(r["ablation"])),
    )
    return [(r["ablation"], r["wfo"]["pass_ratio"], r["wfo"]["oos_sharpe_mean"],
             r["verdict"]) for r in ranked]


# ── holdout ─────────────────────────────────────────────────────────────────

def phase_holdout(args) -> dict:
    """ONE run of ONE configuration, and only if an ablation actually beat the
    control. Refuses to overwrite an existing ablation-holdout checkpoint:
    the reserved window is a single-use resource and re-running it under a
    different variant would turn it into a second development window."""
    results = {}
    for name in ABLATIONS:
        ckpt = _dev_checkpoint(name)
        if not ckpt.exists():
            raise SystemExit(f"missing {ckpt} — run --phase dev first")
        results[name] = json.loads(ckpt.read_text(encoding="utf-8"))

    board = _leaderboard(results)
    winner = board[0][0]
    log.info("selection rule -> %s (leaderboard: %s)", winner, board)

    if winner == "A0_baseline":
        log.info("no ablation beat the control; the answer is 'no change'. Round one's "
                 "holdout already stands and is NOT re-run.")
        payload = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "selected": "A0_baseline",
            "holdout_rerun": False,
            "reason": ("no promotable ablation beat the control on the fixed selection "
                       "rule, so the final configuration is unchanged from round one "
                       "and its single holdout evaluation still stands"),
            "leaderboard": board,
            "round_one_holdout": json.loads(
                ROUND_ONE_HOLDOUT_CHECKPOINT.read_text(encoding="utf-8"))
            if ROUND_ONE_HOLDOUT_CHECKPOINT.exists() else None,
        }
        _write_json(ABLATION_HOLDOUT_CHECKPOINT, payload)
        return payload

    if ABLATION_HOLDOUT_CHECKPOINT.exists() and not args.force:
        raise SystemExit(
            f"{ABLATION_HOLDOUT_CHECKPOINT} already exists — the holdout has already "
            "been evaluated once for this round and must not be re-run")

    close, adv, _universe, candidate_pairs, meta = load_panels(args)
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)
    rules = ABLATIONS[winner]
    final_params = results[winner]["final_params"]

    start_ts, end_ts = pd.Timestamp(args.dev_end), pd.Timestamp(args.holdout_end)
    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps,
        max_concurrent_pairs=args.max_concurrent_pairs,
        exit_rules=rules)
    log.info("holdout: ONE run of %s %s over [%s, %s)",
             winner, final_params, start_ts.date(), end_ts.date())
    metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)
    mc = MonteCarloValidator(n_sims=500).run(metrics.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=args.max_concurrent_pairs,
        exit_rules=rules)
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), final_params)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    gates = {
        "has_trades": metrics["n_trades"] > 0,
        "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= float(goal["wfo"]["min_oos_sharpe"]),
        "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_oos_drawdown_threshold(),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= float(goal["monte_carlo"]["min_p5_sharpe"]),
    }
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "selected": winner,
        "holdout_rerun": True,
        "policy_change_requires_human_signoff": winner in POLICY_CHANGE,
        "exit_rules": rules,
        "params": final_params,
        "window": [str(start_ts.date()), str(end_ts.date())],
        "leaderboard": board,
        "metrics": _strip(metrics),
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    _write_json(ABLATION_HOLDOUT_CHECKPOINT, payload)
    return payload


# ── report ──────────────────────────────────────────────────────────────────

def phase_report(args) -> dict:
    dev = {name: json.loads(_dev_checkpoint(name).read_text(encoding="utf-8"))
           for name in ABLATIONS if _dev_checkpoint(name).exists()}
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "strategy": STRATEGY,
        "round": "exit_rule_ablations",
        "selection_rule": ("max WFO pass ratio, tie-break max mean OOS Sharpe, "
                           "tie-break variant order; A3b excluded (over budget)"),
        "stop_z_multiple": STOP_Z_MULTIPLE,
        "round_one_dev": json.loads(ROUND_ONE_DEV_CHECKPOINT.read_text(encoding="utf-8"))
        if ROUND_ONE_DEV_CHECKPOINT.exists() else None,
        "dev": dev,
        "leaderboard": _leaderboard(dev) if dev else [],
        "holdout": json.loads(ABLATION_HOLDOUT_CHECKPOINT.read_text(encoding="utf-8"))
        if ABLATION_HOLDOUT_CHECKPOINT.exists() else None,
    }
    _write_json(REPORT_JSON, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["dev", "holdout", "report", "all"], default="all")
    parser.add_argument("--warmup-start", default="2016-06-01")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--dev-end", default="2024-01-01")
    parser.add_argument("--holdout-end", default="2026-08-01")
    parser.add_argument("--half-spread-bps", type=float, default=3.0)
    parser.add_argument("--max-concurrent-pairs", type=int, default=10)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recompute checkpoints that already exist (NOT for the holdout)")
    args = parser.parse_args()

    phases = ["dev", "holdout", "report"] if args.phase == "all" else [args.phase]
    for phase in phases:
        log.info("=== phase: %s ===", phase)
        {"dev": phase_dev, "holdout": phase_holdout, "report": phase_report}[phase](args)
    log.info("done")


if __name__ == "__main__":
    main()
