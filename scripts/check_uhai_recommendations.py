#!/usr/bin/env python3
"""
Should the self-improve loop be re-run? — a staleness check over this repo's
own promotion history (backtests/logs/promotion_history.jsonl).

Reports, per strategy:
  - how long ago its parameters were last promoted, and
  - whether the LIVE parameters (configs/strategy.yaml's current values,
    re-measured on each run's window) are scoring worse than they used to
    — NOT whether the search's best candidate that run scored worse; a
    rejected candidate says nothing about what's actually deployed.

This RECOMMENDS ONLY. It runs no backtest, touches no config, and never
promotes anything — running scripts/self_improve_loop.py stays a deliberate
act, and that script's own gates remain the only thing that can change a
parameter. Nothing here reads or writes auto_execute.

Synthetic (--demo) history is excluded by default: staleness of real trading
parameters cannot be judged from synthetic runs.

Exit codes (so cron/CI can branch on the result):
    0  nothing to do
    2  at least one strategy looks stale
    1  could not assess (e.g. unreadable history)

Usage:
    python scripts/check_uhai_recommendations.py
    python scripts/check_uhai_recommendations.py --max-age-days 45
    python scripts/check_uhai_recommendations.py --include-synthetic

    # daily at 09:00, mailing only when something is stale
    0 9 * * * cd /path/to/us-stock-trading && .venv/bin/python \
        scripts/check_uhai_recommendations.py || true
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.uhai.client import assess_degradation, days_since_last_promotion
from python.uhai.explain import explain_decision
from python.uhai.history import load_latest_raw_record

STRATEGY_CONFIG_PATH = Path("configs/strategy.yaml")
PARAM_GRIDS_PATH = Path("configs/param_grids.yaml")


def _default_strategies() -> list[str]:
    """Strategies self_improve_loop.py can actually act on: `enabled: true`
    in configs/strategy.yaml AND present in configs/param_grids.yaml (no
    grid, nothing to re-search). Derived at call time rather than
    hardcoded — a hardcoded list drifts the moment a strategy is
    retired/added and silently keeps reporting on strategies nobody can
    promote anymore (this is exactly what happened here: pairs_trading and
    xsection_mean_reversion were RETIRED 2026-09-11 — see configs/strategy.yaml
    — and this script kept hardcoding just those two)."""
    try:
        strategy_cfg = yaml.safe_load(STRATEGY_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        grid_cfg = yaml.safe_load(PARAM_GRIDS_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        return []

    enabled = {
        name for name, block in strategy_cfg.items()
        if isinstance(block, dict) and block.get("enabled") is True
    }
    return sorted(enabled & set(grid_cfg))


def main() -> int:
    strategies_choices = _default_strategies()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=strategies_choices + ["all"], default="all")
    parser.add_argument("--max-age-days", type=float, default=30.0,
                        help="flag a strategy whose last promotion is older than this")
    parser.add_argument("--recent-runs", type=int, default=3,
                        help="how many of the newest runs count as 'recent'")
    parser.add_argument("--include-synthetic", action="store_true",
                        help="also read --demo runs (for testing this script only)")
    parser.add_argument("--explain", action="store_true",
                        help="also print a UHAI-backed explanation of the most recent "
                             "record for each flagged strategy (falls back to a local, "
                             "LLM-free rendering if no sidecar is reachable — see "
                             "python/uhai/explain.py)")
    args = parser.parse_args()

    # Keep the report readable: the history reader's own INFO lines would
    # otherwise interleave with it.
    logging.basicConfig(level=logging.WARNING)

    strategies = strategies_choices if args.strategy == "all" else [args.strategy]
    if not strategies:
        print("No enabled strategy has a param grid to monitor "
              f"(checked {STRATEGY_CONFIG_PATH} against {PARAM_GRIDS_PATH}).")
        return 0
    stale: list[str] = []

    for strategy in strategies:
        print(f"\n{strategy}")

        try:
            age_days = days_since_last_promotion(
                strategy, include_synthetic_history=args.include_synthetic)
            verdict = assess_degradation(
                strategy,
                recent_runs=args.recent_runs,
                include_synthetic_history=args.include_synthetic,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  could not assess: {exc}")
            return 1

        if age_days is None:
            print("  last promotion : never — current values are the checked-in defaults")
        else:
            print(f"  last promotion : {age_days:.0f} days ago")
            if age_days > args.max_age_days:
                stale.append(strategy)

        print(f"  performance    : {verdict.reason}")
        if verdict.n_recent:
            print(f"                   ({verdict.n_recent} recent vs "
                  f"{verdict.n_earlier} earlier run(s))")
        if verdict.is_degraded and strategy not in stale:
            stale.append(strategy)

    print()
    if not stale:
        print("Nothing looks stale.")
        return 0

    print(f"Worth re-running: {', '.join(stale)}")
    print("\nReport-only first, then review before dropping --no-write:")
    for strategy in stale:
        print(f"  python scripts/self_improve_loop.py --strategy {strategy} --no-write")

    if args.explain:
        print("\nWhy (most recent record on file):")
        for strategy in stale:
            record = load_latest_raw_record(
                strategy, include_synthetic=args.include_synthetic)
            if record is None:
                print(f"  {strategy}: no promotion_history.jsonl record to explain")
                continue
            print(f"  {strategy}: {explain_decision(strategy, record)}")

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
