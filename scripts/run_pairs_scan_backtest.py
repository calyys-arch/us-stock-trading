"""
Scanned-universe pairs-trading validation runner.

Tests the hypothesis named in `backtests/reports/strategy_review_summary.md`
§2.1 / §4.4: `pairs_trading`'s NO-GO is a rare-event / sample-size failure
(8 trades in ~7 years on one hardcoded pair), so scanning a universe of
candidate pairs — point-in-time, never with hindsight — should raise the
trade count into statistically meaningful territory and let the gates
actually discriminate.

This script never writes to `configs/strategy.yaml` and never touches
`auto_execute`: a GO below is a promotion CANDIDATE for human review, per
`python/backtest/promotion.py`'s human-in-the-loop write path. Gate
thresholds come from `configs/goal.yaml` unchanged.

Phases (each independently resumable — see `--phase`):

  scan     Build the point-in-time cointegration scan schedule over the FULL
           price history. This is the expensive phase (one CADF fit per
           candidate pair per scan date) and is checkpointed to JSONL after
           every scan date, so an interrupted run resumes without losing
           work. Computed once; replayed by every fold and every candidate.
  dev      Walk-forward + Monte Carlo + gates on the DEVELOPMENT window only.
           All parameter selection happens here.
  holdout  Evaluate the ONE final configuration exactly once on the reserved
           recent window. Refuses to run without a completed dev phase, so
           the holdout cannot be peeked at during development.
  report   Merge checkpoints into backtests/reports/pairs_scan_report.json.

Usage:
    python scripts/run_pairs_scan_backtest.py --phase all
    python scripts/run_pairs_scan_backtest.py --phase scan      # resumable
    python scripts/run_pairs_scan_backtest.py --phase dev
    python scripts/run_pairs_scan_backtest.py --phase holdout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
# The position manager logs every open/close at INFO. Across a full grid
# search that is hundreds of thousands of lines of noise that would bury the
# progress output (and slow the run down measurably).
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("run_pairs_scan_backtest")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import (
    build_pairs_scan_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    load_param_grid,
    load_wfo_config,
    max_oos_drawdown_threshold,
    preflight_check,
)
from python.backtest.pairs_scan_engine import (
    DEFAULT_HALF_SPREAD_BPS,
    MAX_CONCURRENT_PAIRS,
    STRESS_HALF_SPREAD_MULTIPLIER,
    build_scan_schedule,
    candidate_pairs_from_buckets,
    load_pairs_universe,
)
from python.backtest.walk_forward import WalkForwardOptimizer
from python.data.price_cache import get_cached_price_panel

STRATEGY = "pairs_trading"
CACHE_DIR = Path("backtests/reports/_pairs_scan_cache")
SCAN_CHECKPOINT = CACHE_DIR / "scan_schedule.jsonl"
DEV_CHECKPOINT = CACHE_DIR / "_checkpoint_dev.json"
HOLDOUT_CHECKPOINT = CACHE_DIR / "_checkpoint_holdout.json"
REPORT_JSON = Path("backtests/reports/pairs_scan_report.json")
_CAPITAL = 1_000_000.0


# ── Data ────────────────────────────────────────────────────────────────────

def load_panels(args) -> tuple[pd.DataFrame, pd.DataFrame, dict, list, dict]:
    """Returns (close_wide, adv_wide, universe_cfg, candidate_pairs, meta)."""
    universe = load_pairs_universe()
    symbols = sorted({s for codes in universe["buckets"].values() for s in codes})
    panel, quality_flags, meta = get_cached_price_panel(
        symbols, args.warmup_start, args.holdout_end, refresh=args.refresh_data)

    close = panel["close"].unstack("code").sort_index()
    adv = panel["adv_20d_dollars"].unstack("code").sort_index()

    # Drop any symbol whose history starts after the requested warmup start —
    # trading a short series would quietly give the scan a different (and
    # shorter) lookback for that name than for every other name.
    warmup_ts = pd.Timestamp(args.warmup_start)
    tolerance = warmup_ts + pd.Timedelta(days=10)
    too_short = [c for c in close.columns if close[c].first_valid_index() is None
                 or close[c].first_valid_index() > tolerance]
    if too_short:
        log.warning("dropping %d symbols whose history starts after %s: %s",
                    len(too_short), tolerance.date(), sorted(too_short))
        close = close.drop(columns=too_short)
        adv = adv.drop(columns=too_short)

    buckets = {b: [s for s in codes if s in close.columns]
               for b, codes in universe["buckets"].items()}
    candidate_pairs = candidate_pairs_from_buckets(buckets)

    meta = {
        "sources": {k: len(v) for k, v in meta["sources"].items()},
        "n_symbols": int(close.shape[1]),
        "n_symbols_dropped_short_history": len(too_short),
        "symbols_dropped": sorted(too_short),
        "n_candidate_pairs": len(candidate_pairs),
        "bucket_sizes": {b: len(c) for b, c in buckets.items()},
        "first_date": str(close.index.min().date()),
        "last_date": str(close.index.max().date()),
        "n_trading_days": int(len(close)),
        "n_symbols_with_data_quality_flags": len(quality_flags),
    }
    return close, adv, universe, candidate_pairs, meta


# ── Phases ──────────────────────────────────────────────────────────────────

def phase_scan(args) -> dict:
    close, _adv, _universe, candidate_pairs, meta = load_panels(args)
    base_cfg = _strategy_cfg()
    log.info("scan: %d candidate pairs over %d trading days (%s..%s)",
             len(candidate_pairs), len(close), meta["first_date"], meta["last_date"])
    schedule = build_scan_schedule(
        close, candidate_pairs,
        lookback_days=base_cfg["coint_lookback_days"],
        revalidate_every_days=base_cfg["revalidate_every_days"],
        checkpoint_path=SCAN_CHECKPOINT,
    )
    passing = [len(v) for v in schedule.values()]
    summary = {
        "n_scan_dates": len(schedule),
        "n_candidate_pairs": len(candidate_pairs),
        "eligible_pairs_per_scan_min": int(min(passing)) if passing else 0,
        "eligible_pairs_per_scan_median": float(pd.Series(passing).median()) if passing else 0.0,
        "eligible_pairs_per_scan_max": int(max(passing)) if passing else 0,
        "data": meta,
    }
    log.info("scan complete: %s", summary)
    return summary


def _strategy_cfg() -> dict:
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)[STRATEGY]


def _load_schedule(close: pd.DataFrame, candidate_pairs: list, base_cfg: dict) -> dict:
    if not SCAN_CHECKPOINT.exists():
        raise SystemExit(f"missing {SCAN_CHECKPOINT} — run --phase scan first")
    schedule = build_scan_schedule(
        close, candidate_pairs,
        lookback_days=base_cfg["coint_lookback_days"],
        revalidate_every_days=base_cfg["revalidate_every_days"],
        checkpoint_path=SCAN_CHECKPOINT,
    )
    return schedule


def _full_window_run(fn, start: pd.Timestamp, end: pd.Timestamp, params: dict) -> dict:
    return fn(start.to_pydatetime(), end.to_pydatetime(), params)


def phase_dev(args) -> dict:
    close, adv, _universe, candidate_pairs, meta = load_panels(args)
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.dev_end)
    param_grid = load_param_grid(STRATEGY)
    n_days = int(((close.index >= start_ts) & (close.index < end_ts)).sum())
    preflight = preflight_check(STRATEGY, base_cfg, param_grid, total_trading_days=n_days)
    wfo_cfg = load_wfo_config(STRATEGY)

    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps,
        max_concurrent_pairs=args.max_concurrent_pairs)

    log.info("dev: WFO over [%s, %s), %d candidates, %d trading days",
             start_ts.date(), end_ts.date(), len(param_grid), n_days)
    candidate_wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())
    candidate_wfo.print_summary()

    log.info("dev: baseline WFO (incumbent configs/strategy.yaml params only)")
    baseline_wfo = WalkForwardOptimizer(fn, wfo_cfg, [{}]).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())
    baseline_wfo.print_summary()

    if not candidate_wfo.folds:
        raise SystemExit("dev window too short for a single WFO fold")

    final_params = dict(candidate_wfo.folds[-1].best_params)
    full = _full_window_run(fn, start_ts, end_ts, final_params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=args.max_concurrent_pairs)
    stress = _full_window_run(stress_fn, start_ts, end_ts, final_params)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    gates = {
        "wfo_go": candidate_wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(candidate_wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(candidate_wfo),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
    }

    result = {
        "phase": "dev",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "data": meta,
        "half_spread_bps": args.half_spread_bps,
        "max_concurrent_pairs": args.max_concurrent_pairs,
        "preflight": preflight,
        "param_grid_size": len(param_grid),
        "final_params": final_params,
        "incumbent_params": {k: base_cfg[k] for k in final_params} if final_params else {},
        "candidate_wfo": candidate_wfo.to_dict(),
        "baseline_wfo": baseline_wfo.to_dict(),
        "full_window": _strip(full),
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    _write_json(DEV_CHECKPOINT, result)
    return result


def phase_holdout(args) -> dict:
    """ONE evaluation of ONE configuration. Deliberately refuses to run before
    the dev phase has fixed that configuration."""
    if not DEV_CHECKPOINT.exists():
        raise SystemExit(f"missing {DEV_CHECKPOINT} — run --phase dev first; the "
                         "holdout may only be evaluated after the final config is fixed")
    dev = json.loads(DEV_CHECKPOINT.read_text(encoding="utf-8"))
    final_params = dev["final_params"]

    close, adv, _universe, candidate_pairs, meta = load_panels(args)
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)

    start_ts, end_ts = pd.Timestamp(args.dev_end), pd.Timestamp(args.holdout_end)
    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps,
        max_concurrent_pairs=args.max_concurrent_pairs)
    log.info("holdout: single run of %s over [%s, %s)", final_params, start_ts.date(), end_ts.date())
    metrics = _full_window_run(fn, start_ts, end_ts, final_params)
    mc = MonteCarloValidator(n_sims=500).run(metrics.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=args.half_spread_bps * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=args.max_concurrent_pairs)
    stress = _full_window_run(stress_fn, start_ts, end_ts, final_params)

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
        "phase": "holdout",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start_ts.date()), str(end_ts.date())],
        "data": meta,
        "params": final_params,
        "half_spread_bps": args.half_spread_bps,
        "metrics": _strip(metrics),
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
        "strategy": STRATEGY,
        "mode": "scanned_universe",
        "dev": json.loads(DEV_CHECKPOINT.read_text(encoding="utf-8")) if DEV_CHECKPOINT.exists() else None,
        "holdout": json.loads(HOLDOUT_CHECKPOINT.read_text(encoding="utf-8")) if HOLDOUT_CHECKPOINT.exists() else None,
    }
    _write_json(REPORT_JSON, out)
    log.info("wrote %s", REPORT_JSON)
    return out


# ── helpers ─────────────────────────────────────────────────────────────────

def _strip(metrics: dict) -> dict:
    """Drop the raw daily-return vector from a metrics dict before it is
    serialized — it is only needed to feed the Monte Carlo bootstrap and
    would otherwise dominate the JSON."""
    return {k: v for k, v in metrics.items() if k != "daily_returns"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)          # atomic: a killed process never leaves a half-written checkpoint
    log.info("checkpoint written: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["scan", "dev", "holdout", "report", "all"], default="all")
    parser.add_argument("--warmup-start", default="2016-06-01",
                        help="data start; must precede --start by >= coint_lookback_days trading days")
    parser.add_argument("--start", default="2018-01-01", help="development window start")
    parser.add_argument("--dev-end", default="2024-01-01",
                        help="development window end AND holdout window start")
    parser.add_argument("--holdout-end", default="2026-08-01", help="holdout window end")
    parser.add_argument("--half-spread-bps", type=float, default=DEFAULT_HALF_SPREAD_BPS)
    parser.add_argument("--max-concurrent-pairs", type=int, default=MAX_CONCURRENT_PAIRS)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    phases = ["scan", "dev", "holdout", "report"] if args.phase == "all" else [args.phase]
    for phase in phases:
        log.info("=== phase: %s ===", phase)
        {"scan": phase_scan, "dev": phase_dev,
         "holdout": phase_holdout, "report": phase_report}[phase](args)
    log.info("done")


if __name__ == "__main__":
    main()
