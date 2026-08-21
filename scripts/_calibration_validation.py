"""
Crash-resilient checkpoint runner for the SLIPPAGE CALIBRATION re-validation
(Step 3 of the calibrated-cost-model task): for each of the six already-
validated intraday signals, re-runs the SAME WFO + Monte Carlo + 2x-slippage-
stress pipeline as scripts/run_intraday_backtest.py, comparing the flat
2.0bps half-spread cost model ("old") against the calibrated per-symbol
half-spread ("new", scripts/calibrate_slippage_spreads.py ->
backtests/reports/calibrated_spreads.json), HOLDING PARAMS FIXED so the
comparison isolates the cost-model change (not a fresh grid search).

Two phases per signal, run and checkpointed SEPARATELY:

  "old"  — the already-established best/candidate params + baseline metrics
           under the flat cost model.
           * orb_vwap_regime / vwap_band_fade / vp_breakout: these are
             already on disk (backtests/reports/new_signals_report.json,
             a real run against the real 20-symbol universe) — loaded with
             ZERO recomputation.
           * sweep_reclaim / fvg_retest / orb_vwap: backtests/reports/
             intraday_backtest_report.md/.json's real-data entries for
             these three were lost (superseded on disk by a later
             `--demo` pipeline-validation run that overwrites the SAME
             path — see run_intraday_backtest.py's REPORT_PATH). There is
             no way to recover the exact prior candidate params other than
             re-running the same real per-fold-optimized grid search
             (param_grid=None, configs/param_grids.yaml's full grid) that
             originally produced them — this is NOT "searching for new
             params from scratch", it is recovering already-established-
             but-since-overwritten data via the documented standard
             invocation. Report this honestly.

  "new"  — SAME fixed single candidate (`old`'s candidate_params`, forced
           across every WFO fold via param_grid=[candidate_params] — no
           re-optimization) re-run under the calibrated per-symbol
           half_spread_bps_by_symbol override. Cheap (one candidate per
           fold instead of a whole grid) BY DESIGN — this is what makes
           the six-signal re-validation tractable without re-searching
           parameter space, per the task's explicit instruction.

Resilience (2026-08 lesson: this host has been interrupted mid-run by
reboots/connection drops before, losing hours of WFO progress — see
scripts/_resume_new_signals_validation.py's docstring for the prior
incident):
  1. Per-signal, per-phase checkpoint files
     (backtests/reports/_checkpoint_calib_<signal>.json) — same
     coarse-grained pattern as _resume_new_signals_validation.py.
  2. FINER-grained: every individual backtest_fn(start, end, params) call
     inside the WFO fold loop / stress test is memoized to disk
     (backtests/reports/_calib_cache/<hash>.json), keyed by
     (signal, cost_tag, start, end, params). A crash mid-fold-loop
     therefore loses at most the ONE in-flight call, not the whole
     multi-hour per-fold-optimized grid search — the process can simply
     be re-invoked and every already-computed fold/candidate replays from
     cache instantly.

Usage:
    python scripts/_calibration_validation.py <signal> old
    python scripts/_calibration_validation.py <signal> new
    python scripts/_calibration_validation.py <signal> both   # old then new
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_intraday_backtest import (  # noqa: E402
    GOAL_PATH,
    NEW_REPORT_JSON_PATH,
    NEW_SIGNALS,
    SIGNAL_WARMUP_DAYS,
    SIGNALS,
    STRATEGY_PATH,
    _load_bars_for_args,
    _load_yaml,
)
from python.backtest.intraday_engine import IntradayBacktestConfig  # noqa: E402
from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.optimize import (  # noqa: E402
    build_intraday_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    check_min_trades_gate,
    check_profit_factor_gate,
    load_param_grid,
    load_wfo_config,
    max_oos_drawdown_threshold,
    preflight_check,
    run_intraday_stress_test,
)
from python.backtest.walk_forward import WalkForwardOptimizer

ALL_SIGNALS = SIGNALS + NEW_SIGNALS  # ["sweep_reclaim","fvg_retest","orb_vwap","orb_vwap_regime","vwap_band_fade","vp_breakout"]
CHECKPOINT_DIR = Path("backtests/reports")
CACHE_DIR = Path("backtests/reports/_calib_cache")
CALIBRATED_SPREADS_PATH = Path("backtests/reports/calibrated_spreads.json")
START, END = "2025-08-01", "2026-07-01"


def _checkpoint_path(signal_name: str) -> Path:
    return CHECKPOINT_DIR / f"_checkpoint_calib_{signal_name}.json"


def _load_checkpoint(signal_name: str) -> dict:
    p = _checkpoint_path(signal_name)
    if p.exists():
        state = json.loads(p.read_text(encoding="utf-8"))
        state.setdefault("old_fixed", None)
        return state
    return {"signal": signal_name, "old": None, "old_fixed": None, "new": None}


def _save_phase_result(signal_name: str, phase_key: str, result: dict) -> None:
    """Writes ONE phase's result into the checkpoint, re-reading the
    CURRENT on-disk state first and updating only `phase_key` — NOT a
    naive load-at-start/save-at-end around the whole (old/old_fixed/new)
    lifecycle. This script is routinely invoked as multiple CONCURRENT
    background processes for the SAME signal (e.g. `old_fixed` and `new`
    both depend only on `old` and were launched in parallel once `old`
    was ready) — a load-once/save-once pattern would race: whichever
    process's in-memory snapshot (taken before the other's write) saves
    LAST silently clobbers the other's already-checkpointed phase. This
    happened once during development (`new`'s result was lost when
    `old_fixed` finished later and overwrote it) — read-merge-write on
    every single phase completion is the fix, not "don't run concurrently"
    (concurrency across phases/signals is the whole point of the
    background-job design)."""
    p = _checkpoint_path(signal_name)
    state = _load_checkpoint(signal_name)
    state[phase_key] = result
    p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def load_calibrated_spreads() -> dict[str, float]:
    """symbol -> calibrated median half-spread bps (scripts/
    calibrate_slippage_spreads.py's output). Symbols flagged `suspect` are
    EXCLUDED (fall back to the flat constant) rather than trusted at face
    value — same "flag, don't blindly apply" discipline as the calibration
    script itself."""
    if not CALIBRATED_SPREADS_PATH.exists():
        raise FileNotFoundError(
            f"{CALIBRATED_SPREADS_PATH} not found — run scripts/calibrate_slippage_spreads.py first"
        )
    payload = json.loads(CALIBRATED_SPREADS_PATH.read_text(encoding="utf-8"))
    out = {}
    for sym, s in payload["symbols"].items():
        if s.get("suspect"):
            continue
        out[sym] = float(s["median_bps"])
    return out


# ── disk-memoized backtest_fn wrapper (finer-grained crash resilience) ─────

def _cache_key_path(signal_name: str, cost_tag: str, start, end, params: dict) -> Path:
    raw = json.dumps(
        {"signal": signal_name, "cost": cost_tag, "start": start.isoformat(),
         "end": end.isoformat(), "params": params},
        sort_keys=True, default=str,
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{signal_name}__{cost_tag}__{digest}.json"


def _memoize(fn, signal_name: str, cost_tag: str):
    def wrapped(start, end, params):
        path = _cache_key_path(signal_name, cost_tag, start, end, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        result = fn(start, end, params)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, default=str), encoding="utf-8")
        return result
    return wrapped


# ── one signal/cost-model run (mirrors run_intraday_backtest.run_signal,
#    but with the memoized fn threaded through the WFO fold loop AND the
#    stress test, and an explicit cost_tag for cache-namespacing) ──────────

def run_signal_memoized(
    signal_name: str,
    bars_by_symbol: dict,
    data_label: str,
    start_ts,
    end_ts,
    param_grid: list[dict],
    cost_tag: str,
    half_spread_bps_by_symbol: dict[str, float] | None = None,
) -> dict:
    import pandas as pd

    base_cfg = _load_yaml(STRATEGY_PATH)[signal_name]
    goal = _load_yaml(GOAL_PATH)
    wfo_cfg = load_wfo_config(signal_name)
    n_bdays = len(pd.bdate_range(start_ts, end_ts))
    preflight = preflight_check(signal_name, base_cfg, param_grid, total_trading_days=n_bdays)

    warmup_days = SIGNAL_WARMUP_DAYS.get(signal_name, 1)
    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=half_spread_bps_by_symbol)
    raw_fn = build_intraday_backtest_fn(bars_by_symbol, signal_name, base_cfg, engine_cfg=engine_cfg, warmup_days=warmup_days)
    fn = _memoize(raw_fn, signal_name, cost_tag)

    t0 = time.time()
    print(f"[{signal_name}/{cost_tag}] WFO start: {len(param_grid)} candidate(s), window [{start_ts.date()}, {end_ts.date()}]", flush=True)
    wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
    print(f"[{signal_name}/{cost_tag}] WFO done in {time.time()-t0:.0f}s: "
          f"{wfo.passing_folds}/{wfo.total_folds} folds pass, OOS Sharpe mean {wfo.oos_sharpe_mean:+.3f}", flush=True)

    if not wfo.folds:
        return {"signal": signal_name, "decision": "SKIPPED",
                "reason": "window too short for a single WFO fold", "data_label": data_label, "cost_tag": cost_tag}

    candidate_params = dict(wfo.folds[-1].best_params)
    full_metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), candidate_params)
    mc_result = MonteCarloValidator(n_sims=500).run(full_metrics.get("daily_returns", []))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    intraday_goal = goal.get("intraday", {})
    min_trades = int(intraday_goal.get("min_trades_per_oos_fold", 100))
    min_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.3))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))

    t1 = time.time()
    stress_raw_fn = build_intraday_backtest_fn(
        bars_by_symbol, signal_name, base_cfg,
        engine_cfg=IntradayBacktestConfig(stress_slippage_multiplier=stress_mult, half_spread_bps_by_symbol=half_spread_bps_by_symbol),
        warmup_days=warmup_days,
    )
    stress_fn = _memoize(stress_raw_fn, signal_name, f"{cost_tag}_stress{stress_mult:g}x")
    stress_metrics = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), candidate_params)
    print(f"[{signal_name}/{cost_tag}] stress test done in {time.time()-t1:.0f}s: net_pnl={stress_metrics.get('total_net_pnl', 0):.2f}", flush=True)

    gates = {
        "wfo_go": wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(wfo),
        "min_trades_per_oos_fold": check_min_trades_gate(wfo, min_trades),
        "cost_adjusted_profit_factor": check_profit_factor_gate(wfo, min_pf),
        "monte_carlo_p5_sharpe": mc_result.sharpe.p5 >= min_p5,
        f"stress_slippage_{stress_mult:g}x_net_positive": stress_metrics["total_net_pnl"] > 0,
    }
    overall_pass = all(gates.values())

    return {
        "signal": signal_name,
        "cost_tag": cost_tag,
        "decision": "GO" if overall_pass else "NO-GO",
        "data_label": data_label,
        "window": f"{start_ts.date()} .. {end_ts.date()}",
        "n_symbols": len(bars_by_symbol),
        "wfo_folds": wfo.total_folds,
        "wfo_pass_ratio": wfo.pass_ratio,
        "oos_sharpe_mean": wfo.oos_sharpe_mean,
        "candidate_params": candidate_params,
        "full_window_metrics": {k: v for k, v in full_metrics.items() if k != "daily_returns"},
        "stress_metrics": {k: v for k, v in stress_metrics.items() if k != "daily_returns"},
        "mc_p5_sharpe": mc_result.sharpe.p5,
        "gates": gates,
        "sample_size_check": preflight,
    }


def _load_existing_new_signal_report(signal_name: str) -> dict | None:
    if not NEW_REPORT_JSON_PATH.exists():
        return None
    payload = json.loads(NEW_REPORT_JSON_PATH.read_text(encoding="utf-8"))
    for r in payload.get("results", []):
        if r.get("signal") == signal_name:
            r = dict(r)
            r.setdefault("cost_tag", "flat_reused_from_new_signals_report")
            return r
    return None


def run_old_phase(signal_name: str, bars_by_symbol: dict, data_label, start_ts, end_ts) -> dict:
    if signal_name in NEW_SIGNALS:
        existing = _load_existing_new_signal_report(signal_name)
        if existing is not None:
            print(f"[{signal_name}/old] reusing already-established real-data result from "
                  f"{NEW_REPORT_JSON_PATH} (decision={existing['decision']}, "
                  f"candidate_params={existing['candidate_params']}) — zero recomputation", flush=True)
            return existing
        print(f"[{signal_name}/old] WARNING: no existing entry in {NEW_REPORT_JSON_PATH} — recomputing from scratch", flush=True)

    full_grid = load_param_grid(signal_name)
    return run_signal_memoized(signal_name, bars_by_symbol, data_label, start_ts, end_ts, full_grid, cost_tag="flat")


def run_old_fixed_phase(signal_name: str, old_result: dict, bars_by_symbol: dict, data_label, start_ts, end_ts) -> dict:
    """SAME fixed single candidate as `old`/`new`, but at the FLAT cost
    model — this (not the per-fold-reoptimized `old`) is the correct
    apples-to-apples baseline for isolating the cost-model swap: `old`
    (per-fold reoptimization, kept for continuity with the original
    reports) picks a potentially DIFFERENT best-IS candidate per fold,
    so comparing it directly against `new`'s fixed-candidate run would
    conflate "removed per-fold reoptimization" with "changed cost model".
    `old_fixed` vs `new` differ in EXACTLY one thing: half_spread_bps_by_symbol."""
    candidate_params = dict(old_result["candidate_params"])
    return run_signal_memoized(
        signal_name, bars_by_symbol, data_label, start_ts, end_ts,
        param_grid=[candidate_params], cost_tag="flat_fixed",
    )


def run_new_phase(signal_name: str, old_result: dict, bars_by_symbol: dict, data_label, start_ts, end_ts,
                   calibrated_spreads: dict[str, float]) -> dict:
    candidate_params = dict(old_result["candidate_params"])
    return run_signal_memoized(
        signal_name, bars_by_symbol, data_label, start_ts, end_ts,
        param_grid=[candidate_params], cost_tag="calibrated",
        half_spread_bps_by_symbol=calibrated_spreads,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signal", choices=ALL_SIGNALS)
    parser.add_argument("phase", choices=["old", "old_fixed", "new", "both"])
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    signal_name = args.signal
    state = _load_checkpoint(signal_name)

    class _Args:
        demo = False
        full_grid = False
    ns = _Args()
    ns.start, ns.end = args.start, args.end

    print(f"Loading bars for {signal_name} [{args.start}, {args.end}]...", flush=True)
    bars_by_symbol, data_label, start_ts, end_ts = _load_bars_for_args(ns)
    print(f"Loaded {len(bars_by_symbol)} symbols.", flush=True)

    if args.phase in ("old", "both"):
        if state["old"] is not None:
            print(f">>> old phase already checkpointed: {state['old']['decision']} — skipping", flush=True)
        else:
            print(f"\n>>> [{signal_name}] OLD (flat 2.0bps) phase starting...", flush=True)
            r = run_old_phase(signal_name, bars_by_symbol, data_label, start_ts, end_ts)
            state["old"] = r
            _save_phase_result(signal_name, "old", r)
            print(f">>> OLD phase done: {r['decision']} (checkpointed)", flush=True)

    if args.phase in ("old_fixed", "both"):
        if state["old"] is None:
            raise SystemExit(f"{signal_name}: old phase not checkpointed yet — run `... {signal_name} old` first")
        if state["old_fixed"] is not None:
            print(f">>> old_fixed phase already checkpointed: {state['old_fixed']['decision']} — skipping", flush=True)
        else:
            print(f"\n>>> [{signal_name}] OLD_FIXED (flat cost, same fixed candidate as new) phase starting, "
                  f"candidate_params={state['old']['candidate_params']}...", flush=True)
            r = run_old_fixed_phase(signal_name, state["old"], bars_by_symbol, data_label, start_ts, end_ts)
            _save_phase_result(signal_name, "old_fixed", r)
            print(f">>> OLD_FIXED phase done: {r['decision']} (checkpointed)", flush=True)

    if args.phase in ("new", "both"):
        if state["old"] is None:
            raise SystemExit(f"{signal_name}: old phase not checkpointed yet — run `... {signal_name} old` first")
        if state["new"] is not None:
            print(f">>> new phase already checkpointed: {state['new']['decision']} — skipping", flush=True)
        else:
            calibrated_spreads = load_calibrated_spreads()
            print(f"\n>>> [{signal_name}] NEW (calibrated per-symbol) phase starting, "
                  f"fixed candidate_params={state['old']['candidate_params']}...", flush=True)
            r = run_new_phase(signal_name, state["old"], bars_by_symbol, data_label, start_ts, end_ts, calibrated_spreads)
            _save_phase_result(signal_name, "new", r)
            print(f">>> NEW phase done: {r['decision']} (checkpointed)", flush=True)

    print(f"\nDONE ({signal_name}, phase={args.phase}). Checkpoint: {_checkpoint_path(signal_name)}", flush=True)


if __name__ == "__main__":
    main()
