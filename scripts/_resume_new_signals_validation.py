"""
Crash-resilient checkpoint runner for the NEW_SIGNALS real-data validation
(python scripts/run_intraday_backtest.py --signal new). This host has been
interrupted (reboot, then Cursor closed) TWICE mid-run, each time losing
hours of progress because the normal script only writes its report at the
very end. This wrapper checkpoints after EVERY individual backtest run
(the main per-fold-optimized WFO run, and each full-grid-search candidate
separately) to backtests/reports/_checkpoint_<signal>.json, so a third
interruption loses at most one candidate's worth of computation (~1-30 min)
instead of an entire signal's multi-hour run.

Usage:
    python scripts/_resume_new_signals_validation.py <signal_name>

Resumable: if backtests/reports/_checkpoint_<signal>.json already has a
"main" entry, it is reused instead of recomputed; same for each entry in
"full_grid" (checked by candidate params). Safe to re-invoke after a crash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import argparse

from run_intraday_backtest import (  # noqa: E402
    NEW_SIGNALS,
    _load_bars_for_args,
    run_signal,
)
from python.backtest.optimize import load_param_grid  # noqa: E402

CHECKPOINT_DIR = Path("backtests/reports")


def _checkpoint_path(signal_name: str) -> Path:
    return CHECKPOINT_DIR / f"_checkpoint_{signal_name}.json"


def _load_checkpoint(signal_name: str) -> dict:
    p = _checkpoint_path(signal_name)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"signal": signal_name, "main": None, "full_grid": []}


def _save_checkpoint(signal_name: str, state: dict) -> None:
    p = _checkpoint_path(signal_name)
    p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("signal", choices=NEW_SIGNALS)
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    args = parser.parse_args()
    # run_signal/_load_bars_for_args expect an argparse.Namespace with these
    # attributes (mirrors run_intraday_backtest.py's own --demo/--full-grid
    # flags, both unused here since this is always a real-data, non-full-grid
    # single-run-at-a-time invocation).
    args.demo = False
    args.full_grid = False

    signal_name = args.signal
    state = _load_checkpoint(signal_name)

    print(f"Loading bars for {signal_name} [{args.start}, {args.end}]...", flush=True)
    bars_by_symbol, data_label, start_ts, end_ts = _load_bars_for_args(args)
    print(f"Loaded {len(bars_by_symbol)} symbols.", flush=True)

    if state["main"] is None:
        print(f"\n>>> Running MAIN per-fold-optimized WFO for {signal_name}...", flush=True)
        r = run_signal(signal_name, args, bars_by_symbol, data_label, start_ts, end_ts)
        state["main"] = r
        _save_checkpoint(signal_name, state)
        print(f">>> MAIN done: {r['decision']} (checkpointed)", flush=True)
    else:
        print(f">>> MAIN already checkpointed: {state['main']['decision']} — skipping", flush=True)

    main_result = state["main"]
    worth_scrutiny = (
        main_result["decision"] != "SKIPPED"
        and (main_result["decision"] == "GO" or main_result.get("oos_sharpe_mean", 0.0) > 0)
    )

    if not worth_scrutiny:
        state["full_grid_skipped_reason"] = (
            "main per-fold-optimized WFO run was decisively negative "
            f"(decision={main_result['decision']}, "
            f"OOS Sharpe mean {main_result.get('oos_sharpe_mean', 0.0):+.3f} <= 0) — "
            "a full grid search could only confirm, not overturn, that result"
        )
        _save_checkpoint(signal_name, state)
        print(f">>> Full grid search skipped: {state['full_grid_skipped_reason']}", flush=True)
        print("\nDONE.", flush=True)
        return

    full_grid = load_param_grid(signal_name)
    done_params = {json.dumps(g["params"], sort_keys=True) for g in state["full_grid"]}
    for i, candidate in enumerate(full_grid):
        key = json.dumps(candidate, sort_keys=True)
        if key in done_params:
            print(f">>> [{i + 1}/{len(full_grid)}] {candidate} already checkpointed — skipping", flush=True)
            continue
        print(f"\n>>> [{i + 1}/{len(full_grid)}] running full-grid candidate {candidate}...", flush=True)
        r = run_signal(signal_name, args, bars_by_symbol, data_label, start_ts, end_ts,
                        param_grid=[candidate], quiet=True)
        r["params"] = candidate
        state["full_grid"].append(r)
        _save_checkpoint(signal_name, state)
        print(f"      -> {r['decision']} (OOS Sharpe {r.get('oos_sharpe_mean', 0.0):+.3f}, "
              f"pass_ratio {r.get('wfo_pass_ratio', 0.0):.0%}) [checkpointed]", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
