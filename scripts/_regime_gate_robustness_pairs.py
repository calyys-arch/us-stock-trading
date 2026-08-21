"""
Robustness / sensitivity round for `backtests/reports/regime_gate_report.md`'s
strongest result: `pairs_trading` `pairs_lowfreq_entry_z_4`
(`entry_z=4.0, exit_z=0.5, half_life_multiplier_max_hold=3.0`), gated by
`python/analytics/trend_efficiency_gate.shifted_entry_gate` at
`window=20, reference_window=252`. Full writeup and verdict:
`backtests/reports/regime_gate_robustness_report.md`. This script is only the
compute step (checkpointed, resumable); it does not decide anything.

MANDATE (task brief §0) — read before touching anything below: every
distinct historical window this system has data for has now been used at
least once in this campaign, so there is no virgin holdout window left. This
script does NOT grid-search hunting for a cell that clears the gate and
report only that cell — it runs ONE small, pre-declared grid, justified by
standard technical-analysis convention BEFORE any cell was computed, and
reports EVERY cell (see the report for the full table). It also does not
move the WFO fold boundary to a spot that happens to separate favorable
years into favorable folds — the fold-structure alternatives below are
named, independently-justifiable conventions this repo's own code and
config already document elsewhere, chosen and written up before being run.

PART A — classifier parameter grid (16 cells, fixed BEFORE this script ran)
----------------------------------------------------------------------------
  window            in {10, 14, 20, 50}         (trading days)
  reference_window  in {126, 189, 252, 378}     (trading days)

Neither axis was chosen by looking at how it performs:
  - window: 10 (a common short "recent price action" length, e.g. ROC-10),
    14 (the single most standard TA lookback -- RSI, ADX, ATR all default to
    14), 20 (one trading month; this classifier's own committed value, and
    the Bollinger-Band standard), 50 (the standard "medium-term" moving
    average length).
  - reference_window: 126 (~6 trading months), 189 (~9 trading months), 252
    (~1 trading year -- this classifier's own committed value, and the same
    convention `regime.py`'s `min_train` already uses), 378 (~18 trading
    months). These are the four round-number month-multiples named in this
    task's own brief.

For each of the 16 (window, reference_window) cells: build the gated entry
signal from SPY's close (unchanged classifier code, unchanged frozen
strategy params), then compute (a) the full 2018-2026 history's continuous
metrics (Sharpe, cost-adjusted PF, drawdown, Monte Carlo p5 Sharpe,
`configs/goal.yaml` verdict) and (b) the standard `pairs_trading` WFO fold
pass ratio (`configs/goal.yaml`'s existing `is_days=1008, oos_days=126,
step_days=126` override -- WFO Convention A below). The ungated baseline is
identical for every cell (it does not depend on the classifier at all) and
is loaded directly from `backtests/reports/_regime_gate_phase2_cache/
pairs_ungated.json` rather than recomputed.

PART B — WFO fold-structure convention comparison (3 conventions, at the
anchor cell window=20/reference_window=252 only)
----------------------------------------------------------------------------
Tests whether Phase 2's WFO NO-GO (fold pass ratio 50.0%, gated) is an
artifact of one particular fold boundary rather than a property of the
classifier itself. Three STANDARD conventions, named and justified before
running any of them:

  A. Rolling fixed-window (the existing `configs/goal.yaml` override for
     `pairs_trading`, already reported in `regime_gate_report.md` §2.2):
     is_days=1008 (~4y), oos_days=126 (~6mo), step_days=126, sliding IS
     window. Reused directly from the Phase 2 cache, not recomputed.
  B. Anchored/expanding window: SAME is/oos/step lengths as (A), but the IS
     window's START stays fixed at the study start and its END grows by
     step_days each fold (`WFOConfig(anchored=True)`, added to
     `python/backtest/walk_forward.py` for this task) -- the other of the
     two textbook walk-forward conventions (rolling vs. anchored/expanding;
     e.g. Pardo, "The Evaluation and Optimization of Trading Strategies",
     ch. 5), not a fold rule invented to favor this result.
  C. Rolling fixed-window at this repo's own DEFAULT in-sample length: is_
     days=504 (~2y -- `WFOConfig.is_days`'s own class-level default and
     comment, "~2 trading years in-sample", and the value every OTHER
     strategy in `configs/goal.yaml`'s wfo block uses before any
     per-strategy override), oos_days=126, step_days=126. `pairs_trading`
     specifically overrides this to 1008 in production because
     `coint_lookback_days=252` warmup would otherwise eat too much of a
     504-day IS window (see `python/backtest/optimize.py`'s own docstring),
     but 504 is still a live, already-documented, non-fabricated convention
     in this exact codebase -- not a boundary chosen to move 2022/2008 into
     favorable folds.

PART C -- fold-order-randomization / block bootstrap (substitute for a
holdout that does not exist)
----------------------------------------------------------------------------
`python/analytics/robustness_stats.py`'s two functions, applied to
Convention A's already-computed fold results (from the Phase 2 cache) and
full-window daily returns (anchor cell) -- see that module's docstring for
exactly what each does and does NOT prove.

Usage:
    python scripts/_regime_gate_robustness_pairs.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("regime_gate_robustness_pairs")

from python.analytics.robustness_stats import bootstrap_fold_pass_ratio, moving_block_bootstrap_sharpe
from python.analytics.trend_efficiency_gate import shifted_entry_gate
from python.backtest.optimize import build_pairs_scan_backtest_fn, load_wfo_config
from python.backtest.pairs_scan_engine import DEFAULT_HALF_SPREAD_BPS, MAX_CONCURRENT_PAIRS
from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig
from run_pairs_scan_backtest import _load_schedule, _strategy_cfg, load_panels

from _regime_gate_phase2_pairs import CACHE_DIR as PHASE2_CACHE_DIR
from _regime_gate_phase2_pairs import (
    CONFIG_NAME,
    FROZEN_PARAMS,
    FULL_END,
    FULL_START,
    _full_window_result,
    _panel_args,
)

CACHE_DIR = Path("backtests/reports/_regime_gate_robustness_cache")
REPORT_JSON = Path("backtests/reports/regime_gate_robustness_pairs.json")

# ── Part A: pre-declared grid (16 cells) ────────────────────────────────────
GRID_WINDOW = [10, 14, 20, 50]
GRID_REFERENCE_WINDOW = [126, 189, 252, 378]
ANCHOR_CELL = (20, 252)  # regime_gate_report.md's originally-tested cell

MIN_PASS_FOLDS_RATIO = 0.60  # configs/goal.yaml wfo.min_pass_folds_ratio


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("checkpoint written: %s", path)


def _run_cell(fn_builder, spy_close: pd.Series, window: int, reference_window: int, wfo_cfg: WFOConfig) -> dict:
    entry_gate = shifted_entry_gate(spy_close, window=window, reference_window=reference_window)
    fn_gated = fn_builder(entry_gate)

    wfo = WalkForwardOptimizer(fn_gated, wfo_cfg, [FROZEN_PARAMS]).run(
        pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime())

    full_window = _full_window_result(fn_gated, FROZEN_PARAMS)

    return {
        "window": window,
        "reference_window": reference_window,
        "n_days_gate_on": int(entry_gate.sum()),
        "n_days_total": len(entry_gate),
        "wfo": wfo.to_dict(),
        "full_window_2018_2026": full_window,
    }


def main() -> None:
    close, adv, _universe, candidate_pairs, meta = load_panels(_panel_args())
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)
    log.info("panel/schedule loaded: %s", meta)

    spy_close = close["SPY"].dropna()

    def fn_builder(entry_gate):
        return build_pairs_scan_backtest_fn(
            close, adv, schedule, base_cfg,
            half_spread_bps=DEFAULT_HALF_SPREAD_BPS, max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
            entry_gate=entry_gate,
        )

    wfo_cfg_a = load_wfo_config("pairs_trading")  # Convention A: is=1008/oos=126/step=126, rolling
    log.info("Convention A (rolling, production override): %s", wfo_cfg_a)

    # ── Part A: 16-cell grid, WFO Convention A + full-window per cell ───────
    grid_cells = []
    for window in GRID_WINDOW:
        for reference_window in GRID_REFERENCE_WINDOW:
            ckpt = CACHE_DIR / f"grid_w{window}_r{reference_window}.json"
            if ckpt.exists():
                log.info("resuming grid cell (window=%d, reference_window=%d) from %s",
                         window, reference_window, ckpt)
                grid_cells.append(json.loads(ckpt.read_text(encoding="utf-8")))
                continue
            log.info("=== grid cell: window=%d, reference_window=%d ===", window, reference_window)
            cell = _run_cell(fn_builder, spy_close, window, reference_window, wfo_cfg_a)
            grid_cells.append(cell)
            _write_json(ckpt, cell)

    # ── Part B: WFO fold-structure convention comparison, anchor cell only ──
    anchor_window, anchor_ref = ANCHOR_CELL
    entry_gate_anchor = shifted_entry_gate(spy_close, window=anchor_window, reference_window=anchor_ref)
    fn_gated_anchor = fn_builder(entry_gate_anchor)
    fn_ungated = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=DEFAULT_HALF_SPREAD_BPS, max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
    )

    conventions = {
        "A_rolling_production_1008_126_126": WFOConfig(is_days=1008, oos_days=126, step_days=126, anchored=False),
        "B_anchored_expanding_1008_126_126": WFOConfig(is_days=1008, oos_days=126, step_days=126, anchored=True),
        "C_rolling_repo_default_504_126_126": WFOConfig(is_days=504, oos_days=126, step_days=126, anchored=False),
    }

    fold_convention_results: dict = {}
    phase2_ungated_wfo = json.loads((PHASE2_CACHE_DIR / "pairs_ungated.json").read_text(encoding="utf-8"))["wfo"]
    phase2_gated_wfo = json.loads((PHASE2_CACHE_DIR / "pairs_gated.json").read_text(encoding="utf-8"))["wfo"]

    for conv_name, cfg in conventions.items():
        ckpt = CACHE_DIR / f"convention_{conv_name}.json"
        if ckpt.exists():
            log.info("resuming convention %s from %s", conv_name, ckpt)
            fold_convention_results[conv_name] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue
        log.info("=== WFO convention: %s ===", conv_name)
        if conv_name == "A_rolling_production_1008_126_126":
            # Already computed by Phase 2 -- reuse verbatim rather than
            # rerunning the exact same backtest a second time.
            result = {"gated": phase2_gated_wfo, "ungated": phase2_ungated_wfo}
        else:
            gated_wfo = WalkForwardOptimizer(fn_gated_anchor, cfg, [FROZEN_PARAMS]).run(
                pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime())
            ungated_wfo = WalkForwardOptimizer(fn_ungated, cfg, [FROZEN_PARAMS]).run(
                pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime())
            result = {"gated": gated_wfo.to_dict(), "ungated": ungated_wfo.to_dict()}
        fold_convention_results[conv_name] = result
        _write_json(ckpt, result)

    # ── Part C: fold-order-randomization + block bootstrap ──────────────────
    def _fold_passes(wfo_dict: dict) -> list[bool]:
        return [bool(f["oos_pass"]) for f in wfo_dict["folds"]]

    bootstrap_results = {
        "fold_pass_ratio_bootstrap": {
            "gated_convention_A": bootstrap_fold_pass_ratio(
                _fold_passes(phase2_gated_wfo), min_pass_ratio=MIN_PASS_FOLDS_RATIO),
            "ungated_convention_A": bootstrap_fold_pass_ratio(
                _fold_passes(phase2_ungated_wfo), min_pass_ratio=MIN_PASS_FOLDS_RATIO),
        },
    }

    # Aggregate OOS daily returns (Convention A, anchor cell) for the moving
    # block bootstrap -- same series `regime_gate_phase2_pairs.py` used for
    # its own aggregate-OOS Monte Carlo, reused here rather than recomputed.
    def _aggregate_oos_returns(wfo_dict: dict) -> list[float]:
        out: list[float] = []
        for f in wfo_dict["folds"]:
            out.extend(f["oos_metrics"].get("daily_returns", []))
        return out

    gated_oos_returns = _aggregate_oos_returns(phase2_gated_wfo)
    ungated_oos_returns = _aggregate_oos_returns(phase2_ungated_wfo)
    bootstrap_results["moving_block_bootstrap_sharpe_aggregate_oos"] = {
        "gated_convention_A": moving_block_bootstrap_sharpe(gated_oos_returns, block_size=21, n_boot=2000),
        "ungated_convention_A": moving_block_bootstrap_sharpe(ungated_oos_returns, block_size=21, n_boot=2000),
    }

    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Robustness/sensitivity round for regime_gate_report.md's "
            "pairs_trading gated result. Part A: 16-cell classifier grid "
            "(window x reference_window), every cell reported. Part B: 3 "
            "WFO fold-structure conventions at the anchor cell (20, 252). "
            "Part C: fold-order-randomization bootstrap + moving block "
            "bootstrap Sharpe, substitute checks given no virgin holdout "
            "window remains (see backtests/reports/"
            "regime_gate_robustness_report.md for the full honest reading)."
        ),
        "anchor_cell": {"window": anchor_window, "reference_window": anchor_ref},
        "grid": {
            "window_values": GRID_WINDOW,
            "reference_window_values": GRID_REFERENCE_WINDOW,
            "n_cells": len(GRID_WINDOW) * len(GRID_REFERENCE_WINDOW),
            "cells": grid_cells,
        },
        "wfo_fold_conventions": fold_convention_results,
        "bootstrap": bootstrap_results,
    }
    _write_json(REPORT_JSON, out)
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
