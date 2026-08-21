"""
Crash-resilient checkpoint runner for the orb_vwap RESCUE investigation
(backtests/reports/orb_vwap_rescue_report.md).

`orb_vwap` is the only signal in this system with a genuinely positive raw
edge (flat-cost mean OOS Sharpe +1.407, 62% WFO pass ratio) that is NO-GO
purely on cost drag (calibrated cost-adjusted PF 0.871 vs the 1.3 gate).
This script tests, one lever at a time, whether the cost-to-edge ratio can
be attacked hard enough to clear `configs/goal.yaml`'s gates UNCHANGED:

  Lever 1  universe   — restrict to the calibrated tight-spread names only
                        (the calibration report's §(d) blames trading
                        broadly across all 20 for the cost blowup).
  Lever 2  entry cap  — cap entries per symbol per session (an opening-range
                        break should be a once-per-session event; the
                        as-shipped signal re-fires on every re-cross).
  Lever 3  stop       — OR extreme +/- k*ATR instead of the raw OR extreme.
  Lever 4  target     — an R-multiple profit target instead of target=None.

Discipline this script enforces (see the report for the full write-up):

  * Gate thresholds are read from configs/goal.yaml UNCHANGED. Nothing here
    writes to configs/strategy.yaml and nothing flips `auto_execute` — a GO
    is a promotion CANDIDATE for human review only (see
    python/backtest/promotion.py's `_FORBIDDEN_WRITE_KEYS` path).
  * HOLDOUT SEPARATION. Every development/tuning config runs on the DEV
    window only (`--window dev`). The last two months of available 1-minute
    history are the FINAL HOLDOUT (`--window holdout`) and are evaluated
    exactly ONCE, for the single best dev configuration, at the end.
  * Headline verdicts use the CALIBRATED per-symbol half-spreads
    (backtests/reports/calibrated_spreads.json); `--cost flat` exists only
    for secondary context.
  * Every candidate is FIXED across every WFO fold (param_grid=[candidate],
    no per-fold re-optimization), so each row of the report's lever table is
    one config evaluated consistently — the same `old_fixed`/`new`
    discipline as scripts/_calibration_validation.py.

Resilience (same prior art as scripts/_calibration_validation.py /
scripts/_resume_new_signals_validation.py — a prior session lost days of
work to non-durable background processes):
  1. Per-config checkpoint JSON in `backtests/reports/_orb_rescue/`.
  2. Every individual backtest_fn(start, end, params) call inside the WFO
     fold loop AND the stress re-run is memoized to disk under
     `backtests/reports/_orb_rescue_cache/`, keyed by
     (config_id ingredients, start, end, params). A crash loses at most the
     one in-flight fold, and re-invoking the same command replays
     everything already computed instantly.

Usage:
    python scripts/_orb_vwap_rescue.py --list
    python scripts/_orb_vwap_rescue.py A0_asshipped_full20
    python scripts/_orb_vwap_rescue.py B1_tight10 B2_tight6
    python scripts/_orb_vwap_rescue.py HOLDOUT_best      # ONCE, at the end
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd  # noqa: E402

from run_intraday_backtest import (  # noqa: E402
    GOAL_PATH,
    STRATEGY_PATH,
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
    load_wfo_config,
    max_oos_drawdown_threshold,
    preflight_check,
)
from python.backtest.walk_forward import WalkForwardOptimizer  # noqa: E402

SIGNAL = "orb_vwap"
CHECKPOINT_DIR = Path("backtests/reports/_orb_rescue")
CACHE_DIR = Path("backtests/reports/_orb_rescue_cache")
CALIBRATED_SPREADS_PATH = Path("backtests/reports/calibrated_spreads.json")

# ── windows ─────────────────────────────────────────────────────────────────
# Available cached 1-minute history: 2025-07-02 .. 2026-07-31.
# DEV = everything up to 2026-06-01 (the period every prior orb_vwap report
# already looked at, minus the holdout). HOLDOUT = the final two months,
# untouched until the single best dev config is frozen.
DEV_START, DEV_END = "2025-08-01", "2026-06-01"
HOLDOUT_START, HOLDOUT_END = "2026-06-01", "2026-08-01"

# ── universes (calibrated median half-spread, bps, from
#    backtests/reports/slippage_calibration_report.md §(a)) ────────────────
FULL20 = None  # None = whatever configs/universe.yaml's fixed universe holds
TIGHT10 = ["AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC", "META", "AVGO", "AMD", "QCOM"]
TIGHT6 = ["AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC"]


@dataclass
class RescueConfig:
    config_id: str
    lever: str                       # which lever this row of the report isolates
    description: str
    params: dict                     # orb_vwap signal params (fixed across every fold)
    universe: list[str] | None = None
    window: str = "dev"              # "dev" | "holdout"
    cost: str = "calibrated"         # "calibrated" | "flat"
    # Free-text marker for the state of the CODE this config was run under.
    # Included in the memo-cache key so an as-shipped run and a post-fix run
    # of identical params can never collide in the cache.
    code_variant: str = "levered"
    extra: dict = field(default_factory=dict)


# ── the configuration registry (this IS the "how many configs did you
#    evaluate" count the report has to declare) ─────────────────────────────
BASE_PARAMS = {"or_minutes": 5, "vwap_side_filter": True}


def _p(**overrides) -> dict:
    out = dict(BASE_PARAMS)
    out.update(overrides)
    return out


CONFIGS: dict[str, RescueConfig] = {}


def _add(cfg: RescueConfig) -> None:
    CONFIGS[cfg.config_id] = cfg


# Lever 0 — baselines.
_add(RescueConfig(
    "A0_asshipped_full20", "baseline",
    "as-shipped orb_vwap (pre-fix code), full 20-symbol universe, DEV window",
    _p(), FULL20, code_variant="asshipped",
))
_add(RescueConfig(
    "A1_stopfix_full20", "stop-side fix",
    "orb_vwap with the gap-trap inverted-stop correctness fix, full 20 symbols",
    _p(), FULL20,
))

# Lever 1 — tight-spread universe subset.
_add(RescueConfig(
    "B1_tight10", "1: universe",
    "tight-spread top-10 (calibrated half-spread <= 1.89bps)",
    _p(), TIGHT10,
))
_add(RescueConfig(
    "B2_tight6", "1: universe",
    "tight-spread top-6 (calibrated half-spread <= 1.00bps)",
    _p(), TIGHT6,
))

# Lever 3 — ATR stop buffer, tested BEFORE the entry cap (out of the order
# the levers are numbered in) for an evidence-driven reason: with the Lever-0
# stop-side fix in place, a trap fade's structural stop is the breakout bar's
# own extreme, which is only a few ticks from the entry. Whatever the entry
# cap or the target does is measured on top of THAT stop scale, so the stop
# scale has to be settled first or every later lever is read against a
# pathologically tight risk unit. Universe fixed at TIGHT10 (see the report's
# Lever 1 row for why TIGHT6's slightly better PF was not worth the
# min_trades_per_oos_fold headroom it gives up).
_add(RescueConfig(
    "D1_atr025", "3: stop buffer",
    "stop = structural extreme -/+ 0.25 * ATR14",
    _p(stop_atr_buffer_mult=0.25), TIGHT10,
))
_add(RescueConfig(
    "D2_atr050", "3: stop buffer",
    "stop = structural extreme -/+ 0.50 * ATR14",
    _p(stop_atr_buffer_mult=0.50), TIGHT10,
))
_add(RescueConfig(
    "D3_atr100", "3: stop buffer",
    "stop = structural extreme -/+ 1.00 * ATR14",
    _p(stop_atr_buffer_mult=1.00), TIGHT10,
))
# D4 exists to check whether the D1->D3 improvement is a real "stop was too
# tight for 1-minute noise" effect or just the monotone artifact of widening
# the stop until it stops being hit (which also shrinks risk-based position
# size, and would keep improving forever). A plateau/reversal here is the
# informative outcome, not a better number.
_add(RescueConfig(
    "D4_atr200", "3: stop buffer",
    "stop = structural extreme -/+ 2.00 * ATR14 (monotonicity check)",
    _p(stop_atr_buffer_mult=2.00), TIGHT10,
))

# Lever 2 — entries-per-session cap, on top of the best Lever-3 buffer
# (BEST_BUFFER is set from the D-row results before these are launched).
BEST_BUFFER = 1.00
_add(RescueConfig(
    "C1_cap1", "2: entry cap",
    "first qualifying break per symbol per session only",
    _p(stop_atr_buffer_mult=BEST_BUFFER, max_entries_per_session=1), TIGHT10,
))
_add(RescueConfig(
    "C2_cap2", "2: entry cap",
    "at most 2 entries per symbol per session",
    _p(stop_atr_buffer_mult=BEST_BUFFER, max_entries_per_session=2), TIGHT10,
))

# Lever 4 — R-multiple profit target, on top of the best Lever-2 cap
# (BEST_CAP is set from the C-row results before these are launched).
BEST_CAP = 1
_add(RescueConfig(
    "E1_r1", "4: target",
    "1.0R profit target",
    _p(stop_atr_buffer_mult=BEST_BUFFER, max_entries_per_session=BEST_CAP, target_r_multiple=1.0), TIGHT10,
))
_add(RescueConfig(
    "E2_r2", "4: target",
    "2.0R profit target",
    _p(stop_atr_buffer_mult=BEST_BUFFER, max_entries_per_session=BEST_CAP, target_r_multiple=2.0), TIGHT10,
))
_add(RescueConfig(
    "E3_r3", "4: target",
    "3.0R profit target",
    _p(stop_atr_buffer_mult=BEST_BUFFER, max_entries_per_session=BEST_CAP, target_r_multiple=3.0), TIGHT10,
))


# Secondary context only (NOT a verdict, NOT part of the lever search): the
# frozen best configuration re-run under the old flat 2.0bps half-spread
# assumption. Its only job is to answer "is what is left after the four
# levers still a COST problem, or has it become an edge problem?".
_add(RescueConfig(
    "F1_best_flat", "context: flat cost",
    "frozen best config (E2_r2) under the flat 2.0bps assumption instead of calibrated spreads",
    _p(stop_atr_buffer_mult=BEST_BUFFER, max_entries_per_session=BEST_CAP, target_r_multiple=2.0),
    TIGHT10, cost="flat",
))


def register_holdout(params: dict, universe: list[str] | None, note: str) -> RescueConfig:
    cfg = RescueConfig(
        "HOLDOUT_best", "FINAL HOLDOUT",
        f"single best DEV configuration, evaluated ONCE on the untouched holdout ({note})",
        params, universe, window="holdout",
    )
    _add(cfg)
    return cfg


# ── cost model ──────────────────────────────────────────────────────────────

def load_calibrated_spreads() -> dict[str, float]:
    """symbol -> calibrated median half-spread bps. Symbols the calibration
    script flagged `suspect` are EXCLUDED (they fall back to the flat
    constant) — identical handling to scripts/_calibration_validation.py."""
    payload = json.loads(CALIBRATED_SPREADS_PATH.read_text(encoding="utf-8"))
    return {sym: float(s["median_bps"]) for sym, s in payload["symbols"].items() if not s.get("suspect")}


# ── disk memoization ────────────────────────────────────────────────────────

def _cache_path(tag: str, start, end, params: dict) -> Path:
    raw = json.dumps({"tag": tag, "start": start.isoformat(), "end": end.isoformat(), "params": params},
                     sort_keys=True, default=str)
    return CACHE_DIR / f"{tag}__{hashlib.sha1(raw.encode()).hexdigest()[:24]}.json"


def _memoize(fn, tag: str):
    def wrapped(start, end, params):
        path = _cache_path(tag, start, end, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        result = fn(start, end, params)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, default=str), encoding="utf-8")
        tmp.replace(path)   # atomic: a crash mid-write can never leave a truncated cache entry
        return result
    return wrapped


# ── bars ────────────────────────────────────────────────────────────────────

def load_bars(universe: list[str] | None, start: str, end: str) -> tuple[dict, list[str]]:
    from python.data.fixed_universe import load_universe_config
    from python.data.intraday_cache import get_cached_intraday_panel

    symbols = universe if universe is not None else load_universe_config()["symbols"]
    panel = get_cached_intraday_panel(symbols, start, end)
    codes = set(panel.index.get_level_values("code"))
    out = {s: panel.xs(s, level="code").sort_index() for s in symbols if s in codes}
    if not out:
        raise RuntimeError(f"no cached 1-minute bars for {symbols} in [{start}, {end})")
    return out, sorted(out)


# ── evaluation ──────────────────────────────────────────────────────────────

def _cache_tag(cfg: RescueConfig, suffix: str = "") -> str:
    uni = "full20" if cfg.universe is None else f"u{len(cfg.universe)}_{hashlib.sha1(','.join(sorted(cfg.universe)).encode()).hexdigest()[:6]}"
    return f"{cfg.code_variant}__{cfg.cost}__{uni}{suffix}"


def evaluate(cfg: RescueConfig) -> dict:
    base_cfg = _load_yaml(STRATEGY_PATH)[SIGNAL]
    goal = _load_yaml(GOAL_PATH)
    intraday_goal = goal.get("intraday", {})
    min_trades = int(intraday_goal.get("min_trades_per_oos_fold", 100))
    min_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.3))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    start_s, end_s = (DEV_START, DEV_END) if cfg.window == "dev" else (HOLDOUT_START, HOLDOUT_END)
    start_ts, end_ts = pd.Timestamp(start_s), pd.Timestamp(end_s)

    spreads = load_calibrated_spreads() if cfg.cost == "calibrated" else None
    bars_by_symbol, symbols_used = load_bars(cfg.universe, start_s, end_s)
    print(f"[{cfg.config_id}] {len(symbols_used)} symbols, window [{start_s}, {end_s}), "
          f"cost={cfg.cost}, params={cfg.params}", flush=True)

    preflight = preflight_check(SIGNAL, base_cfg, [cfg.params],
                                total_trading_days=len(pd.bdate_range(start_ts, end_ts)))

    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads)
    fn = _memoize(build_intraday_backtest_fn(bars_by_symbol, SIGNAL, base_cfg, engine_cfg=engine_cfg),
                  _cache_tag(cfg))
    stress_fn = _memoize(
        build_intraday_backtest_fn(
            bars_by_symbol, SIGNAL, base_cfg,
            engine_cfg=IntradayBacktestConfig(half_spread_bps_by_symbol=spreads,
                                              stress_slippage_multiplier=stress_mult)),
        _cache_tag(cfg, f"__stress{stress_mult:g}x"))

    result: dict = {
        "config_id": cfg.config_id, "lever": cfg.lever, "description": cfg.description,
        "params": cfg.params, "window": f"{start_s} .. {end_s}", "window_kind": cfg.window,
        "cost_model": "calibrated_per_symbol" if spreads else "flat_2.0bps",
        "n_symbols": len(symbols_used), "symbols": symbols_used,
        "code_variant": cfg.code_variant, "sample_size_check": preflight,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if cfg.window == "dev":
        wfo_cfg = load_wfo_config(SIGNAL)
        t0 = time.time()
        wfo = WalkForwardOptimizer(fn, wfo_cfg, [cfg.params]).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
        print(f"[{cfg.config_id}] WFO done in {time.time()-t0:.0f}s: "
              f"{wfo.passing_folds}/{wfo.total_folds} folds, OOS Sharpe mean {wfo.oos_sharpe_mean:+.3f}", flush=True)
        if not wfo.folds:
            result.update({"decision": "SKIPPED", "reason": "window too short for a single WFO fold"})
            return result
        result.update({
            "wfo_folds": wfo.total_folds, "wfo_passing_folds": wfo.passing_folds,
            "wfo_pass_ratio": wfo.pass_ratio, "oos_sharpe_mean": wfo.oos_sharpe_mean,
            "fold_oos_sharpes": [f.oos_sharpe for f in wfo.folds],
            "fold_oos_trades": [int(f.oos_metrics.get("n_trades", 0)) for f in wfo.folds],
            "fold_oos_profit_factors": [float(f.oos_metrics.get("profit_factor", 0.0)) for f in wfo.folds],
        })

    full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)
    print(f"[{cfg.config_id}] full-window: n_trades={full.get('n_trades')} "
          f"PF={full.get('profit_factor', 0):.3f} net={full.get('total_net_pnl', 0):,.0f} "
          f"| stress net={stress.get('total_net_pnl', 0):,.0f} | mc_p5={mc.sharpe.p5:+.3f}", flush=True)

    result["full_window_metrics"] = {k: v for k, v in full.items() if k != "daily_returns"}
    result["stress_metrics"] = {k: v for k, v in stress.items() if k != "daily_returns"}
    result["mc_p5_sharpe"] = mc.sharpe.p5

    if cfg.window == "dev":
        gates = {
            "wfo_go": wfo.decision == "GO",
            "oos_drawdown_within_limit": check_drawdown_gate(wfo, max_oos_drawdown_threshold()),
            "has_oos_trades": check_has_trades_gate(wfo),
            "min_trades_per_oos_fold": check_min_trades_gate(wfo, min_trades),
            "cost_adjusted_profit_factor": check_profit_factor_gate(wfo, min_pf),
            "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
            f"stress_slippage_{stress_mult:g}x_net_positive": stress["total_net_pnl"] > 0,
        }
    else:
        # The holdout is 2 months — shorter than a single is_days=90 +
        # oos_days=30 WFO fold, so there is no honest per-fold pass ratio to
        # compute. It is evaluated as ONE out-of-sample window with the
        # params frozen from the dev phase: the gates that ARE well-defined
        # on a single window (cost-adjusted PF, Monte Carlo p5, 2x-slippage
        # stress, trade count) are checked; `wfo_go` is reported as N/A
        # rather than faked from a truncated fold.
        gates = {
            "cost_adjusted_profit_factor": float(full.get("profit_factor", 0.0)) >= min_pf,
            "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
            f"stress_slippage_{stress_mult:g}x_net_positive": stress["total_net_pnl"] > 0,
            "has_trades": int(full.get("n_trades", 0)) > 0,
        }
    result["gates"] = gates
    result["decision"] = "GO" if all(gates.values()) else "NO-GO"
    return result


# ── checkpointing ───────────────────────────────────────────────────────────

def checkpoint_path(config_id: str) -> Path:
    return CHECKPOINT_DIR / f"{config_id}.json"


def run_config(cfg: RescueConfig, force: bool = False) -> dict:
    path = checkpoint_path(cfg.config_id)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f">>> {cfg.config_id} already checkpointed ({existing.get('decision')}) — skipping", flush=True)
        return existing
    t0 = time.time()
    result = evaluate(cfg)
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["config_spec"] = asdict(cfg)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    print(f">>> {cfg.config_id}: {result['decision']} (checkpointed, {result['elapsed_s']}s)", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_ids", nargs="*", help="one or more registry config ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if a checkpoint exists")
    ap.add_argument("--holdout-params", help="JSON dict of frozen params for HOLDOUT_best")
    ap.add_argument("--holdout-universe", help="comma-separated symbols for HOLDOUT_best")
    ap.add_argument("--holdout-note", default="")
    args = ap.parse_args()

    if args.list:
        for cid, c in CONFIGS.items():
            print(f"{cid:24s} [{c.lever}] {c.description}")
        return

    if args.holdout_params:
        universe = args.holdout_universe.split(",") if args.holdout_universe else None
        register_holdout(json.loads(args.holdout_params), universe, args.holdout_note)

    for cid in args.config_ids:
        if cid not in CONFIGS:
            raise SystemExit(f"unknown config id {cid!r} (use --list)")
        run_config(CONFIGS[cid], force=args.force)


if __name__ == "__main__":
    main()
