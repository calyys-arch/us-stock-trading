"""
Crash-resilient checkpoint runner for the `absorption_breakout` investigation
(backtests/reports/absorption_breakout_investigation_report.md) — the
genuine "Option A" pivot from `l2_absorption`'s retirement
(backtests/reports/l2_absorption_validation_report.md,
backtests/reports/signal_status.md): trade CONTINUATION when a heavy-volume
touch of a level FAILS (closes through it) rather than fading a bounce.

Two independent pieces, run via this one script (same checkpointing
discipline as scripts/_l2_absorption_validation.py):

  DIAG  Cheap diagnostic (task step 1, run FIRST): take l2_absorption's
        EXISTING entries (evaluate_l2_absorption completely unmodified —
        same trigger: high-volume touch + close-back-on-defended-side) and
        simply INVERT the direction of every fired signal (long<->short,
        stop/target mirrored to the opposite side at the SAME price
        distance from entry — which is the same ATR distance, since the
        original distance was itself `stop_atr_mult * ATR`). This is
        implemented by monkeypatching the `evaluate_l2_absorption` name
        INSIDE `python.backtest.intraday_engine`'s module namespace for the
        duration of one evaluate() call (see `_invert_signal`/
        `_patched_evaluator` below) — `l2_absorption.py` itself is NEVER
        edited, and the patch is restored immediately after, so this has
        zero effect on any other run in this script or any other process.
        Answers: is l2_absorption's entry trigger "just backwards" (gross
        PF should invert toward/past 1.0) or "uninformative regardless of
        direction" (gross PF stays in a similarly unprofitable range)?

  A/B/HOLDOUT  The real, intended test of Option A: `absorption_breakout`
        (python/microstructure/signals/absorption_breakout.py), a genuinely
        NEW entry condition (fires on a level BREAK, not a defended touch),
        run through the exact same WFO/Monte Carlo/cost-adjusted-PF/
        2x-slippage-stress gate pipeline as every other signal in this
        repo. A0 is the official pipeline verdict (full grid, per-fold
        re-optimization, full 20-symbol universe, calibrated cost); B* are
        up to 2-3 well-reasoned rescue levers chosen AFTER diagnosing A0's
        first-run failure mode (not a blind grid search); HOLDOUT_best is
        the single best DEV configuration, evaluated exactly once on the
        untouched final holdout, per a rule declared BEFORE looking at it.

Discipline (identical to scripts/_l2_absorption_validation.py /
scripts/_orb_vwap_rescue.py):
  * Gate thresholds read from configs/goal.yaml UNCHANGED. Nothing here
    writes to configs/strategy.yaml; `auto_execute` is never flipped.
  * HOLDOUT SEPARATION: dev configs use [DEV_START, DEV_END); the final
    holdout [HOLDOUT_START, HOLDOUT_END) is evaluated exactly once, for the
    single best DEV configuration, registered only via --holdout-params
    after DEV levers are settled.
  * Headline verdicts use CALIBRATED per-symbol half-spreads
    (backtests/reports/calibrated_spreads.json).
  * Every non-baseline candidate (B*) is FIXED across every WFO fold
    (param_grid=[candidate], no per-fold re-optimization) — only A0 uses
    the full grid with genuine per-fold re-optimization.

Resilience: per-config checkpoint JSON in
`backtests/reports/_absorption_breakout_validation/`; every individual
backtest_fn(start, end, params) call is memoized to disk under
`backtests/reports/_absorption_breakout_validation_cache/`.

Usage:
    python scripts/_absorption_breakout_validation.py --list
    python scripts/_absorption_breakout_validation.py DIAG_inversion
    python scripts/_absorption_breakout_validation.py A0_grid_full20
    python scripts/_absorption_breakout_validation.py B1_some_lever
    python scripts/_absorption_breakout_validation.py HOLDOUT_best \
        --holdout-params '{"volume_mult": 3.0, ...}' --holdout-universe AAPL,GOOGL,...
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

from run_intraday_backtest import GOAL_PATH, STRATEGY_PATH, _load_yaml  # noqa: E402
import python.backtest.intraday_engine as intraday_engine_mod  # noqa: E402
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
)
from python.backtest.walk_forward import WalkForwardOptimizer  # noqa: E402
from python.microstructure.signals import MicroSignal  # noqa: E402

SIGNAL = "absorption_breakout"
L2_SIGNAL = "l2_absorption"
CHECKPOINT_DIR = Path("backtests/reports/_absorption_breakout_validation")
CACHE_DIR = Path("backtests/reports/_absorption_breakout_validation_cache")
CALIBRATED_SPREADS_PATH = Path("backtests/reports/calibrated_spreads.json")

# ── windows (identical to scripts/_l2_absorption_validation.py, for direct
#    cross-investigation comparability) ─────────────────────────────────────
DEV_START, DEV_END = "2025-08-01", "2026-06-01"
HOLDOUT_START, HOLDOUT_END = "2026-06-01", "2026-08-01"

FULL20 = None  # None = whatever configs/universe.yaml's fixed universe holds
TIGHT10 = ["AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC", "META", "AVGO", "AMD", "QCOM"]
TIGHT6 = ["AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC"]

# l2_absorption's own official-run baseline params (configs/strategy.yaml),
# used verbatim (not re-optimized) for the DIAG inversion diagnostic per the
# task brief.
L2_BASELINE_PARAMS = {"volume_mult": 3.0, "touch_atr_mult": 0.25, "stop_atr_mult": 0.5}


# ── cost model / bars (identical helpers to _l2_absorption_validation.py) ──

def load_calibrated_spreads() -> dict[str, float]:
    payload = json.loads(CALIBRATED_SPREADS_PATH.read_text(encoding="utf-8"))
    return {sym: float(s["median_bps"]) for sym, s in payload["symbols"].items() if not s.get("suspect")}


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


# ── disk memoization (identical pattern to _l2_absorption_validation.py) ──

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
        tmp.replace(path)
        return result
    return wrapped


# ── DIAG: crude direction-inversion diagnostic (task step 1) ───────────────

def _invert_signal(sig: MicroSignal) -> MicroSignal:
    """long<->short, stop/target MIRRORED to the opposite side of
    `entry_price` at the SAME price distance (== same ATR distance, since
    l2_absorption's own stop/target distances are themselves
    `stop_atr_mult * ATR` / `target_r_multiple * risk`). The entry trigger
    (which bar fires, at what price) is completely untouched — only the
    direction and the two price levels that depend on direction change."""
    inv_direction = "short" if sig.direction == "long" else "long"
    stop_dist = abs(sig.entry_price - sig.stop_price)
    new_stop = sig.entry_price + stop_dist if inv_direction == "short" else sig.entry_price - stop_dist
    new_target = None
    if sig.target_price is not None:
        target_dist = abs(sig.entry_price - sig.target_price)
        new_target = sig.entry_price - target_dist if inv_direction == "short" else sig.entry_price + target_dist
    return MicroSignal(
        symbol=sig.symbol, strategy=sig.strategy, direction=inv_direction,
        signal_time=sig.signal_time, entry_price=sig.entry_price,
        stop_price=new_stop, target_price=new_target,
        order_type=sig.order_type, expiry_time=sig.expiry_time,
        context={**sig.context, "diagnostic_inverted_from_l2_absorption": True},
    )


def _make_inverting_evaluator(original_evaluate_l2_absorption):
    def _inverted_evaluate_l2_absorption(bars, symbol: str = "", **kwargs):
        sig = original_evaluate_l2_absorption(bars, symbol=symbol, **kwargs)
        return None if sig is None else _invert_signal(sig)
    return _inverted_evaluate_l2_absorption


def run_inversion_diagnostic(force: bool = False) -> dict:
    """Full A0-equivalent window/universe/cost, l2_absorption's baseline
    params (UNMODIFIED entry trigger), calibrated cost, DEV window — run
    TWICE: once with the real (fade) direction, once with every fired
    signal's direction inverted (stop/target mirrored, entry untouched).
    No WFO/grid search either time (the task asks for a direct before/after
    comparison on ONE fixed, named configuration, not a re-optimization)."""
    path = CHECKPOINT_DIR / "DIAG_inversion.json"
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f">>> DIAG_inversion already checkpointed — skipping", flush=True)
        return existing

    t0 = time.time()
    spreads = load_calibrated_spreads()
    bars_by_symbol, symbols_used = load_bars(FULL20, DEV_START, DEV_END)
    base_cfg = _load_yaml(STRATEGY_PATH)[L2_SIGNAL]
    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads)
    start_ts, end_ts = pd.Timestamp(DEV_START), pd.Timestamp(DEV_END)

    print(f"[DIAG_inversion] {len(symbols_used)} symbols, window [{DEV_START}, {DEV_END}), "
          f"params={L2_BASELINE_PARAMS} (l2_absorption baseline, unmodified trigger)", flush=True)

    # ORIGINAL direction (real evaluate_l2_absorption, byte-for-byte).
    fn_original = _memoize(
        build_intraday_backtest_fn(bars_by_symbol, L2_SIGNAL, base_cfg, engine_cfg=engine_cfg),
        "diag_original",
    )
    original_metrics = fn_original(start_ts.to_pydatetime(), end_ts.to_pydatetime(), L2_BASELINE_PARAMS)
    print(f"[DIAG_inversion] ORIGINAL direction: n_trades={original_metrics.get('n_trades')} "
          f"PF_net={original_metrics.get('profit_factor', 0):.4f} "
          f"PF_gross={original_metrics.get('profit_factor_gross', 0):.4f} "
          f"net_pnl={original_metrics.get('total_net_pnl', 0):,.0f} "
          f"gross_pnl={original_metrics.get('gross_pnl', 0):,.0f}", flush=True)

    # INVERTED direction: monkeypatch the module-global `evaluate_l2_absorption`
    # name INSIDE python.backtest.intraday_engine (NOT l2_absorption.py
    # itself) for the duration of this one call, then restore it
    # unconditionally (try/finally) so nothing else in this process — or
    # any other config run afterward in this same script invocation — is
    # ever affected by the patch.
    original_fn_ref = intraday_engine_mod.evaluate_l2_absorption
    intraday_engine_mod.evaluate_l2_absorption = _make_inverting_evaluator(original_fn_ref)
    try:
        fn_inverted = _memoize(
            build_intraday_backtest_fn(bars_by_symbol, L2_SIGNAL, base_cfg, engine_cfg=engine_cfg),
            "diag_inverted",
        )
        inverted_metrics = fn_inverted(start_ts.to_pydatetime(), end_ts.to_pydatetime(), L2_BASELINE_PARAMS)
    finally:
        intraday_engine_mod.evaluate_l2_absorption = original_fn_ref
        assert intraday_engine_mod.evaluate_l2_absorption is original_fn_ref  # patch fully undone

    print(f"[DIAG_inversion] INVERTED direction: n_trades={inverted_metrics.get('n_trades')} "
          f"PF_net={inverted_metrics.get('profit_factor', 0):.4f} "
          f"PF_gross={inverted_metrics.get('profit_factor_gross', 0):.4f} "
          f"net_pnl={inverted_metrics.get('total_net_pnl', 0):,.0f} "
          f"gross_pnl={inverted_metrics.get('gross_pnl', 0):,.0f}", flush=True)

    result = {
        "config_id": "DIAG_inversion",
        "description": (
            "l2_absorption's EXISTING entries (evaluate_l2_absorption unmodified), baseline params, "
            "full 20-symbol universe, calibrated cost, DEV window -- direction of every fired signal "
            "inverted (stop/target mirrored to the opposite side at the same price/ATR distance), "
            "entry trigger completely unchanged."
        ),
        "params": L2_BASELINE_PARAMS,
        "window": f"{DEV_START} .. {DEV_END}",
        "n_symbols": len(symbols_used), "symbols": symbols_used,
        "cost_model": "calibrated_per_symbol",
        "original_direction_metrics": {k: v for k, v in original_metrics.items() if k != "daily_returns"},
        "inverted_direction_metrics": {k: v for k, v in inverted_metrics.items() if k != "daily_returns"},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    print(f">>> DIAG_inversion: checkpointed ({result['elapsed_s']}s)", flush=True)
    return result


# ── A/B/HOLDOUT: absorption_breakout's own WFO/MC/stress pipeline ─────────

@dataclass
class ABConfig:
    config_id: str
    lever: str
    description: str
    params: dict
    universe: list[str] | None = None
    window: str = "dev"          # "dev" | "holdout"
    cost: str = "calibrated"     # "calibrated" | "flat"
    grid: bool = False           # True: configs/param_grids.yaml full grid, per-fold reoptimization
    extra: dict = field(default_factory=dict)


CONFIGS: dict[str, ABConfig] = {}


def _add(cfg: ABConfig) -> None:
    CONFIGS[cfg.config_id] = cfg


BASE_PARAMS = {"volume_mult": 3.0, "breakout_atr_mult": 0.0, "stop_atr_mult": 0.5}


def _p(**overrides) -> dict:
    out = dict(BASE_PARAMS)
    out.update(overrides)
    return out


# A: baseline — the official pipeline verdict.
_add(ABConfig(
    "A0_grid_full20", "baseline",
    "configs/param_grids.yaml full grid (27 candidates), per-fold WFO reoptimization, full 20-symbol "
    "universe, calibrated cost -- the standard pipeline's own verdict for absorption_breakout.",
    {}, FULL20, grid=True,
))

# B: rescue levers -- registered dynamically (see main()) AFTER A0's failure
# mode is diagnosed, per this repo's diagnose-before-lever discipline. Two
# candidate levers pre-wired here (added regardless of whether they end up
# used, so --list shows them; main() decides whether to actually run them):
_add(ABConfig(
    "B1_tight10", "1: universe",
    "tight-spread top-10 (calibrated half-spread <= 1.89bps), fixed baseline params -- tests whether "
    "cost/liquidity concentration is the dominant failure mode (same lever/universe as l2_absorption's "
    "and orb_vwap's own rescue investigations, for cross-signal comparability).",
    _p(), TIGHT10,
))
_add(ABConfig(
    "B2_tight6", "1: universe",
    "tight-spread top-6 (calibrated half-spread <= 1.00bps), fixed baseline params.",
    _p(), TIGHT6,
))
_add(ABConfig(
    "B3_clearance_tight10", "2: min-breakout-clearance filter (beyond A0's own grid ceiling)",
    "breakout_atr_mult=0.5 -- PAST configs/param_grids.yaml's tested ceiling (0.3), which A0's own "
    "per-fold WFO re-optimization picked in EVERY one of 7 folds (see A0_grid_full20.json's "
    "fold_best_params) -- direct diagnosis-driven test of whether requiring even more clearance beyond "
    "the level (filtering more marginal/noise breaks) continues to help or has already plateaued, on "
    "TIGHT10 (isolating this one dimension from the universe lever above).",
    _p(breakout_atr_mult=0.5), TIGHT10,
))
_add(ABConfig(
    "B4_wide_stop_tight10", "3: wider stop",
    "stop_atr_mult=1.0 (vs baseline 0.5) on TIGHT10 -- tests whether the stop is too tight for a "
    "continuation trade (structural noise around the just-broken level triggering premature stops).",
    _p(stop_atr_mult=1.0), TIGHT10,
))
# B5: STACK the two levers that each individually moved the needle
# (B2_tight6 and B3_clearance_tight10) -- a natural, non-blind extension of
# the SAME two already-diagnosed dimensions (not a new third hypothesis),
# same spirit as l2_absorption's rescue picking its single best universe
# setting: does the improvement compound, or does one lever's gain just
# subsume the trades the other would also have filtered out?
_add(ABConfig(
    "B5_tight6_clearance", "1+2: universe AND min-breakout-clearance filter combined",
    "TIGHT6 (best universe setting from B1/B2) AND breakout_atr_mult=0.5 (best clearance setting from "
    "B3) together -- tests whether the two independently-helpful levers compound.",
    _p(breakout_atr_mult=0.5), TIGHT6,
))


def register_holdout(params: dict, universe: list[str] | None, note: str) -> ABConfig:
    cfg = ABConfig(
        "HOLDOUT_best", "FINAL HOLDOUT",
        f"single best DEV configuration, evaluated ONCE on the untouched holdout ({note})",
        params, universe, window="holdout",
    )
    _add(cfg)
    return cfg


def _cache_tag(cfg: ABConfig, suffix: str = "") -> str:
    uni = "full20" if cfg.universe is None else f"u{len(cfg.universe)}_{hashlib.sha1(','.join(sorted(cfg.universe)).encode()).hexdigest()[:6]}"
    return f"{cfg.cost}__{uni}{suffix}"


def evaluate(cfg: ABConfig) -> dict:
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
          f"cost={cfg.cost}, grid={cfg.grid}, params={cfg.params}", flush=True)

    param_grid = load_param_grid(SIGNAL) if cfg.grid else [cfg.params]

    preflight = preflight_check(SIGNAL, base_cfg, param_grid,
                                total_trading_days=len(pd.bdate_range(start_ts, end_ts)))

    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads)
    fn = _memoize(build_intraday_backtest_fn(bars_by_symbol, SIGNAL, base_cfg, engine_cfg=engine_cfg),
                  _cache_tag(cfg))

    result: dict = {
        "config_id": cfg.config_id, "lever": cfg.lever, "description": cfg.description,
        "params": cfg.params, "grid": cfg.grid, "window": f"{start_s} .. {end_s}",
        "window_kind": cfg.window, "cost_model": "calibrated_per_symbol" if spreads else "flat_2.0bps",
        "n_symbols": len(symbols_used), "symbols": symbols_used,
        "sample_size_check": preflight, "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if cfg.window == "dev":
        wfo_cfg = load_wfo_config(SIGNAL)
        t0 = time.time()
        wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
        print(f"[{cfg.config_id}] WFO done in {time.time()-t0:.0f}s: "
              f"{wfo.passing_folds}/{wfo.total_folds} folds, OOS Sharpe mean {wfo.oos_sharpe_mean:+.3f}", flush=True)
        if not wfo.folds:
            result.update({"decision": "SKIPPED", "reason": "window too short for a single WFO fold"})
            return result
        best_params = dict(wfo.folds[-1].best_params)
        result.update({
            "wfo_folds": wfo.total_folds, "wfo_passing_folds": wfo.passing_folds,
            "wfo_pass_ratio": wfo.pass_ratio, "oos_sharpe_mean": wfo.oos_sharpe_mean,
            "fold_oos_sharpes": [f.oos_sharpe for f in wfo.folds],
            "fold_oos_trades": [int(f.oos_metrics.get("n_trades", 0)) for f in wfo.folds],
            "fold_oos_profit_factors": [float(f.oos_metrics.get("profit_factor", 0.0)) for f in wfo.folds],
            "fold_oos_profit_factors_gross": [float(f.oos_metrics.get("profit_factor_gross", 0.0)) for f in wfo.folds],
            "fold_best_params": [f.best_params for f in wfo.folds],
            "most_recent_fold_best_params": best_params,
        })
        eval_params = best_params
    else:
        eval_params = cfg.params

    stress_fn = _memoize(
        build_intraday_backtest_fn(
            bars_by_symbol, SIGNAL, base_cfg,
            engine_cfg=IntradayBacktestConfig(half_spread_bps_by_symbol=spreads,
                                              stress_slippage_multiplier=stress_mult)),
        _cache_tag(cfg, f"__stress{stress_mult:g}x"))

    full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), eval_params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), eval_params)
    print(f"[{cfg.config_id}] full-window: n_trades={full.get('n_trades')} "
          f"PF_net={full.get('profit_factor', 0):.3f} PF_gross={full.get('profit_factor_gross', 0):.3f} "
          f"net_pnl={full.get('total_net_pnl', 0):,.0f} gross_pnl={full.get('gross_pnl', 0):,.0f} "
          f"costs={full.get('total_costs', 0):,.0f} "
          f"| stress net={stress.get('total_net_pnl', 0):,.0f} | mc_p5={mc.sharpe.p5:+.3f}", flush=True)

    result["full_window_metrics"] = {k: v for k, v in full.items() if k != "daily_returns"}
    result["stress_metrics"] = {k: v for k, v in stress.items() if k != "daily_returns"}
    result["mc_p5_sharpe"] = mc.sharpe.p5
    result["eval_params"] = eval_params

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
        gates = {
            "cost_adjusted_profit_factor": float(full.get("profit_factor", 0.0)) >= min_pf,
            "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
            f"stress_slippage_{stress_mult:g}x_net_positive": stress["total_net_pnl"] > 0,
            "has_trades": int(full.get("n_trades", 0)) > 0,
        }
    result["gates"] = gates
    result["decision"] = "GO" if all(gates.values()) else "NO-GO"
    return result


def checkpoint_path(config_id: str) -> Path:
    return CHECKPOINT_DIR / f"{config_id}.json"


def run_config(cfg: ABConfig, force: bool = False) -> dict:
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
    ap.add_argument("config_ids", nargs="*", help="one or more registry config ids, or 'DIAG_inversion'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if a checkpoint exists")
    ap.add_argument("--holdout-params", help="JSON dict of frozen params for HOLDOUT_best")
    ap.add_argument("--holdout-universe", help="comma-separated symbols for HOLDOUT_best")
    ap.add_argument("--holdout-note", default="")
    args = ap.parse_args()

    if args.list:
        print("DIAG_inversion            [diagnostic] invert l2_absorption's existing entries (task step 1)")
        for cid, c in CONFIGS.items():
            print(f"{cid:24s} [{c.lever}] {c.description}")
        return

    if args.holdout_params:
        universe = args.holdout_universe.split(",") if args.holdout_universe else None
        register_holdout(json.loads(args.holdout_params), universe, args.holdout_note)

    for cid in args.config_ids:
        if cid == "DIAG_inversion":
            run_inversion_diagnostic(force=args.force)
            continue
        if cid not in CONFIGS:
            raise SystemExit(f"unknown config id {cid!r} (use --list)")
        run_config(CONFIGS[cid], force=args.force)


if __name__ == "__main__":
    main()
