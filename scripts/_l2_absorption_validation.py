"""
Crash-resilient checkpoint runner for l2_absorption's end-to-end validation
(backtests/reports/l2_absorption_validation_report.md).

`l2_absorption` (S4) is the LAST untested microstructure signal in this repo
(backtests/reports/signal_status.md) — every other one (sweep_reclaim,
fvg_retest, orb_vwap, orb_vwap_regime, vwap_band_fade, vp_breakout) is
already RETIRED/NO-GO. This script runs the exact same validation discipline
already established for those signals (see scripts/run_intraday_backtest.py)
PLUS l2_absorption-specific rescue levers, using the same methodology
template as scripts/_orb_vwap_rescue.py:

  A  baseline    — the "official" pipeline verdict: configs/param_grids.yaml's
                   full l2_absorption grid, genuine per-fold WFO
                   re-optimization (WalkForwardOptimizer picks the best IS
                   candidate every fold, same as `--signal l2_absorption`),
                   full 20-symbol universe, calibrated per-symbol slippage.
                   This is run FIRST and its gross-vs-net profit factor
                   (python/backtest/intraday_engine.py's diagnostic fields)
                   diagnoses the dominant failure mode before any lever is
                   chosen — see the module docstring's "no gross edge" vs
                   "cost-killed" vs "fold-unstable" distinction, same
                   diagnostic axis backtests/reports/orb_vwap_rescue_report.md
                   used.
  B  universe    — restrict to the tight-spread subset of the 20-symbol
                   universe (identical TIGHT10/TIGHT6 lists to
                   scripts/_orb_vwap_rescue.py, for apples-to-apples
                   comparison across both signals' cost-sensitivity).
  C  target      — target_r_multiple sweep (the l2_absorption lever added
                   2026-08-14, identical convention to orb_vwap's own
                   target_r_multiple rescue lever).
  D  regime gate — python/analytics/trend_efficiency_gate.shifted_entry_gate
                   applied as an entry filter: l2_absorption is a fade/bounce
                   AT a level (mean-reversion family, not trend-following —
                   see l2_absorption.py's module docstring), so the gate is
                   used at its NATIVE polarity (mean-reversion strategies may
                   trade when trend efficiency is at/below its own trailing
                   median) — no inversion needed, unlike if this were applied
                   to a trend-following signal. Implemented here as a
                   SESSION-DATE FILTER on `bars_by_symbol` (drop entire
                   session-dates where the gate is closed) rather than a new
                   intraday_engine.py code path: l2_absorption's
                   evaluate_l2_absorption never reads prior_day_bars/
                   prior_close (unlike sweep_reclaim/orb_vwap), so dropping
                   whole non-gated sessions before the bars ever reach
                   run_intraday_backtest is behaviorally identical to a
                   proper in-engine gate for this one signal, with zero risk
                   to the shared engine code path other signals depend on.
                   Daily closes come from data/history/<SYMBOL>.csv (longer
                   history than the 1-minute cache, needed for the gate's
                   252-day reference window to be populated from the START
                   of the DEV backtest window — see that module's docstring).

  order_flow_imbalance_score / print_lag_score confirmation filter — NOT
  attempted. Investigated and ruled INFEASIBLE for the DEV/HOLDOUT windows
  below: data/ticks/ and data/depth/ (where these are computed from) only
  cover 2026-08-04..2026-08-14, entirely AFTER the cached 1-minute OHLCV
  history this signal is validated against ends (2026-07-31) — there is no
  overlapping day to even compute the filter's efficacy on, let alone tune
  or validate it. Documented as a data-availability gap in the validation
  report rather than silently skipped.

Discipline this script enforces (identical to _orb_vwap_rescue.py):

  * Gate thresholds are read from configs/goal.yaml UNCHANGED. Nothing here
    writes to configs/strategy.yaml and nothing flips `auto_execute` — a GO
    is a promotion CANDIDATE for human review only.
  * HOLDOUT SEPARATION. Every development/tuning config runs on the DEV
    window only (`--window` implied by config registry, dev configs use
    [DEV_START, DEV_END)). The last two months of available 1-minute history
    are the FINAL HOLDOUT and are evaluated exactly ONCE, for the single best
    dev configuration, at the end (`HOLDOUT_best`, registered only via
    --holdout-params after DEV levers are settled).
  * Headline verdicts use the CALIBRATED per-symbol half-spreads
    (backtests/reports/calibrated_spreads.json); `--cost flat` exists only
    for secondary context.
  * Every non-baseline candidate is FIXED across every WFO fold
    (param_grid=[candidate], no per-fold re-optimization) — only the A0/A1
    baseline configs use the full grid with genuine per-fold
    re-optimization, to answer "what would the standard pipeline actually
    pick and trade".

Resilience (same prior art as scripts/_orb_vwap_rescue.py /
scripts/_resume_new_signals_validation.py):
  1. Per-config checkpoint JSON in `backtests/reports/_l2_absorption_validation/`.
  2. Every individual backtest_fn(start, end, params) call inside the WFO
     fold loop AND the stress re-run is memoized to disk under
     `backtests/reports/_l2_absorption_validation_cache/`, keyed by
     (config_id ingredients, start, end, params). A crash loses at most the
     one in-flight fold, and re-invoking the same command replays everything
     already computed instantly.

Usage:
    python scripts/_l2_absorption_validation.py --list
    python scripts/_l2_absorption_validation.py A0_grid_full20
    python scripts/_l2_absorption_validation.py B1_tight10 B2_tight6
    python scripts/_l2_absorption_validation.py HOLDOUT_best \
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

from run_intraday_backtest import (  # noqa: E402
    GOAL_PATH,
    STRATEGY_PATH,
    _load_yaml,
    run_signal,
)
from python.analytics.trend_efficiency_gate import (  # noqa: E402
    DEFAULT_REFERENCE_WINDOW,
    DEFAULT_WINDOW,
    shifted_entry_gate,
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
)
from python.backtest.walk_forward import WalkForwardOptimizer  # noqa: E402

SIGNAL = "l2_absorption"
CHECKPOINT_DIR = Path("backtests/reports/_l2_absorption_validation")
CACHE_DIR = Path("backtests/reports/_l2_absorption_validation_cache")
CALIBRATED_SPREADS_PATH = Path("backtests/reports/calibrated_spreads.json")
DAILY_HISTORY_DIR = Path("data/history")

# ── windows ─────────────────────────────────────────────────────────────────
# Available cached 1-minute history: 2025-07-02 .. 2026-07-31 (confirmed via
# python.data.intraday_cache.get_cached_intraday_panel, 2026-08-14). Same
# DEV/HOLDOUT split as scripts/_orb_vwap_rescue.py, reused verbatim for
# direct comparability between the two signals' cost-sensitivity findings.
DEV_START, DEV_END = "2025-08-01", "2026-06-01"
HOLDOUT_START, HOLDOUT_END = "2026-06-01", "2026-08-01"

# ── universes (calibrated median half-spread, bps, from
#    backtests/reports/calibrated_spreads.json, all 20 symbols non-suspect) ──
FULL20 = None  # None = whatever configs/universe.yaml's fixed universe holds
TIGHT10 = ["AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC", "META", "AVGO", "AMD", "QCOM"]
TIGHT6 = ["AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC"]


@dataclass
class L2Config:
    config_id: str
    lever: str                       # which lever this row of the report isolates
    description: str
    params: dict                     # l2_absorption signal params (ignored if grid=True)
    universe: list[str] | None = None
    window: str = "dev"              # "dev" | "holdout"
    cost: str = "calibrated"         # "calibrated" | "flat"
    grid: bool = False               # True: configs/param_grids.yaml's full grid,
                                      # per-fold re-optimization (the "what would the
                                      # standard pipeline pick" baseline). False: `params`
                                      # held FIXED across every fold (a rescue lever).
    regime_gate: bool = False        # True: trend-efficiency mean-reversion entry filter
    extra: dict = field(default_factory=dict)


CONFIGS: dict[str, L2Config] = {}


def _add(cfg: L2Config) -> None:
    CONFIGS[cfg.config_id] = cfg


BASE_PARAMS = {"volume_mult": 3.0, "touch_atr_mult": 0.25, "stop_atr_mult": 0.5}


def _p(**overrides) -> dict:
    out = dict(BASE_PARAMS)
    out.update(overrides)
    return out


# ── A: baseline — the official pipeline verdict ──────────────────────────────
_add(L2Config(
    "A0_grid_full20", "baseline",
    "configs/param_grids.yaml full grid (27 candidates), per-fold WFO "
    "reoptimization, full 20-symbol universe, calibrated cost — the "
    "standard pipeline's own verdict (equivalent to "
    "`run_intraday_backtest.py --signal l2_absorption`, but with calibrated "
    "rather than flat cost, and DEV-window-only)",
    {}, FULL20, grid=True,
))
# Secondary context only: identical grid search under the flat 2.0bps
# assumption, to see how much of A0's result (if any) is attributable to the
# calibrated-cost switch itself vs the signal's own edge.
_add(L2Config(
    "A1_grid_full20_flat", "context: flat cost",
    "same as A0 but flat 2.0bps cost instead of calibrated per-symbol spreads",
    {}, FULL20, grid=True, cost="flat",
))

# ── B: tight-spread universe subset (fixed baseline params) ─────────────────
_add(L2Config(
    "B1_tight10", "1: universe",
    "tight-spread top-10 (calibrated half-spread <= 1.89bps), fixed baseline params",
    _p(), TIGHT10,
))
_add(L2Config(
    "B2_tight6", "1: universe",
    "tight-spread top-6 (calibrated half-spread <= 1.00bps), fixed baseline params",
    _p(), TIGHT6,
))

# ── C: target_r_multiple sweep (universe fixed at TIGHT10, pending B's
#    result — if B's tightening doesn't help, TIGHT10 is still used here so
#    every C-row lever is measured against the SAME base as the B-row that
#    will be cited as its starting point, not a moving baseline) ────────────
_add(L2Config(
    "C1_r1", "2: target",
    "1.0R profit target (vs no target / time-stop-only)",
    _p(target_r_multiple=1.0), TIGHT10,
))
_add(L2Config(
    "C2_r2", "2: target",
    "2.0R profit target",
    _p(target_r_multiple=2.0), TIGHT10,
))
_add(L2Config(
    "C3_r3", "2: target",
    "3.0R profit target",
    _p(target_r_multiple=3.0), TIGHT10,
))

# ── D: trend-efficiency mean-reversion regime gate (universe fixed at
#    TIGHT10, fixed baseline params — isolates the gate's OWN effect from
#    the target-R lever above) ───────────────────────────────────────────────
_add(L2Config(
    "D1_regime_gate", "3: regime gate",
    "trend-efficiency mean-reversion entry gate (native polarity — trade "
    "only when NOT in a persistent trend), TIGHT10, fixed baseline params",
    _p(), TIGHT10, regime_gate=True,
))


def register_holdout(params: dict, universe: list[str] | None, regime_gate: bool, note: str) -> L2Config:
    cfg = L2Config(
        "HOLDOUT_best", "FINAL HOLDOUT",
        f"single best DEV configuration, evaluated ONCE on the untouched holdout ({note})",
        params, universe, window="holdout", regime_gate=regime_gate,
    )
    _add(cfg)
    return cfg


# ── cost model ──────────────────────────────────────────────────────────────

def load_calibrated_spreads() -> dict[str, float]:
    payload = json.loads(CALIBRATED_SPREADS_PATH.read_text(encoding="utf-8"))
    return {sym: float(s["median_bps"]) for sym, s in payload["symbols"].items() if not s.get("suspect")}


# ── regime gate (trend-efficiency, native mean-reversion polarity) ─────────

def _regime_open_dates(symbol: str) -> set:
    """Session dates (normalized, tz-naive midnight Timestamps) on which the
    trend-efficiency gate is OPEN (mean-reversion strategies may trade) for
    `symbol`, computed from data/history/<symbol>.csv's full daily-close
    history (2016-2026 for most symbols) rather than the shorter 1-minute
    cache — see this module's docstring and
    python/analytics/trend_efficiency_gate.py's own docstring for why a
    252-day reference window needs history predating the DEV backtest
    window's own start. Symbols with insufficient trailing history (NBIS,
    SNDK — see the validation report's caveat) simply have fewer OPEN dates
    early in the window; `shifted_entry_gate` already fails CLOSED (not
    silently open) for any undecided day, so this is a conservative gap, not
    a look-ahead one."""
    path = DAILY_HISTORY_DIR / f"{symbol}.csv"
    if not path.exists():
        return set()
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    gate = shifted_entry_gate(daily["close"], window=DEFAULT_WINDOW, reference_window=DEFAULT_REFERENCE_WINDOW)
    return set(pd.DatetimeIndex(gate[gate].index).normalize())


def apply_regime_gate(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, bars in bars_by_symbol.items():
        open_dates = _regime_open_dates(sym)
        mask = bars.index.normalize().isin(open_dates)
        out[sym] = bars.loc[mask]
    return out


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
        tmp.replace(path)
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

def _cache_tag(cfg: L2Config, suffix: str = "") -> str:
    uni = "full20" if cfg.universe is None else f"u{len(cfg.universe)}_{hashlib.sha1(','.join(sorted(cfg.universe)).encode()).hexdigest()[:6]}"
    gate = "_gated" if cfg.regime_gate else ""
    return f"{cfg.cost}__{uni}{gate}{suffix}"


def evaluate(cfg: L2Config) -> dict:
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
    if cfg.regime_gate:
        bars_by_symbol = apply_regime_gate(bars_by_symbol)
    print(f"[{cfg.config_id}] {len(symbols_used)} symbols, window [{start_s}, {end_s}), "
          f"cost={cfg.cost}, grid={cfg.grid}, regime_gate={cfg.regime_gate}, params={cfg.params}", flush=True)

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
        "n_symbols": len(symbols_used), "symbols": symbols_used, "regime_gate": cfg.regime_gate,
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
        # Same honest single-window holdout gate set as
        # scripts/_orb_vwap_rescue.py's holdout branch (see that script's
        # comment): the holdout is 2 months, shorter than one full WFO fold
        # (is_days=90 + oos_days=30), so there is no per-fold pass ratio to
        # compute here; `wfo_go` is simply not part of the holdout's gate set.
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


def run_config(cfg: L2Config, force: bool = False) -> dict:
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
    ap.add_argument("--holdout-regime-gate", action="store_true")
    ap.add_argument("--holdout-note", default="")
    args = ap.parse_args()

    if args.list:
        for cid, c in CONFIGS.items():
            print(f"{cid:24s} [{c.lever}] {c.description}")
        return

    if args.holdout_params:
        universe = args.holdout_universe.split(",") if args.holdout_universe else None
        register_holdout(json.loads(args.holdout_params), universe, args.holdout_regime_gate, args.holdout_note)

    for cid in args.config_ids:
        if cid not in CONFIGS:
            raise SystemExit(f"unknown config id {cid!r} (use --list)")
        run_config(CONFIGS[cid], force=args.force)


if __name__ == "__main__":
    main()
