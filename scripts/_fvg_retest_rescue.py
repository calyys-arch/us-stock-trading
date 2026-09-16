"""
Crash-resilient checkpoint runner for the fvg_retest RESCUE investigation
(backtests/reports/fvg_retest_rescue_report.md).

Same method as scripts/_orb_vwap_rescue.py (see that file's docstring and
backtests/reports/orb_vwap_rescue_report.md for the full precedent this
mirrors). `fvg_retest` was re-examined by scripts/_gross_edge_diag.py on
2026-09-16 and found to have a genuinely positive GROSS (pre-cost) edge on
configs/universe.yaml's 20-symbol universe — gross profit factor 1.307,
48.7% win rate — unlike sweep_reclaim (gross PF < 1 even at zero cost, a
falsified thesis on the low-liquidity universe). The net cost-adjusted PF
was 0.195 at the time of the original 2026-08-13 retirement, almost
entirely from two causes this script isolates:

  Lever 1  universe   — restrict to the calibrated tight-spread names only
                        (the same lever that was orb_vwap's single biggest
                        win: +0.181..+0.234 PF).
  Lever 2  entry cap  — cap entries per symbol per session (~4.7
                        trades/symbol/session as shipped; only the first
                        entry of a session is expected to carry the pattern's
                        real edge, by analogy with orb_vwap's entry-ordinal
                        diagnostic).
  Lever 3  target     — loosen the hardcoded 1:1 R:R target (the literal
                        cause named in the original 2026-08-13 retirement
                        docstring: "a strict 1:1 risk:reward target ... needs
                        a win rate far above what the pattern actually has").

Unlike orb_vwap, fvg_retest needed NO stop-side correctness fix (its module
docstring already documents that class of defect was caught and fixed for
ITS target price before orb_vwap's Lever 0 was even found), so there is no
"A0 -> A1" corrected-baseline step here — A0 IS the correct baseline.

Discipline this script enforces (identical to _orb_vwap_rescue.py, with ONE
deliberate deviation — see below):

  * Gate thresholds are read from configs/goal.yaml UNCHANGED. Nothing here
    writes to configs/strategy.yaml and nothing flips `auto_execute` — a GO
    is a promotion CANDIDATE for human review only.
  * HOLDOUT SEPARATION. Every development/tuning config runs on the DEV
    window only (`--window dev`, hardcoded per-config below). The final two
    months of available 1-minute history are the FINAL HOLDOUT and are
    evaluated exactly ONCE, for the single best dev configuration.
  * Headline verdicts use the CALIBRATED per-symbol half-spreads
    (backtests/reports/calibrated_spreads.json).
  * Every candidate is FIXED across every WFO fold (param_grid=[candidate],
    no per-fold re-optimization).
  * TIGHT10/TIGHT6 are computed HERE, fresh, from the current
    calibrated_spreads.json (NOT copy-pasted from _orb_vwap_rescue.py's
    hardcoded lists, which are for configs/universe.yaml's SYMBOL SET AT THE
    TIME of that 2026-08-13 investigation — the universe has since been
    refreshed, e.g. QCOM/MU's relative order changed).

  DEVIATION FROM _orb_vwap_rescue.py's evaluate(): that script hand-rolled
  its own gate dict (WFO / drawdown / has_oos_trades / min_trades_per_fold
  every-fold / cost_adjusted_PF >= 1.3 every-fold / MC p5 / stress net P&L
  positive at 2.0x). configs/goal.yaml's `intraday` block has since been
  rewritten (2026-08-18, see its own comment there) to a pooled-PF /
  pooled-trades / hard-vs-soft split (survival PF >= 1.0 HARD via pooled
  fold concatenation, edge PF >= 1.1 SOFT, stress at 1.5x measured as PF
  not raw P&L, min_trades_per_oos_fold demoted to a soft quiet-month floor
  of 8) implemented in scripts/run_intraday_backtest.py's `run_signal` /
  `assemble_intraday_gates`. Re-deriving that policy by hand here would risk
  silently drifting from it the next time it changes. DEV-window configs
  below therefore call `run_signal` DIRECTLY (same function every other
  report in this repo uses) instead of reimplementing gate logic; only the
  HOLDOUT path (which `run_signal` cannot serve — a 2-month window is
  shorter than one WFO fold) computes gates manually, using the SAME
  goal.yaml keys read at call time so it cannot drift from a hardcoded
  1.3/2.0x either.

Resilience: per-config checkpoint JSON + disk-memoized per-fold backtests,
same as _orb_vwap_rescue.py.

Usage:
    python scripts/_fvg_retest_rescue.py --list
    python scripts/_fvg_retest_rescue.py A0_asshipped_full20
    python scripts/_fvg_retest_rescue.py B1_tight10 B2_tight6
    python scripts/_fvg_retest_rescue.py HOLDOUT_best --holdout-params '{...}' --holdout-universe A,B,C
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
    _pf_clears,
    run_signal,
)
from python.backtest.intraday_engine import IntradayBacktestConfig  # noqa: E402
from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.optimize import (  # noqa: E402
    build_intraday_backtest_fn,
    check_pooled_profit_factor_gate,
    preflight_check,
    run_intraday_stress_test,
)

SIGNAL = "fvg_retest"
CHECKPOINT_DIR = Path("backtests/reports/_fvg_rescue")
CACHE_DIR = Path("backtests/reports/_fvg_rescue_cache")
CALIBRATED_SPREADS_PATH = Path("backtests/reports/calibrated_spreads.json")

# ── windows (identical to _orb_vwap_rescue.py, same cached history) ────────
DEV_START, DEV_END = "2025-08-01", "2026-06-01"
HOLDOUT_START, HOLDOUT_END = "2026-06-01", "2026-08-01"

FULL20 = None  # None = whatever configs/universe.yaml's fixed universe holds


def _tight_universes() -> tuple[list[str], list[str]]:
    """(TIGHT10, TIGHT6): the 10/6 symbols with the lowest calibrated median
    half-spread, computed fresh from the CURRENT calibrated_spreads.json —
    not hardcoded, because configs/universe.yaml's symbol set can be (and
    has been) refreshed since any prior investigation that hardcoded a
    list."""
    payload = json.loads(CALIBRATED_SPREADS_PATH.read_text(encoding="utf-8"))
    rows = [(sym, float(s["median_bps"])) for sym, s in payload["symbols"].items() if not s.get("suspect")]
    rows.sort(key=lambda r: r[1])
    return [r[0] for r in rows[:10]], [r[0] for r in rows[:6]]


TIGHT10, TIGHT6 = _tight_universes()


@dataclass
class RescueConfig:
    config_id: str
    lever: str
    description: str
    params: dict
    universe: list[str] | None = None
    window: str = "dev"
    cost: str = "calibrated"
    code_variant: str = "levered"
    extra: dict = field(default_factory=dict)


BASE_PARAMS = {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10}


def _p(**overrides) -> dict:
    out = dict(BASE_PARAMS)
    out.update(overrides)
    return out


CONFIGS: dict[str, RescueConfig] = {}


def _add(cfg: RescueConfig) -> None:
    CONFIGS[cfg.config_id] = cfg


# Baseline — as-shipped params, full 20-symbol universe. No stop-side /
# target-side correctness fix needed here (see module docstring), so this
# IS the correct baseline, unlike orb_vwap's A0.
_add(RescueConfig(
    "A0_asshipped_full20", "baseline",
    "as-shipped fvg_retest (vol_mult=2.0, entry_pct=0.5, expiry_bars=10, "
    "max_entries=unlimited, target_r=1.0), full 20-symbol universe, DEV window",
    _p(), FULL20,
))

# Lever 1 — tight-spread universe subset.
_add(RescueConfig(
    "B1_tight10", "1: universe",
    f"tight-spread top-10 by calibrated half-spread: {TIGHT10}",
    _p(), TIGHT10,
))
_add(RescueConfig(
    "B2_tight6", "1: universe",
    f"tight-spread top-6 by calibrated half-spread: {TIGHT6}",
    _p(), TIGHT6,
))

# Lever 2 — entries-per-session cap. BEST_UNIVERSE is filled in by the
# driver after inspecting the B-row results (see main()/--best-universe).


def cap_configs(universe: list[str] | None, universe_tag: str) -> None:
    for cap in (1, 2, 3):
        _add(RescueConfig(
            f"C{cap}_cap{cap}_{universe_tag}", "2: entry cap",
            f"at most {cap} entries per symbol per session, universe={universe_tag}",
            _p(max_entries_per_session=cap), universe,
        ))


def target_configs(universe: list[str] | None, universe_tag: str, best_cap: int | None) -> None:
    base = {"max_entries_per_session": best_cap} if best_cap is not None else {}
    for r in (1.5, 2.0, 3.0):
        tag = str(r).replace(".", "")
        _add(RescueConfig(
            f"E_r{tag}_{universe_tag}", "3: target",
            f"{r}R profit target on top of cap={best_cap}, universe={universe_tag}",
            _p(target_r_multiple=r, **base), universe,
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

def _cache_tag(cfg: RescueConfig, suffix: str = "") -> str:
    uni = "full20" if cfg.universe is None else f"u{len(cfg.universe)}_{hashlib.sha1(','.join(sorted(cfg.universe)).encode()).hexdigest()[:6]}"
    return f"{cfg.code_variant}__{cfg.cost}__{uni}{suffix}"


def evaluate_dev(cfg: RescueConfig) -> dict:
    """DEV-window configs call run_signal() directly — the exact function
    scripts/run_intraday_backtest.py uses for every other signal's official
    report — so the hard/soft gate split and pooled PF/trades thresholds can
    never drift from configs/goal.yaml's current policy."""
    spreads = load_calibrated_spreads() if cfg.cost == "calibrated" else None
    start_s, end_s = DEV_START, DEV_END
    bars_by_symbol, symbols_used = load_bars(cfg.universe, start_s, end_s)
    print(f"[{cfg.config_id}] {len(symbols_used)} symbols, window [{start_s}, {end_s}), "
          f"cost={cfg.cost}, params={cfg.params}", flush=True)

    t0 = time.time()
    r = run_signal(
        SIGNAL, None, bars_by_symbol, data_label=f"fvg_retest rescue {cfg.config_id}",
        start_ts=pd.Timestamp(start_s), end_ts=pd.Timestamp(end_s),
        param_grid=[cfg.params], quiet=False,
        half_spread_bps_by_symbol=spreads,
    )
    elapsed = time.time() - t0
    print(f"[{cfg.config_id}] run_signal done in {elapsed:.0f}s -> {r['decision']}", flush=True)

    r["config_id"] = cfg.config_id
    r["lever"] = cfg.lever
    r["description"] = cfg.description
    r["window_kind"] = "dev"
    r["n_symbols"] = len(symbols_used)
    r["symbols"] = symbols_used
    r["params"] = r.get("candidate_params", cfg.params)
    r["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    return r


def evaluate_holdout(cfg: RescueConfig) -> dict:
    """The final holdout is 2 months — shorter than a single is_days=90 +
    oos_days=30 WFO fold — so run_signal() would just report SKIPPED
    ("window too short for a single WFO fold"). This evaluates it as ONE
    out-of-sample window instead, using the SAME goal.yaml keys `run_signal`
    reads (survival PF hard-gated via pooled concatenation of a single
    window == the window itself, MC p5, stress PF at the configured
    multiplier, has_trades) so this cannot drift from the live policy the
    way a hardcoded 1.3/2.0x copy would."""
    base_cfg = _load_yaml(STRATEGY_PATH)[SIGNAL]
    goal = _load_yaml(GOAL_PATH)
    intraday_goal = goal.get("intraday", {})
    min_survival_pf = float(intraday_goal.get("min_survival_profit_factor", 1.0))
    min_edge_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.1))
    min_stress_pf = float(intraday_goal.get("min_stress_profit_factor", 1.0))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    start_s, end_s = HOLDOUT_START, HOLDOUT_END
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
    full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))
    stress = run_intraday_stress_test(
        bars_by_symbol, SIGNAL, base_cfg, cfg.params, start_ts.to_pydatetime(), end_ts.to_pydatetime(),
        stress_slippage_multiplier=stress_mult, half_spread_bps_by_symbol=spreads,
    )
    print(f"[{cfg.config_id}] full-window: n_trades={full.get('n_trades')} "
          f"PF={full.get('profit_factor', 0):.3f} net={full.get('total_net_pnl', 0):,.0f} "
          f"| stress PF={stress.get('profit_factor', 0):.3f} | mc_p5={mc.sharpe.p5:+.3f}", flush=True)

    gates = {
        "survival_profit_factor": _pf_clears(full.get("profit_factor", 0.0), min_survival_pf),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        f"stress_slippage_{stress_mult:g}x_profit_factor": _pf_clears(stress.get("profit_factor", 0.0), min_stress_pf),
        "has_trades": int(full.get("n_trades", 0)) > 0,
    }
    soft_gates = {"edge_profit_factor_1.1": _pf_clears(full.get("profit_factor", 0.0), min_edge_pf)}

    return {
        "config_id": cfg.config_id, "lever": cfg.lever, "description": cfg.description,
        "params": cfg.params, "window": f"{start_s} .. {end_s}", "window_kind": "holdout",
        "cost_model": "calibrated_per_symbol" if spreads else "flat_2.0bps",
        "n_symbols": len(symbols_used), "symbols": symbols_used,
        "sample_size_check": preflight,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "full_window_metrics": {k: v for k, v in full.items() if k != "daily_returns"},
        "stress_metrics": {k: v for k, v in stress.items() if k != "daily_returns"},
        "mc_p5_sharpe": mc.sharpe.p5,
        "gates": gates, "soft_gates": soft_gates,
        "decision": "GO" if all(gates.values()) else "NO-GO",
    }


def evaluate(cfg: RescueConfig) -> dict:
    return evaluate_dev(cfg) if cfg.window == "dev" else evaluate_holdout(cfg)


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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--add-cap-configs", choices=["full20", "tight10", "tight6"],
                     help="register C1/C2/C3 cap configs on the given universe tag before running")
    ap.add_argument("--add-target-configs", choices=["full20", "tight10", "tight6"],
                     help="register E_r1.5/2/3 target configs on the given universe tag before running")
    ap.add_argument("--best-cap", type=int, default=None,
                     help="max_entries_per_session to fix when registering target configs")
    ap.add_argument("--holdout-params", help="JSON dict of frozen params for HOLDOUT_best")
    ap.add_argument("--holdout-universe", help="comma-separated symbols for HOLDOUT_best")
    ap.add_argument("--holdout-note", default="")
    args = ap.parse_args()

    tag_to_universe = {"full20": FULL20, "tight10": TIGHT10, "tight6": TIGHT6}
    if args.add_cap_configs:
        cap_configs(tag_to_universe[args.add_cap_configs], args.add_cap_configs)
    if args.add_target_configs:
        target_configs(tag_to_universe[args.add_target_configs], args.add_target_configs, args.best_cap)

    if args.list:
        for cid, c in CONFIGS.items():
            print(f"{cid:28s} [{c.lever}] {c.description}")
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
