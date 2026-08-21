"""
Round 3 of the `absorption_breakout` investigation (backtests/reports/
absorption_breakout_investigation_report.md's round-3 section) — unblocking
round 2's Lever 1 (macro/sector beta alignment filter), which round 2
reported as BLOCKED for lack of QQQ/SPY/XLK 1-minute history (no reachable
IB Gateway, yfinance capped at ~7 days). This round builds a Futu OpenD
historical-kline client (python/data/futu_price_source.py), backfills
QQQ/SPY/XLK via it (scripts/backfill_futu_symbols.py), and — since the
backfill succeeded (see report) — actually tests Lever 1 here.

## Lever 1: macro/sector beta alignment filter — IMPLEMENTED as an external
## gate (python/analytics/macro_beta_gate.py), NOT a new signal parameter

Applied via the SAME "monkeypatch the module-global evaluator name inside
python.backtest.intraday_engine for the duration of one run, then restore
it" technique this investigation's own round-1 script
(_absorption_breakout_validation.py's `_make_inverting_evaluator`) already
used for its DIAG inversion test — chosen deliberately over adding a new
tunable parameter to absorption_breakout.py because (a) the gate has NO
tunable threshold (hard "non-negative", not a fitted cutoff — see
macro_beta_gate.py's own docstring for why), matching orb_vwap_regime.py's
own hardcoded (not gridded) regime rule, and (b) absorption_breakout.py
is ALREADY at param_guard.py's MAX_FREE_PARAMETERS=5 ceiling as of round 2
(micro_stop_cents was the 5th) — a genuinely new *tunable* 6th parameter
would breach that ceiling outright, but a hardcoded, always-on-or-off
external gate does not count as a free parameter under that discipline
(the same reason orb_vwap_regime's regime threshold never appeared in
SIGNAL_PARAM_KEYS either).

Diagnosis-first, not aggregate-PF-only: since round 1/2's diagnosed failure
mode is P&L CONCENTRATION (holdout's top 5-of-204 trades exceed the entire
net P&L; DEV-window B5's own Monte Carlo p5 Sharpe is deeply negative at
-3.975), this script reports a trade-concentration diagnostic (top-5-trade
share of total net P&L, win rate, trade count) for every config, not just
PF, per the investigation brief's explicit instruction.

## Pre-declared selection rule (declared HERE, before E1 was run)

Primary metric = full-window GROSS profit factor over B5's own DEV value
(0.966) — IDENTICAL metric and IDENTICAL >=0.05 improvement bar round 2's
own micro-stop test (C1/C2 vs B2_tight6) used, for direct cross-round
comparability. E2 (macro gate STACKED on round 2's best micro_stop_cents,
C2's 0.02) is only registered/run if E1 clears this bar — mechanically
enforced in main(), identical to round 2's D1 gating logic.

## Holdout discipline (identical to round 2, re-verified here)

round 1's ONLY reserved holdout (2026-06-01..2026-08-01) has already been
evaluated exactly once (HOLDOUT_best.json) — NEVER touched by this script.
`check_holdout_freshness()` re-runs round 2's exact freshness check against
CURRENT disk state (as of 2026-08-15, one day later than round 2's check) —
see its own docstring for what changed.

Resilience: per-config checkpoint JSON in
`backtests/reports/_absorption_breakout_round3/`; every individual
backtest_fn(start, end, params) call is memoized to disk under
`backtests/reports/_absorption_breakout_round3_cache/`.

Usage:
    python scripts/_absorption_breakout_round3.py --check-freshness
    python scripts/_absorption_breakout_round3.py --list
    python scripts/_absorption_breakout_round3.py E1_macrogate_on_b5
    python scripts/_absorption_breakout_round3.py E2_macrogate_plus_microstop_on_b5
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
from _absorption_breakout_validation import (  # noqa: E402
    DEV_END, DEV_START, HOLDOUT_END, HOLDOUT_START, TIGHT6,
    load_calibrated_spreads, load_bars,
)
import python.backtest.intraday_engine as intraday_engine_mod  # noqa: E402
from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    metrics_from_report,
    run_intraday_backtest,
)
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
from python.analytics.macro_beta_gate import DEFAULT_INDEX_SYMBOLS, compute_macro_momentum, macro_gate_ok  # noqa: E402

SIGNAL = "absorption_breakout"
CHECKPOINT_DIR = Path("backtests/reports/_absorption_breakout_round3")
CACHE_DIR = Path("backtests/reports/_absorption_breakout_round3_cache")
TICKS_DIR = Path("data/ticks")

# Round 1's frozen DEV winner (B5_tight6_clearance) — the base every round-3
# config below starts from, UNCHANGED. micro_stop_cents added here explicitly
# (None) for symmetry with round 2's BASE_PARAMS even though B5 predates it.
B5_PARAMS = {"volume_mult": 3.0, "breakout_atr_mult": 0.5, "stop_atr_mult": 0.5,
             "target_r_multiple": None, "micro_stop_cents": None}
# Round 2's winning (but round-2-threshold-missing) micro-stop value, reused
# here only if E1 clears this round's own pre-declared bar.
ROUND2_BEST_MICRO_STOP_CENTS = 0.02

# Round 2 found 5 trading days of raw tick data strictly after round 1's
# holdout (2026-08-04/05/06/12/13). Re-checked fresh here, one day later.
_PRIOR_ROUND_FRESH_TICK_DATES = ["20260804", "20260805", "20260806", "20260812", "20260813"]


def _p(**overrides) -> dict:
    out = dict(B5_PARAMS)
    out.update(overrides)
    return out


# ── holdout-freshness re-check (round 2's check, re-run against today's disk) ──

def check_holdout_freshness() -> dict:
    """Re-verifies round 2's freshness finding against CURRENT disk state.
    Does NOT touch round 1's holdout checkpoint/window. Two things can only
    have changed since round 2 (2026-08-14): (a) more live-captured tick
    days may have accumulated under data/ticks/, (b) this round's OWN Futu
    backfill added QQQ/SPY/XLK bars under data/history_1m/ — but that does
    NOT create a fresh holdout for the TRADED symbols (AAPL/GOOGL/NVDA/MSFT/
    PLTR/INTC), which remain IBKR-sourced and capped at 2026-07 (see
    canonical_1m_bar_cache below) — the macro ETFs' own fresher coverage is
    irrelevant to whether the SIGNAL's underlying instruments have new bars
    to trade on."""
    universe_dir = Path("data/history_1m")
    last_months = {}
    for sym_dir in sorted(universe_dir.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name in DEFAULT_INDEX_SYMBOLS:
            continue  # exclude this round's own QQQ/SPY/XLK backfill — not a traded symbol
        months = sorted(p.stem for p in sym_dir.glob("*.parquet"))
        if months:
            last_months[sym_dir.name] = months[-1]

    tick_symbols = sorted(p.name for p in TICKS_DIR.iterdir()) if TICKS_DIR.exists() else []
    tick_dates_by_symbol = {}
    for sym in tick_symbols:
        dates = sorted(p.stem for p in (TICKS_DIR / sym).glob("*.jsonl"))
        tick_dates_by_symbol[sym] = dates
    common_tick_dates = sorted(set.intersection(*[set(d) for d in tick_dates_by_symbol.values()])) \
        if tick_dates_by_symbol else []
    new_since_round2 = sorted(set(common_tick_dates) - set(_PRIOR_ROUND_FRESH_TICK_DATES))

    n_days = len(common_tick_dates)
    summary = {
        "canonical_1m_bar_cache_traded_symbols": {
            "last_closed_month_per_symbol": last_months,
            "uniform_last_month": sorted(set(last_months.values())),
            "verdict": ("Still no new 1-minute bars for the TRADED (non-macro) symbols past round "
                        "1's holdout window -- IB Gateway remains the only backfill path for the fixed "
                        "trading universe (kept IBKR-sourced for backtest-vs-live consistency; this "
                        "round's Futu client was scoped ONLY to QQQ/SPY/XLK, per the task brief -- not "
                        "used to re-source the trading universe itself)."),
        },
        "raw_tick_data": {
            "symbols_captured": len(tick_symbols),
            "dates_common_to_all_symbols": common_tick_dates,
            "n_fresh_trading_days": n_days,
            "new_dates_since_round2": new_since_round2,
            "verdict": (
                f"{n_days} trading days of raw tick data now exist strictly after round 1's holdout "
                f"({'no new days since round 2' if not new_since_round2 else f'{len(new_since_round2)} new day(s) since round 2: ' + ', '.join(new_since_round2)}). "
                "Still far too few (non-contiguous, single-digit trading days) to form even one 30-day "
                "OOS fold or a meaningful Monte Carlo bootstrap -- same conclusion as round 2, not "
                "re-used as a real holdout here."
            ),
        },
        "decision": (
            "No adequately-sized fresh holdout available as of this round either. Lever 1 diagnosis/"
            "selection below uses ONLY the DEV-window WFO folds (2025-08-01..2026-06-01, identical to "
            "rounds 1-2); round 1's reserved holdout (2026-06-01..2026-08-01) is NEVER evaluated a "
            "second time. Any round-3 'improved' configuration is IN-SAMPLE-ONLY evidence pending "
            "either a real IB-Gateway-backed forward extension of the canonical 1-minute cache, or "
            "live/paper forward validation."
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


# ── macro momentum loading (this round's own QQQ/SPY/XLK Futu-sourced cache) ──

def load_macro_momentum(start: str, end: str) -> pd.DataFrame:
    """DEV-window (or any [start, end)) composite QQQ/SPY/XLK 1m/5m momentum
    — see python/analytics/macro_beta_gate.py. Loaded from
    data/history_1m/<QQQ|SPY|XLK>/ (this round's own Futu-sourced backfill,
    python/data/futu_price_source.py), via the SAME
    get_cached_intraday_panel helper every other symbol in this repo uses —
    no special-casing of the data source at the read layer."""
    from python.data.intraday_cache import get_cached_intraday_panel

    panel = get_cached_intraday_panel(list(DEFAULT_INDEX_SYMBOLS), start, end)
    codes = set(panel.index.get_level_values("code"))
    index_bars = {s: panel.xs(s, level="code").sort_index() for s in DEFAULT_INDEX_SYMBOLS if s in codes}
    missing = set(DEFAULT_INDEX_SYMBOLS) - set(index_bars)
    if missing:
        raise RuntimeError(f"load_macro_momentum: missing cached bars for {sorted(missing)} in [{start}, {end})")
    return compute_macro_momentum(index_bars)


def _make_macro_gated_evaluator(original_evaluate_absorption_breakout, macro_momentum: pd.DataFrame):
    """Wraps the REAL evaluate_absorption_breakout: run it unmodified first
    (entry trigger, stop, target all byte-for-byte identical), THEN discard
    the signal (return None) if the macro gate blocks its direction at its
    OWN signal_time. Same "wrap, don't touch the signal module" technique as
    round 1's `_make_inverting_evaluator` for evaluate_l2_absorption."""
    def _gated_evaluate_absorption_breakout(bars, symbol: str = "", **kwargs):
        sig = original_evaluate_absorption_breakout(bars, symbol=symbol, **kwargs)
        if sig is None:
            return None
        if not macro_gate_ok(macro_momentum, sig.direction, sig.signal_time):
            return None
        return sig
    return _gated_evaluate_absorption_breakout


def _trade_concentration(trades: list, top_n: int = 5) -> dict:
    """Diagnostic for the SPECIFIC failure mode this lever targets (P&L
    concentration in a handful of large trades -> Monte Carlo p5 fragility)
    -- not just aggregate PF. `net_without_top_n` answers exactly the
    question round 1's §5.1 asked of the holdout: does the edge survive
    removing its biggest winners?"""
    pnls = sorted((float(t.net_pnl) for t in trades), reverse=True)
    total_net = sum(pnls)
    top = pnls[:top_n]
    wins = [p for p in pnls if p > 0]
    return {
        "n_trades": len(pnls),
        "total_net_pnl": total_net,
        f"top_{top_n}_trades_sum": sum(top),
        f"top_{top_n}_trades_share_of_total_net_pnl": (sum(top) / total_net if total_net != 0 else None),
        f"net_pnl_excluding_top_{top_n}": total_net - sum(top),
        "win_rate": (len(wins) / len(pnls) if pnls else None),
        "avg_winner": (sum(wins) / len(wins) if wins else None),
        "avg_loser": (sum(p for p in pnls if p <= 0) / max(len(pnls) - len(wins), 1) if len(pnls) > len(wins) else None),
    }


# ── disk memoization (identical pattern to prior rounds) ────────────────────

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


@dataclass
class R3Config:
    config_id: str
    lever: str
    description: str
    params: dict
    universe: list[str]
    apply_macro_gate: bool = True


CONFIGS: dict[str, R3Config] = {}


def _add(cfg: R3Config) -> None:
    CONFIGS[cfg.config_id] = cfg


_add(R3Config(
    "E1_macrogate_on_b5", "1: macro/sector beta alignment filter (isolated)",
    "Round 1's B5 base (TIGHT6, breakout_atr_mult=0.5, stop_atr_mult=0.5) + the macro/sector beta "
    "alignment gate (QQQ/SPY/XLK composite 1m AND 5m momentum non-negative in the trade's direction) "
    "-- ONLY this lever added on top of B5, isolating its own marginal effect exactly as round 1's B3 "
    "isolated clearance and round 2's C1/C2 isolated the micro-stop.",
    dict(B5_PARAMS), TIGHT6, apply_macro_gate=True,
))
# E2 is registered dynamically in main() ONLY if E1 clears the pre-declared
# bar (>=0.05 gross PF improvement over B5) -- see module docstring's
# "Pre-declared selection rule" and main()'s enforcement of it.


def evaluate_dev(cfg: R3Config, macro_momentum: pd.DataFrame | None) -> dict:
    base_cfg = _load_yaml(STRATEGY_PATH)[SIGNAL]
    goal = _load_yaml(GOAL_PATH)
    intraday_goal = goal.get("intraday", {})
    min_trades = int(intraday_goal.get("min_trades_per_oos_fold", 100))
    min_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.3))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    start_ts, end_ts = pd.Timestamp(DEV_START), pd.Timestamp(DEV_END)
    spreads = load_calibrated_spreads()
    bars_by_symbol, symbols_used = load_bars(cfg.universe, DEV_START, DEV_END)
    print(f"[{cfg.config_id}] {len(symbols_used)} symbols, window [{DEV_START}, {DEV_END}) (DEV, "
          f"NEVER the holdout), params={cfg.params}, macro_gate={cfg.apply_macro_gate}", flush=True)

    param_grid = [cfg.params]
    preflight = preflight_check(SIGNAL, base_cfg, param_grid,
                                total_trading_days=len(pd.bdate_range(start_ts, end_ts)))

    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads)
    stress_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads, stress_slippage_multiplier=stress_mult)

    result: dict = {
        "config_id": cfg.config_id, "lever": cfg.lever, "description": cfg.description,
        "params": cfg.params, "apply_macro_gate": cfg.apply_macro_gate,
        "window": f"{DEV_START} .. {DEV_END}", "window_kind": "dev",
        "cost_model": "calibrated_per_symbol", "n_symbols": len(symbols_used), "symbols": symbols_used,
        "sample_size_check": preflight, "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Monkeypatch the module-global `evaluate_absorption_breakout` name
    # inside python.backtest.intraday_engine for the ENTIRE duration of this
    # config's WFO + full-window + stress runs, then unconditionally restore
    # it (try/finally) -- identical discipline to round 1's DIAG inversion
    # patch. One precomputed `macro_momentum` DataFrame is shared across
    # every fold/symbol/bar in this run (the gate's feature is
    # symbol-independent -- same QQQ/SPY/XLK series regardless of which
    # traded symbol is being evaluated).
    original_fn_ref = intraday_engine_mod.evaluate_absorption_breakout
    if cfg.apply_macro_gate:
        assert macro_momentum is not None, "apply_macro_gate=True requires macro_momentum"
        intraday_engine_mod.evaluate_absorption_breakout = _make_macro_gated_evaluator(
            original_fn_ref, macro_momentum,
        )
    try:
        fn = _memoize(build_intraday_backtest_fn(bars_by_symbol, SIGNAL, base_cfg, engine_cfg=engine_cfg),
                      f"dev__{cfg.config_id}")
        stress_fn = _memoize(
            build_intraday_backtest_fn(bars_by_symbol, SIGNAL, base_cfg, engine_cfg=stress_cfg),
            f"dev__{cfg.config_id}__stress{stress_mult:g}x")

        wfo_cfg = load_wfo_config(SIGNAL)
        t0 = time.time()
        wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
        print(f"[{cfg.config_id}] WFO done in {time.time()-t0:.0f}s: "
              f"{wfo.passing_folds}/{wfo.total_folds} folds, OOS Sharpe mean {wfo.oos_sharpe_mean:+.3f}",
              flush=True)

        full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)
        mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))
        stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)

        # Trade-level concentration diagnostic: re-run un-memoized (cheap --
        # one pass, not a WFO/grid sweep) directly through run_intraday_backtest
        # so we have access to individual trades, which the memoized
        # metrics-only backtest_fn() above deliberately discards.
        warmup_start = start_ts - pd.Timedelta(days=1)
        sliced = {s: b.loc[(b.index >= warmup_start) & (b.index < end_ts)] for s, b in bars_by_symbol.items()}
        sliced = {s: b for s, b in sliced.items() if not b.empty}
        report = run_intraday_backtest(sliced, SIGNAL, cfg.params, engine_cfg)
        in_window_trades = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
        concentration = _trade_concentration(in_window_trades, top_n=5)
    finally:
        intraday_engine_mod.evaluate_absorption_breakout = original_fn_ref
        assert intraday_engine_mod.evaluate_absorption_breakout is original_fn_ref

    print(f"[{cfg.config_id}] full-window: n_trades={full.get('n_trades')} "
          f"PF_net={full.get('profit_factor', 0):.3f} PF_gross={full.get('profit_factor_gross', 0):.3f} "
          f"net_pnl={full.get('total_net_pnl', 0):,.0f} gross_pnl={full.get('gross_pnl', 0):,.0f} "
          f"| stress net={stress.get('total_net_pnl', 0):,.0f} | mc_p5={mc.sharpe.p5:+.3f} "
          f"| top5_share={concentration.get('top_5_trades_share_of_total_net_pnl')}", flush=True)

    result.update({
        "wfo_folds": wfo.total_folds, "wfo_passing_folds": wfo.passing_folds,
        "wfo_pass_ratio": wfo.pass_ratio, "oos_sharpe_mean": wfo.oos_sharpe_mean,
        "fold_oos_sharpes": [f.oos_sharpe for f in wfo.folds],
        "fold_oos_trades": [int(f.oos_metrics.get("n_trades", 0)) for f in wfo.folds],
        "full_window_metrics": {k: v for k, v in full.items() if k != "daily_returns"},
        "stress_metrics": {k: v for k, v in stress.items() if k != "daily_returns"},
        "mc_p5_sharpe": mc.sharpe.p5,
        "trade_concentration_diagnostic": concentration,
    })
    gates = {
        "wfo_go": wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(wfo),
        "min_trades_per_oos_fold": check_min_trades_gate(wfo, min_trades),
        "cost_adjusted_profit_factor": check_profit_factor_gate(wfo, min_pf),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        f"stress_slippage_{stress_mult:g}x_net_positive": stress["total_net_pnl"] > 0,
    }
    result["gates"] = gates
    result["decision"] = "GO" if all(gates.values()) else "NO-GO"
    return result


def checkpoint_path(config_id: str) -> Path:
    return CHECKPOINT_DIR / f"{config_id}.json"


def run_config(cfg: R3Config, macro_momentum: pd.DataFrame | None, force: bool = False) -> dict:
    path = checkpoint_path(cfg.config_id)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f">>> {cfg.config_id} already checkpointed ({existing.get('decision')}) — skipping", flush=True)
        return existing
    t0 = time.time()
    result = evaluate_dev(cfg, macro_momentum)
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
    ap.add_argument("config_ids", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check-freshness", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.check_freshness:
        check_holdout_freshness()
        return

    if args.list:
        for cid, c in CONFIGS.items():
            print(f"{cid:36s} [{c.lever}] {c.description}")
        return

    # Every registered config in this script uses the macro gate, so just
    # compute it once up front whenever any config is requested.
    macro_momentum = load_macro_momentum(DEV_START, DEV_END) if args.config_ids else None

    for cid in args.config_ids:
        if cid == "E2_macrogate_plus_microstop_on_b5":
            e1_path = checkpoint_path("E1_macrogate_on_b5")
            if not e1_path.exists():
                raise SystemExit("E2 requires E1_macrogate_on_b5 to be run first (see --list)")
            e1 = json.loads(e1_path.read_text(encoding="utf-8"))
            e1_gross_pf = float(e1["full_window_metrics"]["profit_factor_gross"])
            b5 = json.loads(Path("backtests/reports/_absorption_breakout_validation/B5_tight6_clearance.json")
                             .read_text(encoding="utf-8"))
            b5_gross_pf = float(b5["full_window_metrics"]["profit_factor_gross"])
            print(f"[E2 gate check] E1 gross PF {e1_gross_pf:.3f} vs B5 control {b5_gross_pf:.3f} "
                  f"(delta {e1_gross_pf - b5_gross_pf:+.3f}, bar is >=+0.05, pre-declared BEFORE E1 ran)")
            if e1_gross_pf - b5_gross_pf < 0.05:
                raise SystemExit(
                    f"PRE-DECLARED SELECTION RULE NOT MET: E1 did not beat B5 by >=0.05 gross PF -- "
                    "per this script's own docstring, E2 is not run; Lever 1's isolated result is "
                    "reported as negative/inconclusive instead."
                )
            _add(R3Config(
                "E2_macrogate_plus_microstop_on_b5", "1+2: macro gate AND micro-stop combined",
                f"E1's macro gate STACKED onto round 2's best micro_stop_cents "
                f"({ROUND2_BEST_MICRO_STOP_CENTS}) on top of round 1's B5 base -- only run because E1 "
                "cleared this round's own pre-declared >=0.05 gross-PF bar over B5.",
                _p(micro_stop_cents=ROUND2_BEST_MICRO_STOP_CENTS), TIGHT6, apply_macro_gate=True,
            ))
        if cid not in CONFIGS:
            raise SystemExit(f"unknown config id {cid!r} (use --list)")
        run_config(CONFIGS[cid], macro_momentum, force=args.force)


if __name__ == "__main__":
    main()
