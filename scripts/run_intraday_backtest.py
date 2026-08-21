"""
Intraday microstructure signal WFO validation CLI
(docs/microstructure_pivot_plan.md §4c).

Runs walk-forward optimization for sweep_reclaim / fvg_retest / orb_vwap
against the fixed universe's cached 1-minute bars (python/data/
intraday_cache.py, built by scripts/backfill_intraday.py), applies the
intraday research GO (configs/goal.yaml `intraday` block: survival AND of
drawdown / has_oos_trades / pooled PF >= 1.0 / stress PF, with WFO,
trade-count, Monte Carlo p5, and the 1.1 edge PF recorded as warnings),
and writes an honest GO/NO-GO report to backtests/reports/intraday_backtest_report.md.

Same discipline as scripts/run_backtest.py / self_improve_loop.py: report-
only, no config write-back — a signal earning a GO here is evidence to
review, not an automatic promotion (no auto_execute flip happens anywhere
in this script).

Runtime note: this is an offline research tool, not something meant to run
inside a request/response loop — one signal's full param_grids.yaml grid
search across a real ~12-month/20-symbol universe (or the multi-month
--demo window) can take anywhere from a couple of minutes to well over an
hour depending on how "busy" (trades/session) that signal turns out to be
on the given data, since python/backtest/intraday_engine.py's per-bar cost
model overhead scales with total fill count, not just bar count. Run it in
the background and check backtests/reports/intraday_backtest_report.md when
done, rather than waiting on it interactively.

RETIRED (2026-08-13, `l2_absorption` added 2026-08-14): all seven signals
this CLI covers (`sweep_reclaim`, `fvg_retest`, `orb_vwap`,
`orb_vwap_regime`, `vwap_band_fade`, `vp_breakout`, `l2_absorption`) are
confirmed NO-GO with no remaining unexplored rescue angle — see
`backtests/reports/strategy_review_summary.md`,
`backtests/reports/orb_vwap_rescue_report.md`, and
`backtests/reports/l2_absorption_validation_report.md`. `--signal all` /
`--signal new` (the batch/default modes) now no-op unless
`--include-retired` is passed, so this script stops burning compute by
default (this was already true before `l2_absorption` was added — neither
batch mode ever iterated it). Naming an individual signal explicitly
(`--signal <name>`) always still runs it in full — the code, tests, and
this CLI itself all stay correct and usable for a deliberate re-run; see
`RETIRED_SIGNALS` below and `backtests/reports/signal_status.md` for the
single-page status manifest.

Usage:
    python scripts/run_intraday_backtest.py --demo
    python scripts/run_intraday_backtest.py --signal sweep_reclaim --start 2025-08-01 --end 2026-07-01
    python scripts/run_intraday_backtest.py --signal all --include-retired --start 2025-08-01 --end 2026-07-01
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

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
log = logging.getLogger("run_intraday_backtest")

from python.backtest.intraday_engine import IntradayBacktestConfig
from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import (
    build_intraday_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    check_min_trades_gate,
    check_pooled_profit_factor_gate,
    check_pooled_trades_gate,
    check_profit_factor_gate,
    load_param_grid,
    load_wfo_config,
    max_oos_drawdown_threshold,
    preflight_check,
    run_intraday_stress_test,
)
from python.backtest.walk_forward import WalkForwardOptimizer

SIGNALS = ["sweep_reclaim", "fvg_retest", "orb_vwap"]
# New signal hypotheses (2026-08-06, backtests/reports/new_signals_report.md)
# — genuinely new ideas, NOT re-tuned variants of the three signals above
# (all three of which are already NO-GO, see SIGNALS' report). Kept in a
# SEPARATE list/report rather than merged into SIGNALS so the original
# three-signal report's content stays exactly what it was.
NEW_SIGNALS = ["orb_vwap_regime", "vwap_band_fade", "vp_breakout"]

# l2_absorption (S4, Phase 3) — kept in its OWN list, separate from
# SIGNALS/NEW_SIGNALS, so it is always explicitly runnable by name
# (`--signal l2_absorption`, no --include-retired needed) and never joins
# the "all"/"new" batch aliases (which only ever iterate SIGNALS/
# NEW_SIGNALS) — it also gets its own report path
# (backtests/reports/l2_absorption_backtest_report.md, see main()) rather
# than sharing intraday_backtest_report.md/new_signals_report.md.
L2_ABSORPTION_SIGNALS = ["l2_absorption"]
# auction_reclaim (2026-08-18) — Creamer-style investigation. Own list so
# `--signal auction_reclaim` is always runnable and never joins the retired
# `--signal all` / `--signal new` batches.
AUCTION_RECLAIM_SIGNALS = ["auction_reclaim"]
# vsa_effort (2026-08-18) — Wyckoff/VSA investigation. Own list so
# `--signal vsa_effort` is always runnable and never joins retired batches.
VSA_EFFORT_SIGNALS = ["vsa_effort"]
# Volume-book signals (2026-08-18) — Williams/Coulling no-demand and
# Granville OBV B-2/S-2. Own list so they never join retired batches.
VOLUME_BOOK_SIGNALS = ["vsa_no_demand", "obv_divergence"]

# ── RETIRED_SIGNALS (2026-08-13, l2_absorption added 2026-08-14) ───────────
# ALL SEVEN signals below (SIGNALS + NEW_SIGNALS + L2_ABSORPTION_SIGNALS) are
# confirmed NO-GO with no remaining unexplored rescue angle — see
# backtests/reports/strategy_review_summary.md (full diagnostic review),
# backtests/reports/orb_vwap_rescue_report.md (orb_vwap's dedicated rescue
# attempt, also NO-GO), and backtests/reports/l2_absorption_validation_report.md
# (l2_absorption's end-to-end validation + 4 rescue levers, also NO-GO —
# the WORST of the seven: gross, pre-cost profit factor never exceeds 0.69
# in any configuration tested, unlike orb_vwap's genuine-but-cost-killed
# edge). Single-page status manifest: backtests/reports/signal_status.md.
#
# This list exists to stop `main()`'s batch modes (`--signal all` / `--signal
# new`, i.e. the modes a scheduled/default run would use) from re-running
# SIGNALS/NEW_SIGNALS by default and burning compute/attention on an
# already-settled question — see main()'s `--include-retired` handling
# below. `l2_absorption`'s inclusion here does NOT change `--signal all`/
# `--signal new` behavior (neither batch alias ever iterated
# L2_ABSORPTION_SIGNALS to begin with) — it only makes the "NOTE: ... is
# RETIRED" message below print for it too, now that it is genuinely
# retired. Retirement does NOT remove anything: every signal's evaluate_*
# function, param grid (configs/param_grids.yaml), and test coverage
# (tests/test_intraday_signals.py, tests/test_new_intraday_signals.py) stay
# exactly as they were, and naming a retired signal explicitly
# (`--signal sweep_reclaim`, `--signal l2_absorption`, etc.) always still
# runs it end-to-end — retirement only changes what runs WITHOUT being
# asked for by name.
RETIRED_SIGNALS = SIGNALS + NEW_SIGNALS + L2_ABSORPTION_SIGNALS
# orb_vwap_regime needs extra calendar-day warmup beyond the standard 1-day
# (python/backtest/optimize.py's build_intraday_backtest_fn default) so its
# regime gate (python/backtest/intraday_engine.py:_daily_trending_flags,
# trailing 20 TRADING days) has real trailing daily-close history at the
# START of every WFO fold's IS/OOS window, not just the first fold — 35
# calendar days comfortably covers 20 trading days with a buffer for
# weekends/holidays. Every other signal here needs no more than the default.
SIGNAL_WARMUP_DAYS = {"orb_vwap_regime": 35}
GOAL_PATH = Path("configs/goal.yaml")
STRATEGY_PATH = Path("configs/strategy.yaml")
REPORT_PATH = Path("backtests/reports/intraday_backtest_report.md")
# Machine-readable sibling of REPORT_PATH — same run, same content, JSON
# instead of markdown. Exists specifically so downstream tooling (e.g.
# scripts/sync_uhai.py's distillation into GreyCat triples) has a stable
# structured source instead of parsing the markdown report.
REPORT_JSON_PATH = Path("backtests/reports/intraday_backtest_report.json")
NEW_REPORT_PATH = Path("backtests/reports/new_signals_report.md")
NEW_REPORT_JSON_PATH = Path("backtests/reports/new_signals_report.json")
# l2_absorption's OWN CLI-driven report — separate from both the above (this
# is the raw `run_signal()`/`_render_signal_section()` output of a plain
# `--signal l2_absorption` invocation). The AUTHORITATIVE l2_absorption
# writeup is backtests/reports/l2_absorption_validation_report.md, produced
# by scripts/_l2_absorption_validation.py (which covers the parameter-grid
# search, variant levers, and the reserved holdout — this CLI's single-run
# output is a useful cross-check/reproduction path, not the primary report).
L2_REPORT_PATH = Path("backtests/reports/l2_absorption_backtest_report.md")
L2_REPORT_JSON_PATH = Path("backtests/reports/l2_absorption_backtest_report.json")
AUCTION_RECLAIM_REPORT_PATH = Path("backtests/reports/auction_reclaim_backtest_report.md")
AUCTION_RECLAIM_REPORT_JSON_PATH = Path("backtests/reports/auction_reclaim_backtest_report.json")
VSA_EFFORT_REPORT_PATH = Path("backtests/reports/vsa_effort_backtest_report.md")
VSA_EFFORT_REPORT_JSON_PATH = Path("backtests/reports/vsa_effort_backtest_report.json")
VOLUME_BOOK_REPORT_PATH = Path("backtests/reports/volume_book_signals_backtest_report.md")
VOLUME_BOOK_REPORT_JSON_PATH = Path("backtests/reports/volume_book_signals_backtest_report.json")


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pf_clears(value, floor: float) -> bool:
    """True when a profit-factor value meets `floor`. inf (all winners,
    no losses) clears; NaN / missing / unparseable does not."""
    try:
        pf = float(value)
    except (TypeError, ValueError):
        return False
    if pf != pf:
        return False
    return pf >= floor


def assemble_intraday_gates(
    *,
    wfo_go: bool,
    oos_drawdown_ok: bool,
    has_oos_trades: bool,
    min_trades_ok: bool,
    survival_pf_ok: bool,
    edge_pf_ok: bool,
    mc_ok: bool,
    stress_mult: float,
    stress_pf_ok: bool,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Split official intraday research checks into hard vs soft.

    Hard AND flips GO/NO-GO: drawdown, has_oos_trades, survival PF, and
    stress PF (1.5x costs still PF >= floor — not "net PnL > 0").
    Soft is recorded only: WFO, pooled trade count, Monte Carlo p5, and
    the 1.1 edge PF bar.
    """
    hard = {
        "oos_drawdown_within_limit": oos_drawdown_ok,
        "has_oos_trades": has_oos_trades,
        "cost_adjusted_profit_factor": survival_pf_ok,
        f"stress_slippage_{stress_mult:g}x_pf_ge_1": stress_pf_ok,
    }
    soft = {
        "wfo_go": wfo_go,
        "min_trades_per_oos_fold": min_trades_ok,
        "edge_profit_factor": edge_pf_ok,
        "monte_carlo_p5_sharpe": mc_ok,
    }
    return hard, soft


def _synthetic_intraday_bars(symbols: list[str], n_days: int = 90, seed: int = 11) -> dict[str, pd.DataFrame]:
    """Offline synthetic 1-minute RTH bars for --demo — PIPELINE validation
    only, same honesty convention as run_backtest.py's _synthetic_panel: a
    pure random walk has no genuine microstructure edge baked in, so a
    NO-GO here demonstrates the gates are working, not that the real
    signals are bad."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        sym_rng = np.random.default_rng(seed + i + 1)
        frames = []
        price = 50.0 + sym_rng.uniform(-20, 20)
        for d in days:
            idx = pd.date_range(f"{d.date()} 09:30", periods=390, freq="1min")
            rets = sym_rng.normal(0, 0.0006, 390)
            closes = price * np.cumprod(1 + rets)
            opens = np.concatenate([[price], closes[:-1]])
            highs = np.maximum(opens, closes) * (1 + np.abs(sym_rng.normal(0, 0.0004, 390)))
            lows = np.minimum(opens, closes) * (1 - np.abs(sym_rng.normal(0, 0.0004, 390)))
            volumes = sym_rng.integers(500, 5000, 390).astype(float)
            frames.append(pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=idx,
            ))
            price = float(closes[-1])
        out[sym] = pd.concat(frames)
    return out


def _load_real_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    from python.data.intraday_cache import get_cached_intraday_panel

    panel = get_cached_intraday_panel(symbols, start, end)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym in panel.index.get_level_values("code"):
            out[sym] = panel.xs(sym, level="code").sort_index()
    return out


def _load_bars_for_args(args) -> tuple[dict, str, pd.Timestamp, pd.Timestamp]:
    """Loads the (symbols, bars) universe ONCE per script invocation — every
    signal run against the same --start/--end/--demo flags shares the exact
    same window/universe (apples-to-apples comparability requirement), so
    there is no reason to re-read 20 symbols x 12 months of parquet per
    signal (previously: once per `run_signal` call)."""
    if args.demo:
        symbols = [f"SYN{i:02d}" for i in range(5)]
        bars_by_symbol = _synthetic_intraday_bars(symbols, n_days=90)
        data_label = "SYNTHETIC DEMO DATA — pipeline validation only, no real edge implied"
        start_ts = pd.Timestamp("2025-01-02")
        end_ts = pd.Timestamp("2025-01-02") + pd.Timedelta(days=140)
        return bars_by_symbol, data_label, start_ts, end_ts

    from python.data.fixed_universe import load_universe_config

    universe_cfg = load_universe_config()
    symbols = universe_cfg["symbols"]
    bars_by_symbol = _load_real_bars(symbols, args.start, args.end)
    if not bars_by_symbol:
        raise RuntimeError(
            f"no cached 1-minute bars for any universe symbol in [{args.start}, {args.end}] — "
            "run scripts/backfill_intraday.py first"
        )
    data_label = (f"fixed top-{universe_cfg['top_n']} universe "
                  f"(computed_at={universe_cfg['computed_at']}), 1m bars via data/history_1m/")
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    return bars_by_symbol, data_label, start_ts, end_ts


def run_signal(
    signal_name: str,
    args,
    bars_by_symbol: dict,
    data_label: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    param_grid: list[dict] | None = None,
    quiet: bool = False,
    half_spread_bps_by_symbol: dict[str, float] | None = None,
    engine_cfg: IntradayBacktestConfig | None = None,
) -> dict:
    """`param_grid=None` (the default) uses configs/param_grids.yaml's full
    grid, with genuine per-fold re-optimization (WalkForwardOptimizer picks
    the best IS candidate each fold) — this is the normal "what would we
    actually trade" path. Passing a SINGLE-candidate grid instead (see
    `run_full_grid_search` below) forces every fold to that one candidate,
    which is how the "full grid search" report section is built: run the
    WFO/MC/stress pipeline once per grid candidate, independently, so a
    marginal-looking result cannot be explained by per-fold cherry-picking
    across the grid.

    `half_spread_bps_by_symbol` (default None -> identical flat-cost
    behavior as before this parameter existed) plugs a calibrated
    per-symbol half-spread override (scripts/calibrate_slippage_spreads.py)
    into BOTH the main WFO run and the stress re-run below — see
    IntradayBacktestConfig.half_spread_bps_by_symbol's docstring in
    python/backtest/intraday_engine.py."""
    base_cfg = _load_yaml(STRATEGY_PATH)[signal_name]
    goal = _load_yaml(GOAL_PATH)
    if param_grid is None:
        param_grid = load_param_grid(signal_name)
    wfo_cfg = load_wfo_config(signal_name)

    n_bdays = len(pd.bdate_range(start_ts, end_ts))
    preflight = preflight_check(signal_name, base_cfg, param_grid, total_trading_days=n_bdays)

    warmup_days = SIGNAL_WARMUP_DAYS.get(signal_name, 1)
    engine_cfg = engine_cfg or IntradayBacktestConfig(
        half_spread_bps_by_symbol=half_spread_bps_by_symbol,
    )
    if half_spread_bps_by_symbol and not engine_cfg.half_spread_bps_by_symbol:
        engine_cfg = IntradayBacktestConfig(
            half_spread_bps_by_symbol=half_spread_bps_by_symbol,
            chart_minutes=engine_cfg.chart_minutes,
            time_stop_minutes=engine_cfg.time_stop_minutes,
            signal_filter_overrides=dict(engine_cfg.signal_filter_overrides),
        )
    fn = build_intraday_backtest_fn(bars_by_symbol, signal_name, base_cfg, engine_cfg=engine_cfg, warmup_days=warmup_days)

    if not quiet:
        print(f"\n=== {signal_name} | window [{start_ts.date()}, {end_ts.date()}] | "
              f"{len(param_grid)} candidates | {len(bars_by_symbol)} symbols ===")
        print(f"    data: {data_label}")

    wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
    if not quiet:
        wfo.print_summary()

    if not wfo.folds:
        return {"signal": signal_name, "decision": "SKIPPED",
                "reason": "window too short for a single WFO fold", "data_label": data_label}

    # Most-recent fold's winner = the parameters we would trade tomorrow —
    # same convention as scripts/self_improve_loop.py.
    candidate_params = dict(wfo.folds[-1].best_params)
    full_metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), candidate_params)
    mc_result = MonteCarloValidator(n_sims=500).run(full_metrics.get("daily_returns", []))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    intraday_goal = goal.get("intraday", {})
    min_trades = int(intraday_goal.get("min_trades_per_oos_fold", 100))
    min_trades_total = int(intraday_goal.get("min_trades_total_oos", 0))
    trades_mode = str(intraday_goal.get("min_trades_mode", "every_fold"))
    min_survival_pf = float(intraday_goal.get("min_survival_profit_factor", 1.0))
    min_edge_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.1))
    min_stress_pf = float(intraday_goal.get("min_stress_profit_factor", 1.0))
    pf_mode = str(intraday_goal.get("profit_factor_mode", "every_fold"))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))

    stress_metrics = run_intraday_stress_test(
        bars_by_symbol, signal_name, base_cfg, candidate_params,
        start_ts.to_pydatetime(), end_ts.to_pydatetime(), stress_slippage_multiplier=stress_mult,
        warmup_days=warmup_days, half_spread_bps_by_symbol=half_spread_bps_by_symbol,
        base_engine_cfg=engine_cfg,
    )

    if pf_mode == "pooled":
        survival_pf_ok = check_pooled_profit_factor_gate(wfo, min_survival_pf)
        edge_pf_ok = check_pooled_profit_factor_gate(wfo, min_edge_pf)
    else:
        survival_pf_ok = check_profit_factor_gate(wfo, min_survival_pf)
        edge_pf_ok = check_profit_factor_gate(wfo, min_edge_pf)

    if trades_mode == "pooled":
        min_trades_ok = check_pooled_trades_gate(wfo, min_trades_total or min_trades)
    else:
        min_trades_ok = check_min_trades_gate(wfo, min_trades)

    hard_gates, soft_gates = assemble_intraday_gates(
        wfo_go=wfo.decision == "GO",
        oos_drawdown_ok=check_drawdown_gate(wfo, max_oos_drawdown_threshold()),
        has_oos_trades=check_has_trades_gate(wfo),
        min_trades_ok=min_trades_ok,
        survival_pf_ok=survival_pf_ok,
        edge_pf_ok=edge_pf_ok,
        mc_ok=mc_result.sharpe.p5 >= min_p5,
        stress_mult=stress_mult,
        stress_pf_ok=_pf_clears(stress_metrics.get("profit_factor", 0.0), min_stress_pf),
    )
    overall_pass = all(hard_gates.values())

    if not quiet:
        print(f"    hard gates: { {k: ('PASS' if v else 'FAIL') for k, v in hard_gates.items()} }")
        print(f"    warnings:   { {k: ('PASS' if v else 'WARN') for k, v in soft_gates.items()} }")
        if not preflight.get("sample_size_sufficient", True):
            print(f"    WARNING sufficient_sample_size: only {preflight['total_trading_days']} trading days "
                  f"for {preflight['n_free_parameters']} free params (need >= {preflight['required_days']})")
        print(f"    -> {'GO' if overall_pass else 'NO-GO'}")

    return {
        "signal": signal_name,
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
        "gates": hard_gates,
        "soft_gates": soft_gates,
        "chart_minutes": int(engine_cfg.chart_minutes),
        "time_stop_minutes": int(engine_cfg.time_stop_minutes),
        "sample_size_check": preflight,
        "cost_model": "calibrated_per_symbol" if half_spread_bps_by_symbol else "flat_2.0bps",
    }


def run_full_grid_search(
    signal_name: str,
    args,
    bars_by_symbol: dict,
    data_label: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict]:
    """Re-runs the ENTIRE WFO/MC/stress pipeline once per individual grid
    candidate (configs/param_grids.yaml), each candidate FIXED across every
    fold (no per-fold re-optimization) — this is what lets the report
    honestly say "NONE of these N candidates pass", not just "the grid's
    best-per-fold pick doesn't pass" (which per-fold re-optimization in the
    main `run_signal` call already reports, but can't rule out that a
    DIFFERENT single candidate might have looked better if evaluated
    consistently across the whole window). Mirrors how
    backtests/reports/intraday_backtest_report.md's orb_vwap section
    reports its 6-candidate exhaustive grid search."""
    full_grid = load_param_grid(signal_name)
    results = []
    print(f"    running full grid search ({len(full_grid)} candidates) for {signal_name}...")
    for i, candidate in enumerate(full_grid):
        r = run_signal(signal_name, args, bars_by_symbol, data_label, start_ts, end_ts,
                        param_grid=[candidate], quiet=True)
        r["params"] = candidate
        results.append(r)
        print(f"      [{i + 1}/{len(full_grid)}] {candidate} -> {r['decision']} "
              f"(OOS Sharpe {r.get('oos_sharpe_mean', 0.0):+.3f}, pass_ratio {r.get('wfo_pass_ratio', 0.0):.0%})")
    return results


def _render_signal_section(r: dict) -> list[str]:
    lines = [f"## {r['signal']}", ""]
    if r["decision"] == "SKIPPED":
        lines.append(f"- SKIPPED: {r['reason']}")
        lines.append(f"- Data: {r['data_label']}")
        lines.append("")
        return lines

    lines.append(f"- Data: {r['data_label']}")
    lines.append(f"- Window: {r['window']} ({r['n_symbols']} symbols)")
    lines.append(f"- WFO: {r['wfo_folds']} folds, pass ratio {r['wfo_pass_ratio']:.0%}, "
                 f"OOS Sharpe mean {r['oos_sharpe_mean']:+.3f}")
    lines.append(f"- Monte Carlo p5 Sharpe (full window, candidate params): {r['mc_p5_sharpe']:+.3f}")
    fm = r["full_window_metrics"]
    lines.append(f"- Full-window metrics: n_trades={fm.get('n_trades')}, "
                 f"total_net_pnl={fm.get('total_net_pnl', 0):.2f}, "
                 f"profit_factor={fm.get('profit_factor', 0):.2f}, "
                 f"signals_emitted={fm.get('signals_emitted')}, signals_filled={fm.get('signals_filled')}")
    sm = r["stress_metrics"]
    lines.append(f"- Stress slippage: total_net_pnl={sm.get('total_net_pnl', 0):.2f}, "
                 f"profit_factor={sm.get('profit_factor', 0):.2f}")
    lines.append(f"- Candidate params: `{r['candidate_params']}`")
    lines.append("")
    lines.append("**Acceptance gates (hard AND — flip GO/NO-GO):**")
    for gate, passed in r["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] {gate}")
    soft = r.get("soft_gates") or {}
    if soft:
        lines.append("")
        lines.append("**Warnings (recorded, do not flip GO):**")
        for gate, passed in soft.items():
            flag = "OK" if passed else "WARNING"
            lines.append(f"- {flag} {gate}")
    ssc = r.get("sample_size_check", {})
    if "sample_size_sufficient" in ssc:
        flag = "OK" if ssc["sample_size_sufficient"] else "WARNING (below Chan's rule of thumb)"
        lines.append(f"- sufficient_sample_size: {flag} — {ssc['total_trading_days']} trading days "
                    f"available, {ssc['required_days']} required for {ssc['n_free_parameters']} free params "
                    "(this rule of thumb was derived for one-row-per-day daily strategies; treat it as an "
                    "informational flag here, not a hard requirement, since each intraday trading day "
                    "carries many more independent trade observations than one daily-bar row)")
    lines.append("")

    full_grid = r.get("full_grid")
    if full_grid:
        any_pass = any(g["decision"] == "GO" for g in full_grid)
        lines.append(f"**Full grid search ({len(full_grid)} candidates):** any_pass={any_pass}")
        for g in full_grid:
            failed = sorted(name for name, ok in g["gates"].items() if not ok)
            soft_failed = sorted(name for name, ok in (g.get("soft_gates") or {}).items() if not ok)
            failed_str = f" | failed: {', '.join(failed)}" if failed else ""
            warn_str = f" | warn: {', '.join(soft_failed)}" if soft_failed else ""
            pf = g["full_window_metrics"].get("profit_factor", 0.0)
            stress_pf = g["stress_metrics"].get("profit_factor", 0.0)
            lines.append(
                f"- `{g['params']}` \u2192 {g['decision']} | OOS Sharpe {g['oos_sharpe_mean']:+.3f} | "
                f"pass_ratio {g['wfo_pass_ratio']:.0%} | PF={pf:.3f} | mc_p5={g['mc_p5_sharpe']:+.3f} | "
                f"stress_pf={stress_pf:.2f}{failed_str}{warn_str}"
            )
        lines.append("")
    elif r.get("full_grid_skipped_reason"):
        lines.append(f"**Full grid search:** not run — {r['full_grid_skipped_reason']}")
        lines.append("")

    lines.append(f"**Overall: {r['decision']}**")
    lines.append("")
    return lines


def write_report(results: list[dict]) -> Path:
    lines = [
        "# Intraday Microstructure Signal Backtest Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> Report-only: a GO decision below is evidence to review, not an automatic",
        "> promotion. `configs/strategy.yaml`'s `auto_execute` for these signals stays",
        "> `false` regardless — see docs/microstructure_pivot_plan.md §4c/§6 for the",
        "> full validation/observe-before-auto discipline this system follows.",
        "",
    ]
    for r in results:
        lines.extend(_render_signal_section(r))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def write_report_json(results: list[dict], path: Path = REPORT_JSON_PATH) -> Path:
    import json

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_new_signals_report(results: list[dict]) -> Path:
    """New signal hypotheses (2026-08-06) — SEPARATE file from
    REPORT_PATH/intraday_backtest_report.md so the original three-signal
    report's content stays exactly as it was (docs/microstructure_pivot_plan.md
    §4c discipline: report every phase's results honestly, never overwrite
    a prior honest NO-GO writeup)."""
    lines = [
        "# New Intraday Signal Hypotheses — Backtest Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> Report-only, same discipline as intraday_backtest_report.md: a GO decision",
        "> below is evidence to review, not an automatic promotion. `configs/strategy.yaml`'s",
        "> `auto_execute` for these signals stays `false` regardless, and no promoted",
        "> params are written to configs/strategy.yaml by this script (see",
        "> python/backtest/promotion.py's `_FORBIDDEN_WRITE_KEYS`/human-in-the-loop write path)",
        "> — see docs/microstructure_pivot_plan.md §4c/§6 for the full discipline.",
        "",
        "These are THREE NEW signal hypotheses, deliberately NOT re-tuned variants of",
        "sweep_reclaim / fvg_retest / orb_vwap (all three already NO-GO — see",
        "intraday_backtest_report.md above/separately). Same fixed 20-symbol universe,",
        "same backtest window, and the exact same WFO/Monte Carlo/param-guard/2x-slippage-",
        "stress validation gates as those three.",
        "",
        f"## New signal hypotheses — {datetime.now(timezone.utc).date().isoformat()}",
        "",
    ]
    for r in results:
        lines.extend(_render_signal_section(r))

    NEW_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEW_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return NEW_REPORT_PATH


def write_l2_absorption_report(results: list[dict]) -> Path:
    """l2_absorption (2026-08-14) — SEPARATE file, same reasoning as
    write_new_signals_report above. See L2_REPORT_PATH's comment: this is a
    single-run cross-check, not the authoritative validation writeup."""
    lines = [
        "# l2_absorption — Single-Run Backtest Report (CLI cross-check)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> This is a plain `scripts/run_intraday_backtest.py --signal l2_absorption` run:",
        "> one per-fold-optimized WFO pass over `configs/param_grids.yaml`'s `l2_absorption`",
        "> grid, no variant levers applied. The AUTHORITATIVE l2_absorption validation",
        "> (parameter grid search + variant levers + reserved holdout) is",
        "> `backtests/reports/l2_absorption_validation_report.md`, produced by",
        "> `scripts/_l2_absorption_validation.py`. Same report-only discipline as",
        "> intraday_backtest_report.md / new_signals_report.md: a GO decision below is",
        "> evidence to review, not an automatic promotion; `auto_execute` stays `false`.",
        "",
    ]
    for r in results:
        lines.extend(_render_signal_section(r))

    L2_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    L2_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return L2_REPORT_PATH


def write_auction_reclaim_report(results: list[dict]) -> Path:
    """auction_reclaim (2026-08-18) — dedicated file so this investigation
    does not rewrite retired-signal reports."""
    lines = [
        "# auction_reclaim — Creamer-style Auction Reclaim Backtest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> Bar-only 5-minute proxy of Christopher Creamer's public process",
        "> (prior-session value bias + fib discount/premium outside value area +",
        "> two-bar absorption/reclaim). No footprint, no options GEX. A GO here",
        "> is evidence to review, not an automatic `auto_execute` promotion.",
        "",
    ]
    for r in results:
        lines.extend(_render_signal_section(r))

    AUCTION_RECLAIM_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUCTION_RECLAIM_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return AUCTION_RECLAIM_REPORT_PATH


def write_vsa_effort_report(results: list[dict]) -> Path:
    """vsa_effort (2026-08-18) — dedicated file so this investigation
    does not rewrite retired-signal reports."""
    lines = [
        "# vsa_effort — Wyckoff/VSA Effort-Without-Result Backtest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> 5-minute bar proxy of Wyckoff 1910 effort-vs-result (poor response",
        "> to size) plus the publicly named VSA test bar (no supply / no demand).",
        "> GEX is environment only. A GO here is evidence to review, not an",
        "> automatic `auto_execute` promotion.",
        "",
    ]
    for r in results:
        lines.extend(_render_signal_section(r))

    VSA_EFFORT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VSA_EFFORT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return VSA_EFFORT_REPORT_PATH


def _merge_volume_book_results(results: list[dict]) -> list[dict]:
    """Keep the other volume-book signal when this CLI is invoked for one
    name at a time (otherwise a finished vsa_no_demand report would be
    wiped by the subsequent obv_divergence run)."""
    by_signal = {}
    if VOLUME_BOOK_REPORT_JSON_PATH.exists():
        try:
            prev = json.loads(VOLUME_BOOK_REPORT_JSON_PATH.read_text(encoding="utf-8"))
            for row in prev.get("results") or []:
                name = row.get("signal")
                if name in VOLUME_BOOK_SIGNALS:
                    by_signal[name] = row
        except (OSError, json.JSONDecodeError):
            pass
    for row in results:
        name = row.get("signal")
        if name:
            by_signal[name] = row
    return [by_signal[s] for s in VOLUME_BOOK_SIGNALS if s in by_signal] or list(results)


def write_volume_book_report(results: list[dict]) -> Path:
    """vsa_no_demand / obv_divergence (2026-08-18) — dedicated file so
    these investigations do not rewrite retired-signal reports."""
    results = _merge_volume_book_results(results)
    lines = [
        "# Volume-book signals — vsa_no_demand + obv_divergence Backtest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> vsa_no_demand: Williams/Coulling narrow-bar no-demand / no-selling-",
        "> pressure, confirmed by the next 5-minute bar. obv_divergence:",
        "> Granville B-2 / S-2 on session 5-minute On-Balance Volume.",
        "> GEX is environment only. A GO here is evidence to review, not an",
        "> automatic `auto_execute` promotion.",
        "",
    ]
    for r in results:
        lines.extend(_render_signal_section(r))

    VOLUME_BOOK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOLUME_BOOK_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return VOLUME_BOOK_REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS + NEW_SIGNALS + L2_ABSORPTION_SIGNALS
                                           + AUCTION_RECLAIM_SIGNALS + VSA_EFFORT_SIGNALS
                                           + VOLUME_BOOK_SIGNALS
                                           + ["all", "new"],
                         default="all")
    parser.add_argument("--demo", action="store_true", help="offline synthetic 1m bars (no IB/cache required)")
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--full-grid", action="store_true",
                         help="also run an exhaustive per-candidate grid search for every signal run "
                              "(default: only for a signal whose main run's WFO decision is GO or has a "
                              "positive OOS Sharpe mean — mirrors how intraday_backtest_report.md's orb_vwap "
                              "section was produced)")
    parser.add_argument("--include-retired", action="store_true",
                         help="required to actually run the RETIRED_SIGNALS batch via `--signal all` / "
                              "`--signal new` (all six signals this CLI covers are RETIRED — confirmed "
                              "NO-GO, see backtests/reports/strategy_review_summary.md and "
                              "backtests/reports/signal_status.md). Naming an individual signal explicitly "
                              "(`--signal sweep_reclaim`, etc.) never needs this flag — it always runs.")
    args = parser.parse_args()

    if args.signal in ("all", "new"):
        if not args.include_retired:
            print(
                f"'--signal {args.signal}' would run only RETIRED signals "
                f"({', '.join(RETIRED_SIGNALS)}) — all confirmed NO-GO, see "
                "backtests/reports/strategy_review_summary.md and "
                "backtests/reports/signal_status.md. Nothing was run. Pass "
                "--include-retired to force this batch anyway, or name a signal "
                "explicitly with --signal <name> (always allowed)."
            )
            return
        signals, is_new = (SIGNALS, False) if args.signal == "all" else (NEW_SIGNALS, True)
    else:
        signals, is_new = [args.signal], args.signal in NEW_SIGNALS
        if args.signal in RETIRED_SIGNALS:
            print(f"NOTE: '{args.signal}' is RETIRED (confirmed NO-GO) — running anyway because it was "
                  "named explicitly. See backtests/reports/signal_status.md.")

    bars_by_symbol, data_label, start_ts, end_ts = _load_bars_for_args(args)

    results = []
    for name in signals:
        r = run_signal(name, args, bars_by_symbol, data_label, start_ts, end_ts)
        if is_new and r["decision"] != "SKIPPED":
            worth_scrutiny = r["decision"] == "GO" or r["oos_sharpe_mean"] > 0
            if args.full_grid or worth_scrutiny:
                r["full_grid"] = run_full_grid_search(name, args, bars_by_symbol, data_label, start_ts, end_ts)
            else:
                r["full_grid_skipped_reason"] = (
                    "main per-fold-optimized WFO run was decisively negative "
                    f"(decision={r['decision']}, OOS Sharpe mean {r['oos_sharpe_mean']:+.3f} <= 0) — "
                    "a full grid search could only confirm, not overturn, that result "
                    "(pass --full-grid to force it anyway)"
                )
        results.append(r)

    if args.signal in L2_ABSORPTION_SIGNALS:
        out_path = write_l2_absorption_report(results)
        json_path = write_report_json(results, path=L2_REPORT_JSON_PATH)
    elif args.signal in AUCTION_RECLAIM_SIGNALS:
        out_path = write_auction_reclaim_report(results)
        json_path = write_report_json(results, path=AUCTION_RECLAIM_REPORT_JSON_PATH)
    elif args.signal in VSA_EFFORT_SIGNALS:
        out_path = write_vsa_effort_report(results)
        json_path = write_report_json(results, path=VSA_EFFORT_REPORT_JSON_PATH)
    elif args.signal in VOLUME_BOOK_SIGNALS:
        results = _merge_volume_book_results(results)
        out_path = write_volume_book_report(results)
        json_path = write_report_json(results, path=VOLUME_BOOK_REPORT_JSON_PATH)
    elif is_new:
        out_path = write_new_signals_report(results)
        json_path = write_report_json(results, path=NEW_REPORT_JSON_PATH)
    else:
        out_path = write_report(results)
        json_path = write_report_json(results, path=REPORT_JSON_PATH)
    print(f"\nReport written to {out_path} (machine-readable: {json_path})")
    for r in results:
        print(f"  {r['signal']}: {r['decision']}")


if __name__ == "__main__":
    main()
