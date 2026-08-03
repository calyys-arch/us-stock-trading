"""
Intraday microstructure signal WFO validation CLI
(docs/microstructure_pivot_plan.md §4c).

Runs walk-forward optimization for sweep_reclaim / fvg_retest / orb_vwap
against the fixed universe's cached 1-minute bars (python/data/
intraday_cache.py, built by scripts/backfill_intraday.py), applies the
shared WFO + Monte Carlo gates PLUS the intraday-specific gates
(configs/goal.yaml `intraday` block: minimum OOS trade count, cost-adjusted
profit factor, and a mandatory 2x-slippage stress re-run), and writes an
honest GO/NO-GO report to backtests/reports/intraday_backtest_report.md.

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

Usage:
    python scripts/run_intraday_backtest.py --demo
    python scripts/run_intraday_backtest.py --signal sweep_reclaim --start 2025-08-01 --end 2026-07-01
    python scripts/run_intraday_backtest.py --signal all --start 2025-08-01 --end 2026-07-01
"""
from __future__ import annotations

import argparse
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

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import (
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

SIGNALS = ["sweep_reclaim", "fvg_retest", "orb_vwap"]
GOAL_PATH = Path("configs/goal.yaml")
STRATEGY_PATH = Path("configs/strategy.yaml")
REPORT_PATH = Path("backtests/reports/intraday_backtest_report.md")
# Machine-readable sibling of REPORT_PATH — same run, same content, JSON
# instead of markdown. Exists specifically so downstream tooling (e.g.
# scripts/sync_uhai.py's distillation into GreyCat triples) has a stable
# structured source instead of parsing the markdown report.
REPORT_JSON_PATH = Path("backtests/reports/intraday_backtest_report.json")


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def run_signal(signal_name: str, args) -> dict:
    base_cfg = _load_yaml(STRATEGY_PATH)[signal_name]
    goal = _load_yaml(GOAL_PATH)
    param_grid = load_param_grid(signal_name)
    wfo_cfg = load_wfo_config(signal_name)

    if args.demo:
        symbols = [f"SYN{i:02d}" for i in range(5)]
        bars_by_symbol = _synthetic_intraday_bars(symbols, n_days=90)
        data_label = "SYNTHETIC DEMO DATA — pipeline validation only, no real edge implied"
        start_ts = pd.Timestamp("2025-01-02")
        end_ts = pd.Timestamp("2025-01-02") + pd.Timedelta(days=140)
    else:
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

    n_bdays = len(pd.bdate_range(start_ts, end_ts))
    preflight = preflight_check(signal_name, base_cfg, param_grid, total_trading_days=n_bdays)

    fn = build_intraday_backtest_fn(bars_by_symbol, signal_name, base_cfg)

    print(f"\n=== {signal_name} | window [{start_ts.date()}, {end_ts.date()}] | "
          f"{len(param_grid)} candidates | {len(bars_by_symbol)} symbols ===")
    print(f"    data: {data_label}")

    wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
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
    min_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.3))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))

    stress_metrics = run_intraday_stress_test(
        bars_by_symbol, signal_name, base_cfg, candidate_params,
        start_ts.to_pydatetime(), end_ts.to_pydatetime(), stress_slippage_multiplier=stress_mult,
    )

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

    print(f"    gates: { {k: ('PASS' if v else 'FAIL') for k, v in gates.items()} }")
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
        "gates": gates,
        "sample_size_check": preflight,
    }


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
        lines.append(f"## {r['signal']}")
        lines.append("")
        if r["decision"] == "SKIPPED":
            lines.append(f"- SKIPPED: {r['reason']}")
            lines.append(f"- Data: {r['data_label']}")
            lines.append("")
            continue
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
        lines.append(f"- 2x-slippage stress: total_net_pnl={sm.get('total_net_pnl', 0):.2f}")
        lines.append(f"- Candidate params: `{r['candidate_params']}`")
        lines.append("")
        lines.append("**Acceptance gates:**")
        for gate, passed in r["gates"].items():
            lines.append(f"- [{'x' if passed else ' '}] {gate}")
        ssc = r.get("sample_size_check", {})
        if "sample_size_sufficient" in ssc:
            flag = "OK" if ssc["sample_size_sufficient"] else "WARNING (below Chan's rule of thumb)"
            lines.append(f"- sufficient_sample_size: {flag} — {ssc['total_trading_days']} trading days "
                        f"available, {ssc['required_days']} required for {ssc['n_free_parameters']} free params "
                        "(this rule of thumb was derived for one-row-per-day daily strategies; treat it as an "
                        "informational flag here, not a hard requirement, since each intraday trading day "
                        "carries many more independent trade observations than one daily-bar row)")
        lines.append("")
        lines.append(f"**Overall: {r['decision']}**")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def write_report_json(results: list[dict]) -> Path:
    import json

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return REPORT_JSON_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS + ["all"], default="all")
    parser.add_argument("--demo", action="store_true", help="offline synthetic 1m bars (no IB/cache required)")
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    args = parser.parse_args()

    signals = SIGNALS if args.signal == "all" else [args.signal]
    results = [run_signal(name, args) for name in signals]

    out_path = write_report(results)
    json_path = write_report_json(results)
    print(f"\nReport written to {out_path} (machine-readable: {json_path})")
    for r in results:
        print(f"  {r['signal']}: {r['decision']}")


if __name__ == "__main__":
    main()
