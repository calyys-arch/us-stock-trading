"""
Self-improving strategy loop: WFO grid search -> validation gates ->
baseline comparison -> (auto) write-back to configs/strategy.yaml ->
append-only decision history.

One ITERATION, per strategy:
  1. Read the CURRENT parameters from configs/strategy.yaml (the incumbent
     baseline — may already include a previous iteration's promotion).
  2. Pre-flight: parameter-discipline checks (python/backtest/param_guard.py
     via optimize.preflight_check).
  3. Candidate search: WalkForwardOptimizer over configs/param_grids.yaml
     (per-fold IS re-optimization, OOS validation).
  4. Baseline run: the SAME folds and data with the incumbent parameters
     only ([{}] grid) — apples-to-apples OOS comparison.
  5. Gates: WFO GO + per-fold OOS drawdown ceiling + has-trades + Monte
     Carlo p5 Sharpe on a full-window run of the candidate.
  6. python/backtest/promotion.py decides; a promotion rewrites ONLY the
     parameter values in configs/strategy.yaml (comments preserved,
     auto_execute NEVER touched — the system stays observe-only).

Multiple --iterations move the data end-date forward step-by-step (oldest
first), so iteration N+1 re-optimizes with iteration N's promoted values as
the new incumbent — the "loop" in self-improving loop.

Usage:
    python scripts/self_improve_loop.py --demo                       # synthetic, offline
    python scripts/self_improve_loop.py --demo --no-write            # report-only
    python scripts/self_improve_loop.py --strategy pairs_trading --start 2018-01-01 --end 2025-01-01
    python scripts/self_improve_loop.py --iterations 3 --refresh-data
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
log = logging.getLogger("self_improve_loop")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import (
    build_pairs_backtest_fn,
    build_xsection_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    load_param_grid,
    load_wfo_config,
    max_oos_drawdown_threshold,
    preflight_check,
)
from python.backtest.promotion import evaluate_and_promote
from python.backtest.walk_forward import WalkForwardOptimizer

STRATEGY_CONFIG_PATH = Path("configs/strategy.yaml")
GOAL_PATH = Path("configs/goal.yaml")
LOG_PATH = Path("backtests/reports/self_improvement_log.md")

STRATEGIES = ["pairs_trading", "xsection_mean_reversion"]


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_backtest_fn(strategy: str, args, base_cfg: dict):
    """Returns (backtest_fn, data_label, universe_fingerprint, data_source)."""
    if args.demo:
        from run_backtest import _synthetic_pair, _synthetic_panel

        if strategy == "pairs_trading":
            prices_a, prices_b = _synthetic_pair()
            fn = build_pairs_backtest_fn("SYNA", "SYNB", prices_a, prices_b, base_cfg)
        else:
            panel = _synthetic_panel()
            codes = sorted(panel.index.get_level_values(1).unique())
            fn = build_xsection_backtest_fn(panel, codes, base_cfg)
        return fn, "SYNTHETIC DEMO DATA — pipeline validation only", "demo", "synthetic"

    from python.data.fixed_universe import load_universe_config, universe_fingerprint
    from python.data.price_cache import get_cached_price_panel

    if strategy == "pairs_trading":
        panel, _flags, meta = get_cached_price_panel(
            [args.pair_a, args.pair_b], args.start, args.end, refresh=args.refresh_data)
        prices_a = panel.xs(args.pair_a.upper(), level=1)["close"]
        prices_b = panel.xs(args.pair_b.upper(), level=1)["close"]
        fn = build_pairs_backtest_fn(args.pair_a.upper(), args.pair_b.upper(),
                                     prices_a, prices_b, base_cfg)
        sources = "+".join(sorted(meta["sources"]))
        return fn, f"{args.pair_a}/{args.pair_b} daily bars via price cache ({sources})", "", sources

    universe_cfg = load_universe_config()
    symbols = universe_cfg["symbols"]
    panel, _flags, meta = get_cached_price_panel(symbols, args.start, args.end, refresh=args.refresh_data)
    fn = build_xsection_backtest_fn(panel, symbols, base_cfg)
    sources = "+".join(sorted(meta["sources"]))
    label = (f"fixed top-{universe_cfg['top_n']} universe "
             f"(computed_at={universe_cfg['computed_at']}) via price cache ({sources})")
    return fn, label, universe_fingerprint(universe_cfg), sources


def run_iteration(strategy: str, args, iteration: int, end_ts: pd.Timestamp) -> dict:
    """One optimize->validate->promote cycle for one strategy. Returns a
    summary dict for the markdown log."""
    base_cfg = _load_yaml(STRATEGY_CONFIG_PATH)[strategy]
    goal = _load_yaml(GOAL_PATH)
    param_grid = load_param_grid(strategy)
    start_ts = pd.Timestamp(args.start)
    # Business-day approximation of the WFO window's trading-day count —
    # good enough for the sufficient_sample_size WARNING gate (param_guard's
    # rule of thumb doesn't need exact exchange-holiday precision here).
    n_bdays = len(pd.bdate_range(start_ts, end_ts))
    preflight = preflight_check(strategy, base_cfg, param_grid, total_trading_days=n_bdays)
    wfo_cfg = load_wfo_config(strategy)

    fn, data_label, uni_fp, data_source = _build_backtest_fn(strategy, args, base_cfg)

    print(f"\n=== iter {iteration} | {strategy} | window [{start_ts.date()}, {end_ts.date()}] "
          f"| {len(param_grid)} candidates ===")
    print(f"    data: {data_label}")

    candidate_wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())
    candidate_wfo.print_summary()

    if not candidate_wfo.folds:
        print("    window too short for a single WFO fold — skipping")
        return {"strategy": strategy, "iteration": iteration, "decision": "SKIPPED",
                "reason": "window too short for a single WFO fold", "data_label": data_label}

    # Most-recent fold's winner = the parameters we would trade tomorrow.
    candidate_params = dict(candidate_wfo.folds[-1].best_params)

    baseline_wfo = WalkForwardOptimizer(fn, wfo_cfg, [{}]).run(
        start_ts.to_pydatetime(), end_ts.to_pydatetime())

    # Monte Carlo over a full-window run of the candidate parameters.
    full_metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), candidate_params)
    mc_result = MonteCarloValidator(n_sims=500).run(full_metrics.get("daily_returns", []))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    gates = {
        "wfo_go": candidate_wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(
            candidate_wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(candidate_wfo),
        "monte_carlo_p5_sharpe": mc_result.sharpe.p5 >= min_p5,
    }
    min_improvement = float(goal.get("live_promotion", {}).get("min_oos_sharpe_improvement", 0.0))

    baseline_params = {k: base_cfg[k] for k in candidate_params}
    record = evaluate_and_promote(
        strategy_name=strategy,
        candidate_params=candidate_params,
        baseline_params=baseline_params,
        candidate_oos_sharpe=candidate_wfo.oos_sharpe_mean,
        baseline_oos_sharpe=baseline_wfo.oos_sharpe_mean,
        gates=gates,
        wfo_summary={
            "decision": candidate_wfo.decision,
            "total_folds": candidate_wfo.total_folds,
            "passing_folds": candidate_wfo.passing_folds,
            "pass_ratio": round(candidate_wfo.pass_ratio, 3),
        },
        min_improvement=min_improvement,
        write_config=not args.no_write,
        universe_fingerprint=uni_fp,
        data_source=data_source,
        iteration=iteration,
    )

    print(f"    baseline OOS Sharpe: {baseline_wfo.oos_sharpe_mean:+.3f} "
          f"| candidate OOS Sharpe: {candidate_wfo.oos_sharpe_mean:+.3f}")
    print(f"    gates: { {k: ('PASS' if v else 'FAIL') for k, v in gates.items()} }")
    if not preflight.get("sample_size_sufficient", True):
        print(f"    WARNING sufficient_sample_size: only {preflight['total_trading_days']} trading days "
              f"for {preflight['n_free_parameters']} free params (need >= {preflight['required_days']})")
    print(f"    -> {record.decision}: {record.reason}")
    if record.decision == "PROMOTED" and args.no_write:
        print("    (--no-write: configs/strategy.yaml NOT modified)")

    return {
        "strategy": strategy,
        "iteration": iteration,
        "window": f"{start_ts.date()} .. {end_ts.date()}",
        "data_label": data_label,
        "decision": record.decision,
        "reason": record.reason,
        "gates": gates,
        "sample_size_check": preflight,
        "candidate_params": candidate_params,
        "baseline_params": baseline_params,
        "candidate_oos_sharpe": candidate_wfo.oos_sharpe_mean,
        "baseline_oos_sharpe": baseline_wfo.oos_sharpe_mean,
        "wfo_pass_ratio": candidate_wfo.pass_ratio,
        "wfo_folds": candidate_wfo.total_folds,
        "mc_p5_sharpe": mc_result.sharpe.p5,
        "config_written": record.config_written,
    }


def append_markdown_log(summaries: list[dict], args) -> None:
    lines = [
        "",
        f"## Run {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Mode: {'demo (synthetic)' if args.demo else 'real data'}"
        + (" | report-only (--no-write)" if args.no_write else " | auto-write enabled"),
        f"- Window: {args.start} .. {args.end} | iterations: {args.iterations}",
        "- Full machine-readable history: `backtests/logs/promotion_history.jsonl`",
        "",
    ]
    for s in summaries:
        lines.append(f"### iter {s['iteration']} — {s['strategy']}: **{s['decision']}**")
        lines.append("")
        if s["decision"] == "SKIPPED":
            lines.append(f"- {s['reason']}")
            lines.append("")
            continue
        lines.append(f"- Window: {s['window']}")
        lines.append(f"- Data: {s['data_label']}")
        lines.append(f"- WFO: {s['wfo_folds']} folds, pass ratio {s['wfo_pass_ratio']:.0%}")
        lines.append(f"- OOS Sharpe: baseline {s['baseline_oos_sharpe']:+.3f} -> "
                     f"candidate {s['candidate_oos_sharpe']:+.3f}")
        lines.append(f"- Monte Carlo p5 Sharpe: {s['mc_p5_sharpe']:+.3f}")
        gate_str = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in s["gates"].items())
        lines.append(f"- Gates: {gate_str}")
        ssc = s.get("sample_size_check", {})
        if "sample_size_sufficient" in ssc:
            flag = "OK" if ssc["sample_size_sufficient"] else "WARNING (below Chan's rule of thumb)"
            lines.append(f"- sufficient_sample_size: {flag} — {ssc['total_trading_days']} trading days "
                        f"available, {ssc['required_days']} required for {ssc['n_free_parameters']} free params")
        lines.append(f"- Candidate params: `{s['candidate_params']}` "
                     f"(baseline: `{s['baseline_params']}`)")
        lines.append(f"- Reason: {s['reason']}")
        lines.append(f"- configs/strategy.yaml written: {s['config_written']}")
        lines.append("")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        header = [
            "# Self-Improvement Loop Log",
            "",
            "Append-only, human-readable record of every self-improve run",
            "(scripts/self_improve_loop.py). Machine-readable decision history:",
            "`backtests/logs/promotion_history.jsonl`. Promotions only ever rewrite strategy",
            "PARAMETERS — `auto_execute` stays false (observe-only) regardless.",
            "",
        ]
        LOG_PATH.write_text("\n".join(header + lines), encoding="utf-8")
    else:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print(f"\nLog appended to {LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=STRATEGIES + ["both"], default="both")
    parser.add_argument("--demo", action="store_true", help="offline synthetic data")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--pair-a", default="XLE")
    parser.add_argument("--pair-b", default="XOP")
    parser.add_argument("--iterations", type=int, default=1,
                        help="N>1 moves the end date forward step-by-step so each iteration "
                             "re-optimizes on top of the previous one's promoted params")
    parser.add_argument("--iteration-step-days", type=int, default=126,
                        help="calendar days the end date advances between iterations")
    parser.add_argument("--no-write", action="store_true",
                        help="report-only: never modify configs/strategy.yaml")
    parser.add_argument("--refresh-data", action="store_true",
                        help="force re-fetch of cached price data")
    args = parser.parse_args()

    if args.demo:
        # Synthetic generators produce 1000 business days from 2018-01-02.
        args.start, args.end = "2018-01-02", "2021-11-01"

    strategies = STRATEGIES if args.strategy == "both" else [args.strategy]
    final_end = pd.Timestamp(args.end)

    summaries: list[dict] = []
    for i in range(args.iterations):
        end_i = final_end - pd.Timedelta(days=(args.iterations - 1 - i) * args.iteration_step_days)
        for strategy in strategies:
            summaries.append(run_iteration(strategy, args, iteration=i, end_ts=end_i))

    append_markdown_log(summaries, args)

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  iter {s['iteration']} {s['strategy']}: {s['decision']}")


if __name__ == "__main__":
    main()
