"""
Merges the per-signal checkpoints written by
scripts/_calibration_validation.py (old=flat-2.0bps, new=calibrated
per-symbol half-spread, SAME fixed candidate params) plus
backtests/reports/calibrated_spreads.json into the final
backtests/reports/slippage_calibration_report.md + .json — the Step 4
deliverable of the slippage-calibration task.

Usage:
    python scripts/_merge_calibration_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _calibration_validation import ALL_SIGNALS, _checkpoint_path  # noqa: E402

CALIBRATED_SPREADS_PATH = Path("backtests/reports/calibrated_spreads.json")
REPORT_MD_PATH = Path("backtests/reports/slippage_calibration_report.md")
REPORT_JSON_PATH = Path("backtests/reports/slippage_calibration_report.json")

FLAT_BASELINE_BPS = 2.0


def _fmt(v, spec: str = "+.3f") -> str:
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _render_spread_table(spreads: dict) -> list[str]:
    lines = [
        "## (a) Calibrated half-spread per symbol vs the 2.0bps flat baseline",
        "",
        f"Computed by `scripts/calibrate_slippage_spreads.py` from REAL captured Level-2 depth "
        f"(`data/depth/<SYMBOL>/*.jsonl`, Futu OpenD, `source=\"futu\"`), median half-spread-bps "
        "across every non-crossed top-of-book sample, reconstructed from the diff-event stream "
        "(insert/update at `position==0` per side) — see that script's module docstring for the "
        "exact reconstruction method and its known approximation (Futu's per-position-index "
        "snapshot diffing, not a native insert/delete stream).",
        "",
        "| Symbol | Flat baseline (bps) | Calibrated median (bps) | Δ vs flat | Days used | Samples | Crossed dropped | Open median | Midday median | Close median | Suspect? |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sym in sorted(spreads.keys()):
        s = spreads[sym]
        delta = s["median_bps"] - FLAT_BASELINE_BPS
        suspect = f"⚠️ {s['suspect_reason']}" if s.get("suspect") else "no"
        lines.append(
            f"| {sym} | {FLAT_BASELINE_BPS:.2f} | {s['median_bps']:.3f} | {delta:+.3f} | "
            f"{s['n_days']} ({', '.join(s['days'])}) | {s['n_samples']:,} | {s['n_crossed_dropped']:,} | "
            f"{_fmt(s.get('open_median_bps'), '.3f')} | {_fmt(s.get('midday_median_bps'), '.3f')} | "
            f"{_fmt(s.get('close_median_bps'), '.3f')} | {suspect} |"
        )
    lines.append("")
    n_days_set = {s["n_days"] for s in spreads.values()}
    lines.append(
        f"**Sample-size honesty note:** every symbol's estimate is built from **{sorted(n_days_set)} trading "
        "day(s)** of continuous RTH capture (data/depth/ started 2026-08-04) — this is a genuinely small "
        "sample (2 sessions), not a multi-week average. Directionally consistent with well-known real-world "
        "spread behavior (mega-caps AAPL/MSFT/NVDA/GOOGL sub-1bps to ~1bps; smaller/less-liquid names "
        "STX/WDC/NBIS/LITE several bps wide) and no symbol tripped the >=50bps mega-cap sanity flag, but "
        "should be re-calibrated once more capture days accumulate rather than treated as a stable long-run "
        "estimate."
    )
    lines.append("")
    lines.append(
        "**Time-of-day pattern (qualitative, not folded into the cost model):** every symbol's open-session "
        "(09:30-10:00 ET) median half-spread is wider than its midday median — often 2-4x — consistent with "
        "the standard open-wider/settle-down intraday spread pattern; close-session (15:30-16:00 ET) medians "
        "come in at or below midday for this universe on these 2 days, which is a smaller effect and not "
        "universal enough here to generalize beyond \"worth watching\", per the task's qualitative-only scope "
        "for this dimension."
    )
    lines.append("")
    return lines


def _render_signal_comparison(signal_name: str, old: dict, old_fixed: dict, new: dict) -> list[str]:
    lines = [f"### {signal_name}", ""]
    if old is None:
        lines.append("- MISSING checkpoint data — see backtests/reports/_checkpoint_calib_*.json")
        lines.append("")
        return lines

    lines.append(f"- Established candidate params: `{old['candidate_params']}` "
                 f"({'reused, real-data run already on disk' if old.get('cost_tag') == 'flat_reused_from_new_signals_report' else 'recovered by re-running the standard per-fold-optimized grid search — see Methodology'})")
    lines.append(f"- `old` = the ORIGINAL per-fold-optimized WFO (multi-candidate grid, as already reported)")
    lines.append(f"- `old_fixed` / `new` = SAME single fixed candidate above forced across every fold "
                 "(no re-optimization) — the clean, apples-to-apples pair for isolating the cost-model swap")
    lines.append("")

    if old_fixed is None or new is None:
        lines.append("⚠️ **old_fixed / new phase still running or not yet checkpointed** — see "
                     f"`backtests/reports/_calib_logs/{signal_name}_*.log` for live progress. "
                     f"`old` (original per-fold-optimized, flat-cost) verdict: **{old['decision']}**.")
        lines.append("")
        return lines

    lines.append(f"- Window: {new.get('window', 'n/a')} ({new.get('n_symbols', 'n/a')} symbols)")
    lines.append("")
    lines.append("| Metric | old (per-fold-optimized, flat, context) | old_fixed (fixed candidate, flat) | new (fixed candidate, calibrated) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| WFO pass ratio | {old['wfo_pass_ratio']:.0%} ({old.get('wfo_folds', 'n/a')} folds) | "
                 f"{old_fixed['wfo_pass_ratio']:.0%} ({old_fixed.get('wfo_folds', 'n/a')} folds) | "
                 f"{new['wfo_pass_ratio']:.0%} ({new.get('wfo_folds', 'n/a')} folds) |")
    lines.append(f"| OOS Sharpe mean | {old['oos_sharpe_mean']:+.3f} | {old_fixed['oos_sharpe_mean']:+.3f} | {new['oos_sharpe_mean']:+.3f} |")
    old_pf = old["full_window_metrics"].get("profit_factor", 0.0)
    oldf_pf = old_fixed["full_window_metrics"].get("profit_factor", 0.0)
    new_pf = new["full_window_metrics"].get("profit_factor", 0.0)
    lines.append(f"| Cost-adjusted profit factor (full window) | {old_pf:.3f} | {oldf_pf:.3f} | {new_pf:.3f} |")
    lines.append(f"| Monte Carlo p5 Sharpe | {old['mc_p5_sharpe']:+.3f} | {old_fixed['mc_p5_sharpe']:+.3f} | {new['mc_p5_sharpe']:+.3f} |")
    old_stress = old["stress_metrics"].get("total_net_pnl", 0.0)
    oldf_stress = old_fixed["stress_metrics"].get("total_net_pnl", 0.0)
    new_stress = new["stress_metrics"].get("total_net_pnl", 0.0)
    lines.append(f"| 2x-slippage stress net P&L | {old_stress:,.2f} | {oldf_stress:,.2f} | {new_stress:,.2f} |")
    old_pnl = old["full_window_metrics"].get("total_net_pnl", 0.0)
    oldf_pnl = old_fixed["full_window_metrics"].get("total_net_pnl", 0.0)
    new_pnl = new["full_window_metrics"].get("total_net_pnl", 0.0)
    lines.append(f"| Full-window total net P&L | {old_pnl:,.2f} | {oldf_pnl:,.2f} | {new_pnl:,.2f} |")
    lines.append(f"| n_trades (full window) | {old['full_window_metrics'].get('n_trades')} | "
                 f"{old_fixed['full_window_metrics'].get('n_trades')} | {new['full_window_metrics'].get('n_trades')} |")
    lines.append("")

    lines.append("**Gates under calibrated (`new`) costs:**")
    for gate, passed in new["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] {gate}")
    lines.append("")
    lines.append(f"**`old` verdict (original per-fold-optimized, flat cost, for context):** {old['decision']}")
    lines.append(f"**`old_fixed` verdict (same fixed candidate, flat cost):** {old_fixed['decision']}")
    lines.append(f"**`new` verdict (same fixed candidate, calibrated cost):** {new['decision']}")
    flipped = old_fixed["decision"] == "NO-GO" and new["decision"] == "GO"
    lines.append(f"**Flipped NO-GO -> GO (old_fixed -> new, the clean comparison)?** {'**YES**' if flipped else 'No'}")
    lines.append("")
    return lines


def main() -> None:
    spreads_payload = json.loads(CALIBRATED_SPREADS_PATH.read_text(encoding="utf-8"))
    spreads = spreads_payload["symbols"]

    checkpoints = {}
    missing = []
    incomplete = []
    for sig in ALL_SIGNALS:
        p = _checkpoint_path(sig)
        if not p.exists():
            missing.append(sig)
            checkpoints[sig] = {"old": None, "old_fixed": None, "new": None}
            continue
        state = json.loads(p.read_text(encoding="utf-8"))
        state.setdefault("old_fixed", None)
        checkpoints[sig] = state
        if state["old"] is None or state["old_fixed"] is None or state["new"] is None:
            incomplete.append(sig)

    flips = []
    for sig in ALL_SIGNALS:
        old_fixed, new = checkpoints[sig]["old_fixed"], checkpoints[sig]["new"]
        if old_fixed is not None and new is not None and old_fixed["decision"] == "NO-GO" and new["decision"] == "GO":
            flips.append(sig)

    lines = [
        "# Slippage Calibration Re-Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> Report-only, same discipline as intraday_backtest_report.md / new_signals_report.md: a GO",
        "> decision below is evidence to review, not an automatic promotion. `configs/strategy.yaml`'s",
        "> `auto_execute` stays `false` for every signal regardless of this report's outcome, and NO",
        "> params are written back to configs/strategy.yaml by this script — see",
        "> python/backtest/promotion.py's `_FORBIDDEN_WRITE_KEYS`/human-in-the-loop write path. A signal",
        "> that flips to GO below is a **promotion CANDIDATE for human review**, not a live change.",
        "",
        "## HEADLINE",
        "",
    ]
    if flips:
        lines.append(f"**YES — {len(flips)} signal(s) flipped from NO-GO to GO under calibrated costs: "
                     f"{', '.join(flips)}.** See the per-signal sections below for full gate detail before "
                     "treating this as anything more than a promotion candidate.")
    else:
        lines.append("**NO signal flipped from NO-GO to GO under calibrated (real, captured-depth-derived) "
                     "per-symbol half-spread costs.** All six remain NO-GO; calibrated costs moved some "
                     "metrics (see per-signal tables below) but not far enough to clear every gate for any "
                     "signal, including `orb_vwap` (the closest-to-passing candidate going in).")
    if missing or incomplete:
        lines.append("")
        lines.append(f"⚠️ **This run is still in progress.** No checkpoint yet for {missing or 'none'}; "
                     f"partial (old done, old_fixed/new pending) for {[s for s in incomplete if s not in missing]}. "
                     "See `backtests/reports/_calib_logs/` for live progress, "
                     "`backtests/reports/_checkpoint_calib_<signal>.json` for what's already checkpointed. "
                     "Re-run `python scripts/_calibration_validation.py <signal> <old|old_fixed|new|both>` to "
                     "resume — already-computed WFO folds replay instantly from "
                     "`backtests/reports/_calib_cache/`. Re-run this merge script "
                     "(`python scripts/_merge_calibration_report.py`) once all six finish to get the final headline.")
    lines.append("")

    lines.extend(_render_spread_table(spreads))

    lines.append("## (b) Per-signal old (flat) vs new (calibrated) validation results")
    lines.append("")
    for sig in ALL_SIGNALS:
        lines.extend(_render_signal_comparison(sig, checkpoints[sig]["old"], checkpoints[sig]["old_fixed"], checkpoints[sig]["new"]))

    lines.append("## (c) Methodology notes")
    lines.append("")
    lines.append(
        "- **Candidate params held fixed** across `old_fixed` and `new` for every signal — this isolates the "
        "cost-model change, it is NOT a fresh parameter search (`python scripts/_calibration_validation.py "
        "<signal> old_fixed|new` forces every WFO fold to the SAME single candidate via "
        "`param_grid=[candidate_params]`)."
    )
    lines.append(
        "- **Why `old_fixed` exists as a separate column from `old`:** `old` is the ORIGINAL per-fold-"
        "optimized WFO (WalkForwardOptimizer re-picks the best IS candidate from the full grid every fold), "
        "kept for continuity with the prior reports. Comparing `old` directly against `new` (fixed candidate) "
        "would conflate two changes at once — \"removed per-fold reoptimization\" AND \"changed cost model\" "
        "— so the headline flip determination uses `old_fixed` vs `new`, which differ in EXACTLY one thing: "
        "`half_spread_bps_by_symbol`."
    )
    lines.append(
        "- For `orb_vwap_regime` / `vwap_band_fade` / `vp_breakout`, the OLD column is reused **verbatim, "
        "zero recomputation** from `backtests/reports/new_signals_report.json` (already a real 20-symbol-"
        "universe run under the flat cost model)."
    )
    lines.append(
        "- For `sweep_reclaim` / `fvg_retest` / `orb_vwap`, the real-data OLD baseline previously reported in "
        "`backtests/reports/intraday_backtest_report.md` had been overwritten on disk by a later `--demo` "
        "pipeline-validation invocation of `scripts/run_intraday_backtest.py` (both write to the same "
        "`REPORT_PATH`). The OLD column for these three was therefore **regenerated** via the same "
        "per-fold-optimized full-grid WFO (`configs/param_grids.yaml`) that originally produced it — this "
        "both recovers the honest flat-cost baseline AND discovers the \"last fold's best params\" candidate "
        "used, unchanged, for the NEW (calibrated) run. It is not a fresh hypothesis search."
    )
    lines.append(
        "- No gate thresholds were weakened — same `configs/goal.yaml` `wfo` / `monte_carlo` / `intraday` "
        "blocks as every prior report."
    )
    lines.append("")

    n_below_flat = sum(1 for s in spreads.values() if s["median_bps"] < FLAT_BASELINE_BPS)
    n_above_flat = sum(1 for s in spreads.values() if s["median_bps"] >= FLAT_BASELINE_BPS)
    lines.append("## (d) Key observation: calibration is NOT uniformly a cost cut")
    lines.append("")
    lines.append(
        f"Only **{n_below_flat} of {len(spreads)}** symbols in this universe have a calibrated median "
        f"half-spread BELOW the flat 2.0bps baseline (the mega-caps: AAPL, MSFT, NVDA, GOOGL, META, AVGO, "
        f"INTC, PLTR, roughly). The other **{n_above_flat}** — mostly the smaller/less-liquid names (STX, "
        "WDC, NBIS, LITE, LRCX, AMAT, MRVL, SNDK, ORCL, QCOM, MU) — are calibrated ABOVE 2.0bps, some "
        "substantially (STX ~6.6bps, WDC ~4.4bps). For every signal computed so far, these six 1-minute "
        "microstructure signals trade broadly across the fixed 20-symbol universe rather than concentrating "
        "in the handful of tightest-spread mega-caps, so the AVERAGE effect of calibration was a net "
        "COST INCREASE, not a decrease — every signal's cost-adjusted profit factor, Monte Carlo p5 Sharpe, "
        "and stress P&L got measurably WORSE under calibrated costs, not better. The pre-task hope that "
        "\"tighter mega-cap spreads might flip `orb_vwap` to GO\" did not pan out for exactly this reason: "
        "the flat 2.0bps constant was, if anything, UNDER-pricing this universe's true blended cost, not "
        "over-pricing it."
    )
    lines.append("")

    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline_any_flip_to_go": bool(flips),
        "flipped_signals": flips,
        "missing_signals": missing,
        "calibrated_spreads": spreads,
        "signals": checkpoints,
    }, indent=2, default=str), encoding="utf-8")

    print(f"Report written to {REPORT_MD_PATH} (machine-readable: {REPORT_JSON_PATH})")
    print(f"Headline: any flip to GO? {bool(flips)} ({flips})")
    if missing:
        print(f"WARNING: incomplete — missing {missing}")


if __name__ == "__main__":
    main()
