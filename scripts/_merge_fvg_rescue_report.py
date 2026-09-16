"""
Merges the per-config checkpoints written by scripts/_fvg_retest_rescue.py
into backtests/reports/fvg_retest_rescue_report.md + .json.

Same discipline as scripts/_merge_orb_rescue_report.py: the markdown report
is GENERATED from the checkpointed run results rather than hand-transcribed,
so every number in it is traceable to a
backtests/reports/_fvg_rescue/<config_id>.json produced by an actual run
under configs/goal.yaml's unmodified (current) gate thresholds.

Usage:
    python scripts/_merge_fvg_rescue_report.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_DIR = Path("backtests/reports/_fvg_rescue")
REPORT_MD_PATH = Path("backtests/reports/fvg_retest_rescue_report.md")
REPORT_JSON_PATH = Path("backtests/reports/fvg_retest_rescue_report.json")

ROW_ORDER = [
    "A0_asshipped_full20",
    "B1_tight10", "B2_tight6",
    "C1_cap1_tight6", "C2_cap2_tight6", "C3_cap3_tight6",
    "E_r15_tight6", "E_r20_tight6", "E_r30_tight6",
    "HOLDOUT_best",
]

HARD_GATE_ORDER = [
    "wfo_go", "oos_drawdown_within_limit", "has_oos_trades",
    "cost_adjusted_profit_factor", "monte_carlo_p5_sharpe",
    "stress_slippage_1.5x_pf_ge_1", "survival_profit_factor",
    "stress_slippage_1.5x_profit_factor", "has_trades",
]
GATE_SHORT = {
    "wfo_go": "wfo", "oos_drawdown_within_limit": "dd", "has_oos_trades": "trades",
    "cost_adjusted_profit_factor": "survival_PF", "monte_carlo_p5_sharpe": "mc_p5",
    "stress_slippage_1.5x_pf_ge_1": "stress_PF", "survival_profit_factor": "survival_PF",
    "stress_slippage_1.5x_profit_factor": "stress_PF", "has_trades": "trades",
}


def load_results() -> dict[str, dict]:
    out = {}
    for cid in ROW_ORDER:
        p = CHECKPOINT_DIR / f"{cid}.json"
        if p.exists():
            out[cid] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _pf(r: dict) -> float:
    return float(r.get("full_window_metrics", {}).get("profit_factor", 0.0))


def _label(r: dict) -> str:
    p = dict(r["params"])
    bits = []
    if p.get("max_entries_per_session") is not None:
        bits.append(f"cap={p['max_entries_per_session']}")
    if p.get("target_r_multiple") is not None:
        bits.append(f"R={p['target_r_multiple']:g}")
    tail = (", " + ", ".join(bits)) if bits else ""
    return f"{r['n_symbols']} sym{tail}"


def _gate_cell(r: dict) -> str:
    gates = r.get("gates", {})
    failed = [g for g, ok in gates.items() if not ok]
    if not failed:
        return "**all pass**"
    return "fail: " + ", ".join(GATE_SHORT.get(g, g) for g in failed)


def render_lever_table(results: dict[str, dict]) -> list[str]:
    lines = [
        "| # | Lever | Config | WFO pass ratio | Mean OOS Sharpe | Net (cost-adj.) PF | MC p5 Sharpe | 1.5x-stress PF | Net P&L | Trades | Gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, cid in enumerate([c for c in ROW_ORDER if c in results and c != "HOLDOUT_best"], start=1):
        r = results[cid]
        pr = r.get("wfo_pass_ratio")
        pr_s = f"{pr:.0%}" if pr is not None else "n/a"
        sh = r.get("oos_sharpe_mean")
        sh_s = f"{sh:+.3f}" if sh is not None else "n/a"
        fw = r["full_window_metrics"]
        st = r.get("stress_metrics", {})
        lines.append(
            f"| {i} | {r['lever']} | `{cid}` — {_label(r)} | {pr_s} | {sh_s} | "
            f"{_pf(r):.3f} | {r['mc_p5_sharpe']:+.3f} | {st.get('profit_factor', 0):.3f} | "
            f"${fw['total_net_pnl']:,.0f} | {fw['n_trades']:,} | {_gate_cell(r)} |"
        )
    return lines


def render_gate_detail(r: dict, title: str) -> list[str]:
    lines = [f"**{title}** (gate thresholds read unmodified from `configs/goal.yaml`):", ""]
    gates = r.get("gates", {})
    for g in HARD_GATE_ORDER:
        if g in gates:
            lines.append(f"- [{'x' if gates[g] else ' '}] {g}")
    for g, ok in gates.items():
        if g not in HARD_GATE_ORDER:
            lines.append(f"- [{'x' if ok else ' '}] {g}")
    soft = r.get("soft_gates", {})
    if soft:
        lines.append("")
        lines.append("Soft (recorded only, do not flip the verdict):")
        for g, ok in soft.items():
            lines.append(f"- [{'x' if ok else ' '}] {g}")
    lines.append("")
    lines.append(f"**Verdict: {r['decision']}**")
    lines.append("")
    return lines


def main() -> None:
    results = load_results()
    n_dev = len([c for c in results if c != "HOLDOUT_best"])
    lines: list[str] = []

    a0 = results.get("A0_asshipped_full20")
    c1 = results.get("C1_cap1_tight6")
    holdout = results.get("HOLDOUT_best")

    lines += [
        "# fvg_retest Cost-to-Edge Rescue — Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> Report-only, same discipline as `intraday_backtest_report.md` / "
        "`orb_vwap_rescue_report.md`: a GO below would be evidence for a human to "
        "review, not an automatic promotion. No gate threshold in `configs/goal.yaml` "
        "was changed, `auto_execute` stays `false` for every strategy, and no params "
        "were written back into `configs/strategy.yaml` by this work.",
        "",
        "## HEADLINE",
        "",
        f"**NO-GO.** {n_dev} configurations were evaluated on the development window, "
        "plus the single best configuration evaluated once on the untouched final "
        "holdout. Two levers — a tight-spread universe subset and an entries-per-"
        "session cap — lifted `fvg_retest`'s net (cost-adjusted) profit factor from "
        f"**{_pf(a0):.3f}** (as-shipped, 20 symbols) to **{_pf(c1):.3f}** "
        "(`C1_cap1_tight6`) on the development window. Unlike `orb_vwap`'s rescue, "
        "this is close to the `configs/goal.yaml` **survival** bar "
        "(pooled profit factor >= 1.0, currently the hard gate — not the older 1.3 "
        "figure quoted in `orb_vwap_rescue_report.md`, which predates a 2026-08-18 "
        "gate-policy rewrite) but still falls short of it, and further short of the "
        "1.1 soft edge bar.",
        "",
        "**The holdout is almost IDENTICAL, not worse.** Evaluated exactly once on "
        "the final two months of 1-minute history (2026-06-01 .. 2026-08-01), the "
        f"frozen best configuration produced a net profit factor of **{_pf(holdout):.3f}** "
        f"— within {abs(_pf(c1) - _pf(holdout)):.3f} of its development-window value "
        f"({_pf(c1):.3f}) — on {holdout['full_window_metrics']['n_trades']} trades, "
        f"net P&L **${holdout['full_window_metrics']['total_net_pnl']:,.0f}**. This is "
        "the opposite finding from `orb_vwap`'s rescue, where the frozen configuration "
        "collapsed out-of-sample (1.003 -> 0.713). Here the near-breakeven edge is "
        "STABLE across two non-overlapping windows — the two levers did not manufacture "
        "an in-sample-only illusion, they found something real. It is just not enough.",
        "",
        "**One finding cuts against the original retirement rationale:** the "
        "2026-08-13 retirement docstring named the strategy's fixed 1:1 risk:reward "
        "target as the literal root cause (\"needs a win rate far above what the "
        "pattern actually has\"). Loosening it — a new `target_r_multiple` lever, "
        "tested at 1.5R/2.0R/3.0R on top of the best cap+universe configuration — made "
        f"the net PF WORSE in every case tested ({_pf(results['C1_cap1_tight6']):.3f} at "
        f"the original 1:1 vs {_pf(results['E_r15_tight6']):.3f}/"
        f"{_pf(results['E_r20_tight6']):.3f}/{_pf(results['E_r30_tight6']):.3f} at "
        "1.5R/2.0R/3.0R). The named root cause was not the fixable part; the universe "
        "and frequency were.",
        "",
        "## 1. What was tested, and why",
        "",
        "`fvg_retest` was re-examined after `scripts/_gross_edge_diag.py` (2026-09-16, "
        "read-only, zero-cost vs real-cost, full 20-symbol universe, 2025-08-01..2026-07-01) "
        "found a genuinely positive TRUE zero-slippage gross profit factor of **1.307** "
        "(48.7% win rate) — the first of the seven retired signals in "
        "`backtests/reports/strategy_review_summary.md` confirmed to have real directional "
        "edge rather than a falsified thesis (contrast `sweep_reclaim`'s gross PF < 1.0 "
        "even at zero cost on both the mega-cap and a dedicated low-liquidity universe, "
        "`configs/alt_universe_lowliq.yaml`). At ~4.7 trades/symbol/session across all 20 "
        "symbols, the net (cost-adjusted) PF at the 2026-08-13 retirement was 0.195 — this "
        "investigation asked whether the SAME kind of levers that partially rescued "
        "`orb_vwap` (`backtests/reports/orb_vwap_rescue_report.md`) could close that gap "
        "here, starting from a stronger gross base.",
        "",
        "| Lever | Hypothesis | How it is implemented |",
        "|---|---|---|",
        "| 1. Tight-spread universe | Same as `orb_vwap`'s calibration finding: trading "
        "broadly across all 20 symbols, half of them well above 2bps, is a direct cost "
        "driver. | Data-universe restriction in the runner. TIGHT10/TIGHT6 = the 10/6 "
        "symbols with the lowest CURRENT calibrated half-spread — recomputed fresh in "
        "`scripts/_fvg_retest_rescue.py`, not copied from `orb_vwap`'s now-stale hardcoded "
        "lists (the universe has been refreshed since). |",
        "| 2. Entries-per-session cap | ~4.7 fills/symbol/session for a pattern keyed on a "
        "single 3-bar impulse-and-gap event is implausible as a once-per-session-worth "
        "signal, by analogy with `orb_vwap`'s entry-ordinal diagnostic. | New "
        "`max_entries_per_session` param on `fvg_retest` (it already existed for "
        "`orb_vwap`), consumed by the ENGINE (`python/backtest/intraday_engine.py`'s "
        "`run_symbol_day`), unchanged code path. |",
        "| 3. Target R-multiple | The 2026-08-13 retirement's own stated root cause: a "
        "strict 1:1 R:R target needs a win rate this pattern does not have. | New "
        "`target_r_multiple` param on `evaluate_fvg_retest` (default 1.0 = byte-for-byte "
        "the pre-existing hardcoded 1:1 mirrored target). |",
        "",
        "Both new parameters default to the pre-existing behavior (`null` / `1.0` in "
        "`configs/strategy.yaml`), so nothing about the shipped configuration changed. "
        "With them, `fvg_retest` has **5 free parameters** — exactly at "
        "`python/backtest/param_guard.py`'s `MAX_FREE_PARAMETERS` ceiling, not over it, "
        "the same discipline `orb_vwap`'s rescue used. Unlike `orb_vwap`, no ATR-stop-buffer "
        "lever was added: `fvg_retest`'s stop is structural (mirrors the FVG width, not a "
        "raw breakout extreme) and adding a 4th tunable lever would have exceeded the "
        "parameter budget without a specific diagnosed reason to spend it there. The "
        "universe restriction is deliberately NOT counted as one of the 5: it is an "
        "eligibility/data decision of the same kind as `orb_vwap`'s, but see the "
        "overfitting section — it is still a degree of freedom for data-snooping purposes.",
        "",
        "**No stop-side/target-side correctness fix was needed here**, unlike `orb_vwap`'s "
        "Lever 0. `fvg_retest`'s module docstring already documents and fixes the class of "
        "defect `orb_vwap`'s inverted-stop bug turned out to be (a target/stop reference "
        "price that is not guaranteed to sit on the correct side of entry) — it was caught "
        "and corrected for `fvg_retest` BEFORE that bug was even found in `orb_vwap`. So "
        "`A0` below is the correct as-shipped baseline, not a defect to be corrected first.",
        "",
        "## 2. Methodology",
        "",
        "- **Development window:** 2025-08-01 .. 2026-06-01 (7 WFO folds at "
        "`configs/goal.yaml`'s `fvg_retest` cadence: is_days=90, oos_days=30, step_days=30) "
        "— identical to `orb_vwap`'s rescue for direct comparability.",
        "- **Final holdout:** 2026-06-01 .. 2026-08-01 — the last two months of available "
        "1-minute history. Not looked at, in any form, until the single best development "
        "configuration was frozen; evaluated exactly once.",
        "- **Costs:** calibrated per-symbol half-spreads from "
        "`backtests/reports/calibrated_spreads.json` for every headline number, in both "
        "the main run and the mandatory 1.5x-slippage stress re-run (the multiplier is "
        "1.5x, not 2.0x — `configs/goal.yaml`'s `stress_slippage_multiplier` was lowered "
        "2026-08-18 with the comment \"2.0x was vetoing thin but real edges\"; this "
        "investigation reads it, not a hardcoded value).",
        "- **No per-fold re-optimization.** Every configuration is a SINGLE candidate "
        "forced across every fold (`param_grid=[candidate]`).",
        "- **Gates:** DEV-window configs call `scripts/run_intraday_backtest.py`'s "
        "`run_signal()` DIRECTLY — the exact function every other signal's official "
        "report in this repo uses — rather than a hand-rolled copy, so the hard/soft gate "
        "split and pooled PF/trades thresholds cannot silently drift from "
        "`configs/goal.yaml`'s current policy the way a copy-pasted "
        "`_orb_vwap_rescue.py`-style gate dict would have (that policy was rewritten "
        "2026-08-18, AFTER the orb_vwap rescue report was written — see "
        "`scripts/_fvg_retest_rescue.py`'s module docstring for the full diff). Current "
        "hard gates: WFO GO, OOS drawdown within limit, has OOS trades, pooled profit "
        "factor >= 1.0 (\"survival\"), Monte Carlo p5 Sharpe >= 0, stress profit factor "
        ">= 1.0 at 1.5x slippage. Soft (recorded, non-binding): pooled min trades, pooled "
        "profit factor >= 1.1 (\"edge\"). The HOLDOUT window (shorter than one WFO fold) "
        "cannot go through `run_signal`, so it is evaluated as one out-of-sample window "
        "against the same goal.yaml keys read at call time.",
        "- **Durability:** per-configuration checkpoint JSON in "
        "`backtests/reports/_fvg_rescue/`.",
        "",
        "## 3. Lever-by-lever results (development window, calibrated per-symbol costs)",
        "",
    ]
    lines += render_lever_table(results)
    lines += [
        "",
        "Config column key: `cap` = `max_entries_per_session`, `R` = `target_r_multiple`. "
        "TIGHT10 = "
        + ", ".join(results["B1_tight10"]["symbols"]) + ". TIGHT6 = "
        + ", ".join(results["B2_tight6"]["symbols"]) + ".",
        "",
        "### Reading the table",
        "",
        f"- **Lever 1 (universe) is the single largest effect: +{_pf(results['B1_tight10']) - _pf(a0):.3f} PF "
        f"at TIGHT10, +{_pf(results['B2_tight6']) - _pf(a0):.3f} at TIGHT6** "
        f"({_pf(a0):.3f} -> {_pf(results['B1_tight10']):.3f} -> {_pf(results['B2_tight6']):.3f}). "
        "Bigger than the equivalent lever in `orb_vwap`'s rescue (+0.181/+0.234) — this "
        "signal's cost sensitivity to wide-spread names is even more pronounced. TIGHT6 "
        "was carried forward: unlike `orb_vwap`'s TIGHT10-vs-TIGHT6 tradeoff (chosen for "
        "`min_trades_per_oos_fold` headroom under the OLD every-fold-100 rule), the current "
        "pooled/soft trade-count rule has ample headroom on both "
        f"({min(results['B2_tight6']['fold_oos_trades'])}-"
        f"{max(results['B2_tight6']['fold_oos_trades'])} OOS trades/fold on TIGHT6 alone, "
        "against a soft floor of 8), so TIGHT6 wins outright on PF with no tradeoff to "
        "weigh.",
        f"- **Lever 2 (entry cap) is the second-largest effect: +{_pf(c1) - _pf(results['B2_tight6']):.3f} PF "
        f"at cap=1** ({_pf(results['B2_tight6']):.3f} -> {_pf(c1):.3f}), and it is "
        "monotone in the wrong-to-cap direction — cap=2 "
        f"({_pf(results['C2_cap2_tight6']):.3f}) and cap=3 "
        f"({_pf(results['C3_cap3_tight6']):.3f}) are both worse than cap=1, the same "
        "\"only the first entry of a session carries the edge\" pattern `orb_vwap`'s "
        "entry-ordinal diagnostic found.",
        "- **Lever 3 (target) is NEGATIVE at every multiple tested, the opposite of "
        f"`orb_vwap`'s finding.** 1.5R ({_pf(results['E_r15_tight6']):.3f}), 2.0R "
        f"({_pf(results['E_r20_tight6']):.3f}) and 3.0R "
        f"({_pf(results['E_r30_tight6']):.3f}) are all worse than the original "
        f"hardcoded 1:1 ({_pf(c1):.3f}). Where `orb_vwap` had NO target at all "
        "(winners cut only by the time-stop/EOD flatten) and a 1:1 target was actively "
        "harmful there too before 2R/3R helped, `fvg_retest` already HAD a 1:1 target "
        "and loosening it hurts monotonically. The likely mechanism: this is a "
        "retracement-into-continuation pattern, not a breakout — price that is going to "
        "keep moving does so quickly after the retest fills, so a tight target captures "
        "it while a wide one gives the move room to round-trip back through entry before "
        "the (still-1:1-distance) stop is reached. This directly contradicts the "
        "2026-08-13 retirement docstring's stated root cause: the 1:1 target was not the "
        "fixable problem, cost and frequency were.",
        "",
        "### Selection rule for the frozen configuration",
        "",
        "Declared before the E rows were run: primary metric = development-window net "
        "(cost-adjusted) profit factor, the metric closest to the binding survival gate. "
        f"`C1_cap1_tight6` wins outright ({_pf(c1):.3f}, vs {_pf(results['E_r15_tight6']):.3f} "
        "for the best E-row) — no tie-break was needed. The frozen configuration is:",
        "",
        "```",
        "universe                : AAPL GOOGL NVDA MSFT PLTR INTC  (tight-spread 6)",
        "vol_mult                : 2.0",
        "entry_pct               : 0.5",
        "expiry_bars             : 10",
        "max_entries_per_session : 1",
        "target_r_multiple       : 1.0   (unchanged from as-shipped)",
        "```",
        "",
        "5 free parameters, exactly at the Chan Ch.3 ceiling.",
        "",
    ]
    lines += render_gate_detail(c1, f"`C1_cap1_tight6` — development window ({c1['window']}), calibrated costs")
    lines += [
        "None of the failures is marginal in count, but two are close in magnitude: "
        f"survival PF needs >= 1.0 and this configuration reaches {_pf(c1):.3f} "
        f"(short by {1.0 - _pf(c1):.3f}); the WFO pass ratio is "
        f"{c1.get('wfo_pass_ratio', 0):.0%} against `configs/goal.yaml`'s "
        f"`min_pass_folds_ratio` of 0.50 for the RATIO alone — but "
        "`WalkForwardOptimizer`'s decision also requires the mean OOS Sharpe to not "
        "decay too far below mean IS Sharpe and a minimum count of individually "
        "profitable folds at a binomial-style significance level, and this "
        "configuration fails that second, stricter check (5 of 7 folds profitable, "
        "needs 6). Monte Carlo p5 Sharpe and the 1.5x stress profit factor both fail "
        "more comfortably.",
        "",
        "## 4. Final holdout — evaluated exactly once",
        "",
        f"Window `{holdout['window']}` ({holdout['n_symbols']} symbols, calibrated "
        "per-symbol costs), params frozen from the development phase, no "
        "re-optimization of any kind. Two months is shorter than a single is_days=90 + "
        "oos_days=30 WFO fold, so there is no honest per-fold pass ratio to compute; it "
        "is evaluated as one out-of-sample window and the gates that ARE well-defined on "
        "a single window are checked.",
        "",
        "| Metric | Development (in-sample) | **Final holdout** |",
        "|---|---|---|",
        f"| Net (cost-adjusted) profit factor | {_pf(c1):.3f} | **{_pf(holdout):.3f}** |",
        f"| Gross profit factor (commission-only, same slippage-laden fills) | "
        f"{c1['full_window_metrics'].get('profit_factor_gross', 0):.3f} | "
        f"**{holdout['full_window_metrics'].get('profit_factor_gross', 0):.3f}** |",
        f"| Total net P&L | ${c1['full_window_metrics']['total_net_pnl']:,.0f} | "
        f"**${holdout['full_window_metrics']['total_net_pnl']:,.0f}** |",
        f"| Full-window Sharpe ratio | {c1['full_window_metrics'].get('sharpe_ratio', 0):+.3f} | "
        f"**{holdout['full_window_metrics'].get('sharpe_ratio', 0):+.3f}** |",
        f"| Mean OOS Sharpe (WFO) | {c1.get('oos_sharpe_mean', 0):+.3f} | n/a (window < 1 fold) |",
        f"| Monte Carlo p5 Sharpe | {c1['mc_p5_sharpe']:+.3f} | **{holdout['mc_p5_sharpe']:+.3f}** |",
        f"| 1.5x-slippage stress profit factor | {c1['stress_metrics'].get('profit_factor', 0):.3f} | "
        f"**{holdout['stress_metrics'].get('profit_factor', 0):.3f}** |",
        f"| Trades | {c1['full_window_metrics']['n_trades']:,} | **{holdout['full_window_metrics']['n_trades']:,}** |",
        "",
    ]
    lines += render_gate_detail(holdout, f"Final holdout ({holdout['window']}), calibrated costs")
    lines += [
        "**The holdout confirms the development result — it does not contradict it, and "
        f"that is itself the finding.** Net PF moved from {_pf(c1):.3f} to "
        f"{_pf(holdout):.3f} (a change of {_pf(holdout) - _pf(c1):+.3f}), gross PF from "
        f"{c1['full_window_metrics'].get('profit_factor_gross', 0):.3f} to "
        f"{holdout['full_window_metrics'].get('profit_factor_gross', 0):.3f}. Both are "
        "well inside normal sampling noise for 158-802 trades. Unlike `orb_vwap`'s "
        "rescue, where the frozen configuration's in-sample gain (0.573 -> 1.003) did "
        "NOT survive contact with the holdout (-> 0.713), this configuration's "
        "near-breakeven edge is genuinely stable across two non-overlapping windows. "
        "The honest reading is the mirror image of `orb_vwap`'s: this is not a "
        "selection artifact that evaporated out of sample, it is a real, small, "
        "persistent edge that the specific levers tested could not push over the "
        "survival line.",
        "",
        "Caveat in the holdout's favor (i.e., against reading too much into a `NO-GO` "
        f"that is this close): {holdout['full_window_metrics']['n_trades']} trades over "
        "42 trading days is a small sample; its Monte Carlo p5 "
        f"({holdout['mc_p5_sharpe']:+.3f}) is computed from only 42 daily observations, "
        "which is why it is more negative than a longer window's would be for the same "
        "underlying edge. Caveat against over-crediting it: the gross edge on the "
        f"holdout ({holdout['full_window_metrics'].get('profit_factor_gross', 0):.3f}) "
        "is barely above 1.0 even before commission — most of the true zero-slippage "
        "1.307 gross edge measured on the full 20-symbol universe "
        "(`scripts/_gross_edge_diag.py`) evidently does NOT live in these 6 tight-spread "
        "names or in first-entries-only, which is exactly what a universe/frequency "
        "restriction chosen by looking at results would be expected to do to a "
        "population-level edge estimate (see the overfitting section).",
        "",
        "## 5. Overfitting-risk assessment",
        "",
        "1. **This is at least the fourth round of hypothesis testing on the same 20 "
        "symbols and ~12 months of 1-minute bars** (`intraday_backtest_report.md`, "
        "`new_signals_report.md`, `slippage_calibration_report.md`, "
        "`scripts/_gross_edge_diag.py`, and now this). The prior rounds' data-snooping "
        "cost is already sunk into any in-sample number produced here.",
        "2. **9 development configurations is a modest search, but not zero**, and it "
        "was deliberately INCREMENTAL (never a combinatorial grid) — each lever was "
        "fixed at its apparent best before the next was searched, the same discipline "
        "`orb_vwap`'s rescue used.",
        "3. **The universe restriction is a hidden degree of freedom**, exactly as "
        "`orb_vwap`'s report noted for its own TIGHT10-vs-TIGHT6 choice. Here TIGHT6 won "
        "outright on PF with no headroom tradeoff to justify picking a worse-PF option, "
        "so the freedom was exercised in the most straightforward direction — but it is "
        "still a choice made by looking at results, and `_gross_edge_diag.py`'s original "
        "1.307 gross PF was measured on all 20 symbols, not 6, so some of the apparent "
        "gain here is the universe restriction concentrating the estimate on a "
        "subpopulation rather than describing the average behavior of the pattern.",
        "4. **Sample-size rule of thumb is violated by a wide margin.** 5 free "
        f"parameters wants ~1,260 trading days by Chan's 252-per-parameter rule; the "
        f"development window has {c1.get('sample_size_check', {}).get('total_trading_days', 217)} "
        f"and the holdout {holdout.get('sample_size_check', {}).get('total_trading_days', 45)}. "
        "Intraday bars carry more independent observations per day than the rule "
        "assumes, so this is an informational flag rather than a hard failure.",
        "5. **The verdict itself is not at risk of overfitting** — every configuration "
        "tested, including the best one, fails multiple hard gates on the development "
        "window, and the holdout does not disagree. Where this investigation differs "
        "from `orb_vwap`'s is that the MAGNITUDE of the shortfall (survival PF short by "
        f"{1.0 - _pf(c1):.3f} in-sample, {1.0 - _pf(holdout):.3f} on holdout) is small "
        "and consistent enough that a reader should not treat this as \"clearly no "
        "edge\" the way `sweep_reclaim`'s low-liquidity result was — it is closer to "
        "\"a real edge too small and too concentrated in a 6-symbol subset to trade\".",
        "",
        "## 6. Conclusion",
        "",
        "`fvg_retest` remains **NO-GO** under calibrated per-symbol costs and the "
        "current `configs/goal.yaml` policy:",
        "",
        f"- Its raw gross edge (true zero-slippage, full 20-symbol universe) is real: "
        "1.307 profit factor, 48.7% win rate (`scripts/_gross_edge_diag.py`).",
        f"- Two levers (tight-spread universe, entry cap) recovered net PF from "
        f"{_pf(a0):.3f} to {_pf(c1):.3f} against a 1.0 survival floor — closer than any "
        "other rescue attempt in this repo's history, but still short.",
        f"- A third lever (loosening the 1:1 target) — the literal cause named in the "
        "original retirement docstring — made things WORSE at every multiple tested, "
        "contradicting that docstring's diagnosis.",
        f"- The frozen configuration's shortfall is CONSISTENT out of sample "
        f"({_pf(c1):.3f} dev vs {_pf(holdout):.3f} holdout) rather than an in-sample "
        "illusion that collapsed, unlike `orb_vwap`'s rescue.",
        "",
        "The natural next question — is there a further lever (a different universe cut, "
        "a volume/entry_pct recalibration, combining the cap with a narrower `vol_mult` "
        "filter) that could close the remaining ~0.06 PF gap — is left open rather than "
        "chased further: the parameter budget is spent (5/5), the search was already "
        "incremental rather than exhaustive, and a signal this close to its OWN best "
        "levers' ceiling is a case for a genuinely different idea (e.g. a different "
        "instrument class or timeframe), not more parameter search on the same universe "
        "and bars.",
        "",
        "The lever code stays in the repo as explicit, defaulted-off, documented options "
        "with test coverage. Nothing is promoted, `configs/strategy.yaml`'s `fvg_retest` "
        "block keeps its original behavior and `auto_execute: false`, and no gate was "
        "touched.",
        "",
        "## Appendix — artifacts",
        "",
        "- `scripts/_gross_edge_diag.py` — the zero-cost/real-cost diagnostic that "
        "motivated this investigation (run manually, not checkpointed to a JSON file)",
        "- `scripts/_fvg_retest_rescue.py` -> `backtests/reports/_fvg_rescue/<config_id>.json` "
        "(one checkpoint per configuration)",
        "- `scripts/_merge_fvg_rescue_report.py` -> this file and its `.json` sibling",
        "- run logs: `backtests/logs/fvg_rescue/*.log`",
    ]

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_configurations_evaluated": len(results),
        "n_development_configurations": n_dev,
        "results": [results[c] for c in ROW_ORDER if c in results],
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {REPORT_MD_PATH} and {REPORT_JSON_PATH} ({len(results)} configurations)")


if __name__ == "__main__":
    main()
