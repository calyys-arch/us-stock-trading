"""
Merges the per-config checkpoints written by scripts/_orb_vwap_rescue.py
into backtests/reports/orb_vwap_rescue_report.md + .json.

Same prior art / discipline as scripts/_merge_calibration_report.py: the
markdown report is GENERATED from the checkpointed run results rather than
hand-transcribed, so every number in it is traceable to a
`backtests/reports/_orb_rescue/<config_id>.json` produced by an actual
backtest run under `configs/goal.yaml`'s unmodified gate thresholds.

Usage:
    python scripts/_merge_orb_rescue_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

CHECKPOINT_DIR = Path("backtests/reports/_orb_rescue")
DIAGNOSTICS_PATH = Path("backtests/reports/_orb_vwap_diagnostics.json")
REPORT_MD_PATH = Path("backtests/reports/orb_vwap_rescue_report.md")
REPORT_JSON_PATH = Path("backtests/reports/orb_vwap_rescue_report.json")

# The order rows appear in the lever table (= the order the levers were
# actually tested in; see the report's methodology section).
ROW_ORDER = [
    "A0_asshipped_full20", "A1_stopfix_full20",
    "B1_tight10", "B2_tight6",
    "D1_atr025", "D2_atr050", "D3_atr100", "D4_atr200",
    "C1_cap1", "C2_cap2",
    "E1_r1", "E2_r2", "E3_r3",
    "F1_best_flat",
    "HOLDOUT_best",
]

GATE_ORDER = [
    "wfo_go", "oos_drawdown_within_limit", "has_oos_trades", "min_trades_per_oos_fold",
    "cost_adjusted_profit_factor", "monte_carlo_p5_sharpe", "stress_slippage_2x_net_positive",
]


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
    bits = [f"or={p.get('or_minutes')}"]
    if p.get("max_entries_per_session") is not None:
        bits.append(f"cap={p['max_entries_per_session']}")
    if p.get("stop_atr_buffer_mult"):
        bits.append(f"atr={p['stop_atr_buffer_mult']:g}")
    if p.get("target_r_multiple") is not None:
        bits.append(f"R={p['target_r_multiple']:g}")
    return f"{r['n_symbols']} sym, " + ", ".join(bits)


def _gate_cell(r: dict) -> str:
    gates = r.get("gates", {})
    failed = [g for g, ok in gates.items() if not ok]
    if not failed:
        return "**all pass**"
    short = {
        "wfo_go": "wfo", "oos_drawdown_within_limit": "dd", "has_oos_trades": "trades",
        "min_trades_per_oos_fold": "min_trades", "cost_adjusted_profit_factor": "PF",
        "monte_carlo_p5_sharpe": "mc_p5", "stress_slippage_2x_net_positive": "stress",
        "has_trades": "trades",
    }
    return "fail: " + ", ".join(short.get(g, g) for g in failed)


def render_lever_table(results: dict[str, dict]) -> list[str]:
    lines = [
        "| # | Lever | Config | WFO pass ratio | Mean OOS Sharpe | Cost-adj. PF | MC p5 Sharpe | 2x-stress net P&L | Trades | Gates |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    skip = {"HOLDOUT_best", "F1_best_flat"}   # rendered in their own sections
    for i, cid in enumerate([c for c in ROW_ORDER if c in results and c not in skip], start=1):
        r = results[cid]
        pr = r.get("wfo_pass_ratio")
        pr_s = f"{pr:.0%} ({r.get('wfo_passing_folds')}/{r.get('wfo_folds')})" if pr is not None else "n/a"
        sh = r.get("oos_sharpe_mean")
        sh_s = f"{sh:+.3f}" if sh is not None else "n/a"
        lines.append(
            f"| {i} | {r['lever']} | `{cid}` — {_label(r)} | {pr_s} | {sh_s} | "
            f"{_pf(r):.3f} | {r['mc_p5_sharpe']:+.3f} | "
            f"${r['stress_metrics']['total_net_pnl']:,.0f} | "
            f"{r['full_window_metrics']['n_trades']:,} | {_gate_cell(r)} |"
        )
    return lines


def render_gate_detail(r: dict, title: str) -> list[str]:
    lines = [f"**{title}** (gate thresholds read unmodified from `configs/goal.yaml`):", ""]
    gates = r.get("gates", {})
    for g in GATE_ORDER:
        if g in gates:
            lines.append(f"- [{'x' if gates[g] else ' '}] {g}")
    for g, ok in gates.items():
        if g not in GATE_ORDER:
            lines.append(f"- [{'x' if ok else ' '}] {g}")
    lines.append("")
    lines.append(f"**Verdict: {r['decision']}**")
    lines.append("")
    return lines


def main() -> None:
    results = load_results()
    diag = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8")) if DIAGNOSTICS_PATH.exists() else {}
    n_dev = len([c for c in results if c != "HOLDOUT_best"])
    prose = build_prose(results, diag, n_dev)
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(prose), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_configurations_evaluated": len(results),
        "n_development_configurations": n_dev,
        "diagnostics": diag,
        "results": [results[c] for c in ROW_ORDER if c in results],
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {REPORT_MD_PATH} and {REPORT_JSON_PATH} ({len(results)} configurations)")


def build_prose(results: dict[str, dict], diag: dict, n_dev: int) -> list[str]:
    from _orb_rescue_prose import render  # noqa: E402  (prose lives next door for readability)

    return render(results, diag, n_dev, render_lever_table, render_gate_detail, _pf)


if __name__ == "__main__":
    main()
