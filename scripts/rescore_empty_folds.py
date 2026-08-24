"""
Re-judge stored WFO runs under the empty-fold rule, without re-running them.

Why this exists. `python/backtest/walk_forward.py` used to score a fold that
produced no fills as a PASS: with no trades on either side both Sharpes are
0.0, so the decay test degenerates to `0.0 >= 0` and the absolute test to
`0.0 >= 0.0`. Harmless while `wfo_go` was a warning; decision-flipping once it
became a hard gate. Empty folds are now excluded from the pass ratio, and a
run where too few folds traded is INCONCLUSIVE rather than GO or NO-GO.

Any cell whose WFO finished before that change carries a pass ratio computed
over all folds, including empty ones. The fix is a pure re-reading of numbers
already on disk: every fold's `oos_metrics` is stored, so the trade counts
needed to identify empty folds are already there. Re-running instead would
cost roughly half an hour per fold to recompute answers that have not changed.

Scope, and what this deliberately does NOT touch. Only the WFO verdict and the
`wfo_go` gate are recomputed. The other hard gates read the full-window,
stress and Monte Carlo replays, which depend on `candidate_params` — and where
the LAST fold was empty, those params came from a tie at Sharpe 0.0 on an
empty in-sample window rather than from a real selection. Cells in that state
are flagged `candidate_params_from_empty_fold` and NOT silently rescored,
because their downstream evidence describes a parameter set nothing chose.
Fixing those honestly means re-running, so this script names them instead of
papering over them.

Usage:
    .venv/bin/python scripts/rescore_empty_folds.py            # report only
    .venv/bin/python scripts/rescore_empty_folds.py --write    # apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.backtest.walk_forward import WFOConfig  # noqa: E402

REPORT_JSON = ROOT / "backtests/reports/entry_hypothesis_gate_report.json"


def audit(payload: dict) -> list[dict]:
    """Find cells whose pass ratio counted folds that never traded.

    Cells store aggregates (`fold_oos_trades`, `wfo_passing_folds`) rather
    than whole fold objects, so the new ratio is reconstructed arithmetically.
    That reconstruction is only EXACT when every empty fold was one of the
    old rule's automatic passes, which required its in-sample Sharpe to be 0
    too. An empty OOS fold under a profitable in-sample window failed the
    decay test and was already counted as a failure — and in-sample Sharpes
    are not stored per fold. So a lower and an upper bound are reported, and a
    gate flip is only claimed when both bounds agree.
    """
    findings = []
    for key, cell in sorted((payload.get("cells") or {}).items()):
        trades = cell.get("fold_oos_trades")
        if not isinstance(trades, list) or not trades:
            continue
        n_empty = sum(1 for t in trades if not t)
        if not n_empty:
            continue
        total = len(trades)
        old_passing = int(cell.get("wfo_passing_folds") or 0)
        evaluable = total - n_empty
        # Every empty fold might have been an automatic pass (lower bound on
        # the new numerator) or none of them were (upper bound).
        lo = max(old_passing - n_empty, 0) / evaluable if evaluable else 0.0
        hi = min(old_passing, evaluable) / evaluable if evaluable else 0.0
        min_evaluable = WFOConfig().min_evaluable_folds_ratio
        min_pass = WFOConfig().min_pass_folds_ratio
        inconclusive = (evaluable / total) < min_evaluable
        stored_go = bool((cell.get("route_gates") or {}).get("wfo_go"))
        new_go_lo = (not inconclusive) and lo >= min_pass
        new_go_hi = (not inconclusive) and hi >= min_pass
        findings.append({
            "cell": key,
            "total_folds": total,
            "empty_folds": n_empty,
            "evaluable_folds": evaluable,
            "stored_pass_ratio": float(cell.get("wfo_pass_ratio") or 0.0),
            "stored_wfo_go": stored_go,
            "new_decision": "INCONCLUSIVE" if inconclusive else (
                "GO" if new_go_lo and new_go_hi else
                "NO-GO" if not new_go_hi else "AMBIGUOUS"),
            "new_pass_ratio_lo": lo,
            "new_pass_ratio_hi": hi,
            "exact": lo == hi,
            "flips_gate": (new_go_lo == new_go_hi) and stored_go != new_go_hi,
            # The last fold decides candidate_params, and therefore the
            # full-window, stress and Monte Carlo replays. If it was empty
            # those params were a tie at Sharpe 0.0 on an empty window, and
            # nothing downstream can be salvaged by re-reading the file.
            "candidate_params_from_empty_fold": not trades[-1],
        })
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="apply the rescored wfo_go / pass ratio to the report JSON")
    args = ap.parse_args()

    if not REPORT_JSON.exists():
        print(f"missing {REPORT_JSON}")
        return 1
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    findings = audit(payload)

    if not findings:
        print("沒有任何格子含零成交的折，不需要重評。")
        return 0

    print(f"{len(findings)} 格含零成交的折：\n")
    needs_rerun = []
    for f in findings:
        ratio = (f"{f['new_pass_ratio_lo']:.0%}" if f["exact"]
                 else f"{f['new_pass_ratio_lo']:.0%}~{f['new_pass_ratio_hi']:.0%}")
        flag = "  <- 翻閘門" if f["flips_gate"] else ""
        print(f"  {f['cell']}")
        print(f"    空折 {f['empty_folds']}/{f['total_folds']}（可評估 {f['evaluable_folds']}）；"
              f"通過率 {f['stored_pass_ratio']:.0%} -> {ratio}；"
              f"wfo_go {f['stored_wfo_go']} -> {f['new_decision']}{flag}")
        if f["candidate_params_from_empty_fold"]:
            needs_rerun.append(f["cell"])
            print("    !! 最後一折是空的：candidate_params 是空窗上的平手結果，"
                  "全窗／壓力／MC 都建立在沒有被真正選出的參數上，必須重跑")
    print()

    if needs_rerun:
        print("以下格子無法只靠重讀檔案修正，需要重跑：")
        for c in needs_rerun:
            print(f"  {c}")
        print()

    if not args.write:
        print("（唯讀。加 --write 才寫回。）")
        return 0

    cells = payload["cells"]
    written = 0
    for f in findings:
        if f["candidate_params_from_empty_fold"] or f["new_decision"] == "AMBIGUOUS":
            # Refuse to write a number this script cannot establish. Writing a
            # bound as if it were a measurement is the failure mode the whole
            # exercise exists to avoid.
            continue
        cell = cells[f["cell"]]
        cell["wfo_pass_ratio"] = f["new_pass_ratio_hi"]
        cell["wfo_decision"] = f["new_decision"]
        cell["wfo_evaluable_folds"] = f["evaluable_folds"]
        gates = cell.get("route_gates") or {}
        if "wfo_go" in gates:
            gates["wfo_go"] = f["new_decision"] == "GO"
        cell["rescored_empty_folds"] = {
            "reason": (
                "folds with zero fills were previously scored as passes; "
                "excluded from the pass ratio per walk_forward.FoldResult."
                "is_evaluable"
            ),
            "prior_pass_ratio": f["stored_pass_ratio"],
            "prior_wfo_go": f["stored_wfo_go"],
            "empty_folds": f["empty_folds"],
            "pass_ratio_was_reconstructed": not f["exact"],
        }
        written += 1
    if not written:
        print("沒有可以安全寫回的格子。")
        return 0
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {REPORT_JSON}（{written} 格）")
    print("接著重跑 scripts/run_entry_hypothesis_gates.py --resume 以重產 md／PDF。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
