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

Detection must not fail silently. Only cells run AFTER the per-fold detail was
added to run_intraday_backtest.py carry `fold_oos_trades`, and the first
version of this script `continue`d past cells without it — so it audited 2 of
19 cells and printed "nothing to rescore", which is the one output an audit
tool must never produce when it has not actually looked. Cells with no usable
per-fold basis are now reported as UNVERIFIABLE.

For those, `fold_oos_sharpes` is accepted as a fallback basis: an OOS Sharpe of
exactly 0.0 is the empty-fold signature (the degenerate value the old rule
scored as a pass), and a fold that really traded returning exactly 0.0 to full
float precision does not happen. Where even that is missing, the bug's
DIRECTION bounds the exposure: counting empty folds as passes can only inflate
`wfo_go`, never deflate it, so a cell already at `wfo_go: FAIL` is unaffected
whatever its folds did. Only PASS cells need per-fold evidence.

LOG_DERIVED_FOLDS backfills the two pre-fix PASS cells from their run logs,
with line references, because the terminal logs are ephemeral and outside the
repo: transcribing the numbers into version control is what makes the
correction reproducible instead of asserted.

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

# Per-fold OOS Sharpe and the OLD rule's pass flag, transcribed from the run
# logs of the two cells that finished before run_intraday_backtest.py started
# recording per-fold detail. These are the only two cells whose stored
# `wfo_go` is PASS, so they are the only two where the empty-fold bug can have
# changed a gate letter.
LOG_DERIVED_FOLDS = {
    "auction_reclaim_15m": {
        "source": "terminals/120074.txt lines 57-78 (run 2026-08-20 14:02-14:34)",
        "oos_sharpes": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "is_sharpes": [7.621, 2.755, 3.406, 0.0, 0.0, 0.0, 0.0, 0.0],
        "old_pass": [False, False, False, True, True, True, True, True],
    },
    "absorption_breakout_15m": {
        "source": "terminals/120085.txt lines 26-36",
        "oos_sharpes": [0.0, 0.0, 0.0, 0.0, -10.215, -13.886, 0.0, 1.213],
        "is_sharpes": [3.452, 0.0, 0.0, 0.0, 4.653, -1.559, -1.836, -3.037],
        "old_pass": [False, True, True, True, False, False, True, True],
    },
}


def _empty_basis(cell: dict, key: str) -> tuple[list[bool] | None, list[bool] | None, str]:
    """Per-fold (is_empty, old_pass) flags and the basis they came from.

    Returns (None, None, reason) when the cell carries no usable per-fold
    evidence, so the caller can report it rather than assume it is clean.
    """
    trades = cell.get("fold_oos_trades")
    passes = cell.get("fold_oos_pass")
    if isinstance(trades, list) and trades:
        empty = [not t for t in trades]
        return empty, (passes if isinstance(passes, list) else None), "fold_oos_trades"

    sharpes = cell.get("fold_oos_sharpes")
    backfill = LOG_DERIVED_FOLDS.get(key)
    basis = "fold_oos_sharpes"
    if not (isinstance(sharpes, list) and sharpes) and backfill:
        sharpes = backfill["oos_sharpes"]
        passes = backfill["old_pass"]
        basis = f"log-derived ({backfill['source']})"
    if isinstance(sharpes, list) and sharpes:
        # Exactly 0.0 is the degenerate value an untraded fold reports.
        empty = [float(s) == 0.0 for s in sharpes]
        return empty, (passes if isinstance(passes, list) else None), basis
    return None, None, "no per-fold detail stored"


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
    findings, unverifiable = [], []
    min_evaluable = WFOConfig().min_evaluable_folds_ratio
    min_pass = WFOConfig().min_pass_folds_ratio

    for key, cell in sorted((payload.get("cells") or {}).items()):
        stored_go = bool((cell.get("route_gates") or {}).get("wfo_go"))
        empty, old_pass, basis = _empty_basis(cell, key)
        if empty is None:
            # The bug only ever turned a fail into a pass, so a stored FAIL is
            # already the fail-closed answer and needs no fold evidence.
            unverifiable.append({
                "cell": key, "stored_wfo_go": stored_go, "reason": basis,
                "gate_at_risk": stored_go,
            })
            continue
        if not any(empty):
            continue

        total = len(empty)
        n_empty = sum(empty)
        evaluable = total - n_empty
        stored_ratio = float(cell.get("wfo_pass_ratio") or 0.0)

        if old_pass and len(old_pass) == total:
            # Exact: the fix only reclassifies empty folds, so a traded fold
            # keeps its old verdict and the new ratio is directly countable.
            kept = sum(1 for e, p in zip(empty, old_pass) if not e and p)
            lo = hi = (kept / evaluable) if evaluable else 0.0
        else:
            old_passing = int(cell.get("wfo_passing_folds")
                              or round(stored_ratio * total))
            lo = max(old_passing - n_empty, 0) / evaluable if evaluable else 0.0
            hi = min(old_passing, evaluable) / evaluable if evaluable else 0.0

        inconclusive = (evaluable / total) < min_evaluable
        new_go_lo = (not inconclusive) and lo >= min_pass
        new_go_hi = (not inconclusive) and hi >= min_pass
        findings.append({
            "cell": key,
            "basis": basis,
            "total_folds": total,
            "empty_folds": n_empty,
            "evaluable_folds": evaluable,
            "stored_pass_ratio": stored_ratio,
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
            "candidate_params_from_empty_fold": bool(empty[-1]),
        })
    return findings, unverifiable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="apply the rescored wfo_go / pass ratio to the report JSON")
    args = ap.parse_args()

    if not REPORT_JSON.exists():
        print(f"missing {REPORT_JSON}")
        return 1
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    findings, unverifiable = audit(payload)

    if unverifiable:
        at_risk = [u for u in unverifiable if u["gate_at_risk"]]
        print(f"{len(unverifiable)} 格沒有逐折明細，無法直接查核："
              f"{'、'.join(u['cell'] for u in unverifiable)}")
        print("  空折只會把 wfo_go 從 FAIL 推成 PASS，不會反向，"
              f"所以其中已是 FAIL 的 {len(unverifiable) - len(at_risk)} 格不受影響。")
        if at_risk:
            print("  !! 以下格子 wfo_go=PASS 且無逐折證據，閘門字母有風險：")
            for u in at_risk:
                print(f"       {u['cell']}（{u['reason']}）")
        print()

    if not findings:
        print("沒有任何可查核的格子含零成交的折。")
        return 0

    print(f"{len(findings)} 格含零成交的折：\n")
    unchosen = []
    for f in findings:
        ratio = (f"{f['new_pass_ratio_lo']:.0%}" if f["exact"]
                 else f"{f['new_pass_ratio_lo']:.0%}~{f['new_pass_ratio_hi']:.0%}")
        flag = "  <- 翻閘門" if f["flips_gate"] else ""
        print(f"  {f['cell']}  [{f['basis']}]")
        print(f"    空折 {f['empty_folds']}/{f['total_folds']}（可評估 {f['evaluable_folds']}）；"
              f"通過率 {f['stored_pass_ratio']:.0%} -> {ratio}；"
              f"wfo_go {f['stored_wfo_go']} -> {f['new_decision']}{flag}")
        if f["candidate_params_from_empty_fold"]:
            unchosen.append(f["cell"])
            print("    !! 最後一折是空的：candidate_params 是空窗上的平手結果，"
                  "全窗／壓力／MC 都建立在沒有被真正選出的參數上")
    print()

    if unchosen:
        print("以下格子的 wfo_go 可以精確更正，但其餘硬閘門要重跑才可信：")
        for c in unchosen:
            print(f"  {c}")
        print()

    if not args.write:
        print("（唯讀。加 --write 才寫回。）")
        return 0

    cells = payload["cells"]
    written = 0
    for f in findings:
        if f["new_decision"] == "AMBIGUOUS":
            # Refuse to write a number this script cannot establish. Writing a
            # bound as if it were a measurement is the failure mode the whole
            # exercise exists to avoid.
            continue
        cell = cells[f["cell"]]
        cell["wfo_pass_ratio"] = f["new_pass_ratio_hi"]
        cell["wfo_decision"] = f["new_decision"]
        cell["wfo_evaluable_folds"] = f["evaluable_folds"]
        for container in ("route_gates", "gates"):
            gates = cell.get(container) or {}
            if "wfo_go" in gates:
                gates["wfo_go"] = f["new_decision"] == "GO"
        cell["rescored_empty_folds"] = {
            "reason": (
                "folds with zero fills were previously scored as passes; "
                "excluded from the pass ratio per walk_forward.FoldResult."
                "is_evaluable"
            ),
            "basis": f["basis"],
            "prior_pass_ratio": f["stored_pass_ratio"],
            "prior_wfo_go": f["stored_wfo_go"],
            "empty_folds": f["empty_folds"],
            "pass_ratio_was_reconstructed": not f["exact"],
        }
        if f["candidate_params_from_empty_fold"]:
            # wfo_go above is exact. Everything downstream is not: it describes
            # a parameter set that a tie at Sharpe 0.0 on an empty in-sample
            # window handed over, so mark it rather than let it read as measured.
            cell["candidate_params_unchosen"] = {
                "reason": (
                    "the last fold was empty, so candidate_params came from a "
                    "0.0-Sharpe tie on an empty in-sample window; the "
                    "full-window, stress and Monte Carlo gates below describe "
                    "parameters nothing selected"
                ),
                "affected_gates": [
                    "cost_adjusted_profit_factor", "monte_carlo_p5_sharpe",
                    "stress_slippage_1.5x_pf_ge_1", "oos_drawdown_within_limit",
                    "has_oos_trades",
                ],
                "resolution": "re-run this cell under the fixed engine",
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
