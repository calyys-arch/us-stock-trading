"""
Re-runs the FULL WFO pipeline at a reduced position size, to test whether a
sizing change that flips a gate on a single replay survives walk-forward.

Why this exists. scripts/run_stress_vs_size.py showed that auction_reclaim_5m's
1.5x stress profit factor crosses 1.0 once position size drops to 0.75x, for
only ~9% of the absolute return. That was a single-path replay of the frozen
candidate params over the whole window, which is exactly the basis the gate
report's `stress_metrics` uses — so the comparison is apples-to-apples and the
flip is real ON THAT BASIS.

It is also not sufficient, and the reason is the point of this script. Of the
four hard gates, three (drawdown, has_oos_trades, pooled cost-adjusted PF) do
read genuinely out-of-sample WFO folds — but each only tests SURVIVAL, not
consistency: drawdown merely has to sit inside a limit, has_oos_trades merely
has to exceed zero, and the pooled PF concatenates every fold into one basket
where a handful of large winners can carry a strategy whose per-fold behaviour
is dreadful. The fourth, the stress gate, is the one in-sample member: it
replays `candidate_params` over the same full window those params were chosen
on.

Everything that actually tests consistency — `wfo_go` (1 of 8 folds pass here),
`min_trades_per_oos_fold`, and Monte Carlo p5 — is filed under "warnings" and
cannot change the decision. So a sizing change that flips the lone in-sample
gate can hand the pipeline a GO while every consistency measure stays where it
was, which would demonstrate a hole in the gate set rather than an edge.

So the question this script answers is NOT "does stress PF flip at 0.75x" —
that is already known — but "do the WALK-FORWARD numbers move at all?" A
smaller position cuts participation impact quadratically, which genuinely
changes per-fold OOS returns, so the pass ratio and OOS Sharpe are entitled to
improve. If they do not, the honest reading is that shrinking size only shrank
the measurement noise around a zero (or negative) edge.

Method. The baseline is READ from the existing gate report rather than re-run:
that cell was itself produced by a fresh WFO under the currently committed
code (it carries a `superseded` record for the unreproducible import it
replaced), so re-running it would burn an hour to reproduce a number we
already own. Only the reduced-size arm is executed, through the same
`run_signal` the gate runner calls, with `risk_per_trade_pct` and
`max_notional_pct` scaled together.

Both must be scaled together: sizing is min(risk-implied shares,
notional-cap shares) and for this cohort the notional cap is almost always the
binding one (average trade notional sits at $199.9k against a $200k cap), so
scaling risk alone would leave nearly every trade the same size.

Results are written to their OWN report, deliberately not merged into
entry_hypothesis_gate_report.json: that matrix documents one consistent cost
and sizing regime, and silently mixing a 0.75x cell into it would make the
15x7 grid uninterpretable.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import IntradayBacktestConfig  # noqa: E402

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
OUT_JSON = Path("backtests/reports/sizing_wfo.json")
OUT_MD = Path("backtests/reports/sizing_wfo.md")

# Mirrors scripts/run_intraday_backtest.assemble_intraday_gates. Kept as a
# literal (rather than imported) so a stored arm can still be scored against
# the gate set that produced it: the `arms` in an existing sizing_wfo.json
# were run before WFO/MC were promoted to hard, and silently re-scoring them
# under today's rule would misrepresent what the pipeline actually decided.
HARD_GATES = (
    "wfo_go",
    "oos_drawdown_within_limit",
    "has_oos_trades",
    "cost_adjusted_profit_factor",
    "monte_carlo_p5_sharpe",
    "stress_slippage_1.5x_pf_ge_1",
)
# The gate set in force when the 0.75x arm below was run, and the reason this
# script exists. `_arm` reports both counts so the report can say plainly that
# the GO was an artifact of the older, gameable rule.
LEGACY_HARD_GATES = (
    "oos_drawdown_within_limit",
    "has_oos_trades",
    "cost_adjusted_profit_factor",
    "stress_slippage_1.5x_pf_ge_1",
)


def _fmt(value, spec: str = ".3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _money(value, spec: str = ",.0f", dash: str = "—") -> str:
    if value is None:
        return dash
    amount = float(value)
    return f"-${abs(amount):{spec}}" if amount < 0 else f"${amount:{spec}}"


def _arm(result: dict, factor: float) -> dict:
    """Flattens a run_signal result (or an imported gate cell, which has the
    same shape) down to the fields this comparison turns on."""
    full = result.get("full_window_metrics") or {}
    stress = result.get("stress_metrics") or {}
    gates = dict(result.get("gates") or {})
    soft = dict(result.get("soft_gates") or {})
    return _rescore({
        "size_factor": factor,
        "decision": result.get("decision"),
        "candidate_params": result.get("candidate_params"),
        "wfo_folds": result.get("wfo_folds"),
        "wfo_pass_ratio": result.get("wfo_pass_ratio"),
        "oos_sharpe_mean": result.get("oos_sharpe_mean"),
        "mc_p5_sharpe": result.get("mc_p5_sharpe"),
        "n_trades": full.get("n_trades"),
        "profit_factor": full.get("profit_factor"),
        "total_net_pnl": full.get("total_net_pnl"),
        "max_drawdown": full.get("max_drawdown"),
        "stress_profit_factor": stress.get("profit_factor"),
        "stress_net_pnl": stress.get("total_net_pnl"),
        "gates": gates,
        "soft_gates": soft,
    })


def _rescore(arm: dict) -> dict:
    """Recomputes the gate-set-dependent fields of a stored arm in place.

    `--rerender` reads arms written by earlier versions of this script, which
    predate the current-gate columns. Recomputing them from the stored
    `gates`/`soft_gates` booleans keeps an old JSON renderable without
    re-running the hour-long WFO, and is exact: the booleans themselves never
    depended on which dict a gate was filed under.
    """
    merged = {**(arm.get("soft_gates") or {}), **(arm.get("gates") or {})}
    arm["hard_pass_count"] = sum(1 for k in LEGACY_HARD_GATES if merged.get(k))
    arm["hard_pass_count_current"] = sum(1 for k in HARD_GATES if merged.get(k))
    arm["decision_under_current_gates"] = (
        "GO" if all(merged.get(k) for k in HARD_GATES) else "NO-GO"
    )
    return arm


def _verdict(base: dict, arm: dict) -> str:
    """Reads the reduced-size arm against the baseline, weighting the
    out-of-sample fields over the in-sample ones on purpose."""
    stress_flipped = (not base["gates"].get("stress_slippage_1.5x_pf_ge_1")) and \
        bool(arm["gates"].get("stress_slippage_1.5x_pf_ge_1"))
    all_hard = arm["hard_pass_count"] == len(LEGACY_HARD_GATES)

    def _delta(field: str) -> float | None:
        a, b = arm.get(field), base.get(field)
        return None if (a is None or b is None) else float(a) - float(b)

    d_ratio = _delta("wfo_pass_ratio")
    d_sharpe = _delta("oos_sharpe_mean")
    oos_moved = (d_ratio is not None and d_ratio > 0.01) or \
        (d_sharpe is not None and d_sharpe > 0.5)

    lines: list[str] = []
    if stress_flipped:
        lines.append(
            f"壓力閘門在 {arm['size_factor']:.2f}x 由 FAIL 翻成 PASS"
            f"（壓力 PF {_fmt(base['stress_profit_factor'])} → "
            f"{_fmt(arm['stress_profit_factor'])}），"
            f"四個硬閘門{'全過' if all_hard else '仍未全過'}。"
        )
    else:
        lines.append(
            f"壓力閘門在 {arm['size_factor']:.2f}x 仍是 "
            f"{'PASS' if arm['gates'].get('stress_slippage_1.5x_pf_ge_1') else 'FAIL'}"
            f"（壓力 PF {_fmt(arm['stress_profit_factor'])}）。"
        )

    lines.append(
        f"樣本外指標：WFO 通過率 {_fmt(base['wfo_pass_ratio'], '.3f')} → "
        f"{_fmt(arm['wfo_pass_ratio'], '.3f')}，"
        f"OOS Sharpe 平均 {_fmt(base['oos_sharpe_mean'], '.2f')} → "
        f"{_fmt(arm['oos_sharpe_mean'], '.2f')}，"
        f"MC p5 {_fmt(base['mc_p5_sharpe'], '.2f')} → {_fmt(arm['mc_p5_sharpe'], '.2f')}。"
    )

    if oos_moved:
        lines.append(
            "**樣本外證據確實跟著改善** — 縮小部位不只是把樣本內閘門推過門檻，"
            "衝擊成本下降真的改變了逐折的 OOS 報酬。這是值得往下走的方向，"
            "但仍要先確認通過率高到足以支撐一個真實的參數選擇規則。"
        )
    elif all_hard:
        lines.append(
            "**但樣本外證據沒有跟著動，所以這個 GO 是假陽性。** 翻掉的那一個閘門"
            "（壓力 PF）正好是舊制四個硬閘門裡唯一的樣本內指標——它用 "
            "`candidate_params` 在挑出這組參數的同一段全窗上重播。另外三個雖然真的是"
            "樣本外，但它們只測「活得下來」而不測「穩定」：回撤只要在上限內、"
            "有 OOS 成交只要大於零、彙總 OOS PF 把所有折併成一個池子，"
            "少數幾筆大贏就能撐起來。真正測穩定性的 wfo_go（8 折過 1 折）與 MC p5 "
            "當時都只是 warning，動不了決策。"
        )
        lines.append(
            f"**這個漏洞已於 2026-08-22 修補**：wfo_go 與 monte_carlo_p5_sharpe "
            f"已升為硬閘門（見 configs/goal.yaml 與 assemble_intraday_gates）。"
            f"同一組數字在現行閘門下是 "
            f"{arm.get('decision_under_current_gates', 'NO-GO')}"
            f"（{arm.get('hard_pass_count_current')}/{len(HARD_GATES)} 硬閘門），"
            "縮小部位再也換不到 GO。"
        )
    else:
        lines.append(
            "樣本外證據沒有跟著動，硬閘門也沒有全過。縮小部位在這格沒有買到任何東西。"
        )
    if arm.get("candidate_params") != base.get("candidate_params"):
        lines.append(
            f"注意：最後一折選出的參數改變了（{base.get('candidate_params')} → "
            f"{arm.get('candidate_params')}），所以兩臂不是同一組參數的純 sizing 對照，"
            "而是「在這個 sizing 下 WFO 會選什麼」的對照。"
        )
    return " ".join(lines)


def _render_md(payload: dict) -> str:
    base, arms = payload["baseline"], payload["arms"]
    rows = [base, *arms]
    L: list[str] = [
        "# 縮小部位之後，走動最佳化怎麼說",
        "",
        f"- 產生時間：{payload['generated_at']}",
        f"- 格子：`{payload['cell']}`（訊號 `{payload['signal']}`，"
        f"{payload['chart_minutes']}m 圖）",
        f"- 視窗：{payload['window']}",
        f"- 方法：{payload['method']}",
        "",
        "壓力對部位掃描（`stress_vs_size.md`）顯示這格的 1.5 倍壓力 PF 在 0.75x 部位跨過 1.0，"
        "而絕對報酬只掉 9%。那是凍結參數的單路徑重播，基礎與閘門報告的 `stress_metrics` 相同，"
        "所以那個翻正在該基礎上是真的。",
        "",
        "這一輪跑的時候，硬閘門是舊制的四個。其中三個（回撤、有 OOS 成交、彙總成本後 PF）"
        "確實讀的是樣本外的 WFO 折，但它們只測「活得下來」而不測「穩定」："
        "回撤只要在上限內、有成交只要大於零、彙總 PF 把所有折併成一個池子，"
        "少數幾筆大贏就能撐起逐折表現很糟的策略。第四個（壓力 PF）是唯一的樣本內成員，"
        "用 `candidate_params` 在挑出這組參數的同一段全窗上重播。而真正測穩定性的 "
        "`wfo_go` 與 Monte Carlo p5 當時都只是 warning，動不了決策。",
        "",
        "因此本報告要回答的不是「壓力 PF 會不會翻」，而是"
        "**「走動最佳化的數字會不會跟著動」**。答案是沒有，所以那個 GO 是假陽性——"
        "這個結果直接促成 2026-08-22 把 `wfo_go` 與 `monte_carlo_p5_sharpe` 升為硬閘門。"
        "下表同時列出兩套閘門集合下的決策：「當時決策」是這一輪實際跑出來的，"
        "「現行決策」是同一組布林值套上修補後的規則。",
        "",
        "## 逐臂比較",
        "",
        "| 部位 | 當時決策 | 舊制硬閘門 | 現行決策 | 現行硬閘門 | WFO 通過率 | "
        "OOS Sharpe | MC p5 | 筆數 | PF | 淨額 | 壓力 PF | 壓力淨額 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = f"{row['size_factor']:.2f}x" + (" (基準)" if row is base else "")
        L.append(
            f"| {label} | {row.get('decision') or '—'} | "
            f"{row['hard_pass_count']}/{len(LEGACY_HARD_GATES)} | "
            f"{row.get('decision_under_current_gates') or '—'} | "
            f"{row.get('hard_pass_count_current')}/{len(HARD_GATES)} | "
            f"{_fmt(row.get('wfo_pass_ratio'), '.3f')} | "
            f"{_fmt(row.get('oos_sharpe_mean'), '.2f')} | "
            f"{_fmt(row.get('mc_p5_sharpe'), '.2f')} | "
            f"{row.get('n_trades') or '—'} | {_fmt(row.get('profit_factor'))} | "
            f"{_money(row.get('total_net_pnl'))} | "
            f"{_fmt(row.get('stress_profit_factor'))} | "
            f"{_money(row.get('stress_net_pnl'))} |"
        )
    L.append("")
    L.append("## 硬閘門逐項（現行閘門集合）")
    L.append("")
    L.append("| 閘門 | 制度 | " + " | ".join(f"{r['size_factor']:.2f}x" for r in rows) + " |")
    L.append("|---|---|" + "---|" * len(rows))
    for gate in HARD_GATES:
        era = "舊制即為硬" if gate in LEGACY_HARD_GATES else "**2026-08-22 升為硬**"
        cells = " | ".join(
            ("PASS" if {**(r["soft_gates"] or {}), **(r["gates"] or {})}.get(gate)
             else "FAIL")
            for r in rows
        )
        L.append(f"| `{gate}` | {era} | {cells} |")
    L.append("")
    for arm in arms:
        L.append(f"- **判讀（{arm['size_factor']:.2f}x）**：{arm['verdict']}")
    L.append("")
    return "\n".join(L) + "\n"


def _persist(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="auction_reclaim_5m")
    parser.add_argument(
        "--size-factors", default="0.75",
        help="comma-separated reduced-size arms to run; the 1.00x baseline is "
             "read from the gate report, never re-run",
    )
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument(
        "--rerender", action="store_true",
        help="recompute verdicts from the existing JSON and rewrite the "
             "markdown, without re-running the hour-long WFO",
    )
    args = parser.parse_args()

    if args.rerender:
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        _rescore(payload["baseline"])
        for arm in payload.get("arms", []):
            _rescore(arm)
            arm["verdict"] = _verdict(payload["baseline"], arm)
            print(f"  {arm['size_factor']:.2f}x: {arm['verdict']}", flush=True)
        _persist(payload)
        print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
        return 0

    report = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    cell = (report.get("cells") or {}).get(args.cell)
    if not cell:
        raise SystemExit(f"{args.cell} not in {GATE_REPORT}")

    signal_name = cell["signal"]
    minutes = int(cell["chart_minutes"])
    factors = [float(f) for f in args.size_factors.split(",") if f.strip()]

    from run_intraday_backtest import _load_bars_for_args, run_signal

    signal_args = SimpleNamespace(demo=False, start=args.start, end=args.end)
    bars, data_label, start_ts, end_ts = _load_bars_for_args(signal_args)

    baseline = _arm(cell, 1.0)
    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "cell": args.cell,
        "signal": signal_name,
        "chart_minutes": minutes,
        "window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "method": (
            "baseline read from entry_hypothesis_gate_report.json (fresh WFO "
            "under committed code); reduced-size arms run through the same "
            "run_signal() the gate runner uses, scaling risk_per_trade_pct and "
            "max_notional_pct together"
        ),
        "data_label": data_label,
        "baseline": baseline,
        "arms": [],
    }
    print(f"baseline 1.00x: {baseline['decision']} "
          f"hard={baseline['hard_pass_count']}/{len(HARD_GATES)} "
          f"WFO={_fmt(baseline['wfo_pass_ratio'], '.3f')} "
          f"OOS Sharpe={_fmt(baseline['oos_sharpe_mean'], '.2f')} "
          f"壓力PF={_fmt(baseline['stress_profit_factor'])}", flush=True)

    for factor in factors:
        base_cfg = IntradayBacktestConfig(
            chart_minutes=max(minutes, 1),
            time_stop_minutes=time_stop_for(max(minutes, 1)),
        )
        engine_cfg = replace(
            base_cfg,
            risk_per_trade_pct=base_cfg.risk_per_trade_pct * factor,
            max_notional_pct=base_cfg.max_notional_pct * factor,
        )
        print(f"\n=== {args.cell} @ {factor:.2f}x "
              f"(risk {engine_cfg.risk_per_trade_pct:.4f}, "
              f"notional cap {engine_cfg.max_notional_pct:.4f}) ===", flush=True)
        result = run_signal(
            signal_name, signal_args, bars, data_label, start_ts, end_ts,
            engine_cfg=engine_cfg,
        )
        arm = _arm(result, factor)
        arm["verdict"] = _verdict(baseline, arm)
        payload["arms"].append(arm)
        _persist(payload)
        print(f"    => {arm['verdict']}", flush=True)

    _persist(payload)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
