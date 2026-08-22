"""15 entry hypotheses × 7 independent gates × charts.

Intraday cells reuse scripts/run_intraday_backtest.py run_signal.
Daily hypotheses are recorded as not-applicable to 1m/5m/15m (they
are daily-bar strategies). Official hard-AND research GO is unchanged.

Usage:
    .venv/bin/python -u scripts/run_entry_hypothesis_gates.py --init-report
    .venv/bin/python -u scripts/run_entry_hypothesis_gates.py --import-existing
    .venv/bin/python -u scripts/run_entry_hypothesis_gates.py --hypothesis vsa_effort --chart 5 --resume
    .venv/bin/python -u scripts/run_entry_hypothesis_gates.py --resume
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.analytics.volume_route_policy import (
    CANONICAL_GATES,
    GATE_SHORT,
    collect_route_gates,
    list_routes,
    load_catalog,
    score_all,
    time_stop_for,
)
from python.analytics.entry_hypothesis_import import collect_existing_cells
from python.backtest.intraday_engine import IntradayBacktestConfig
from scripts.run_intraday_backtest import _load_bars_for_args, run_signal

ENTRY_CATALOG = Path("configs/entry_hypothesis_tests.yaml")
CHART_RUN_ORDER = (5, 15, 1)
REPORT_PATH = Path("backtests/reports/entry_hypothesis_gate_report.md")
REPORT_JSON_PATH = Path("backtests/reports/entry_hypothesis_gate_report.json")
DECOMPOSITION_JSON = Path("backtests/reports/slippage_decomposition.json")

THESIS = {
    "pairs_trading": "共整合配對價差 z-score 偏離後回歸",
    "xsection_mean_reversion": "昨天輸給橫截面平均的今天回歸",
    "daily_range_breakout": "日線收盤穿出近 N 日高低，明天開盤順勢",
    "absorption_breakout": "大量打穿近期高低且收盤穿過 → 順勢突破",
    "auction_reclaim": "前一日價值區外 fib 位置，吸收後收回",
    "vsa_effort": "大量卻走不動 + 輕量測試棒",
    "vsa_no_demand": "窄幅量縮的無需求／無賣壓，下一根無法延續",
    "obv_divergence": "價格創新高／低，當日 OBV 不確認",
    "l2_absorption": "大量碰到位能但守住 → fade",
    "sweep_reclaim": "刺破昨日高低／整數關後收回",
    "fvg_retest": "回測公平價值缺口後延續",
    "orb_vwap": "開盤區間突破 + VWAP",
    "orb_vwap_regime": "orb_vwap 再加 20 日趨勢過濾",
    "vwap_band_fade": "VWAP 帶外延伸做 fade",
    "vp_breakout": "成交量分布價值區突破",
}


def _cell_key(route: str, chart_minutes: int) -> str:
    return f"{route}_daily" if int(chart_minutes) == 0 else f"{route}_{int(chart_minutes)}m"


def _chart_label(minutes: int) -> str:
    return "daily" if int(minutes) == 0 else f"{int(minutes)}m"


def _verdict(strategies: list[dict], route_name: str, minutes: int, short: str) -> str:
    for row in strategies:
        if (
            row.get("route") == route_name
            and int(row.get("chart_minutes", -1)) == minutes
            and row.get("combination") == short
        ):
            return "PASS" if row.get("decision") == "GO" else "FAIL"
    return "—"


def _decomposition_index() -> dict[str, dict]:
    """Per-cell zero-slippage replay results, keyed by cell, or {} if absent.

    Produced by scripts/run_slippage_decomposition.py. Optional on purpose:
    the gate report must still render on a machine that has never run it.
    """
    if not DECOMPOSITION_JSON.exists():
        return {}
    try:
        payload = json.loads(DECOMPOSITION_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        row["cell"]: row
        for row in payload.get("results", [])
        if row.get("cell") and not row.get("error")
    }


def _cell_meta(cells: dict, route_name: str, minutes: int, decomp: dict | None = None) -> str:
    cell = cells.get(_cell_key(route_name, minutes))
    if not cell:
        return "未跑"
    if cell.get("decision") == "SKIPPED":
        return f"SKIPPED ({cell.get('reason')})"
    fm = cell.get("full_window_metrics") or {}
    src = cell.get("imported_from")
    tag = f", imported={src}" if src else ""
    if cell.get("superseded"):
        tag += "、重跑取代匯入值"
    pf = fm.get("profit_factor")
    pf_s = f"{float(pf):.2f}" if pf is not None else "n/a"
    row = (decomp or {}).get(_cell_key(route_name, minutes)) or {}
    pre = row.get("pf_pre_cost")
    if pre is not None:
        ratio = row.get("edge_cost_ratio")
        ratio_s = f"{float(ratio):.2f}x" if ratio is not None else "n/a"
        tag += f"，稅前 PF={float(pre):.2f}／每筆邊緣對成本 {ratio_s}"
    return (
        f"official={cell.get('decision')}, trades={fm.get('n_trades')}, "
        f"PF={pf_s}{tag}"
    )


def _decomposition_section(decomp: dict[str, dict]) -> list[str]:
    """Zero-slippage baseline for every replayable cell.

    This exists because `profit_factor_gross` was read as a pre-cost figure and
    used to retire signals for "having no edge". It is pre-COMMISSION only —
    `IntradayTrade.gross_pnl` is computed from slipped fill prices — so the
    retirement reasoning needed a genuine zero-cost replay to stand on.
    """
    if not decomp:
        return []
    rows = sorted(
        decomp.values(),
        key=lambda r: (int(r.get("chart_minutes") or 0), -(r.get("edge_cost_ratio") or -99)),
    )
    slip = sum(float(r.get("slippage") or 0.0) for r in rows)
    comm = sum(float(r.get("commission") or 0.0) for r in rows)
    flipped = [r for r in rows
               if (r.get("pf_pre_cost") or 0) >= 1.0 and (r.get("pf_normal") or 0) < 1.0]
    L = [
        "## 成本拆解：關掉滑價之後還剩什麼",
        "",
        "`profit_factor_gross` 是 **pre-commission**，不是 pre-cost — `IntradayTrade.gross_pnl`",
        "由已經過 `_slippage_price` 的成交價算出，`costs` 只含佣金。先前有多格是以",
        "「`profit_factor_gross` < 1，所以沒有毛邊緣」為由退役的，那個推論站不住。",
        "下表是凍結參數、同一份棒資料各重播兩次（正常成本／完全零成本）的結果。",
        "",
        "| 格子 | 圖 | 筆數 | 成本後 PF | 稅前 PF | 每筆邊緣 | 每筆成本 | 倍數 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    def _money(value) -> str:
        amount = float(value or 0.0)
        return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"

    for r in rows:
        pre = r.get("pf_pre_cost")
        L.append(
            f"| `{r['cell']}` | {r.get('chart_minutes')}m | {r.get('n_trades_zero')} | "
            f"{float(r.get('pf_normal') or 0):.3f} | "
            f"{'—' if pre is None else f'**{float(pre):.3f}**'} | "
            f"{_money(r.get('edge_per_trade'))} | {_money(r.get('cost_per_trade'))} | "
            f"{float(r.get('edge_cost_ratio') or 0):.2f} |"
        )
    total = slip + comm
    L.extend([
        "",
        f"- 滑價佔總成本 **{(slip / total if total else 0):.1%}**"
        f"（滑價 ${slip:,.0f}／佣金 ${comm:,.0f}）。成本優化的槓桿幾乎全在成交價，不在佣金方案。",
        f"- {len(flipped)}／{len(rows)} 格在關掉成本後 PF 翻正，"
        "所以「沒有毛邊緣」對這些格子是錯的退役理由。",
        "- 但沒有一格因此得救：每筆成本穩定落在 $150–270、與週期幾乎無關，"
        "而 1m 訊號每筆邊緣只有 $3–27。正確的說法是「邊緣小於執行成本」，不是「沒有邊緣」。",
        "- 「倍數」= 每筆稅前邊緣 ÷ 每筆總成本。壓力閘門把滑價乘 1.5，倍數需在滑價那塊留餘裕才可能存活。",
        "- 完整逐格拆解：`backtests/reports/slippage_decomposition.md`",
        "",
    ])
    return L


def _supersede_section(cells: dict) -> list[str]:
    """Cells whose imported numbers were replaced by a re-run, with the old values.

    Kept in the report rather than only in the JSON so the audit trail is
    visible to a reader who never opens the raw payload.
    """
    replaced = sorted(
        ((k, c["superseded"]) for k, c in cells.items() if c.get("superseded")),
        key=lambda kv: kv[0],
    )
    if not replaced:
        return []
    L = [
        "## 重跑取代的匯入值",
        "",
        "以下格子原本沿用舊報告的數字，但那些數字無法用現行已進版控的程式碼重現，",
        "已用重跑結果取代。舊值保留在 JSON 的 `superseded` 欄位供稽核。",
        "",
        "| 格子 | 舊來源 | 舊筆數 | 舊 PF | 舊淨額 | 新筆數 | 新 PF | 新淨額 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, old in replaced:
        new = cells[key].get("full_window_metrics") or {}

        def _n(value, spec: str) -> str:
            return "—" if value is None else format(float(value), spec)

        L.append(
            f"| `{key}` | `{old.get('imported_from') or '本地'}` | "
            f"{old.get('n_trades') or '—'} | {_n(old.get('profit_factor'), '.3f')} | "
            f"{_n(old.get('total_net_pnl'), ',.0f')} | "
            f"{new.get('n_trades') or '—'} | {_n(new.get('profit_factor'), '.3f')} | "
            f"{_n(new.get('total_net_pnl'), ',.0f')} |"
        )
    L.append("")
    return L


def _render(payload: dict) -> str:
    catalog = payload.get("catalog_obj") or load_catalog(ENTRY_CATALOG)
    routes = list_routes(catalog)
    cells = payload.get("cells") or {}
    strategies = payload.get("strategies") or []
    lines = [
        "# 15 個進場假設 × 7 閘門獨立計分 × 時間框架",
        "",
        f"Generated: {payload.get('generated_at', '')}",
        "",
        "## 方法",
        "",
        "這份報告回答的是：**每一個進場假設，單獨面對每一道閘門，在哪一種決策圖上過或不過。**",
        "官方研究 GO 仍是 hard AND，這裡**刻意不 AND**。",
        "",
        "計分規則：每個（假設 × 圖）只跑一次 WFO／Monte Carlo／1.5x 成本壓力測試，",
        "記下七個布林值，然後各閘門各判一次。PASS 只表示那一道閘門過了。",
        "",
        "七道閘門（「官方」欄標示它是否參與 hard AND，2026-08-22 起 `wfo_go` 與",
        "`monte_carlo_p5_sharpe` 由 warning 升為硬閘門，理由見 `sizing_wfo.md`）：",
        "",
        "| 閘門 | 單獨在問什麼 | 官方 |",
        "|---|---|---|",
        "| `wfo_go` | 走步最佳化本身是否判 GO（折通過比例／OOS Sharpe） | 硬（2026-08-22 升級） |",
        "| `oos_drawdown_within_limit` | 每個 OOS 折的最大回撤是否 ≤ 25% | 硬 |",
        "| `has_oos_trades` | 是否至少有一個 OOS 折真正平倉過 | 硬 |",
        "| `min_trades_per_oos_fold` | 全部 OOS 折合計成交是否 ≥ 40（pooled） | 軟 |",
        "| `cost_adjusted_profit_factor` | 成本後 pooled PF 是否 ≥ 1.0 | 硬 |",
        "| `monte_carlo_p5_sharpe` | bootstrap 第 5 百分位 Sharpe 是否 ≥ 0 | 硬（2026-08-22 升級） |",
        "| `stress_slippage_1.5x_pf_ge_1` | 成本 1.5 倍後 PF 是否仍 ≥ 1 | 硬（唯一樣本內） |",
        "",
        "硬閘門裡只有壓力閘門是樣本內的（用 `candidate_params` 在挑出它的同一段全窗重播）。",
        "把它當成唯一可翻的搖擺票，就能靠縮小部位換到 GO —— `sizing_wfo.md` 記錄了",
        "`auction_reclaim_5m` 在 0.75x 部位下正是這樣拿到假陽性 GO 的（8 折過 1 折、",
        "OOS Sharpe −6.95、MC p5 −2.22）。升級 `wfo_go` 與 `monte_carlo_p5_sharpe` 之後",
        "這條路被堵住；此變更只會讓 GO 變 NO-GO，不會反向，且未改動本矩陣任何一格的決策。",
        "",
        "## 為什麼不是每格都有 1m／5m／15m",
        "",
        "- **日線假設**（`pairs_trading`、`xsection_mean_reversion`、`daily_range_breakout`）",
        "  用的是日線價差／橫截面／日線突破，沒有「決策圖分鐘」。這三欄標 N/A，另留 `daily`。",
        "- **日內、決策圖可重採樣**（VSA／OBV／auction／absorption_breakout）：同一套棒規則",
        "  跑在收盤後的 1／5／15 分鐘棒上。原始資料永遠是 1 分鐘快取。",
        "- **日內、1 分鐘原生**（sweep／FVG／ORB／VWAP band／VP／l2_absorption）：",
        "  參數用「根數」或「開盤分鐘」，硬改成 15 分鐘圖會變成另一個假設。這份報告只跑 1m。",
        "",
        f"- Data: {payload.get('data_label', '（尚未載入 1m 快取）')}",
        f"- Window: [{payload.get('window', {}).get('start')} .. {payload.get('window', {}).get('end')})",
        f"- Time stop: `max(10, 2 * chart_minutes)`（僅日內格子）",
        f"- Mode: 七閘門獨立，無 AND",
        "",
    ]
    pending = payload.get("pending") or []
    if pending:
        lines.append(f"**尚未跑完的 WFO 格子：** {', '.join(pending)}")
        lines.append("")

    decomp = _decomposition_index()
    lines.extend(_decomposition_section(decomp))
    lines.extend(_supersede_section(cells))

    for route in routes:
        lines.append(f"## `{route.name}`")
        lines.append("")
        lines.append(f"- 進場假設：{THESIS.get(route.name, '')}")
        lines.append(f"- 種類：`{route.kind}`")
        lines.append(f"- 訊號：`{route.signal}`")
        lines.append("")
        if route.kind == "daily":
            lines.append("| 時間框架 | WFO 摘要 |")
            lines.append("|---|---|")
            lines.append(f"| daily | {_cell_meta(cells, route.name, 0, decomp)} |")
            lines.append("")
            lines.append("| 閘門 | daily | 1m | 5m | 15m |")
            lines.append("|---|---|---|---|---|")
            for full in CANONICAL_GATES:
                short = GATE_SHORT[full]
                vd = _verdict(strategies, route.name, 0, short)
                lines.append(f"| `{full}` | {vd} | N/A | N/A | N/A |")
            lines.append("")
            continue

        planned = tuple(route.charts)
        lines.append("| 時間框架 | WFO 摘要 |")
        lines.append("|---|---|")
        for minutes in (1, 5, 15):
            if minutes not in planned:
                lines.append(f"| {minutes}m | 不適用（{route.kind}） |")
            else:
                lines.append(f"| {minutes}m | {_cell_meta(cells, route.name, minutes, decomp)} |")
        lines.append("")
        lines.append("| 閘門 | 1m | 5m | 15m |")
        lines.append("|---|---|---|---|")
        for full in CANONICAL_GATES:
            short = GATE_SHORT[full]
            v1 = _verdict(strategies, route.name, 1, short) if 1 in planned else "N/A"
            v5 = _verdict(strategies, route.name, 5, short) if 5 in planned else "N/A"
            v15 = _verdict(strategies, route.name, 15, short) if 15 in planned else "N/A"
            lines.append(f"| `{full}` | {v1} | {v5} | {v15} |")
        lines.append("")

    go = sum(1 for s in strategies if s.get("decision") == "GO")
    lines.append("## 合計")
    lines.append("")
    lines.append(f"- 已計分裁決：{len(strategies)}")
    lines.append(f"- PASS：{go}")
    lines.append(f"- FAIL：{len(strategies) - go}")
    lines.append("")
    lines.append("PASS 不是 live 晉升，也不是官方 GO。`auto_execute` 不會因為這份表翻成 true。")
    lines.append("")
    return "\n".join(lines)


def _persist(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    slim = {k: v for k, v in payload.items() if k != "catalog_obj"}
    REPORT_JSON_PATH.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    REPORT_PATH.write_text(_render(payload), encoding="utf-8")


def _score_cells(cells: dict, catalog: dict) -> list[dict]:
    rows = []
    for cell in cells.values():
        gates = cell.get("route_gates") or {}
        if not gates:
            continue
        route = cell["route"]
        minutes = int(cell["chart_minutes"])
        for score in score_all(gates, route=route, chart_minutes=minutes, catalog=catalog):
            rows.append({
                "strategy_id": score.strategy_id,
                "route": route,
                "signal": cell.get("signal"),
                "chart_minutes": minutes,
                "combination": GATE_SHORT.get(score.required[0], score.strategy_id),
                "decision": score.decision,
                "required": list(score.required),
                "results": score.results,
                "failed": list(score.failed),
            })
    return rows


def _load_resume(path: Path, start: str, end: str) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    window = payload.get("window") or {}
    if window.get("start") != start or window.get("end") != end:
        raise SystemExit(
            f"--resume json window {window} != [{start}, {end}) — delete "
            f"{path} or pass matching --start/--end"
        )
    return payload


def main() -> int:
    catalog = load_catalog(ENTRY_CATALOG)
    routes = {r.name: r for r in list_routes(catalog)}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hypothesis", choices=["all", *routes], default="all")
    parser.add_argument("--chart", choices=["all", "0", "1", "5", "15"], default="all")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-report", action="store_true", help="write the empty matrix + method, no WFO")
    parser.add_argument(
        "--import-existing",
        action="store_true",
        help="seed cells from official reports already on disk (no WFO)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "re-run the selected cells even when --resume already has them. "
            "Needed to replace an imported cell whose published numbers cannot "
            "be reproduced by the current code"
        ),
    )
    args = parser.parse_args()

    selected = list(routes) if args.hypothesis == "all" else [args.hypothesis]
    if args.chart == "all":
        selected_charts = None
    else:
        selected_charts = {int(args.chart)}

    planned = []
    for name in selected:
        route = routes[name]
        charts = route.charts
        if selected_charts is not None:
            charts = tuple(c for c in charts if c in selected_charts)
        for minutes in charts:
            planned.append(_cell_key(name, minutes))

    # `pending` must stay catalog-wide: a scoped run (--hypothesis X --chart 15)
    # otherwise reports an empty backlog while most of the matrix is unrun.
    all_keys = [
        _cell_key(route.name, minutes)
        for route in routes.values()
        for minutes in route.charts
    ]

    payload = {
        "window": {"start": args.start, "end": args.end},
        "data_label": "",
        "catalog": str(ENTRY_CATALOG),
        "combination_mode": "single_gates",
        "cells": {},
        "strategies": [],
        "pending": planned,
        "catalog_obj": catalog,
    }

    if args.init_report or args.import_existing:
        if REPORT_JSON_PATH.exists():
            prior = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
            payload["cells"] = prior.get("cells") or {}
            payload["data_label"] = prior.get("data_label") or ""
            payload["window"] = prior.get("window") or payload["window"]
        if args.import_existing:
            imported = collect_existing_cells()
            for key, cell in imported.items():
                if key in payload["cells"] and not payload["cells"][key].get("imported_from"):
                    continue
                if key not in payload["cells"]:
                    payload["cells"][key] = cell
                    print(f"    import {key} <- {cell.get('imported_from')}", flush=True)
            payload["data_label"] = payload.get("data_label") or (
                "official reports imported; missing cells still need WFO"
            )
        payload["strategies"] = _score_cells(payload["cells"], catalog)
        payload["pending"] = [k for k in all_keys if k not in payload["cells"]]
        _persist(payload)
        print(f"Wrote {REPORT_PATH} pending={payload['pending']}")
        return 0

    bars_by_symbol, data_label, start_ts, end_ts = _load_bars_for_args(args)
    window = {
        "start": args.start if not args.demo else str(start_ts.date()),
        "end": args.end if not args.demo else str(end_ts.date()),
    }
    prior = _load_resume(REPORT_JSON_PATH, window["start"], window["end"]) if args.resume else {}
    cells = dict(prior.get("cells") or {})
    payload.update({
        "window": window,
        "data_label": data_label,
        "cells": cells,
        "pending": [k for k in all_keys if k not in cells],
    })

    signal_args = SimpleNamespace(demo=args.demo, start=args.start, end=args.end)
    run_order = [m for m in CHART_RUN_ORDER if selected_charts is None or m in selected_charts]
    if selected_charts and 0 in selected_charts:
        run_order = [0, *run_order]

    for minutes in run_order or [0]:
        for name in selected:
            route = routes[name]
            if minutes not in route.charts:
                continue
            key = _cell_key(name, minutes)
            if key in cells and not (args.force and key in planned):
                print(f"    skip {key}", flush=True)
                continue
            superseded = None
            if key in cells:
                prior_cell = cells[key]
                prior_fw = prior_cell.get("full_window_metrics") or {}
                superseded = {
                    "imported_from": prior_cell.get("imported_from"),
                    "decision": prior_cell.get("decision"),
                    "n_trades": prior_fw.get("n_trades"),
                    "profit_factor": prior_fw.get("profit_factor"),
                    "total_net_pnl": prior_fw.get("total_net_pnl"),
                    "reason": (
                        "replaced by a re-run under committed code; the published "
                        "numbers could not be reproduced from the same window and "
                        "frozen params"
                    ),
                }
                print(f"    force re-run {key} (was "
                      f"{prior_cell.get('imported_from') or 'locally run'})", flush=True)
            if route.kind == "daily":
                cells[key] = {
                    "signal": route.signal,
                    "route": name,
                    "decision": "SKIPPED",
                    "reason": "daily-bar hypothesis — 1m/5m/15m do not apply; daily WFO is a separate runner",
                    "chart_minutes": 0,
                    "data_label": data_label,
                }
                payload["cells"] = cells
                payload["pending"] = [k for k in all_keys if k not in cells]
                payload["strategies"] = _score_cells(cells, catalog)
                _persist(payload)
                print(f"    {key}: SKIPPED (daily)", flush=True)
                continue
            engine_cfg = IntradayBacktestConfig(
                chart_minutes=max(minutes, 1),
                time_stop_minutes=time_stop_for(max(minutes, 1)),
            )
            print(
                f"\n=== {name} signal={route.signal} chart={_chart_label(minutes)} ===",
                flush=True,
            )
            result = run_signal(
                route.signal, signal_args, bars_by_symbol, data_label, start_ts, end_ts,
                engine_cfg=engine_cfg,
            )
            result["route"] = name
            result["chart_minutes"] = minutes
            result["route_gates"] = collect_route_gates(
                result.get("gates") or {}, result.get("soft_gates") or {},
            )
            if superseded is not None:
                result["superseded"] = superseded
            cells[key] = result
            payload["cells"] = cells
            payload["pending"] = [k for k in all_keys if k not in cells]
            payload["strategies"] = _score_cells(cells, catalog)
            _persist(payload)
            print(
                f"    {key}: official={result.get('decision')} "
                f"vector={ {k: ('P' if v else 'F') for k, v in result['route_gates'].items()} }",
                flush=True,
            )

    payload["cells"] = cells
    payload["pending"] = [k for k in all_keys if k not in cells]
    payload["strategies"] = _score_cells(cells, catalog)
    _persist(payload)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
