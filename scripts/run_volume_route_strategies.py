"""VSA / OBV dual-route WFO: 7 separate gates × 1m / 5m / 15m.

Two entry routes (`vsa_no_demand`, `obv_divergence`). Each (route, chart)
cell runs one WFO. The seven gates are then scored independently — 7 VSA
tests and 7 OBV tests, compared across three decision charts. Official
research GO is unchanged and `auto_execute` stays false.

`--rescore-only` rewrites the 7-gate scoreboard from existing JSON.

Usage:
    .venv/bin/python -u scripts/run_volume_route_strategies.py --demo
    .venv/bin/python -u scripts/run_volume_route_strategies.py --rescore-only
    .venv/bin/python -u scripts/run_volume_route_strategies.py --resume
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
from python.backtest.intraday_engine import IntradayBacktestConfig
from scripts.run_intraday_backtest import _load_bars_for_args, run_signal

CHART_RUN_ORDER = (5, 15, 1)
REPORT_PATH = Path("backtests/reports/volume_route_strategies.md")
REPORT_JSON_PATH = Path("backtests/reports/volume_route_strategies.json")


def _cell_key(route: str, chart_minutes: int) -> str:
    return f"{route}_{int(chart_minutes)}m"


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


def _verdict(strategies: list[dict], route_name: str, minutes: int, short: str) -> str:
    for row in strategies:
        if (
            row.get("route") == route_name
            and int(row.get("chart_minutes") or 0) == minutes
            and row.get("combination") == short
        ):
            return "PASS" if row.get("decision") == "GO" else "FAIL"
    return "—"


def _cell_meta(cells: dict, route_name: str, minutes: int) -> str:
    cell = cells.get(_cell_key(route_name, minutes))
    if not cell:
        return "未跑"
    if cell.get("decision") == "SKIPPED":
        return f"SKIPPED ({cell.get('reason')})"
    fm = cell.get("full_window_metrics") or {}
    return (
        f"official={cell.get('decision')}, trades={fm.get('n_trades')}, "
        f"PF={float(fm.get('profit_factor') or 0):.2f}"
    )


def _render(payload: dict) -> str:
    cells = payload.get("cells") or {}
    strategies = payload.get("strategies") or []
    lines = [
        "# VSA / OBV 單閘門對照報告（1m / 5m / 15m）",
        "",
        f"Generated: {payload.get('generated_at', '')}",
        "",
        "> 每條路線、每個時間框架只跑一次 WFO。七個閘門各自獨立計分，",
        "> 不是 hard AND，也不是 128 種子集。官方研究 GO 維持原樣。",
        "> PASS/FAIL 不是 `auto_execute` 晉升。",
        "",
        f"- Data: {payload.get('data_label', '')}",
        f"- Window: [{payload.get('window', {}).get('start')} .. {payload.get('window', {}).get('end')})",
        f"- Time stop: `{payload.get('time_stop_rule', 'max(10, 2 * chart_minutes)')}`",
        f"- Mode: `{payload.get('combination_mode', 'single_gates')}`",
        f"- Tests: VSA 7 + OBV 7，各對 1m / 5m / 15m（共 42 個裁決）",
        "",
    ]
    pending = payload.get("pending") or []
    if pending:
        lines.append(f"**尚未跑完的 WFO 格子：** {', '.join(pending)}")
        lines.append("")

    for route_name, title in (("vsa", "VSA（`vsa_no_demand`）"), ("obv", "OBV（`obv_divergence`）")):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 時間框架 | WFO 摘要 |")
        lines.append("|---|---|")
        for minutes in (1, 5, 15):
            lines.append(f"| {minutes}m | {_cell_meta(cells, route_name, minutes)} |")
        lines.append("")
        lines.append("| 閘門 | 1m | 5m | 15m |")
        lines.append("|---|---|---|---|")
        for full in CANONICAL_GATES:
            short = GATE_SHORT[full]
            v1 = _verdict(strategies, route_name, 1, short)
            v5 = _verdict(strategies, route_name, 5, short)
            v15 = _verdict(strategies, route_name, 15, short)
            lines.append(f"| `{full}` | {v1} | {v5} | {v15} |")
        lines.append("")

    go = sum(1 for s in strategies if s.get("decision") == "GO")
    lines.append("## 合計")
    lines.append("")
    lines.append(f"- 已計分裁決：{len(strategies)}（滿格為 42）")
    lines.append(f"- PASS：{go}")
    lines.append(f"- FAIL：{len(strategies) - go}")
    lines.append("")
    return "\n".join(lines)


def _persist(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    REPORT_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    REPORT_PATH.write_text(_render(payload), encoding="utf-8")


def _score_cells(cells: dict, catalog: dict) -> list[dict]:
    rows = []
    for key, cell in cells.items():
        gates = cell.get("route_gates") or {}
        if not gates:
            continue
        route = cell["route"]
        minutes = int(cell["chart_minutes"])
        prefix = f"{route}_{minutes}m_"
        for score in score_all(gates, route=route, chart_minutes=minutes, catalog=catalog):
            rows.append({
                "strategy_id": score.strategy_id,
                "route": route,
                "signal": cell.get("signal"),
                "chart_minutes": minutes,
                "combination": score.strategy_id[len(prefix):],
                "decision": score.decision,
                "required": list(score.required),
                "results": score.results,
                "failed": list(score.failed),
            })
    return rows


def main() -> int:
    catalog = load_catalog()
    routes = {r.name: r for r in list_routes(catalog)}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", choices=["all", *routes], default="all")
    parser.add_argument("--chart", choices=["all", "1", "5", "15"], default="all")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="rewrite the 7-gate scoreboard from existing JSON; do not run WFO",
    )
    args = parser.parse_args()

    if args.rescore_only:
        if not REPORT_JSON_PATH.exists():
            raise SystemExit(f"no report to rescore: {REPORT_JSON_PATH}")
        payload = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
        payload["combination_mode"] = "single_gates"
        payload["catalog"] = "configs/volume_route_strategies.yaml"
        payload["strategies"] = _score_cells(payload.get("cells") or {}, catalog)
        _persist(payload)
        print(f"Rescored {REPORT_PATH} ({len(payload['strategies'])} single-gate rows)")
        _print_summary(payload)
        return 0

    selected_routes = list(routes) if args.route == "all" else [args.route]
    selected_charts = list(CHART_RUN_ORDER) if args.chart == "all" else [int(args.chart)]
    selected_charts = [m for m in CHART_RUN_ORDER if m in selected_charts]

    bars_by_symbol, data_label, start_ts, end_ts = _load_bars_for_args(args)
    window = {"start": args.start if not args.demo else str(start_ts.date()),
              "end": args.end if not args.demo else str(end_ts.date())}

    planned = [_cell_key(r, m) for m in selected_charts for r in selected_routes]
    prior = _load_resume(REPORT_JSON_PATH, window["start"], window["end"]) if args.resume else {}
    cells = dict(prior.get("cells") or {})

    payload = {
        "window": window,
        "data_label": data_label,
        "time_stop_rule": "max(10, 2 * chart_minutes)",
        "catalog": "configs/volume_route_strategies.yaml",
        "combination_mode": "single_gates",
        "cells": cells,
        "strategies": [],
        "pending": [k for k in planned if k not in cells],
    }

    signal_args = SimpleNamespace(demo=args.demo, start=args.start, end=args.end)
    for minutes in selected_charts:
        for route_name in selected_routes:
            key = _cell_key(route_name, minutes)
            if key in cells:
                print(f"    skip {key} (resume): official={cells[key].get('decision')}", flush=True)
                continue
            route = routes[route_name]
            engine_cfg = IntradayBacktestConfig(
                chart_minutes=minutes,
                time_stop_minutes=time_stop_for(minutes),
            )
            print(
                f"\n=== route={route_name} signal={route.signal} "
                f"chart={minutes}m time_stop={engine_cfg.time_stop_minutes}m ===",
                flush=True,
            )
            result = run_signal(
                route.signal, signal_args, bars_by_symbol, data_label, start_ts, end_ts,
                engine_cfg=engine_cfg,
            )
            result["route"] = route_name
            result["route_gates"] = collect_route_gates(
                result.get("gates") or {}, result.get("soft_gates") or {},
            )
            cells[key] = result
            payload["cells"] = cells
            payload["pending"] = [k for k in planned if k not in cells]
            payload["strategies"] = _score_cells(cells, catalog)
            _persist(payload)
            print(
                f"    {key}: official={result.get('decision')} "
                f"vector={ {k: ('P' if v else 'F') for k, v in result['route_gates'].items()} }",
                flush=True,
            )

    payload["cells"] = cells
    payload["pending"] = [k for k in planned if k not in cells]
    payload["strategies"] = _score_cells(cells, catalog)
    _persist(payload)
    print(f"\nReport written to {REPORT_PATH} (machine-readable: {REPORT_JSON_PATH})")
    _print_summary(payload)
    return 0


def _print_summary(payload: dict) -> None:
    strategies = payload.get("strategies") or []
    cells = payload.get("cells") or {}
    for key, cell in cells.items():
        route = cell.get("route")
        minutes = int(cell.get("chart_minutes") or 0)
        bits = []
        for full in CANONICAL_GATES:
            short = GATE_SHORT[full]
            bits.append(f"{short}={_verdict(strategies, route, minutes, short)}")
        print(f"  {key}: official={cell.get('decision')} | {' '.join(bits)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
