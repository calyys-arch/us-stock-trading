"""Signal-internal filter ablation for vsa_no_demand / obv_divergence.

Leave-one-out + each-only + bare vs all-on, same WFO-winner params and
full-year window as backtests/reports/volume_book_signals_backtest_report.md.

This is NOT an 8-candidate WFO and does not change research GO AND.
`require_*` flags are engine overrides, not Chan free parameters.

Usage:
    .venv/bin/python -u scripts/ablate_volume_book_filters.py
    .venv/bin/python -u scripts/ablate_volume_book_filters.py --only vsa_no_demand
    .venv/bin/python -u scripts/ablate_volume_book_filters.py --resume
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from python.backtest.intraday_engine import (
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)

REPORT_PATH = Path("backtests/reports/volume_book_filter_ablation.md")
REPORT_JSON_PATH = Path("backtests/reports/volume_book_filter_ablation.json")
PARTIAL_DIR = Path("backtests/reports/volume_book_filter_ablation_partial")

WFO_PARAMS = {
    "vsa_no_demand": {
        "spread_atr_max": 0.4,
        "stop_atr_mult": 0.3,
        "vol_lookback": 2,
        "target_r_multiple": 1.5,
    },
    "obv_divergence": {
        "lookback_bars": 10,
        "obv_lag_frac": 0.35,
        "stop_atr_mult": 0.3,
        "target_r_multiple": 1.5,
    },
}

VSA_VARIANTS: list[tuple[str, dict]] = [
    ("all_on", {"require_location": True, "require_confirm": True, "require_volume": True}),
    ("no_location", {"require_location": False, "require_confirm": True, "require_volume": True}),
    ("no_confirm", {"require_location": True, "require_confirm": False, "require_volume": True}),
    ("no_volume", {"require_location": True, "require_confirm": True, "require_volume": False}),
    ("only_location", {"require_location": True, "require_confirm": False, "require_volume": False}),
    ("only_confirm", {"require_location": False, "require_confirm": True, "require_volume": False}),
    ("only_volume", {"require_location": False, "require_confirm": False, "require_volume": True}),
    ("bare_core", {"require_location": False, "require_confirm": False, "require_volume": False}),
]

OBV_VARIANTS: list[tuple[str, dict]] = [
    ("all_on", {"require_location": True, "require_obv_lag": True}),
    ("no_location", {"require_location": False, "require_obv_lag": True}),
    ("no_obv_lag", {"require_location": True, "require_obv_lag": False}),
    ("only_location", {"require_location": True, "require_obv_lag": False}),
    ("only_obv_lag", {"require_location": False, "require_obv_lag": True}),
    ("bare_core", {"require_location": False, "require_obv_lag": False}),
]

VARIANTS = {
    "vsa_no_demand": VSA_VARIANTS,
    "obv_divergence": OBV_VARIANTS,
}


def _partial_path(signal_name: str, variant: str) -> Path:
    return PARTIAL_DIR / f"{signal_name}_{variant}.json"


def _load_real_bars(symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    from python.data.intraday_cache import get_cached_intraday_panel

    panel = get_cached_intraday_panel(symbols, start, end)
    codes = set(panel.index.get_level_values("code"))
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols:
        if sym in codes:
            out[sym] = panel.xs(sym, level="code").sort_index()
        else:
            missing.append(sym)
    return out, missing


def _summarize(metrics: dict, n_wins: int) -> dict:
    n = int(metrics.get("n_trades", 0))
    win_rate = (n_wins / n) if n else 0.0
    daily = metrics.get("daily_returns") or []
    row = {
        "n_trades": n,
        "signals_emitted": int(metrics.get("signals_emitted", 0)),
        "signals_filled": int(metrics.get("signals_filled", 0)),
        "total_net_pnl": float(metrics.get("total_net_pnl", 0.0)),
        "gross_pnl": float(metrics.get("gross_pnl", 0.0)),
        "total_costs": float(metrics.get("total_costs", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "profit_factor_gross": float(metrics.get("profit_factor_gross", 0.0)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "win_rate": win_rate,
        "n_wins": n_wins,
        "n_days": int(metrics.get("n_days", 0) or len(daily)),
    }
    if "profit_factor_stress_1_5x" in metrics:
        row["profit_factor_stress_1_5x"] = float(metrics["profit_factor_stress_1_5x"])
    return row


def _run_one(
    bars_by_symbol: dict[str, pd.DataFrame],
    signal_name: str,
    params: dict,
    overrides: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    engine_cfg = IntradayBacktestConfig(
        chart_minutes=5,
        signal_filter_overrides=dict(overrides),
    )
    warmup_start = start_ts - pd.Timedelta(days=7)
    sliced = {}
    for symbol, bars in bars_by_symbol.items():
        window = bars.loc[(bars.index >= warmup_start) & (bars.index < end_ts)]
        if not window.empty:
            sliced[symbol] = window
    report = run_intraday_backtest(sliced, signal_name, params, engine_cfg)
    in_window = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
    filtered = IntradayBacktestReport(
        trades=in_window,
        signals_emitted=report.signals_emitted,
        signals_filled=report.signals_filled,
    )
    metrics = metrics_from_report(filtered, engine_cfg.capital)
    n_wins = sum(1 for t in in_window if t.net_pnl > 0)
    return _summarize(metrics, n_wins)


def _fmt_row(signal: str, variant: str, overrides: dict, r: dict) -> str:
    return (
        f"| `{signal}` | `{variant}` | {r['n_trades']} | {r['signals_emitted']} | "
        f"{r['signals_filled']} | ${r['total_net_pnl']:,.0f} | ${r['gross_pnl']:,.0f} | "
        f"${r['total_costs']:,.0f} | {r['profit_factor']:.3f} | {r['profit_factor_gross']:.3f} | "
        f"{r['win_rate']:.1%} | {r['sharpe']:+.3f} | {r['max_drawdown']:.1%} | {r['n_days']} |"
    )


def _render(payload: dict) -> str:
    start = payload["window"]["start"]
    end = payload["window"]["end"]
    lines = [
        "# Volume-book 訊號內部濾網 ablation",
        "",
        f"- 視窗：`[{start}, {end})`",
        f"- 宇宙：{payload['data_label']}",
        f"- 缺資料（未納入）：{', '.join(payload['missing_symbols']) or '無'}",
        f"- `chart_minutes=5`；參數為官方 volume-book 報告的 **WFO winner**，不是 yaml 預設",
        f"- 這不是 8-candidate WFO，也不是 research GO 重跑",
        f"- GEX 本來就不是 veto，未動；`spread_atr_max` 窄幅未拆",
        f"- 新旗標不進 `SIGNAL_PARAM_KEYS` / `param_grids.yaml` / Chan 自由參數",
        f"- 產生時間：{payload['generated_at']}",
        "",
        "## WFO winner 參數",
        "",
        f"- `vsa_no_demand`: `{payload['wfo_params']['vsa_no_demand']}`",
        f"- `obv_divergence`: `{payload['wfo_params']['obv_divergence']}`",
        "",
        "官方全年 stamp（同一視窗、同一 winner；本 ablation 的 `all_on` 應接近）：",
        "",
        "- vsa_no_demand：245 trades, net −$22,309, PF 0.80 — **NO-GO**",
        "- obv_divergence：4120 trades, net −$966,938, PF 0.34 — **NO-GO**",
        "",
    ]
    header = (
        "| Signal | Variant | Trades | Emitted | Filled | Net PnL | Gross PnL | Costs | "
        "PF (net) | PF (gross) | Win rate | Sharpe | Max DD | n_days |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for signal, variants in VARIANTS.items():
        have = any(f"{signal}:{variant}" in payload["results"] for variant, _ in variants)
        if not have:
            continue
        lines.extend([f"## `{signal}`", "", header, sep])
        for variant, overrides in variants:
            key = f"{signal}:{variant}"
            if key not in payload["results"]:
                continue
            r = payload["results"][key]
            lines.append(_fmt_row(signal, variant, overrides, r))
        lines.append("")
    lines.extend([
        "## 解讀（腳本自動表；敘事見同目錄報告正文）",
        "",
        "- leave-one-out：對照 `all_on`，看拿掉哪一個濾網使交易變多、PF 變好或變更差。",
        "- each-only：單濾網本身有沒有獨立邊緣。",
        "- 官方 stamp 仍是 NO-GO。此表不構成上紙上或改 `LIVE_SIGNALS` 的建議。",
        "",
    ])
    return "\n".join(lines)


def _write_payload(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(_render(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablate vsa_no_demand / obv_divergence internal filters")
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--only", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--resume", action="store_true", help="skip variants that already have a partial json")
    args = parser.parse_args()

    from python.data.fixed_universe import load_universe_config

    universe_cfg = load_universe_config()
    symbols = universe_cfg["symbols"]
    load_start = (pd.Timestamp(args.start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    print(
        f"loading 1m cache for {len(symbols)} symbols [{load_start}, {args.end}) ...",
        flush=True,
    )
    bars_by_symbol, missing = _load_real_bars(symbols, load_start, args.end)
    if not bars_by_symbol:
        raise SystemExit(
            f"no cached 1-minute bars for any universe symbol in [{load_start}, {args.end}]"
        )
    if missing:
        print(f"missing cache for {len(missing)} symbols: {missing}", flush=True)
    data_label = (
        f"fixed top-{universe_cfg['top_n']} universe "
        f"(computed_at={universe_cfg['computed_at']}), 1m bars via data/history_1m/"
    )
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    signals = [args.only] if args.only else list(VARIANTS)

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "window": {"start": args.start, "end": args.end},
        "data_label": data_label,
        "missing_symbols": missing,
        "symbols_used": sorted(bars_by_symbol),
        "wfo_params": WFO_PARAMS,
        "chart_minutes": 5,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "results": {},
    }
    if args.resume and REPORT_JSON_PATH.exists():
        saved = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
        payload["results"].update(saved.get("results") or {})

    for signal in signals:
        params = WFO_PARAMS[signal]
        for variant, overrides in VARIANTS[signal]:
            key = f"{signal}:{variant}"
            partial = _partial_path(signal, variant)
            if args.resume and (key in payload["results"] or partial.exists()):
                if key not in payload["results"] and partial.exists():
                    payload["results"][key] = json.loads(partial.read_text(encoding="utf-8"))
                r = payload["results"][key]
                print(
                    f"    skip {signal} {variant} (resume): trades={r['n_trades']} "
                    f"net=${r['total_net_pnl']:,.0f} pf={r['profit_factor']:.3f}",
                    flush=True,
                )
                continue
            print(f"    running {signal} {variant} {overrides} ...", flush=True)
            result = _run_one(bars_by_symbol, signal, params, overrides, start_ts, end_ts)
            result["filters"] = dict(overrides)
            payload["results"][key] = result
            payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            partial.write_text(json.dumps(result, indent=2), encoding="utf-8")
            _write_payload(payload)
            print(
                f"    {signal} {variant}: trades={result['n_trades']} "
                f"net=${result['total_net_pnl']:,.0f} pf={result['profit_factor']:.3f} "
                f"win={result['win_rate']:.1%}",
                flush=True,
            )

    _write_payload(payload)
    print(_render(payload))
    print(f"wrote {REPORT_PATH} and {REPORT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
