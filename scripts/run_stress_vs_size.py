"""
Asks whether shrinking position size lets a cell clear the 1.5x slippage
stress gate — the gate that is currently the only hard failure standing
between auction_reclaim 5m and a GO.

Why this is a separate question from run_execution_cost_study.py. That script
measures the edge-to-cost RATIO under size scaling, which is the right lens
for a cell whose cost-adjusted PF is already below 1. auction_reclaim 5m is
not that case: its cost-adjusted PF is 1.198 (PASS) and it dies only on
`stress_slippage_1.5x_pf_ge_1` at 0.970. The stress gate multiplies BOTH
slippage terms by 1.5, so the question is narrower and directly decidable —
at what size, if any, does stressed PF cross 1.0?

The arithmetic that makes this plausible rather than wishful. Stress scales
slippage by 1.5 at fixed size. Shrinking size by f scales the half-spread
component by f and the participation-impact component by f^2, while gross
edge scales by f. So stressed cost per unit of edge falls with f whenever any
of the slippage is impact rather than spread — measured 50/50 on this cell,
so the stressed cost ratio approaches 1.5 * 0.5 = 0.75 of its baseline value
as f goes to zero. A gate failing at 0.970 does not need much.

What works against it, and why this must be measured rather than derived:
  * `min_commission` is a floor of $1 per leg. As size shrinks, commission
    stops scaling and becomes a rising share of cost, which eventually
    reverses the gain.
  * Smaller size changes fill prices, so the trade set itself can shift.
  * Position sizing is min(risk-implied, notional-capped) shares; scaling
    both knobs by f does not guarantee every trade's size scales by exactly
    f, because the binding constraint can change.

Method: for each size factor, replay the frozen params twice — once at normal
cost, once at `stress_slippage_multiplier=1.5` — and report both profit
factors alongside the gate verdict. No WFO and no re-optimization: this
answers "does the gate flip", not "is the cell a GO". A flip here justifies a
full forced WFO re-run, it does not replace one.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
STRATEGY_PATH = Path("configs/strategy.yaml")
OUT_JSON = Path("backtests/reports/stress_vs_size.json")
OUT_MD = Path("backtests/reports/stress_vs_size.md")

STRESS_MULTIPLIER = 1.5
SIZE_FACTORS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1)
_MAX_WARMUP_DAYS = 40


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run(bars, signal_name, base_cfg, params, sig_keys, cfg, warmup_days, start_ts, end_ts) -> dict:
    merged = {**base_cfg, **params}
    sig_params = {k: merged[k] for k in sig_keys if k in merged}
    warmup_start = start_ts - pd.Timedelta(days=warmup_days)
    sliced = {
        symbol: window
        for symbol, df in bars.items()
        if not (window := df.loc[(df.index >= warmup_start) & (df.index < end_ts)]).empty
    }
    report = run_intraday_backtest(sliced, signal_name, sig_params, cfg)
    in_window = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
    metrics = metrics_from_report(IntradayBacktestReport(trades=in_window), cfg.capital)
    notionals = [t.shares * t.entry_price for t in in_window]
    return {
        "n_trades": int(metrics["n_trades"]),
        "profit_factor": metrics["profit_factor"],
        "total_net_pnl": float(metrics["total_net_pnl"]),
        "commission": float(metrics["total_costs"]),
        "max_drawdown": metrics.get("max_drawdown"),
        "avg_notional": (sum(notionals) / len(notionals)) if notionals else None,
    }


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


def _render_md(payload: dict) -> str:
    L = [
        "# 縮小部位能不能翻過 1.5 倍壓力閘門",
        "",
        f"- 產生時間：{payload['generated_at']}",
        f"- 視窗：{payload['window']}",
        f"- 壓力倍率：{STRESS_MULTIPLIER}x（同時作用在半價差與參與度衝擊）",
        "- 方法：凍結參數、單路徑重播，無 WFO、無重新最佳化",
        "",
        "壓力閘門把滑價乘 1.5。縮小部位讓半價差成本隨 f 線性下降、衝擊成本隨 f² 下降，"
        "而邊緣只隨 f 下降，所以只要滑價裡有一部分是衝擊，壓力後的每單位邊緣成本就會隨部位縮小而下降。"
        "反方向的力量是 `min_commission` 每腿 $1 的下限——部位夠小時佣金不再等比縮小，佔比反而上升。"
        "所以這件事只能量，不能推。",
        "",
    ]
    for row in payload.get("results", []):
        L.append(f"## `{row['cell']}`")
        L.append("")
        if row.get("error"):
            L.append(f"- 執行失敗：`{row['error']}`")
            L.append("")
            continue
        L.append(f"- 訊號 `{row['signal']}`，{row['chart_minutes']}m 圖，warmup {row['warmup_days']} 天")
        L.append("")
        L.append("| 倍率 | 筆數 | 平均名目 | 正常 PF | 壓力 PF | 正常淨額 | 壓力淨額 | 壓力閘門 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for e in row.get("sweep", []):
            L.append(
                f"| {e['factor']:.2f}x | {e.get('n_trades', '—')} | "
                f"{_money(e.get('avg_notional'))} | {_fmt(e.get('pf_normal'))} | "
                f"**{_fmt(e.get('pf_stress'))}** | {_money(e.get('net_normal'))} | "
                f"{_money(e.get('net_stress'))} | {'PASS' if e.get('stress_pass') else 'FAIL'} |"
            )
        L.append("")
        L.append(f"- **判讀**：{row.get('verdict', '')}")
        L.append("")
    return "\n".join(L) + "\n"


def _verdict(sweep: list[dict]) -> str:
    passing = [e for e in sweep if e.get("stress_pass")]
    base = next((e for e in sweep if e["factor"] == 1.0), None)
    if not passing:
        best = max((e for e in sweep if e.get("pf_stress") is not None),
                   key=lambda e: e["pf_stress"], default=None)
        if best is None:
            return "無法判讀（重播沒有成交）"
        return (
            f"**任何部位大小都過不了壓力閘門**，最好的是 {best['factor']:.2f}x 的壓力 PF "
            f"{best['pf_stress']:.3f}。縮小部位的方向對，但幅度不足。"
        )
    largest = max(passing, key=lambda e: e["factor"])
    note = ""
    if base is not None and base.get("net_normal"):
        shrink = largest.get("net_normal", 0) / base["net_normal"] if base["net_normal"] else 0
        note = (
            f" 代價是絕對報酬縮到基準的 {shrink:.0%}"
            f"（{_money(base['net_normal'])} → {_money(largest.get('net_normal'))}），"
            "因為部位小了；比值改善不等於賺得更多。"
        )
    return (
        f"**在 {largest['factor']:.2f}x 及更小的部位下壓力閘門翻成 PASS**"
        f"（壓力 PF {largest['pf_stress']:.3f}）。{note}"
        " 這是單路徑結果，要當結論必須用同樣的 sizing 重跑一次完整 WFO。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default="auction_reclaim_5m")
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    args = parser.parse_args()

    cells = (json.loads(GATE_REPORT.read_text(encoding="utf-8")).get("cells") or {})
    strategy = _load_yaml(STRATEGY_PATH)

    from run_intraday_backtest import SIGNAL_WARMUP_DAYS, _load_real_bars
    from python.backtest.optimize import SIGNAL_PARAM_KEYS
    from python.data.fixed_universe import load_universe_config

    universe = load_universe_config()
    load_start = str((pd.Timestamp(args.start) - pd.Timedelta(days=_MAX_WARMUP_DAYS)).date())
    print(f"loading 1m bars [{load_start}, {args.end}) for {len(universe['symbols'])} symbols...",
          flush=True)
    bars = _load_real_bars(universe["symbols"], load_start, args.end)
    if not bars:
        raise SystemExit("no cached 1m bars")
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "stress_multiplier": STRESS_MULTIPLIER,
        "size_factors": list(SIZE_FACTORS),
        "results": [],
    }

    for key in [c.strip() for c in args.cells.split(",") if c.strip()]:
        cell = cells.get(key)
        if not cell or not cell.get("candidate_params"):
            print(f"  !! {key}: no frozen params — skipped", flush=True)
            continue
        signal_name = cell["signal"]
        params = dict(cell["candidate_params"])
        minutes = int(cell["chart_minutes"])
        warmup = SIGNAL_WARMUP_DAYS.get(signal_name, 1)
        sig_keys = SIGNAL_PARAM_KEYS[signal_name]
        base_cfg = strategy[signal_name]
        row: dict = {
            "cell": key, "signal": signal_name, "chart_minutes": minutes,
            "warmup_days": warmup, "frozen_params": params, "sweep": [],
        }
        print(f"\n=== {key} ({signal_name}, {minutes}m, warmup {warmup}d) ===", flush=True)
        # Appended up front so each size factor's incremental _persist writes a
        # report that already contains this cell's partial sweep.
        payload["results"].append(row)

        try:
            for factor in SIZE_FACTORS:
                def cfg_for(stress: float) -> IntradayBacktestConfig:
                    cfg = IntradayBacktestConfig(
                        chart_minutes=max(minutes, 1),
                        time_stop_minutes=time_stop_for(max(minutes, 1)),
                        stress_slippage_multiplier=stress,
                    )
                    if factor == 1.0:
                        return cfg
                    return replace(
                        cfg,
                        risk_per_trade_pct=cfg.risk_per_trade_pct * factor,
                        max_notional_pct=cfg.max_notional_pct * factor,
                    )

                normal = _run(bars, signal_name, base_cfg, params, sig_keys,
                              cfg_for(1.0), warmup, start_ts, end_ts)
                stress = _run(bars, signal_name, base_cfg, params, sig_keys,
                              cfg_for(STRESS_MULTIPLIER), warmup, start_ts, end_ts)
                pf_stress = stress["profit_factor"]
                entry = {
                    "factor": factor,
                    "n_trades": normal["n_trades"],
                    "avg_notional": normal.get("avg_notional"),
                    "pf_normal": normal["profit_factor"],
                    "pf_stress": pf_stress,
                    "net_normal": normal["total_net_pnl"],
                    "net_stress": stress["total_net_pnl"],
                    "stress_pass": bool(pf_stress is not None and float(pf_stress) >= 1.0),
                }
                row["sweep"].append(entry)
                print(f"    {factor:.2f}x n={normal['n_trades']:5d} "
                      f"名目={_money(normal.get('avg_notional')):>10} "
                      f"PF={_fmt(normal['profit_factor'])} "
                      f"壓力PF={_fmt(pf_stress)} "
                      f"{'PASS' if entry['stress_pass'] else 'FAIL'}", flush=True)
                _persist(payload)
            row["verdict"] = _verdict(row["sweep"])
            print(f"    => {row['verdict']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    !! {row['error']}", flush=True)

        _persist(payload)

    _persist(payload)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _persist(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
