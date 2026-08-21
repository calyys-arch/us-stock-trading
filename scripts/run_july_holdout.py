"""
July-2026 HOLDOUT read on the entry-hypothesis cells that sit one gate away
from the official intraday AND (configs/goal.yaml `intraday`: drawdown,
has_oos_trades, pooled PF >= min_survival_profit_factor, stress PF >=
min_stress_profit_factor).

Why this window is a real holdout, not another slice of the same data: every
WFO in backtests/reports/entry_hypothesis_gate_report.json ran on
[2025-08-01, 2026-07-01) — end-exclusive, see
python/backtest/optimize.py's build_intraday_backtest_fn (`bars.index <
end_ts`, trades filtered to `exit_time < end_ts`). data/history_1m/ already
extends to 2026-07-31. So 2026-07 was cached but never read by any fold, any
grid search, or any rescue attempt. No new data fetch is needed.

What this script deliberately does NOT do:
  * no walk-forward, no per-fold re-optimization, no grid search. Each cell
    is replayed with its FROZEN `candidate_params` (the last fold's winner
    already recorded in the gate report) on one contiguous path. Nothing
    here can select a parameter, so nothing here can be in-sample.
  * no writes to entry_hypothesis_gate_report.json. That report is defined
    on the dev window; mixing a different window into its cells would make
    its gate vectors uninterpretable. Output goes to its own files.
  * no change to any official decision and no touch to `auto_execute`.
    A holdout is evidence, not a promotion path.

Cost-model sensitivity: each cell is replayed twice, under the flat 2.0 bps
half-spread the dev-window runs used and under the per-symbol calibrated
half-spreads from scripts/calibrate_slippage_spreads.py
(backtests/reports/calibrated_spreads.json). The calibrated set averages
2.15 bps -- almost exactly the flat placeholder -- but spans 0.33 bps (AAPL)
to 5.36 bps (STX), so the flat number is not a uniform over- or
under-charge; it is a per-name misallocation whose sign depends entirely on
which symbols a signal actually trades. Reporting one number would hide
that, so both are reported side by side as a band. CAVEAT: the calibrated
spreads come from 8 sessions of 2026-08 L2 depth, applied here to a 2026-07
window -- directionally informative about relative per-name liquidity, not a
measured 2026-07 spread.

Monte Carlo is reported as `insufficient` rather than as a number when the
path has fewer than MIN_MC_OBSERVATIONS trading days with activity: a p5
Sharpe bootstrapped from a handful of daily returns is noise wearing a
decimal point, and 21 sessions cannot supply more.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    run_intraday_backtest,
)
from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.optimize import (  # noqa: E402
    SIGNAL_PARAM_KEYS,
    build_intraday_backtest_fn,
    max_oos_drawdown_threshold,
    run_intraday_stress_test,
)

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
SPREADS_JSON = Path("backtests/reports/calibrated_spreads.json")
GOAL_PATH = Path("configs/goal.yaml")
STRATEGY_PATH = Path("configs/strategy.yaml")
OUT_JSON = Path("backtests/reports/july_holdout_report.json")
OUT_MD = Path("backtests/reports/july_holdout_report.md")

HOLDOUT_START = "2026-07-01"
HOLDOUT_END = "2026-08-01"
# Dev window every gate-report cell was optimized on, for the "is this
# really untouched" assertion below.
DEV_END = "2026-07-01"

# Below this many active trading days a bootstrap p5 is not reported.
MIN_MC_OBSERVATIONS = 30

# Cells one gate short of the official AND, plus auction 5m as the
# two-gates-short control (same signal as auction 15m on a faster chart).
DEFAULT_CELLS = (
    "vsa_no_demand_5m",
    "vsa_effort_15m",
    "auction_reclaim_15m",
    "auction_reclaim_5m",
)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _calibrated_spreads() -> dict[str, float]:
    if not SPREADS_JSON.exists():
        raise SystemExit(
            f"{SPREADS_JSON} missing — run scripts/calibrate_slippage_spreads.py first"
        )
    payload = json.loads(SPREADS_JSON.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for symbol, rec in (payload.get("symbols") or {}).items():
        if rec.get("suspect"):
            continue
        median = rec.get("median_bps")
        if median is not None:
            out[symbol] = float(median)
    return out


def _top_day_concentration(daily_returns, top: int = 5) -> dict:
    """Share of the path's NET total contributed by its `top` best days.

    Day-level, not trade-level: the engine's per-trade list is rebuilt
    separately below for per-symbol attribution, but concentration is the
    same fragility question at either resolution and the daily series is
    what Monte Carlo actually resamples, so this is the figure that
    explains a p5 result rather than merely accompanying it.

    `net_is_negligible` guards the ratio's degenerate regime. When winners
    and losers nearly cancel, net approaches zero and the share explodes
    (a 9,000% "concentration" says nothing about tail risk — it says the
    denominator is noise). In that regime the ratio is suppressed in
    favour of the flag, because a profit factor computed on a net of tens
    of dollars is not a measurement of anything.
    """
    values = [float(v) for v in (daily_returns or [])]
    if not values:
        return {"n_days": 0, "top_n": top, "top_share_of_net": None, "net_is_negligible": None}
    total = sum(values)
    best = sorted(values, reverse=True)[:top]
    top_sum = sum(best)
    negligible = bool(top_sum) and abs(total) < 0.1 * abs(top_sum)
    return {
        "n_days": len(values),
        "top_n": top,
        "top_sum": top_sum,
        "net_sum": total,
        "top_share_of_net": None if (negligible or not total) else top_sum / total,
        "net_is_negligible": negligible,
    }


def _per_symbol_attribution(
    bars_by_symbol: dict[str, pd.DataFrame],
    signal_name: str,
    base_cfg: dict,
    params: dict,
    engine_cfg: IntradayBacktestConfig,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warmup_days: int,
) -> list[dict]:
    """Net PnL and fill count per symbol for one replayed path.

    The gate report stores only pooled metrics, which is why the flat-vs-
    calibrated spread question could not be answered from it: a per-name
    cost correction is meaningless without knowing which names carry the
    PnL. Slicing/filtering mirrors build_intraday_backtest_fn exactly so
    these trades are the same set its metrics were computed from.
    """
    warmup_start = start_ts - pd.Timedelta(days=warmup_days)
    sliced = {}
    for symbol, bars in bars_by_symbol.items():
        window = bars.loc[(bars.index >= warmup_start) & (bars.index < end_ts)]
        if not window.empty:
            sliced[symbol] = window
    if not sliced:
        return []
    merged = {**base_cfg, **params}
    sig_params = {k: merged[k] for k in SIGNAL_PARAM_KEYS[signal_name] if k in merged}
    report = run_intraday_backtest(sliced, signal_name, sig_params, engine_cfg)
    agg: dict[str, dict] = {}
    for trade in report.trades:
        if not (start_ts <= trade.exit_time < end_ts):
            continue
        row = agg.setdefault(trade.symbol, {"symbol": trade.symbol, "n_trades": 0, "net_pnl": 0.0})
        row["n_trades"] += 1
        row["net_pnl"] += float(trade.net_pnl)
    return sorted(agg.values(), key=lambda r: r["net_pnl"])


def _replay(
    cell_key: str,
    cell: dict,
    bars_by_symbol: dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    stress_mult: float,
    min_survival_pf: float,
    min_stress_pf: float,
    max_dd: float,
    spreads: dict[str, float] | None,
    cost_label: str,
) -> dict:
    signal_name = cell["signal"]
    params = dict(cell["candidate_params"])
    minutes = int(cell["chart_minutes"])
    base_cfg = _load_yaml(STRATEGY_PATH)[signal_name]
    warmup_days = 1

    engine_cfg = IntradayBacktestConfig(
        chart_minutes=max(minutes, 1),
        time_stop_minutes=time_stop_for(max(minutes, 1)),
        half_spread_bps_by_symbol=spreads,
    )
    fn = build_intraday_backtest_fn(
        bars_by_symbol, signal_name, base_cfg,
        engine_cfg=engine_cfg, warmup_days=warmup_days,
    )
    metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), params)
    daily_returns = metrics.get("daily_returns", [])

    stress = run_intraday_stress_test(
        bars_by_symbol, signal_name, base_cfg, params,
        start_ts.to_pydatetime(), end_ts.to_pydatetime(),
        stress_slippage_multiplier=stress_mult, warmup_days=warmup_days,
        half_spread_bps_by_symbol=spreads, base_engine_cfg=engine_cfg,
    )

    n_obs = len(daily_returns or [])
    if n_obs >= MIN_MC_OBSERVATIONS:
        mc = MonteCarloValidator(n_sims=500).run(daily_returns)
        mc_p5 = float(mc.sharpe.p5)
        mc_note = None
    else:
        mc_p5 = None
        mc_note = (
            f"insufficient: {n_obs} active trading days < {MIN_MC_OBSERVATIONS} "
            "— a bootstrap p5 here would be noise, not a robustness verdict"
        )

    net_pf = metrics.get("profit_factor")
    stress_pf = stress.get("profit_factor")
    n_trades = int(metrics.get("n_trades") or 0)
    dd = metrics.get("max_drawdown")
    gates = {
        "has_trades": n_trades > 0,
        "drawdown_within_limit": dd is None or float(dd) >= -abs(max_dd),
        "profit_factor_ge_floor": net_pf is not None and float(net_pf) >= min_survival_pf,
        f"stress_slippage_{stress_mult:g}x_pf_ge_{min_stress_pf:g}": (
            stress_pf is not None and float(stress_pf) >= min_stress_pf
        ),
    }

    return {
        "cell": cell_key,
        "signal": signal_name,
        "chart_minutes": minutes,
        "cost_model": cost_label,
        "frozen_params": params,
        "holdout_window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "metrics": {k: v for k, v in metrics.items() if k != "daily_returns"},
        "stress_metrics": {k: v for k, v in stress.items() if k != "daily_returns"},
        "mc_p5_sharpe": mc_p5,
        "mc_note": mc_note,
        "concentration": _top_day_concentration(daily_returns),
        "per_symbol": _per_symbol_attribution(
            bars_by_symbol, signal_name, base_cfg, params,
            engine_cfg, start_ts, end_ts, warmup_days,
        ),
        "gates": gates,
        "gates_passed": sum(1 for v in gates.values() if v),
        "gates_total": len(gates),
    }


def _fmt(value, spec: str = ".3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _render_md(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# July-2026 holdout — 凍結參數單路徑重播")
    lines.append("")
    lines.append(f"- 產生時間：{payload['generated_at']}")
    lines.append(f"- Holdout 視窗：**{payload['holdout_window']}**（{payload['n_sessions']} 個交易日，end-exclusive）")
    lines.append(f"- 開發視窗（所有 WFO 用的）：{payload['dev_window']}")
    lines.append(f"- 資料：{payload['data_label']}")
    lines.append(f"- 壓力倍數：{payload['stress_multiplier']:g}×；PF 地板 {payload['min_survival_pf']:g}；壓力 PF 地板 {payload['min_stress_pf']:g}")
    lines.append("")
    lines.append("**這不是 WFO。** 每格用它在開發視窗最後一折已經選定的參數，"
                 "在 7 月單跑一條路徑；這裡沒有任何參數挑選動作，所以不可能是樣本內。"
                 "本檔不寫回 `entry_hypothesis_gate_report.json`（那份報告定義在開發視窗上），"
                 "也不改任何官方決策或 `auto_execute`。")
    lines.append("")
    lines.append("## 總表")
    lines.append("")
    lines.append("| 格子 | 成本模型 | 筆數 | 毛 PF | 淨 PF | 壓力 PF | 淨損益 | 回撤 | MC p5 | 四門 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in payload["results"]:
        m = row["metrics"]
        mc = row["mc_note"] and "樣本不足" or _fmt(row["mc_p5_sharpe"], ".2f")
        lines.append(
            f"| `{row['cell']}` | {row['cost_model']} | {int(m.get('n_trades') or 0)} | "
            f"{_fmt(m.get('profit_factor_gross'))} | {_fmt(m.get('profit_factor'))} | "
            f"{_fmt(row['stress_metrics'].get('profit_factor'))} | "
            f"{_fmt(m.get('total_net_pnl'), ',.0f')} | {_fmt(m.get('max_drawdown'), '.2%')} | "
            f"{mc} | {row['gates_passed']}/{row['gates_total']} |"
        )
    lines.append("")
    for row in payload["results"]:
        m = row["metrics"]
        lines.append(f"## `{row['cell']}` — {row['cost_model']}")
        lines.append("")
        lines.append(f"- 凍結參數：`{json.dumps(row['frozen_params'], sort_keys=True)}`")
        lines.append(f"- 訊號發出 / 成交：{m.get('signals_emitted')} / {m.get('signals_filled')}")
        lines.append(f"- 毛損益 {_fmt(m.get('gross_pnl'), ',.0f')}；成本 {_fmt(m.get('total_costs'), ',.0f')}；"
                     f"淨 {_fmt(m.get('total_net_pnl'), ',.0f')}")
        conc = row["concentration"]
        if conc.get("net_is_negligible"):
            lines.append(
                f"- 淨額趨近零：前 {conc['top_n']} 個最佳日合計 "
                f"{_fmt(conc['top_sum'], ',.4f')} 對上淨額 {_fmt(conc['net_sum'], ',.4f')}"
                f"（共 {conc['n_days']} 個活躍日）——贏虧幾乎互相抵銷，"
                "這條路徑的 PF 不構成任何量測"
            )
        elif conc.get("top_share_of_net") is not None:
            lines.append(f"- 前 {conc['top_n']} 個最佳交易日佔淨額：{_fmt(conc['top_share_of_net'], '.1%')}"
                         f"（共 {conc['n_days']} 個活躍日）")
        if row["mc_note"]:
            lines.append(f"- Monte Carlo：{row['mc_note']}")
        gate_txt = "、".join(
            f"{k} {'PASS' if v else 'FAIL'}" for k, v in row["gates"].items()
        )
        lines.append(f"- 四門：{gate_txt}")
        if row["per_symbol"]:
            lines.append("")
            lines.append("| 標的 | 筆數 | 淨損益 |")
            lines.append("|---|---:|---:|")
            for rec in row["per_symbol"]:
                lines.append(f"| {rec['symbol']} | {rec['n_trades']} | {_fmt(rec['net_pnl'], ',.0f')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    parser.add_argument("--start", default=HOLDOUT_START)
    parser.add_argument("--end", default=HOLDOUT_END)
    parser.add_argument(
        "--rerender",
        action="store_true",
        help="rebuild the markdown from the existing JSON (no backtests)",
    )
    args = parser.parse_args()

    if args.rerender:
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for row in payload.get("results") or []:
            conc = row.get("concentration") or {}
            top_sum, net_sum = conc.get("top_sum"), conc.get("net_sum")
            if top_sum is None or net_sum is None:
                continue
            negligible = bool(top_sum) and abs(net_sum) < 0.1 * abs(top_sum)
            conc["net_is_negligible"] = negligible
            conc["top_share_of_net"] = None if (negligible or not net_sum) else top_sum / net_sum
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        OUT_MD.write_text(_render_md(payload), encoding="utf-8")
        print(f"Re-rendered {OUT_MD} from {OUT_JSON}")
        return 0

    if pd.Timestamp(args.start) < pd.Timestamp(DEV_END):
        raise SystemExit(
            f"refusing to run: --start {args.start} precedes the dev window end {DEV_END}; "
            "that would not be a holdout"
        )

    report = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    dev_window = report.get("window") or {}
    if dev_window.get("end") != DEV_END:
        raise SystemExit(
            f"refusing to run: gate report dev window ends {dev_window.get('end')!r}, "
            f"expected {DEV_END!r} — the holdout boundary assumption no longer holds"
        )
    cells = report.get("cells") or {}

    goal = _load_yaml(GOAL_PATH).get("intraday", {})
    stress_mult = float(goal.get("stress_slippage_multiplier", 1.5))
    min_survival_pf = float(goal.get("min_survival_profit_factor", 1.0))
    min_stress_pf = float(goal.get("min_stress_profit_factor", 1.0))
    max_dd = max_oos_drawdown_threshold()

    from python.data.fixed_universe import load_universe_config

    universe_cfg = load_universe_config()
    symbols = universe_cfg["symbols"]

    sys.path.insert(0, str(Path(__file__).parent))
    from run_intraday_backtest import _load_real_bars

    load_start = str((pd.Timestamp(args.start) - pd.Timedelta(days=7)).date())
    bars_by_symbol = _load_real_bars(symbols, load_start, args.end)
    if not bars_by_symbol:
        raise SystemExit(f"no cached 1m bars in [{load_start}, {args.end})")

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    sessions = sorted({
        ts.date()
        for bars in bars_by_symbol.values()
        for ts in bars.loc[(bars.index >= start_ts) & (bars.index < end_ts)].index
    })
    data_label = (f"fixed top-{universe_cfg['top_n']} universe "
                  f"(computed_at={universe_cfg['computed_at']}), 1m bars via data/history_1m/")

    print(f"holdout [{start_ts.date()}, {end_ts.date()}) — {len(sessions)} sessions, "
          f"{len(bars_by_symbol)} symbols", flush=True)
    if sessions:
        print(f"    first={sessions[0]} last={sessions[-1]}", flush=True)

    calibrated = _calibrated_spreads()
    cost_models = [("flat_2.0bps", None), ("calibrated_per_symbol", calibrated)]
    med = statistics.median(calibrated.values()) if calibrated else float("nan")
    print(f"    calibrated spreads: {len(calibrated)} symbols, median {med:.3f} bps "
          f"(flat baseline 2.0 bps)", flush=True)

    results = []
    for cell_key in [c.strip() for c in args.cells.split(",") if c.strip()]:
        cell = cells.get(cell_key)
        if not cell:
            print(f"  !! {cell_key}: not in gate report — skipped", flush=True)
            continue
        if not cell.get("candidate_params"):
            print(f"  !! {cell_key}: no frozen params — skipped", flush=True)
            continue
        for cost_label, spreads in cost_models:
            print(f"\n=== {cell_key} | {cost_label} ===", flush=True)
            row = _replay(
                cell_key, cell, bars_by_symbol, start_ts, end_ts,
                stress_mult, min_survival_pf, min_stress_pf, max_dd,
                spreads, cost_label,
            )
            m = row["metrics"]
            print(f"    trades={int(m.get('n_trades') or 0)} "
                  f"gross_pf={_fmt(m.get('profit_factor_gross'))} "
                  f"net_pf={_fmt(m.get('profit_factor'))} "
                  f"stress_pf={_fmt(row['stress_metrics'].get('profit_factor'))} "
                  f"net=${_fmt(m.get('total_net_pnl'), ',.0f')} "
                  f"gates={row['gates_passed']}/{row['gates_total']}", flush=True)
            if row["mc_note"]:
                print(f"    mc: {row['mc_note']}", flush=True)
            results.append(row)

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "holdout_window": f"{start_ts.date()} .. {end_ts.date()}",
        "dev_window": f"{dev_window.get('start')} .. {dev_window.get('end')}",
        "n_sessions": len(sessions),
        "data_label": data_label,
        "stress_multiplier": stress_mult,
        "min_survival_pf": min_survival_pf,
        "min_stress_pf": min_stress_pf,
        "min_mc_observations": MIN_MC_OBSERVATIONS,
        "method": (
            "frozen last-fold params, single contiguous path, no WFO and no "
            "re-optimization; evidence only, does not alter official decisions "
            "or auto_execute"
        ),
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
