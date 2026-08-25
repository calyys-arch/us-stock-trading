"""
Can volatility targeting close the last of the drawdown gap, and is it
doing anything a constant exposure cut would not?

Where this picks up. On a daily horizon the diversified buy-and-hold portfolio
clears every gate except drawdown, at -38.5% overall and -33.3% through 2022
against a -25% limit (backtests/reports/diversification_study.md). Breadth was
free and closed a third of the gap. The remaining tool is position size.

Volatility targeting scales total exposure by target_vol / recent realized
vol, so exposure falls automatically when markets get violent — which is when
drawdowns happen. That is the mechanism. But it also means average exposure
drops, and LESS EXPOSURE ALONE CUTS DRAWDOWN. Any study that reports vol
targeting against an unscaled baseline is partly measuring the leverage
reduction and calling it risk management.

So the control arm matters more than the treatment:

  unscaled        diversified buy-and-hold, exposure 1.0 throughout.
  constant_scale  a FIXED exposure, set to the average exposure the vol
                  targeted arm actually used. Same money at risk on average,
                  no timing. If this matches the vol targeted arm, then
                  targeting bought nothing and the honest recommendation is
                  "hold less", which is simpler and has no turnover.
  vol_target      exposure re-set from trailing realized volatility.

Costs. Vol targeting rebalances as volatility moves, and that turnover is
real: every change in exposure is charged one-way costs on the traded
difference. A rebalance band suppresses churn from noise. Reporting vol
targeting without this charge would credit it with returns it cannot capture.

No leverage. Exposure is capped at 1.0, so the arms differ only in how much
they DE-risk. Allowing >1 would introduce financing cost and margin mechanics
this study does not model.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from scripts.run_daily_baseline_gates import (  # noqa: E402
    _max_dd, _profit_factor, _sharpe, load_panel, TRADING_DAYS,
)

OUT_JSON = ROOT / "backtests/reports/vol_target_study.json"
OUT_MD = ROOT / "backtests/reports/vol_target_study.md"

WINDOW_START = "2018-06-01"
STRESS_WINDOW = ("2021-11-01", "2023-01-31")
MAX_DD_LIMIT = -0.25

VOL_LOOKBACK = 60          # trading days of realized vol
REBALANCE_BAND = 0.10      # only re-size when exposure drifts >10% relative
ONE_WAY_COST_BPS = 4.0     # half-spread plus commission on the traded change
MAX_EXPOSURE = 1.0
TARGET_VOLS = (0.15, 0.20, 0.25, 0.30, 0.35)


def gross_returns(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight daily returns over whatever names exist that day.

    Weights are decided from the prior close (`shift(1)`), so a name entering
    the panel is not credited with the return of the day it appeared.
    """
    rets = panel.pct_change(fill_method=None)
    hold = panel.notna().astype(float).shift(1).fillna(0.0)
    n = hold.sum(axis=1)
    weights = hold.div(n.where(n > 0), axis=0).fillna(0.0)
    return (weights * rets).sum(axis=1).dropna()


def exposure_path(gross: pd.Series, target_vol: float) -> pd.Series:
    """Exposure per day from trailing realized vol, with a rebalance band.

    Realized vol is computed over days strictly BEFORE the day it sizes:
    `.shift(1)` after the rolling window, so no day's exposure is informed by
    its own return.
    """
    realized = gross.rolling(VOL_LOOKBACK).std(ddof=1) * np.sqrt(TRADING_DAYS)
    raw = (target_vol / realized.shift(1)).clip(upper=MAX_EXPOSURE)

    out, current = [], np.nan
    for value in raw.to_numpy():
        if np.isnan(value):
            out.append(np.nan)
            continue
        if np.isnan(current) or abs(value - current) / max(current, 1e-9) > REBALANCE_BAND:
            current = value
        out.append(current)
    return pd.Series(out, index=gross.index)


def apply_exposure(gross: pd.Series, exposure: pd.Series) -> pd.Series:
    """Net returns after scaling, charging one-way costs on exposure changes."""
    exp = exposure.reindex(gross.index)
    valid = exp.notna()
    gross, exp = gross[valid], exp[valid]
    turnover = exp.diff().abs().fillna(0.0)
    cost = turnover * (ONE_WAY_COST_BPS / 10_000.0)
    return (exp * gross - cost).dropna()


def _score(net: pd.Series, exposure_mean: float, sims: int) -> dict:
    mc = MonteCarloValidator(n_sims=sims, seed=42).run(
        [float(v) for v in net.tolist()])
    stress = net.loc[(net.index >= STRESS_WINDOW[0])
                     & (net.index <= STRESS_WINDOW[1])]
    dd = _max_dd(net)
    return {
        "n_days": int(len(net)),
        "mean_exposure": float(exposure_mean),
        "sharpe_annualized": _sharpe(net),
        "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
        "realized_vol": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": dd,
        "stress_max_drawdown": _max_dd(stress) if len(stress) else None,
        "profit_factor": _profit_factor(net),
        "mc_p5_sharpe": float(mc.sharpe.p5),
        "drawdown_gate_pass": dd >= MAX_DD_LIMIT,
    }


def run_vol_target_wfo(panel: pd.DataFrame, cost_multiplier: float = 1.0) -> dict:
    """Walk-forward the target vol instead of choosing it by inspection.

    This is the load-bearing test of the whole study. Reading a table of five
    target vols and reporting the one that passed is selection on the test
    set: the 15% figure would be a decision made with knowledge of the answer.
    Walk-forward picks the target on in-sample data and is graded on the
    out-of-sample window that follows, which is the only version of the claim
    worth anything.

    `cost_multiplier` scales trading costs for the stress arm.
    """
    from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig

    grid = [{"target_vol": tv} for tv in TARGET_VOLS]

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        # The vol estimate needs history from before the window, or a fold
        # shorter than VOL_LOOKBACK yields an all-NaN exposure, no position,
        # and a silent 0.0 -- the same trap that made a 200-day average read
        # 0/25 positive folds in the baseline study. Warmup ends strictly
        # before `start`, so it adds no look-ahead.
        warmup = panel.loc[panel.index < start].tail(VOL_LOOKBACK * 2)
        window = panel.loc[(panel.index >= start) & (panel.index < end)]
        if len(window) < 30:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        gross_all = gross_returns(pd.concat([warmup, window]))
        exp = exposure_path(gross_all, float(params["target_vol"]))
        net_all = apply_exposure_scaled(gross_all, exp, cost_multiplier)
        net = net_all.loc[net_all.index >= window.index[0]]
        if net.empty:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        rebalances = int((exp.reindex(net.index).diff().abs() > 1e-12).sum())
        return {
            "sharpe_ratio": _sharpe(net),
            "n_trades": max(rebalances, 1),
            "total_net_pnl": float(net.sum()),
            "profit_factor": _profit_factor(net),
            "max_drawdown": _max_dd(net),
            "daily_returns": [float(v) for v in net.tolist()],
        }

    cfg = WFOConfig(is_days=504, oos_days=126, step_days=126)
    result = WalkForwardOptimizer(backtest_fn, cfg, grid).run(
        panel.index[0].to_pydatetime(), panel.index[-1].to_pydatetime())

    chosen = [f.best_params.get("target_vol") for f in result.folds
              if f.is_evaluable]
    oos_returns, oos_dds, oos_trades = [], [], []
    for f in result.folds:
        if not f.is_evaluable:
            continue
        oos_returns.extend((f.oos_metrics or {}).get("daily_returns") or [])
        oos_dds.append(float((f.oos_metrics or {}).get("max_drawdown") or 0.0))
        oos_trades.append(int((f.oos_metrics or {}).get("n_trades") or 0))
    pooled = pd.Series(oos_returns)
    return {
        "decision": result.decision,
        "total_folds": result.total_folds,
        "evaluable_folds": result.evaluable_folds,
        "positive_folds": result.positive_folds,
        "required_positive_folds": result.required_positive_folds,
        "oos_sharpe_mean": result.oos_sharpe_mean,
        "chosen_target_vols": chosen,
        "pooled_oos_sharpe": _sharpe(pooled) if len(pooled) > 1 else 0.0,
        "pooled_oos_profit_factor": _profit_factor(pooled) if len(pooled) else 0.0,
        "worst_fold_drawdown": min(oos_dds) if oos_dds else 0.0,
        "min_trades_in_a_fold": min(oos_trades) if oos_trades else 0,
        "pooled_oos_days": int(len(pooled)),
    }


def apply_exposure_scaled(gross: pd.Series, exposure: pd.Series,
                          cost_multiplier: float) -> pd.Series:
    exp = exposure.reindex(gross.index)
    valid = exp.notna()
    gross, exp = gross[valid], exp[valid]
    turnover = exp.diff().abs().fillna(0.0)
    cost = turnover * (ONE_WAY_COST_BPS * cost_multiplier / 10_000.0)
    return (exp * gross - cost).dropna()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--wfo", action="store_true",
                    help="walk-forward the target vol and score all 7 gates")
    args = ap.parse_args()

    panel = load_panel(min_rows=250)
    panel = panel.loc[panel.index >= pd.Timestamp(WINDOW_START)]
    gross = gross_returns(panel)
    print(f"universe: {panel.shape[1]} 檔，{len(gross)} 個交易日 "
          f"({gross.index[0].date()} -> {gross.index[-1].date()})", flush=True)

    cells = {}
    # Baseline on the same days the targeted arms can score, so the comparison
    # is not partly a different number of days.
    warm = gross.index[VOL_LOOKBACK:]
    base = gross.loc[warm]
    cells["unscaled"] = _score(base, 1.0, args.sims)
    print(f"\n== unscaled ==  曝險 1.00  Sharpe "
          f"{cells['unscaled']['sharpe_annualized']:.2f}  "
          f"MaxDD {cells['unscaled']['max_drawdown']:.1%}  "
          f"2022 {cells['unscaled']['stress_max_drawdown']:.1%}", flush=True)

    for tv in TARGET_VOLS:
        exp = exposure_path(gross, tv)
        net = apply_exposure(gross, exp)
        mean_exp = float(exp.reindex(net.index).mean())
        key = f"vol_target_{tv:.2f}"
        cells[key] = _score(net, mean_exp, args.sims)

        # Control: the SAME average exposure, held constant.
        flat = pd.Series(mean_exp, index=gross.index)
        net_flat = apply_exposure(gross.loc[net.index], flat.loc[net.index])
        cells[f"constant_{tv:.2f}"] = _score(net_flat, mean_exp, args.sims)

        a, b = cells[key], cells[f"constant_{tv:.2f}"]
        print(f"\n== 目標波動 {tv:.0%} ==  平均曝險 {mean_exp:.2f}", flush=True)
        print(f"   vol_target : Sharpe {a['sharpe_annualized']:.2f}  "
              f"CAGR {a['cagr']:.1%}  MaxDD {a['max_drawdown']:.1%}  "
              f"2022 {a['stress_max_drawdown']:.1%}  "
              f"閘門 {'PASS' if a['drawdown_gate_pass'] else 'FAIL'}", flush=True)
        print(f"   constant   : Sharpe {b['sharpe_annualized']:.2f}  "
              f"CAGR {b['cagr']:.1%}  MaxDD {b['max_drawdown']:.1%}  "
              f"2022 {b['stress_max_drawdown']:.1%}  "
              f"閘門 {'PASS' if b['drawdown_gate_pass'] else 'FAIL'}", flush=True)
        print(f"   → 動態相對固定：回撤差 "
              f"{(a['max_drawdown']-b['max_drawdown'])*100:+.1f} 個百分點，"
              f"Sharpe 差 {a['sharpe_annualized']-b['sharpe_annualized']:+.2f}",
              flush=True)

    gates = None
    if args.wfo:
        print("\n=== 走步最佳化：讓它自己在樣本外選目標波動 ===", flush=True)
        wfo = run_vol_target_wfo(panel)
        stress = run_vol_target_wfo(panel, cost_multiplier=1.5)
        gates = {
            "wfo_go": wfo["decision"] == "GO",
            "has_oos_trades": wfo["min_trades_in_a_fold"] > 0,
            "min_trades_per_oos_fold": wfo["min_trades_in_a_fold"] >= 40,
            "oos_drawdown_within_limit": wfo["worst_fold_drawdown"] >= MAX_DD_LIMIT,
            "cost_adjusted_profit_factor": wfo["pooled_oos_profit_factor"] >= 1.0,
            "stress_slippage_1.5x_pf_ge_1": stress["pooled_oos_profit_factor"] >= 1.0,
        }
        print(f"   判決 {wfo['decision']}  "
              f"{wfo['positive_folds']}/{wfo['evaluable_folds']} 折為正"
              f"（需 {wfo['required_positive_folds']}）", flush=True)
        print(f"   各折選到的目標波動: {wfo['chosen_target_vols']}", flush=True)
        print(f"   彙總 OOS: {wfo['pooled_oos_days']} 日，Sharpe "
              f"{wfo['pooled_oos_sharpe']:.2f}，PF "
              f"{wfo['pooled_oos_profit_factor']:.2f}", flush=True)
        print(f"   最差單折回撤 {wfo['worst_fold_drawdown']:.1%}；"
              f"最少單折再平衡 {wfo['min_trades_in_a_fold']} 次", flush=True)
        print(f"   1.5x 成本壓力 PF {stress['pooled_oos_profit_factor']:.2f} "
              f"(判決 {stress['decision']})", flush=True)
        for g, ok in gates.items():
            print(f"     {'PASS' if ok else 'FAIL'}  {g}", flush=True)

    payload = {
        "run_at": datetime.utcnow().isoformat(),
        "wfo": (wfo if args.wfo else None),
        "wfo_stress_1.5x": (stress if args.wfo else None),
        "gates": gates,
        "window_start": WINDOW_START,
        "stress_window": list(STRESS_WINDOW),
        "max_dd_limit": MAX_DD_LIMIT,
        "vol_lookback": VOL_LOOKBACK,
        "rebalance_band": REBALANCE_BAND,
        "one_way_cost_bps": ONE_WAY_COST_BPS,
        "max_exposure": MAX_EXPOSURE,
        "n_symbols": int(panel.shape[1]),
        "cells": cells,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _render(p: dict) -> str:
    L = [
        "# 波動目標化：它有沒有做到「少持有」以外的事",
        "",
        f"視窗 {p['window_start']} 起，{p['n_symbols']} 檔跨產業宇宙。"
        f"波動回看 {p['vol_lookback']} 日、再平衡帶 {p['rebalance_band']:.0%}、"
        f"單邊成本 {p['one_way_cost_bps']:.0f} bps、曝險上限 {p['max_exposure']:.1f}"
        "（不使用槓桿）。",
        "",
        "**`constant` 那一列是這份研究的重點。** 它把曝險固定在動態版本實際用到的"
        "**平均值**上。少持有本身就會降低回撤，所以只跟未縮放的基準比，"
        "等於把降槓桿的效果算成風險管理的功勞。動態版本要贏過同平均曝險的固定版本，"
        "才算真的做了事。",
        "",
        "| 組別 | 平均曝險 | Sharpe | CAGR | 實現波動 | 最大回撤 | 2022 回撤 | 回撤閘門 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, c in p["cells"].items():
        L.append(
            f"| `{key}` | {c['mean_exposure']:.2f} | "
            f"{c['sharpe_annualized']:.2f} | {c['cagr']:.1%} | "
            f"{c['realized_vol']:.1%} | {c['max_drawdown']:.1%} | "
            f"{c['stress_max_drawdown']:.1%} | "
            f"{'PASS' if c['drawdown_gate_pass'] else 'FAIL'} |"
        )
    L += [
        "",
        "曝險由**嚴格早於**它所規模的那一天的報酬算出（rolling 後再 `shift(1)`），"
        "所以沒有任何一天的部位看到自己的報酬。宇宙仍帶倖存者偏誤"
        "（名單取自 2026 年的流動性快照），絕對水位不能當預測。",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
