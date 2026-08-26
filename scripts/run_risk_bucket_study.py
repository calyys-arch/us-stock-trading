"""
Does bucketing by asset class beat naive equal weight, or just look tidier?

The complaint this answers. Strategy V1 equal-weights 120 instruments, and
classifying that list showed what equal weight actually buys: 83 of the 120 are
US equity (54 single names, 23 sector ETFs, 6 broad-market ETFs), so US beta
gets 69% of the book by count. Worse, those three groups OVERLAP -- SPY holds
NVDA, XLK holds NVDA, SMH holds NVDA, and NVDA is in there on its own. Equal
weight by DOLLAR is not equal by RISK, and the true factor concentration is
higher than the headline 82%-in-equities suggests.

The fix, specified before looking at any result. Two changes, no fitted
parameters, because a weighting scheme tuned on the same window it is graded
on would be worth nothing:

  1. Collapse the three US equity groups into ONE bucket. They are one beta;
     counting them as three is the double-count.
  2. Weight the four remaining buckets -- US equity, international equity,
     bonds, commodities -- inversely to their own trailing 60-day realized
     volatility, renormalized to 1. Equal weight WITHIN each bucket. Inverse
     volatility is the textbook parameter-free allocation: no expected returns
     to estimate, no covariance to invert, nothing to fit.

Both arms then get the identical 15% volatility brake from
run_vol_target_study, so the only difference measured is the weighting.

Cost symmetry matters. Bucket weights drift daily as volatilities move, and
that turnover is as real as the brake's. Charging the brake but not the
reweighting would hand risk parity a free advantage, so bucket turnover pays
the same one-way cost, and the same 10% rebalance band suppresses noise churn.

Note on the holdout. 2024-01..2026-08 was already used once, to report V1.
Looking again is a second look at the same window, so it is reported as a
test of an a-priori rule rather than as a fresh holdout -- the scheme was not
chosen by consulting it. Read the walk-forward as the load-bearing number.
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
from python.portfolio.strategy_v1 import (  # noqa: E402
    COLLAPSE, bucket_weighted_gross, load_prices, load_universe,
)
from scripts.run_daily_baseline_gates import (  # noqa: E402
    TRADING_DAYS, _max_dd, _profit_factor, _sharpe,
)
from scripts.run_vol_target_study import (  # noqa: E402
    MAX_DD_LIMIT, ONE_WAY_COST_BPS, REBALANCE_BAND, STRESS_WINDOW,
    TARGET_VOLS, VOL_LOOKBACK, WINDOW_START, apply_exposure, exposure_path,
    gross_returns,
)

OUT_JSON = ROOT / "backtests/reports/risk_bucket_study.json"
OUT_MD = ROOT / "backtests/reports/risk_bucket_study.md"

TARGET_VOL = 0.15
HOLDOUT_START = "2024-01-01"

ARMS = ("equal_weight", "bucket_equal", "bucket_rp")


def gross_for_arm(panel: pd.DataFrame, bucket_of: dict[str, str], arm: str
                  ) -> tuple[pd.Series, pd.DataFrame | None]:
    if arm == "equal_weight":
        return gross_returns(panel), None
    return bucket_weighted_gross(
        panel, bucket_of, "equal" if arm == "bucket_equal" else "rp",
        VOL_LOOKBACK, REBALANCE_BAND, ONE_WAY_COST_BPS)


def _score(net: pd.Series, sims: int, mean_exposure: float) -> dict:
    mc = MonteCarloValidator(n_sims=sims, seed=42).run(
        [float(v) for v in net.tolist()])
    stress = net.loc[(net.index >= STRESS_WINDOW[0])
                     & (net.index <= STRESS_WINDOW[1])]
    dd = _max_dd(net)
    return {
        "n_days": int(len(net)),
        "mean_exposure": float(mean_exposure),
        "sharpe_annualized": _sharpe(net),
        "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
        "realized_vol": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": dd,
        "stress_max_drawdown": _max_dd(stress) if len(stress) else None,
        "profit_factor": _profit_factor(net),
        "mc_p5_sharpe": float(mc.sharpe.p5),
        "drawdown_gate_pass": bool(dd >= MAX_DD_LIMIT),
    }


def _braked(gross: pd.Series, target_vol: float
            ) -> tuple[pd.Series, float, pd.Series]:
    exp = exposure_path(gross, target_vol)
    net = apply_exposure(gross, exp)
    aligned = exp.reindex(net.index)
    return net, float(aligned.mean()), aligned


def run_wfo(panel: pd.DataFrame, bucket_of: dict[str, str], arm: str) -> dict:
    """Walk-forward the target vol under a given weighting scheme."""
    from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        # Warmup ends strictly before `start`, so it adds no look-ahead. Both
        # the bucket volatilities and the brake need it, and a fold shorter
        # than the lookback would otherwise yield an all-NaN allocation, no
        # position, and a silent 0.0.
        warmup = panel.loc[panel.index < start].tail(VOL_LOOKBACK * 3)
        window = panel.loc[(panel.index >= start) & (panel.index < end)]
        if len(window) < 30:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        combined = pd.concat([warmup, window])
        gross = gross_for_arm(combined, bucket_of, arm)[0]
        exp = exposure_path(gross, float(params["target_vol"]))
        net_all = apply_exposure(gross, exp)
        net = net_all.loc[net_all.index >= window.index[0]]
        if net.empty:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        rebalances = int((exp.reindex(net.index).diff().abs() > 1e-12).sum())
        return {
            "sharpe_ratio": _sharpe(net),
            # `n_trades` gates FoldResult.is_evaluable, whose question is "did
            # this fold produce an observation". For a vol-targeted basket the
            # answer is the number of days held, not the number of exposure
            # changes: a fold that sat at exposure 1.0 the whole quarter still
            # earned the basket's return. Clamping exposure changes to max(n, 1)
            # answered it the wrong way twice -- it made the guard vacuous, and
            # it fed a rebalance count of 1 to gates that report turnover.
            "n_trades": int(len(net)),
            "exposure_changes": rebalances,
            "total_net_pnl": float(net.sum()),
            "profit_factor": _profit_factor(net),
            "max_drawdown": _max_dd(net),
            "daily_returns": [float(v) for v in net.tolist()],
        }

    cfg = WFOConfig(
        is_days=504, oos_days=126, step_days=126,
        selection_objective="sharpe_subject_to_drawdown",
        selection_max_drawdown=abs(MAX_DD_LIMIT),
    )
    result = WalkForwardOptimizer(
        backtest_fn, cfg, [{"target_vol": tv} for tv in TARGET_VOLS]
    ).run(panel.index[0].to_pydatetime(), panel.index[-1].to_pydatetime())

    pooled = [r for f in result.folds for r in (f.oos_metrics.get("daily_returns") or [])]
    series = pd.Series(pooled)
    return {
        "decision": result.decision,
        "n_folds": len(result.folds),
        "positive_folds": getattr(result, "positive_folds", None),
        "required_positive_folds": getattr(result, "required_positive_folds", None),
        "oos_days": len(pooled),
        "oos_sharpe": _sharpe(series) if len(series) else None,
        "oos_profit_factor": _profit_factor(series) if len(series) else None,
        "oos_max_drawdown": _max_dd(series) if len(series) else None,
        "chosen_targets": [f.best_params.get("target_vol") for f in result.folds],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--wfo", action="store_true",
                    help="also walk-forward both arms (slow)")
    args = ap.parse_args()

    symbols, bucket_of = load_universe()
    panel = load_prices(symbols).loc[WINDOW_START:]
    print(f"籃子 {panel.shape[1]} 檔，{panel.index[0].date()} -> "
          f"{panel.index[-1].date()}\n")

    raw = {arm: gross_for_arm(panel, bucket_of, arm) for arm in ARMS}
    common = raw["equal_weight"][0].index
    for arm in ARMS:
        common = common.intersection(raw[arm][0].index)
    gross_by_arm = {arm: raw[arm][0].loc[common] for arm in ARMS}
    weights = raw["bucket_rp"][1]

    payload: dict = {
        "run_at": datetime.now().isoformat(),
        "window": [str(common[0].date()), str(common[-1].date())],
        "target_vol": TARGET_VOL,
        "n_symbols": int(panel.shape[1]),
        "buckets": {b: int(sum(1 for s in panel.columns
                               if COLLAPSE.get(bucket_of.get(s, ""), "") == b))
                    for b in sorted(set(COLLAPSE.values()))},
        "arms": {}, "holdout": {}, "bucket_weights_recent": {},
    }

    for name in ARMS:
        net, mean_exp, exposure = _braked(gross_by_arm[name], TARGET_VOL)
        payload["arms"][name] = _score(net, args.sims, mean_exp)
        hold = net.loc[net.index >= HOLDOUT_START]
        # Re-measure exposure on the holdout slice. Passing the full-sample
        # `mean_exp` made every holdout row report the 1,948-day average, which
        # showed up as a mean_exposure byte-identical to the full-window arm.
        hold_exp = float(exposure.reindex(hold.index).mean()) \
            if len(hold) else float("nan")
        payload["holdout"][name] = _score(hold, args.sims, hold_exp) \
            if len(hold) > 60 else None

    latest = weights.dropna().iloc[-1]
    payload["bucket_weights_recent"] = {k: float(v) for k, v in latest.items()}
    payload["bucket_weights_mean"] = {
        k: float(v) for k, v in weights.dropna().mean().items()}

    if args.wfo:
        for name in ARMS:
            print(f"走進式 {name} ...")
            payload.setdefault("wfo", {})[name] = run_wfo(panel, bucket_of, name)
        payload["wfo_measured_at"] = payload["run_at"]
        payload["wfo_window"] = payload["window"]
    elif OUT_JSON.exists():
        # A run without --wfo used to overwrite the file and delete the recorded
        # walk-forward results, leaving the in-sample `arms` table as the only
        # thing in it -- while paper_track_v1.BACKTEST still cited this file for
        # an out-of-sample benchmark. Carry the old section forward, stamped
        # with the window it was actually measured on so staleness is visible.
        prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        if "wfo" in prior:
            payload["wfo"] = prior["wfo"]
            payload["wfo_measured_at"] = prior.get("wfo_measured_at")
            payload["wfo_window"] = prior.get("wfo_window")
            payload["wfo_is_stale"] = prior.get("wfo_window") != payload["window"]
            if payload["wfo_is_stale"]:
                print(f"  注意：沿用 {payload['wfo_measured_at']} 的走進式結果，"
                      f"但視窗已變動。要更新請加 --wfo。")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    header = {"equal_weight": "等權 V1", "bucket_equal": "四桶各 25%",
              "bucket_rp": "桶逆波動"}
    print("  " + " " * 20 + "".join(f"{header[a]:>14}" for a in ARMS))
    for label, key in (("Sharpe", "sharpe_annualized"), ("CAGR", "cagr"),
                       ("實現波動", "realized_vol"),
                       ("最大回撤", "max_drawdown"),
                       ("2022 壓力回撤", "stress_max_drawdown"),
                       ("獲利因子", "profit_factor"),
                       ("MC p5 Sharpe", "mc_p5_sharpe"),
                       ("平均曝險", "mean_exposure")):
        cells = "".join(f"{payload['arms'][a][key]:>14.3f}" for a in ARMS)
        print(f"  {label:<20}{cells}")

    print("\n  桶權重：四桶各 25% 是固定的；逆波動的實際落點如下")
    for k, v in sorted(latest.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<16}最近 {v:>6.1%}   期間平均 "
              f"{payload['bucket_weights_mean'][k]:>6.1%}")

    if payload["holdout"]["bucket_rp"]:
        print(f"\n  {HOLDOUT_START} 之後")
        for name in ARMS:
            h = payload["holdout"][name]
            print(f"    {header[name]:<12} Sharpe {h['sharpe_annualized']:>5.2f}  "
                  f"CAGR {h['cagr']:>6.1%}  回撤 {h['max_drawdown']:>6.1%}  "
                  f"PF {h['profit_factor']:>4.2f}")

    if args.wfo:
        print("\n  走進式（載重數字）")
        for name in ARMS:
            w = payload["wfo"][name]
            print(f"    {header[name]:<12} {w['decision']:<6} "
                  f"{w['oos_days']} 個樣本外日  Sharpe {w['oos_sharpe']:.2f}  "
                  f"PF {w['oos_profit_factor']:.2f}  "
                  f"回撤 {w['oos_max_drawdown']:.1%}  "
                  f"折 {w['positive_folds']}/{w['n_folds']}")

    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
