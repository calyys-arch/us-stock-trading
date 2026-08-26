"""How much of Strategy V1 is actually exposed to survivorship bias, and does
dropping that part cost anything?

    .venv/bin/python scripts/run_survivorship_exposure_study.py --wfo

THE PROBLEM, STATED PRECISELY. data/history was assembled from a 2026 liquidity
snapshot, so every instrument in it is one that still existed in 2026. A
backtest over that universe never holds a name that failed, which inflates the
result by however much the failures would have cost.

WHY THE USUAL FRAMING IS WRONG HERE. "Survivorship bias" is usually shorthand
for excluded bankruptcies, and the attrition number this repo measured -- 6 of
25 point-in-time 2016 S&P 500 names unavailable a decade later, 24% -- invites
reading all 24% as value destruction. It is not. Those six were ABC
(AmerisourceBergen, renamed Cencora), ADS (Alliance Data, renamed Bread
Financial), ACE (merged into Chubb and continues as CB), ADT (taken private at a
premium), AET (Aetna, acquired by CVS at a premium) and AGN (Allergan, acquired
by AbbVie at a premium). Three renames and three premium acquisitions. Zero
bankruptcies. Renames are a data-plumbing problem, not survivorship, and
excluding a premium acquisition makes a survivor-only backtest UNDERSTATE that
name's return.

The failure mode that does inflate a backtest is different: a name that falls,
gets dropped from the index on the way down, and is therefore absent from any
current-members universe. Chesapeake and Bed Bath & Beyond are the era's
examples. All of them sit in the 479 names never fetched, so the alphabetical
25-name sample says nothing about the effect that matters, and yfinance will
not serve partial history for a delisted ticker anyway.

SO THIS DOES NOT TRY TO MEASURE THE HAIRCUT. It bounds the exposure and then
prices the escape route.

  Exposure. Under bucket weighting, single companies carry 16.3% of the book:
      the four buckets are 25% each, and only the US equity bucket contains
      single names, at 65% of that bucket. The other 83.7% is ETFs, which do
      close but return NAV to holders rather than going to zero, so their bias
      is a weak selection effect rather than an excluded -100%.
  Escape route. Arm B drops all 54 single names and runs the same rule on the
      66 ETFs alone. If that costs little, survivorship stops being a risk to
      manage and becomes a variant to choose.
  Breakeven. Arm C charges the single-name sleeve a synthetic annual drag and
      reports how large the drag must be before shipped V1 loses to the ETF-only
      arm. That converts an unmeasurable quantity into a threshold you can
      compare against published estimates, which for surviving-members-only US
      equity backtests generally run 1-4% a year.
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

from python.portfolio.strategy_v1 import (  # noqa: E402
    COLLAPSE, bucket_weighted_gross, load_prices, load_universe,
    target_weights,
)
from scripts.run_daily_baseline_gates import (  # noqa: E402
    TRADING_DAYS, _max_dd, _profit_factor, _sharpe,
)
from scripts.run_risk_bucket_study import (  # noqa: E402
    HOLDOUT_START, MAX_DD_LIMIT, TARGET_VOL, TARGET_VOLS, WINDOW_START,
)
from scripts.run_vol_target_study import (  # noqa: E402
    ONE_WAY_COST_BPS, REBALANCE_BAND, VOL_LOOKBACK, apply_exposure,
    exposure_path,
)

OUT_JSON = ROOT / "backtests/reports/survivorship_exposure_study.json"
# The groups whose members are single companies, and so can actually fail. An
# ETF that closes liquidates at NAV; it does not print -100%.
SINGLE_NAME_GROUPS = ("us_single_name",)
# Annual drags to charge the single-name sleeve in the breakeven scan.
DRAGS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12)


def split_universe() -> tuple[list[str], list[str], dict[str, str]]:
    symbols, bucket_of = load_universe()
    singles = [s for s in symbols if bucket_of[s] in SINGLE_NAME_GROUPS]
    etfs = [s for s in symbols if bucket_of[s] not in SINGLE_NAME_GROUPS]
    return singles, etfs, bucket_of


def _score(net: pd.Series, exposure: pd.Series | None = None) -> dict:
    if net.empty:
        return {}
    dd = _max_dd(net)
    out = {
        "n_days": int(len(net)),
        "sharpe_annualized": _sharpe(net),
        "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
        "realized_vol": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": dd,
        "profit_factor": _profit_factor(net),
        "drawdown_gate_pass": bool(dd >= MAX_DD_LIMIT),
    }
    if exposure is not None:
        out["mean_exposure"] = float(exposure.reindex(net.index).mean())
    return out


def braked(gross: pd.Series, target_vol: float = TARGET_VOL
           ) -> tuple[pd.Series, pd.Series]:
    exp = exposure_path(gross, target_vol)
    return apply_exposure(gross, exp), exp


def gross_for(panel: pd.DataFrame, symbols: list[str],
              bucket_of: dict[str, str]) -> pd.Series:
    """Bucket-equal gross returns over a chosen subset of the universe."""
    held = [s for s in symbols if s in panel.columns]
    return bucket_weighted_gross(panel[held], bucket_of, "equal", VOL_LOOKBACK,
                                 REBALANCE_BAND, ONE_WAY_COST_BPS)[0]


def sleeve_returns(panel: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """Equal-weight return of one sleeve, weights set from the prior close."""
    held = [s for s in symbols if s in panel.columns]
    rets = panel[held].pct_change(fill_method=None)
    hold = panel[held].notna().astype(float).shift(1).fillna(0.0)
    n = hold.sum(axis=1)
    w = hold.div(n.where(n > 0), axis=0).fillna(0.0)
    priced = (w > 0) & rets.notna()
    return (w * rets).sum(axis=1).where(priced.any(axis=1))


def shipped_gross_with_drag(panel: pd.DataFrame, bucket_of: dict[str, str],
                            singles: list[str], annual_drag: float
                            ) -> pd.Series:
    """Shipped V1, with the single-name sleeve charged `annual_drag` per year.

    The drag stands in for the return of names that would have been held and
    failed. It is applied only to the single-name sleeve's share of the book, so
    a 3% sleeve drag is roughly 3% * 16.3% = 0.5% at the portfolio level.
    """
    symbols = list(bucket_of)
    gross = gross_for(panel, symbols, bucket_of)
    if annual_drag <= 0:
        return gross
    weights = target_weights([s for s in symbols if s in panel.columns],
                             bucket_of, "bucket")
    share = sum(weights[s] for s in singles if s in weights)
    daily = annual_drag / TRADING_DAYS
    return gross - share * daily


def run_wfo(panel: pd.DataFrame, bucket_of: dict[str, str],
            symbols: list[str]) -> dict:
    from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        warmup = panel.loc[panel.index < start].tail(VOL_LOOKBACK * 3)
        window = panel.loc[(panel.index >= start) & (panel.index < end)]
        if len(window) < 30:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        combined = pd.concat([warmup, window])
        gross = gross_for(combined, symbols, bucket_of)
        exp = exposure_path(gross, float(params["target_vol"]))
        net = apply_exposure(gross, exp)
        net = net.loc[net.index >= window.index[0]]
        if net.empty:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        changes = int((exp.reindex(net.index).diff().abs() > 1e-12).sum())
        return {
            "sharpe_ratio": _sharpe(net),
            "n_trades": int(len(net)),
            "exposure_changes": changes,
            "total_net_pnl": float(net.sum()),
            "profit_factor": _profit_factor(net),
            "max_drawdown": _max_dd(net),
            "daily_returns": [float(v) for v in net.tolist()],
        }

    cfg = WFOConfig(is_days=504, oos_days=126, step_days=126,
                    selection_objective="sharpe_subject_to_drawdown",
                    selection_max_drawdown=abs(MAX_DD_LIMIT))
    result = WalkForwardOptimizer(
        backtest_fn, cfg, [{"target_vol": tv} for tv in TARGET_VOLS]
    ).run(panel.index[0].to_pydatetime(), panel.index[-1].to_pydatetime())

    pooled = pd.Series([r for f in result.folds
                        for r in (f.oos_metrics.get("daily_returns") or [])])
    return {
        "decision": result.decision,
        "n_folds": len(result.folds),
        "positive_folds": result.positive_folds,
        "required_positive_folds": result.required_positive_folds,
        "oos_days": int(len(pooled)),
        "oos_sharpe": _sharpe(pooled) if len(pooled) else None,
        "oos_profit_factor": _profit_factor(pooled) if len(pooled) else None,
        "oos_max_drawdown": _max_dd(pooled) if len(pooled) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wfo", action="store_true",
                    help="also walk-forward both arms (slower)")
    args = ap.parse_args()

    singles, etfs, bucket_of = split_universe()
    symbols = list(bucket_of)
    panel = load_prices(symbols).loc[WINDOW_START:]

    weights = target_weights([s for s in symbols if s in panel.columns],
                             bucket_of, "bucket")
    single_share = sum(weights[s] for s in singles if s in weights)

    print(f"策略 V1 存活偏誤敞口   {panel.index[0].date()} -> "
          f"{panel.index[-1].date()}")
    print(f"  單一公司 {len(singles)} 檔，加權佔比 {single_share:.1%}"
          f"（真正可能倒掉的部分）")
    print(f"  ETF     {len(etfs)} 檔，加權佔比 {1 - single_share:.1%}"
          f"（清算還 NAV，不歸零）\n")

    payload: dict = {
        "run_at": datetime.now().isoformat(),
        "window": [str(panel.index[0].date()), str(panel.index[-1].date())],
        "n_single_names": len(singles),
        "n_etfs": len(etfs),
        "single_name_weight_share": single_share,
        "attrition_note": (
            "6 of 25 point-in-time 2016 S&P 500 names unavailable in 2026: "
            "ABC/ADS/ACE are renames or continuations, ADT/AET/AGN are premium "
            "acquisitions, zero are bankruptcies. The alphabetical sample "
            "misses the decline-then-delist mode that actually inflates a "
            "survivor-only backtest."),
        "arms": {}, "holdout": {}, "sleeves": {}, "breakeven": [],
    }

    arms = {"shipped_v1": symbols, "etf_only": etfs}
    for name, subset in arms.items():
        gross = gross_for(panel, subset, bucket_of)
        net, exp = braked(gross)
        payload["arms"][name] = _score(net, exp)
        hold = net.loc[net.index >= HOLDOUT_START]
        payload["holdout"][name] = _score(hold, exp) if len(hold) > 60 else None

    for name, subset in (("single_names_only", singles), ("etfs_only", etfs)):
        payload["sleeves"][name] = _score(sleeve_returns(panel, subset).dropna())

    header = {"shipped_v1": "出貨版 V1", "etf_only": "純 ETF"}
    print("  " + " " * 18 + "".join(f"{header[a]:>14}" for a in arms))
    for label, key in (("Sharpe", "sharpe_annualized"), ("CAGR", "cagr"),
                       ("實現波動", "realized_vol"),
                       ("最大回撤", "max_drawdown"),
                       ("獲利因子", "profit_factor"),
                       ("平均曝險", "mean_exposure")):
        cells = "".join(f"{payload['arms'][a][key]:>14.3f}" for a in arms)
        print(f"  {label:<18}{cells}")

    print(f"\n  兩個袖子各自的無煞車報酬（等權，未經桶配置）")
    for name, label in (("single_names_only", f"54 檔單一股票"),
                        ("etfs_only", f"66 檔 ETF")):
        s = payload["sleeves"][name]
        print(f"    {label:<16} Sharpe {s['sharpe_annualized']:>5.2f}  "
              f"CAGR {s['cagr']:>6.1%}  波動 {s['realized_vol']:>5.1%}  "
              f"回撤 {s['max_drawdown']:>6.1%}")

    print(f"\n  {HOLDOUT_START} 之後")
    for name in arms:
        h = payload["holdout"][name]
        if h:
            print(f"    {header[name]:<12} Sharpe {h['sharpe_annualized']:>5.2f}"
                  f"  CAGR {h['cagr']:>6.1%}  回撤 {h['max_drawdown']:>6.1%}")

    etf_sharpe = payload["arms"]["etf_only"]["sharpe_annualized"]
    print(f"\n  單一股票袖子的年化拖累要多大，出貨版才輸給純 ETF "
          f"(Sharpe {etf_sharpe:.3f})")
    crossed = None
    for drag in DRAGS:
        gross = shipped_gross_with_drag(panel, bucket_of, singles, drag)
        net, exp = braked(gross)
        row = _score(net, exp)
        row["sleeve_annual_drag"] = drag
        row["portfolio_annual_drag"] = drag * single_share
        payload["breakeven"].append(row)
        worse = row["sharpe_annualized"] < etf_sharpe
        if worse and crossed is None:
            crossed = drag
        print(f"    袖子拖累 {drag:>5.1%}（組合層面 {drag * single_share:>5.2%}）"
              f"  Sharpe {row['sharpe_annualized']:>5.3f}"
              f"  回撤 {row['max_drawdown']:>6.1%}"
              f"  {'<= 已輸給純 ETF' if worse else ''}")
    payload["breakeven_sleeve_drag"] = crossed

    # The drag above is a pure return haircut, so it barely touches drawdown
    # (-24.6% to -24.9% across the whole scan). Real failures arrive as
    # idiosyncratic crashes, which a smooth drag cannot represent. Bound that
    # separately: a single name is a fixed slice of the book, so total wipeouts
    # cost at most their combined weight, and the 54-way split inside a 16.3%
    # sleeve is what makes the ceiling low.
    per_name = single_share / len(singles)
    payload["tail_bound"] = {
        "weight_per_single_name": per_name,
        "cost_if_n_go_to_zero": {str(n): n * per_name for n in (1, 3, 5, 10)},
    }
    print(f"\n  尾端上界（拖累是平滑的，真實倒閉不是）")
    print(f"    每檔單一股票佔全書 {per_name:.2%}")
    for n in (1, 3, 5, 10):
        print(f"    同時 {n:>2} 檔完全歸零，最多損失全書 {n * per_name:.1%}")
    print(f"    54 檔攤在 16.3% 的袖子裡，這就是上界壓得低的原因")

    if args.wfo:
        print()
        for name, subset in arms.items():
            print(f"走進式 {name} ...", flush=True)
            payload.setdefault("wfo", {})[name] = run_wfo(panel, bucket_of,
                                                          subset)
        print("\n  走進式")
        for name in arms:
            w = payload["wfo"][name]
            print(f"    {header[name]:<12} {w['decision']:<6} "
                  f"{w['oos_days']} 日  Sharpe {w['oos_sharpe']:.2f}  "
                  f"PF {w['oos_profit_factor']:.2f}  "
                  f"回撤 {w['oos_max_drawdown']:.1%}  "
                  f"折 {w['positive_folds']}/{w['n_folds']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
