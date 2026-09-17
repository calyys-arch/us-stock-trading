#!/usr/bin/env python3
"""Measure the impact of VIX-based dynamic exposure adjustment on Strategy V1.

Compares baseline V1 (bucket-equal, volatility targeting alone) against V1 +
VIX overlay, over the full `data/history` cache.

Usage:
    .venv/bin/python scripts/run_v1_vix_study.py --save
    (needs network access to fetch ^VIX via yfinance: run the shell command
    with required_permissions ["network"] or ["full_network"])

THE HYPOTHESIS. Volatility targeting already scales exposure based on realized
vol, but it's backward-looking (uses trailing 60 days). VIX is forward-looking
(implied vol from options) and spikes before drawdowns. Adding a VIX overlay
should reduce drawdowns at some CAGR cost; net positive only if Sharpe holds
up or improves and drawdown genuinely shrinks.

THE TEST. Run V1's real, shipped pipeline -- bucket_weighted_gross(scheme=
"equal") feeding exposure_path(target_vol=0.15) -- exactly as
scripts/strategy_v1_positions.py and scripts/run_vol_target_study.py define
it. Then take that SAME base exposure and multiply by the VIX-bracket scalar
from python.portfolio.risk_controls.adjust_exposure_for_vix before charging
turnover costs on the adjusted series. Compare Sharpe, CAGR, max drawdown,
Calmar and MC p5 Sharpe.

THIRD ARM: constant_scale. The VIX overlay does two things at once: (1) it
times exposure down specifically when VIX is elevated, and (2) it simply runs
at a LOWER AVERAGE exposure than the baseline, because the overlay can only
scale down, never up. run_vol_target_study.py's own methodology note applies
here verbatim: "LESS EXPOSURE ALONE CUTS DRAWDOWN. Any study that reports
[a de-risking mechanism] against [a less de-risked] baseline is partly
measuring the leverage reduction and calling it risk management." So this
script also runs a FIXED exposure, set to the VIX arm's own realized mean
exposure, on the same gross series. If constant_scale matches vol_target+VIX,
the VIX brackets bought nothing beyond "hold a bit less" -- which is simpler
and has no extra turnover.

CORRECTNESS NOTE (2026-09-17). The previous version of this script had two
bugs that meant it had NEVER completed a real run:
  Bug A: called bucket_weighted_gross(..., scheme="bucket", ...). That is not
    a recognized scheme -- strategy_v1.bucket_weighted_gross only branches on
    scheme == "equal" (the live bucket_equal weighting); anything else,
    including the string "bucket", silently falls into the inverse-vol
    "bucket_rp" branch. So the old script was scoring a strategy V1 does not
    run.
  Bug B: tried to read exp_df["exposure"] from bucket_weighted_gross's second
    return value, which is a bucket-WEIGHT DataFrame (columns are bucket names
    like us_equity/bonds/...), not a total-book exposure fraction. That column
    never existed, so every invocation crashed with KeyError: 'exposure' before
    printing a single number.
  This version fixes both: scheme="equal" for the real bucket_equal gross
  return series, and a separately-computed exposure_path() (imported from
  scripts/run_vol_target_study.py, the same function scripts/strategy_v1_positions.py
  effectively mirrors) for the 0-1 book exposure fraction.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.portfolio.strategy_v1 import (
    load_universe, load_prices, bucket_weighted_gross
)
from python.portfolio.risk_controls import adjust_exposure_for_vix
from python.simulation.hist_data_us import fetch_daily_bars
from python.backtest.monte_carlo import MonteCarloValidator
from scripts.run_vol_target_study import (
    exposure_path, apply_exposure, VOL_LOOKBACK, REBALANCE_BAND,
    ONE_WAY_COST_BPS,
)

VOL_TARGET = 0.15
TRADING_DAYS = 252
MC_SIMS = 2000

OUTPUT_JSON = Path("backtests/v1_vix_study.json")


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def _max_dd(equity: pd.Series) -> float:
    running_max = equity.expanding().max()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    if years < 0.01:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def fetch_vix(start: str, end: str) -> pd.Series:
    """Fetch ^VIX history from yfinance. Needs network access."""
    print(f"  抓取 ^VIX 數據 ({start} 到 {end}) ...")
    vix_df = fetch_daily_bars("^VIX", start, end)
    vix = vix_df["close"].rename("VIX")
    print(f"    ^VIX: {len(vix)} 日，{vix.index[0].date()} 到 {vix.index[-1].date()}")
    return vix


def summarize(name: str, net: pd.Series, mean_exposure: float) -> dict:
    """Compute summary metrics for a strategy, including MC p5 Sharpe."""
    equity = (1 + net).cumprod()
    sharpe = _sharpe(net)
    cagr = _cagr(equity)
    dd = _max_dd(equity)
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    mc = MonteCarloValidator(n_sims=MC_SIMS, seed=42).run(
        [float(v) for v in net.tolist()])

    return {
        "name": name,
        "n_days": int(len(net)),
        "mean_exposure": float(mean_exposure),
        "sharpe_ratio": float(sharpe),
        "cagr": float(cagr),
        "max_drawdown": float(dd),
        "calmar_ratio": float(calmar),
        "realized_vol": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "mc_p5_sharpe": float(mc.sharpe.p5),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true",
                    help=f"Save results to {OUTPUT_JSON}")
    ap.add_argument("--sims", type=int, default=MC_SIMS)
    args = ap.parse_args()

    print("=== V1 + VIX 動態風控研究（全歷史，data/history 快取）===\n")

    print("載入 V1 universe ...")
    # NOTE: load_universe() returns the FINE-grained bucket map (e.g.
    # "us_single_name", "us_sector_etf", ...). bucket_weighted_gross() collapses
    # it internally via COLLAPSE (see collapsed_of() in strategy_v1.py) -- do
    # NOT pre-collapse it here. scripts/strategy_v1_positions.py,
    # scripts/paper_track_v1.py and scripts/run_risk_bucket_study.py all pass
    # the fine map straight through unchanged; pre-collapsing it (as an
    # earlier draft of this script did, mirroring a mistake in the task
    # instructions) makes collapsed_of() look up COLLAPSE["us_equity"], which
    # does not exist, so every symbol falls out and bucket_returns() returns
    # an empty frame (ZeroDivisionError downstream).
    symbols, bucket_of = load_universe()

    print("載入價格 ...")
    panel = load_prices(symbols)
    print(f"  {len(panel)} 日，{len(symbols)} 檔標的")
    print(f"  期間：{panel.index[0].date()} 到 {panel.index[-1].date()}")

    # === V1 的真實管線：bucket_equal gross, 15% 波動目標 ===
    print("\n計算 V1 實盤管線（scheme=equal，15% 波動目標）...")
    gross, _bucket_weights = bucket_weighted_gross(
        panel, bucket_of, "equal", VOL_LOOKBACK, REBALANCE_BAND, ONE_WAY_COST_BPS
    )
    base_exposure = exposure_path(gross, VOL_TARGET)
    net_baseline = apply_exposure(gross, base_exposure)
    mean_exp_baseline = float(base_exposure.reindex(net_baseline.index).mean())

    # Fetch VIX over the same span (plus a little margin) and align.
    start = (panel.index[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (panel.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    vix = fetch_vix(start, end)
    vix_aligned = vix.reindex(base_exposure.index, method="ffill")
    missing_vix = int(vix_aligned.isna().sum())
    if missing_vix:
        print(f"  警告：{missing_vix} 天沒有可用的 ^VIX 值（ffill 後仍缺）")

    # === V1 + VIX overlay：同一組 base_exposure 再乘 VIX 分級 scalar ===
    print("計算 V1 + VIX 疊加 ...")
    adjusted_exposure = adjust_exposure_for_vix(base_exposure, vix_aligned)
    net_with_vix = apply_exposure(gross, adjusted_exposure)
    mean_exp_vix = float(adjusted_exposure.reindex(net_with_vix.index).mean())

    # === constant_scale control: SAME mean exposure as the VIX arm, but
    # fixed all the time (no timing at all) ===
    flat = pd.Series(mean_exp_vix, index=gross.index)
    net_constant = apply_exposure(gross, flat)

    baseline = summarize("V1 基準（bucket_equal + 15% 波動目標，無 VIX）",
                         net_baseline, mean_exp_baseline)
    with_vix = summarize("V1 + VIX 疊加", net_with_vix, mean_exp_vix)
    constant_scale = summarize("固定曝險（= VIX 疊加的平均曝險，無擇時）",
                               net_constant, mean_exp_vix)

    # Compare
    print("\n" + "=" * 92)
    print(f"{'指標':<20} {'基準':>14} {'+ VIX':>14} {'固定同曝險':>14} {'VIX-基準':>14}")
    print("=" * 92)

    metrics = [
        ("Sharpe ratio", "sharpe_ratio", False),
        ("CAGR", "cagr", True),
        ("最大回撤", "max_drawdown", True),
        ("Calmar ratio", "calmar_ratio", False),
        ("實現波動", "realized_vol", True),
        ("平均曝險", "mean_exposure", True),
        ("MC p5 Sharpe", "mc_p5_sharpe", False),
    ]

    for label, key, is_pct in metrics:
        base_val = baseline[key]
        vix_val = with_vix[key]
        const_val = constant_scale[key]
        delta = vix_val - base_val

        if is_pct:
            print(f"{label:<20} {base_val:>13.1%} {vix_val:>13.1%} {const_val:>13.1%} {delta:>+13.1%}")
        else:
            print(f"{label:<20} {base_val:>14.2f} {vix_val:>14.2f} {const_val:>14.2f} {delta:>+14.2f}")

    print("=" * 92)

    sharpe_improved = with_vix["sharpe_ratio"] > baseline["sharpe_ratio"]
    dd_improved = with_vix["max_drawdown"] > baseline["max_drawdown"]  # less negative is better
    beats_constant_dd = with_vix["max_drawdown"] > constant_scale["max_drawdown"]
    beats_constant_sharpe = with_vix["sharpe_ratio"] > constant_scale["sharpe_ratio"]

    print("\n結論（僅描述性比較，非決策依據，見報告）：")
    if sharpe_improved and dd_improved:
        print("  VIX 疊加同時改善了 Sharpe 和回撤（相對基準）")
    elif sharpe_improved:
        print("  VIX 疊加改善了 Sharpe 但回撤更差（相對基準）")
    elif dd_improved:
        print("  VIX 疊加改善了回撤但 Sharpe 更差（相對基準）")
    else:
        print("  VIX 疊加兩者都沒有改善（相對基準）")
    if beats_constant_dd and beats_constant_sharpe:
        print("  且同時贏過「同平均曝險但不擇時」的固定版本 → 擇時本身有貢獻")
    else:
        print("  但沒有同時贏過「同平均曝險但不擇時」的固定版本 → 部分或全部改善"
              "可能只是曝險降低的功勞，不是 VIX 擇時的功勞")

    if args.save:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "period": [str(panel.index[0].date()), str(panel.index[-1].date())],
            "n_symbols": len(symbols),
            "vix_missing_days_after_ffill": missing_vix,
            "baseline": baseline,
            "with_vix": with_vix,
            "constant_scale_control": constant_scale,
            "delta_vix_vs_baseline": {
                "sharpe_delta": with_vix["sharpe_ratio"] - baseline["sharpe_ratio"],
                "dd_delta": with_vix["max_drawdown"] - baseline["max_drawdown"],
                "cagr_delta": with_vix["cagr"] - baseline["cagr"],
                "mc_p5_sharpe_delta": with_vix["mc_p5_sharpe"] - baseline["mc_p5_sharpe"],
            },
            "delta_vix_vs_constant_scale": {
                "sharpe_delta": with_vix["sharpe_ratio"] - constant_scale["sharpe_ratio"],
                "dd_delta": with_vix["max_drawdown"] - constant_scale["max_drawdown"],
                "mc_p5_sharpe_delta": with_vix["mc_p5_sharpe"] - constant_scale["mc_p5_sharpe"],
            },
            "config": {
                "vol_target": VOL_TARGET,
                "vol_lookback": VOL_LOOKBACK,
                "rebalance_band": REBALANCE_BAND,
                "cost_bps": ONE_WAY_COST_BPS,
                "scheme": "equal",
                "mc_sims": args.sims,
            }
        }
        OUTPUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n結果已儲存：{OUTPUT_JSON}")


if __name__ == "__main__":
    main()
