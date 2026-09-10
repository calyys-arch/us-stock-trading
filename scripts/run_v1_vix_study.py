#!/usr/bin/env python3
"""Measure the impact of VIX-based dynamic exposure adjustment on Strategy V1.

Compares baseline V1 (volatility targeting alone) against V1 + VIX overlay.

Usage:
    .venv/bin/python scripts/run_v1_vix_study.py --save

THE HYPOTHESIS. Volatility targeting already scales exposure based on realized vol,
but it's backward-looking (uses past 60 days). VIX is forward-looking (implied vol
from options) and spikes before drawdowns. Adding a VIX overlay should:
  - Reduce drawdowns (by de-risking early in stress periods)
  - Slightly reduce CAGR (by missing some recovery vol)
  - Net positive if Sharpe improves and drawdown shrinks

THE TEST. Run V1's bucket-equal volatility targeting logic over the full history,
then re-run it with VIX-based exposure scaling. Compare:
  - Sharpe ratio (risk-adjusted return)
  - Max drawdown (tail risk)
  - CAGR (absolute return)
  - Calmar ratio (CAGR / abs(max_dd))

BENCHMARK. The baseline is the shipped V1: bucket-equal, 15% vol target, 10%
rebalance band, no VIX overlay. Backtest Sharpe = 0.82 over 2008-2026.

VIX DATA. We'll fetch ^VIX from yfinance. It's available from 1990-01-02 onward.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Use the same infra as run_risk_bucket_study.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.portfolio.strategy_v1 import (
    load_universe, load_prices, bucket_weighted_gross, COLLAPSE
)
from python.portfolio.risk_controls import adjust_exposure_for_vix
from python.simulation.hist_data_us import fetch_daily_bars


REBALANCE_BAND = 0.10
VOL_TARGET = 0.15
VOL_LOOKBACK = 60
ONE_WAY_COST_BPS = 4.0

OUTPUT_JSON = Path("backtests/v1_vix_study.json")


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)


def _max_dd(equity: pd.Series) -> float:
    running_max = equity.expanding().max()
    dd = (equity - running_max) / running_max
    return dd.min()


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / 252.0
    if years < 0.01:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def fetch_vix() -> pd.Series:
    """Fetch ^VIX history from yfinance."""
    print("  抓取 ^VIX 數據 ...")
    vix_df = fetch_daily_bars("^VIX", "1990-01-01", "2026-12-31")
    vix = vix_df["close"].rename("VIX")
    print(f"    ^VIX: {len(vix)} 日，{vix.index[0].date()} 到 {vix.index[-1].date()}")
    return vix


def run_v1_baseline(panel: pd.DataFrame, bucket_of: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Run baseline V1: bucket-equal, vol-targeting, no VIX overlay."""
    print("\n運行 V1 基準版本（無 VIX 調整）...")
    gross, exp_df = bucket_weighted_gross(
        panel, bucket_of, scheme="bucket", vol_lookback=VOL_LOOKBACK,
        band=REBALANCE_BAND, one_way_cost_bps=ONE_WAY_COST_BPS
    )
    exposure = exp_df["exposure"]
    
    # Apply exposure
    exp_aligned = exposure.reindex(gross.index)
    valid = exp_aligned.notna()
    gross_valid = gross[valid]
    exp_valid = exp_aligned[valid]
    
    turnover = exp_valid.diff().abs()
    if len(turnover):
        turnover.iloc[0] = exp_valid.iloc[0]
    turnover = turnover.fillna(0.0)
    
    cost = turnover * (ONE_WAY_COST_BPS / 10_000.0)
    net = gross_valid * exp_valid - cost
    
    return net, exposure


def run_v1_with_vix(panel: pd.DataFrame, bucket_of: dict, vix: pd.Series
                    ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run V1 with VIX overlay: vol-targeting base, then scale by VIX."""
    print("\n運行 V1 + VIX 疊加版本 ...")
    
    # Get base exposure from vol targeting
    gross, exp_df = bucket_weighted_gross(
        panel, bucket_of, scheme="bucket", vol_lookback=VOL_LOOKBACK,
        band=REBALANCE_BAND, one_way_cost_bps=ONE_WAY_COST_BPS
    )
    base_exposure = exp_df["exposure"]
    
    # Align VIX to the same dates
    vix_aligned = vix.reindex(base_exposure.index, method="ffill")
    
    # Apply VIX adjustment
    adjusted_exposure = adjust_exposure_for_vix(base_exposure, vix_aligned)
    
    # Calculate net returns
    exp_aligned = adjusted_exposure.reindex(gross.index)
    valid = exp_aligned.notna()
    gross_valid = gross[valid]
    exp_valid = exp_aligned[valid]
    
    turnover = exp_valid.diff().abs()
    if len(turnover):
        turnover.iloc[0] = exp_valid.iloc[0]
    turnover = turnover.fillna(0.0)
    
    cost = turnover * (ONE_WAY_COST_BPS / 10_000.0)
    net = gross_valid * exp_valid - cost
    
    return net, base_exposure, adjusted_exposure


def summarize(name: str, net: pd.Series) -> dict:
    """Compute summary metrics for a strategy."""
    equity = (1 + net).cumprod()
    sharpe = _sharpe(net)
    cagr = _cagr(equity)
    dd = _max_dd(equity)
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    
    return {
        "name": name,
        "n_days": len(net),
        "sharpe_ratio": float(sharpe),
        "cagr": float(cagr),
        "max_drawdown": float(dd),
        "calmar_ratio": float(calmar),
        "realized_vol": float(net.std() * np.sqrt(252)),
        "total_return": float(equity.iloc[-1] - 1.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true",
                    help=f"Save results to {OUTPUT_JSON}")
    args = ap.parse_args()
    
    print("=== V1 + VIX 動態風控研究 ===\n")
    
    # Load universe and prices
    print("載入 V1 universe ...")
    symbols, bucket_of = load_universe()
    bucket_of = {sym: COLLAPSE[bucket_of[sym]] for sym in symbols}
    
    print("載入價格 ...")
    panel = load_prices(symbols)
    print(f"  {len(panel)} 日，{len(symbols)} 檔標的")
    print(f"  期間：{panel.index[0].date()} 到 {panel.index[-1].date()}")
    
    # Fetch VIX
    vix = fetch_vix()
    
    # Run baseline
    net_baseline, exp_baseline = run_v1_baseline(panel, bucket_of)
    baseline = summarize("V1 基準（無 VIX）", net_baseline)
    
    # Run with VIX
    net_vix, exp_base, exp_vix = run_v1_with_vix(panel, bucket_of, vix)
    with_vix = summarize("V1 + VIX 疊加", net_vix)
    
    # Compare
    print("\n" + "=" * 70)
    print(f"{'指標':<20} {'基準':>12} {'+ VIX':>12} {'變化':>12}")
    print("=" * 70)
    
    metrics = [
        ("Sharpe ratio", "sharpe_ratio", False),
        ("CAGR", "cagr", True),
        ("最大回撤", "max_drawdown", True),
        ("Calmar ratio", "calmar_ratio", False),
        ("實現波動", "realized_vol", True),
    ]
    
    for label, key, is_pct in metrics:
        base_val = baseline[key]
        vix_val = with_vix[key]
        delta = vix_val - base_val
        
        if is_pct:
            print(f"{label:<20} {base_val:>11.1%} {vix_val:>11.1%} {delta:>+11.1%}")
        else:
            print(f"{label:<20} {base_val:>12.2f} {vix_val:>12.2f} {delta:>+12.2f}")
    
    print("=" * 70)
    
    # Interpretation
    sharpe_improved = with_vix["sharpe_ratio"] > baseline["sharpe_ratio"]
    dd_improved = with_vix["max_drawdown"] > baseline["max_drawdown"]  # Less negative is better
    
    print("\n結論：")
    if sharpe_improved and dd_improved:
        print("  ✅ VIX 疊加改善了 Sharpe 和回撤，建議採用")
    elif sharpe_improved:
        print("  ⚠️  VIX 疊加改善了 Sharpe 但回撤更差，需權衡")
    elif dd_improved:
        print("  ⚠️  VIX 疊加改善了回撤但 Sharpe 更差，需權衡")
    else:
        print("  ❌ VIX 疊加沒有改善，不建議採用")
    
    # Save
    if args.save:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "baseline": baseline,
            "with_vix": with_vix,
            "improvement": {
                "sharpe_delta": with_vix["sharpe_ratio"] - baseline["sharpe_ratio"],
                "dd_delta": with_vix["max_drawdown"] - baseline["max_drawdown"],
                "cagr_delta": with_vix["cagr"] - baseline["cagr"],
            },
            "config": {
                "vol_target": VOL_TARGET,
                "vol_lookback": VOL_LOOKBACK,
                "rebalance_band": REBALANCE_BAND,
                "cost_bps": ONE_WAY_COST_BPS,
            }
        }
        OUTPUT_JSON.write_text(json.dumps(result, indent=2))
        print(f"\n結果已儲存：{OUTPUT_JSON}")


if __name__ == "__main__":
    main()
