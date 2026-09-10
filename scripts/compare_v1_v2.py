#!/usr/bin/env python3
"""Compare V1 and V2 strategies and test portfolio combinations.

Usage:
    .venv/bin/python scripts/compare_v1_v2.py
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.portfolio.strategy_v1 import (
    load_universe, load_prices, bucket_weighted_gross, COLLAPSE
)


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


def run_v1() -> pd.Series:
    """Run V1 strategy and return daily returns."""
    print("運行 V1（bucket-equal volatility targeting）...")
    
    symbols, bucket_of = load_universe()
    bucket_of = {sym: COLLAPSE[bucket_of[sym]] for sym in symbols}
    panel = load_prices(symbols)
    
    gross, exp_df = bucket_weighted_gross(
        panel, bucket_of, scheme="bucket", vol_lookback=60,
        band=0.10, one_way_cost_bps=4.0
    )
    
    print(f"  {len(gross)} 日")
    return gross


def run_v2() -> pd.Series:
    """Run V2 strategy and return daily returns."""
    print("運行 V2（cross-sectional momentum）...")
    
    import scripts.run_v2_momentum as v2
    
    panel = v2.load_universe()
    weights = v2.generate_signals(panel, mode="long-only")
    returns, metrics = v2.backtest(panel, weights, charge_costs=True)
    
    print(f"  {len(returns)} 日")
    return returns


def compare(v1_returns: pd.Series, v2_returns: pd.Series):
    """Compare V1, V2, and their combinations."""
    
    # Align to common dates
    common_dates = v1_returns.index.intersection(v2_returns.index)
    v1 = v1_returns.loc[common_dates]
    v2 = v2_returns.loc[common_dates]
    
    print(f"\n共同期間：{len(common_dates)} 日")
    print(f"  {common_dates[0].date()} 到 {common_dates[-1].date()}")
    
    # Calculate combined portfolios
    combo_50_50 = 0.5 * v1 + 0.5 * v2
    combo_70_30 = 0.7 * v1 + 0.3 * v2  # Conservative: more V1
    combo_30_70 = 0.3 * v1 + 0.7 * v2  # Aggressive: more V2
    
    # Calculate metrics for each
    strategies = {
        "V1 (100%)": v1,
        "V2 (100%)": v2,
        "V1 70% + V2 30%": combo_70_30,
        "V1 50% + V2 50%": combo_50_50,
        "V1 30% + V2 70%": combo_30_70,
    }
    
    results = []
    for name, rets in strategies.items():
        equity = (1 + rets).cumprod()
        results.append({
            "策略": name,
            "Sharpe": _sharpe(rets),
            "CAGR": _cagr(equity),
            "回撤": _max_dd(equity),
            "波動": rets.std() * np.sqrt(252),
        })
    
    df = pd.DataFrame(results)
    
    # Calculate correlation
    corr = v1.corr(v2)
    
    print("\n" + "=" * 70)
    print("策略對比")
    print("=" * 70)
    print(f"\nV1 與 V2 相關性：{corr:.2f}")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}" 
                       if abs(x) < 10 else f"{x:.1%}"))
    
    # Find best combo
    best = df.loc[df["Sharpe"].idxmax()]
    print(f"\n最佳 Sharpe：{best['策略']} (Sharpe {best['Sharpe']:.2f})")
    
    # Improvement over V1
    v1_sharpe = df[df["策略"] == "V1 (100%)"]["Sharpe"].values[0]
    best_sharpe = best["Sharpe"]
    improvement = (best_sharpe - v1_sharpe) / v1_sharpe * 100
    
    print(f"相比 V1 單獨：Sharpe 提升 {improvement:+.1f}%")
    
    # Check if drawdown improved
    v1_dd = df[df["策略"] == "V1 (100%)"]["回撤"].values[0]
    best_dd = best["回撤"]
    dd_improvement = (best_dd - v1_dd) * 100  # In percentage points
    
    if best_dd > v1_dd:
        print(f"              回撤改善 {dd_improvement:+.1f}pp")
    else:
        print(f"              回撤惡化 {dd_improvement:.1f}pp")


def main():
    print("=" * 70)
    print("V1 vs V2 策略對比")
    print("=" * 70)
    print()
    
    # Run both strategies
    v1_returns = run_v1()
    v2_returns = run_v2()
    
    # Compare
    compare(v1_returns, v2_returns)
    
    print("\n" + "=" * 70)
    print("結論：")
    print("=" * 70)
    print("""
如果相關性 < 0.5 且組合 Sharpe > V1：
  → V2 值得開發（提供 diversification benefit）
  → 下一步：V2 紙上交易 3-6 個月

如果相關性 > 0.7 或組合沒改善：
  → V2 不值得（太像 V1 或沒有加分）
  → 專注優化 V1 即可
    """)


if __name__ == "__main__":
    main()
