#!/usr/bin/env python3
"""Strategy V2: Cross-sectional Momentum (12-1).

Ranks stocks by their past 12-month return (excluding the most recent month to
avoid short-term reversal) and goes long the top quintile, optionally shorts
the bottom quintile.

Usage:
    # Long-only version (recommended for retail)
    .venv/bin/python scripts/run_v2_momentum.py --mode long-only --save

    # Long-short market-neutral version
    .venv/bin/python scripts/run_v2_momentum.py --mode long-short --save

    # Compare correlation with V1
    .venv/bin/python scripts/run_v2_momentum.py --compare-v1

LOGIC. On the first trading day of each month:
  1. Calculate each stock's return over days [-252, -21] (12 months, skip last month)
  2. Rank all stocks by this return
  3. Long-only: equal-weight the top 20%
  4. Long-short: equal-weight top 20% (long) and bottom 20% (short), 50% each side

WHY SKIP THE LAST MONTH? Short-term (1-month) returns tend to reverse, while
medium-term (12-month) returns continue. The 12-1 formation period captures
momentum while avoiding the reversal effect.

COST MODEL. Monthly rebalancing with realistic costs:
  - Commission: IBKR Pro Fixed ($1 minimum per order)
  - Spread: 4bps one-way
  - Turnover: ~40% per month (half the portfolio changes)

At 500 stocks × 40% turnover × 12 months = 2,400 orders/year
→ ~$2,600/year on $100k book (2.6%)

This is HIGH. V2 needs gross Sharpe > 1.0 to survive these costs.

UNIVERSE. Uses all cached symbols in data/history/ with sufficient data
(at least 252 + 21 = 273 trading days = ~15 months). Currently ~589 symbols.

BENCHMARK. Academic momentum portfolios typically achieve Sharpe ~0.5-0.8 on
large-cap US stocks. We're aiming for Sharpe > 0.5 post-cost on long-only,
or Sharpe > 0.7 on long-short (which has higher turnover but no market beta).
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.portfolio.broker_costs import load_schedule

# Cache directory
HISTORY = Path("data/history")


# Constants
FORMATION_DAYS = 252  # 12 months
SKIP_DAYS = 21        # Skip last month to avoid reversal
MIN_HISTORY = FORMATION_DAYS + SKIP_DAYS + 21  # Need extra buffer
REBALANCE_FREQ = "ME"  # Monthly (month-end)
TOP_PERCENTILE = 0.80  # Top 20% (above 80th percentile)
BOTTOM_PERCENTILE = 0.20  # Bottom 20% (below 20th percentile)
ONE_WAY_COST_BPS = 4.0

OUTPUT_JSON = Path("backtests/v2_momentum_study.json")


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


def load_universe(min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """Load all cached price data with sufficient history.
    
    Returns:
        DataFrame with daily prices, columns = symbols, index = dates
    """
    print("載入 universe ...")
    
    # Read all CSV files from cache
    csv_files = sorted(HISTORY.glob("*.csv"))
    
    cols = {}
    for csv_file in csv_files:
        symbol = csv_file.stem
        try:
            df = pd.read_csv(csv_file, parse_dates=["date"]).set_index("date")
            if "close" in df.columns and len(df) >= min_history:
                cols[symbol] = df["close"].astype(float)
        except Exception:
            continue  # Skip broken files
    
    panel = pd.DataFrame(cols).sort_index()
    
    print(f"  {len(panel)} 日 × {len(panel.columns)} 檔")
    print(f"  期間：{panel.index[0].date()} 到 {panel.index[-1].date()}")
    
    return panel


def calculate_momentum_scores(panel: pd.DataFrame, as_of: pd.Timestamp,
                              formation_days: int = FORMATION_DAYS,
                              skip_days: int = SKIP_DAYS) -> pd.Series:
    """Calculate momentum scores (12-month return, skip last month).
    
    Args:
        panel: Price panel (dates × symbols)
        as_of: Date to calculate scores for
        formation_days: Lookback period (default 252 = 12 months)
        skip_days: Skip recent days to avoid reversal (default 21 = 1 month)
    
    Returns:
        Series of momentum scores (symbol → score), NaN for insufficient data
    """
    # Get the slice from as_of going back
    upto = panel.loc[:as_of]
    
    if len(upto) < formation_days + skip_days:
        return pd.Series(dtype=float)
    
    # Price at end of formation period (skip_days ago)
    p_recent = upto.iloc[-(skip_days + 1)]
    
    # Price at start of formation period (formation_days + skip_days ago)
    p_old = upto.iloc[-(formation_days + skip_days + 1)]
    
    # Return over the formation period
    returns = (p_recent / p_old - 1.0).dropna()
    
    return returns


def generate_signals(panel: pd.DataFrame, mode: str = "long-only",
                    rebalance_freq: str = "ME") -> pd.DataFrame:
    """Generate monthly momentum signals.
    
    Args:
        panel: Price panel
        mode: "long-only" or "long-short"
        rebalance_freq: "M" for monthly
    
    Returns:
        DataFrame of target weights (dates × symbols), rebalanced monthly
    """
    print(f"\n生成 {mode} momentum 訊號（{rebalance_freq} 再平衡）...")
    
    # Get month-end dates
    month_ends = panel.resample(rebalance_freq).last().index
    
    weights_records = []
    
    # Debug: track holdings over time
    n_holdings_history = []
    
    for i, rebal_date in enumerate(month_ends, 1):
        if i % 12 == 0:
            print(f"  處理 {rebal_date.date()} ({i}/{len(month_ends)})")
        
        # Calculate momentum scores
        scores = calculate_momentum_scores(panel, rebal_date)
        
        if len(scores) < 50:  # Need minimum universe size
            if i <= 3:  # Debug early months
                print(f"    {rebal_date.date()}: 跳過，scores 太少 ({len(scores)})")
            continue
        
        # Rank stocks
        ranks = scores.rank(pct=True)
        
        if mode == "long-only":
            # Long top 20%
            longs = ranks[ranks >= TOP_PERCENTILE].index
            n_long = len(longs)
            
            if n_long == 0:
                if i <= 3:
                    print(f"    {rebal_date.date()}: 跳過，沒有 longs")
                continue
            
            # Debug first few months
            if i <= 3:
                print(f"    {rebal_date.date()}: {n_long} 檔持倉（{len(scores)} 檔有 scores）")
                print(f"      Top 5: {scores.nlargest(5).to_dict()}")
            
            n_holdings_history.append(n_long)
            
            weights = pd.Series(0.0, index=panel.columns)
            weights[longs] = 1.0 / n_long
            
        elif mode == "long-short":
            # Long top 20%, short bottom 20%
            longs = ranks[ranks >= TOP_PERCENTILE].index
            shorts = ranks[ranks <= BOTTOM_PERCENTILE].index
            
            n_long = len(longs)
            n_short = len(shorts)
            
            if n_long == 0 or n_short == 0:
                continue
            
            weights = pd.Series(0.0, index=panel.columns)
            weights[longs] = 0.5 / n_long   # 50% long
            weights[shorts] = -0.5 / n_short  # 50% short
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        weights_records.append({
            'date': rebal_date,
            'weights': weights
        })
    
    # Build weights DataFrame
    if not weights_records:
        raise RuntimeError("No valid rebalance dates generated")
    
    dates = [r['date'] for r in weights_records]
    weights_df = pd.DataFrame(
        [r['weights'].values for r in weights_records],
        index=dates,
        columns=panel.columns
    )
    
    # Forward-fill weights to daily (hold until next rebalance)
    all_dates = panel.index
    weights_daily = weights_df.reindex(all_dates, method='ffill').fillna(0.0)
    
    print(f"  生成 {len(weights_df)} 次再平衡訊號")
    
    # Better diagnostic: count non-zero weights
    n_holdings_per_day = (weights_daily.abs() > 1e-6).sum(axis=1)
    print(f"  持倉統計：")
    print(f"    平均：{n_holdings_per_day.mean():.1f} 檔")
    print(f"    中位數：{n_holdings_per_day.median():.0f} 檔")
    print(f"    最少：{n_holdings_per_day.min():.0f} 檔")
    print(f"    最多：{n_holdings_per_day.max():.0f} 檔")
    
    if n_holdings_history:
        print(f"  再平衡時的持倉數：平均 {np.mean(n_holdings_history):.1f} 檔")
    
    return weights_daily


def backtest(panel: pd.DataFrame, weights: pd.DataFrame,
            charge_costs: bool = True) -> tuple[pd.Series, dict]:
    """Run backtest with given weights.
    
    Args:
        panel: Price panel (dates × symbols)
        weights: Target weights (dates × symbols)
        charge_costs: Whether to charge trading costs
    
    Returns:
        (net_returns, metrics_dict)
    """
    print("\n運行回測 ...")
    
    # Calculate returns
    returns = panel.pct_change()
    
    # Align weights and returns
    common_dates = weights.index.intersection(returns.index)
    weights = weights.loc[common_dates]
    returns = returns.loc[common_dates]
    
    # Calculate gross returns
    gross_returns = (weights.shift(1) * returns).sum(axis=1)
    
    # Calculate turnover
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # Initial position
    
    if charge_costs:
        schedule = load_schedule()
        
        # Count number of orders per rebalance
        # (non-zero weight changes)
        n_orders = (weights.diff().abs() > 1e-6).sum(axis=1)
        n_orders.iloc[0] = (weights.iloc[0].abs() > 1e-6).sum()
        
        # Estimate commission (assume avg order size for cost calculation)
        # This is approximate - real cost depends on actual $ amounts
        commission_per_rebal = n_orders * 1.00  # $1 minimum dominates
        
        # Spread cost
        spread_cost = turnover * (ONE_WAY_COST_BPS / 10_000.0)
        
        # Total cost as fraction of capital (assume $100k capital)
        capital = 100_000.0
        total_cost = (commission_per_rebal / capital) + spread_cost
        
        net_returns = gross_returns - total_cost
    else:
        net_returns = gross_returns
    
    net_returns = net_returns.dropna()
    
    # Calculate metrics
    equity = (1 + net_returns).cumprod()
    
    metrics = {
        "n_days": len(net_returns),
        "sharpe_ratio": _sharpe(net_returns),
        "cagr": _cagr(equity),
        "max_drawdown": _max_dd(equity),
        "realized_vol": net_returns.std() * np.sqrt(252),
        "avg_turnover": turnover.mean(),
        "total_return": equity.iloc[-1] - 1.0,
        "n_rebalances": (turnover > 0.01).sum(),
    }
    
    print(f"  {metrics['n_days']} 日")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  CAGR: {metrics['cagr']:.1%}")
    print(f"  回撤: {metrics['max_drawdown']:.1%}")
    print(f"  再平衡: {metrics['n_rebalances']} 次")
    
    return net_returns, metrics


def compare_with_v1(v2_returns: pd.Series, v1_returns_path: Path) -> dict:
    """Compare V2 with V1 returns (if available).
    
    Returns:
        {
            'correlation': float,
            'v1_sharpe': float,
            'v2_sharpe': float,
            'combined_sharpe': float (50/50 blend),
        }
    """
    if not v1_returns_path.exists():
        print(f"\n⚠️  V1 returns not found at {v1_returns_path}")
        return {}
    
    print(f"\n對比 V1 與 V2 ...")
    
    # Load V1 returns (would need to be saved by run_risk_bucket_study.py)
    # For now, return placeholder
    print("  (需要先保存 V1 returns 才能對比)")
    
    return {}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["long-only", "long-short"],
                    default="long-only",
                    help="Strategy mode (default: long-only)")
    ap.add_argument("--save", action="store_true",
                    help=f"Save results to {OUTPUT_JSON}")
    ap.add_argument("--compare-v1", action="store_true",
                    help="Compare correlation with V1")
    ap.add_argument("--no-costs", action="store_true",
                    help="Don't charge trading costs (for gross return)")
    args = ap.parse_args()
    
    print("=" * 70)
    print(f"策略 V2：Cross-sectional Momentum ({args.mode})")
    print("=" * 70)
    print()
    
    # Load universe
    panel = load_universe()
    
    # Generate signals
    weights = generate_signals(panel, mode=args.mode)
    
    # Backtest
    net_returns, metrics = backtest(panel, weights, 
                                   charge_costs=not args.no_costs)
    
    # Compare with V1
    if args.compare_v1:
        v1_comparison = compare_with_v1(
            net_returns,
            Path("backtests/v1_returns.csv")
        )
        metrics['v1_comparison'] = v1_comparison
    
    # Summary
    print("\n" + "=" * 70)
    print("結果總結")
    print("=" * 70)
    print(f"模式：{args.mode}")
    print(f"Sharpe ratio：{metrics['sharpe_ratio']:.2f}")
    print(f"CAGR：{metrics['cagr']:.1%}")
    print(f"最大回撤：{metrics['max_drawdown']:.1%}")
    print(f"實現波動：{metrics['realized_vol']:.1%}")
    print(f"平均月換手：{metrics['avg_turnover'] * 100:.0f}%")
    
    # Verdict
    print("\n判斷：")
    if metrics['sharpe_ratio'] > 0.7 if args.mode == "long-short" else 0.5:
        if metrics['max_drawdown'] > -0.30:
            print("  ✅ 通過：Sharpe 和回撤都在可接受範圍")
            print(f"     下一步：與 V1 組合測試，看能否提升整體 Sharpe")
        else:
            print("  ⚠️  Sharpe 足夠但回撤過大，需要風控")
    else:
        print(f"  ❌ Sharpe 不足（目標 > {0.7 if args.mode == 'long-short' else 0.5}）")
        print("     可能原因：成本太高、universe 太小、或 momentum 效應不存在")
    
    # Save
    if args.save:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "mode": args.mode,
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()},
            "config": {
                "formation_days": FORMATION_DAYS,
                "skip_days": SKIP_DAYS,
                "rebalance_freq": REBALANCE_FREQ,
                "cost_bps": ONE_WAY_COST_BPS,
                "universe_size": len(panel.columns),
            },
            "run_at": datetime.now().isoformat(),
        }
        OUTPUT_JSON.write_text(json.dumps(result, indent=2))
        print(f"\n結果已儲存：{OUTPUT_JSON}")


if __name__ == "__main__":
    main()
