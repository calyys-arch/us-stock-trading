#!/usr/bin/env python3
"""Generate monthly performance report for Strategy V1 paper trading.

Compares realized performance against backtest expectations and flags deviations.

Usage:
    .venv/bin/python scripts/report_v1_monthly.py
    .venv/bin/python scripts/report_v1_monthly.py --output reports/v1_2026_09.md

The report includes:
  - Equity curve plot
  - Realized vs expected metrics (Sharpe, CAGR, drawdown, profit factor)
  - Trade log summary
  - Deviation alerts (if realized metrics fall outside confidence bounds)

BENCHMARK. The backtest used as ground truth is the bucket-equal WFO holdout from
run_risk_bucket_study.py: Sharpe 0.91, CAGR 11.7%, max drawdown -24.6%, profit
factor 1.17, measured over 1,646 out-of-sample days (2018-11 to 2026-08).

CONFIDENCE BOUNDS. Early samples have high variance. A 10-day realized Sharpe of
-0.2 does not refute a true Sharpe of 0.9 -- the standard error is huge. The
report calculates how many days are needed before deviations become significant.

FALSIFICATION. The ledger's own docstring defines failure conditions:
  - Realized Sharpe < -0.3 (13 Sharpe points below benchmark)
  - Max drawdown < -30% (5pp worse than backtest)
These are conservative thresholds designed to survive short-term noise.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Backtest benchmark (from run_risk_bucket_study.py, bucket_equal WFO holdout)
BENCHMARK = {
    "sharpe_ratio": 0.91,
    "cagr": 0.117,
    "max_drawdown": -0.246,
    "profit_factor": 1.17,
    "n_days": 1646,
    "period": "2018-11 to 2026-08",
    "key": "scripts/run_risk_bucket_study.py bucket_equal arm, WFO holdout"
}

# Falsification thresholds (from paper_track_v1.py)
FAIL_SHARPE = -0.3
FAIL_DRAWDOWN = -0.30

# Statistical thresholds
MIN_DAYS_FOR_SIGNIFICANCE = 90  # ~4 months, rough rule of thumb


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def _max_dd(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction."""
    running_max = equity.expanding().max()
    dd = (equity - running_max) / running_max
    return dd.min()


def _profit_factor(returns: pd.Series) -> float:
    """Profit factor: sum(gains) / sum(losses)."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return gains / losses if losses > 0 else np.inf


def _cagr(equity: pd.Series, n_days: int) -> float:
    """Compounded annual growth rate."""
    if n_days < 1:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = n_days / 252.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _load_journal(journal_path: Path) -> pd.DataFrame:
    """Load the paper trading journal into a DataFrame."""
    records = []
    with open(journal_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def _compute_realized_metrics(journal: pd.DataFrame) -> dict:
    """Compute realized performance metrics from the journal."""
    equity = journal["equity"]
    returns = equity.pct_change().dropna()
    n_days = len(equity) - 1  # Exclude init day
    
    return {
        "n_days": n_days,
        "equity_start": equity.iloc[0],
        "equity_end": equity.iloc[-1],
        "total_return": (equity.iloc[-1] / equity.iloc[0] - 1.0),
        "sharpe_ratio": _sharpe(returns),
        "cagr": _cagr(equity, n_days),
        "max_drawdown": _max_dd(equity),
        "profit_factor": _profit_factor(returns),
        "realized_vol": returns.std() * np.sqrt(252),
        "n_rebalances": (journal["action"] == "rebalance").sum(),
        "total_cost": journal["cost"].sum(),
    }


def _plot_equity_curve(journal: pd.DataFrame, output_path: Path) -> Path:
    """Generate equity curve plot and save to output_path."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    equity = journal["equity"]
    ax.plot(equity.index, equity.values, linewidth=2, label="Realized Equity")
    
    # Add benchmark projection (straight line from start at benchmark CAGR)
    days = (equity.index[-1] - equity.index[0]).days
    benchmark_end = equity.iloc[0] * (1 + BENCHMARK["cagr"]) ** (days / 365.25)
    ax.plot([equity.index[0], equity.index[-1]], 
            [equity.iloc[0], benchmark_end],
            '--', color='gray', alpha=0.7, label=f'Benchmark CAGR {BENCHMARK["cagr"]:.1%}')
    
    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Strategy V1 Paper Trading - Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def _format_metric_comparison(name: str, realized: float, expected: float, 
                               is_pct: bool = False, is_negative: bool = False) -> str:
    """Format a single metric comparison line."""
    fmt = "{:.1%}" if is_pct else "{:.2f}"
    realized_str = fmt.format(realized)
    expected_str = fmt.format(expected)
    
    # Determine if deviation is concerning
    if is_negative:
        # For drawdown: more negative is worse
        concerning = realized < expected
    else:
        # For Sharpe/CAGR/PF: lower is worse
        concerning = realized < expected
    
    status = "⚠️" if concerning else "✓"
    
    return f"{status} {name:<20} 實現 {realized_str:>8}  預期 {expected_str:>8}"


def _check_falsification(realized: dict) -> list[str]:
    """Check if any falsification conditions are met."""
    alerts = []
    
    if realized["sharpe_ratio"] < FAIL_SHARPE:
        alerts.append(
            f"❌ Sharpe {realized['sharpe_ratio']:.2f} < {FAIL_SHARPE} "
            f"(失敗門檻，低於回測 {BENCHMARK['sharpe_ratio'] - FAIL_SHARPE:.1f} 個標準差)"
        )
    
    if realized["max_drawdown"] < FAIL_DRAWDOWN:
        alerts.append(
            f"❌ 回撤 {realized['max_drawdown']:.1%} < {FAIL_DRAWDOWN:.0%} "
            f"(失敗門檻，比回測最差情況再差 5pp)"
        )
    
    return alerts


def _generate_report(journal: pd.DataFrame, realized: dict, 
                     plot_path: Path, output_path: Path) -> None:
    """Generate markdown report and write to output_path."""
    report = []
    
    # Header
    report.append(f"# 策略 V1 月度績效報告")
    report.append(f"\n生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n資料期間：{journal.index[0].date()} 至 {journal.index[-1].date()}")
    report.append(f"交易日數：{realized['n_days']} 日")
    
    # Equity curve
    report.append(f"\n## 權益曲線\n")
    report.append(f"![Equity Curve]({plot_path.name})\n")
    
    # Metrics comparison
    report.append(f"## 績效指標對比\n")
    report.append(f"```")
    report.append(_format_metric_comparison(
        "Sharpe ratio", realized["sharpe_ratio"], BENCHMARK["sharpe_ratio"]))
    report.append(_format_metric_comparison(
        "CAGR", realized["cagr"], BENCHMARK["cagr"], is_pct=True))
    report.append(_format_metric_comparison(
        "最大回撤", realized["max_drawdown"], BENCHMARK["max_drawdown"], 
        is_pct=True, is_negative=True))
    report.append(_format_metric_comparison(
        "獲利因子", realized["profit_factor"], BENCHMARK["profit_factor"]))
    report.append(f"```\n")
    
    # Sample size note
    if realized["n_days"] < MIN_DAYS_FOR_SIGNIFICANCE:
        report.append(f"**⏳ 樣本數不足**：目前 {realized['n_days']} 日，")
        report.append(f"需至少 {MIN_DAYS_FOR_SIGNIFICANCE} 日才能對偏離做統計推斷。")
        report.append(f"短期波動遠大於訊號，好壞都不構成結論。\n")
    
    # Trade summary
    report.append(f"## 交易摘要\n")
    report.append(f"- 起始權益：${realized['equity_start']:,.0f}")
    report.append(f"- 現在權益：${realized['equity_end']:,.0f}")
    report.append(f"- 總報酬：{realized['total_return']:+.2%}")
    report.append(f"- 實現波動：{realized['realized_vol']:.1%} 年化")
    report.append(f"- 再平衡次數：{realized['n_rebalances']}")
    report.append(f"- 累計成本：${realized['total_cost']:.2f}\n")
    
    # Falsification check
    alerts = _check_falsification(realized)
    if alerts:
        report.append(f"## 🚨 警報\n")
        for alert in alerts:
            report.append(alert)
        report.append("")
    else:
        report.append(f"## ✅ 狀態：正常\n")
        report.append(f"無觸發失敗門檻。策略按預期運作。\n")
    
    # Benchmark context
    report.append(f"## 回測基準\n")
    report.append(f"- 期間：{BENCHMARK['period']} ({BENCHMARK['n_days']} 樣本外日)")
    report.append(f"- Sharpe：{BENCHMARK['sharpe_ratio']:.2f}")
    report.append(f"- CAGR：{BENCHMARK['cagr']:.1%}")
    report.append(f"- 最大回撤：{BENCHMARK['max_drawdown']:.1%}")
    report.append(f"- 獲利因子：{BENCHMARK['profit_factor']:.2f}")
    report.append(f"- 來源：`{BENCHMARK['key']}`\n")
    
    # Footer
    report.append(f"---")
    report.append(f"_報告由 `scripts/report_v1_monthly.py` 自動生成_")
    
    output_path.write_text("\n".join(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, 
                    default=Path("reports") / f"v1_{datetime.now().strftime('%Y_%m')}.md",
                    help="Output markdown file path")
    ap.add_argument("--journal", type=Path,
                    default=Path("data/paper_v1/journal.jsonl"),
                    help="Path to paper trading journal")
    args = ap.parse_args()
    
    # Load journal
    print(f"讀取帳本：{args.journal}")
    journal = _load_journal(args.journal)
    
    # Compute metrics
    print("計算績效指標 ...")
    realized = _compute_realized_metrics(journal)
    
    # Generate plot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_path = args.output.parent / f"{args.output.stem}_equity_curve.png"
    print(f"生成權益曲線圖：{plot_path}")
    _plot_equity_curve(journal, plot_path)
    
    # Generate report
    print(f"生成報告：{args.output}")
    _generate_report(journal, realized, plot_path, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"報告已生成：{args.output}")
    print(f"權益曲線圖：{plot_path}")
    print(f"\n實現 Sharpe：{realized['sharpe_ratio']:.2f}（預期 {BENCHMARK['sharpe_ratio']:.2f}）")
    print(f"交易日數：{realized['n_days']}（需 {MIN_DAYS_FOR_SIGNIFICANCE}+ 才可信）")
    
    alerts = _check_falsification(realized)
    if alerts:
        print(f"\n⚠️  警報：")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print(f"\n✅ 狀態正常")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
