"""
STRATEGY V1 — Bucketed multi-asset basket with a volatility brake.
Run this to get the exact share counts to hold today.

    .venv/bin/python scripts/strategy_v1_positions.py --capital 100000 --fractional

WHAT IT DOES, in four rules.

  1. Hold the 120 instruments in config/strategy_v1_universe.json. That list is
     frozen on disk on purpose: it is the exact panel every number below was
     scored on, so it cannot drift as the price cache grows.
  2. Split the book equally across four asset buckets -- US equity,
     international equity, bonds, commodities -- then equally within each
     bucket. The three US equity groups (single names, sector ETFs, broad
     ETFs) collapse into one bucket because they are one beta: SPY holds NVDA,
     XLK holds NVDA, SMH holds NVDA, and NVDA is in there on its own.
  3. Scale TOTAL exposure to hit 15% annualized volatility: exposure =
     0.15 / (realized vol of the basket over the last 60 trading days), capped
     at 1.0 so it never borrows. The rest sits in cash.
  4. Only re-size when that number moves more than 10% relative to what is
     currently held. Without the band, volatility noise churns the whole book
     and pays spread for nothing.

WHY RULE 2 EXISTS. The first version equal-weighted all 120 instruments, which
sounds neutral and is not: 83 of the 120 are US equity, so US beta got 69% of
the book purely because of how many tickers happened to be cached. Bucketing
makes the allocation a decision instead of an artifact. Measured over 1,646
walk-forward out-of-sample days (backtests/reports/risk_bucket_study.json):

    weighting            Sharpe    PF   max DD   positive folds
    all-120 equal          0.91  1.18   -31.4%           15/19
    four buckets @ 25%     0.89  1.17   -24.6%           16/19
    buckets @ inverse vol  0.75  1.15   -19.2%           15/19

Bucketing gives up 0.02 Sharpe and buys 6.8 points of drawdown, and it is the
only arm whose out-of-sample drawdown stays inside the -25% limit. Inverse-
volatility weighting cuts drawdown further but is not worth it here: with no
leverage allowed it parks 55% of the book in bonds, runs at 9.6% realized vol
against a 15% target, and halves the return to reach it.

WHAT YOU ARE ACTUALLY HOLDING. Equal buckets put 25% in US equity, 25%
international equity, 25% bonds, 25% commodities. This is a diversified
multi-asset portfolio with a volatility brake -- not market-neutral and not an
alpha source. In a decade when global equities and commodities both go
nowhere, this goes roughly nowhere. What the brake buys is drawdown, and that
part is measured rather than asserted.

WHAT ELSE WAS MEASURED (vol_target_study.json, _policy_holdout, _gfc_stress,
diversification_study.json, risk_bucket_study.json):

  2024-01..2026-08 at equal buckets: Sharpe 1.64, CAGR 22.4%, drawdown -11.9%,
      PF 1.34. That window was used once already to report the all-120 version,
      so read it as a second look at an a-priori rule, not a fresh holdout.
  2018-06..2026-08 at a fixed 15% target, equal buckets: Sharpe 0.89,
      CAGR 11.5%, drawdown -24.6%.
  2006-2010 replay on 46 equity ETFs, target frozen, no bonds or commodities:
      drawdown -29.9% where unscaled was -56.2%, but profit factor only 1.05.

Read that last line as the honest cost. Through a 2008-style decade this
roughly breaks even. The brake nearly halved the crash, and it was the timing
rather than merely holding less: a CONSTANT exposure at the same average still
lost 38.8%, so about 9 points came from the brake reacting. But halving a 56%
drawdown still leaves 30%.

THE REMAINING KNOWN WEAKNESS. The universe is survivor-flattered.
data/history was assembled from a 2026 liquidity snapshot, so every name in it
is a name that still existed in 2026. Measured attrition on a true
point-in-time 2016 S&P 500 list was 6 of 25 names over ten years, and the
names that vanish are acquisitions and failures. Expect live results below the
backtest for this reason alone.

TURNOVER AND COST. Bucket weights are fixed, so constituents only trade when
the frozen list changes; exposure only trades when the band trips. That was
tens of round trips per year across the whole book, so cost is a rounding
error against a multi-week holding period -- unlike the intraday work in this
repo, where cost per trade ran 5 to 100 times the edge per trade.
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
from scripts.run_vol_target_study import (  # noqa: E402
    MAX_EXPOSURE, ONE_WAY_COST_BPS, REBALANCE_BAND, VOL_LOOKBACK,
    gross_returns,
)

TRADING_DAYS = 252
TARGET_VOL = 0.15
STATE_FILE = ROOT / "backtests/reports/strategy_v1_state.json"
# Refuse to size orders off prices staler than this many trading days. Silently
# dropping a stale name would quietly change the strategy, so this warns and
# holds the line instead.
MAX_STALE_DAYS = 5


def current_exposure(gross: pd.Series, target_vol: float) -> tuple[float, float]:
    """Realized vol and the exposure it implies, capped at no-leverage."""
    realized = float(gross.tail(VOL_LOOKBACK).std(ddof=1) * np.sqrt(TRADING_DAYS))
    if realized <= 0:
        return 0.0, 0.0
    return realized, min(MAX_EXPOSURE, target_vol / realized)


def _capital_for_drag(weights: dict[str, float], prices: pd.Series,
                      exposure: float, limit: float) -> float:
    """Smallest round capital whose integer-share rounding drag stays under
    `limit`. Scanned rather than solved: the drag is a sawtooth in capital, so
    there is no closed form."""
    if exposure <= 0:
        return 0.0
    for capital in range(50_000, 20_000_001, 25_000):
        invested = 0.0
        for symbol, weight in weights.items():
            price = float(prices.get(symbol, 0.0))
            if price > 0:
                invested += int(capital * exposure * weight / price) * price
        target = capital * exposure
        if target and (target - invested) / target < limit:
            return float(capital)
    return 20_000_000.0


def held_last_time() -> float | None:
    if not STATE_FILE.exists():
        return None
    try:
        return float(json.loads(STATE_FILE.read_text())["exposure"])
    except (ValueError, KeyError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, required=True,
                    help="account equity to allocate, in dollars")
    ap.add_argument("--target-vol", type=float, default=TARGET_VOL,
                    help="annualized volatility target (default 0.15)")
    ap.add_argument("--weights", choices=("bucket", "naive"), default="bucket",
                    help="bucket = four asset buckets at 25%% each (default, "
                         "and the arm that passes the drawdown limit); "
                         "naive = all instruments equal, the first version")
    ap.add_argument("--fractional", action="store_true",
                    help="allow fractional shares (IBKR supports these on most "
                         "US names) so the target weights are held exactly")
    ap.add_argument("--commit", action="store_true",
                    help="record today's exposure so the band works next run")
    args = ap.parse_args()

    symbols, bucket_of = load_universe()
    panel = load_prices(symbols)
    missing = sorted(set(symbols) - set(panel.columns))
    as_of = panel.index[-1]
    held = [s for s in symbols if s in panel.columns]

    last_row = len(panel) - 1
    stale = sorted(
        s for s in panel.columns
        if last_row - int(panel.index.get_indexer(
            [panel[s].last_valid_index()])[0]) > MAX_STALE_DAYS)

    gross = (gross_returns(panel) if args.weights == "naive"
             else bucket_weighted_gross(panel, bucket_of, "equal",
                                        VOL_LOOKBACK, REBALANCE_BAND,
                                        ONE_WAY_COST_BPS)[0])
    realized, want = current_exposure(gross, args.target_vol)
    holding = held_last_time()

    if holding is None:
        exposure, action = want, "初次建立部位"
    elif abs(want - holding) / max(holding, 1e-9) > REBALANCE_BAND:
        exposure, action = want, f"調整曝險 {holding:.2f} -> {want:.2f}"
    else:
        exposure, action = holding, (
            f"不動作：目標 {want:.2f} 與現有 {holding:.2f} 差距未超過 "
            f"{REBALANCE_BAND:.0%} 再平衡帶")

    weights = target_weights(held, bucket_of, args.weights)
    prices = panel.loc[as_of]

    scheme_label = ("四桶各 25%（等權於桶內）" if args.weights == "bucket"
                    else "全部標的等權（第一版）")
    print("策略 V1 — 多資產 + 波動剎車")
    print(f"資料截至 {as_of.date()}   帳戶資金 ${args.capital:,.0f}")
    print(f"權重規則 {scheme_label}\n")
    print(f"  籃子           {len(held)} 檔（凍結名單 {len(symbols)} 檔）")
    print(f"  近 {VOL_LOOKBACK} 日實現波動  {realized:.1%} 年化")
    print(f"  目標波動       {args.target_vol:.0%}")
    print(f"  → 目標曝險     {want:.2f}"
          + ("（觸及不借貸上限 1.00）" if want >= MAX_EXPOSURE else ""))
    print(f"  → 本次採用     {exposure:.2f}   [{action}]")
    print(f"  投入 ${args.capital * exposure:,.0f}"
          f"   現金 ${args.capital * (1 - exposure):,.0f}")
    if missing:
        print(f"\n  !! 凍結名單有 {len(missing)} 檔在快取中找不到：{missing}")
    if stale:
        print(f"\n  !! {len(stale)} 檔價格超過 {MAX_STALE_DAYS} 個交易日未更新，"
              f"股數會用舊價算：{stale[:12]}")

    rows = []
    for symbol in sorted(held):
        price = float(prices.get(symbol, float("nan")))
        if not price > 0:
            continue
        dollars = args.capital * exposure * weights[symbol]
        shares = round(dollars / price, 4) if args.fractional \
            else float(int(dollars / price))
        if shares > 0:
            rows.append((symbol, COLLAPSE.get(bucket_of[symbol], "?"), price,
                         shares, shares * price))

    qty = "{:>9.4f}" if args.fractional else "{:>9.0f}"
    print(f"\n{'標的':<7}{'桶':<14}{'價格':>9}{'股數':>9}{'金額':>11}")
    print("-" * 50)
    for symbol, bucket, price, shares, value in rows:
        print(f"{symbol:<7}{bucket:<14}{price:>9.2f}"
              + qty.format(shares) + f"{value:>11,.0f}")
    invested = sum(r[4] for r in rows)
    target = args.capital * exposure
    drag = (target - invested) / target if target else 0.0
    print("-" * 50)
    print(f"{'合計':<21}{'':>9}{'':>9}{invested:>11,.0f}")
    print(f"  目標投入 ${target:,.0f}，實際 ${invested:,.0f}，"
          f"取整缺口 {drag:.1%}")

    by_bucket: dict[str, float] = {}
    for _, bucket, _, _, value in rows:
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + value
    print("\n實際桶配置（目標每桶 25% 的投入金額）：")
    for bucket, value in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        share = value / invested if invested else 0.0
        print(f"  {bucket:<14}${value:>10,.0f}   佔投入 {share:>6.1%}"
              f"   佔帳戶 {value / args.capital:>6.1%}")

    skipped = len(held) - len(rows)
    if not args.fractional and (skipped or drag > 0.02):
        need = _capital_for_drag(weights, prices, exposure, 0.02)
        print(f"\n  整股取整讓實際配置偏離目標 {drag:.1%}"
              + (f"，其中 {skipped} 檔連一股都買不起" if skipped else "") + "。")
        print(f"  兩個解法：加 --fractional 用碎股（IBKR 多數美股支援，"
              f"可精確配置）；或把資金提高到約 ${need:,.0f} 讓缺口降到 2% 以下。")

    if args.commit:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "exposure": exposure, "as_of": str(as_of.date()),
            "target_vol": args.target_vol, "weights": args.weights,
            "n_names": len(rows), "realized_vol": realized,
            "recorded_at": datetime.now().isoformat(),
        }, indent=2), encoding="utf-8")
        print(f"\n  已記錄曝險 {exposure:.2f} 於 {STATE_FILE.name}。")
    else:
        print(f"\n  未記錄。照這個下單後請加 --commit 再跑一次，"
              f"否則下次無法判斷再平衡帶。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
