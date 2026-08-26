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
    four buckets @ 25%     0.91  1.17   -24.6%           16/19
    buckets @ inverse vol  0.75  1.15   -19.2%           15/19

Bucketing costs nothing in Sharpe and buys 6.8 points of drawdown, and it is the
only arm whose drawdown stays inside the -25% limit.

READ THAT TABLE WITH TWO CAVEATS, both measured rather than suspected.

  The drawdown column is not independent of the full-window number quoted
      below. The pooled walk-forward drawdown and the fixed-15%-target
      full-window drawdown are both -0.24597529605973933 -- identical to every
      digit, both troughing on 2020-03-18. They are one measurement printed
      twice, so the walk-forward adds no independent support to the -24.6%
      claim, which is the claim the bucketing decision rests on.
  The walk-forward validates a policy nobody runs. It re-picks the target each
      fold (0.15 ten times, 0.20 once, 0.25 eight times) while the shipped rule
      is a fixed 15%. Numerically the gap is small -- pooled 0.907 against
      full-sample 0.911 -- because any target at or above 0.20 just pins
      exposure at the 1.0 cap, which is also exactly why the two drawdowns
      collapse onto the same number.

Inverse-
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

  2024-01..2026-08 at equal buckets: Sharpe 1.65, CAGR 22.7%, drawdown -11.9%,
      PF 1.34.
  2018-06..2026-08 at a fixed 15% target, equal buckets: Sharpe 0.91,
      CAGR 11.7%, drawdown -24.6%.

TWO CLAIMS OF DIFFERENT QUALITY, which earlier drafts blurred together.

  The 15% target is clean. It was picked on a development window ending
      2023-12-29 as the largest target whose drawdown still respected the limit,
      then checked once on 658 days it had never seen (vol_target_policy_holdout
      .json: Sharpe 1.86, drawdown -15.8%). That is a real holdout.
  The bucketing is not clean. Three candidate weightings were specified up
      front, but the winner was chosen by reading walk-forward results that span
      2018-2026 -- including those same 658 days. Three candidates on a drawdown
      criterion is mild selection, not a fishing expedition, and the losing arm
      failed by 6.8 points rather than a hair. Still: the 2024-2026 numbers
      above are not an out-of-sample test of the bucketing decision, and the
      forward paper record is the first thing that will be.
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

TURNOVER AND COST, and why the cost stress test is weak evidence. Bucket
weights are fixed, so constituents only trade when the frozen list changes;
exposure only trades when the band trips. Measured: 30 exposure changes in
eight years, about 4 a year, total charged turnover 3.2x the book, roughly
13bps over the whole sample. Cost really is a rounding error here -- unlike the
intraday work in this repo, where cost per trade ran 5 to 100 times the edge.

But that is also why `stress_slippage_1.5x_pf_ge_1: true` proves almost
nothing. Multiplying 13bps by 1.5 moves pooled Sharpe by 0.0005. The gate
passing means nearly nothing was charged, not that the edge survives costs. And
the larger real cost is charged at zero: the backtest re-weights to
equal-within-bucket every day for free, which is 0.5% one-way turnover a day, or
about 5bps a year -- three times the only cost it does charge. That omission is
small enough not to matter (about 0.004 of Sharpe) but it is the bigger of the
two, and the ledger pays it monthly rather than pretending it is free.

WHAT THE 16-OF-19 FOLD COUNT ACTUALLY TESTS. In 14 of the 19 out-of-sample
folds, exposure sat pinned at 1.00 for the entire window and the brake never
moved. That is the design working -- a vol target capped at 1.0 is correctly
inert whenever realized vol is under target -- but it means the fold breadth is
overwhelmingly a statement about the bucketed basket, not about the sizing rule.
The brake is exercised in 5 folds, and those are the ones that matter (2020,
2022): fold 1 ran exposure down to 0.33.
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

    # Size only what can be priced, but re-derive the weights over that subset
    # so the four buckets still sit at 25% each. The sizing loop used to skip an
    # unpriced name and leave its share of the book in cash, then report the hole
    # as "integer rounding" -- a cause that cannot produce it under --fractional.
    # Three missing commodity prices silently took that bucket to 18.9%.
    prices = panel.ffill().loc[as_of]
    priceable = [s for s in held if float(prices.get(s, float("nan"))) > 0]
    unpriceable = sorted(set(held) - set(priceable))
    weights = target_weights(priceable, bucket_of, args.weights)

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
    if unpriceable:
        print(f"\n  !! {len(unpriceable)} 檔完全沒有價格，已從本次配置剔除，"
              f"權重在其餘標的上重新歸一（桶配比維持 25%）：{unpriceable[:12]}")

    rows = []
    for symbol in sorted(priceable):
        price = float(prices[symbol])
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
    label = "取整缺口" if args.fractional is False else "缺口"
    print(f"  目標投入 ${target:,.0f}，實際 ${invested:,.0f}，{label} {drag:.1%}"
          + ("（整股取整所致）" if not args.fractional else
             "（零股模式下應為 0；非零表示還有其他原因）"))

    by_bucket: dict[str, float] = {}
    for _, bucket, _, _, value in rows:
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + value
    print("\n實際桶配置（目標每桶 25% 的投入金額）：")
    for bucket, value in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        share = value / invested if invested else 0.0
        print(f"  {bucket:<14}${value:>10,.0f}   佔投入 {share:>6.1%}"
              f"   佔帳戶 {value / args.capital:>6.1%}")

    skipped = len(priceable) - len(rows)
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
