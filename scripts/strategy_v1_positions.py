"""
STRATEGY V1 — Equal-weight multi-asset basket with a volatility brake.
Run this to get the exact share counts to hold today.

    .venv/bin/python scripts/strategy_v1_positions.py --capital 100000

WHAT IT DOES, in three rules.

  1. Hold the 120 instruments in config/strategy_v1_universe.json at equal
     dollar weight. That list is frozen on disk on purpose: it is the exact
     panel the measurements below were scored on, so it cannot drift as the
     price cache grows.
  2. Scale TOTAL exposure to hit 15% annualized volatility: exposure =
     0.15 / (realized vol of the basket over the last 60 trading days), capped
     at 1.0 so it never borrows. Calm markets run near fully invested,
     violent ones get cut automatically. The rest sits in cash.
  3. Only re-size when that number moves more than 10% relative to what is
     currently held. Without the band, volatility noise churns the whole book
     and pays spread for nothing.

WHAT YOU ARE ACTUALLY HOLDING. By count the basket is 45% US single names,
19% US sector ETFs, 5% US broad-market ETFs, 13% international equity ETFs,
9% bond ETFs, 8% commodity ETFs. Equal weight therefore puts about 82% in
equities. This is a diversified, equity-heavy multi-asset portfolio with a
volatility brake -- not a market-neutral strategy and not an alpha source. In
a decade when global equities go nowhere, this goes roughly nowhere. What the
brake buys is drawdown, and that part is measured rather than asserted.

WHAT WAS MEASURED (backtests/reports/vol_target_study.json, _policy_holdout,
_gfc_stress, diversification_study.json):

  2024-01..2026-08, 658 days never used to fit or choose anything:
      Sharpe 1.86, CAGR 29.8%, max drawdown -15.8%, profit factor 1.38,
      Monte Carlo p5 Sharpe +0.88.
  Walk-forward, 1,646 out-of-sample days, target re-chosen each fold:
      Sharpe 1.19, profit factor 1.25, still 1.25 at 1.5x costs.
  2018-06..2026-08 full window at a fixed 15% target:
      Sharpe 1.19, max drawdown -23.3%.
  2006-2010 replay on 46 equity ETFs, target frozen, no bonds or commodities:
      drawdown -29.9% where unscaled was -56.2%, profit factor only 1.05,
      Monte Carlo p5 -0.65.

Read that last line as the honest cost of what this is. Through a 2008 it
roughly breaks even and still draws down about 30%. The brake nearly halved
the crash, and it was the timing rather than merely holding less: a CONSTANT
exposure at the same average still lost 38.8%, so about 9 points came from
the brake reacting. But halving a 56% drawdown still leaves 30%.

TWO KNOWN WEAKNESSES, neither of them fixed here.

  Naive equal weight double-counts US tech. SPY, QQQ, XLK, SMH, SOXX and IGV
  overlap each other and overlap the NVDA/AMD/MU/AVGO single names held
  alongside them. Equal weight by DOLLAR is not equal by RISK, so the true
  factor exposure is more concentrated than the 82% figure suggests.

  The universe is survivor-flattered. data/history was assembled from a 2026
  liquidity snapshot, so every name in it is a name that still existed in
  2026. Measured attrition on a true point-in-time 2016 S&P 500 list was 6 of
  25 names over ten years, and the names that vanish are acquisitions and
  failures. Expect live results below the backtest for this reason alone.

TURNOVER AND COST. Constituent weights change only when the frozen list
changes; exposure changes only when the band trips. That was tens of round
trips per year across the whole book, so cost is a rounding error against a
multi-week holding period -- unlike the intraday work in this repo, where
cost per trade ran 5 to 100 times the edge per trade.
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

from scripts.run_daily_baseline_gates import HISTORY  # noqa: E402
from scripts.run_vol_target_study import (  # noqa: E402
    MAX_EXPOSURE, REBALANCE_BAND, VOL_LOOKBACK, gross_returns,
)

TRADING_DAYS = 252
TARGET_VOL = 0.15
UNIVERSE_FILE = ROOT / "config/strategy_v1_universe.json"
STATE_FILE = ROOT / "backtests/reports/strategy_v1_state.json"
# Refuse to size orders off prices staler than this many trading days. Silently
# dropping a stale name would quietly change the strategy, so this warns and
# holds the line instead.
MAX_STALE_DAYS = 5


def load_universe() -> tuple[list[str], dict[str, str]]:
    spec = json.loads(UNIVERSE_FILE.read_text())
    label: dict[str, str] = {}
    for bucket, syms in spec["buckets"].items():
        for s in syms:
            label[s] = bucket
    return sorted(label), label


def load_prices(symbols: list[str]) -> pd.DataFrame:
    cols = {}
    for s in symbols:
        path = HISTORY / f"{s}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        cols[s] = df["close"].astype(float)
    return pd.DataFrame(cols).sort_index()


def current_exposure(gross: pd.Series, target_vol: float) -> tuple[float, float]:
    """Realized vol and the exposure it implies, capped at no-leverage."""
    realized = float(gross.tail(VOL_LOOKBACK).std(ddof=1) * np.sqrt(TRADING_DAYS))
    if realized <= 0:
        return 0.0, 0.0
    return realized, min(MAX_EXPOSURE, target_vol / realized)


def _capital_for_drag(prices: pd.Series, exposure: float, n: int,
                      limit: float) -> float:
    """Smallest round capital whose integer-share rounding drag stays under
    `limit`. Scanned rather than solved: the drag is a sawtooth in capital, so
    there is no closed form."""
    if exposure <= 0 or n == 0:
        return 0.0
    for capital in range(50_000, 5_000_001, 25_000):
        per = capital * exposure / n
        invested = sum(int(per / p) * p for p in prices if p > 0)
        target = capital * exposure
        if target and (target - invested) / target < limit:
            return float(capital)
    return 5_000_000.0


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
    ap.add_argument("--fractional", action="store_true",
                    help="allow fractional shares (IBKR supports these on most "
                         "US names) so equal weight is held exactly")
    ap.add_argument("--commit", action="store_true",
                    help="record today's exposure so the band works next run")
    args = ap.parse_args()

    symbols, bucket_of = load_universe()
    panel = load_prices(symbols)
    missing = sorted(set(symbols) - set(panel.columns))
    as_of = panel.index[-1]

    staleness = {s: int(panel.index.get_indexer([panel[s].last_valid_index()])[0])
                 for s in panel.columns}
    last_row = len(panel) - 1
    stale = sorted(s for s, i in staleness.items()
                   if last_row - i > MAX_STALE_DAYS)

    gross = gross_returns(panel)
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

    prices = panel.loc[as_of].dropna()
    n = len(prices)
    per_name = args.capital * exposure / n if n else 0.0

    print("策略 V1 — 等權多資產 + 波動剎車")
    print(f"資料截至 {as_of.date()}   帳戶資金 ${args.capital:,.0f}\n")
    print(f"  籃子           {n} 檔（凍結名單 {len(symbols)} 檔，等權）")
    print(f"  近 {VOL_LOOKBACK} 日實現波動  {realized:.1%} 年化")
    print(f"  目標波動       {args.target_vol:.0%}")
    print(f"  → 目標曝險     {want:.2f}"
          + ("（觸及不借貸上限 1.00）" if want >= MAX_EXPOSURE else ""))
    print(f"  → 本次採用     {exposure:.2f}   [{action}]")
    print(f"  投入 ${args.capital * exposure:,.0f}"
          f"   現金 ${args.capital * (1 - exposure):,.0f}"
          f"   每檔約 ${per_name:,.0f}")
    if missing:
        print(f"\n  !! 凍結名單有 {len(missing)} 檔在快取中找不到：{missing}")
    if stale:
        print(f"\n  !! {len(stale)} 檔價格超過 {MAX_STALE_DAYS} 個交易日未更新，"
              f"股數會用舊價算：{stale[:12]}")
        print(f"     先跑一次 refresh 再下單。")

    rows = []
    for symbol in sorted(prices.index):
        price = float(prices[symbol])
        if price <= 0:
            continue
        shares = round(per_name / price, 4) if args.fractional \
            else float(int(per_name / price))
        if shares > 0:
            rows.append((symbol, bucket_of.get(symbol, "?"), price, shares,
                         shares * price))

    qty = "{:>9.4f}" if args.fractional else "{:>9.0f}"
    print(f"\n{'標的':<7}{'類別':<18}{'價格':>9}{'股數':>9}{'金額':>11}")
    print("-" * 54)
    for symbol, bucket, price, shares, value in rows:
        print(f"{symbol:<7}{bucket:<18}{price:>9.2f}"
              + qty.format(shares) + f"{value:>11,.0f}")
    invested = sum(r[4] for r in rows)
    target = args.capital * exposure
    print("-" * 54)
    print(f"{'合計':<25}{'':>9}{'':>9}{invested:>11,.0f}")
    drag = (target - invested) / target if target else 0.0
    print(f"  目標投入 ${target:,.0f}，實際 ${invested:,.0f}，"
          f"取整缺口 {drag:.1%}")

    by_bucket: dict[str, float] = {}
    for _, bucket, _, _, value in rows:
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + value
    print("\n實際資產配置：")
    for bucket, value in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:<18}${value:>10,.0f}   {value / args.capital:>6.1%} 帳戶")

    skipped = n - len(rows)
    if not args.fractional and (skipped or drag > 0.02):
        need = _capital_for_drag(prices, exposure, n, 0.02)
        print(f"\n  整股取整讓實際配置偏離等權 {drag:.1%}"
              + (f"，其中 {skipped} 檔連一股都買不起" if skipped else "") + "。")
        print(f"  兩個解法：加 --fractional 用碎股（IBKR 多數美股支援，"
              f"可精確等權）；或把資金提高到約 ${need:,.0f} 讓缺口降到 2% 以下。")

    if args.commit:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "exposure": exposure, "as_of": str(as_of.date()),
            "target_vol": args.target_vol, "n_names": n,
            "realized_vol": realized,
            "recorded_at": datetime.now().isoformat(),
        }, indent=2), encoding="utf-8")
        print(f"\n  已記錄曝險 {exposure:.2f} 於 {STATE_FILE.name}。")
    else:
        print(f"\n  未記錄。照這個下單後請加 --commit 再跑一次，"
              f"否則下次無法判斷再平衡帶。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
