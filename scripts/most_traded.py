"""
Find the most heavily-traded US stocks year-to-date, ranked by actual
volume data (not intuition/memory of "popular" tickers).

Ranking metric: average daily SHARE volume over the YTD window — this is a
better proxy for "how many people/trades are involved" than average DOLLAR
volume, which just rewards a high share price (e.g. a $500 stock trading 1M
shares/day shows up as more dollar volume than a $10 stock trading 40M
shares/day, even though the $10 stock clearly has more individual trades/
participants). Both metrics are reported so you can judge either way.

Universe: the full S&P 500 (for broad large/mega-cap coverage) UNIONED with
an explicit watchlist of well-known high-volume tickers that are NOT
currently S&P 500 members (meme/retail-favorite/EV/crypto-adjacent names
that are frequently among the most-traded tickers on US exchanges but don't
meet S&P 500's profitability/index-committee criteria).

Usage:
    .venv/bin/python scripts/most_traded.py --top 10 --limit 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from python.data.sp500_universe import fetch_current_constituents

# Well-known high-volume US-listed names historically NOT in the S&P 500
# (excluded by profitability/committee criteria despite huge trading volume).
_NON_SP500_HIGH_VOLUME_WATCHLIST = [
    "SOFI", "RIVN", "LCID", "NIO", "GME", "AMC", "COIN", "MARA", "RIOT",
    "PLUG", "NU", "PBR", "VALE", "ITUB", "BBD", "AAL", "CLSK", "AFRM",
    "CCL", "UBER", "LYFT", "SNAP", "PARA", "WBD", "F", "T", "PFE",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--limit", type=int, default=150, help="cap S&P 500 sample size for a fast interactive pull")
    args = ap.parse_args()

    print("Fetching current S&P 500 constituent list (Wikipedia)...")
    constituents = fetch_current_constituents()
    sector_map = dict(zip(constituents["symbol"], constituents["sector"]))
    all_sp500 = sorted(sector_map)

    if args.limit < len(all_sp500):
        step = len(all_sp500) / args.limit
        sp500_sample = sorted({all_sp500[int(i * step)] for i in range(args.limit)})
    else:
        sp500_sample = all_sp500

    watchlist = [s for s in _NON_SP500_HIGH_VOLUME_WATCHLIST if s not in sector_map]
    symbols = sorted(set(sp500_sample) | set(watchlist))
    for s in watchlist:
        sector_map[s] = "(not S&P 500)"
    print(f"Universe: {len(symbols)} symbols "
          f"({len(sp500_sample)} evenly-sampled S&P 500 + {len(watchlist)} watchlist non-index names)")

    print("Downloading year-to-date daily bars (sequential yfinance calls; threads=True hangs in this env)...")
    raw = yf.download(
        symbols, period="ytd", auto_adjust=True, progress=False,
        group_by="ticker", threads=False,
    )

    rows = []
    for sym in symbols:
        try:
            sub = raw[sym][["Close", "Volume"]].dropna()
        except (KeyError, TypeError):
            continue
        if len(sub) < 10:
            continue
        avg_share_vol = float(sub["Volume"].mean())
        avg_dollar_vol = float((sub["Close"] * sub["Volume"]).mean())
        rows.append({
            "symbol": sym,
            "sector": sector_map.get(sym, "?"),
            "n_days": len(sub),
            "avg_daily_share_volume": avg_share_vol,
            "avg_daily_dollar_volume": avg_dollar_vol,
        })

    if not rows:
        print("No data fetched — aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)}/{len(symbols)} symbols with usable YTD data.")

    by_shares = df.sort_values("avg_daily_share_volume", ascending=False).head(args.top)
    by_dollars = df.sort_values("avg_daily_dollar_volume", ascending=False).head(args.top)

    print(f"\n=== Top {args.top} by AVERAGE DAILY SHARE VOLUME, YTD {pd.Timestamp.now().year} ===")
    print(f"{'Symbol':<8}{'Sector':<26}{'Avg shares/day':>18}{'Avg $ volume/day':>20}")
    for _, r in by_shares.iterrows():
        print(f"{r['symbol']:<8}{r['sector'][:24]:<26}{r['avg_daily_share_volume']:>18,.0f}"
              f"{r['avg_daily_dollar_volume']:>20,.0f}")

    print(f"\n=== Top {args.top} by AVERAGE DAILY DOLLAR VOLUME, YTD {pd.Timestamp.now().year} (for comparison) ===")
    print(f"{'Symbol':<8}{'Sector':<26}{'Avg shares/day':>18}{'Avg $ volume/day':>20}")
    for _, r in by_dollars.iterrows():
        print(f"{r['symbol']:<8}{r['sector'][:24]:<26}{r['avg_daily_share_volume']:>18,.0f}"
              f"{r['avg_daily_dollar_volume']:>20,.0f}")


if __name__ == "__main__":
    main()
