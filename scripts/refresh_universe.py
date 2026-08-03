"""
Build/refresh the fixed top-N backtest universe (configs/universe.yaml).

Ranks the point-in-time (S&P 500 UNION Nasdaq-100) pool by trailing 60-day
average dollar volume as of --as-of, keeps the top N, and writes the list +
metadata to configs/universe.yaml. Also pre-warms the local price cache
(data/history/) for the selected names back to --check-history-from, and
WARNS about any selected symbol whose available history starts after that
date (e.g. a recent IPO) — those names will thin out the early years of a
backtest (see configs/strategy.yaml min_universe_size).

Usage:
    python scripts/refresh_universe.py                      # top 20, 60d ranking, full pool
    python scripts/refresh_universe.py --limit 120          # subsample pool (faster smoke run)
    python scripts/refresh_universe.py --refresh-data       # force re-fetch of cached prices
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

from python.data.fixed_universe import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TOP_N,
    UNIVERSE_CONFIG_PATH,
    save_universe_config,
    select_fixed_top_n,
)
from python.data.liquid_universe import combined_index_membership
from python.data.price_cache import first_available_dates, get_cached_price_panel

# Calendar-day buffer so the trailing 60-TRADING-day ranking window always
# has enough bars regardless of holidays.
_RANKING_FETCH_BUFFER_DAYS = 150


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="trailing trading-day window for the dollar-volume ranking")
    parser.add_argument("--as-of", default=str(pd.Timestamp.now().date()),
                        help="ranking snapshot date (YYYY-MM-DD, default today)")
    parser.add_argument("--check-history-from", default="2018-01-01",
                        help="warn about selected symbols whose history starts after this date; "
                             "also pre-warms the price cache for them from this date")
    parser.add_argument("--limit", type=int, default=0,
                        help="evenly subsample the candidate pool to N names (0 = full pool). "
                             "Faster for smoke runs; the ranking is then approximate.")
    parser.add_argument("--refresh-data", action="store_true",
                        help="force re-fetch of cached price data")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)

    print(f"Building candidate pool (point-in-time S&P 500 UNION Nasdaq-100 as of {as_of.date()})...")
    pool = sorted(combined_index_membership(as_of))
    if args.limit and args.limit < len(pool):
        step = len(pool) / args.limit
        pool = [pool[int(i * step)] for i in range(args.limit)]
        print(f"POOL SUBSAMPLED to {len(pool)} names (--limit) — ranking is approximate")
    print(f"Candidate pool: {len(pool)} symbols")

    fetch_start = as_of - pd.Timedelta(days=_RANKING_FETCH_BUFFER_DAYS)
    print(f"Fetching ranking window [{fetch_start.date()}, {as_of.date()}] via price cache...")
    panel, quality_flags, meta = get_cached_price_panel(pool, fetch_start, as_of, refresh=args.refresh_data)
    if quality_flags:
        print(f"WARNING: data-quality flags on {len(quality_flags)} symbols: {sorted(quality_flags)}")
    print(f"Data sources: { {src: len(syms) for src, syms in meta['sources'].items()} } "
          f"(cache hits: {len(meta['from_cache'])}, fetched: {len(meta['fetched'])})")

    present = sorted(set(panel.index.get_level_values(1)))
    missing_pool = sorted(set(pool) - set(present))
    if missing_pool:
        print(f"NOTE: {len(missing_pool)} pool symbols had no price data and were excluded from ranking")

    selected = select_fixed_top_n(present, panel, as_of, top_n=args.top_n, lookback_days=args.lookback_days)
    print(f"\nSelected top {len(selected)} by trailing {args.lookback_days}d avg dollar volume:")
    print("  " + ", ".join(selected))

    history_from = pd.Timestamp(args.check_history_from)
    print(f"\nPre-warming price cache for selected names from {history_from.date()}...")
    get_cached_price_panel(selected, history_from, as_of, refresh=args.refresh_data)

    firsts = first_available_dates(selected)
    late_starters = {
        s: d for s, d in firsts.items() if d > history_from + pd.Timedelta(days=10)
    }
    no_data = [s for s in selected if s not in firsts]
    for s, d in sorted(late_starters.items()):
        print(f"WARNING: {s} history starts {d.date()} (after {history_from.date()}) — "
              f"early backtest years will run with a thinner universe")
    for s in no_data:
        print(f"WARNING: {s} has NO cached history at all")

    save_universe_config(
        selected, as_of, args.lookback_days,
        source_pool_label=f"sp500_union_nasdaq100_pit_{as_of.date()}"
                          + (f"_subsampled_{len(pool)}" if args.limit else ""),
    )
    print(f"\nWrote {UNIVERSE_CONFIG_PATH} ({len(selected)} symbols, computed_at={as_of.date()})")


if __name__ == "__main__":
    main()
