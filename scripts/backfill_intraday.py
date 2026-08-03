"""
Backfill the local 1-minute bar cache (data/history_1m/) for the fixed
backtest universe via IB Gateway.

Resumable by construction: python/data/intraday_cache.backfill_symbol_months
writes each (symbol, month) to disk as soon as it is fetched, so killing
this script and re-running it later just skips whatever is already cached
(months strictly before the current calendar month are treated as closed
and never re-fetched; --force overrides that).

Pacing note: IB's historical-data soft limit (~60 requests / 10 min) means
20 symbols x 12 months = 240 requests takes roughly 240 x 10s ≈ 40 minutes
at this module's conservative 0.1 req/s budget. Run this in the background
(scripts/backfill_intraday.py --months 12 &) and re-run later to pick up
where it left off if interrupted.

Usage:
    python scripts/backfill_intraday.py                  # last 12 months, universe.yaml symbols
    python scripts/backfill_intraday.py --months 3        # shorter smoke run
    python scripts/backfill_intraday.py --symbols AAPL,NVDA
    python scripts/backfill_intraday.py --force           # re-fetch closed months too
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
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

from python.data.intraday_cache import (
    CACHE_DIR,
    backfill_symbol_months,
    cached_symbol_coverage,
    month_range,
)
from python.data.ibkr_price_source import IbkrHistoricalUnavailable, open_ib_connection

UNIVERSE_CONFIG_PATH = Path("configs/universe.yaml")


def _universe_symbols() -> list[str]:
    with open(UNIVERSE_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return list(cfg.get("fixed_universe", {}).get("symbols", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="", help="comma-separated override; default = configs/universe.yaml")
    parser.add_argument("--months", type=int, default=12, help="trailing N months to backfill")
    parser.add_argument("--force", action="store_true", help="re-fetch even closed (already-cached) months")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--client-id", type=int, default=None,
                         help="override ibkr.historical_client_id (avoid clashing with a running backtest)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or _universe_symbols()
    if not symbols:
        print("ERROR: no symbols to backfill (configs/universe.yaml empty and --symbols not given)")
        sys.exit(1)

    now = pd.Timestamp.now()
    start = (now - pd.DateOffset(months=args.months)).replace(day=1)
    months = month_range(start, now)

    print(f"Backfilling {len(symbols)} symbols x {len(months)} months "
          f"({months[0]:%Y-%m} .. {months[-1]:%Y-%m}) into {args.cache_dir}")
    print(f"Symbols: {', '.join(symbols)}")
    print("This uses IB's ~60 req/10min pacing budget — expect roughly "
          f"{len(symbols) * len(months) * 10 / 60:.0f} minutes for a full (non-resumed) run.\n")

    try:
        ib = open_ib_connection(client_id_override=args.client_id)
    except IbkrHistoricalUnavailable as exc:
        print(f"ERROR: {exc}")
        print("Intraday backfill has NO fallback data source (yfinance 1-minute history "
              "only covers ~7 days) — IB Gateway must be running and reachable.")
        sys.exit(1)

    totals = {"fetched": 0, "skipped": 0, "empty": 0, "failed": 0}
    try:
        for symbol in symbols:
            summary = backfill_symbol_months(
                symbol, months, ib, cache_dir=args.cache_dir, force=args.force,
            )
            for key in totals:
                totals[key] += len(summary[key])
            print(f"  {symbol}: fetched={len(summary['fetched'])} skipped={len(summary['skipped'])} "
                  f"empty={len(summary['empty'])} failed={len(summary['failed'])}")
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass

    print(f"\nDone. Totals: {totals}")
    coverage = cached_symbol_coverage(symbols, cache_dir=args.cache_dir)
    thin = {s: c["n_months"] for s, c in coverage.items() if c["n_months"] < len(months)}
    if thin:
        print(f"NOTE: symbols with fewer cached months than requested (gaps or fetch failures): {thin}")


if __name__ == "__main__":
    main()
