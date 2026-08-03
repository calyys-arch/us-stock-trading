"""
Refresh the report-only diagnostic layer's event data caches:
  - SEC EDGAR 8-K filings (full history — data/filings/8k/<SYMBOL>.json)
  - Finnhub company news (free tier: ~last 12 months — data/news/)
  - Finnhub earnings calendar (data/calendar/earnings_<year>.json)
  - Finnhub economic calendar (premium-gated; skipped gracefully on 403)

Symbols default to the fixed backtest universe (configs/universe.yaml,
built by scripts/refresh_universe.py) plus the default pairs tickers.

Usage:
    python scripts/refresh_event_data.py                          # universe + XLE/XOP
    python scripts/refresh_event_data.py --symbols AAPL,MSFT
    python scripts/refresh_event_data.py --since 2018-01-01 --skip-news
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

from python.data.edgar_client import EdgarClient
from python.data.finnhub_client import FinnhubClient


def _default_symbols() -> list[str]:
    symbols = {"XLE", "XOP"}  # default pairs tickers (scripts/run_backtest.py)
    try:
        from python.data.fixed_universe import load_universe_config

        symbols.update(load_universe_config()["symbols"])
    except Exception as exc:
        print(f"NOTE: fixed universe unavailable ({exc}) — using pairs tickers only")
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="",
                        help="comma-separated tickers (default: fixed universe + XLE,XOP)")
    parser.add_argument("--since", default="2018-01-01",
                        help="8-K backfill start date (also the earnings-calendar range start)")
    parser.add_argument("--skip-edgar", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-calendar", action="store_true")
    args = parser.parse_args()

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else _default_symbols())
    since = pd.Timestamp(args.since).date()
    today = pd.Timestamp.now().date()
    print(f"Refreshing event data for {len(symbols)} symbols, since {since}")

    if not args.skip_edgar:
        print("\n-- SEC EDGAR 8-K --")
        edgar = EdgarClient()
        for symbol in symbols:
            cached = edgar.get_cached_8k(symbol)
            filings = edgar.refresh_8k(symbol) if cached is not None else edgar.backfill_8k(symbol, since=since)
            print(f"  {symbol}: {len(filings)} filings"
                  + (" (incremental)" if cached is not None else " (backfill)"))

    finnhub = FinnhubClient()
    if not args.skip_news:
        print("\n-- Finnhub company news (free tier: ~12 months back) --")
        news_start = max(since, (pd.Timestamp.now() - pd.Timedelta(days=365)).date())
        for symbol in symbols:
            rows = finnhub.company_news(symbol, news_start, today)
            print(f"  {symbol}: {len(rows)} headlines cached [{news_start} .. {today}]")

    if not args.skip_calendar:
        print("\n-- Finnhub calendars --")
        earnings = finnhub.earnings_calendar(since, today)
        print(f"  earnings calendar: {len(earnings)} rows [{since} .. {today}]")
        econ = finnhub.economic_calendar(since, today)
        print(f"  economic calendar: {len(econ)} rows (empty = premium-gated or no key)")

    print("\nDone. Caches: data/filings/, data/news/, data/calendar/")


if __name__ == "__main__":
    main()
