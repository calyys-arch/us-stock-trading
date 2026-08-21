"""
Backfill the local 1-minute bar cache (data/history_1m/) for arbitrary
symbols via Futu OpenD's historical K-line API
(python/data/futu_price_source.py), as an alternative to
scripts/backfill_intraday.py (IBKR-only, unreachable in some environments —
see backtests/reports/absorption_breakout_investigation_report.md's round-3
section for why this script exists).

Not a replacement for backfill_intraday.py's role backfilling the fixed
20-symbol trading universe (that stays IBKR-sourced for backtest-vs-live
consistency, per ibkr_price_source.py's own module docstring) — this script
is for symbols OUTSIDE that universe needed for auxiliary research signals
(e.g. QQQ/SPY/XLK for a macro/sector-beta alignment filter), where Futu is
the only reachable historical 1-minute source in this environment.

ALWAYS run with --check-quota-only first against a small symbol before a
real backfill — Futu's historical-kline quota is consumed once PER DISTINCT
STOCK CODE ever queried (not per request/date-range — see
futu_price_source.py's module docstring for how this was verified), so a
tiny probe costs the same 1 quota point the real fetch will anyway, but lets
you see the account's quota budget before committing to a whole symbol list.

Usage:
    python scripts/backfill_futu_symbols.py --check-quota-only --symbols QQQ
    python scripts/backfill_futu_symbols.py --symbols QQQ,SPY,XLK --start 2025-08-01 --end 2026-08-01
    python scripts/backfill_futu_symbols.py --symbols QQQ,SPY,XLK --start 2025-08-01 --end 2026-08-01 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

from python.data.futu_price_source import (
    FutuHistoricalUnavailable,
    backfill_symbol_months,
    check_history_kline_quota,
    open_futu_quote_context,
)
from python.data.intraday_cache import CACHE_DIR, cached_symbol_coverage, month_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", required=True, help="comma-separated, e.g. QQQ,SPY,XLK")
    parser.add_argument("--start", default="2025-08-01", help="first day of history to backfill")
    parser.add_argument("--end", default=str(pd.Timestamp.now().date()), help="last day of history to backfill")
    parser.add_argument("--force", action="store_true", help="re-fetch even closed (already-cached) months")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--check-quota-only", action="store_true",
                         help="just print the account's current historical-kline quota usage and exit")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("ERROR: --symbols is required (comma-separated)")
        sys.exit(1)

    try:
        ctx = open_futu_quote_context()
    except FutuHistoricalUnavailable as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    try:
        quota = check_history_kline_quota(ctx)
        print(f"Futu historical-kline quota: used={quota['used']} remaining={quota['remaining']}")
        if quota["detail"]:
            print(f"  Previously-unlocked codes: {[d.get('code') for d in quota['detail']]}")
        if args.check_quota_only:
            return
        if quota["remaining"] < len(symbols):
            print(f"ERROR: remaining quota ({quota['remaining']}) is less than the number of "
                  f"NEW symbols requested ({len(symbols)}) — stopping rather than risk exhausting "
                  "the account's budget. Re-run with --check-quota-only to inspect first.")
            sys.exit(1)

        months = month_range(pd.Timestamp(args.start), pd.Timestamp(args.end))
        print(f"Backfilling {len(symbols)} symbols x {len(months)} months "
              f"({months[0]:%Y-%m} .. {months[-1]:%Y-%m}) into {args.cache_dir}")
        print(f"Symbols: {', '.join(symbols)}\n")

        totals = {"fetched": 0, "skipped": 0, "empty": 0, "failed": 0}
        for symbol in symbols:
            try:
                summary = backfill_symbol_months(
                    symbol, months, ctx, cache_dir=args.cache_dir, force=args.force,
                )
            except FutuHistoricalUnavailable as exc:
                print(f"  {symbol}: FAILED — {exc}")
                totals["failed"] += len(months)
                continue
            for key in totals:
                totals[key] += len(summary[key])
            print(f"  {symbol}: fetched={len(summary['fetched'])} skipped={len(summary['skipped'])} "
                  f"empty={len(summary['empty'])} failed={len(summary['failed'])}")

        print(f"\nDone. Totals: {totals}")
        coverage = cached_symbol_coverage(symbols, cache_dir=args.cache_dir)
        thin = {s: c["n_months"] for s, c in coverage.items() if c["n_months"] < len(months)}
        if thin:
            print(f"NOTE: symbols with fewer cached months than requested (gaps or fetch failures): {thin}")

        quota_after = check_history_kline_quota(ctx)
        print(f"Quota after backfill: used={quota_after['used']} remaining={quota_after['remaining']}")
    finally:
        try:
            ctx.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
