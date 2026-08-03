"""
Backfill the local FINRA ATS weekly dark-pool-volume cache (data/finra_ats/)
for the fixed backtest universe.

Two-phase, resumable strategy (python/data/finra_ats.py has the full
rationale for why these are two different API calls):

  1. --phase recent: one targeted per-symbol query against `weeklySummary`,
     which empirically covers roughly the trailing ~3-4 years. Fast (one
     paginated call per symbol) — always run this first.
  2. --phase historic: fills in weeks OLDER than whatever `weeklySummary`
     already covers, one full market-wide `weeklySummaryHistoric` call PER
     CALENDAR WEEK (that dataset can't be filtered by symbol server-side),
     filtered down to our universe client-side. This is the slow part —
     backfilling 2018-2025 is ~365 weekly calls, each pulling and filtering
     a market-wide page set. Resumable: data/finra_ats/_meta.json tracks
     which weeks have already been fetched (successfully — even if a week
     had zero matches for our universe, e.g. before any of them IPO'd) so a
     killed/re-run backfill only fetches what's still missing.

Usage:
    python scripts/backfill_finra_ats.py --phase recent
    python scripts/backfill_finra_ats.py --phase historic --start 2018-01-01 --end 2023-01-01
    python scripts/backfill_finra_ats.py --phase historic --symbols AAPL,NVDA --start 2018-01-01
"""
from __future__ import annotations

import argparse
import json
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
log = logging.getLogger("backfill_finra_ats")

from python.data.finra_ats import CACHE_DIR, fetch_all_recent_weeks, fetch_historic_week, save_weeks

UNIVERSE_CONFIG_PATH = Path("configs/universe.yaml")


def _universe_symbols() -> list[str]:
    with open(UNIVERSE_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return list(cfg.get("fixed_universe", {}).get("symbols", []))


def _meta_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "_meta.json"


def _load_meta(cache_dir: Path) -> dict:
    path = _meta_path(cache_dir)
    if not path.exists():
        return {"historic_weeks_done": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_meta(cache_dir: Path, meta: dict) -> None:
    _meta_path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
    _meta_path(cache_dir).write_text(json.dumps(meta), encoding="utf-8")


def run_recent(symbols: list[str], cache_dir: Path) -> None:
    for symbol in symbols:
        weeks = fetch_all_recent_weeks(symbol)
        if weeks:
            save_weeks(symbol, weeks, cache_dir=cache_dir)
        log.info("recent: %s -> %d weeks cached (%s .. %s)", symbol, len(weeks),
                 weeks[0]["week_start_date"] if weeks else "n/a",
                 weeks[-1]["week_start_date"] if weeks else "n/a")


def run_historic(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path, force: bool) -> None:
    meta = _load_meta(cache_dir)
    done = set(meta.get("historic_weeks_done", []))

    mondays = pd.date_range(start - pd.Timedelta(days=start.weekday()), end, freq="W-MON")
    total, skipped, empty, fetched = len(mondays), 0, 0, 0

    for week in mondays:
        week_str = week.strftime("%Y-%m-%d")
        if not force and week_str in done:
            skipped += 1
            continue
        by_symbol = fetch_historic_week(week_str, set(symbols))
        if by_symbol:
            for symbol, row in by_symbol.items():
                save_weeks(symbol, [row], cache_dir=cache_dir)
            fetched += 1
        else:
            empty += 1
        done.add(week_str)
        meta["historic_weeks_done"] = sorted(done)
        _save_meta(cache_dir, meta)
        if (fetched + empty) % 20 == 0:
            log.info("historic: %s done (%d/%d weeks processed this run)", week_str, fetched + empty, total - skipped)

    log.info("historic backfill done: %d weeks fetched, %d empty (no universe match), %d already-cached skipped",
              fetched, empty, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["recent", "historic"], required=True)
    parser.add_argument("--symbols", default="", help="comma-separated override; default = configs/universe.yaml")
    parser.add_argument("--start", default="2018-01-01", help="historic phase only")
    parser.add_argument("--end", default=None, help="historic phase only; default = today")
    parser.add_argument("--force", action="store_true", help="historic phase only: re-fetch already-done weeks")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or _universe_symbols()
    if not symbols:
        print("ERROR: no symbols to backfill (configs/universe.yaml empty and --symbols not given)")
        sys.exit(1)
    cache_dir = Path(args.cache_dir)

    if args.phase == "recent":
        print(f"Fetching FINRA weeklySummary (recent, ~3-4yr rolling window) for {len(symbols)} symbols...")
        run_recent(symbols, cache_dir)
    else:
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
        start = pd.Timestamp(args.start)
        n_weeks = int((end - start).days / 7) + 1
        print(f"Fetching FINRA weeklySummaryHistoric for {len(symbols)} symbols, "
              f"{start.date()} .. {end.date()} (~{n_weeks} weekly market-wide calls, resumable).")
        run_historic(symbols, start, end, cache_dir, args.force)


if __name__ == "__main__":
    main()
