"""
Fetch a point-in-time 2016 index membership and measure what is missing.

Why. Cross-sectional momentum measured a beta of +0.016 -- genuine market
neutrality, the first non-beta return stream in this project -- and no edge,
on a median of 12 positions. Six names long and six short is stock picking,
not a factor, and the cause is that only ~33 names in data/history have
pre-2024 history. The factor needs hundreds of names.

The obstacle everyone waves at is survivorship, and this script exists to stop
waving. `scripts/_altuni_build_universe.py` can pull the Wikipedia REVISION
that was live at a past date, which yields the index as it ACTUALLY STOOD then
-- 504 tickers at 2016-01-01, including names like ACE (acquired by Chubb weeks
later) that no present-day list contains. Membership is therefore already
solved in this repo. What is not solved is whether PRICES exist for the names
that later vanished.

So the measurement that matters is not "how many of the 504 have ten years of
data". A name acquired in 2019 SHOULD stop in 2019; that is correct data, not
missing data, and a backtest where names drop out when they delist is exactly
what removes survivorship bias. The question is whether the vanished names
come back with history up to their disappearance, or with nothing at all.
Three outcomes are counted separately:

  full      data through the end of the window: survived, or at least is still
            listed. These are the only names a naive present-day universe
            would have given us.
  truncated data that starts on time and stops early. THE VALUABLE CASE: a
            delisting or acquisition captured with its real history, usable in
            a point-in-time backtest.
  missing   no data at all, or a start date well after the window opens. These
            are the irreducible hole, and their share is the honest size of
            the survivorship problem that remains.

Nothing here is a backtest. It fetches, caches, and counts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data/history_pit2016"
OUT_JSON = ROOT / "backtests/reports/pit_universe_coverage.json"

AS_OF = datetime(2016, 1, 1)
START, END = "2016-01-01", "2026-08-01"
# A name whose data starts more than this far after the window opens was not
# really covered, whatever it returned.
LATE_START_DAYS = 180
# A name still reporting within this margin of the window's end counts as full.
EARLY_STOP_DAYS = 90


def _membership(as_of: datetime) -> tuple[list[str], int, str]:
    spec = importlib.util.spec_from_file_location(
        "_altuni", ROOT / "scripts/_altuni_build_universe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sp500_members_asof_via_revision(as_of)


def _classify(path: Path) -> tuple[str, dict]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or len(df) == 0:
        return "missing", {"rows": 0}
    ix = pd.to_datetime(df[cols["date"]]).sort_values()
    first, last = ix.iloc[0], ix.iloc[-1]
    info = {"rows": int(len(df)), "first": str(first.date()),
            "last": str(last.date())}
    if (first - pd.Timestamp(START)).days > LATE_START_DAYS:
        return "missing", info
    if (pd.Timestamp(END) - last).days > EARLY_STOP_DAYS:
        return "truncated", info
    return "full", info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="actually download; otherwise only classify the cache")
    ap.add_argument("--limit", type=int, default=0,
                    help="fetch only the first N symbols (for a quick probe)")
    args = ap.parse_args()

    members, revid, revurl = _membership(AS_OF)
    print(f"S&P 500 成分 @ {AS_OF.date()}: {len(members)} 檔 "
          f"(Wikipedia revision {revid})")
    print(f"  {revurl}\n")

    symbols = members[:args.limit] if args.limit else members

    if args.fetch:
        CACHE.mkdir(parents=True, exist_ok=True)
        from python.data.price_cache import get_cached_price_panel
        print(f"抓取 {len(symbols)} 檔 {START} -> {END} 到 {CACHE} ...",
              flush=True)
        # Returns (panel, quality_flags, source_map); the middle element is a
        # per-symbol quality dict, not a source label, and printing it dumps
        # thousands of lines.
        panel, _quality, _sources = get_cached_price_panel(
            symbols, START, END, cache_dir=CACHE)
        print(f"  面板 {panel.shape}\n", flush=True)

    buckets: dict[str, list] = {"full": [], "truncated": [], "missing": []}
    details = {}
    for symbol in symbols:
        path = CACHE / f"{symbol}.csv"
        if not path.exists():
            buckets["missing"].append(symbol)
            details[symbol] = {"rows": 0, "note": "no cache file"}
            continue
        kind, info = _classify(path)
        buckets[kind].append(symbol)
        details[symbol] = info

    total = len(symbols)
    print("== 覆蓋率 ==")
    for kind in ("full", "truncated", "missing"):
        n = len(buckets[kind])
        print(f"  {kind:10s} {n:4d} 檔  ({n/total:5.1%})")
    usable = len(buckets["full"]) + len(buckets["truncated"])
    print(f"\n  可用於點入時間回測: {usable} 檔 ({usable/total:.1%})")
    print(f"  無法補救的缺口:     {len(buckets['missing'])} 檔 "
          f"({len(buckets['missing'])/total:.1%})")

    if buckets["truncated"]:
        print(f"\n  truncated 抽樣（退市／併購，帶真實歷史）:")
        for s in sorted(buckets["truncated"])[:12]:
            d = details[s]
            print(f"    {s:6s} {d.get('first')} -> {d.get('last')}  "
                  f"{d.get('rows')} 列")

    payload = {
        "run_at": datetime.utcnow().isoformat(),
        "as_of": str(AS_OF.date()),
        "wikipedia_revision_id": revid,
        "wikipedia_revision_url": revurl,
        "window": [START, END],
        "n_members": total,
        "counts": {k: len(v) for k, v in buckets.items()},
        "usable_for_pit_backtest": usable,
        "buckets": buckets,
        "details": details,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
