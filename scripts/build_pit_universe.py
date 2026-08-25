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
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data/history_pit2016"
OUT_JSON = ROOT / "backtests/reports/pit_universe_coverage.json"
# Which symbols a fetch actually asked for. Without this, a missing CSV is
# ambiguous -- never requested, or requested and genuinely not available -- and
# an interrupted fetch would be read as catastrophic survivorship attrition.
ATTEMPTED_JSON = CACHE / "_attempted.json"

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


def _record_attempted(symbols: list[str]) -> None:
    """Union this request into the attempted set, so an interrupted fetch does
    not erase what an earlier probe already established."""
    prior = _attempted()
    ATTEMPTED_JSON.write_text(
        json.dumps(sorted(set(symbols) | prior), indent=0), encoding="utf-8")


def _attempted() -> set[str]:
    if not ATTEMPTED_JSON.exists():
        return set()
    try:
        return set(json.loads(ATTEMPTED_JSON.read_text()))
    except (ValueError, TypeError):
        return set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="actually download; otherwise only classify the cache")
    ap.add_argument("--limit", type=int, default=0,
                    help="fetch only the first N symbols (for a quick probe)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the coverage report even if the existing "
                         "one covers more symbols than this run")
    args = ap.parse_args()

    members, revid, revurl = _membership(AS_OF)
    print(f"S&P 500 成分 @ {AS_OF.date()}: {len(members)} 檔 "
          f"(Wikipedia revision {revid})")
    print(f"  {revurl}\n")

    symbols = members[:args.limit] if args.limit else members

    if args.fetch:
        CACHE.mkdir(parents=True, exist_ok=True)
        _record_attempted(symbols)
        from python.data.price_cache import get_cached_price_panel
        print(f"抓取 {len(symbols)} 檔 {START} -> {END} 到 {CACHE} ...",
              flush=True)
        # Returns (panel, quality_flags, source_map); the middle element is a
        # per-symbol quality dict, not a source label, and printing it dumps
        # thousands of lines.
        try:
            panel, _quality, _sources = get_cached_price_panel(
                symbols, START, END, cache_dir=CACHE)
            print(f"  面板 {panel.shape}\n", flush=True)
        except RuntimeError as exc:
            # Every symbol still needing a fetch came back empty, which is what
            # a batch of delisted tickers looks like. Confirming absence is the
            # measurement here, not a failure, so classification continues on
            # whatever the cache already holds.
            print(f"  這批沒有任何標的可抓（{exc}）。"
                  f"對已下市代號而言這就是結果，繼續分類快取。\n", flush=True)

    attempted = _attempted()
    buckets: dict[str, list] = {"full": [], "truncated": [], "missing": [],
                               "not_attempted": []}
    details = {}
    for symbol in symbols:
        path = CACHE / f"{symbol}.csv"
        if not path.exists():
            # A missing file only means "gone" if the fetch actually asked for
            # it. Otherwise it means "not yet tried", and counting the two
            # together turns an interrupted download into a 96% attrition
            # claim.
            kind = "missing" if symbol in attempted else "not_attempted"
            buckets[kind].append(symbol)
            details[symbol] = {
                "rows": 0,
                "note": ("requested but nothing returned" if kind == "missing"
                         else "never requested — says nothing about the symbol"),
            }
            continue
        kind, info = _classify(path)
        buckets[kind].append(symbol)
        details[symbol] = info

    total = len(symbols)
    n_tried = total - len(buckets["not_attempted"])
    print("== 覆蓋率 ==")
    for kind in ("full", "truncated", "missing", "not_attempted"):
        n = len(buckets[kind])
        print(f"  {kind:14s} {n:4d} 檔  ({n/total:5.1%} of members)")

    usable = len(buckets["full"]) + len(buckets["truncated"])
    if not n_tried:
        print("\n  尚未抓取任何資料，無法談流失率。先跑 --fetch。")
    else:
        print(f"\n  以「實際請求過的 {n_tried} 檔」為分母：")
        print(f"    可用於點入時間回測: {usable} 檔 ({usable/n_tried:.1%})")
        print(f"    確認消失（倖存者偏誤缺口）: {len(buckets['missing'])} 檔 "
              f"({len(buckets['missing'])/n_tried:.1%})")
    if buckets["not_attempted"]:
        print(f"\n  !! {len(buckets['not_attempted'])} 檔從未請求過，"
              f"不計入流失率。上面的比率只描述已抓取的子集，"
              f"不是這 {total} 檔名單的倖存率。")
        print(f"     跑 --fetch 補齊後這個警告才會消失。")

    if buckets["truncated"]:
        print(f"\n  truncated 抽樣（退市／併購，帶真實歷史）:")
        for s in sorted(buckets["truncated"])[:12]:
            d = details[s]
            print(f"    {s:6s} {d.get('first')} -> {d.get('last')}  "
                  f"{d.get('rows')} 列")

    payload = {
        "run_at": datetime.now(UTC).isoformat(),
        "as_of": str(AS_OF.date()),
        "wikipedia_revision_id": revid,
        "wikipedia_revision_url": revurl,
        "window": [START, END],
        "n_members": total,
        "n_attempted": n_tried,
        "counts": {k: len(v) for k, v in buckets.items()},
        "usable_for_pit_backtest": usable,
        # Guard against this file being read as a survivorship estimate when
        # most of the list was never downloaded.
        "coverage_is_representative": not buckets["not_attempted"],
        "attrition_rate_of_attempted": (
            len(buckets["missing"]) / n_tried if n_tried else None),
        "buckets": buckets,
        "details": details,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # A classify-only run over a half-filled cache is strictly less informative
    # than an earlier run that actually requested more symbols. Overwriting
    # silently already destroyed one probe's result; refuse rather than repeat
    # it. backtests/ is gitignored, so there is no history to recover from.
    prior_attempted = 0
    if OUT_JSON.exists():
        try:
            prior_attempted = int(json.loads(OUT_JSON.read_text())
                                  .get("n_attempted") or 0)
        except (ValueError, TypeError):
            prior_attempted = 0
    if prior_attempted > n_tried and not args.force:
        print(f"\n拒絕覆寫 {OUT_JSON.name}：既有紀錄請求過 {prior_attempted} 檔，"
              f"本次只有 {n_tried} 檔。加 --force 才會覆寫。")
        return 1

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
