"""
Track 1 (alt_universe_frequency_exploration): construct a point-in-time,
liquidity-BANDED mid-cap-ish candidate universe — deliberately one liquidity
tier BELOW configs/universe.yaml's fixed top-20 mega-cap list, to test
whether a wider-spread / lower-HFT-competition universe retains more edge
than it costs (see backtests/reports/strategy_review_summary.md's
xsection_mean_reversion diagnosis).

Point-in-time discipline (the exact bias pairs_scan_report.md's round one
caught and fixed for configs/universe.yaml/pairs_trading):
  1. Candidate pool = S&P 500 membership AS OF `AS_OF_DATE` (2016-11-01).
  2. Liquidity ranking uses ONLY a trailing 60-trading-day dollar-volume
     window ending STRICTLY BEFORE AS_OF_DATE (python/data/liquid_universe.py
     band_by_trailing_dollar_volume) — mechanical, no discretion, no look-
     ahead into any later date.
  3. The selected band is then held FIXED across the whole backtest window
     (same "one snapshot, applied forward in time, honestly dated" pattern
     as configs/universe.yaml — but dated at the START of the backtest
     rather than "today, applied backward", which corrects the exact
     survivorship caveat fixed_universe.py's own docstring flags).

DATA-AVAILABILITY FINDING (2026-08-13, see the report's data-availability
section): python/data/sp500_universe.py's walk-backward-from-current-list
algorithm depends on a "Selected changes to the components" table that no
longer exists on the LIVE Wikipedia page (confirmed: fetch_wiki_tables now
returns only 2 tables there — current constituents + a navbox, no changes
table). sp500_universe.py is left completely UNCHANGED (not this
exploration's job to fix production code; it may work again if Wikipedia
restores the table, or for other as_of dates). Instead, step 1 above is done
by fetching the Wikipedia ARTICLE REVISION that was live at/before
AS_OF_DATE via the MediaWiki Action API and parsing THAT revision's own
constituent table directly — arguably a MORE direct point-in-time method
than walking back a changes log (it is literally what the page said on that
date), and it needs no changes table at all.

Liquidity band: ranks 150-220 (up to 70 names) of the ~500-name S&P 500 pool
by trailing dollar volume. Rationale: excludes the top ~150 (mega/large-cap
names that overlap heavily with configs/universe.yaml's universe and the
microstructure signals' mega-cap universe — the segment already shown to be
heavily arbitraged), while stopping well short of the S&P 500's illiquid
tail (ranks 450+), where the spread-cost estimate itself would be unreliable
and this strategy's own $50k-notional-per-name order size could plausibly
move the market (the user's explicit "avoid microcaps" constraint). S&P 500
membership itself already screens out true micro-caps (min market cap
requirement), so even the bottom of this pool is a real, continuously-listed
company — not an illiquid curiosity.

Usage:
    python scripts/_altuni_build_universe.py
Writes configs/alt_universe_midcap.yaml and prints the trailing-dollar-volume
distribution for the report.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from python.data.liquid_universe import band_by_trailing_dollar_volume, rank_by_trailing_dollar_volume
from python.data.price_cache import get_cached_price_panel
from python.data.wiki_fetch import _REQUEST_HEADERS, fetch_wiki_tables

AS_OF_DATE = datetime(2016, 11, 1)
RANK_WINDOW_START = "2016-07-15"   # >= 60 trading days before AS_OF_DATE
RANK_WINDOW_END = "2016-11-01"
RANK_LOOKBACK_DAYS = 60
BAND_START_RANK = 150
BAND_END_RANK = 220
OUT_PATH = Path("configs/alt_universe_midcap.yaml")


def _wikipedia_revision_id_before(title: str, as_of: datetime) -> int:
    """Latest revision id at or before `as_of` (MediaWiki Action API)."""
    rvstart = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
        f"&titles={title}&rvlimit=1&rvprop=ids|timestamp&rvstart={rvstart}"
        "&rvdir=older&format=json"
    )
    req = urllib.request.Request(url, headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    revisions = page.get("revisions")
    if not revisions:
        raise RuntimeError(f"no Wikipedia revision of {title!r} found at/before {as_of}")
    return int(revisions[0]["revid"])


def sp500_members_asof_via_revision(as_of: datetime) -> tuple[list[str], int, str]:
    """Point-in-time S&P 500 tickers by fetching the Wikipedia REVISION live
    at/before `as_of` and parsing its own constituent table (whatever that
    revision's column name for the ticker happened to be). Returns
    (tickers, revision_id, revision_url) for provenance logging."""
    title = "List_of_S%26P_500_companies"
    revid = _wikipedia_revision_id_before(title, as_of)
    url = f"https://en.wikipedia.org/w/index.php?title={title}&oldid={revid}"
    tables = fetch_wiki_tables(url)
    const_table = tables[0]
    ticker_col = next(
        c for c in const_table.columns
        if "ticker" in str(c).lower() or "symbol" in str(c).lower()
    )
    tickers = (
        const_table[ticker_col].astype(str).str.strip()
        .str.replace(".", "-", regex=False)  # yfinance convention (BRK.B -> BRK-B)
        .tolist()
    )
    tickers = sorted({t for t in tickers if t and t.lower() != "nan"})
    return tickers, revid, url


def main() -> None:
    print(f"Fetching S&P 500 point-in-time membership as of {AS_OF_DATE.date()} "
          "via historical Wikipedia revision ...")
    members, revid, revurl = sp500_members_asof_via_revision(AS_OF_DATE)
    print(f"  {len(members)} point-in-time S&P 500 members as of {AS_OF_DATE.date()} "
          f"(Wikipedia revision {revid}: {revurl})")

    print(f"Fetching trailing {RANK_LOOKBACK_DAYS}-day price/volume window "
          f"[{RANK_WINDOW_START}, {RANK_WINDOW_END}) for ranking "
          f"({len(members)} symbols, this is the slow step)...")
    panel, quality_flags, meta = get_cached_price_panel(
        members, RANK_WINDOW_START, RANK_WINDOW_END,
        cache_dir="data/history_altuni_rank",
    )
    fetched_codes = sorted(panel.index.get_level_values(1).unique())
    missing = sorted(set(members) - set(fetched_codes))
    print(f"  {len(fetched_codes)}/{len(members)} symbols fetched "
          f"(source={meta.get('fetched_source')}); "
          f"{len(missing)} missing: {missing[:15]}{'...' if len(missing) > 15 else ''}")

    full_rank = rank_by_trailing_dollar_volume(
        fetched_codes, panel, AS_OF_DATE, lookback_days=RANK_LOOKBACK_DAYS)
    print(f"\nTrailing {RANK_LOOKBACK_DAYS}d dollar-volume distribution "
          f"({len(full_rank)} ranked names):")
    for rank in (1, 20, 50, 100, 150, 175, 200, 220, 250, 300, 400, len(full_rank)):
        if rank <= len(full_rank):
            code = full_rank.index[rank - 1]
            print(f"  rank {rank:4d}: {code:6s} ${full_rank.iloc[rank - 1]:>15,.0f}/day")

    band = band_by_trailing_dollar_volume(
        fetched_codes, panel, AS_OF_DATE,
        band_start_rank=BAND_START_RANK, band_end_rank=BAND_END_RANK,
        lookback_days=RANK_LOOKBACK_DAYS,
    )
    band_dv = full_rank.loc[band]
    print(f"\nSelected band [{BAND_START_RANK}, {BAND_END_RANK}): {len(band)} symbols")
    print(f"  dollar-volume range in band: ${band_dv.min():,.0f} .. ${band_dv.max():,.0f} /day")
    print(f"  symbols: {sorted(band)}")

    doc = {
        "alt_universe_midcap": {
            "symbols": sorted(band),
            "n_symbols": len(band),
            "method": (
                "Point-in-time S&P 500 membership via the Wikipedia article REVISION "
                f"live at/before {AS_OF_DATE.date()} (revision {revid}, {revurl}) — NOT "
                "today's constituents applied backward, and NOT sp500_universe.py's "
                "walk-backward-from-current-list algorithm (its 'Selected changes' table "
                "no longer exists on the live Wikipedia page as of this exploration; see "
                "scripts/_altuni_build_universe.py's module docstring). Ranked by trailing "
                f"{RANK_LOOKBACK_DAYS}-trading-day mean dollar volume over "
                f"[{RANK_WINDOW_START}, {AS_OF_DATE.date()}) (strictly before as_of, no "
                f"look-ahead), liquidity BAND ranks [{BAND_START_RANK}, {BAND_END_RANK}) "
                "selected (excludes the most-liquid top tier, which overlaps "
                "configs/universe.yaml's mega-cap universe, and the illiquid tail)."
            ),
            "as_of": str(AS_OF_DATE.date()),
            "wikipedia_revision_id": revid,
            "wikipedia_revision_url": revurl,
            "band_start_rank": BAND_START_RANK,
            "band_end_rank": BAND_END_RANK,
            "rank_lookback_days": RANK_LOOKBACK_DAYS,
            "dollar_volume_range_usd_per_day": [float(band_dv.min()), float(band_dv.max())],
            "caveat": (
                "NOT a true small-cap/Russell-2000 universe — this codebase has no "
                "point-in-time Russell 2000 membership source, and building one reliably "
                "was out of scope for this exploration (see "
                "backtests/reports/alt_universe_frequency_exploration.md). This is a "
                "liquidity tier ONE STEP DOWN from the existing mega-cap universe, still "
                "drawn from S&P 500 membership (which itself screens out true micro-caps), "
                "not a genuinely small-cap universe."
            ),
        }
    }
    header = (
        "# Alt liquidity-band universe for the alt-universe/frequency exploration\n"
        "# (backtests/reports/alt_universe_frequency_exploration.md). Generated by\n"
        "# scripts/_altuni_build_universe.py. Point-in-time constructed — see 'method'\n"
        "# below and the report for the full survivorship-bias discussion. This is a\n"
        "# research artifact, NOT configs/universe.yaml — nothing reads this file by\n"
        "# default and no production path is affected by it.\n"
    )
    OUT_PATH.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
