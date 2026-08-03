"""
Point-in-time Nasdaq-100 universe construction — same rationale and same
survivorship-bias discipline as sp500_universe.py, mirrored for the
Nasdaq-100 index.

Why this exists (user-confirmed requirement): restricting the tradeable
universe to the S&P 500 alone systematically excludes many of the most
actively-traded, retail-popular US equities. The S&P Index Committee applies
a profitability screen and a discretionary review before adding a name —
several huge-volume Nasdaq-listed companies sit outside the S&P 500 for
years after becoming liquid, heavily-traded names (recent mega-cap IPOs,
newly-profitable growth companies, etc.). Nasdaq-100 membership is unioned
with S&P 500 in python/data/liquid_universe.py specifically to close this
gap, then narrowed further by ACTUAL trailing trading volume (not index
membership at all) so the final tradeable list tracks "where the volume
really is" rather than either index's committee-driven inclusion rules.

Method: identical walk-backward-from-current-list algorithm as
sp500_universe.py (shared implementation in index_membership.py), using
Wikipedia's "List of NASDAQ-100 companies" current constituent table and its
"Changes in the composition" history table.

Same explicit limitation applies as sp500_universe.py: this is NOT a
professional point-in-time database (see README.md "Known limitations
(MVP)") — Wikipedia's changes table coverage/completeness is not guaranteed
for the entire history.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import index_membership
from .wiki_fetch import fetch_wiki_tables

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def fetch_current_constituents() -> pd.DataFrame:
    """Returns a DataFrame with columns [symbol, security, sector,
    sub_industry] for the CURRENT Nasdaq-100 constituents."""
    tables = fetch_wiki_tables(_WIKI_URL)
    current = tables[0]
    current = current.rename(columns={
        "Ticker": "symbol", "Company": "security",
        "ICB Industry[1]": "sector", "ICB Subsector[1]": "sub_industry",
    })
    current["symbol"] = current["symbol"].astype(str).str.replace(".", "-", regex=False)  # yfinance convention
    keep = [c for c in ("symbol", "security", "sector", "sub_industry") if c in current.columns]
    return current[keep].drop_duplicates(subset="symbol").reset_index(drop=True)


def fetch_membership_changes() -> pd.DataFrame:
    """Returns a DataFrame with columns [date, added_ticker, removed_ticker]
    parsed from the "Changes in the composition" table. That table has
    MultiIndex columns (Date / Added-Ticker / Added-Security / Removed-Ticker
    / Removed-Security / Reason) as rendered by Wikipedia, unlike the S&P 500
    page's flat header — flattened here to match sp500_universe.py's shape."""
    tables = fetch_wiki_tables(_WIKI_URL)
    changes = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("Added" in c for c in cols) and any("Removed" in c for c in cols):
            changes = t
            break
    if changes is None:
        raise RuntimeError("nasdaq100_universe: could not locate the membership-changes table on Wikipedia")

    changes.columns = ["date", "added_ticker", "added_security", "removed_ticker", "removed_security", "reason"]
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"])
    changes["added_ticker"] = changes["added_ticker"].astype(str).str.replace(".", "-", regex=False)
    changes["removed_ticker"] = changes["removed_ticker"].astype(str).str.replace(".", "-", regex=False)
    return changes[["date", "added_ticker", "removed_ticker"]]


def nasdaq100_point_in_time_membership(
    as_of: datetime,
    current_constituents: pd.DataFrame | None = None,
    changes: pd.DataFrame | None = None,
) -> set[str]:
    """Reconstruct Nasdaq-100 membership as of `as_of`, mirroring
    sp500_universe.sp500_point_in_time_membership. Named `nasdaq100_...` to
    stay unambiguous alongside that module's same-shaped wrapper and
    index_membership.py's shared generic implementation."""
    current_constituents = current_constituents if current_constituents is not None else fetch_current_constituents()
    changes = changes if changes is not None else fetch_membership_changes()
    return index_membership.point_in_time_membership(as_of, current_constituents, changes)


def universe_by_day(
    dates: list[datetime],
    current_constituents: pd.DataFrame | None = None,
    changes: pd.DataFrame | None = None,
) -> dict:
    current_constituents = current_constituents if current_constituents is not None else fetch_current_constituents()
    changes = changes if changes is not None else fetch_membership_changes()
    return index_membership.universe_by_day(dates, current_constituents, changes)
