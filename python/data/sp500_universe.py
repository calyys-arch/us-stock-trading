"""
Point-in-time S&P 500 universe construction — addresses Chan's explicit
survivorship-bias warning (backtesting on TODAY's S&P 500 constituents back
through history inflates results because it omits every company that was
removed for underperforming/bankruptcy/delisting).

Method: Wikipedia's "List of S&P 500 companies" page publishes both (a) the
CURRENT constituent table and (b) a "Selected changes to the components"
table of historical additions/removals with dates. Starting from the
current list, we walk BACKWARD in time undoing each change (an addition on
date D means the added ticker was NOT a member before D; a removal on date D
means the removed ticker WAS a member before D) to reconstruct membership as
of any past date.

Explicit limitation (documented in README.md "Known limitations (MVP)"):
this is NOT a professional point-in-time database. Wikipedia's changes
table is not guaranteed complete for the entire history (coverage is best
from ~2015 onward) and ticker symbols can be renamed/re-used. Treat this as
a "much better than naive current-constituents-only" approximation, not a
CRSP-grade universe. `docs/us_equity_health_check.md` must carry a
disclaimer banner referencing this limitation whenever this module is used.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_current_constituents() -> pd.DataFrame:
    """Returns a DataFrame with columns [symbol, security, sector,
    sub_industry] for the CURRENT S&P 500 constituents."""
    tables = pd.read_html(_WIKI_URL)
    current = tables[0]
    current = current.rename(columns={
        "Symbol": "symbol", "Security": "security",
        "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry",
    })
    current["symbol"] = current["symbol"].str.replace(".", "-", regex=False)  # yfinance convention
    return current[["symbol", "security", "sector", "sub_industry"]]


def fetch_membership_changes() -> pd.DataFrame:
    """Returns a DataFrame with columns [date, added, removed] parsed from
    the "Selected changes to the components" table."""
    tables = pd.read_html(_WIKI_URL)
    changes = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("Added" in c for c in cols) and any("Removed" in c for c in cols):
            changes = t
            break
    if changes is None:
        raise RuntimeError("sp500_universe: could not locate the membership-changes table on Wikipedia")

    changes.columns = ["date", "added_ticker", "added_security", "removed_ticker", "removed_security", "reason"]
    changes = changes.iloc[1:] if changes.iloc[0]["date"] == "Date" else changes
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"])
    changes["added_ticker"] = changes["added_ticker"].astype(str).str.replace(".", "-", regex=False)
    changes["removed_ticker"] = changes["removed_ticker"].astype(str).str.replace(".", "-", regex=False)
    return changes[["date", "added_ticker", "removed_ticker"]]


def point_in_time_membership(
    as_of: datetime,
    current_constituents: pd.DataFrame | None = None,
    changes: pd.DataFrame | None = None,
) -> set[str]:
    """Reconstruct S&P 500 membership as of `as_of` by undoing every
    membership change that occurred AFTER `as_of`, walking backward from the
    current constituent list."""
    current_constituents = current_constituents if current_constituents is not None else fetch_current_constituents()
    changes = changes if changes is not None else fetch_membership_changes()

    membership = set(current_constituents["symbol"].tolist())

    later_changes = changes[changes["date"] > pd.Timestamp(as_of)].sort_values("date", ascending=False)
    for _, row in later_changes.iterrows():
        added = row["added_ticker"]
        removed = row["removed_ticker"]
        if added and added != "nan" and added in membership:
            membership.discard(added)   # wasn't a member before this addition
        if removed and removed != "nan":
            membership.add(removed)     # WAS a member before this removal

    return membership


def universe_by_day(
    dates: list[datetime],
    current_constituents: pd.DataFrame | None = None,
    changes: pd.DataFrame | None = None,
) -> dict:
    """Convenience wrapper for backtest/vector_engine.run_vector_backtest's
    `universe_by_day` argument — computes point-in-time membership for every
    date in `dates` (caches the Wikipedia tables so they are fetched once)."""
    current_constituents = current_constituents if current_constituents is not None else fetch_current_constituents()
    changes = changes if changes is not None else fetch_membership_changes()

    result = {}
    for d in dates:
        result[d] = sorted(point_in_time_membership(d, current_constituents, changes))
    return result
