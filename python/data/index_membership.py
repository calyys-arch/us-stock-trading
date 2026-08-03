"""
Shared point-in-time index-membership reconstruction algorithm, used by both
sp500_universe.py and nasdaq100_universe.py.

Method (Chan's explicit survivorship-bias warning — see either caller
module's docstring): start from the CURRENT constituent list and walk
BACKWARD in time undoing each membership change — an addition on date D
means the added ticker was NOT a member before D; a removal on date D means
the removed ticker WAS a member before D.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


def point_in_time_membership(
    as_of: datetime,
    current_constituents: pd.DataFrame,
    changes: pd.DataFrame,
) -> set[str]:
    """`current_constituents` needs a `symbol` column; `changes` needs
    `date`, `added_ticker`, `removed_ticker` columns."""
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
    current_constituents: pd.DataFrame,
    changes: pd.DataFrame,
) -> dict:
    return {d: sorted(point_in_time_membership(d, current_constituents, changes)) for d in dates}
