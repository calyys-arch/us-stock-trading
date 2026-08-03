"""
Tests for python/data/index_membership.py and python/data/liquid_universe.py
— added when the tradeable universe was expanded from S&P-500-only to
(S&P 500 UNION Nasdaq-100) narrowed by trailing dollar volume (2026-07-28).

All tests here use synthetic in-memory DataFrames (no network) — the actual
Wikipedia-fetching functions (sp500_universe.fetch_current_constituents,
nasdaq100_universe.fetch_current_constituents, etc.) are exercised manually/
in scripts, not in the offline test suite, matching this repo's existing
convention (sp500_universe.py itself has no network-dependent unit tests).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from python.data import index_membership
from python.data.liquid_universe import top_by_trailing_dollar_volume


def _constituents(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols})


def _changes(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: list of (date_str, added_ticker, removed_ticker); use '' for
    whichever side didn't change."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "added_ticker": a, "removed_ticker": r} for d, a, r in rows]
    )


def test_point_in_time_membership_undoes_future_addition():
    """A ticker added AFTER as_of must not appear in that as_of's membership."""
    current = _constituents(["AAPL", "MSFT", "NEWCO"])
    changes = _changes([("2026-06-01", "NEWCO", "")])  # NEWCO added on 2026-06-01

    before = index_membership.point_in_time_membership(datetime(2026, 1, 1), current, changes)
    after = index_membership.point_in_time_membership(datetime(2026, 7, 1), current, changes)

    assert "NEWCO" not in before
    assert "NEWCO" in after


def test_point_in_time_membership_restores_future_removal():
    """A ticker removed AFTER as_of must still appear in that as_of's
    membership (it hadn't been removed yet)."""
    current = _constituents(["AAPL", "MSFT"])  # OLDCO no longer in the current list
    changes = _changes([("2026-06-01", "", "OLDCO")])  # OLDCO removed on 2026-06-01

    before = index_membership.point_in_time_membership(datetime(2026, 1, 1), current, changes)
    after = index_membership.point_in_time_membership(datetime(2026, 7, 1), current, changes)

    assert "OLDCO" in before
    assert "OLDCO" not in after


def _volume_panel(volumes: dict[str, list[float]], dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for code, vols in volumes.items():
        for d, v in zip(dates, vols):
            rows.append({"date": d, "code": code, "close": 100.0, "volume": v})
    return pd.DataFrame(rows).set_index(["date", "code"]).sort_index()


def test_top_by_trailing_dollar_volume_ranks_higher_volume_first():
    dates = pd.bdate_range("2026-01-02", periods=10)
    panel = _volume_panel({"HIGH": [1_000_000] * 10, "LOW": [10_000] * 10, "MID": [100_000] * 10}, dates)

    ranked = top_by_trailing_dollar_volume(
        candidates=["HIGH", "LOW", "MID"], price_panel=panel,
        as_of=dates[-1] + pd.Timedelta(days=1), lookback_days=20, top_k=3,
    )
    assert ranked == ["HIGH", "MID", "LOW"]


def test_top_by_trailing_dollar_volume_ignores_future_volume_spike():
    """Corrupting volume ON/AFTER as_of must not change the ranking as of
    an earlier as_of — same look-ahead-bias contract as
    tests/test_lookahead_bias.py's strategy-level checks."""
    dates = pd.bdate_range("2026-01-02", periods=20)
    as_of = dates[10]

    panel = _volume_panel(
        {"A": [500_000] * 20, "B": [400_000] * 20},
        dates,
    )
    baseline = top_by_trailing_dollar_volume(["A", "B"], panel, as_of, lookback_days=5, top_k=2)

    corrupted = panel.copy()
    future_mask = corrupted.index.get_level_values(0) >= as_of
    corrupted.loc[future_mask, "volume"] = 999_999_999  # B "becomes" huge, but only in the future

    mutated = top_by_trailing_dollar_volume(["A", "B"], corrupted, as_of, lookback_days=5, top_k=2)

    assert baseline == mutated == ["A", "B"]


def test_top_by_trailing_dollar_volume_handles_empty_history():
    dates = pd.bdate_range("2026-01-02", periods=3)
    panel = _volume_panel({"A": [1.0, 1.0, 1.0]}, dates)
    ranked = top_by_trailing_dollar_volume(["A", "B"], panel, as_of=dates[0], lookback_days=5, top_k=5)
    # as_of is the very first date in the panel -> no prior history at all
    assert ranked == ["A", "B"]
