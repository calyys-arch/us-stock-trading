"""
price_cache tests — no network: _fetch_remote is monkeypatched with a fake
that records calls and returns a synthetic panel.

Coverage:
  - first request fetches and caches; identical second request is served
    entirely from disk (zero fetch calls);
  - a WIDER date range triggers a re-fetch (requested-range coverage check);
  - a NARROWER sub-range is a cache hit even for a symbol whose first bar
    is after the sub-range start (IPO case — requested-range, not
    bar-range, comparison);
  - refresh=True forces a re-fetch;
  - panel output contract (MultiIndex, adv_20d_dollars present, slice
    bounds respected) and meta source labeling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.data import price_cache


def _fake_panel(symbols: list[str], start, end, first_date_override: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for code in symbols:
        dates = pd.bdate_range(start, end)
        if first_date_override and code in first_date_override:
            dates = dates[dates >= pd.Timestamp(first_date_override[code])]
        close = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        df = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1e6,
        }, index=dates)
        df.index.name = "date"
        df["adv_20d_dollars"] = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean()
        df["code"] = code
        frames.append(df.reset_index().set_index(["date", "code"]))
    return pd.concat(frames).sort_index()


@pytest.fixture
def fake_fetch(monkeypatch):
    calls: list[dict] = []

    def _fetch(symbols, start, end, broker_config_path):
        calls.append({"symbols": list(symbols), "start": start, "end": end})
        return _fake_panel(symbols, start, end, first_date_override={"LATEIPO": "2020-06-01"}), "fake_source"

    monkeypatch.setattr(price_cache, "_fetch_remote", _fetch)
    return calls


def test_fetch_then_cache_hit(tmp_path, fake_fetch):
    panel1, flags1, meta1 = price_cache.get_cached_price_panel(
        ["AAA", "BBB"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    assert len(fake_fetch) == 1
    assert meta1["fetched"] == ["AAA", "BBB"]
    assert meta1["sources"] == {"fake_source": ["AAA", "BBB"]}

    panel2, _flags2, meta2 = price_cache.get_cached_price_panel(
        ["AAA", "BBB"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    assert len(fake_fetch) == 1, "second identical request must not fetch"
    assert meta2["fetched"] == []
    assert meta2["from_cache"] == ["AAA", "BBB"]
    pd.testing.assert_frame_equal(panel1.sort_index(), panel2.sort_index())


def test_wider_range_refetches_narrower_hits_cache(tmp_path, fake_fetch):
    price_cache.get_cached_price_panel(["AAA"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    assert len(fake_fetch) == 1

    # narrower sub-range -> cache hit
    price_cache.get_cached_price_panel(["AAA"], "2020-03-01", "2020-06-30", cache_dir=tmp_path)
    assert len(fake_fetch) == 1

    # wider range -> must re-fetch
    price_cache.get_cached_price_panel(["AAA"], "2019-01-01", "2020-12-31", cache_dir=tmp_path)
    assert len(fake_fetch) == 2


def test_late_ipo_subrange_is_still_a_cache_hit(tmp_path, fake_fetch):
    """LATEIPO has no bars before 2020-06-01. After caching the full-year
    request, asking for [2020-01-01, 2020-12-31] again (which includes the
    bar-less early months) must NOT re-fetch — coverage compares the
    REQUESTED range, not the first available bar."""
    price_cache.get_cached_price_panel(["LATEIPO"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    assert len(fake_fetch) == 1

    panel, _flags, meta = price_cache.get_cached_price_panel(
        ["LATEIPO"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    assert len(fake_fetch) == 1
    assert meta["from_cache"] == ["LATEIPO"]
    assert panel.index.get_level_values(0).min() >= pd.Timestamp("2020-06-01")


def test_refresh_forces_refetch(tmp_path, fake_fetch):
    price_cache.get_cached_price_panel(["AAA"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    price_cache.get_cached_price_panel(["AAA"], "2020-01-01", "2020-12-31",
                                       cache_dir=tmp_path, refresh=True)
    assert len(fake_fetch) == 2


def test_panel_contract_and_slicing(tmp_path, fake_fetch):
    price_cache.get_cached_price_panel(["AAA"], "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    panel, _flags, _meta = price_cache.get_cached_price_panel(
        ["AAA"], "2020-04-01", "2020-05-31", cache_dir=tmp_path)
    assert set(["open", "high", "low", "close", "volume", "adv_20d_dollars"]) <= set(panel.columns)
    assert panel.index.names == ["date", "code"]
    dates = panel.index.get_level_values(0)
    assert dates.min() >= pd.Timestamp("2020-04-01")
    assert dates.max() <= pd.Timestamp("2020-05-31")


def test_first_available_dates(tmp_path, fake_fetch):
    price_cache.get_cached_price_panel(["AAA", "LATEIPO"], "2020-01-01", "2020-12-31",
                                       cache_dir=tmp_path)
    firsts = price_cache.first_available_dates(["AAA", "LATEIPO", "MISSING"], cache_dir=tmp_path)
    assert firsts["AAA"] == pd.Timestamp("2020-01-01")
    assert firsts["LATEIPO"] == pd.Timestamp("2020-06-01")
    assert "MISSING" not in firsts
