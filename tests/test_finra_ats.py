"""
finra_ats tests — HTTP calls are mocked (a fake client with a scripted
`.post()`), so these never touch the network. Response shapes mirror the
REAL payloads captured 2026-07-29 (see module docstring): `weeklySummary`
returns one row per (week, reporting MPID), `weeklySummaryHistoric` rejects
a symbol filter and must be pulled market-wide then filtered client-side.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from python.data.finra_ats import (
    _aggregate_by_week,
    elevated_vs_baseline,
    fetch_all_recent_weeks,
    fetch_historic_week,
    load_cached_weeks,
    save_weeks,
    trailing_baseline_ratio,
    weekly_participation_ratio,
)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeClient:
    """Scripted client: `responses` is a list consumed in order, one per
    `.post()` call, so tests can assert exact pagination behaviour."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "payload": json})
        if not self._responses:
            return _FakeResponse([])
        return self._responses.pop(0)


def _no_wait_limiter():
    class _Unlimited:
        def try_acquire(self):
            return True

    return _Unlimited()


def _mpid_row(symbol, week, mpid, shares, trades=10):
    return {
        "issueSymbolIdentifier": symbol, "weekStartDate": week, "MPID": mpid,
        "totalWeeklyShareQuantity": shares, "totalWeeklyTradeCount": trades,
        "summaryTypeCode": "ATS_W_SMBL_FIRM",
    }


# ── aggregation ──────────────────────────────────────────────────────────────

def test_aggregate_by_week_sums_across_mpid():
    rows = [
        _mpid_row("AAPL", "2024-01-01", "KCGM", 1000),
        _mpid_row("AAPL", "2024-01-01", "LQNA", 500),
        _mpid_row("AAPL", "2024-01-08", "KCGM", 2000),
    ]
    agg = _aggregate_by_week(rows)
    assert agg["2024-01-01"]["total_shares"] == 1500
    assert agg["2024-01-08"]["total_shares"] == 2000


# ── fetch_all_recent_weeks ───────────────────────────────────────────────────

def test_fetch_all_recent_weeks_pages_until_no_new_week():
    page1 = [_mpid_row("AAPL", "2024-01-01", "KCGM", 1000), _mpid_row("AAPL", "2024-01-08", "KCGM", 900)]
    page2 = [_mpid_row("AAPL", "2024-01-08", "LQNA", 100), _mpid_row("AAPL", "2024-01-15", "KCGM", 800)]
    page3_no_new = [_mpid_row("AAPL", "2024-01-15", "LQNA", 50)]  # no new week -> stop
    client = _FakeClient([_FakeResponse(page1), _FakeResponse(page2), _FakeResponse(page3_no_new)])

    weeks = fetch_all_recent_weeks("AAPL", http_client=client, limiter=_no_wait_limiter())

    assert [w["week_start_date"] for w in weeks] == ["2024-01-01", "2024-01-08", "2024-01-15"]
    assert next(w for w in weeks if w["week_start_date"] == "2024-01-08")["total_shares"] == 1000
    assert len(client.calls) == 3  # stopped after the "no new week" page, not exhausting all responses


def test_fetch_all_recent_weeks_stops_on_empty_page():
    client = _FakeClient([_FakeResponse([_mpid_row("MSFT", "2024-02-05", "KCGM", 300)]), _FakeResponse([])])
    weeks = fetch_all_recent_weeks("MSFT", http_client=client, limiter=_no_wait_limiter())
    assert len(weeks) == 1
    assert len(client.calls) == 2


def test_fetch_all_recent_weeks_handles_204_no_content():
    client = _FakeClient([_FakeResponse([], status_code=204)])
    weeks = fetch_all_recent_weeks("ZZZZ", http_client=client, limiter=_no_wait_limiter())
    assert weeks == []


def test_fetch_all_recent_weeks_survives_request_error():
    class _RaisingClient:
        def post(self, url, json=None, headers=None):
            raise RuntimeError("network down")

    weeks = fetch_all_recent_weeks("AAPL", http_client=_RaisingClient(), limiter=_no_wait_limiter())
    assert weeks == []


# ── fetch_historic_week ──────────────────────────────────────────────────────

def test_fetch_historic_week_filters_to_requested_symbols_client_side():
    market_wide_page = [
        _mpid_row("AAPL", "2018-06-25", "KCGM", 500),
        _mpid_row("TSLA", "2018-06-25", "KCGM", 700),  # not in our universe
        _mpid_row("AAPL", "2018-06-25", "LQNA", 100),
    ]
    client = _FakeClient([_FakeResponse(market_wide_page), _FakeResponse([])])
    result = fetch_historic_week("2018-06-25", {"AAPL", "NVDA"}, http_client=client, limiter=_no_wait_limiter())
    assert set(result) == {"AAPL"}
    assert result["AAPL"]["total_shares"] == 600
    # Payload sent must NOT try to filter by symbol server-side (API rejects it).
    sent_fields = {f["fieldName"] for f in client.calls[0]["payload"]["compareFilters"]}
    assert sent_fields == {"weekStartDate", "tierIdentifier"}


def test_fetch_historic_week_empty_when_no_universe_match():
    client = _FakeClient([_FakeResponse([_mpid_row("TSLA", "2018-06-25", "KCGM", 700)])])
    result = fetch_historic_week("2018-06-25", {"AAPL"}, http_client=client, limiter=_no_wait_limiter())
    assert result == {}


# ── cache round-trip ─────────────────────────────────────────────────────────

def test_save_and_load_cached_weeks_roundtrip(tmp_path):
    weeks = [
        {"week_start_date": "2024-01-01", "total_shares": 1500, "total_trades": 20},
        {"week_start_date": "2024-01-08", "total_shares": 2000, "total_trades": 25},
    ]
    save_weeks("AAPL", weeks, cache_dir=tmp_path)
    df = load_cached_weeks("AAPL", cache_dir=tmp_path)
    assert list(df.index) == ["2024-01-01", "2024-01-08"]
    assert df.loc["2024-01-08", "total_shares"] == 2000


def test_save_weeks_dedups_last_write_wins(tmp_path):
    save_weeks("AAPL", [{"week_start_date": "2024-01-01", "total_shares": 1000, "total_trades": 10}], cache_dir=tmp_path)
    save_weeks("AAPL", [{"week_start_date": "2024-01-01", "total_shares": 1500, "total_trades": 15}], cache_dir=tmp_path)
    df = load_cached_weeks("AAPL", cache_dir=tmp_path)
    assert len(df) == 1
    assert df.loc["2024-01-01", "total_shares"] == 1500


def test_load_cached_weeks_missing_file_returns_empty_frame(tmp_path):
    df = load_cached_weeks("NOPE", cache_dir=tmp_path)
    assert df.empty


# ── derived ratios / flags ───────────────────────────────────────────────────

def test_weekly_participation_ratio_needs_cached_week(tmp_path):
    assert weekly_participation_ratio("AAPL", "2024-01-01", 1_000_000, cache_dir=tmp_path) is None
    save_weeks("AAPL", [{"week_start_date": "2024-01-01", "total_shares": 400_000, "total_trades": 10}], cache_dir=tmp_path)
    ratio = weekly_participation_ratio("AAPL", "2024-01-01", 1_000_000, cache_dir=tmp_path)
    assert ratio == pytest.approx(0.4)


def test_weekly_participation_ratio_rejects_nonpositive_volume(tmp_path):
    save_weeks("AAPL", [{"week_start_date": "2024-01-01", "total_shares": 400_000, "total_trades": 10}], cache_dir=tmp_path)
    assert weekly_participation_ratio("AAPL", "2024-01-01", 0, cache_dir=tmp_path) is None


def test_trailing_baseline_ratio_needs_enough_history(tmp_path):
    save_weeks("AAPL", [{"week_start_date": "2024-01-01", "total_shares": 100_000, "total_trades": 10}], cache_dir=tmp_path)
    assert trailing_baseline_ratio("AAPL", "2024-01-08", lookback_weeks=12, cache_dir=tmp_path) is None


def test_elevated_vs_baseline_flags_spike(tmp_path):
    weeks = [{"week_start_date": f"2024-01-{d:02d}", "total_shares": 100_000, "total_trades": 10} for d in (1, 8, 15, 22)]
    save_weeks("AAPL", weeks, cache_dir=tmp_path)
    save_weeks("AAPL", [{"week_start_date": "2024-01-29", "total_shares": 500_000, "total_trades": 40}], cache_dir=tmp_path)
    assert elevated_vs_baseline("AAPL", "2024-01-29", lookback_weeks=4, cache_dir=tmp_path) is True


def test_elevated_vs_baseline_normal_week_is_false(tmp_path):
    weeks = [{"week_start_date": f"2024-01-{d:02d}", "total_shares": 100_000, "total_trades": 10} for d in (1, 8, 15, 22)]
    save_weeks("AAPL", weeks, cache_dir=tmp_path)
    save_weeks("AAPL", [{"week_start_date": "2024-01-29", "total_shares": 105_000, "total_trades": 10}], cache_dir=tmp_path)
    assert elevated_vs_baseline("AAPL", "2024-01-29", lookback_weeks=4, cache_dir=tmp_path) is False


def test_elevated_vs_baseline_unknown_week_returns_none(tmp_path):
    assert elevated_vs_baseline("AAPL", "2024-01-29", cache_dir=tmp_path) is None
