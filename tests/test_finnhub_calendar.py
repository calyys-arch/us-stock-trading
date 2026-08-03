"""
FinnhubEarningsCalendar tests — no real network calls (httpx.Client is
replaced with a fake that records calls and returns canned responses).

Coverage:
  - No API key configured -> fails safe (False for every ticker, no crash).
  - A configured key with a mocked response -> correct tickers return True.
  - Caching: two lookups on the SAME day trigger only ONE HTTP call (Finnhub
    lesson from rate_limiter.py's docstring: never re-fetch per-ticker).
  - A failed HTTP call -> fails safe (False / stale cache, no crash).
"""
from __future__ import annotations

from datetime import date

import pytest

from python.interfaces.finnhub_calendar import FinnhubEarningsCalendar


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload: dict | None = None, raise_exc: Exception | None = None) -> None:
        self.payload = payload or {"earningsCalendar": []}
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append({"url": url, "params": params})
        if self.raise_exc:
            raise self.raise_exc
        return _FakeResponse(self.payload)


def test_no_api_key_fails_safe():
    calendar = FinnhubEarningsCalendar(api_key="", http_client=_FakeHttpClient())
    assert calendar.is_earnings_today("AAPL") is False
    assert calendar.is_earnings_today("MSFT") is False


def test_configured_key_returns_correct_tickers():
    fake = _FakeHttpClient(payload={"earningsCalendar": [
        {"symbol": "AAPL", "date": "2026-07-28"},
        {"symbol": "msft", "date": "2026-07-28"},  # lowercase — must be normalized
    ]})
    calendar = FinnhubEarningsCalendar(api_key="fake-key", http_client=fake)

    assert calendar.is_earnings_today("AAPL") is True
    assert calendar.is_earnings_today("MSFT") is True
    assert calendar.is_earnings_today("GOOGL") is False


def test_same_day_lookups_use_cache_not_repeated_http_calls():
    fake = _FakeHttpClient(payload={"earningsCalendar": [{"symbol": "AAPL"}]})
    calendar = FinnhubEarningsCalendar(api_key="fake-key", http_client=fake)

    for _ in range(5):
        calendar.is_earnings_today("AAPL")
        calendar.is_earnings_today("MSFT")

    assert len(fake.calls) == 1, "expected exactly one HTTP call across many same-day lookups"


def test_stale_cache_persists_across_days_without_refetch_if_forced():
    """Sanity check that _ensure_fresh only refetches when the cached day
    actually differs from `today` (not a live-clock test — directly exercises
    the day-comparison branch)."""
    fake = _FakeHttpClient(payload={"earningsCalendar": [{"symbol": "AAPL"}]})
    calendar = FinnhubEarningsCalendar(api_key="fake-key", http_client=fake)

    calendar._ensure_fresh(date(2026, 7, 28))
    calendar._ensure_fresh(date(2026, 7, 28))
    assert len(fake.calls) == 1

    calendar._ensure_fresh(date(2026, 7, 29))
    assert len(fake.calls) == 2


def test_fetch_failure_fails_safe_without_raising():
    fake = _FakeHttpClient(raise_exc=RuntimeError("network down"))
    calendar = FinnhubEarningsCalendar(api_key="fake-key", http_client=fake)

    result = calendar.is_earnings_today("AAPL")
    assert result is False


def test_fetch_failure_keeps_previous_cache_instead_of_wiping_it():
    fake = _FakeHttpClient(payload={"earningsCalendar": [{"symbol": "AAPL"}]})
    calendar = FinnhubEarningsCalendar(api_key="fake-key", http_client=fake)
    calendar._ensure_fresh(date(2026, 7, 28))
    assert calendar.is_earnings_today("AAPL") is True

    fake.raise_exc = RuntimeError("network down")
    calendar._ensure_fresh(date(2026, 7, 29))
    # fetch failed on the new day -> should keep serving yesterday's cached
    # set rather than silently returning False for a name that may still
    # genuinely be reporting today (best-effort staleness over a hard reset).
    assert calendar.is_earnings_today("AAPL") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
