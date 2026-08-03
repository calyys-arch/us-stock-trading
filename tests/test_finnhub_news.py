"""
FinnhubNewsSignal tests — no real network calls.

Coverage:
  - No API key -> both has_company_news_today and
    general_market_headlines_today fail safe (False / []).
  - has_company_news_today: only counts headlines whose unix `datetime`
    actually falls on the queried ET day (Finnhub's from/to date params are
    not a hard guarantee the API won't ever include an adjacent-day item).
  - has_company_news_today results are cached per (day, symbol) — repeated
    lookups for the SAME symbol/day do not refetch.
  - general_market_headlines_today is cached per day and filters out
    off-day items the same way.
  - A fetch failure fails safe without raising.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from python.interfaces.finnhub_news import FinnhubNewsSignal, _ET


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload=None, raise_exc: Exception | None = None) -> None:
        self.payload = payload if payload is not None else []
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append({"url": url, "params": params})
        if self.raise_exc:
            raise self.raise_exc
        return _FakeResponse(self.payload)


def _ts_on(day_str: str, hour: int = 12) -> int:
    dt = datetime.fromisoformat(day_str).replace(hour=hour, tzinfo=_ET)
    return int(dt.timestamp())


def test_no_api_key_fails_safe():
    signal = FinnhubNewsSignal(api_key="", http_client=_FakeHttpClient())
    assert signal.has_company_news_today("AAPL") is False
    assert signal.general_market_headlines_today() == []


def test_company_news_true_when_headline_matches_today():
    today = datetime.now(_ET).date().isoformat()
    fake = _FakeHttpClient(payload=[{"headline": "AAPL beats", "datetime": _ts_on(today)}])
    signal = FinnhubNewsSignal(api_key="fake-key", http_client=fake)

    assert signal.has_company_news_today("AAPL") is True


def test_company_news_false_when_only_adjacent_day_items_returned():
    fake = _FakeHttpClient(payload=[{"headline": "old news", "datetime": _ts_on("2020-01-01")}])
    signal = FinnhubNewsSignal(api_key="fake-key", http_client=fake)

    assert signal.has_company_news_today("AAPL") is False


def test_company_news_cached_per_symbol_per_day():
    today = datetime.now(_ET).date().isoformat()
    fake = _FakeHttpClient(payload=[{"headline": "AAPL news", "datetime": _ts_on(today)}])
    signal = FinnhubNewsSignal(api_key="fake-key", http_client=fake)

    for _ in range(5):
        signal.has_company_news_today("AAPL")
    assert len(fake.calls) == 1

    signal.has_company_news_today("MSFT")
    assert len(fake.calls) == 2


def test_company_news_fetch_failure_fails_safe():
    fake = _FakeHttpClient(raise_exc=RuntimeError("network down"))
    signal = FinnhubNewsSignal(api_key="fake-key", http_client=fake)

    assert signal.has_company_news_today("AAPL") is False


def test_general_market_headlines_filters_to_today_and_caches():
    today = datetime.now(_ET).date().isoformat()
    fake = _FakeHttpClient(payload=[
        {"headline": "Fed holds rates", "source": "Reuters", "url": "http://x", "datetime": _ts_on(today)},
        {"headline": "stale item", "source": "Reuters", "url": "http://y", "datetime": _ts_on("2020-01-01")},
    ])
    signal = FinnhubNewsSignal(api_key="fake-key", http_client=fake)

    headlines = signal.general_market_headlines_today()
    assert len(headlines) == 1
    assert headlines[0]["headline"] == "Fed holds rates"

    signal.general_market_headlines_today()
    assert len(fake.calls) == 1, "expected exactly one HTTP call across repeated same-day lookups"


def test_general_market_headlines_fetch_failure_fails_safe():
    fake = _FakeHttpClient(raise_exc=RuntimeError("network down"))
    signal = FinnhubNewsSignal(api_key="fake-key", http_client=fake)

    assert signal.general_market_headlines_today() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
