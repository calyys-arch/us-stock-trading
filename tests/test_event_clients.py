"""
FinnhubClient + EdgarClient tests — no real network (fake httpx-like
clients), tmp_path caches.

Key behaviors under test:
  - Finnhub month-level news caching: one HTTP call per (symbol, month),
    re-served from disk afterwards; empty months cached as empty.
  - No API key -> fail-safe empty results, no crash.
  - 403 -> endpoint disabled for the process (no repeat requests).
  - has_news_data three-valued logic (True / False / None=no coverage).
  - EDGAR ticker->CIK mapping, 8-K filtering from the columnar submissions
    payload, RSS merge dedup, and has_8k_near three-valued logic.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from python.data.edgar_client import EdgarClient
from python.data.finnhub_client import FinnhubClient


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200, content: bytes = b""):
        self._payload = payload
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeHttp:
    """Routes by substring of the URL; records every call."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None):
        self.calls.append((url, params or {}))
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, _FakeResponse):
                    return payload
                return _FakeResponse(payload)
        return _FakeResponse({}, status_code=404)


# ── Finnhub ──────────────────────────────────────────────────────────────────

def _finnhub(tmp_path, routes, api_key="k"):
    return FinnhubClient(
        api_key=api_key, http_client=_FakeHttp(routes),
        news_cache_dir=tmp_path / "news", calendar_cache_dir=tmp_path / "calendar",
    )


def test_company_news_caches_per_month(tmp_path):
    rows = [{"datetime": 1717200000, "headline": "x", "symbol": "AAPL"}]
    client = _finnhub(tmp_path, {"/company-news": rows})
    out1 = client.company_news("AAPL", date(2024, 6, 1), date(2024, 7, 31))
    n_calls = len(client._client.calls)
    assert n_calls == 2  # one fetch per month: June + July
    assert len(out1) == 2  # the fake returns 1 row per month fetch

    # second identical request: served from disk, zero new calls
    client.company_news("AAPL", date(2024, 6, 1), date(2024, 7, 31))
    assert len(client._client.calls) == n_calls


def test_company_news_no_key_fails_safe(tmp_path):
    client = _finnhub(tmp_path, {"/company-news": [{"headline": "x"}]}, api_key="")
    assert client.company_news("AAPL", date(2024, 6, 1), date(2024, 6, 30)) == []
    assert client._client.calls == []


def test_403_disables_endpoint_for_process(tmp_path):
    client = _finnhub(tmp_path, {"/calendar/economic": _FakeResponse({}, status_code=403)})
    assert client.economic_calendar(date(2024, 1, 1), date(2024, 6, 30)) == []
    assert client.economic_calendar(date(2024, 7, 1), date(2024, 12, 31)) == []
    econ_calls = [c for c in client._client.calls if "/calendar/economic" in c[0]]
    assert len(econ_calls) == 1, "403 endpoint must not be re-requested"


def test_has_news_data_three_valued(tmp_path):
    client = _finnhub(tmp_path, {})
    # no cache at all -> None (UNKNOWN)
    assert client.has_news_data("AAPL", date(2024, 6, 5)) is None

    # cached month with one article on 2024-06-05
    import time as _time

    ts = int(_time.mktime(date(2024, 6, 5).timetuple())) + 3600
    cache = tmp_path / "news" / "AAPL" / "2024-06.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([{"datetime": ts, "headline": "x"}]))
    assert client.has_news_data("AAPL", date(2024, 6, 5)) is True
    assert client.has_news_data("AAPL", date(2024, 6, 6)) is False


def test_earnings_calendar_cached_per_year_and_filtered(tmp_path):
    rows = [{"symbol": "AAPL", "date": "2024-05-02"}, {"symbol": "MSFT", "date": "2024-10-24"}]
    client = _finnhub(tmp_path, {"/calendar/earnings": {"earningsCalendar": rows}})
    out = client.earnings_calendar(date(2024, 1, 1), date(2024, 6, 30))
    assert [r["symbol"] for r in out] == ["AAPL"]   # Oct row filtered out
    client.earnings_calendar(date(2024, 1, 1), date(2024, 12, 31))
    year_calls = [c for c in client._client.calls if "/calendar/earnings" in c[0]]
    assert len(year_calls) == 1, "year cache must be reused"

    lookup = client.earnings_dates_by_symbol(date(2024, 1, 1), date(2024, 12, 31))
    assert lookup == {"AAPL": {"2024-05-02"}, "MSFT": {"2024-10-24"}}


# ── EDGAR ────────────────────────────────────────────────────────────────────

_TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "8-K/A"],
            "accessionNumber": ["acc-1", "acc-2", "acc-3"],
            "filingDate": ["2024-05-02", "2024-05-03", "2024-05-10"],
            "acceptanceDateTime": ["2024-05-02T16:30:00.000Z", "2024-05-03T10:00:00.000Z",
                                   "2024-05-10T09:00:00.000Z"],
            "items": ["2.02,9.01", "", "2.02"],
        },
        "files": [],
    }
}


def _edgar(tmp_path, routes):
    return EdgarClient(http_client=_FakeHttp(routes), cache_dir=tmp_path / "filings",
                       user_agent="test-suite test@example.com")


def test_ticker_to_cik(tmp_path):
    client = _edgar(tmp_path, {"company_tickers.json": _TICKER_MAP})
    assert client.ticker_to_cik("aapl") == "0000320193"
    assert client.ticker_to_cik("ZZZZ") is None


def test_backfill_filters_to_8k_forms(tmp_path):
    client = _edgar(tmp_path, {"company_tickers.json": _TICKER_MAP,
                               "submissions/CIK0000320193.json": _SUBMISSIONS})
    filings = client.backfill_8k("AAPL", since=date(2024, 1, 1))
    assert [f["accession"] for f in filings] == ["acc-1", "acc-3"]
    assert all(f["form"] in ("8-K", "8-K/A") for f in filings)
    assert filings[0]["items"] == "2.02,9.01"


def test_has_8k_near_three_valued(tmp_path):
    client = _edgar(tmp_path, {"company_tickers.json": _TICKER_MAP,
                               "submissions/CIK0000320193.json": _SUBMISSIONS})
    # no cache yet -> None
    assert client.has_8k_near("AAPL", date(2024, 5, 2)) is None
    client.backfill_8k("AAPL")
    assert client.has_8k_near("AAPL", date(2024, 5, 2)) is True
    assert client.has_8k_near("AAPL", date(2024, 5, 3), window_days=1) is True   # +/-1 day window
    assert client.has_8k_near("AAPL", date(2024, 7, 1)) is False


def test_rss_merge_dedups_on_accession(tmp_path):
    atom = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>urn:tag:sec.gov,2008:accession-number=acc-1</id>
    <category term="8-K" label="form type"/>
    <updated>2024-05-02T16:30:00-04:00</updated>
  </entry>
  <entry>
    <id>urn:tag:sec.gov,2008:accession-number=acc-9</id>
    <category term="8-K" label="form type"/>
    <updated>2024-06-01T09:00:00-04:00</updated>
  </entry>
</feed>"""
    client = _edgar(tmp_path, {
        "company_tickers.json": _TICKER_MAP,
        "submissions/CIK0000320193.json": _SUBMISSIONS,
        "browse-edgar": _FakeResponse({}, content=atom),
    })
    client.backfill_8k("AAPL")
    merged = client.refresh_8k("AAPL")
    accessions = [f["accession"] for f in merged]
    assert accessions.count("acc-1") == 1          # deduped
    assert "acc-9" in accessions                    # new RSS entry merged
    assert client.eight_k_dates_by_symbol(["AAPL"])["AAPL"] >= {"2024-05-02", "2024-06-01"}
