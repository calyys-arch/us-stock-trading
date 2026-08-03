"""
Finnhub HISTORICAL news + calendar client for the report-only signal-trap
diagnostic layer (python/signals/trap_detector.py).

Scope split with python/interfaces/finnhub_calendar.py: that module answers
ONE live question ("does this ticker report earnings TODAY?") for the
running engine, refreshing once per ET day with no disk state. THIS module
serves research/backtest needs — historical company news, earnings-calendar
ranges, and the economic calendar — with a DISK cache (data/news/,
data/calendar/) so a backtest over 2018-2025 doesn't re-download the same
months on every run.

Free-tier honesty (user confirmed free plan, 2026-07-28):
  - /company-news only returns roughly the last 12 months on the free
    plan. Backtests further back get empty months — cached as empty so we
    don't re-ask — and SEC EDGAR 8-K filings (python/data/edgar_client.py)
    are the full-history event source instead. Trap-detector news
    qualifiers treat "no news data available" as UNKNOWN, never as
    "no news happened".
  - /calendar/economic is NOT included in the free plan for most keys; a
    403 marks the endpoint unavailable for the process and returns empty.

Rate limiting: free tier allows 60 req/min. A shared token bucket
(python/core/rate_limiter.py) at 0.9 req/s stays under it while letting a
year-long backfill finish in minutes. Callers see blocking waits, not
errors, when the bucket is empty.

Key handling mirrors finnhub_calendar.py: FINNHUB_API_KEY from the process
env or a local .env (python-dotenv). No key -> warn once, return empty
results (fail-safe; the diagnostic layer just reports "news evidence
unavailable").
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"
NEWS_CACHE_DIR = Path("data/news")
CALENDAR_CACHE_DIR = Path("data/calendar")

_LIMITER_NAME = "finnhub"
_REQUESTS_PER_SECOND = 0.9


def _month_starts(start: date, end: date) -> list[tuple[date, date]]:
    """[(month_first_day, month_last_day_clamped)] covering [start, end]."""
    import pandas as pd

    out = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        month_end = (pd.Timestamp(cursor) + pd.offsets.MonthEnd(0)).date()
        out.append((max(cursor, start), min(month_end, end)))
        cursor = (pd.Timestamp(cursor) + pd.offsets.MonthBegin(1)).date()
    return out


class FinnhubClient:
    """Disk-cached Finnhub REST client. All getters return plain lists of
    dicts (Finnhub's own JSON rows) and NEVER raise on network/auth
    problems — they log once and return what the cache has (possibly
    nothing). The trap-detector layer must degrade to 'evidence
    unavailable', not crash a backtest."""

    def __init__(
        self,
        api_key: str | None = None,
        http_client=None,
        news_cache_dir: str | Path = NEWS_CACHE_DIR,
        calendar_cache_dir: str | Path = CALENDAR_CACHE_DIR,
    ) -> None:
        if api_key is None:
            import os

            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.environ.get("FINNHUB_API_KEY", "")
        self._api_key = api_key

        if http_client is None:
            import httpx

            http_client = httpx.Client(timeout=15.0)
        self._client = http_client

        from ..core.rate_limiter import RateLimitConfig, registry

        self._limiter = registry.get(
            _LIMITER_NAME, RateLimitConfig(requests_per_second=_REQUESTS_PER_SECOND, daily_quota=None)
        )
        self._news_cache_dir = Path(news_cache_dir)
        self._calendar_cache_dir = Path(calendar_cache_dir)
        self._warned_no_key = False
        self._unavailable_endpoints: set[str] = set()

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> list | dict | None:
        """One rate-limited GET. Returns parsed JSON, or None on any failure
        (logged). A 403 marks the endpoint plan-unavailable for the rest of
        the process so we don't burn quota re-asking."""
        if not self._api_key:
            if not self._warned_no_key:
                log.warning("FinnhubClient: no FINNHUB_API_KEY — all lookups return empty "
                            "(see .env.example)")
                self._warned_no_key = True
            return None
        if endpoint in self._unavailable_endpoints:
            return None

        while not self._limiter.try_acquire():
            time.sleep(0.2)
        try:
            resp = self._client.get(f"{BASE_URL}{endpoint}", params={**params, "token": self._api_key})
            if resp.status_code == 403:
                log.warning("FinnhubClient: %s returned 403 (not in current plan) — "
                            "disabled for this process", endpoint)
                self._unavailable_endpoints.add(endpoint)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("FinnhubClient: %s failed (%s) — returning cached/empty", endpoint, exc)
            return None

    @staticmethod
    def _read_cache(path: Path) -> list | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("FinnhubClient: unreadable cache %s — refetching", path)
            return None

    @staticmethod
    def _write_cache(path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")

    # ── company news ─────────────────────────────────────────────────────────

    def company_news(self, symbol: str, start: date, end: date, refresh: bool = False) -> list[dict]:
        """Headlines/summaries for `symbol` over [start, end], cached one
        JSON file per (symbol, month). Free tier: months older than ~1 year
        come back empty — cached as empty (a re-run must not re-ask)."""
        symbol = symbol.upper()
        rows: list[dict] = []
        for month_start, month_end in _month_starts(start, end):
            cache_path = self._news_cache_dir / symbol / f"{month_start:%Y-%m}.json"
            cached = None if refresh else self._read_cache(cache_path)
            if cached is None:
                fetched = self._get(
                    "/company-news",
                    {"symbol": symbol, "from": month_start.isoformat(), "to": month_end.isoformat()},
                )
                if fetched is None:
                    cached = self._read_cache(cache_path) or []
                else:
                    cached = fetched if isinstance(fetched, list) else []
                    self._write_cache(cache_path, cached)
            rows.extend(cached)
        return rows

    def has_news_data(self, symbol: str, day: date) -> bool | None:
        """Three-valued evidence for the trap detector: True (news exists
        that day), False (we HAVE coverage for that month and there was no
        news), None (no cache and no fetchable data -> UNKNOWN)."""
        symbol = symbol.upper()
        cache_path = self._news_cache_dir / symbol / f"{day:%Y-%m}.json"
        cached = self._read_cache(cache_path)
        if cached is None:
            return None
        day_start = int(time.mktime(day.timetuple()))
        day_end = day_start + 86400
        return any(day_start <= int(row.get("datetime", 0)) < day_end for row in cached)

    # ── calendars ────────────────────────────────────────────────────────────

    def earnings_calendar(self, start: date, end: date, refresh: bool = False) -> list[dict]:
        """Earnings-calendar rows for [start, end], cached per calendar
        year. Row shape: {symbol, date, hour, epsEstimate, ...} (Finnhub's
        earningsCalendar list)."""
        rows: list[dict] = []
        for year in range(start.year, end.year + 1):
            cache_path = self._calendar_cache_dir / f"earnings_{year}.json"
            cached = None if refresh else self._read_cache(cache_path)
            if cached is None:
                fetched = self._get(
                    "/calendar/earnings",
                    {"from": f"{year}-01-01", "to": f"{year}-12-31"},
                )
                if fetched is None:
                    cached = self._read_cache(cache_path) or []
                else:
                    cached = fetched.get("earningsCalendar", []) if isinstance(fetched, dict) else []
                    self._write_cache(cache_path, cached)
            rows.extend(cached)
        return [r for r in rows if r.get("date") and start.isoformat() <= r["date"] <= end.isoformat()]

    def economic_calendar(self, start: date, end: date, refresh: bool = False) -> list[dict]:
        """Economic-calendar rows (FOMC, CPI, NFP...). Premium-gated on
        most free keys — a 403 disables the endpoint and this returns
        whatever is cached (usually nothing)."""
        cache_path = self._calendar_cache_dir / f"economic_{start:%Y%m%d}_{end:%Y%m%d}.json"
        cached = None if refresh else self._read_cache(cache_path)
        if cached is None:
            fetched = self._get(
                "/calendar/economic", {"from": start.isoformat(), "to": end.isoformat()}
            )
            if fetched is None:
                cached = self._read_cache(cache_path) or []
            else:
                cached = fetched.get("economicCalendar", []) if isinstance(fetched, dict) else []
                self._write_cache(cache_path, cached)
        return cached

    def earnings_dates_by_symbol(self, start: date, end: date) -> dict[str, set[str]]:
        """{symbol: {iso_date, ...}} — the trap-report annotator's lookup
        shape for 'signal fell near an earnings date'."""
        out: dict[str, set[str]] = {}
        for row in self.earnings_calendar(start, end):
            symbol = str(row.get("symbol", "")).upper().strip()
            if symbol and row.get("date"):
                out.setdefault(symbol, set()).add(row["date"])
        return out
