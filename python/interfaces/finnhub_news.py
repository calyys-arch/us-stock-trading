"""
Finnhub news client — TWO independent signals, both optional add-ons on top
of the earnings-calendar integration in `finnhub_calendar.py`:

  1. `has_company_news_today(code)` — per-symbol company news (Finnhub
     `/company-news`). This is the real provider behind `DataEngine`'s
     SECOND (and, until now, permanently unwired) news injection point:
     `DataEngine(news_event_checker=...)` -> `MarketSnapshot.has_news_event`
     (see python/core/data_engine.py __init__ and python/core/types.py's
     "News / calendar flags" section). Same rationale as `is_earnings_today`
     (finnhub_calendar.py's docstring): a same-day company-specific headline
     (M&A, guidance, downgrade, litigation, ...) is a news/event-driven price
     move, not the noise-driven mean reversion Chan's cross-sectional
     strategy assumes — so the universe builder should be able to exclude
     names with news today, same as it excludes earnings-today names.

  2. `general_market_headlines_today()` — market-wide headlines (Finnhub
     `/news?category=general`), deliberately exposed as a plain headline
     list rather than a boolean gate. There is no principled threshold for
     "how many general-news items = an excludable day" (general news feeds
     always carry dozens of items regardless of whether anything unusual
     happened), and architecture-rules.mdc's Chan discipline requires every
     guard/config key to have a REAL enforcement point, not an invented
     magic number — so this is informational only (e.g. printed at the top
     of scripts/pick_10.py) and is NOT wired into any exclusion filter.

Caching / rate-limiting discipline (see finnhub_calendar.py's docstring for
the same argument in more depth):
  - General market news is cached ONCE per ET calendar day, like the
    earnings calendar — a single request covers the whole day, so no
    separate rate limiter is layered on top (the day-cache IS the limit).
  - Company news is fetched PER SYMBOL and genuinely can be requested for
    dozens/hundreds of distinct tickers within seconds of each other (e.g.
    scripts/pick_10.py looping over its universe) — unlike the calendar or
    general-news case, that access pattern really can exceed Finnhub's free
    tier (60 requests/minute), so this class owns a real per-instance
    token-bucket limiter (python/core/rate_limiter.py) and BLOCKS (briefly
    sleeps) rather than silently skipping, since a skipped symbol would
    mean permanently missing news for that name for the rest of the day.

Fails safe exactly like finnhub_calendar.py: no API key or a request
failure -> logs a warning once per key/day and returns False / an empty
headline list, never raises.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from ..core.rate_limiter import RateLimitConfig, RateLimiter

log = logging.getLogger(__name__)

load_dotenv()

_ET = ZoneInfo("America/New_York")
_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_GENERAL_NEWS_URL = "https://finnhub.io/api/v1/news"
_RATE_LIMIT_MAX_WAIT_SEC = 30.0


class FinnhubNewsSignal:
    """`.has_company_news_today(code)` is the callable meant for
    `DataEngine(news_event_checker=...)`. `.general_market_headlines_today()`
    is informational-only — see module docstring for why it is not a gate."""

    def __init__(self, api_key: str | None = None, http_client: httpx.Client | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY", "")
        self._client = http_client or httpx.Client(timeout=10.0)
        self._lock = threading.Lock()

        self._company_news_cache: dict[tuple[date, str], bool] = {}
        self._general_news_cache: tuple[date, list[dict]] | None = None

        self._warned_no_key = False
        self._warned_fetch_failure = False

        # Real per-instance limiter — see module docstring: company-news
        # lookups are per-symbol and can legitimately fire dozens/hundreds
        # of times within seconds, unlike the day-level caches above.
        self._company_news_limiter = RateLimiter("finnhub_company_news", RateLimitConfig(requests_per_second=1.0))

    def has_company_news_today(self, code: str) -> bool:
        code = code.upper().strip()
        today = datetime.now(_ET).date()
        with self._lock:
            cached = self._company_news_cache.get((today, code))
            if cached is not None:
                return cached
            result = self._fetch_company_news(code, today)
            self._company_news_cache[(today, code)] = result
            return result

    def general_market_headlines_today(self) -> list[dict]:
        today = datetime.now(_ET).date()
        with self._lock:
            if self._general_news_cache is not None and self._general_news_cache[0] == today:
                return self._general_news_cache[1]
            headlines = self._fetch_general_news()
            self._general_news_cache = (today, headlines)
            return headlines

    def _no_key_warning(self, what: str) -> None:
        if not self._warned_no_key:
            log.warning(
                "FinnhubNewsSignal: no FINNHUB_API_KEY configured — %s will return an "
                "empty/False result (see .env.example)", what,
            )
            self._warned_no_key = True

    def _fetch_company_news(self, code: str, day: date) -> bool:
        if not self._api_key:
            self._no_key_warning("has_company_news_today")
            return False

        deadline = time.monotonic() + _RATE_LIMIT_MAX_WAIT_SEC
        while not self._company_news_limiter.try_acquire():
            if time.monotonic() > deadline:
                log.warning(
                    "FinnhubNewsSignal: rate limiter wait exceeded %.0fs for %s — "
                    "treating as no-news-today for this symbol", _RATE_LIMIT_MAX_WAIT_SEC, code,
                )
                return False
            time.sleep(0.05)

        try:
            resp = self._client.get(
                _COMPANY_NEWS_URL,
                params={"symbol": code, "from": day.isoformat(), "to": day.isoformat(), "token": self._api_key},
            )
            resp.raise_for_status()
            rows = resp.json()
        except Exception:
            self._log_fetch_failure(f"company news for {code}")
            return False

        self._warned_fetch_failure = False
        return any(self._is_on_day(row.get("datetime"), day) for row in rows or [])

    def _fetch_general_news(self) -> list[dict]:
        if not self._api_key:
            self._no_key_warning("general_market_headlines_today")
            return []

        try:
            resp = self._client.get(
                _GENERAL_NEWS_URL,
                params={"category": "general", "token": self._api_key},
            )
            resp.raise_for_status()
            rows = resp.json()
        except Exception:
            self._log_fetch_failure("general market news")
            return []

        self._warned_fetch_failure = False
        today = datetime.now(_ET).date()
        return [
            {
                "headline": row.get("headline", ""),
                "source": row.get("source", ""),
                "url": row.get("url", ""),
                "datetime": row.get("datetime"),
            }
            for row in rows or []
            if self._is_on_day(row.get("datetime"), today)
        ]

    def _log_fetch_failure(self, what: str) -> None:
        if not self._warned_fetch_failure:
            log.exception("FinnhubNewsSignal: fetch failed for %s — failing safe", what)
            self._warned_fetch_failure = True

    @staticmethod
    def _is_on_day(unix_ts, day: date) -> bool:
        if not unix_ts:
            return False
        try:
            return datetime.fromtimestamp(unix_ts, tz=_ET).date() == day
        except (OverflowError, OSError, ValueError):
            return False
