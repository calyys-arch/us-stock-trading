"""
Finnhub earnings-calendar client — fills the `is_earnings_today` injection
point on `python/core/data_engine.py`'s `ReferenceData` (see that module's
docstring: DataEngine itself must stay a pure tick-processing component with
no DB/network access — architecture-rules.mdc "Layer Ownership"). This
module is the network-touching provider; `dashboard/engine_bridge.py` is the
only place that wires it into `ReferenceData`.

Why this exists: Chan's cross-sectional mean-reversion strategy assumes a
stock's price move is a statistical reversion signal — a move driven by a
same-day earnings surprise is a fundamentally different (news/event-driven)
process and should not be traded as if it were noise-driven mean reversion.
`is_earnings_today` lets the strategy layer exclude those names from the
day's cross-section (the actual exclusion logic belongs in the strategy/
universe layer, not here — this module only answers "does this ticker
report earnings today?").

API: Finnhub's free tier (https://finnhub.io) covers `/calendar/earnings`
at 60 requests/minute. Requires a `FINNHUB_API_KEY` environment variable
(see `.env.example`); this module loads a local `.env` file via
python-dotenv if present, then falls back to whatever is already in the
process environment.

Caching discipline: `DataEngine` calls `is_earnings_today(code)` once per
snapshot per instrument — with dozens of instruments emitting snapshots
every few seconds, a naive per-call network request would blow through
Finnhub's rate limit almost immediately (forex-trading lesson #5, see
python/core/rate_limiter.py's docstring). Instead, `FinnhubEarningsCalendar`
fetches the FULL day's earnings-reporting ticker set ONCE per calendar day
(ET) and serves every subsequent lookup for that day out of an in-memory
set — one API call covers an entire trading day regardless of how many
instruments/snapshots query it. At most 1 request/~86400s is nowhere near
Finnhub's 60/minute free-tier ceiling, so this day-level cache IS the rate
limit; a separate token-bucket limiter (python/core/rate_limiter.py) was
deliberately NOT layered on top — it would add no real protection here and
would risk suppressing a legitimate same-day retry after a transient
network failure.

Fail-safe default (matches the pre-existing behavior when nothing is
wired): if no API key is configured, or the Finnhub request fails, this
logs a warning ONCE per day and returns False for every ticker rather than
raising — a missing/broken calendar feed must never crash the data pipeline
or block trading; it should just mean "earnings-day exclusion is inactive
today," which is exactly today's status quo before this module existed.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

_ET = ZoneInfo("America/New_York")
_FINNHUB_BASE_URL = "https://finnhub.io/api/v1/calendar/earnings"


class FinnhubEarningsCalendar:
    """Call `.is_earnings_today(code)` from a `ReferenceData(is_earnings_today=...)`
    binding. Thread-safe; refreshes its cached ticker set at most once per
    ET calendar day."""

    def __init__(self, api_key: str | None = None, http_client: httpx.Client | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY", "")
        self._client = http_client or httpx.Client(timeout=10.0)
        self._lock = threading.Lock()
        self._cached_day: date | None = None
        self._cached_tickers: set[str] = set()
        self._warned_no_key = False
        self._warned_fetch_failure = False

    def is_earnings_today(self, code: str) -> bool:
        today = datetime.now(_ET).date()
        self._ensure_fresh(today)
        return code.upper().strip() in self._cached_tickers

    def _ensure_fresh(self, today: date) -> None:
        with self._lock:
            if self._cached_day == today:
                return
            self._cached_tickers = self._fetch(today)
            self._cached_day = today

    def _fetch(self, day: date) -> set[str]:
        if not self._api_key:
            if not self._warned_no_key:
                log.warning(
                    "FinnhubEarningsCalendar: no FINNHUB_API_KEY configured — "
                    "is_earnings_today will return False for every ticker (see .env.example)"
                )
                self._warned_no_key = True
            return set()

        try:
            resp = self._client.get(
                _FINNHUB_BASE_URL,
                params={"from": day.isoformat(), "to": day.isoformat(), "token": self._api_key},
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            if not self._warned_fetch_failure:
                log.exception(
                    "FinnhubEarningsCalendar: fetch failed for %s — is_earnings_today will "
                    "return stale/empty results until a future call succeeds", day.isoformat(),
                )
                self._warned_fetch_failure = True
            return self._cached_tickers

        self._warned_fetch_failure = False
        rows = payload.get("earningsCalendar", [])
        return {row["symbol"].upper().strip() for row in rows if row.get("symbol")}
