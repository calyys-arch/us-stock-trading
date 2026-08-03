"""
Token-bucket rate limiter for external APIs.

forex-trading lesson #5 (docs/lessons_from_forex_trading.md): Alpha
Vantage's 25 requests/day limit was exhausted early in the day, after which
the system kept polling every ~15 seconds anyway, producing thousands of
wasted rejected-request log lines for the rest of the day instead of
backing off. This module gives every external API call site a shared,
explicit budget with a hard daily quota AND a smooth per-second token
bucket, so a burst of requests can't silently blow through the daily quota
before anyone notices.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    requests_per_second: float = 5.0
    daily_quota: int | None = None    # None = no hard daily cap


class RateLimiter:
    """Thread-safe token-bucket limiter with an optional hard daily quota.

    Usage:
        limiter = RateLimiter("yfinance", RateLimitConfig(requests_per_second=2, daily_quota=None))
        if not limiter.try_acquire():
            ... skip / back off ...
    """

    def __init__(self, name: str, config: RateLimitConfig) -> None:
        self.name = name
        self.cfg = config
        # Bucket capacity must be at least ONE whole token: try_acquire needs
        # tokens >= 1.0 to grant a request, so capping capacity at a
        # sub-1.0 requests_per_second (e.g. IBKR historical pacing at 0.1/s)
        # would make the bucket permanently ungrantable — every caller doing
        # `while not try_acquire(): sleep(...)` would hang forever.
        self._capacity = max(1.0, config.requests_per_second)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._daily_count = 0
        self._daily_reset_at = time.monotonic() + 86400
        self._quota_exhausted_logged = False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self.cfg.requests_per_second,
        )
        self._last_refill = now

        if now >= self._daily_reset_at:
            self._daily_count = 0
            self._daily_reset_at = now + 86400
            self._quota_exhausted_logged = False

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()

            if self.cfg.daily_quota is not None and self._daily_count >= self.cfg.daily_quota:
                if not self._quota_exhausted_logged:
                    log.warning(
                        "RateLimiter[%s]: daily quota (%d) exhausted — refusing further "
                        "requests until reset (single alert; no further spam this cycle)",
                        self.name, self.cfg.daily_quota,
                    )
                    self._quota_exhausted_logged = True
                return False

            if self._tokens < 1.0:
                return False

            self._tokens -= 1.0
            self._daily_count += 1
            return True

    @property
    def daily_count(self) -> int:
        return self._daily_count


class RateLimiterRegistry:
    """Process-wide registry so every call site shares the SAME limiter
    instance for a given API name, instead of each accidentally creating its
    own bucket (which would silently multiply the effective rate)."""

    def __init__(self) -> None:
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def get(self, name: str, config: RateLimitConfig | None = None) -> RateLimiter:
        with self._lock:
            if name not in self._limiters:
                self._limiters[name] = RateLimiter(name, config or RateLimitConfig())
            return self._limiters[name]


registry = RateLimiterRegistry()
