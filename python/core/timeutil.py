"""
Single source of truth for timestamp parsing / unit conversion.

Forex-trading lesson (docs/lessons_from_forex_trading.md #3): the Dukascopy
tick loader once treated microsecond epoch timestamps as nanoseconds,
silently shrinking simulated holding times by 1000x and invalidating an
entire research cycle before it was caught. Every timestamp conversion in
this codebase MUST go through one of the functions below, each of which
hard-asserts the resulting year is sane.

Historical Note: nothing here is exchange-timezone-aware — that is
python/core/calendar.py's job (NYSE sessions, holidays, early closes). This
module only guards against unit-of-epoch mistakes.
"""
from __future__ import annotations

from datetime import datetime, timezone

_MIN_SANE_YEAR = 2000
_MAX_SANE_YEAR = 2100


def _assert_sane(dt: datetime, source_desc: str) -> datetime:
    if not (_MIN_SANE_YEAR <= dt.year <= _MAX_SANE_YEAR):
        raise ValueError(
            f"timeutil: parsed timestamp {dt.isoformat()} from {source_desc} "
            f"has an insane year ({dt.year}); this almost always means an "
            f"epoch-unit bug (ns vs us vs ms vs s). Refusing to proceed."
        )
    return dt


def from_epoch_seconds(value: float, source_desc: str = "unknown") -> datetime:
    dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    return _assert_sane(dt, source_desc)


def from_epoch_millis(value: float, source_desc: str = "unknown") -> datetime:
    return from_epoch_seconds(float(value) / 1_000.0, source_desc)


def from_epoch_micros(value: float, source_desc: str = "unknown") -> datetime:
    return from_epoch_seconds(float(value) / 1_000_000.0, source_desc)


def from_epoch_nanos(value: float, source_desc: str = "unknown") -> datetime:
    return from_epoch_seconds(float(value) / 1_000_000_000.0, source_desc)


def from_epoch_auto(value: float, source_desc: str = "unknown") -> datetime:
    """Infer the epoch unit from magnitude and convert.

    Boundaries (roughly, for dates between 2001 and 2286):
      seconds : 1e9 .. 1e10
      millis  : 1e12 .. 1e13
      micros  : 1e15 .. 1e16
      nanos   : 1e18 .. 1e19

    Prefer an explicit from_epoch_* call when the source's unit is known —
    this function exists only for third-party data whose unit isn't
    documented, and it still runs the same sanity assertion as a backstop.
    """
    v = abs(float(value))
    if v < 1e11:
        return from_epoch_seconds(value, source_desc)
    if v < 1e14:
        return from_epoch_millis(value, source_desc)
    if v < 1e17:
        return from_epoch_micros(value, source_desc)
    return from_epoch_nanos(value, source_desc)


def from_iso(value: str, source_desc: str = "unknown") -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _assert_sane(dt, source_desc)
