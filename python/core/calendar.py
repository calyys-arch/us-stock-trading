"""
NYSE trading calendar — replaces forex-trading's 24/5 session assumption.

forex-trading has no concept of exchange holidays, early closes, or DST
transitions (it just checks UTC weekday/hour). US equities need all three:
Thanksgiving/Christmas holidays, the day-after-Thanksgiving 13:00 ET early
close, and the fact that "09:30-16:00 ET" shifts relative to UTC across the
March/November DST boundary. Wraps `exchange_calendars` (XNYS) rather than
hand-rolling holiday tables, which is exactly the kind of subtly-wrong
one-off logic that caused the forex timestamp bug.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# Regular session hours (Chan / SEC convention). Actual open/close for a
# given date (early closes etc.) come from exchange_calendars when available.
_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)
# Flatten day-trading positions before the close to avoid a last-minute
# unfilled market order carrying into an unintended overnight position
# (mirrors forex-trading's live_watchdog UTC-21:00 EOD flatten pattern).
_INTRADAY_FLATTEN_BUFFER = timedelta(minutes=5)


@lru_cache(maxsize=1)
def _xnys():
    try:
        import exchange_calendars as xcals

        return xcals.get_calendar("XNYS")
    except Exception:
        return None


def _to_et(dt: datetime) -> datetime:
    """Timezone-naive datetimes represent ET wall-clock time throughout
    this live pipeline (python/interfaces/market_data.py's SimulatedFeed
    virtual clock starts at 09:30 to mean 09:30 ET;
    python/microstructure/context.py's whole bar-index convention is a
    tz-naive ET DatetimeIndex — see that module's docstring; a MicroSignal
    built from those bars therefore carries a tz-naive `signal_time` too).
    `.replace(tzinfo=...)` ATTACHES ET without converting the wall-clock
    value, which is exactly what "this naive value already IS ET" means —
    unlike `.astimezone()`, which (a) assumes the *server's local*
    timezone for naive stdlib datetimes (silently wrong on a non-ET host)
    and (b) raises TypeError outright for a naive `pandas.Timestamp`
    (pandas is stricter here than stdlib datetime), which crashed
    ExecutionGateway._on_microstructure_order the first time a live
    microstructure signal actually reached it. Timezone-AWARE datetimes
    (every real IBKR tick — always UTC, see python/interfaces/ibkr_feed.py)
    are still correctly CONVERTED to ET as before."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def is_trading_day(dt: datetime) -> bool:
    cal = _xnys()
    d = _to_et(dt).date()
    if cal is not None:
        return bool(cal.is_session(d.isoformat()))
    # Fallback (exchange_calendars not installed): weekday-only, no holidays.
    # Only used for quick local smoke tests — never for backtest research,
    # which must have exchange_calendars installed.
    return d.weekday() < 5


def session_open_close(dt: datetime) -> tuple[datetime, datetime] | None:
    """Return (open, close) in ET for the trading day containing `dt`,
    or None if `dt`'s date is not a trading day. Accounts for early closes
    (e.g. day after Thanksgiving) via exchange_calendars when available."""
    cal = _xnys()
    d = _to_et(dt).date()
    if cal is not None:
        if not cal.is_session(d.isoformat()):
            return None
        open_utc = cal.session_open(d.isoformat()).to_pydatetime()
        close_utc = cal.session_close(d.isoformat()).to_pydatetime()
        return open_utc.astimezone(_ET), close_utc.astimezone(_ET)
    if d.weekday() >= 5:
        return None
    open_et = datetime.combine(d, _RTH_OPEN, tzinfo=_ET)
    close_et = datetime.combine(d, _RTH_CLOSE, tzinfo=_ET)
    return open_et, close_et


def is_regular_trading_hours(dt: datetime) -> bool:
    session = session_open_close(dt)
    if session is None:
        return False
    open_et, close_et = session
    now_et = _to_et(dt)
    return open_et <= now_et < close_et


def is_intraday_flatten_window(dt: datetime) -> bool:
    """True once we're within the pre-close buffer — day-trading strategies
    must be flat or actively flattening by this point."""
    session = session_open_close(dt)
    if session is None:
        return False
    _, close_et = session
    now_et = _to_et(dt)
    return (close_et - _INTRADAY_FLATTEN_BUFFER) <= now_et < close_et


def next_trading_day(dt: datetime) -> datetime:
    cal = _xnys()
    d = _to_et(dt).date()
    if cal is not None:
        nxt = cal.next_session(d.isoformat())
        return nxt.to_pydatetime().astimezone(_ET)
    probe = dt
    for _ in range(10):
        probe = probe + timedelta(days=1)
        if is_trading_day(probe):
            return probe
    raise RuntimeError("next_trading_day: fallback search exceeded 10 days")
