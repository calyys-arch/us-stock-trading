"""
Event blackout windows for intraday microstructure signals
(RiskEngine.qualify_microstructure_order) — earnings/8-K/economic-calendar
proximity is a HARD REJECT here, unlike python/signals/trap_detector.py's
identical-looking checks which are report-only. The two modules read the
SAME on-disk caches (data/news/, data/filings/, data/calendar/, populated by
scripts/refresh_event_data.py) but are deliberately NOT the same code: this
one lives in python/core (the risk-gating layer) and trap_report.py lives in
python/signals (the report-only diagnostic layer) — core must not depend on
signals (see architecture-rules.mdc's layering), so the ~20 lines of cache
readers below are intentionally duplicated rather than imported across that
boundary. If the cache format changes, update BOTH.

Cache-only, no network — same "reproducible/offline at decision time"
contract as every other evidence source in this system.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NEWS_DIR = Path("data/news")
FILINGS_DIR = Path("data/filings")
CALENDAR_DIR = Path("data/calendar")


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _earnings_dates_cached(symbol: str, year: int) -> set[str]:
    rows = _load_json(CALENDAR_DIR / f"earnings_{year}.json") or []
    symbol = symbol.upper()
    return {row["date"] for row in rows if str(row.get("symbol", "")).upper() == symbol and row.get("date")}


def _eight_k_acceptance_times_cached(symbol: str) -> list[str]:
    """`acceptance_datetime` (exact timestamp, unlike the day-only
    `filing_date`) for every cached 8-K — this is the one event type in
    this module with real intraday-time granularity."""
    cached = _load_json(FILINGS_DIR / "8k" / f"{symbol.upper()}.json")
    if cached is None:
        return []
    return [f["acceptance_datetime"] for f in cached.get("filings", []) if f.get("acceptance_datetime")]


def _econ_dates_cached() -> set[str]:
    dates: set[str] = set()
    if CALENDAR_DIR.exists():
        for path in CALENDAR_DIR.glob("economic_*.json"):
            for row in _load_json(path) or []:
                if row.get("time"):
                    dates.add(str(row["time"])[:10])
    return dates


def is_event_blackout(symbol: str, now: pd.Timestamp, window_minutes: int = 30) -> bool:
    """True if `now` falls in a blackout window for `symbol`.

    Granularity note: earnings-calendar and general-economic-calendar
    caches are DAY-only (no reliable intraday time), so those two block the
    ENTIRE calendar day regardless of `window_minutes` — there is no finer
    evidence to gate on. 8-K filings DO carry an exact `acceptance_datetime`
    in the cache, so those use a real +/-`window_minutes` window. This
    asymmetry is intentional, not an oversight: blocking a whole day on
    coarse evidence is the conservative failure mode; a minutes-level
    filter needs minutes-level evidence to mean anything.

    Missing cache data degrades to False/"no blackout", not None/"unknown"
    (unlike trap_detector's report-only near-event flags) — this function
    gates a HARD order reject, so it needs a definite answer; an empty
    data/calendar/ or data/filings/ is a reason to run
    scripts/refresh_event_data.py, not license to silently trade through
    every earnings date."""
    now = pd.Timestamp(now)
    day = str(now.date())

    if day in _econ_dates_cached():
        return True
    if day in _earnings_dates_cached(symbol, now.year):
        return True

    window = pd.Timedelta(minutes=window_minutes)
    for ts_str in _eight_k_acceptance_times_cached(symbol):
        try:
            event_ts = pd.Timestamp(ts_str)
        except Exception:
            continue
        if event_ts.tzinfo is not None and now.tzinfo is None:
            event_ts = event_ts.tz_localize(None)
        elif event_ts.tzinfo is None and now.tzinfo is not None:
            now = now.tz_localize(None)
        if abs(now - event_ts) <= window:
            return True
    return False
