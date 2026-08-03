"""
python/core/event_blackout.py tests — cache-only, tmp_path-isolated so these
never touch the real data/ directory. Confirms the day-only vs
minute-level asymmetry documented in is_event_blackout's docstring:
earnings/econ block the WHOLE day, 8-K uses a real +/-window on its
acceptance_datetime.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from python.core import event_blackout as eb


@pytest.fixture(autouse=True)
def _isolated_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "NEWS_DIR", tmp_path / "news")
    monkeypatch.setattr(eb, "FILINGS_DIR", tmp_path / "filings")
    monkeypatch.setattr(eb, "CALENDAR_DIR", tmp_path / "calendar")
    return tmp_path


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_no_cache_at_all_is_not_a_blackout():
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 10:00")) is False


def test_earnings_day_blocks_the_whole_day(_isolated_cache_dirs):
    _write_json(
        _isolated_cache_dirs / "calendar" / "earnings_2024.json",
        [{"symbol": "AAPL", "date": "2024-06-03"}],
    )
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 09:31")) is True
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 15:59")) is True
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-04 09:31")) is False


def test_earnings_day_only_blocks_the_matching_symbol(_isolated_cache_dirs):
    _write_json(
        _isolated_cache_dirs / "calendar" / "earnings_2024.json",
        [{"symbol": "MSFT", "date": "2024-06-03"}],
    )
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 10:00")) is False


def test_econ_event_day_blocks_every_symbol(_isolated_cache_dirs):
    _write_json(
        _isolated_cache_dirs / "calendar" / "economic_2024.json",
        [{"time": "2024-06-03T14:00:00Z", "event": "FOMC"}],
    )
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 10:00")) is True
    assert eb.is_event_blackout("ZZZZ", pd.Timestamp("2024-06-03 10:00")) is True


def test_eight_k_uses_minute_level_window(_isolated_cache_dirs):
    _write_json(
        _isolated_cache_dirs / "filings" / "8k" / "AAPL.json",
        {"filings": [{"acceptance_datetime": "2024-06-03T14:30:00", "filing_date": "2024-06-03"}]},
    )
    # Inside the +/-30min window around 14:30 -> blackout.
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 14:45:00"), window_minutes=30) is True
    # Same day but well outside the window -> NOT blocked by the 8-K
    # (day-only earnings/econ caches are empty here, so this should be False).
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 09:35:00"), window_minutes=30) is False


def test_eight_k_window_respects_configured_minutes(_isolated_cache_dirs):
    _write_json(
        _isolated_cache_dirs / "filings" / "8k" / "AAPL.json",
        {"filings": [{"acceptance_datetime": "2024-06-03T14:30:00", "filing_date": "2024-06-03"}]},
    )
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 14:35:00"), window_minutes=2) is False
    assert eb.is_event_blackout("AAPL", pd.Timestamp("2024-06-03 14:31:00"), window_minutes=2) is True
