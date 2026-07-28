"""
DataRecorder — SQLite-backed point-in-time storage for daily bars, snapshots,
and signals. Mirrors forex-trading's data/market.db pattern (WAL-mode
SQLite for concurrent dashboard reads while the live engine writes).

Point-in-time discipline: `record_daily_bar()` stores a `recorded_at`
timestamp separate from the bar's own `date` — this lets later analysis
distinguish "the close price for 2026-01-15" from "what we believed the
close price for 2026-01-15 was as of the time we recorded it", which
matters if a data vendor later restates a bar (rare but real, e.g. after a
correction to a dividend adjustment).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
"""


class DataRecorder:
    def __init__(self, db_path: str | Path = "data/market.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_daily_bar(self, code: str, date: str, open_: float, high: float, low: float, close: float, volume: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO daily_bars (code, date, open, high, low, close, volume, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (code, date, open_, high, low, close, volume, datetime.utcnow().isoformat()),
            )

    def record_signal(self, signal_id: str, strategy: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO signals (id, strategy, payload, created_at) VALUES (?, ?, ?, ?)",
                (signal_id, strategy, json.dumps(payload, default=str), datetime.utcnow().isoformat()),
            )

    def record_execution_report(self, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO execution_reports (payload, created_at) VALUES (?, ?)",
                (json.dumps(payload, default=str), datetime.utcnow().isoformat()),
            )

    def get_daily_bars(self, code: str, start: str, end: str):
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT date, open, high, low, close, volume FROM daily_bars "
                "WHERE code = ? AND date >= ? AND date <= ? ORDER BY date",
                (code, start, end),
            )
            return cursor.fetchall()

    def get_signals(self, strategy: str | None = None, limit: int = 500):
        with self._connect() as conn:
            if strategy:
                cursor = conn.execute(
                    "SELECT id, strategy, payload, created_at FROM signals WHERE strategy = ? "
                    "ORDER BY created_at DESC LIMIT ?", (strategy, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT id, strategy, payload, created_at FROM signals ORDER BY created_at DESC LIMIT ?", (limit,),
                )
            return cursor.fetchall()
