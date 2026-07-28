"""
DashboardState — single in-memory source of truth the FastAPI app reads
from and the (future) live engine writes to. Kept intentionally minimal for
the MVP: no Greycat, no NiceGUI — this is a plain data holder consumed by
dashboard/app.py's REST endpoints, mirroring forex-trading's
dashboard/state.py role but without any UI framework coupling (the UI is a
separate React/Vite app, not server-rendered).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DashboardState:
    running: bool = False
    mode: str = "observe"          # "observe" | "auto" — mirrors ExecutionGateway.mode
    started_at: str | None = None

    # Strategy A — pairs
    open_pairs: list = field(default_factory=list)          # [OpenPairPosition-like dicts]
    pair_candidates: list = field(default_factory=list)      # latest pair_scanner.scan() results

    # Strategy B — cross-sectional
    latest_portfolio_target: dict | None = None
    latest_portfolio_weights: dict = field(default_factory=dict)

    # Shared
    recent_signals: list = field(default_factory=list)       # newest first, capped
    recent_execution_reports: list = field(default_factory=list)
    latest_backtest_summary: dict | None = None
    account_summary: dict = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    _MAX_SIGNALS = 200
    _MAX_REPORTS = 200

    def push_signal(self, signal: dict) -> None:
        with self._lock:
            self.recent_signals.insert(0, signal)
            del self.recent_signals[self._MAX_SIGNALS:]

    def push_execution_report(self, report: dict) -> None:
        with self._lock:
            self.recent_execution_reports.insert(0, report)
            del self.recent_execution_reports[self._MAX_REPORTS:]

    def set_portfolio_target(self, target: dict) -> None:
        with self._lock:
            self.latest_portfolio_target = target
            self.latest_portfolio_weights = target.get("weights", {})

    def set_backtest_summary(self, summary: dict) -> None:
        with self._lock:
            self.latest_backtest_summary = summary

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "mode": self.mode,
                "started_at": self.started_at,
                "open_pairs": list(self.open_pairs),
                "pair_candidates": list(self.pair_candidates),
                "latest_portfolio_target": self.latest_portfolio_target,
                "latest_portfolio_weights": dict(self.latest_portfolio_weights),
                "recent_signals": list(self.recent_signals[:50]),
                "recent_execution_reports": list(self.recent_execution_reports[:50]),
                "latest_backtest_summary": self.latest_backtest_summary,
                "account_summary": dict(self.account_summary),
                "server_time": datetime.utcnow().isoformat(),
            }


state = DashboardState()
