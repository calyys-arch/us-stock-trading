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

    # Data source — set once from configs/broker.yaml at EngineRuntime
    # construction, NOT a runtime toggle (mirrors strategy.yaml's
    # auto_execute philosophy: this is an operational decision, not a
    # dashboard button). "simulated" needs no external connection at all;
    # "ibkr_paper" requires IB Gateway/TWS to be running and logged into the
    # paper account — ibkr_*_connected reflect the REAL connection state so
    # the UI never silently pretends a live session exists. "futu_live"
    # needs scripts/capture_market_microstructure.py --source futu already
    # running (OpenD logged in for THAT process, not this one) —
    # futu_live_feed_active reflects the same "a tick actually arrived"
    # liveness signal as ibkr_feed_connected, via file-tailing instead of a
    # direct connection (see python/interfaces/futu_live_feed.py).
    data_source: str = "simulated"      # "simulated" | "ibkr_paper" | "futu_live"
    ibkr_broker_connected: bool = False  # meaningful only when data_source == "ibkr_paper"
    ibkr_feed_connected: bool = False    # meaningful only when data_source == "ibkr_paper"
    futu_live_feed_active: bool = False  # meaningful only when data_source == "futu_live"

    # Paper-forward experiment status (2026-08-15) — which strategies the
    # gateway currently has in its auto-execute allowlist, and whether the
    # pairs trend-efficiency gate is open. Empty / closed by default
    # (observe mode, fail-closed).
    armed_strategies: list = field(default_factory=list)
    pairs_regime_gate_open: bool = False
    pairs_regime_gate_reason: str = "not_yet_evaluated"
    live_gate_regime: str = "undecided"
    live_gate_policy: dict = field(default_factory=dict)

    # Live tick-feed symbol universe (EngineRuntime.symbols) — used by the
    # dashboard's Symbol Chart panel as quick-select chips. NOT the same as
    # configs/universe.yaml's larger backtest/research universe.
    symbols: list = field(default_factory=list)

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
                "data_source": self.data_source,
                "ibkr_broker_connected": self.ibkr_broker_connected,
                "ibkr_feed_connected": self.ibkr_feed_connected,
                "futu_live_feed_active": self.futu_live_feed_active,
                "armed_strategies": list(self.armed_strategies),
                "pairs_regime_gate_open": self.pairs_regime_gate_open,
                "pairs_regime_gate_reason": self.pairs_regime_gate_reason,
                "live_gate_regime": self.live_gate_regime,
                "live_gate_policy": dict(self.live_gate_policy),
                "symbols": list(self.symbols),
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
