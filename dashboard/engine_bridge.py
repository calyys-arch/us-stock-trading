"""
Engine bridge — wires MessageBus + DataEngine + ExecutionGateway together
and mirrors every published event into DashboardState so the FastAPI REST
endpoints (dashboard/app.py) have something live to serve. This is the
equivalent of forex-trading's dashboard/engine_bridge.py, trimmed to this
MVP's scope (no NiceGUI, no Greycat, no news pipeline).

Safety default: the engine always starts in mode="observe" — auto-execution
requires an explicit, separate opt-in (mirrors forex-trading's design where
`auto_execute` and the gateway mode are two independent gates; see
tests/test_config_enforcement.py). This bridge intentionally does NOT expose
a "flip to auto" REST endpoint in the MVP — enabling live order submission
is a deliberate, out-of-band operational decision, not a dashboard toggle.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from python.core.bus import MessageBus
from python.core.data_engine import DataEngine, ReferenceData
from python.core.execution_gateway import ExecutionGateway
from python.core.sim_broker import SimBroker
from python.interfaces.market_data import SimulatedFeed

from .state import DashboardState

log = logging.getLogger(__name__)


class EngineRuntime:
    def __init__(self, state: DashboardState, symbols: list[str] | None = None) -> None:
        self.state = state
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        self.bus = MessageBus()
        self.broker = SimBroker()
        self.gateway = ExecutionGateway(self.bus, self.broker, mode="observe", auto_execute_strategies=set())
        self.data_engine = DataEngine(self.bus, reference_data=ReferenceData())
        self._task: asyncio.Task | None = None

        self.bus.subscribe("snapshot", self._on_snapshot)
        self.bus.subscribe("execution_report", self._on_execution_report)

    async def _on_snapshot(self, snap) -> None:
        # MVP: no strategy wired to live snapshots yet (both MVP strategies
        # operate on daily bars, not tick snapshots) — this handler exists so
        # the dashboard can show "engine is alive and receiving data" via
        # account_summary/heartbeat rather than pretending strategies are
        # live-evaluating every tick.
        self.state.account_summary = self.broker.get_account_summary()

    async def _on_execution_report(self, report: dict) -> None:
        self.state.push_execution_report(report)

    async def start(self) -> None:
        if self._task and not self._task.done():
            log.warning("EngineRuntime.start: already running")
            return
        self.state.running = True
        self.state.started_at = datetime.utcnow().isoformat()
        feed = SimulatedFeed(self.symbols, duration_seconds=3600.0)

        async def _run():
            try:
                async for tick in feed.stream():
                    await self.data_engine.process_tick(tick)
            except asyncio.CancelledError:
                log.info("EngineRuntime: feed loop cancelled")
            except Exception:
                log.exception("EngineRuntime: feed loop crashed")
            finally:
                self.state.running = False

        self._task = asyncio.create_task(_run())
        log.info("EngineRuntime: started with %d symbols (SimulatedFeed)", len(self.symbols))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        log.info("EngineRuntime: stopped")
