"""
Engine bridge — wires MessageBus + DataEngine + ExecutionGateway together
and mirrors every published event into DashboardState so the FastAPI REST
endpoints (dashboard/app.py) have something live to serve. This is the
equivalent of forex-trading's dashboard/engine_bridge.py, trimmed to this
MVP's scope (no NiceGUI, no Greycat). Earnings/news reference data ARE wired
in (python/interfaces/finnhub_calendar.py, python/interfaces/finnhub_news.py)
but the live engine still only runs DataEngine's tick pipeline — there is no
daily scheduler here that calls a PortfolioStrategy; see README.md's "Known
limitations" for where the earnings/news exclusion actually takes effect
today (scripts/pick_10.py).

Safety default: the engine always starts in mode="observe" — auto-execution
requires an explicit, separate opt-in (mirrors forex-trading's design where
`auto_execute` and the gateway mode are two independent gates; see
tests/test_config_enforcement.py). This bridge intentionally does NOT expose
a "flip to auto" REST endpoint in the MVP — enabling live order submission
is a deliberate, out-of-band operational decision, not a dashboard toggle.

Data source is the SAME kind of deliberate, config-file-only decision — see
configs/broker.yaml — across all THREE options:
  - data_source: simulated  — in-memory SimulatedFeed + SimBroker, zero
    external connections (the checked-in default).
  - data_source: ibkr_paper — real IB Gateway/TWS connection for BOTH feed
    and broker; DashboardState.ibkr_broker_connected / ibkr_feed_connected
    reflect the true connection state.
  - data_source: futu_live  — real Futu-sourced tick/L2 data, but via
    python/interfaces/futu_live_feed.py's FutuLiveFeed, which TAILS the
    JSONL files scripts/capture_market_microstructure.py --source futu is
    already writing (data/ticks/, data/depth/) rather than opening a
    second Futu/OpenD connection of its own. Feed-only: the broker stays
    SimBroker (observe-only real-data mode, see docs/microstructure_pivot_plan.md
    §7's Phase 2 — this never touches ExecutionGateway's mode or broker
    selection).
Clicking "Start (Paper)" in the UI does NOT by itself require any external
process to be running for the "simulated" default; it only does for
ibkr_paper (IB Gateway/TWS) or futu_live (the capture script + OpenD it
depends on, already running independently of this dashboard) — the
dashboard never silently pretends to be live when it isn't.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from python.analytics.macro_beta_gate import LiveMacroGate, load_index_1m_bars
from python.backtest.intraday_engine import SIGNAL_PARAM_KEYS
from python.core.bus import MessageBus
from python.core.data_engine import DataEngine, ReferenceData
from python.core.execution_gateway import ExecutionGateway
from python.core.paper_forward import ABSORPTION_BREAKOUT_UNIVERSE, PAPER_AUTO_ALLOWLIST, RETIRED_MICRO_SIGNALS
from python.core.risk_engine import RiskEngine, load_risk_config
from python.core.sim_broker import SimBroker
from python.interfaces.finnhub_calendar import FinnhubEarningsCalendar
from python.interfaces.finnhub_news import FinnhubNewsSignal
from python.interfaces.futu_live_feed import FutuLiveFeed
from python.interfaces.market_data import SimulatedFeed
from python.microstructure.signal_journal import SignalJournal

from .live_microstructure_scheduler import LIVE_SIGNALS, MicrostructureScheduler
from .live_pairs_scheduler import REGIME_PROXY_SYMBOL, LivePairsScheduler
from .state import DashboardState

log = logging.getLogger(__name__)

_BROKER_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "broker.yaml"
_STRATEGY_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "strategy.yaml"
_VALID_DATA_SOURCES = ("simulated", "ibkr_paper", "futu_live")
# Pre-close flatten check cadence — matches the general "poll, don't
# schedule to the second" pattern used elsewhere in this bridge (e.g. the
# 2s dashboard state poll); 20s is frequent enough relative to
# calendar.is_intraday_flatten_window's 5-minute buffer to reliably catch
# the window without needing a precise one-shot timer.
_FLATTEN_CHECK_INTERVAL_SEC = 20.0
_PAIRS_EVAL_INTERVAL_SEC = 120.0
_MACRO_REFRESH_INTERVAL_SEC = 300.0


def _load_strategy_config() -> dict:
    try:
        with open(_STRATEGY_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("engine_bridge: %s not found", _STRATEGY_CONFIG_PATH)
        return {}


def _load_paper_auto_strategies() -> set[str]:
    """Strategies that may be armed for PAPER auto-execution.

    Intersection of (a) `auto_execute: true` in configs/strategy.yaml,
    (b) `enabled: true`, and (c) PAPER_AUTO_ALLOWLIST. Retired
    microstructure names are excluded even if a human later flips their
    yaml flag — that is the live footgun this function exists to close.
    This is a paper-forward-test / regime-gated arm, NOT a WFO GO
    promotion.
    """
    cfg = _load_strategy_config()
    armed: set[str] = set()
    for name, strategy_cfg in cfg.items():
        if not isinstance(strategy_cfg, dict):
            continue
        if name in RETIRED_MICRO_SIGNALS:
            continue
        if name not in PAPER_AUTO_ALLOWLIST:
            continue
        if strategy_cfg.get("enabled") and strategy_cfg.get("auto_execute"):
            armed.add(name)
    return armed


def _load_live_signal_params() -> dict[str, dict]:
    """Per-signal parameter dicts for the three LIVE microstructure signals
    (dashboard.live_microstructure_scheduler.LIVE_SIGNALS — l2_absorption
    deliberately excluded), read from configs/strategy.yaml using the SAME
    key set (python/backtest/intraday_engine.SIGNAL_PARAM_KEYS) the WFO
    backtester itself reads — so live evaluation always uses whatever
    parameters were actually promoted/validated, never a separately
    hand-maintained copy that could silently drift from it."""
    try:
        with open(_STRATEGY_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning(
            "engine_bridge: %s not found — live microstructure scheduler will use each signal's "
            "hardcoded evaluate_*() defaults", _STRATEGY_CONFIG_PATH,
        )
        return {}
    out: dict[str, dict] = {}
    for signal_name in LIVE_SIGNALS:
        base_cfg = cfg.get(signal_name, {}) or {}
        out[signal_name] = {k: base_cfg[k] for k in SIGNAL_PARAM_KEYS[signal_name] if k in base_cfg}
    return out


def _load_broker_config() -> dict:
    cfg: dict = {}
    try:
        with open(_BROKER_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("engine_bridge: %s not found — defaulting to data_source=simulated", _BROKER_CONFIG_PATH)

    cfg.setdefault("data_source", "simulated")
    ibkr = cfg.setdefault("ibkr", {}) or {}
    ibkr.setdefault("host", "127.0.0.1")
    ibkr.setdefault("feed_port", 4002)
    ibkr.setdefault("broker_port", 4002)
    ibkr.setdefault("feed_client_id", 11)
    ibkr.setdefault("broker_client_id", 21)
    cfg["ibkr"] = ibkr

    if cfg["data_source"] not in _VALID_DATA_SOURCES:
        log.warning(
            "engine_bridge: configs/broker.yaml data_source=%r is invalid (expected one of %s) — "
            "falling back to 'simulated'", cfg["data_source"], _VALID_DATA_SOURCES,
        )
        cfg["data_source"] = "simulated"
    return cfg


class EngineRuntime:
    def __init__(self, state: DashboardState, symbols: list[str] | None = None) -> None:
        self.state = state
        self.symbols = symbols or list(ABSORPTION_BREAKOUT_UNIVERSE)
        self.bus = MessageBus()

        self.broker_cfg = _load_broker_config()
        self.data_source: str = self.broker_cfg["data_source"]
        self.state.data_source = self.data_source
        self.state.symbols = list(self.symbols)

        # Safe, zero-connection default. Only swapped for a real IbkrBroker
        # inside start() — and only if configs/broker.yaml opted into
        # data_source: ibkr_paper — so constructing EngineRuntime (which
        # happens once at FastAPI process startup) never itself requires IB
        # Gateway/TWS to be running.
        self.broker = SimBroker()
        # Live snapshot price cache, keyed by symbol — wired into the
        # gateway's flatten price source below so
        # flatten_intraday_positions/flatten_position/emergency_flatten_all
        # actually work tonight instead of silently skip+warn (see
        # ExecutionGateway.set_price_lookup's docstring). Updated from
        # every "snapshot" event in _on_snapshot.
        self._latest_prices: dict[str, float] = {}
        self.gateway = ExecutionGateway(self.bus, self.broker, mode="observe", auto_execute_strategies=set())
        self.gateway.set_price_lookup(lambda code: self._latest_prices.get(code, 0.0))
        self.gateway.enable_gate_policy(True)
        # Both fail safe (return False) when FINNHUB_API_KEY is unset — see
        # .env.example — so wiring them in unconditionally is safe even
        # without a key configured.
        self._earnings_calendar = FinnhubEarningsCalendar()
        self._news_signal = FinnhubNewsSignal()
        self.data_engine = DataEngine(
            self.bus,
            reference_data=ReferenceData(is_earnings_today=self._earnings_calendar.is_earnings_today),
            news_event_checker=self._news_signal.has_company_news_today,
        )
        # Persists across the whole session (PDTTracker/DailyLossTracker
        # state must survive across bars, not be recreated per-signal) —
        # owned by EngineRuntime, not the gateway, since it gates SIGNAL
        # qualification (upstream of the bus), not order submission itself.
        self.risk_engine = RiskEngine(load_risk_config())
        # Signal journal (docs/microstructure_pivot_plan.md §7/§9): records
        # EVERY live microstructure MicroSignal that fires — approved or
        # RiskEngine-rejected — to data/signal_journal/<date>.jsonl. Always
        # constructed (not opt-in) for the live dashboard runtime; tests
        # that build a bare MicrostructureScheduler directly leave its own
        # `signal_journal` param at the default None (no disk writes) —
        # see that module's docstring.
        self.signal_journal = SignalJournal()
        self._macro_gate = LiveMacroGate(live_fetch=True)
        self.micro_scheduler = MicrostructureScheduler(
            self.bus, self.risk_engine, _load_live_signal_params(),
            get_account_equity=lambda: self.state.account_summary.get("NetLiquidation", 0.0),
            signal_journal=self.signal_journal,
            live_universe=frozenset(ABSORPTION_BREAKOUT_UNIVERSE),
            macro_gate=self._macro_gate,
        )
        strat_cfg = _load_strategy_config()
        self.pairs_scheduler = LivePairsScheduler(
            self.bus, self.risk_engine, strat_cfg.get("pairs_trading") or {},
            get_account_equity=lambda: self.state.account_summary.get("NetLiquidation", 0.0),
        )
        self._task: asyncio.Task | None = None
        self._flatten_task: asyncio.Task | None = None
        self._pairs_task: asyncio.Task | None = None
        self._macro_task: asyncio.Task | None = None
        self._ibkr_broker = None  # cached real IbkrBroker instance, once (re)connected

        self.bus.subscribe("snapshot", self._on_snapshot)
        self.bus.subscribe("execution_report", self._on_execution_report)

        if self.data_source == "ibkr_paper":
            log.warning(
                "EngineRuntime: configs/broker.yaml data_source=ibkr_paper — Start (Paper) will "
                "require IB Gateway/TWS running and logged into the PAPER account on %s:%s "
                "(feed) / %s:%s (broker).",
                self.broker_cfg["ibkr"]["host"], self.broker_cfg["ibkr"]["feed_port"],
                self.broker_cfg["ibkr"]["host"], self.broker_cfg["ibkr"]["broker_port"],
            )

    async def _on_snapshot(self, snap) -> None:
        self.state.account_summary = self.broker.get_account_summary()
        if self.data_source == "ibkr_paper":
            # A tick actually arrived, so the feed side of the IBKR
            # connection is demonstrably alive right now.
            self.state.ibkr_feed_connected = True
        elif self.data_source == "futu_live":
            # Mirrors ibkr_feed_connected's "a tick actually arrived"
            # liveness signal — true once FutuLiveFeed has successfully
            # tailed at least one real trade line for ANY symbol.
            self.state.futu_live_feed_active = True
        self._latest_prices[snap.code] = snap.price
        await self.micro_scheduler.on_snapshot(snap)

    async def _on_execution_report(self, report: dict) -> None:
        self.state.push_execution_report(report)

    async def start(self) -> None:
        if self._task and not self._task.done():
            log.warning("EngineRuntime.start: already running")
            return

        if self.data_source == "ibkr_paper":
            await self._connect_ibkr_broker()
            feed = self._make_ibkr_feed()
        elif self.data_source == "futu_live":
            # Feed-only swap — self.broker stays SimBroker (never touched
            # here), matching data_source: simulated's own scope. No new
            # Futu/OpenD connection is opened; see FutuLiveFeed's module
            # docstring for why (it tails scripts/capture_market_microstructure.py
            # --source futu's already-running output instead).
            feed = self._make_futu_live_feed()
        else:
            feed = SimulatedFeed(self.symbols, duration_seconds=3600.0)

        self.state.running = True
        self.state.started_at = datetime.utcnow().isoformat()

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
                if self.data_source == "ibkr_paper":
                    self.state.ibkr_feed_connected = False
                elif self.data_source == "futu_live":
                    self.state.futu_live_feed_active = False

        self._task = asyncio.create_task(_run())
        if self._flatten_task is None or self._flatten_task.done():
            self._flatten_task = asyncio.create_task(self._flatten_loop())
        if self._pairs_task is None or self._pairs_task.done():
            self._pairs_task = asyncio.create_task(self._pairs_loop())
        if self._macro_task is None or self._macro_task.done():
            self._macro_task = asyncio.create_task(self._macro_refresh_loop())
        # Fire-and-forget: best-effort context only (YDH/YDL, gap-trap),
        # must never delay/block Start — a symbol with no cached prior
        # session (data/history_1m/) just runs without that context, see
        # MicrostructureScheduler.seed_prior_session's docstring.
        asyncio.create_task(self._seed_prior_session_context())
        asyncio.create_task(self._seed_paper_forward_context())
        log.info(
            "EngineRuntime: started with %d symbols (data_source=%s)",
            len(self.symbols), self.data_source,
        )

    async def _flatten_loop(self) -> None:
        """Periodic pre-close flatten check for open intraday/microstructure
        positions — the live scheduler's half of
        python/core/calendar.py:is_intraday_flatten_window (the other half,
        the actual window-time check + never-a-market-order flatten logic,
        already lives in ExecutionGateway.flatten_intraday_positions; this
        loop's only job is to actually CALL that periodically during a live
        session, which nothing did before tonight). Runs continuously while
        the engine is started, regardless of observe/auto mode — same as
        the method it calls, this is a safety net, not gated by the
        auto-trading arm switch (in observe mode there are simply never any
        real positions for it to find)."""
        try:
            while True:
                await asyncio.sleep(_FLATTEN_CHECK_INTERVAL_SEC)
                try:
                    await self.gateway.flatten_intraday_positions("microstructure")
                except Exception:
                    log.exception("EngineRuntime: pre-close flatten check failed")
        except asyncio.CancelledError:
            pass

    async def _pairs_loop(self) -> None:
        """Periodic daily-bar pairs evaluation. Exits always run; new
        entries only fire when the trend-efficiency gate is open."""
        try:
            while True:
                await asyncio.sleep(_PAIRS_EVAL_INTERVAL_SEC)
                try:
                    result = await self.pairs_scheduler.evaluate_once()
                    self.state.pairs_regime_gate_open = self.pairs_scheduler.regime_gate_open
                    self.state.pairs_regime_gate_reason = self.pairs_scheduler.regime_gate_reason
                    self._sync_gate_policy()
                    self.state.open_pairs = [
                        {
                            "code_a": p.code_a, "code_b": p.code_b,
                            "side": p.side.value, "entry_z": p.entry_z,
                            "entry_time": p.entry_time.isoformat() if hasattr(p.entry_time, "isoformat") else str(p.entry_time),
                        }
                        for p in self.pairs_scheduler.position_manager.open_positions
                    ]
                    if result.get("entries") or result.get("exits"):
                        log.info("EngineRuntime: pairs cycle %s", result)
                except Exception:
                    log.exception("EngineRuntime: pairs evaluation cycle failed")
        except asyncio.CancelledError:
            pass

    async def _macro_refresh_loop(self) -> None:
        """Re-fetch QQQ/SPY/XLK 1m during the session so the fail-closed
        macro gate has the current minute, not only the bars from Start."""
        try:
            while True:
                await asyncio.sleep(_MACRO_REFRESH_INTERVAL_SEC)
                try:
                    loop = asyncio.get_event_loop()
                    index_bars = await loop.run_in_executor(
                        None, lambda: load_index_1m_bars(lookback_days=5, fetch_live=True),
                    )
                    self._macro_gate.set_index_bars(index_bars)
                    log.info(
                        "EngineRuntime: macro beta gate refreshed (%d index symbols)",
                        len(index_bars),
                    )
                except Exception:
                    log.exception("EngineRuntime: macro beta gate refresh failed — gate stays on last seed")
        except asyncio.CancelledError:
            pass

    async def _seed_paper_forward_context(self) -> None:
        """Best-effort: load cached QQQ/SPY/XLK 1m for the macro gate and
        a daily close panel for pairs + the SPY regime proxy. Failures
        leave the respective gate CLOSED (no trade), never ungated."""
        loop = asyncio.get_event_loop()

        def _load_macro() -> dict:
            return load_index_1m_bars(lookback_days=5, fetch_live=True)

        try:
            index_bars = await loop.run_in_executor(None, _load_macro)
            self._macro_gate.set_index_bars(index_bars)
            log.info(
                "EngineRuntime: macro beta gate seeded with %d index symbols (%s)",
                len(index_bars), ",".join(sorted(index_bars)) or "none — fail-closed",
            )
        except Exception:
            log.exception("EngineRuntime: macro beta gate seed failed — gate stays fail-closed")

        def _load_regime_proxy():
            from python.analytics.trend_efficiency_gate import load_regime_proxy_close
            return load_regime_proxy_close(REGIME_PROXY_SYMBOL)

        try:
            import pandas as pd

            regime = await loop.run_in_executor(None, _load_regime_proxy)
            if regime is None or regime.empty:
                log.warning("EngineRuntime: regime proxy missing — live gates stay fail-closed")
            else:
                self.pairs_scheduler.set_regime_close(regime)
                self.pairs_scheduler.evaluate_regime_gate(pd.Timestamp.now().normalize())
                self.state.pairs_regime_gate_open = self.pairs_scheduler.regime_gate_open
                self.state.pairs_regime_gate_reason = self.pairs_scheduler.regime_gate_reason
                self._sync_gate_policy()
                log.info(
                    "EngineRuntime: regime proxy seeded (%d days); gate %s (%s)",
                    len(regime),
                    "OPEN" if self.pairs_scheduler.regime_gate_open else "CLOSED",
                    self.pairs_scheduler.regime_gate_reason,
                )
        except Exception:
            log.exception("EngineRuntime: regime proxy seed failed — live gates stay fail-closed")

        self._pairs_seed_task = asyncio.create_task(self._seed_pairs_panel())

    async def _seed_pairs_panel(self) -> None:
        """Background: 66-ETF daily panel. Must not block Start or the
        tape classifier — absorption uses the SPY proxy seeded above."""
        loop = asyncio.get_event_loop()

        def _load_pairs_panel():
            import pandas as pd

            from python.backtest.pairs_scan_engine import load_pairs_universe, pairs_buckets
            from python.data.price_cache import get_cached_price_panel

            universe = load_pairs_universe()
            symbols = sorted({c for codes in pairs_buckets(universe).values() for c in codes})
            if REGIME_PROXY_SYMBOL not in symbols:
                symbols.append(REGIME_PROXY_SYMBOL)
            end = pd.Timestamp.now().normalize()
            start = end - pd.Timedelta(days=800)
            panel, _flags, _meta = get_cached_price_panel(symbols, start, end)
            return panel["close"].unstack("code").sort_index()

        try:
            import pandas as pd

            close = await loop.run_in_executor(None, _load_pairs_panel)
            regime = close[REGIME_PROXY_SYMBOL] if REGIME_PROXY_SYMBOL in close.columns else None
            self.pairs_scheduler.set_panels(close, regime)
            if self.pairs_scheduler.regime_gate_reason == "not_yet_evaluated":
                self.pairs_scheduler.evaluate_regime_gate(pd.Timestamp.now().normalize())
                self.state.pairs_regime_gate_open = self.pairs_scheduler.regime_gate_open
                self.state.pairs_regime_gate_reason = self.pairs_scheduler.regime_gate_reason
                self._sync_gate_policy()
            log.info(
                "EngineRuntime: pairs panel seeded (%d days x %d names); regime gate %s (%s)",
                len(close), close.shape[1],
                "OPEN" if self.pairs_scheduler.regime_gate_open else "CLOSED",
                self.pairs_scheduler.regime_gate_reason,
            )
        except Exception:
            log.exception("EngineRuntime: pairs panel seed failed — pairs stay idle / fail-closed")

    async def _seed_prior_session_context(self) -> None:
        """Best-effort: load each symbol's most recent CACHED 1-minute
        session (data/history_1m/, via python/data/intraday_cache.py) as
        prior-day context for sweep_reclaim's YDH/YDL and orb_vwap's
        gap-trap rule. Cache-only (never a live IB fetch, matching
        get_cached_intraday_panel's own contract) and entirely optional —
        a symbol with no cached session, or if the cache read fails
        outright, just runs live tonight without that context rather than
        blocking Start or crashing it."""
        import pandas as pd

        from python.data.intraday_cache import get_cached_intraday_panel

        loop = asyncio.get_event_loop()
        end = pd.Timestamp.now().normalize()
        start = end - pd.Timedelta(days=10)

        def _load() -> dict[str, tuple]:
            out: dict[str, tuple] = {}
            for symbol in self.symbols:
                try:
                    panel = get_cached_intraday_panel([symbol], start, end)
                    df = panel.xs(symbol, level="code").sort_index()
                except Exception:
                    continue
                prior_dates = [d for d in sorted(set(df.index.normalize())) if d < end]
                if not prior_dates:
                    continue
                bars = df.loc[df.index.normalize() == prior_dates[-1]]
                if not bars.empty:
                    out[symbol] = (bars, float(bars["close"].iloc[-1]))
            return out

        try:
            sessions = await loop.run_in_executor(None, _load)
        except Exception:
            log.exception("EngineRuntime: prior-session microstructure context seeding failed — continuing without it")
            return

        for symbol, (bars, prior_close) in sessions.items():
            self.micro_scheduler.seed_prior_session(symbol, bars, prior_close)
        log.info(
            "EngineRuntime: seeded prior-session microstructure context for %d/%d symbols",
            len(sessions), len(self.symbols),
        )

    def _make_ibkr_feed(self):
        from python.interfaces.ibkr_feed import IbkrFeed

        ibkr_cfg = self.broker_cfg["ibkr"]
        return IbkrFeed(
            self.symbols,
            host=ibkr_cfg["host"],
            port=int(ibkr_cfg["feed_port"]),
            client_id=int(ibkr_cfg["feed_client_id"]),
        )

    def _make_futu_live_feed(self) -> FutuLiveFeed:
        """No broker.yaml settings needed here (unlike ibkr_cfg above) —
        FutuLiveFeed only reads local JSONL files at their standard paths
        (python/interfaces/ibkr_tick_capture.TICKS_DIR/DEPTH_DIR); it never
        opens a Futu/OpenD socket itself, so none of configs/broker.yaml's
        `futu:` connection settings (host/port/rsa_key_path — those are for
        scripts/capture_market_microstructure.py --source futu, which must
        already be running) apply to this feed."""
        return FutuLiveFeed(self.symbols)

    async def _connect_ibkr_broker(self) -> None:
        """Connect (or reuse) the real IbkrBroker. The blocking connect
        (IbkrBroker.__init__ dials IB Gateway/TWS synchronously, up to ~10s)
        runs in a thread executor so the FastAPI event loop — and the
        /api/state poll the UI relies on every 2s — stays responsive while
        waiting on IB Gateway/TWS instead of freezing the whole server."""
        from python.interfaces.ibkr_broker import IbkrBroker

        if self._ibkr_broker is not None and self._ibkr_broker.is_connected:
            self.broker = self._ibkr_broker
            self.gateway.set_broker(self._ibkr_broker)
            self.state.ibkr_broker_connected = True
            await self._refresh_account_summary()
            return

        ibkr_cfg = self.broker_cfg["ibkr"]
        loop = asyncio.get_event_loop()
        try:
            broker = await loop.run_in_executor(
                None,
                lambda: IbkrBroker(
                    host=ibkr_cfg["host"],
                    port=int(ibkr_cfg["broker_port"]),
                    client_id=int(ibkr_cfg["broker_client_id"]),
                ),
            )
        except Exception:
            log.exception("EngineRuntime: IbkrBroker construction raised unexpectedly")
            self.state.ibkr_broker_connected = False
            return

        self._ibkr_broker = broker
        self.broker = broker
        self.gateway.set_broker(broker)
        self.state.ibkr_broker_connected = broker.is_connected
        if not self.state.ibkr_broker_connected:
            log.warning(
                "EngineRuntime: IbkrBroker did not connect — is IB Gateway/TWS running and "
                "logged into the PAPER account on %s:%s? Orders will be rejected until it "
                "reconnects (retried automatically on the next Start).",
                ibkr_cfg["host"], ibkr_cfg["broker_port"],
            )
        else:
            # Don't wait for the first live tick (DataEngine's "snapshot"
            # event) to show real account numbers — NetLiquidation/
            # BuyingPower are available immediately after connect, and
            # outside market hours no tick may arrive for a long time.
            await self._refresh_account_summary()

    async def _refresh_account_summary(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            self.state.account_summary = await loop.run_in_executor(None, self.broker.get_account_summary)
        except Exception:
            log.exception("EngineRuntime: get_account_summary failed")

    def _sync_gate_policy(self) -> None:
        """Push tape + volatility into the situation catalog and snapshot."""
        from python.analytics.gate_policy import (
            REGIME_UNDECIDED,
            classify_volatility,
            regime_from_pairs_gate,
            summarize_active,
        )

        reason = self.pairs_scheduler.regime_gate_reason
        if reason in ("not_yet_evaluated", "") or reason is None:
            tape = REGIME_UNDECIDED
        else:
            tape = regime_from_pairs_gate(self.pairs_scheduler.regime_gate_open)
        vol = classify_volatility(self.pairs_scheduler.regime_close)
        self.gateway.set_market_features(tape, vol)
        summary = summarize_active(tape, vol)
        self.state.live_gate_regime = f"{tape}/{vol}"
        self.state.live_gate_policy = summary

    async def enable_auto_trading(self) -> set[str]:
        """Arms real order submission for the current session ("Start Auto
        Trading" in the dashboard). This flips BOTH keys of the gateway's
        two-key AND gate (see ExecutionGateway.__init__ docstring) in
        memory only — configs/strategy.yaml on disk is never modified, so a
        server restart always comes back up safely in observe mode. Callers
        (dashboard/app.py) are expected to require an explicit confirmation
        before hitting this — it is a genuine "orders may now go to the
        market" switch, not a cosmetic UI toggle."""
        strategies = _load_paper_auto_strategies()
        self.gateway.set_auto_execute_strategies(strategies)
        self.gateway.set_mode("auto")
        self._sync_gate_policy()
        self.state.mode = "auto"
        self.state.armed_strategies = sorted(strategies)
        log.warning(
            "EngineRuntime: PAPER AUTO TRADING ARMED (strategies=%s, data_source=%s) — "
            "paper orders may now be submitted for the allowlisted names only "
            "(NOT a WFO GO promotion)",
            sorted(strategies), self.data_source,
        )
        return strategies

    async def disable_auto_trading(self) -> None:
        self.gateway.set_mode("observe")
        self.gateway.set_auto_execute_strategies(set())
        self.state.mode = "observe"
        self.state.armed_strategies = []
        log.info("EngineRuntime: auto trading disarmed, back to observe mode")

    async def emergency_flatten_all(self) -> list[dict]:
        """"Exit All Positions" — closes every open position at the current
        broker immediately, independent of engine running/mode state (it
        only needs a broker to exist, so it still works after Stop as long
        as the IBKR connection — or SimBroker — is present)."""
        return await self.gateway.emergency_flatten_all()

    async def flatten_position(self, code: str) -> Optional[dict]:
        """"Exit" button next to a single symbol's row in the dashboard's
        Positions panel — closes just that one position."""
        return await self.gateway.flatten_position(code)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._flatten_task:
            self._flatten_task.cancel()
            try:
                await self._flatten_task
            except asyncio.CancelledError:
                pass
            self._flatten_task = None
        if self._pairs_task:
            self._pairs_task.cancel()
            try:
                await self._pairs_task
            except asyncio.CancelledError:
                pass
            self._pairs_task = None
        self.state.running = False
        # Stop always disarms auto trading — the next Start must re-arm it
        # explicitly rather than silently resuming a live-order session.
        await self.disable_auto_trading()

        if self.data_source == "ibkr_paper" and self._ibkr_broker is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._ibkr_broker.disconnect)
            except Exception:
                log.exception("EngineRuntime: IbkrBroker disconnect failed")
            self._ibkr_broker = None
            self.state.ibkr_broker_connected = False
            self.state.ibkr_feed_connected = False
            # Revert to the safe in-memory default so no residual bus
            # handler can ever forward an order to a now-disconnected IBKR
            # session between Stop and the next Start.
            self.broker = SimBroker()
            self.gateway.set_broker(self.broker)

        log.info("EngineRuntime: stopped")
