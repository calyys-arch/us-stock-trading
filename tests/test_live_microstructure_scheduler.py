"""
dashboard/live_microstructure_scheduler.py — the live glue between
DataEngine's bus("snapshot") and the microstructure qualify/publish path.
Decision chart is 5 minutes (1m is ingest only). These tests deliberately
do NOT re-test ExecutionGateway's own two-key auto_execute gate (already
covered by tests/test_never_market_orders.py and
tests/test_config_enforcement.py) — they test that THIS module correctly
feeds that existing, unchanged gate: publishes a qualified order when a
signal fires on a completed 5m bar during RTH, stays silent on an
incomplete 5m bin and outside RTH, never touches l2_absorption, and
survives one symbol's evaluation blowing up.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import dashboard.live_microstructure_scheduler as scheduler_module
from dashboard.live_microstructure_scheduler import LIVE_SIGNALS, MicrostructureScheduler
from python.core.bus import MessageBus
from python.core.execution_gateway import ExecutionGateway
from python.core.risk_engine import RiskConfig, RiskEngine
from python.core.sim_broker import SimBroker
from python.core.types import Candle, MarketSnapshot

# 2026-03-02 is a Monday, comfortably before the March DST switch (2026-03-08).
# 14:30 UTC == 09:30 ET (session open); 14:35 UTC == 09:35 ET.
SESSION_OPEN = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
RTH_START = datetime(2026, 3, 2, 14, 35, tzinfo=timezone.utc)
OUTSIDE_RTH = datetime(2026, 3, 2, 22, 0, tzinfo=timezone.utc)  # 17:00 ET — after the 16:00 close

# absorption_breakout needs max(level, volume, atr lookback)+2 = 22 decision
# bars. 22 closed 5m bins = 110 closed 1m bars (09:30 through 11:19 ET).
_ABSORPTION_CLOSED_1M = 110

_STRATEGY_PARAMS = {
    "sweep_reclaim": {"sweep_min_atr": 0.15, "reclaim_bars": 3, "stop_atr_mult": 0.25},
    "fvg_retest": {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10},
    "orb_vwap": {"or_minutes": 15, "vwap_side_filter": True},
    "absorption_breakout": {"volume_mult": 3.0, "breakout_atr_mult": 0.5, "stop_atr_mult": 0.5},
}


def _candle(symbol: str, ts, o: float, h: float, l: float, c: float, v: float = 1000.0) -> Candle:
    return Candle(code=symbol, timeframe="1m", open=o, high=h, low=l, close=c, volume=v, turnover=o * v, timestamp=ts)


def _absorption_1m_candles(
    symbol: str,
    *,
    n_closed: int = _ABSORPTION_CLOSED_1M,
    start=SESSION_OPEN,
    fire_last_5m_bin: bool = False,
    fire_last_1m: bool = False,
) -> list[Candle]:
    """Closed 1m bars from 09:30 ET plus one in-progress bar (CandleBuilder
    shape). `fire_last_5m_bin` makes the last completed 5m OHLCV a
    high-volume close through the prior 5m range so absorption_breakout
    fires on the decision chart. `fire_last_1m` is a 1m-only breakout that
    would fire if the scheduler still evaluated every 1m bar."""
    candles = []
    for i in range(n_closed):
        ts = start + timedelta(minutes=i)
        in_last_5m = i >= n_closed - 5
        if fire_last_5m_bin and in_last_5m:
            o, h, l, c, v = 100.0, 110.0, 100.0, 109.0, 50_000.0
        elif fire_last_1m and i == n_closed - 1:
            o, h, l, c, v = 100.0, 110.0, 100.0, 109.0, 50_000.0
        else:
            o, h, l, c, v = 100.0, 101.0, 99.0, 100.0, 1000.0
        candles.append(_candle(symbol, ts, o, h, l, c, v))
    last_px = candles[-1].close
    candles.append(_candle(
        symbol, start + timedelta(minutes=n_closed),
        last_px, last_px, last_px, last_px, 100.0,
    ))
    return candles


def _flat_candles(symbol: str, start=SESSION_OPEN, n: int = 111, price: float = 50.0) -> list[Candle]:
    return [_candle(symbol, start + timedelta(minutes=i), price, price, price, price) for i in range(n)]


def _snapshot(symbol: str, candles: list[Candle], timestamp=None) -> MarketSnapshot:
    timestamp = timestamp if timestamp is not None else candles[-1].timestamp
    price = candles[-2].close if len(candles) >= 2 else candles[-1].close
    return MarketSnapshot(
        code=symbol, price=price, volume_today=sum(c.volume for c in candles),
        turnover_today=sum(c.turnover for c in candles), vwap=price,
        atr14=1.0, atr5=1.0, rsi14=50.0, ema8=price, ema20=price,
        bb_upper=price + 1, bb_mid=price, bb_lower=price - 1, vol_ma20=1000.0, vol_ratio=1.0,
        bid_ask_spread_pct=0.001, timestamp=timestamp, is_regular_trading_hours=True,
        candles_1m=candles,
    )


def _make_scheduler(
    bus: MessageBus,
    equity: float = 1_000_000.0,
    signals=None,
    apply_macro_gate: bool = False,
    live_universe=None,
    **kwargs,
) -> MicrostructureScheduler:
    """Pipeline-mechanics helper. Default `signals=("absorption_breakout",)`
    matches LIVE_SIGNALS. `apply_macro_gate=False` so fixture tests are
    not fail-closed by missing QQQ/SPY/XLK bars."""
    risk_engine = RiskEngine(RiskConfig(micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.5))
    return MicrostructureScheduler(
        bus, risk_engine, _STRATEGY_PARAMS, get_account_equity=lambda: equity,
        signals=signals if signals is not None else ("absorption_breakout",),
        apply_macro_gate=apply_macro_gate,
        live_universe=live_universe if live_universe is not None else frozenset({"AAPL", "BADSYM", "WEIRD"}),
        **kwargs,
    )


# ── (a) qualified order published on a real fired signal, during RTH ───────

def test_signal_during_rth_publishes_qualified_micro_order_on_5m_close():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)

    candles = _absorption_1m_candles("AAPL", fire_last_5m_bin=True)
    snap = _snapshot("AAPL", candles)

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)  # let bus.publish's scheduled subscriber tasks run

    asyncio.run(_run())

    orders = [o for o in captured if o.raw.strategy == "absorption_breakout"]
    assert len(orders) == 1
    order = orders[0]
    assert order.raw.symbol == "AAPL"
    assert order.approved is True
    assert order.qty > 0


def test_signal_does_not_fire_on_incomplete_5m_bin():
    """A 1m-only breakout on 11:18 (elapsed since 09:30 ≡ 3 mod 5) must
    not evaluate — the 11:15 5m bin is still open."""
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)
    snap = _snapshot("AAPL", _absorption_1m_candles(
        "AAPL", n_closed=_ABSORPTION_CLOSED_1M - 1, fire_last_1m=True,
    ))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert captured == []


def test_no_signal_no_publish_on_flat_bars():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)
    snap = _snapshot("AAPL", _flat_candles("AAPL"))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert captured == []


# ── (b) nothing evaluated/published outside RTH ─────────────────────────────

def test_outside_rth_nothing_evaluated_or_published():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)

    candles = _absorption_1m_candles("AAPL", fire_last_5m_bin=True)
    snap = _snapshot("AAPL", candles, timestamp=OUTSIDE_RTH)

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert captured == []
    # Confirm bars were never even ingested (evaluation genuinely skipped,
    # not just "evaluated but the result discarded").
    assert "AAPL" not in scheduler._buffers


# ── (c) the existing two-key auto_execute gate still blocks submission ─────

def test_pipeline_feeds_gate_that_still_blocks_in_observe_mode():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    broker = SimBroker()
    # Real ExecutionGateway, default mode="observe" — this test proves the
    # scheduler's published order flows all the way to the gateway and is
    # still correctly blocked, WITHOUT re-testing the gate's own logic
    # (see tests/test_never_market_orders.py / test_config_enforcement.py
    # for that).
    ExecutionGateway(bus, broker, mode="observe", auto_execute_strategies=set())

    snap = _snapshot("AAPL", _absorption_1m_candles("AAPL", fire_last_5m_bin=True))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert broker.get_positions() == {}


def test_pipeline_feeds_gate_that_submits_when_armed():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    broker = SimBroker()
    ExecutionGateway(bus, broker, mode="auto", auto_execute_strategies={"absorption_breakout"})
    reports = []

    async def _cap(report):
        reports.append(report)

    bus.subscribe("execution_report", _cap)
    snap = _snapshot("AAPL", _absorption_1m_candles("AAPL", fire_last_5m_bin=True))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    # SimBroker may fill entry AND the protective stop on the same wide
    # 5m bar (net qty can be 0). The pin is that the armed gate submitted.
    micro = [r for r in reports if r.get("type") == "micro_order"]
    assert micro and micro[0]["entry"]["accepted"] is True
    assert micro[0]["strategy"] == "absorption_breakout"


# ── (d) l2_absorption is never evaluated or published by the live path ─────

def test_l2_absorption_excluded_from_live_signals_constant():
    from python.core.paper_forward import RETIRED_MICRO_SIGNALS

    assert LIVE_SIGNALS == ("absorption_breakout",)
    assert set(LIVE_SIGNALS).isdisjoint(RETIRED_MICRO_SIGNALS)
    for retired in RETIRED_MICRO_SIGNALS:
        assert retired not in LIVE_SIGNALS


def test_l2_absorption_never_dispatched_during_evaluation(monkeypatch):
    called_signal_names: list[str] = []
    real_evaluate = scheduler_module._evaluate_signal

    def _spy(signal_name, *args, **kwargs):
        called_signal_names.append(signal_name)
        return real_evaluate(signal_name, *args, **kwargs)

    monkeypatch.setattr(scheduler_module, "_evaluate_signal", _spy)

    bus = MessageBus()
    scheduler = _make_scheduler(bus, signals=LIVE_SIGNALS, live_universe=frozenset({"AAPL"}))
    snap = _snapshot("AAPL", _absorption_1m_candles("AAPL", fire_last_5m_bin=True))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert "absorption_breakout" in called_signal_names
    assert "l2_absorption" not in called_signal_names
    assert "sweep_reclaim" not in called_signal_names
    assert "fvg_retest" not in called_signal_names
    assert "orb_vwap" not in called_signal_names
    assert set(called_signal_names) <= set(LIVE_SIGNALS)


# ── (e) one symbol's exception does not stop other symbols ─────────────────

def test_one_symbol_exception_does_not_block_other_symbols(monkeypatch):
    real_evaluate = scheduler_module._evaluate_signal

    def _boom_for_badsym(signal_name, bars_so_far, symbol, *args, **kwargs):
        if symbol == "BADSYM":
            raise RuntimeError("simulated evaluation crash for BADSYM")
        return real_evaluate(signal_name, bars_so_far, symbol, *args, **kwargs)

    monkeypatch.setattr(scheduler_module, "_evaluate_signal", _boom_for_badsym)

    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)

    bad_snap = _snapshot("BADSYM", _absorption_1m_candles("BADSYM", fire_last_5m_bin=True))
    good_snap = _snapshot("AAPL", _absorption_1m_candles("AAPL", fire_last_5m_bin=True))

    async def _run():
        # Must not raise, even though BADSYM's evaluation blows up internally.
        await scheduler.on_snapshot(bad_snap)
        await scheduler.on_snapshot(good_snap)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert captured, "AAPL's signal should still have fired despite BADSYM's crash"
    assert all(o.raw.symbol == "AAPL" for o in captured)


def test_single_bad_bar_extraction_does_not_raise(monkeypatch):
    """A malformed snapshot (e.g. candles_1m containing something
    unexpected) must be logged and skipped, never propagate out of
    on_snapshot — the scheduler is called from the live tick pipeline,
    where one bad snapshot must not kill the whole event loop."""
    bus = MessageBus()
    scheduler = _make_scheduler(bus)

    class _BrokenSnapshot:
        code = "WEIRD"
        timestamp = RTH_START

        @property
        def candles_1m(self):
            raise RuntimeError("simulated corrupt snapshot")

    async def _run():
        await scheduler.on_snapshot(_BrokenSnapshot())

    asyncio.run(_run())  # must not raise


# ── open-position bookkeeping (task requirement 3) ──────────────────────────

def test_open_position_count_increments_on_accepted_entry_and_decrements_on_flatten():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    assert scheduler.open_micro_position_count() == 0

    async def _run():
        await bus.publish("execution_report", {
            "type": "micro_order", "symbol": "AAPL", "strategy": "fvg_retest",
            "entry": {"accepted": True}, "protective_stop": {"accepted": True},
        })
        await asyncio.sleep(0)
        assert scheduler.open_micro_position_count() == 1

        await bus.publish("execution_report", {
            "type": "eod_flatten", "code": "AAPL", "strategy": "microstructure",
            "result": {"accepted": True},
        })
        await asyncio.sleep(0)
        assert scheduler.open_micro_position_count() == 0

    asyncio.run(_run())


def test_open_position_count_not_incremented_on_rejected_entry():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)

    async def _run():
        await bus.publish("execution_report", {
            "type": "micro_order", "symbol": "AAPL", "strategy": "fvg_retest",
            "entry": {"accepted": False, "reason": "no_valid_limit_price"},
        })
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert scheduler.open_micro_position_count() == 0


# ── signal journal wiring (task requirement: python/microstructure/signal_journal.py) ──

def test_fired_signal_is_journaled_when_a_journal_is_wired_in(tmp_path):
    """No journal is passed by `_make_scheduler` (see its own docstring) —
    this test wires one in explicitly to prove `_qualify_and_publish`
    actually calls it, without making every other test in this file touch
    disk."""
    from python.microstructure.signal_journal import SignalJournal

    bus = MessageBus()
    risk_engine = RiskEngine(RiskConfig(micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.5))
    journal = SignalJournal(output_dir=tmp_path)
    scheduler = MicrostructureScheduler(
        bus, risk_engine, _STRATEGY_PARAMS, get_account_equity=lambda: 1_000_000.0,
        signal_journal=journal, signals=("absorption_breakout",), apply_macro_gate=False,
        live_universe=frozenset({"AAPL"}),
    )
    snap = _snapshot("AAPL", _absorption_1m_candles("AAPL", fire_last_5m_bin=True))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())

    rows = journal.read_day(snap.timestamp)
    abs_rows = [r for r in rows if r["strategy"] == "absorption_breakout"]
    assert len(abs_rows) == 1
    assert abs_rows[0]["symbol"] == "AAPL"
    assert abs_rows[0]["risk_passed"] is True


def test_journal_failure_does_not_block_bus_publish(monkeypatch):
    """A broken journal (e.g. disk full/permission error) must never stop
    the qualified order from still reaching the bus — journaling is a
    best-effort side channel, not a gate."""
    class _BoomJournal:
        def record(self, *_args, **_kwargs):
            raise OSError("simulated disk failure")

    bus = MessageBus()
    risk_engine = RiskEngine(RiskConfig(micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.5))
    scheduler = MicrostructureScheduler(
        bus, risk_engine, _STRATEGY_PARAMS, get_account_equity=lambda: 1_000_000.0,
        signal_journal=_BoomJournal(), signals=("absorption_breakout",), apply_macro_gate=False,
        live_universe=frozenset({"AAPL"}),
    )
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)
    snap = _snapshot("AAPL", _absorption_1m_candles("AAPL", fire_last_5m_bin=True))

    async def _run():
        await scheduler.on_snapshot(snap)  # must not raise despite the journal blowing up
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert any(o.raw.strategy == "absorption_breakout" for o in captured)
