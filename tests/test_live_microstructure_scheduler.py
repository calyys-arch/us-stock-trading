"""
dashboard/live_microstructure_scheduler.py — the live glue between
DataEngine's bus("snapshot") and sweep_reclaim/fvg_retest/orb_vwap +
RiskEngine.qualify_microstructure_order. These tests deliberately do NOT
re-test ExecutionGateway's own two-key auto_execute gate (already covered
by tests/test_never_market_orders.py and tests/test_config_enforcement.py)
— they test that THIS module correctly feeds that existing, unchanged
gate: publishes a qualified order when a signal fires during RTH, stays
silent outside RTH, never touches l2_absorption, and survives one
symbol's evaluation blowing up.
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

# 2026-03-02 is a Monday, comfortably before the March DST switch (2026-03-08),
# so 14:35 UTC == 09:35 ET — inside RTH but past the open, matching the rest
# of this repo's tests (e.g. tests/test_never_market_orders.py's RTH constant).
RTH_START = datetime(2026, 3, 2, 14, 35, tzinfo=timezone.utc)
OUTSIDE_RTH = datetime(2026, 3, 2, 22, 0, tzinfo=timezone.utc)  # 17:00 ET — after the 16:00 close

_STRATEGY_PARAMS = {
    "sweep_reclaim": {"sweep_min_atr": 0.15, "reclaim_bars": 3, "stop_atr_mult": 0.25},
    "fvg_retest": {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10},
    "orb_vwap": {"or_minutes": 15, "vwap_side_filter": True},
}


def _candle(symbol: str, ts, o: float, h: float, l: float, c: float, v: float = 1000.0) -> Candle:
    return Candle(code=symbol, timeframe="1m", open=o, high=h, low=l, close=c, volume=v, turnover=o * v, timestamp=ts)


def _fvg_candles(symbol: str, start=RTH_START, n: int = 25) -> list[Candle]:
    """Reproduces tests/test_intraday_signals.py's
    test_fvg_retest_fires_on_bullish_gap bar sequence exactly (bullish FVG
    at indices 21/22/23), as a list of live Candle objects: indices 0..23
    are CLOSED bars, index 24 is the current in-progress bar (matching
    DataEngine.CandleBuilder.last()'s own [...closed..., open] shape) — so
    the FVG only "completes" once bar 23 has actually closed."""
    candles = []
    for i in range(n):
        ts = start + timedelta(minutes=i)
        o = h = l = c = 100.0
        v = 1000.0
        if i == 21:
            o, h, l, c = 100.0, 100.5, 99.5, 100.2
        elif i == 22:
            o, h, l, c, v = 100.2, 106.0, 100.1, 105.5, 50_000.0
        elif i == 23:
            o, h, l, c = 105.5, 107.0, 105.2, 106.5
        candles.append(_candle(symbol, ts, o, h, l, c, v))
    return candles


def _flat_candles(symbol: str, start=RTH_START, n: int = 25, price: float = 50.0) -> list[Candle]:
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


def _make_scheduler(bus: MessageBus, equity: float = 1_000_000.0) -> MicrostructureScheduler:
    risk_engine = RiskEngine(RiskConfig(micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.5))
    return MicrostructureScheduler(bus, risk_engine, _STRATEGY_PARAMS, get_account_equity=lambda: equity)


# ── (a) qualified order published on a real fired signal, during RTH ───────

def test_fvg_signal_during_rth_publishes_qualified_micro_order():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    captured = []

    async def _capture(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _capture)

    candles = _fvg_candles("AAPL")
    snap = _snapshot("AAPL", candles)

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)  # let bus.publish's scheduled subscriber tasks run

    asyncio.run(_run())

    # sweep_reclaim and orb_vwap may ALSO legitimately fire on this same
    # bar sequence (a large enough move to form a fresh FVG routinely also
    # crosses a nearby round-number level and/or the opening range) — this
    # test only asserts the fvg_retest signal specifically fired and was
    # correctly qualified/published, not that it was the ONLY signal.
    fvg_orders = [o for o in captured if o.raw.strategy == "fvg_retest"]
    assert len(fvg_orders) == 1
    order = fvg_orders[0]
    assert order.raw.symbol == "AAPL"
    assert order.approved is True
    assert order.qty > 0


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

    candles = _fvg_candles("AAPL")  # would fire fvg_retest if evaluated
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

    snap = _snapshot("AAPL", _fvg_candles("AAPL"))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert broker.get_positions() == {}


def test_pipeline_feeds_gate_that_submits_when_armed():
    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    broker = SimBroker()
    ExecutionGateway(bus, broker, mode="auto", auto_execute_strategies={"fvg_retest"})

    snap = _snapshot("AAPL", _fvg_candles("AAPL"))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    positions = broker.get_positions()
    assert positions.get("AAPL", 0) != 0


# ── (d) l2_absorption is never evaluated or published by the live path ─────

def test_l2_absorption_excluded_from_live_signals_constant():
    assert "l2_absorption" not in LIVE_SIGNALS
    assert set(LIVE_SIGNALS) == {"sweep_reclaim", "fvg_retest", "orb_vwap"}


def test_l2_absorption_never_dispatched_during_evaluation(monkeypatch):
    called_signal_names: list[str] = []
    real_evaluate = scheduler_module._evaluate_signal

    def _spy(signal_name, *args, **kwargs):
        called_signal_names.append(signal_name)
        return real_evaluate(signal_name, *args, **kwargs)

    monkeypatch.setattr(scheduler_module, "_evaluate_signal", _spy)

    bus = MessageBus()
    scheduler = _make_scheduler(bus)
    snap = _snapshot("AAPL", _fvg_candles("AAPL"))

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert "l2_absorption" not in called_signal_names
    assert "fvg_retest" in called_signal_names
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

    bad_snap = _snapshot("BADSYM", _fvg_candles("BADSYM"))
    good_snap = _snapshot("AAPL", _fvg_candles("AAPL"))

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
