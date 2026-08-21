"""Paper-forward experiment wiring (2026-08-15).

NOT a WFO GO promotion. Pins: LIVE_SIGNALS is only absorption_breakout;
retired names cannot auto-execute; macro beta gate fail-closes on missing
index bars; pairs regime gate blocks a synthetic trend and allows a
synthetic mean-reverting series; ExecutionGateway only auto-executes the
allowlisted names.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from python.analytics.macro_beta_gate import LiveMacroGate, compute_macro_momentum, macro_gate_ok
from python.analytics.trend_efficiency_gate import live_entry_allowed, shifted_entry_gate
from python.core.bus import MessageBus
from python.core.execution_gateway import ExecutionGateway
from python.core.paper_forward import (
    ABSORPTION_BREAKOUT_UNIVERSE,
    PAPER_AUTO_ALLOWLIST,
    RETIRED_MICRO_SIGNALS,
)
from python.core.risk_engine import RiskConfig, RiskEngine
from python.core.sim_broker import SimBroker
from python.core.types import (
    CointegrationResult,
    QualifiedMicroOrder,
    QualifiedSpreadOrder,
    SpreadSide,
    SpreadSignal,
)
from python.microstructure.signals import MicroSignal

from dashboard.engine_bridge import _load_paper_auto_strategies
from dashboard.live_microstructure_scheduler import LIVE_SIGNALS, MicrostructureScheduler
from dashboard.live_pairs_scheduler import LivePairsScheduler


RTH = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


def _spread(strategy: str = "pairs_trading") -> SpreadSignal:
    return SpreadSignal(
        id="sig1", strategy=strategy, code_a="AAA", code_b="BBB",
        hedge_ratio=1.0, side=SpreadSide.LONG_SPREAD, z_score=4.5,
        entry_z_threshold=4.0, exit_z_threshold=0.5, spread_mean=0.0,
        half_life_days=10.0, confidence=0.8, timestamp=RTH,
    )


def _micro(strategy: str = "absorption_breakout") -> MicroSignal:
    return MicroSignal(
        symbol="AAPL", strategy=strategy, direction="long",
        signal_time=pd.Timestamp("2026-03-02 10:00:00"),
        entry_price=100.0, stop_price=99.0, target_price=None,
        order_type="next_open",
    )


def test_live_signals_is_only_absorption_breakout():
    assert LIVE_SIGNALS == ("absorption_breakout",)
    assert set(LIVE_SIGNALS).isdisjoint(RETIRED_MICRO_SIGNALS)
    assert ABSORPTION_BREAKOUT_UNIVERSE == ("AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC")


def test_paper_auto_allowlist_excludes_retired():
    assert PAPER_AUTO_ALLOWLIST == frozenset({"absorption_breakout", "pairs_trading"})
    assert PAPER_AUTO_ALLOWLIST.isdisjoint(RETIRED_MICRO_SIGNALS)
    armed = _load_paper_auto_strategies()
    assert armed == {"absorption_breakout", "pairs_trading"}
    assert armed.isdisjoint(RETIRED_MICRO_SIGNALS)


def test_retired_signals_cannot_auto_execute_even_if_gateway_armed():
    """Even a mistaken allowlist grant for a retired name is not how
    enable_auto_trading works — and a retired name must not be in the
    yaml-derived set. This test also pins the gateway: a retired name
    that is NOT in auto_execute_strategies cannot submit."""
    bus = MessageBus()
    broker = SimBroker()
    gw = ExecutionGateway(
        bus, broker, mode="auto",
        auto_execute_strategies=_load_paper_auto_strategies(),
    )
    assert gw._auto_execute_strategies == {"absorption_breakout", "pairs_trading"}
    for retired in RETIRED_MICRO_SIGNALS:
        assert retired not in gw._auto_execute_strategies

    order = QualifiedMicroOrder(
        raw=_micro("sweep_reclaim"), qty=10, entry_limit_price=100.1,
        stop_price=99.0, stop_limit_price=98.9, approved=True,
    )

    async def _run():
        await gw._on_microstructure_order(order)

    asyncio.run(_run())
    assert broker.get_positions() == {}


def test_gateway_auto_executes_only_allowlisted_absorption_breakout():
    bus = MessageBus()
    broker = SimBroker()
    gw = ExecutionGateway(
        bus, broker, mode="auto",
        auto_execute_strategies=_load_paper_auto_strategies(),
    )
    reports = []

    async def _cap(report):
        reports.append(report)

    bus.subscribe("execution_report", _cap)
    order = QualifiedMicroOrder(
        raw=_micro("absorption_breakout"), qty=10, entry_limit_price=100.1,
        stop_price=99.0, stop_limit_price=98.9, approved=True,
    )

    async def _run():
        await gw._on_microstructure_order(order)
        await asyncio.sleep(0)

    asyncio.run(_run())
    # Entry + protective stop both fill on SimBroker (net qty can be 0);
    # the pin is that the allowlisted name was actually submitted.
    assert reports and reports[0].get("type") == "micro_order"
    assert reports[0]["entry"]["accepted"] is True
    assert reports[0]["strategy"] == "absorption_breakout"


def test_gateway_auto_executes_only_allowlisted_pairs():
    bus = MessageBus()
    broker = SimBroker()
    gw = ExecutionGateway(
        bus, broker, mode="auto",
        auto_execute_strategies=_load_paper_auto_strategies(),
    )
    order = QualifiedSpreadOrder(
        raw=_spread("pairs_trading"), qty_a=10, qty_b=10, gross_notional=2000.0,
        estimated_cost=0.0, kelly_fraction_used=0.003, approved=True,
        limit_price_a=100.0, limit_price_b=50.0,
    )

    async def _run():
        await gw._on_spread_order(order)

    asyncio.run(_run())
    pos = broker.get_positions()
    assert pos.get("AAA", 0) != 0 or pos.get("BBB", 0) != 0


def test_gateway_rejects_xsection_even_in_auto_mode():
    bus = MessageBus()
    broker = SimBroker()
    gw = ExecutionGateway(
        bus, broker, mode="auto",
        auto_execute_strategies=_load_paper_auto_strategies(),
    )
    from python.core.types import PortfolioTarget, QualifiedPortfolioOrder

    target = PortfolioTarget(
        strategy="xsection_mean_reversion", as_of=RTH, weights={"AAA": 0.1},
    )
    order = QualifiedPortfolioOrder(
        raw=target, target_shares={"AAA": 10}, approved=True,
        limit_prices={"AAA": 100.0},
    )

    async def _run():
        await gw._on_portfolio_order(order)

    asyncio.run(_run())
    assert broker.get_positions() == {}


def _index_bars(closes, start="2026-03-02 09:30:00"):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1000.0] * len(closes)},
        index=idx,
    )


def test_macro_gate_closed_on_missing_index():
    gate = LiveMacroGate()  # no bars loaded
    assert gate.ok("long", pd.Timestamp("2026-03-02 10:00:00")) is False
    gate.set_index_bars({})
    assert gate.ok("short", pd.Timestamp("2026-03-02 10:00:00")) is False


def test_macro_gate_closed_on_missing_timestamp():
    bars = {"QQQ": _index_bars([100, 101, 102, 103, 104, 105, 106])}
    gate = LiveMacroGate(bars)
    # timestamp not in the frame
    assert gate.ok("long", pd.Timestamp("2026-03-02 15:00:00")) is False


def test_macro_gate_aligns_long_and_blocks_countertrend():
    # steadily rising tape — long should pass, short should fail
    closes = [100, 101, 102, 103, 104, 105, 106]
    bars = {"QQQ": _index_bars(closes), "SPY": _index_bars(closes), "XLK": _index_bars(closes)}
    mom = compute_macro_momentum(bars)
    t = mom.index[-1]
    assert macro_gate_ok(mom, "long", t) is True
    assert macro_gate_ok(mom, "short", t) is False
    gate = LiveMacroGate(bars)
    assert gate.ok("long", t) is True
    assert gate.ok("short", t) is False


def test_scheduler_macro_gate_fail_closed_publishes_rejection():
    from python.core.types import Candle, MarketSnapshot

    bus = MessageBus()
    captured = []

    async def _cap(order):
        captured.append(order)

    bus.subscribe("qualified_micro_order", _cap)
    risk = RiskEngine(RiskConfig(micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.5))
    scheduler = MicrostructureScheduler(
        bus, risk,
        {"absorption_breakout": {"volume_mult": 3.0, "breakout_atr_mult": 0.5, "stop_atr_mult": 0.5}},
        get_account_equity=lambda: 1_000_000.0,
        signals=("absorption_breakout",),
        apply_macro_gate=True,
        macro_gate=LiveMacroGate(),  # empty → fail closed
        live_universe=frozenset({"AAPL"}),
    )

    # 110 closed 1m bars from 09:30 ET = 22 closed 5m decision bars, last
    # 5m bin is a high-volume close through the prior 5m range.
    start = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
    candles = []
    n_closed = 110
    for i in range(n_closed):
        ts = start + pd.Timedelta(minutes=i)
        if i >= n_closed - 5:
            o, h, l, c, v = 100.0, 110.0, 100.0, 109.0, 50_000.0
        else:
            o, h, l, c, v = 100.0, 101.0, 99.0, 100.0, 1000.0
        candles.append(Candle(
            code="AAPL", timeframe="1m", open=o, high=h, low=l, close=c,
            volume=v, turnover=o * v, timestamp=ts,
        ))
    candles.append(Candle(
        code="AAPL", timeframe="1m", open=109.0, high=109.0, low=109.0, close=109.0,
        volume=100, turnover=10900, timestamp=start + pd.Timedelta(minutes=n_closed),
    ))
    snap = MarketSnapshot(
        code="AAPL", price=109.0, volume_today=1, turnover_today=1, vwap=100.0,
        atr14=1.0, atr5=1.0, rsi14=50.0, ema8=100.0, ema20=100.0,
        bb_upper=101, bb_mid=100, bb_lower=99, vol_ma20=1000, vol_ratio=1.0,
        bid_ask_spread_pct=0.001, timestamp=candles[-1].timestamp,
        is_regular_trading_hours=True, candles_1m=candles,
    )

    async def _run():
        await scheduler.on_snapshot(snap)
        await asyncio.sleep(0)

    asyncio.run(_run())
    blocked = [o for o in captured if o.raw.strategy == "absorption_breakout"]
    assert blocked, "5m absorption_breakout must fire so the empty macro gate is exercised"
    for o in blocked:
        assert o.approved is False
        assert o.rejection_reason == "macro_beta_gate_closed_or_missing_index"


def _trend_close(n: int = 400) -> pd.Series:
    """Chop first (pulls the 252-day ER median down), then a persistent
    one-way trend — current ER sits ABOVE its own trailing median → gate
    CLOSED. A pure always-trending series would have ER ≈ its own median
    and would not trip the 'unusually persistent' rule."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    vals = []
    px = 100.0
    split = n - 80
    for i in range(n):
        if i < split:
            px += 0.6 if (i % 2 == 0) else -0.55
        else:
            px *= 1.012
        vals.append(px)
    return pd.Series(vals, index=idx)


def _choppy_close(n: int = 400) -> pd.Series:
    """Trend first (pulls the 252-day ER median up), then two-way chop —
    current ER sits at or below that median → gate OPEN."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    vals = []
    px = 100.0
    split = n - 80
    for i in range(n):
        if i < split:
            px *= 1.008
        else:
            px += 0.8 if (i % 2 == 0) else -0.75
        vals.append(px)
    return pd.Series(vals, index=idx)


def test_pairs_regime_gate_blocks_synthetic_strong_trend():
    close = _trend_close()
    assert live_entry_allowed(close) is False
    assert bool(shifted_entry_gate(close).iloc[-1]) is False


def test_pairs_regime_gate_allows_synthetic_mean_reversion():
    close = _choppy_close()
    assert live_entry_allowed(close) is True


def test_live_pairs_scheduler_blocks_entries_when_gate_closed():
    """Strong-trend SPY proxy → no new entries; the evaluate_once result
    records the closed gate. (Exits would still run if a position were open.)"""
    bus = MessageBus()
    captured = []

    async def _cap(order):
        captured.append(order)

    bus.subscribe("qualified_spread_order", _cap)

    n = 400
    idx = pd.bdate_range("2024-01-02", periods=n)
    # Two legs that would otherwise look cointegrated + a trending SPY.
    spy = _trend_close(n)
    a = spy * 1.0
    b = spy * 0.5 + 10.0
    panel = pd.DataFrame({"SPY": spy, "AAA": a, "BBB": b}, index=idx)

    coint = CointegrationResult(
        code_a="AAA", code_b="BBB", hedge_ratio=2.0, cadf_tstat=-4.5,
        cadf_crit_1pct=-3.9, cadf_crit_5pct=-3.3, cadf_crit_10pct=-3.0,
        is_cointegrated_5pct=True, half_life_days=10.0, spread_mean=0.0,
        spread_std=0.01, computed_at=datetime(2026, 1, 1), lookback_days=252,
    )
    sched = LivePairsScheduler(
        bus, RiskEngine(RiskConfig(paper_max_notional_usd=3000.0)),
        {"entry_z": 4.0, "exit_z": 0.5, "paper_notional_per_leg": 3000.0,
         "half_life_multiplier_max_hold": 3.0},
        get_account_equity=lambda: 1_000_000.0,
        close_panel=panel, regime_close=spy,
        now_fn=lambda: RTH,
    )
    sched._active = [coint]
    sched._latest_by_pair[("AAA", "BBB")] = coint
    sched._last_scan_as_of = pd.Timestamp(idx[-1])

    async def _run():
        return await sched.evaluate_once(RTH)

    result = asyncio.run(_run())
    assert result["gate_open"] is False
    assert result["entries"] == 0
    assert sched.regime_gate_reason == "trend_gate_closed"
    assert captured == []


def test_live_pairs_scheduler_allows_entries_when_gate_open():
    bus = MessageBus()
    captured = []

    async def _cap(order):
        captured.append(order)

    bus.subscribe("qualified_spread_order", _cap)

    n = 400
    idx = pd.bdate_range("2024-01-02", periods=n)
    spy = _choppy_close(n)
    # Spread far below mean so entry_z=4.0 fires a LONG_SPREAD.
    a = pd.Series([100.0] * n, index=idx)
    b = pd.Series([100.0] * (n - 1) + [80.0], index=idx)
    panel = pd.DataFrame({"SPY": spy, "AAA": a, "BBB": b}, index=idx)

    coint = CointegrationResult(
        code_a="AAA", code_b="BBB", hedge_ratio=1.0, cadf_tstat=-4.5,
        cadf_crit_1pct=-3.9, cadf_crit_5pct=-3.3, cadf_crit_10pct=-3.0,
        is_cointegrated_5pct=True, half_life_days=10.0, spread_mean=0.0,
        spread_std=0.01, computed_at=datetime(2026, 1, 1), lookback_days=252,
    )
    sched = LivePairsScheduler(
        bus, RiskEngine(RiskConfig(paper_max_notional_usd=3000.0, require_short_locate=False)),
        {"entry_z": 4.0, "exit_z": 0.5, "paper_notional_per_leg": 3000.0,
         "half_life_multiplier_max_hold": 3.0},
        get_account_equity=lambda: 1_000_000.0,
        close_panel=panel, regime_close=spy,
        now_fn=lambda: RTH,
    )
    sched._active = [coint]
    sched._latest_by_pair[("AAA", "BBB")] = coint
    # Pin scan as-of to "today" so evaluate_once does not rebuild the
    # candidate list from configs/pairs_universe.yaml (AAA/BBB are fixtures).
    sched._last_scan_as_of = pd.Timestamp("2026-03-02")

    async def _run():
        return await sched.evaluate_once(RTH)

    result = asyncio.run(_run())
    assert result["gate_open"] is True
    assert result["entries"] >= 1
    assert any(getattr(o.raw, "strategy", None) == "pairs_trading" for o in captured)


def test_enable_auto_trading_arms_only_allowlisted_names():
    from dashboard.engine_bridge import EngineRuntime
    from dashboard.state import DashboardState

    runtime = EngineRuntime(DashboardState())
    runtime.state.running = True

    async def _run():
        return await runtime.enable_auto_trading()

    armed = asyncio.run(_run())
    assert armed == {"absorption_breakout", "pairs_trading"}
    assert runtime.gateway.mode == "auto"
    assert runtime.gateway._auto_execute_strategies == armed
    assert runtime.state.armed_strategies == sorted(armed)
    for retired in RETIRED_MICRO_SIGNALS:
        assert retired not in armed


def test_paper_max_notional_caps_micro_qty():
    from python.core.types import MarketSnapshot

    engine = RiskEngine(RiskConfig(
        micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.05,
        paper_max_notional_usd=3000.0,
    ))
    snap = MarketSnapshot(
        code="AAPL", price=100.0, volume_today=1, turnover_today=1, vwap=100.0,
        atr14=1.0, atr5=1.0, rsi14=50.0, ema8=100.0, ema20=100.0,
        bb_upper=101, bb_mid=100, bb_lower=99, vol_ma20=1000, vol_ratio=1.0,
        bid_ask_spread_pct=0.001, timestamp=RTH, is_regular_trading_hours=True,
    )
    order = engine.qualify_microstructure_order(
        _micro(), snap, account_equity=1_000_000.0, open_micro_positions=0,
        event_blackout=False, now=RTH,
    )
    assert order.approved is True
    assert order.qty <= 30  # 3000 / 100
    assert order.gross_notional <= 3000.0 + 1e-6
