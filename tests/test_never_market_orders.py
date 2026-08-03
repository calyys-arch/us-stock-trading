"""
"Never a market order" invariant tests (user decision, 2026-07-29: always
Limit/Stop-Limit, control exact execution price, avoid PFOF-routed market-
order fills). Confirms ExecutionGateway._submit_order is the one chokepoint
that reaches SimBroker/IbkrBroker, and that every code path in the class
that reaches it either has a valid bounded price or is hard-rejected —
never a silent market-order fallback.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from python.core.bus import MessageBus
from python.core.execution_gateway import ExecutionGateway
from python.core.risk_engine import RiskConfig, RiskEngine
from python.core.sim_broker import SimBroker
from python.core.types import (
    CointegrationResult,
    PortfolioTarget,
    QualifiedMicroOrder,
    QualifiedPortfolioOrder,
    QualifiedSpreadOrder,
    SpreadSide,
    SpreadSignal,
)

RTH = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)  # 10:00 ET


class _RecordingBroker(SimBroker):
    """SimBroker that also remembers every place_order call's kwargs, so
    tests can assert on order_type/limit_price directly instead of only
    the resulting position."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def place_order(self, code, side, qty, limit_price=0.0, order_type="market", stop_price=0.0, tif="DAY"):
        self.calls.append({"code": code, "side": side, "qty": qty, "limit_price": limit_price,
                            "order_type": order_type, "stop_price": stop_price, "tif": tif})
        return super().place_order(code, side, qty, limit_price=limit_price, order_type=order_type,
                                    stop_price=stop_price, tif=tif)


def _make_spread_signal() -> SpreadSignal:
    return SpreadSignal(
        id="s1", strategy="pairs_trading", code_a="AAA", code_b="BBB", hedge_ratio=1.0,
        side=SpreadSide.LONG_SPREAD, z_score=2.5, entry_z_threshold=2.0, exit_z_threshold=0.5,
        spread_mean=0.0, half_life_days=10.0, confidence=0.8, timestamp=RTH,
    )


@dataclass
class _FakeMicroSignal:
    symbol: str = "AAPL"
    strategy: str = "fvg_retest"
    direction: str = "long"
    signal_time: datetime = RTH
    entry_price: float = 100.0
    stop_price: float = 99.0
    target_price: float | None = 102.0
    context: dict = field(default_factory=dict)


# ── _submit_order chokepoint ─────────────────────────────────────────────────

def test_submit_order_refuses_market_order_outright():
    gw = ExecutionGateway(MessageBus(), _RecordingBroker())
    result = gw._submit_order("AAPL", "buy", 100, limit_price=150.0, order_type="market")
    assert result["accepted"] is False
    assert result["reason"] == "market_orders_disabled"
    assert gw._broker.calls == []  # never reached the broker


def test_submit_order_rejects_limit_with_no_price():
    gw = ExecutionGateway(MessageBus(), _RecordingBroker())
    result = gw._submit_order("AAPL", "buy", 100, limit_price=0.0, order_type="limit")
    assert result["accepted"] is False
    assert result["reason"] == "no_valid_limit_price"


def test_submit_order_rejects_stop_limit_with_missing_stop_price():
    gw = ExecutionGateway(MessageBus(), _RecordingBroker())
    result = gw._submit_order("AAPL", "sell", 100, limit_price=99.0, order_type="stop_limit", stop_price=0.0)
    assert result["accepted"] is False
    assert result["reason"] == "no_valid_stop_limit_price"


def test_submit_order_accepts_valid_limit_order():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker)
    result = gw._submit_order("AAPL", "buy", 100, limit_price=150.0, order_type="limit")
    assert result["accepted"] is True
    assert broker.calls[0]["order_type"] == "limit"


# ── spread orders always go out as limit orders ─────────────────────────────

def test_spread_order_never_submits_market_and_uses_computed_limit_prices():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker, mode="auto", auto_execute_strategies={"pairs_trading"})
    order = QualifiedSpreadOrder(
        raw=_make_spread_signal(), qty_a=100, qty_b=100, gross_notional=20000.0,
        estimated_cost=0.0, kelly_fraction_used=0.05, approved=True,
        limit_price_a=101.0, limit_price_b=49.0,
    )
    asyncio.run(gw._on_spread_order(order))
    assert len(broker.calls) == 2
    for call in broker.calls:
        assert call["order_type"] != "market"
        assert call["limit_price"] > 0


def test_spread_order_with_zero_limit_price_is_rejected_not_market():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker, mode="auto", auto_execute_strategies={"pairs_trading"})
    order = QualifiedSpreadOrder(
        raw=_make_spread_signal(), qty_a=100, qty_b=100, gross_notional=20000.0,
        estimated_cost=0.0, kelly_fraction_used=0.05, approved=True,
        limit_price_a=0.0, limit_price_b=49.0,  # leg A missing a valid price
    )
    asyncio.run(gw._on_spread_order(order))
    # Leg A must have been rejected at the chokepoint, never silently market.
    assert broker.get_positions().get("AAA", 0) == 0
    assert broker.get_positions().get("BBB", 0) != 0


# ── portfolio orders always go out as limit orders ──────────────────────────

def test_portfolio_order_uses_per_code_limit_prices():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker, mode="auto", auto_execute_strategies={"xsection_mean_reversion"})
    target = PortfolioTarget(strategy="xsection_mean_reversion", as_of=RTH, weights={"AAA": 0.02})
    order = QualifiedPortfolioOrder(
        raw=target, target_shares={"AAA": 100}, gross_notional=10000.0, approved=True,
        limit_prices={"AAA": 101.5},
    )
    asyncio.run(gw._on_portfolio_order(order))
    assert broker.calls[0]["order_type"] == "limit"
    assert broker.calls[0]["limit_price"] == 101.5


# ── flatten paths skip rather than submit a market order ───────────────────

def test_flatten_intraday_positions_skips_without_price_lookup():
    broker = _RecordingBroker()
    broker.place_order("AAA", "buy", 100, limit_price=100.0, order_type="limit")
    gw = ExecutionGateway(MessageBus(), broker, flatten_buffer_check=False)
    asyncio.run(gw.flatten_intraday_positions("xsection_mean_reversion"))
    assert broker.get_positions()["AAA"] == 100  # still open — flatten skipped, no market order sent
    assert not any(c["order_type"] == "market" for c in broker.calls)


def test_flatten_intraday_positions_uses_bounded_limit_when_price_lookup_wired():
    broker = _RecordingBroker()
    broker.place_order("AAA", "buy", 100, limit_price=100.0, order_type="limit")
    gw = ExecutionGateway(MessageBus(), broker, flatten_buffer_check=False, price_lookup=lambda code: 100.0)
    asyncio.run(gw.flatten_intraday_positions("xsection_mean_reversion"))
    assert broker.get_positions()["AAA"] == 0  # closed
    close_call = [c for c in broker.calls if c["side"] == "sell"][0]
    assert close_call["order_type"] == "limit"
    assert close_call["limit_price"] < 100.0  # sell -> below reference price


def test_flatten_position_skips_without_price_lookup():
    broker = _RecordingBroker()
    broker.place_order("AAA", "buy", 50, limit_price=10.0, order_type="limit")
    gw = ExecutionGateway(MessageBus(), broker)
    result = asyncio.run(gw.flatten_position("AAA"))
    assert result["result"]["accepted"] is False
    assert result["result"]["reason"] == "no_price_lookup_configured"
    assert broker.get_positions()["AAA"] == 50


def test_emergency_flatten_all_never_submits_market_orders():
    broker = _RecordingBroker()
    broker.place_order("AAA", "buy", 50, limit_price=10.0, order_type="limit")
    broker.place_order("BBB", "sell", 30, limit_price=20.0, order_type="limit")
    gw = ExecutionGateway(MessageBus(), broker, price_lookup=lambda code: {"AAA": 10.0, "BBB": 20.0}[code])
    asyncio.run(gw.emergency_flatten_all())
    assert broker.get_positions().get("AAA", 0) == 0
    assert broker.get_positions().get("BBB", 0) == 0
    assert not any(c["order_type"] == "market" for c in broker.calls)


# ── microstructure orders: limit entry + stop-limit protective exit ────────

def test_microstructure_order_submits_limit_entry_and_stop_limit_exit():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker, mode="auto", auto_execute_strategies={"fvg_retest"})
    engine = RiskEngine(RiskConfig(micro_cancel_after_seconds=0))
    signal = _FakeMicroSignal()
    from python.core.types import MarketSnapshot

    snap = MarketSnapshot(
        code="AAPL", price=100.0, volume_today=1_000_000, turnover_today=1e8,
        vwap=100.0, atr14=1.0, atr5=1.0, rsi14=50.0, ema8=100.0, ema20=100.0,
        bb_upper=101.0, bb_mid=100.0, bb_lower=99.0, vol_ma20=1_000_000, vol_ratio=1.0,
        bid_ask_spread_pct=0.001, timestamp=RTH, is_regular_trading_hours=True,
    )
    order = engine.qualify_microstructure_order(signal, snap, account_equity=1_000_000, open_micro_positions=0)
    assert order.approved

    asyncio.run(gw._on_microstructure_order(order))

    order_types = [c["order_type"] for c in broker.calls]
    assert "market" not in order_types
    assert "limit" in order_types      # entry (+ optional target)
    assert "stop_limit" in order_types  # protective exit


def test_microstructure_order_rejected_upstream_never_reaches_broker():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker, mode="auto", auto_execute_strategies={"fvg_retest"})
    order = QualifiedMicroOrder(
        raw=_FakeMicroSignal(), qty=100, entry_limit_price=101.0, stop_price=99.0,
        stop_limit_price=98.5, approved=False, rejection_reason="daily_loss_kill_switch",
    )
    asyncio.run(gw._on_microstructure_order(order))
    assert broker.calls == []


def test_microstructure_order_blocked_in_observe_mode():
    broker = _RecordingBroker()
    gw = ExecutionGateway(MessageBus(), broker, mode="observe", auto_execute_strategies={"fvg_retest"})
    order = QualifiedMicroOrder(
        raw=_FakeMicroSignal(), qty=100, entry_limit_price=101.0, stop_price=99.0,
        stop_limit_price=98.5, approved=True,
    )
    asyncio.run(gw._on_microstructure_order(order))
    assert broker.calls == []
