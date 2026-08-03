"""
RiskEngine microstructure-signal extensions (qualify_microstructure_order,
DailyLossTracker, marketable_limit_price, load_risk_config) — the Q2
extension work (user decision, 2026-07-29): every qualify_* method computes
a bounded limit price rather than leaving execution price to a market
order, and the daily-loss kill-switch / event-blackout / max-open-positions
gates are specific to intraday microstructure signals (Chan's daily
strategies have no equivalent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from python.core.risk_engine import DailyLossTracker, RiskConfig, RiskEngine, load_risk_config, marketable_limit_price
from python.core.types import MarketSnapshot

RTH = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)  # 10:00 ET, RTH


@dataclass
class _FakeMicroSignal:
    """Stand-in for python.microstructure.signals.MicroSignal — RiskEngine
    only reads a few plain attributes, so a lightweight fake avoids pulling
    in the microstructure package (keeps this test import-light, matching
    QualifiedMicroOrder.raw's deliberately loose typing)."""
    symbol: str = "AAPL"
    strategy: str = "fvg_retest"
    direction: str = "long"
    signal_time: datetime = RTH
    entry_price: float = 100.0
    stop_price: float = 99.0
    target_price: float | None = 102.0
    context: dict = field(default_factory=dict)


def _snap(price: float = 100.0, tradeable: bool = True, rth: bool = True) -> MarketSnapshot:
    return MarketSnapshot(
        code="AAPL", price=price, volume_today=1_000_000, turnover_today=price * 1_000_000,
        vwap=price, atr14=1.0, atr5=1.0, rsi14=50.0, ema8=price, ema20=price,
        bb_upper=price + 1, bb_mid=price, bb_lower=price - 1, vol_ma20=1_000_000, vol_ratio=1.0,
        bid_ask_spread_pct=0.001, timestamp=RTH,
        is_regular_trading_hours=rth, is_halted=not tradeable,
    )


# ── marketable_limit_price ──────────────────────────────────────────────────

def test_marketable_limit_price_buy_is_above_sell_is_below():
    buy = marketable_limit_price(100.0, "buy", buffer_bps=10.0)
    sell = marketable_limit_price(100.0, "sell", buffer_bps=10.0)
    assert buy == pytest.approx(100.1)
    assert sell == pytest.approx(99.9)


def test_marketable_limit_price_nonpositive_price_returns_zero():
    assert marketable_limit_price(0.0, "buy", 10.0) == 0.0
    assert marketable_limit_price(-5.0, "sell", 10.0) == 0.0


# ── qualify_spread_order / qualify_portfolio_order carry limit prices ──────

def test_qualify_spread_order_computes_limit_prices_from_buffer():
    from python.core.types import SpreadSide, SpreadSignal

    signal = SpreadSignal(
        id="s1", strategy="pairs_trading", code_a="AAA", code_b="BBB", hedge_ratio=1.0,
        side=SpreadSide.LONG_SPREAD, z_score=2.5, entry_z_threshold=2.0, exit_z_threshold=0.5,
        spread_mean=0.0, half_life_days=10.0, confidence=0.8, timestamp=RTH,
    )
    tight = RiskEngine(RiskConfig(limit_price_buffer_bps=5.0))
    wide = RiskEngine(RiskConfig(limit_price_buffer_bps=50.0))

    order_tight = tight.qualify_spread_order(signal, _snap(100.0), _snap(50.0), 1_000_000, 0.05)
    order_wide = wide.qualify_spread_order(signal, _snap(100.0), _snap(50.0), 1_000_000, 0.05)

    # LONG_SPREAD: buy leg A (limit above price), sell leg B (limit below price).
    assert order_tight.limit_price_a > 100.0
    assert order_wide.limit_price_a > order_tight.limit_price_a
    assert order_tight.limit_price_b < 50.0
    assert order_wide.limit_price_b < order_tight.limit_price_b


def test_qualify_portfolio_order_computes_limit_prices_per_code():
    from python.core.types import PortfolioTarget

    target = PortfolioTarget(strategy="xsection_mean_reversion", as_of=RTH, weights={"AAA": 0.05, "BBB": -0.03})
    engine = RiskEngine(RiskConfig(limit_price_buffer_bps=20.0))
    order = engine.qualify_portfolio_order(target, {"AAA": _snap(100.0), "BBB": _snap(50.0)}, 1_000_000)

    assert order.limit_prices["AAA"] > 100.0    # long -> buy limit above price
    assert order.limit_prices["BBB"] < 50.0     # short -> sell limit below price


# ── DailyLossTracker ─────────────────────────────────────────────────────────

def test_daily_loss_tracker_accumulates_within_session():
    tracker = DailyLossTracker()
    tracker.record_pnl(RTH, -1000.0)
    tracker.record_pnl(RTH, -500.0)
    assert tracker.session_pnl(RTH) == pytest.approx(-1500.0)


def test_daily_loss_tracker_resets_on_new_session_date():
    tracker = DailyLossTracker()
    tracker.record_pnl(RTH, -1000.0)
    next_day = RTH.replace(day=3)
    assert tracker.session_pnl(next_day) == 0.0


def test_daily_loss_kill_switch_triggers_past_threshold():
    tracker = DailyLossTracker()
    cfg = RiskConfig(max_daily_loss_pct=0.02)
    tracker.record_pnl(RTH, -25_000.0)  # -2.5% of a 1,000,000 account
    assert tracker.kill_switch_triggered(RTH, account_equity=1_000_000.0, cfg=cfg) is True


def test_daily_loss_kill_switch_not_triggered_under_threshold():
    tracker = DailyLossTracker()
    cfg = RiskConfig(max_daily_loss_pct=0.02)
    tracker.record_pnl(RTH, -5_000.0)  # -0.5%
    assert tracker.kill_switch_triggered(RTH, account_equity=1_000_000.0, cfg=cfg) is False


# ── qualify_microstructure_order ────────────────────────────────────────────

def test_qualify_microstructure_order_approves_and_sizes_by_stop_distance():
    engine = RiskEngine(RiskConfig(micro_risk_per_trade_pct=0.01, max_intraday_notional_pct=0.5))
    signal = _FakeMicroSignal(entry_price=100.0, stop_price=99.0)  # $1 stop distance
    order = engine.qualify_microstructure_order(signal, _snap(100.0), account_equity=1_000_000, open_micro_positions=0)

    assert order.approved is True
    # risk_dollars=10,000 / stop_dist=1 -> 10,000 shares by risk; notional cap
    # (0.5 * 1,000,000)/100 = 5,000 shares -> notional cap binds.
    assert order.qty == 5000
    assert order.entry_limit_price > 100.0       # long -> buy limit above price
    assert order.stop_limit_price < 99.0         # long exit -> sell limit below stop trigger
    assert order.target_price == 102.0


def test_qualify_microstructure_order_rejects_on_daily_loss_kill_switch():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=0.02))
    engine.daily_loss.record_pnl(RTH, -25_000.0)
    signal = _FakeMicroSignal()
    order = engine.qualify_microstructure_order(signal, _snap(), account_equity=1_000_000, open_micro_positions=0)
    assert order.approved is False
    assert order.rejection_reason == "daily_loss_kill_switch"


def test_qualify_microstructure_order_rejects_on_event_blackout():
    engine = RiskEngine()
    signal = _FakeMicroSignal()
    order = engine.qualify_microstructure_order(
        signal, _snap(), account_equity=1_000_000, open_micro_positions=0, event_blackout=True,
    )
    assert order.approved is False
    assert order.rejection_reason == "event_blackout"


def test_qualify_microstructure_order_rejects_when_max_open_positions_reached():
    engine = RiskEngine(RiskConfig(max_open_micro_positions=3))
    signal = _FakeMicroSignal()
    order = engine.qualify_microstructure_order(signal, _snap(), account_equity=1_000_000, open_micro_positions=3)
    assert order.approved is False
    assert order.rejection_reason == "max_open_micro_positions"


def test_qualify_microstructure_order_rejects_untradeable_snapshot():
    engine = RiskEngine()
    signal = _FakeMicroSignal()
    order = engine.qualify_microstructure_order(
        signal, _snap(tradeable=False), account_equity=1_000_000, open_micro_positions=0,
    )
    assert order.approved is False
    assert order.rejection_reason in ("snapshot_not_tradeable",)


def test_qualify_microstructure_order_short_direction_prices_correctly():
    engine = RiskEngine()
    signal = _FakeMicroSignal(direction="short", entry_price=100.0, stop_price=101.0, target_price=98.0)
    order = engine.qualify_microstructure_order(signal, _snap(100.0), account_equity=1_000_000, open_micro_positions=0)
    assert order.entry_limit_price < 100.0   # short -> sell limit below price
    assert order.stop_limit_price > 101.0    # short exit -> buy limit above stop trigger


# ── load_risk_config ─────────────────────────────────────────────────────────

def test_load_risk_config_reads_real_yaml_file():
    cfg = load_risk_config("configs/risk.yaml")
    assert cfg.limit_price_buffer_bps == 10.0
    assert cfg.max_daily_loss_pct == 0.02
    assert cfg.require_short_locate is True


def test_load_risk_config_missing_file_falls_back_to_defaults():
    cfg = load_risk_config("configs/does_not_exist.yaml")
    assert cfg == RiskConfig()


def test_load_risk_config_ignores_unknown_yaml_keys(tmp_path):
    path = tmp_path / "risk.yaml"
    path.write_text("limit_price_buffer_bps: 42.0\nmax_gross_leverage: 2.0\nnot_a_real_field: 1\n", encoding="utf-8")
    cfg = load_risk_config(str(path))
    assert cfg.limit_price_buffer_bps == 42.0
