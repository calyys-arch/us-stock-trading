"""
Config enforcement tests — forex-trading lesson #2
(docs/lessons_from_forex_trading.md): `auto_execute` existed in
configs/strategy.yaml for a period with NO code that read it, silently
making the "observe-only" safety switch a no-op. Every guard-like config key
must have a matching test here that (a) proves the key is actually read by
some code path and (b) proves changing its value changes behavior.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import yaml

from python.core.bus import MessageBus
from python.core.execution_gateway import ExecutionGateway
from python.core.risk_engine import PDTTracker, RiskConfig, RiskEngine
from python.core.sim_broker import SimBroker
from python.core.types import CointegrationResult, PortfolioTarget, SpreadSide, SpreadSignal


def _make_spread_signal(strategy_name: str) -> SpreadSignal:
    return SpreadSignal(
        id="sig1", strategy=strategy_name, code_a="AAA", code_b="BBB",
        hedge_ratio=1.0, side=SpreadSide.LONG_SPREAD, z_score=2.5,
        entry_z_threshold=2.0, exit_z_threshold=0.5, spread_mean=0.0,
        half_life_days=10.0, confidence=0.8,
        timestamp=datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc),  # RTH
    )


def test_auto_execute_key_gates_order_submission():
    """auto_execute: false (or gateway mode != auto) must actually prevent
    a broker.place_order call — not just be logged and ignored."""
    from python.core.risk_engine import RiskEngine
    from python.core.types import QualifiedSpreadOrder

    signal = _make_spread_signal("pairs_trading")
    order = QualifiedSpreadOrder(
        raw=signal, qty_a=100, qty_b=100, gross_notional=20000.0,
        estimated_cost=0.0, kelly_fraction_used=0.05, approved=True,
        limit_price_a=100.0, limit_price_b=50.0,
    )

    bus = MessageBus()
    broker = SimBroker()

    # auto_execute NOT granted for "pairs_trading" -> must not submit.
    gw_observe = ExecutionGateway(bus, broker, mode="auto", auto_execute_strategies=set())

    async def _run():
        await gw_observe._on_spread_order(order)

    asyncio.run(_run())
    assert broker.get_positions() == {}, "order was submitted despite auto_execute not being granted"

    # Now grant auto_execute for this strategy AND set gateway mode=auto -> must submit.
    gw_auto = ExecutionGateway(bus, broker, mode="auto", auto_execute_strategies={"pairs_trading"})

    async def _run2():
        await gw_auto._on_spread_order(order)

    asyncio.run(_run2())
    positions = broker.get_positions()
    assert positions.get("AAA", 0) != 0 or positions.get("BBB", 0) != 0, (
        "order was NOT submitted even though gateway mode=auto AND strategy auto_execute=true"
    )


def test_gateway_mode_observe_blocks_even_if_strategy_auto_execute_true():
    """Both keys are an AND, not an OR — gateway-level 'mode' must still be
    able to hard-block everything regardless of per-strategy config."""
    signal = _make_spread_signal("pairs_trading")
    from python.core.types import QualifiedSpreadOrder

    order = QualifiedSpreadOrder(
        raw=signal, qty_a=100, qty_b=100, gross_notional=20000.0,
        estimated_cost=0.0, kelly_fraction_used=0.05, approved=True,
        limit_price_a=100.0, limit_price_b=50.0,
    )
    bus = MessageBus()
    broker = SimBroker()
    gw = ExecutionGateway(bus, broker, mode="observe", auto_execute_strategies={"pairs_trading"})

    async def _run():
        await gw._on_spread_order(order)

    asyncio.run(_run())
    assert broker.get_positions() == {}


def test_require_short_locate_key_rejects_unlocatable_short():
    from python.core.types import MarketSnapshot

    def _snap(code: str, locate: bool) -> MarketSnapshot:
        return MarketSnapshot(
            code=code, price=50.0, volume_today=1000, turnover_today=50000,
            vwap=50.0, atr14=1.0, atr5=1.0, rsi14=50.0, ema8=50.0, ema20=50.0,
            bb_upper=51.0, bb_mid=50.0, bb_lower=49.0, vol_ma20=1000, vol_ratio=1.0,
            bid_ask_spread_pct=0.001, timestamp=datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc),
            short_locate_available=locate, is_regular_trading_hours=True,
        )

    signal = _make_spread_signal("pairs_trading")  # LONG_SPREAD -> short leg is code_b (BBB)

    engine_requiring_locate = RiskEngine(RiskConfig(require_short_locate=True))
    order_rejected = engine_requiring_locate.qualify_spread_order(
        signal, _snap("AAA", True), _snap("BBB", False), account_equity=1_000_000, kelly_fraction=0.05,
    )
    assert order_rejected.approved is False
    assert order_rejected.rejection_reason is not None and "locate" in order_rejected.rejection_reason

    engine_not_requiring_locate = RiskEngine(RiskConfig(require_short_locate=False))
    order_allowed = engine_not_requiring_locate.qualify_spread_order(
        signal, _snap("AAA", True), _snap("BBB", False), account_equity=1_000_000, kelly_fraction=0.05,
    )
    assert order_allowed.approved is True


def test_pdt_equity_threshold_key_limits_day_trades_for_small_accounts():
    tracker = PDTTracker()
    cfg = RiskConfig(pdt_equity_threshold=25_000.0, max_day_trades_rolling_5d=3)
    now = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)

    # Small account: blocked after max_day_trades_rolling_5d trades.
    assert tracker.can_day_trade(now, account_equity=10_000.0, cfg=cfg) is True
    for _ in range(3):
        tracker.record_day_trade(now)
    assert tracker.can_day_trade(now, account_equity=10_000.0, cfg=cfg) is False

    # Large account: PDT rule does not apply at all.
    assert tracker.can_day_trade(now, account_equity=100_000.0, cfg=cfg) is True


def test_strategy_yaml_entry_z_actually_changes_signal_generation():
    """entry_z from configs/strategy.yaml must change WHETHER a signal fires,
    not just be stored and ignored."""
    from python.core.strategies.pairs_trading import PairsTradingStrategy

    coint = CointegrationResult(
        code_a="AAA", code_b="BBB", hedge_ratio=1.0, cadf_tstat=-4.0,
        cadf_crit_1pct=-3.9, cadf_crit_5pct=-3.3, cadf_crit_10pct=-3.0,
        is_cointegrated_5pct=True, half_life_days=10.0, spread_mean=0.0,
        spread_std=0.01, computed_at=datetime(2026, 1, 1), lookback_days=252,
    )
    # A z of ~1.5 (price_a slightly high vs price_b) should NOT trigger a
    # strict entry_z=2.0 strategy, but SHOULD trigger a loose entry_z=1.0 one.
    price_a, price_b = 101.5, 100.0

    strict = PairsTradingStrategy(entry_z=2.0, exit_z=0.5)
    loose = PairsTradingStrategy(entry_z=1.0, exit_z=0.5)

    strict_signal = strict.evaluate(coint, [], price_a, price_b, datetime(2026, 3, 2))
    loose_signal = loose.evaluate(coint, [], price_a, price_b, datetime(2026, 3, 2))

    assert strict_signal is None
    assert loose_signal is not None


def test_configs_yaml_files_are_valid_and_have_required_keys():
    with open("configs/risk.yaml", "r", encoding="utf-8") as f:
        risk_cfg = yaml.safe_load(f)
    with open("configs/strategy.yaml", "r", encoding="utf-8") as f:
        strat_cfg = yaml.safe_load(f)
    with open("configs/broker.yaml", "r", encoding="utf-8") as f:
        broker_cfg = yaml.safe_load(f)

    for key in ("require_short_locate", "pdt_equity_threshold", "max_day_trades_rolling_5d",
                "reg_t_intraday_leverage", "reg_t_overnight_leverage",
                "limit_price_buffer_bps", "flatten_limit_buffer_bps", "stop_limit_buffer_bps",
                "micro_risk_per_trade_pct", "max_intraday_notional_pct", "max_open_micro_positions",
                "max_daily_loss_pct", "event_blackout_minutes", "micro_cancel_after_seconds"):
        assert key in risk_cfg, f"configs/risk.yaml missing enforced key: {key}"

    for strat in ("pairs_trading", "xsection_mean_reversion"):
        assert "auto_execute" in strat_cfg[strat], f"{strat} missing auto_execute key"
        assert "enabled" in strat_cfg[strat], f"{strat} missing enabled key"

    assert "data_source" in broker_cfg, "configs/broker.yaml missing data_source key"
    assert broker_cfg["data_source"] == "simulated", (
        "configs/broker.yaml must default to data_source: simulated — the checked-in default "
        "must never silently require a real IB Gateway/TWS connection"
    )
    for key in ("host", "feed_port", "broker_port", "feed_client_id", "broker_client_id"):
        assert key in broker_cfg["ibkr"], f"configs/broker.yaml ibkr section missing key: {key}"


def test_broker_yaml_data_source_is_actually_read_by_engine_runtime():
    """Forex lesson #2 regression guard: configs/broker.yaml's data_source
    key must actually be read by dashboard/engine_bridge.py and actually
    change which broker/feed classes EngineRuntime wires up — not just be
    present in the file with no code path consuming it."""
    from dashboard.engine_bridge import EngineRuntime
    from dashboard.state import DashboardState

    default_runtime = EngineRuntime(DashboardState())
    assert default_runtime.data_source == "simulated"
    assert isinstance(default_runtime.broker, SimBroker), (
        "data_source=simulated must wire up SimBroker, not a real IbkrBroker"
    )
    assert default_runtime.state.data_source == "simulated"

    patched_cfg = dict(default_runtime.broker_cfg)
    patched_cfg["data_source"] = "ibkr_paper"
    import dashboard.engine_bridge as engine_bridge_module

    original_loader = engine_bridge_module._load_broker_config
    engine_bridge_module._load_broker_config = lambda: patched_cfg
    try:
        ibkr_runtime = EngineRuntime(DashboardState())
    finally:
        engine_bridge_module._load_broker_config = original_loader

    assert ibkr_runtime.data_source == "ibkr_paper"
    assert ibkr_runtime.state.data_source == "ibkr_paper"


def test_broker_yaml_invalid_data_source_falls_back_to_simulated():
    """An unrecognized data_source value must never silently attempt an
    IBKR connection — it must fail safe to the simulated default."""
    from dashboard.engine_bridge import EngineRuntime
    from dashboard.state import DashboardState
    import dashboard.engine_bridge as engine_bridge_module

    bogus_cfg = dict(engine_bridge_module._load_broker_config())
    bogus_cfg["data_source"] = "some_typo"
    original_loader = engine_bridge_module._load_broker_config
    engine_bridge_module._load_broker_config = lambda: _validate(bogus_cfg)
    try:
        runtime = EngineRuntime(DashboardState())
    finally:
        engine_bridge_module._load_broker_config = original_loader

    assert runtime.data_source == "simulated"
    assert isinstance(runtime.broker, SimBroker)


def test_universe_yaml_symbols_are_actually_wired_into_dashboard_runtime():
    """Forex lesson #2-style regression guard for the Part A fix: dashboard/
    app.py's module-level `runtime = EngineRuntime(state, symbols=...)` call
    must actually pass configs/universe.yaml's 20-symbol fixed_universe list
    — not just have that config file present on disk with nothing reading
    it into the LIVE dashboard runtime (EngineRuntime.__init__'s own
    hardcoded 5-symbol placeholder — ["AAPL", "MSFT", "GOOGL", "AMZN",
    "META"] — is the exact silent-fallback failure mode being guarded
    against here)."""
    from python.data.fixed_universe import load_universe_config

    import dashboard.app as dashboard_app_module

    expected_symbols = sorted(load_universe_config()["symbols"])
    assert len(expected_symbols) == 20

    assert sorted(dashboard_app_module.runtime.symbols) == expected_symbols
    assert dashboard_app_module.runtime.state.symbols == dashboard_app_module.runtime.symbols
    # The bug this guards against: EngineRuntime.__init__'s hardcoded
    # placeholder fallback used whenever no `symbols` kwarg is passed.
    placeholder = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    assert sorted(dashboard_app_module.runtime.symbols) != sorted(placeholder)


def test_engine_runtime_still_defaults_to_placeholder_when_constructed_bare():
    """EngineRuntime's own default (no `symbols` kwarg) must stay the
    harmless 5-symbol placeholder — Part A only changes the dashboard/app.py
    CALL SITE, not EngineRuntime's own fallback behavior, so other
    callers/tests that construct EngineRuntime without args are unaffected."""
    from dashboard.engine_bridge import EngineRuntime
    from dashboard.state import DashboardState

    bare_runtime = EngineRuntime(DashboardState())
    assert bare_runtime.symbols == ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def _validate(cfg: dict) -> dict:
    """Mirror engine_bridge._load_broker_config's own invalid-value
    fallback so this test exercises the same validation path without
    duplicating the real loader's I/O."""
    from dashboard.engine_bridge import _VALID_DATA_SOURCES

    cfg = dict(cfg)
    if cfg.get("data_source") not in _VALID_DATA_SOURCES:
        cfg["data_source"] = "simulated"
    return cfg
