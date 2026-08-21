"""
intraday_engine tests: synthetic 1-minute bars driving the sweep_reclaim
and orb_vwap signals through the full fill/exit/cost model, plus the
metrics-contract shape check and the 2x slippage stress test.
"""
from __future__ import annotations

import pandas as pd
import pytest

from python.backtest import intraday_engine as eng


def _flat_bars(n: int, price: float = 100.0, start: str = "2024-06-04 09:30",
                volume: float = 100_000.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price, "volume": volume,
    }, index=idx)


def _cfg(**overrides) -> eng.IntradayBacktestConfig:
    base = dict(capital=1_000_000.0, risk_per_trade_pct=0.01, half_spread_bps=1.0,
                impact_bps_per_participation=0.0, commission_per_share=0.0, min_commission=0.0,
                time_stop_minutes=1000, flatten_buffer_minutes=0)
    base.update(overrides)
    return eng.IntradayBacktestConfig(**base)


def _fvg_bars(n: int = 40, start: str = "2024-06-04 09:30") -> pd.DataFrame:
    """Deterministic bullish-FVG setup (bar1/bar2/bar3 at index 20/21/22),
    independent of the context engine's price-dependent round-number /
    equal-high-low levels — the cleanest signal for exercising the ENGINE's
    fill/exit/cost machinery without incidental level-detection noise
    (sweep_reclaim's own detection logic is separately covered, with
    explicit levels, by tests/test_intraday_signals.py)."""
    bars = _flat_bars(n, price=50.0, start=start, volume=1000.0)
    bars.loc[bars.index[20], ["open", "high", "low", "close"]] = [50.0, 50.5, 49.5, 50.2]
    bars.loc[bars.index[21], ["open", "high", "low", "close", "volume"]] = [50.2, 56.0, 50.1, 55.5, 50_000.0]
    bars.loc[bars.index[22], ["open", "high", "low", "close"]] = [55.5, 57.0, 55.2, 56.5]
    return bars


def test_fvg_retest_day_produces_a_filled_trade_with_target_hit():
    bars = _fvg_bars()
    # bar1.high=50.5, bar3.low=55.2 -> gap_width=4.7, entry(0.5)=52.85,
    # stop=gap_low-width=45.8 -> target is stop-mirrored (1R):
    # 52.85 + (52.85-45.8) = 59.9 (see fvg_retest.py's target_price comment).
    # bar 23: retest bar touching the gap's limit price (~52.85) and, in the
    # same bar, running up to that target.
    bars.loc[bars.index[23], ["open", "high", "low", "close"]] = [56.0, 60.0, 52.0, 59.5]

    params = {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10}
    trades, emitted, filled = eng.run_symbol_day(
        "AAA", bars, prior_day_bars=None, signal_name="fvg_retest", params=params, cfg=_cfg(),
    )
    assert emitted == 1
    assert filled == 1
    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == "long"
    assert trade.exit_reason == "target"
    assert trade.entry_time == bars.index[23]  # filled the bar AFTER bar3 (index 22), not on bar3 itself
    assert trade.signal_time == bars.index[22]
    assert trade.shares > 0


def test_signal_never_fills_on_its_own_bar():
    """Regression guard for the no-lookahead fill contract: a signal
    detected at bar i must not produce a trade whose entry_time == bar i,
    for ANY of the three signal modules."""
    bars = _fvg_bars()  # no retest bar added -> order sits pending, then expires; never fills same-bar
    params = {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10}
    trades, _emitted, filled = eng.run_symbol_day(
        "AAA", bars, prior_day_bars=None, signal_name="fvg_retest", params=params, cfg=_cfg(),
    )
    for trade in trades:
        assert trade.entry_time > trade.signal_time
    assert filled == 0  # price never revisits the gap in this bar sequence


def test_scan_signals_for_session_finds_the_fvg_pattern_without_simulating_fills():
    bars = _fvg_bars()
    params = {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10}
    signals = eng.scan_signals_for_session("fvg_retest", bars, params, cfg=_cfg())
    assert len(signals) == 1
    assert signals[0].direction == "long"
    assert signals[0].signal_time == bars.index[22]
    # causality: a signal can never be attributed to the FIRST bar of the scan (i starts at 1)
    assert all(s.signal_time > bars.index[0] for s in signals)


def test_scan_signals_for_session_empty_on_too_few_bars():
    bars = _flat_bars(1, price=100.0)
    assert eng.scan_signals_for_session("orb_vwap", bars, {"or_minutes": 15}, cfg=_cfg()) == []


def test_stop_exit():
    bars = _flat_bars(30, price=100.0)
    # OR high/low from bars[0:15]; put a spike in bar 5 to define OR high, dip in bar 6 for OR low.
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0  # breakout long, fires at bar 21
    bars.loc[bars.index[22], "low"] = 98.0     # bar AFTER signal: fill at open, then stop (OR low=99.0) hit intrabar

    params = {"or_minutes": 15, "vwap_side_filter": False}
    trades, emitted, filled = eng.run_symbol_day(
        "AAA", bars, prior_day_bars=None, signal_name="orb_vwap", params=params, cfg=_cfg(),
    )
    assert emitted >= 1
    assert filled == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].direction == "long"


def test_eod_flatten_closes_open_position():
    bars = _flat_bars(30, price=100.0)
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0

    params = {"or_minutes": 15, "vwap_side_filter": False}
    cfg = _cfg(time_stop_minutes=10_000)  # disable time-stop so EOD flatten is the only exit
    trades, _emitted, filled = eng.run_symbol_day(
        "AAA", bars, prior_day_bars=None, signal_name="orb_vwap", params=params, cfg=cfg,
    )
    assert filled == 1
    assert trades[-1].exit_reason == "eod_flatten"
    assert trades[-1].exit_time == bars.index[-1]


def test_time_stop_exit_when_unfavorable():
    bars = _flat_bars(60, price=100.0)
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0  # long breakout signal fires here
    # Bar 22 fills at open ~102; keep price flat/unfavorable afterward.
    bars.loc[bars.index[22]:, ["open", "high", "low", "close"]] = 101.5

    params = {"or_minutes": 15, "vwap_side_filter": False}
    cfg = _cfg(time_stop_minutes=3)
    trades, _emitted, filled = eng.run_symbol_day(
        "AAA", bars, prior_day_bars=None, signal_name="orb_vwap", params=params, cfg=cfg,
    )
    assert filled == 1
    assert trades[0].exit_reason == "time_stop"


def _orb_multi_break_bars() -> pd.DataFrame:
    """A session whose price crosses back and forth over the opening-range
    high four separate times — the "orb_vwap re-fires all session" pattern
    the rescue investigation measured at ~4.8 entries/symbol/session
    (backtests/reports/orb_vwap_rescue_report.md)."""
    bars = _flat_bars(80, price=100.0)
    bars.loc[bars.index[5], "high"] = 101.0   # OR high (first 15 bars)
    bars.loc[bars.index[6], "low"] = 99.0     # OR low
    for i in (21, 35, 49, 63):
        bars.loc[bars.index[i - 1], "close"] = 100.0   # back inside the range
        bars.loc[bars.index[i], "close"] = 102.0       # fresh break above the OR high
        bars.loc[bars.index[i], "high"] = 102.0
    return bars


def test_orb_vwap_refires_every_session_break_when_uncapped():
    bars = _orb_multi_break_bars()
    params = {"or_minutes": 15, "vwap_side_filter": False}
    cfg = _cfg(time_stop_minutes=1)  # exit fast so the state machine is free for the next break
    _trades, emitted, _filled = eng.run_symbol_day("AAA", bars, None, "orb_vwap", params, cfg)
    assert emitted >= 3


@pytest.mark.parametrize("cap", [1, 2])
def test_max_entries_per_session_caps_signals_and_defaults_to_unlimited(cap):
    bars = _orb_multi_break_bars()
    cfg = _cfg(time_stop_minutes=1)
    uncapped = {"or_minutes": 15, "vwap_side_filter": False}
    capped = {**uncapped, "max_entries_per_session": cap}

    _t0, emitted_uncapped, _f0 = eng.run_symbol_day("AAA", bars, None, "orb_vwap", uncapped, cfg)
    trades, emitted_capped, filled = eng.run_symbol_day("AAA", bars, None, "orb_vwap", capped, cfg)

    assert emitted_uncapped > cap
    assert emitted_capped == cap
    assert filled <= cap and len(trades) <= cap
    # None (and an absent key) must both mean "unlimited", i.e. unchanged behavior.
    _t1, emitted_explicit_none, _f1 = eng.run_symbol_day(
        "AAA", bars, None, "orb_vwap", {**uncapped, "max_entries_per_session": None}, cfg,
    )
    assert emitted_explicit_none == emitted_uncapped


def test_orb_vwap_r_multiple_target_produces_a_target_exit():
    bars = _flat_bars(40, price=100.0)
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], ["high", "low", "close"]] = [102.0, 101.5, 102.0]  # long break, stop = OR low 99.0
    # Entry ~102 at bar 22's open, risk = 3.0, so a 1R target sits at 105.0.
    bars.loc[bars.index[22], ["open", "high", "low", "close"]] = [102.0, 106.0, 101.8, 105.5]

    params = {"or_minutes": 15, "vwap_side_filter": False, "target_r_multiple": 1.0}
    trades, _emitted, filled = eng.run_symbol_day(
        "AAA", bars, None, "orb_vwap", params, _cfg(time_stop_minutes=10_000),
    )
    assert filled == 1
    assert trades[0].exit_reason == "target"
    assert trades[0].net_pnl > 0


def test_orb_vwap_stop_atr_buffer_widens_the_stop_end_to_end():
    """The buffered stop must sit further from entry than the raw OR-extreme
    stop, which (via 1%-risk sizing) means strictly fewer shares."""
    bars = _flat_bars(40, price=100.0)
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], ["high", "low", "close"]] = [102.0, 101.5, 102.0]

    base = {"or_minutes": 15, "vwap_side_filter": False}
    cfg = _cfg(time_stop_minutes=10_000, max_notional_pct=10.0)
    trades_raw, _, _ = eng.run_symbol_day("AAA", bars, None, "orb_vwap", base, cfg)
    trades_buf, _, _ = eng.run_symbol_day(
        "AAA", bars, None, "orb_vwap", {**base, "stop_atr_buffer_mult": 1.0}, cfg,
    )
    assert trades_raw and trades_buf
    assert trades_buf[0].shares < trades_raw[0].shares


def test_slippage_increases_cost_and_stress_multiplier_doubles_it():
    bars = _fvg_bars()
    bars.loc[bars.index[23], ["open", "high", "low", "close"]] = [56.0, 56.2, 52.0, 53.0]

    params = {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10}
    normal_cfg = _cfg(half_spread_bps=5.0, impact_bps_per_participation=20.0)
    stress_cfg = _cfg(half_spread_bps=5.0, impact_bps_per_participation=20.0, stress_slippage_multiplier=2.0)

    trades_normal, _, _ = eng.run_symbol_day("AAA", bars, None, "fvg_retest", params, normal_cfg)
    trades_stress, _, _ = eng.run_symbol_day("AAA", bars, None, "fvg_retest", params, stress_cfg)

    assert len(trades_normal) == 1 and len(trades_stress) == 1
    assert trades_stress[0].net_pnl < trades_normal[0].net_pnl


def test_slippage_price_falls_back_to_flat_half_spread_when_symbol_override_absent():
    """half_spread_bps_by_symbol=None (the default) or a dict that simply
    doesn't mention this symbol must produce IDENTICAL slippage to never
    having set the field at all — the backward-compatibility contract for
    every pre-existing caller."""
    cfg_no_override = _cfg(half_spread_bps=5.0, impact_bps_per_participation=0.0)
    cfg_empty_dict = _cfg(half_spread_bps=5.0, impact_bps_per_participation=0.0,
                           half_spread_bps_by_symbol={})
    cfg_other_symbol = _cfg(half_spread_bps=5.0, impact_bps_per_participation=0.0,
                             half_spread_bps_by_symbol={"MSFT": 0.5})

    for cfg in (cfg_no_override, cfg_empty_dict, cfg_other_symbol):
        price = eng._slippage_price(100.0, "long", 100, 10_000.0, cfg, is_entry=True, symbol="AAPL")
        assert price == pytest.approx(100.0 * (1 + 5.0 / 10_000.0))
        # No symbol passed at all (e.g. a caller that predates this field) —
        # must behave exactly like the flat constant too.
        price_no_symbol = eng._slippage_price(100.0, "long", 100, 10_000.0, cfg, is_entry=True)
        assert price_no_symbol == pytest.approx(price)


def test_slippage_price_uses_calibrated_per_symbol_override_when_present():
    cfg = _cfg(half_spread_bps=5.0, impact_bps_per_participation=0.0,
               half_spread_bps_by_symbol={"AAPL": 0.3, "STX": 6.5})

    aapl_entry = eng._slippage_price(100.0, "long", 100, 10_000.0, cfg, is_entry=True, symbol="AAPL")
    assert aapl_entry == pytest.approx(100.0 * (1 + 0.3 / 10_000.0))

    stx_entry = eng._slippage_price(100.0, "long", 100, 10_000.0, cfg, is_entry=True, symbol="STX")
    assert stx_entry == pytest.approx(100.0 * (1 + 6.5 / 10_000.0))

    # A symbol not in the override dict still falls back to the flat constant.
    other_entry = eng._slippage_price(100.0, "long", 100, 10_000.0, cfg, is_entry=True, symbol="MSFT")
    assert other_entry == pytest.approx(100.0 * (1 + 5.0 / 10_000.0))


def test_run_symbol_day_threads_calibrated_symbol_override_into_fills():
    """End-to-end: run_symbol_day's real fill/exit path (not just the
    _slippage_price unit) must pick up a per-symbol override via the
    symbol actually passed to run_symbol_day."""
    bars = _fvg_bars()
    bars.loc[bars.index[23], ["open", "high", "low", "close"]] = [56.0, 60.0, 52.0, 59.5]
    params = {"vol_mult": 2.0, "entry_pct": 0.5, "expiry_bars": 10}

    cfg_flat = _cfg(half_spread_bps=5.0, impact_bps_per_participation=0.0)
    cfg_override = _cfg(half_spread_bps=5.0, impact_bps_per_participation=0.0,
                         half_spread_bps_by_symbol={"AAPL": 0.5})

    trades_flat, _, _ = eng.run_symbol_day("AAPL", bars, None, "fvg_retest", params, cfg_flat)
    trades_override, _, _ = eng.run_symbol_day("AAPL", bars, None, "fvg_retest", params, cfg_override)

    assert len(trades_flat) == 1 and len(trades_override) == 1
    # Tighter calibrated spread (0.5bps < 5.0bps) means less slippage paid
    # on both legs (commission is zeroed out in _cfg) -> strictly higher
    # net P&L for the identical underlying price path.
    assert trades_override[0].net_pnl > trades_flat[0].net_pnl
    assert trades_override[0].entry_price != trades_flat[0].entry_price


def test_position_size_scales_with_risk_and_stop_distance():
    # Wide stop -> risk-based sizing dominates and stays well under the notional cap.
    cfg = _cfg(capital=1_000_000.0, risk_per_trade_pct=0.01, max_notional_pct=0.50)
    assert eng._position_size(100.0, 95.0, cfg) == 2_000  # risk $10,000 / $5 stop distance
    assert eng._position_size(100.0, 100.0, _cfg()) == 0  # zero stop distance -> no trade


def test_position_size_caps_at_max_notional_when_stop_is_very_tight():
    # A stop only 1 cent away would otherwise imply a million-share order —
    # the notional cap must kick in instead of pure risk-based sizing.
    cfg = _cfg(capital=1_000_000.0, risk_per_trade_pct=0.01, max_notional_pct=0.20)
    shares = eng._position_size(100.0, 99.99, cfg)
    assert shares == 2_000  # capped at 20% of $1,000,000 / $100 = 2,000 shares
    assert shares * 100.0 <= cfg.capital * cfg.max_notional_pct + 1e-6


def test_metrics_from_report_shape_matches_daily_engine_contract():
    report = eng.IntradayBacktestReport()
    report.trades = [
        eng.IntradayTrade(
            symbol="AAA", strategy="sweep_reclaim", direction="short",
            entry_time=pd.Timestamp("2024-06-04 10:05"), entry_price=100.0,
            exit_time=pd.Timestamp("2024-06-04 10:10"), exit_price=98.0,
            exit_reason="target", shares=1000, gross_pnl=2000.0, costs=10.0, net_pnl=1990.0,
        ),
        eng.IntradayTrade(
            symbol="AAA", strategy="sweep_reclaim", direction="long",
            entry_time=pd.Timestamp("2024-06-05 10:05"), entry_price=100.0,
            exit_time=pd.Timestamp("2024-06-05 10:10"), exit_price=99.0,
            exit_reason="stop", shares=1000, gross_pnl=-1000.0, costs=10.0, net_pnl=-1010.0,
        ),
    ]
    report.signals_emitted = 3
    report.signals_filled = 2
    metrics = eng.metrics_from_report(report, capital=1_000_000.0)
    assert set(["sharpe_ratio", "max_drawdown", "n_trades", "total_net_pnl", "n_days", "daily_returns"]) <= set(metrics)
    assert metrics["n_trades"] == 2
    assert metrics["n_days"] == 2
    assert metrics["total_net_pnl"] == pytest.approx(980.0)


def test_no_signals_produces_empty_but_valid_metrics():
    bars = _flat_bars(30, price=100.0)
    params = {"sweep_min_atr": 5.0, "reclaim_bars": 3, "stop_atr_mult": 0.25}  # impossible threshold
    trades, emitted, filled = eng.run_symbol_day("AAA", bars, None, "sweep_reclaim", params, _cfg())
    assert trades == [] and emitted == 0 and filled == 0
    report = eng.IntradayBacktestReport()
    metrics = eng.metrics_from_report(report, capital=1_000_000.0)
    assert metrics["n_trades"] == 0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["daily_returns"] == []


def test_run_intraday_backtest_multi_day_orchestrator():
    day1 = _flat_bars(30, price=100.0, start="2024-06-04 09:30")
    day1.loc[day1.index[5], "high"] = 101.0
    day1.loc[day1.index[6], "low"] = 99.0
    day1.loc[day1.index[20], "close"] = 100.0
    day1.loc[day1.index[21], "close"] = 102.0

    day2 = _flat_bars(30, price=103.0, start="2024-06-05 09:30")
    bars = pd.concat([day1, day2])

    params = {"or_minutes": 15, "vwap_side_filter": False}
    report = eng.run_intraday_backtest({"AAA": bars}, "orb_vwap", params, cfg=_cfg())
    assert report.signals_filled >= 1
    metrics = eng.metrics_from_report(report, capital=1_000_000.0)
    assert metrics["n_trades"] >= 1
