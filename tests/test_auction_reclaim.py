"""Unit tests for auction_reclaim — Creamer-style 5-minute auction reclaim."""
from __future__ import annotations

import pandas as pd
import pytest

from python.backtest import intraday_engine as eng
from python.microstructure import context as ctx
from python.microstructure.gex import GexSnapshot
from python.microstructure.signals import auction_reclaim as ar


def _session(date: str, start_px: float, end_px: float, *, heavy_high: bool) -> pd.DataFrame:
    idx = pd.date_range(f"{date} 09:30", periods=60, freq="1min")
    closes = [start_px + (end_px - start_px) * (i / 59.0) for i in range(60)]
    mid = (start_px + end_px) / 2.0
    if heavy_high:
        volume = [400.0 if c < mid else 4000.0 for c in closes]
    else:
        volume = [400.0 if c > mid else 4000.0 for c in closes]
    return pd.DataFrame({
        "open": closes,
        "high": [c + 0.05 for c in closes],
        "low": [c - 0.05 for c in closes],
        "close": closes,
        "volume": volume,
    }, index=idx)


def _prior_up_day() -> pd.DataFrame:
    """Most recent session in a value-up stack: closed near the high, volume
    in the upper half so VAL sits above the fib discount pocket."""
    return _session("2024-06-03", 100.0, 110.0, heavy_high=True)


def _prior_down_day() -> pd.DataFrame:
    return _session("2024-06-03", 110.0, 100.0, heavy_high=False)


def _rising_sessions() -> list[pd.DataFrame]:
    """Two completed sessions whose 1h charts make HH + HL (value-up)."""
    return [
        _session("2024-05-31", 90.0, 100.0, heavy_high=True),
        _prior_up_day(),
    ]


def _falling_sessions() -> list[pd.DataFrame]:
    return [
        _session("2024-05-31", 120.0, 110.0, heavy_high=False),
        _prior_down_day(),
    ]


def _overlapping_sessions() -> list[pd.DataFrame]:
    """Last session inside the prior 1h range — sideways, stand aside."""
    return [
        _session("2024-05-31", 100.0, 110.0, heavy_high=True),
        _session("2024-06-03", 104.0, 108.0, heavy_high=True),
    ]


def _prior_balanced_day() -> pd.DataFrame:
    idx = pd.date_range("2024-06-03 09:30", periods=60, freq="1min")
    return pd.DataFrame({
        "open": 105.0, "high": 105.4, "low": 104.6, "close": 105.0, "volume": 1000.0,
    }, index=idx)


def _today_long_setup() -> pd.DataFrame:
    """25 one-minute bars: drift into discount, absorb on the 09:45 5m
    probe, reclaim on the 09:50 5m confirm. Last print is 09:54."""
    idx = pd.date_range("2024-06-04 09:30", periods=25, freq="1min")
    rows = []
    # 09:30-09:44 (15 bars, three complete 5m bins): drift 108 → 103.
    for i in range(15):
        px = 108.0 - i * 0.30
        rows.append((px, px + 0.05, px - 0.05, px, 800.0))
    # 09:45-09:49 probe: sell to 101.5, reject, close 102.3.
    probe = [
        (103.0, 103.1, 102.6, 102.7, 2500.0),
        (102.7, 102.8, 102.2, 102.3, 2500.0),
        (102.3, 102.4, 101.8, 101.9, 2500.0),
        (101.9, 102.0, 101.5, 101.7, 2500.0),
        (101.7, 102.4, 101.55, 102.3, 2500.0),
    ]
    rows.extend(probe)
    # 09:50-09:54 confirm: higher low 101.8, bullish close 102.8.
    confirm = [
        (102.3, 102.5, 102.1, 102.2, 2200.0),
        (102.2, 102.4, 102.0, 102.3, 2200.0),
        (102.3, 102.5, 101.9, 102.4, 2200.0),
        (102.4, 102.6, 102.1, 102.5, 2200.0),
        (102.5, 102.9, 101.85, 102.8, 2200.0),
    ]
    rows.extend(confirm)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_fires_long_on_discount_absorption_and_reclaim():
    sig = ar.evaluate_auction_reclaim(
        _today_long_setup(), _prior_up_day(), symbol="TEST",
        prior_sessions=_rising_sessions(),
    )
    assert sig is not None
    assert sig.direction == "long"
    assert sig.strategy == "auction_reclaim"
    assert sig.symbol == "TEST"
    assert sig.stop_price < sig.entry_price
    assert sig.target_price > sig.entry_price
    assert sig.context["value_bias"] == "up"
    assert sig.context["environment_tf"] == "1h"
    assert sig.context["environment_when"] == "preopen"
    assert sig.context["environment_structure"] == "value_up"
    assert sig.context["preopen_trader_side"] == "buying"
    assert sig.context["preopen_last_volume"] > 0
    assert sig.context["gex_source"] == "unavailable_vol_regime_proxy"
    assert sig.context["tier"].startswith("bar_only_5m_proxy")
    assert sig.context["chart_minutes"] == 5
    assert sig.context["atr_timeframe"] == "5m"
    risk = sig.entry_price - sig.stop_price
    assert sig.target_price == pytest.approx(sig.entry_price + 1.5 * risk)


def test_no_signal_on_balanced_prior_session():
    assert ar.evaluate_auction_reclaim(_today_long_setup(), _prior_balanced_day()) is None


def test_no_signal_without_two_preopen_1h_sessions():
    """One prior day is location, not a pre-open 1h environment read."""
    assert ar.evaluate_auction_reclaim(_today_long_setup(), _prior_up_day()) is None


def test_no_signal_when_preopen_1h_is_sideways():
    assert ar.evaluate_auction_reclaim(
        _today_long_setup(), _prior_up_day(), prior_sessions=_overlapping_sessions(),
    ) is None


def test_preopen_1h_reads_last_two_sessions_not_today():
    env = ar.preopen_1h_environment(_rising_sessions())
    assert env.structure == "value_up"
    assert env.bias == "up"
    assert env.n_hourly_bars >= 2
    today_dump = _today_long_setup()
    frozen = ar.preopen_1h_environment(
        ar.prior_rth_sessions(pd.concat(_rising_sessions() + [today_dump]), today_dump.index[0]),
    )
    assert frozen.structure == "value_up"
    assert frozen.last_high == env.last_high
    assert frozen.last_low == env.last_low
    assert env.trader_side == "buying"
    assert env.trader_momentum.startswith("buying_")
    assert env.trader_pressure is not None and env.trader_pressure > 0
    assert env.last_volume is not None and env.prev_volume is not None
    assert env.hourly[0]["session"] == "prev"
    assert "pressure" in env.hourly[0]


def test_preopen_1h_trader_pace_is_not_price_structure():
    thin = _prior_up_day().copy()
    thin["volume"] = thin["volume"] * 0.2
    fading = ar.preopen_1h_environment([
        _session("2024-05-31", 90.0, 100.0, heavy_high=True),
        thin,
    ])
    assert fading.structure == "value_up"
    assert fading.trader_side == "buying"
    assert fading.trader_pace == "fading"

    heavy = _prior_up_day().copy()
    heavy["volume"] = heavy["volume"] * 2.0
    building = ar.preopen_1h_environment([
        _session("2024-05-31", 90.0, 100.0, heavy_high=True),
        heavy,
    ])
    assert building.structure == "value_up"
    assert building.trader_side == "buying"
    assert building.trader_pace == "building"


def test_preopen_1h_trader_side_independent_of_value_structure():
    """HH+HL (value-up) can still be a sold 1h auction — volume/price
    do not confirm each other."""
    sold_up_range = _session("2024-06-03", 111.0, 103.0, heavy_high=False)
    env = ar.preopen_1h_environment([
        _session("2024-05-31", 90.0, 100.0, heavy_high=True),
        sold_up_range,
    ])
    assert env.structure == "value_up"
    assert env.trader_side == "selling"
    assert env.trader_momentum.startswith("selling_")


def test_preopen_1h_selling_momentum_on_value_down():
    env = ar.preopen_1h_environment(_falling_sessions())
    assert env.structure == "value_down"
    assert env.trader_side == "selling"


def test_no_signal_without_prior_day():
    assert ar.evaluate_auction_reclaim(_today_long_setup(), None) is None


def test_no_signal_mid_five_minute_bin():
    bars = _today_long_setup().iloc[:-1]  # last print 09:53, incomplete 5m
    assert ar.evaluate_auction_reclaim(bars, _prior_up_day(), prior_sessions=_rising_sessions()) is None


def test_no_signal_after_session_window():
    bars = _today_long_setup()
    bars.index = bars.index + pd.Timedelta(hours=2)  # 11:30+
    assert ar.evaluate_auction_reclaim(bars, _prior_up_day(), prior_sessions=_rising_sessions()) is None


def test_no_signal_without_volume():
    bars = _today_long_setup()
    bars.loc[bars.index[-10:], "volume"] = 100.0
    assert ar.evaluate_auction_reclaim(
        bars, _prior_up_day(), min_rel_volume=1.2, prior_sessions=_rising_sessions(),
    ) is None


def test_no_lookahead():
    bars = _today_long_setup()
    prior = _prior_up_day()
    env = _rising_sessions()
    sig = ar.evaluate_auction_reclaim(bars, prior, prior_sessions=env)
    later = bars.copy()
    later.loc[bars.index[-1] + pd.Timedelta(minutes=1)] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    sig2 = ar.evaluate_auction_reclaim(later.iloc[: len(bars)], prior, prior_sessions=env)
    assert sig is not None and sig2 is not None
    assert sig.entry_price == sig2.entry_price
    assert sig.stop_price == sig2.stop_price


def test_short_on_premium_absorption_and_reclaim():
    prior = _prior_down_day()
    idx = pd.date_range("2024-06-04 09:30", periods=25, freq="1min")
    rows = []
    for i in range(15):
        px = 102.0 + i * 0.30
        rows.append((px, px + 0.05, px - 0.05, px, 800.0))
    # Probe into premium: push to 108.5, reject, close 107.7.
    rows.extend([
        (107.0, 107.4, 106.9, 107.3, 2500.0),
        (107.3, 107.8, 107.2, 107.6, 2500.0),
        (107.6, 108.2, 107.5, 108.0, 2500.0),
        (108.0, 108.5, 107.8, 108.2, 2500.0),
        (108.2, 108.5, 107.6, 107.7, 2500.0),
    ])
    # Confirm: lower high, bearish close.
    rows.extend([
        (107.7, 107.9, 107.5, 107.6, 2200.0),
        (107.6, 107.8, 107.4, 107.5, 2200.0),
        (107.5, 107.7, 107.3, 107.4, 2200.0),
        (107.4, 107.6, 107.2, 107.3, 2200.0),
        (107.3, 108.2, 107.0, 107.1, 2200.0),
    ])
    bars = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    sig = ar.evaluate_auction_reclaim(bars, prior, prior_sessions=_falling_sessions())
    assert sig is not None
    assert sig.direction == "short"
    assert sig.stop_price > sig.entry_price


def test_dispatches_through_intraday_engine():
    sig = eng._evaluate_signal(
        "auction_reclaim",
        _today_long_setup(),
        "TEST",
        {"min_rel_volume": 1.2, "min_wick_frac": 0.45, "stop_atr_mult": 0.15, "target_r_multiple": 1.5},
        eng.IntradayBacktestConfig(),
        _prior_up_day(),
        None,
        prior_sessions=_rising_sessions(),
    )
    assert sig is not None
    assert sig.strategy == "auction_reclaim"
    assert sig.symbol == "TEST"


def test_stop_buffer_uses_five_minute_atr_not_one_minute():
    bars = _today_long_setup()
    prior = _prior_up_day()
    sig = ar.evaluate_auction_reclaim(bars, prior, prior_sessions=_rising_sessions())
    assert sig is not None
    atr_1m = float(ctx.atr(bars, period=14).iloc[-1])
    assert sig.context["atr"] > atr_1m
    expected_stop = float(sig.context["probe_low"]) - 0.15 * sig.context["atr"]
    assert sig.stop_price == pytest.approx(expected_stop)


def test_resample_closed_5m_drops_incomplete_bin():
    bars = _today_long_setup()
    complete = ar._resample_closed_5m(bars)
    assert len(complete) == 5
    incomplete = ar._resample_closed_5m(bars.iloc[:-1])
    assert len(incomplete) == 4


def _ticks(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["time", "price", "size", "ticker_direction"])


def _long_footprint_ticks(*, absorb: bool, extra_after_now: list[tuple] | None = None) -> pd.DataFrame:
    """20+ sided prints in the 09:45 probe and 09:50 confirm bins."""
    rows = []
    probe = pd.Timestamp("2024-06-04 09:45:00")
    confirm = pd.Timestamp("2024-06-04 09:50:00")
    probe_side = "SELL" if absorb else "BUY"
    for i in range(22):
        rows.append((probe + pd.Timedelta(seconds=i), 101.55, 10.0, probe_side))
    for i in range(22):
        rows.append((confirm + pd.Timedelta(seconds=i), 102.5, 10.0, "BUY"))
    if extra_after_now:
        rows.extend(extra_after_now)
    return _ticks(rows)


def test_footprint_filter_allows_absorption_and_tags_tier():
    sig = ar.evaluate_auction_reclaim(
        _today_long_setup(), _prior_up_day(), symbol="TEST",
        prior_sessions=_rising_sessions(),
        ticks_so_far=_long_footprint_ticks(absorb=True),
    )
    assert sig is not None
    assert sig.context["tier"].startswith("footprint_5m")


def test_footprint_filter_rejects_when_probe_not_absorbed():
    sig = ar.evaluate_auction_reclaim(
        _today_long_setup(), _prior_up_day(), symbol="TEST",
        prior_sessions=_rising_sessions(),
        ticks_so_far=_long_footprint_ticks(absorb=False),
    )
    assert sig is None


def test_footprint_ignores_prints_after_now():
    future = [(pd.Timestamp("2024-06-04 09:55:00") + pd.Timedelta(seconds=i), 102.5, 50.0, "SELL")
              for i in range(40)]
    sig = ar.evaluate_auction_reclaim(
        _today_long_setup(), _prior_up_day(), symbol="TEST",
        prior_sessions=_rising_sessions(),
        ticks_so_far=_long_footprint_ticks(absorb=True, extra_after_now=future),
    )
    assert sig is not None
    assert sig.context["tier"].startswith("footprint_5m")


def test_gex_snapshot_replaces_vol_regime_label():
    gex = GexSnapshot(
        symbol="QQQ", as_of="2024-06-04", source="synthetic", spot=450.0,
        net_gex=1.5e9, call_gex=2.0e9, put_gex=-0.5e9, regime="positive_gamma",
        call_wall=455.0, put_wall=440.0, gamma_flip=442.0,
    )
    sig = ar.evaluate_auction_reclaim(
        _today_long_setup(), _prior_up_day(), symbol="TEST",
        prior_sessions=_rising_sessions(), gex_snapshot=gex,
    )
    assert sig is not None
    assert sig.context["vol_regime"] == "positive_gamma"
    assert sig.context["gex_source"] == "synthetic"
    assert sig.context["gex_call_wall"] == 455.0
    assert sig.context["tier"].endswith("_gex")
