"""
Context Engine tests — pure computation, synthetic 1-minute bars, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.microstructure import context


def _bars(n: int, start: str = "2024-06-03 09:30", start_price: float = 100.0,
          trend: float = 0.0, freq: str = "1min", volume: float = 1000.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    closes = start_price + np.arange(n) * trend
    df = pd.DataFrame({
        "open": closes,
        "high": closes + 0.1,
        "low": closes - 0.1,
        "close": closes,
        "volume": volume,
    }, index=idx)
    return df


def test_true_range_and_atr_basic():
    bars = _bars(20, trend=0.05)
    tr = context.true_range(bars)
    a = context.atr(bars, period=5)
    assert len(tr) == len(bars)
    assert (a >= 0).all()
    assert not a.isna().any()


def test_session_vwap_matches_manual_calc_for_flat_volume():
    bars = _bars(5, start_price=100.0, trend=1.0, volume=100.0)
    vwap = context.session_vwap(bars)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    expected_last = tp.mean()  # equal volume every bar -> simple average
    assert vwap.iloc[-1] == pytest.approx(expected_last, rel=1e-6)


def test_vwap_bands_ordering():
    bars = _bars(30, trend=0.2, volume=500.0)
    bands = context.vwap_bands(bars)
    last = bands.iloc[-1]
    assert last["lower_2"] <= last["lower_1"] <= last["vwap"] <= last["upper_1"] <= last["upper_2"]


def test_anchored_vwap_nan_before_anchor():
    bars = _bars(10, trend=0.1)
    anchor = bars.index[4]
    av = context.anchored_vwap(bars, anchor)
    assert av.loc[:bars.index[3]].isna().all()
    assert not av.loc[anchor:].isna().any()
    # Anchored VWAP restarts — value at anchor bar equals that bar's typical price.
    tp_at_anchor = (bars.loc[anchor, "high"] + bars.loc[anchor, "low"] + bars.loc[anchor, "close"]) / 3.0
    assert av.loc[anchor] == pytest.approx(tp_at_anchor)


def test_liquidity_levels_ydh_ydl_from_prior_day():
    today = _bars(10, start="2024-06-04 09:30", start_price=105.0)
    prior = _bars(10, start="2024-06-03 09:30", start_price=100.0, trend=0.5)
    levels = context.compute_liquidity_levels(today, prior_day_bars=prior)
    assert levels.ydh == pytest.approx(prior["high"].max())
    assert levels.ydl == pytest.approx(prior["low"].min())


def test_liquidity_levels_pmh_pml_none_when_not_supplied():
    today = _bars(10)
    levels = context.compute_liquidity_levels(today)
    assert levels.pmh is None
    assert levels.pml is None


def test_liquidity_levels_pmh_pml_from_premarket_bars():
    today = _bars(10, start="2024-06-04 09:30")
    premarket = _bars(5, start="2024-06-04 08:00", start_price=98.0, trend=0.3)
    levels = context.compute_liquidity_levels(today, premarket_bars=premarket)
    assert levels.pmh == pytest.approx(premarket["high"].max())
    assert levels.pml == pytest.approx(premarket["low"].min())


def test_equal_highs_detects_repeated_touches():
    # Bars alternate between a flat "equal high" plateau and lower bars.
    idx = pd.date_range("2024-06-04 09:30", periods=8, freq="1min")
    highs = [100.0, 95.0, 100.05, 94.0, 100.02, 93.0, 100.0, 92.0]
    df = pd.DataFrame({
        "open": highs, "high": highs, "low": [h - 1 for h in highs],
        "close": highs, "volume": 1000.0,
    }, index=idx)
    levels = context.compute_liquidity_levels(df, eq_lookback=8, eq_atr_mult=0.02)
    assert len(levels.eq_highs) >= 1
    assert levels.eq_highs[0] == pytest.approx(100.0, abs=0.1)


def test_round_levels_near_scales_with_price():
    assert context._round_step_for_price(9.0) == 0.5
    assert context._round_step_for_price(40.0) == 1.0
    assert context._round_step_for_price(80.0) == 5.0
    assert context._round_step_for_price(500.0) == 10.0
    levels = context._round_levels_near(101.3, span=2)
    assert 100.0 in levels


def test_volume_profile_poc_at_highest_volume_bin():
    idx = pd.date_range("2024-06-04 09:30", periods=6, freq="1min")
    closes = [100.0, 100.0, 105.0, 105.0, 105.0, 110.0]
    volumes = [100.0, 100.0, 5000.0, 5000.0, 5000.0, 100.0]
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes,
    }, index=idx)
    vp = context.volume_profile(df, n_bins=10)
    assert vp.poc == pytest.approx(105.0, abs=1.0)
    assert vp.val <= vp.poc <= vp.vah


def test_volume_profile_empty_bars():
    vp = context.volume_profile(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    assert vp.poc is None


def test_opening_range_high_low_first_n_minutes():
    bars = _bars(30, start="2024-06-04 09:30", start_price=100.0, trend=0.0)
    bars.loc[bars.index[5], "high"] = 150.0  # spike inside the first 15 minutes
    bars.loc[bars.index[20], "high"] = 999.0  # spike AFTER the opening range — must be excluded
    orange = context.opening_range(bars, minutes=15)
    assert orange.high == 150.0
    assert orange.high != 999.0
    assert orange.start == bars.index[0]
    assert orange.end == bars.index[0] + pd.Timedelta(minutes=15)


def test_opening_range_empty_bars():
    orange = context.opening_range(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    assert orange.high is None


def test_compute_context_orchestrator_returns_all_components():
    today = _bars(30, start="2024-06-04 09:30", start_price=100.0, trend=0.1)
    prior = _bars(20, start="2024-06-03 09:30", start_price=98.0, trend=0.1)
    ctx = context.compute_context(today, prior_day_bars=prior)
    assert ctx.liquidity.ydh is not None
    assert len(ctx.vwap) == len(today)
    assert ctx.volume_profile.poc is not None
    assert ctx.opening_range.high is not None
    assert len(ctx.atr14) == len(today)
