"""Unit tests for vsa_effort — Wyckoff/VSA effort-without-result."""
from __future__ import annotations

import pandas as pd
import pytest

from python.backtest import intraday_engine as eng
from python.microstructure.gex import GexSnapshot
from python.microstructure.signals import vsa_effort as vsa


def _prior_up_day() -> pd.DataFrame:
    idx = pd.date_range("2024-06-03 09:30", periods=60, freq="1min")
    closes = [100.0 + 10.0 * (i / 59.0) for i in range(60)]
    mid = 105.0
    volume = [400.0 if c < mid else 4000.0 for c in closes]
    return pd.DataFrame({
        "open": closes, "high": [c + 0.05 for c in closes],
        "low": [c - 0.05 for c in closes], "close": closes, "volume": volume,
    }, index=idx)


def _prior_down_day() -> pd.DataFrame:
    idx = pd.date_range("2024-06-03 09:30", periods=60, freq="1min")
    closes = [110.0 - 10.0 * (i / 59.0) for i in range(60)]
    mid = 105.0
    volume = [400.0 if c > mid else 4000.0 for c in closes]
    return pd.DataFrame({
        "open": closes, "high": [c + 0.05 for c in closes],
        "low": [c - 0.05 for c in closes], "close": closes, "volume": volume,
    }, index=idx)


def _today_spring() -> pd.DataFrame:
    """Six quiet 5m bins drift down, effort dumps then holds, light test."""
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    for i in range(30):
        px = 108.0 - i * 0.12
        rows.append((px, px + 0.04, px - 0.04, px, 800.0))
    # 10:00-10:04 effort: new low 101.0, close held in the upper half.
    rows.extend([
        (104.4, 104.6, 103.6, 103.8, 3500.0),
        (103.8, 104.0, 102.4, 102.6, 3500.0),
        (102.6, 102.8, 101.2, 101.5, 3500.0),
        (101.5, 102.2, 101.0, 101.8, 3500.0),
        (101.8, 104.2, 101.1, 103.6, 3500.0),
    ])
    # 10:05-10:09 test: higher low, light volume, buyers finish the hour.
    rows.extend([
        (102.5, 102.7, 102.2, 102.4, 400.0),
        (102.4, 102.6, 102.1, 102.3, 400.0),
        (102.3, 102.5, 101.8, 102.2, 400.0),
        (102.2, 102.6, 101.9, 102.4, 400.0),
        (102.4, 103.0, 101.85, 102.8, 400.0),
    ])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def _today_upthrust() -> pd.DataFrame:
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    for i in range(30):
        px = 102.0 + i * 0.12
        rows.append((px, px + 0.04, px - 0.04, px, 800.0))
    rows.extend([
        (105.6, 106.8, 105.5, 106.6, 3500.0),
        (106.6, 107.8, 106.5, 107.6, 3500.0),
        (107.6, 108.8, 107.5, 108.6, 3500.0),
        (108.6, 109.2, 108.4, 108.8, 3500.0),
        (108.8, 109.2, 105.8, 106.4, 3500.0),
    ])
    rows.extend([
        (107.5, 107.8, 107.2, 107.3, 400.0),
        (107.3, 107.6, 107.1, 107.2, 400.0),
        (107.2, 107.5, 107.0, 107.1, 400.0),
        (107.1, 107.4, 106.9, 107.0, 400.0),
        (107.0, 108.2, 106.8, 106.9, 400.0),
    ])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_fires_long_on_stopping_volume_and_no_supply_test():
    sig = vsa.evaluate_vsa_effort(_today_spring(), _prior_up_day(), symbol="TEST")
    assert sig is not None
    assert sig.direction == "long"
    assert sig.strategy == "vsa_effort"
    assert sig.context["pattern"] == "spring_no_supply"
    assert sig.stop_price < sig.entry_price
    assert sig.context["effort_rel_volume"] >= 2.0
    assert sig.context["test_rel_volume"] <= 0.85


def test_fires_short_on_upthrust_and_no_demand_test():
    sig = vsa.evaluate_vsa_effort(_today_upthrust(), _prior_down_day(), symbol="TEST")
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["pattern"] == "upthrust_no_demand"
    assert sig.stop_price > sig.entry_price


def test_no_signal_when_test_volume_is_still_heavy():
    bars = _today_spring()
    bars.loc[bars.index[-5:], "volume"] = 4000.0
    assert vsa.evaluate_vsa_effort(bars, _prior_up_day()) is None


def test_no_signal_without_prior_day():
    assert vsa.evaluate_vsa_effort(_today_spring(), None) is None


def test_no_signal_mid_five_minute_bin():
    assert vsa.evaluate_vsa_effort(_today_spring().iloc[:-1], _prior_up_day()) is None


def test_no_lookahead():
    bars = _today_spring()
    prior = _prior_up_day()
    sig = vsa.evaluate_vsa_effort(bars, prior)
    later = bars.copy()
    later.loc[bars.index[-1] + pd.Timedelta(minutes=1)] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    sig2 = vsa.evaluate_vsa_effort(later.iloc[: len(bars)], prior)
    assert sig is not None and sig2 is not None
    assert sig.entry_price == sig2.entry_price
    assert sig.stop_price == sig2.stop_price


def test_dispatches_through_intraday_engine():
    sig = eng._evaluate_signal(
        "vsa_effort",
        _today_spring(),
        "TEST",
        {"effort_vol_mult": 2.0, "test_vol_mult": 0.85, "stop_atr_mult": 0.20, "target_r_multiple": 1.5},
        eng.IntradayBacktestConfig(),
        _prior_up_day(),
        None,
    )
    assert sig is not None
    assert sig.strategy == "vsa_effort"


def test_gex_snapshot_labels_regime_only():
    gex = GexSnapshot(
        symbol="QQQ", as_of="2024-06-04", source="synthetic", spot=450.0,
        net_gex=1.5e9, call_gex=2.0e9, put_gex=-0.5e9, regime="positive_gamma",
        call_wall=455.0, put_wall=440.0, gamma_flip=442.0,
    )
    sig = vsa.evaluate_vsa_effort(_today_spring(), _prior_up_day(), gex_snapshot=gex)
    assert sig is not None
    assert sig.context["vol_regime"] == "positive_gamma"
    assert sig.context["gex_source"] == "synthetic"
    assert sig.context["tier"].endswith("_gex")
