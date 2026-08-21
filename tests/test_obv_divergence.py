"""Unit tests for obv_divergence — Granville B-2 / S-2 on 5-minute OBV."""
from __future__ import annotations

import pandas as pd

from python.backtest import intraday_engine as eng
from python.microstructure.gex import GexSnapshot
from python.microstructure.signals import obv_divergence as obv


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


def _minute_block(start_px: float, end_px: float, vol: float, n: int = 5):
    step = (end_px - start_px) / (n - 1)
    rows = []
    for i in range(n):
        px = start_px + i * step
        hi = max(start_px, end_px, px) + 0.04
        lo = min(start_px, end_px, px) - 0.04
        rows.append((px, hi, lo, px, vol))
    return rows


def _today_price_high_obv_lags() -> pd.DataFrame:
    """Heavy early rally (OBV peaks), heavy selloff (OBV drops), then a
    new price high on light volume so OBV cannot reclaim its high."""
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    rows.extend(_minute_block(104.0, 105.0, 2500.0))
    rows.extend(_minute_block(105.0, 106.0, 2500.0))
    rows.extend(_minute_block(106.0, 107.0, 2500.0))
    rows.extend(_minute_block(107.0, 105.2, 3500.0))
    rows.extend(_minute_block(105.2, 105.6, 400.0))
    rows.extend(_minute_block(105.6, 106.2, 400.0))
    rows.extend(_minute_block(106.2, 106.8, 400.0))
    rows.extend(_minute_block(106.8, 107.4, 400.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def _today_price_low_obv_lags() -> pd.DataFrame:
    """Heavy early selloff (OBV troughs), heavy bounce (OBV rises), then a
    new price low on light volume so OBV cannot reclaim its low."""
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    rows.extend(_minute_block(106.0, 105.0, 2500.0))
    rows.extend(_minute_block(105.0, 104.0, 2500.0))
    rows.extend(_minute_block(104.0, 103.0, 2500.0))
    rows.extend(_minute_block(103.0, 104.8, 3500.0))
    rows.extend(_minute_block(104.8, 104.4, 400.0))
    rows.extend(_minute_block(104.4, 103.8, 400.0))
    rows.extend(_minute_block(103.8, 103.2, 400.0))
    rows.extend(_minute_block(103.2, 102.6, 400.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_fires_short_when_price_high_outpaces_obv():
    sig = obv.evaluate_obv_divergence(_today_price_high_obv_lags(), _prior_down_day(), symbol="TEST")
    assert sig is not None
    assert sig.direction == "short"
    assert sig.strategy == "obv_divergence"
    assert sig.context["pattern"] == "obv_s2_price_leads"
    assert sig.stop_price > sig.entry_price


def test_fires_long_when_price_low_outpaces_obv():
    sig = obv.evaluate_obv_divergence(_today_price_low_obv_lags(), _prior_up_day(), symbol="TEST")
    assert sig is not None
    assert sig.direction == "long"
    assert sig.context["pattern"] == "obv_b2_price_leads"
    assert sig.stop_price < sig.entry_price


def test_no_signal_when_obv_confirms_the_high():
    bars = _today_price_high_obv_lags()
    bars.loc[bars.index[-5:], "volume"] = 8000.0
    assert obv.evaluate_obv_divergence(bars, _prior_down_day()) is None


def test_no_signal_without_prior_day():
    assert obv.evaluate_obv_divergence(_today_price_high_obv_lags(), None) is None


def test_no_signal_mid_five_minute_bin():
    assert obv.evaluate_obv_divergence(_today_price_high_obv_lags().iloc[:-1], _prior_down_day()) is None


def test_no_lookahead():
    bars = _today_price_high_obv_lags()
    prior = _prior_down_day()
    sig = obv.evaluate_obv_divergence(bars, prior)
    later = bars.copy()
    later.loc[bars.index[-1] + pd.Timedelta(minutes=1)] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    sig2 = obv.evaluate_obv_divergence(later.iloc[: len(bars)], prior)
    assert sig is not None and sig2 is not None
    assert sig.entry_price == sig2.entry_price
    assert sig.stop_price == sig2.stop_price


def test_dispatches_through_intraday_engine():
    sig = eng._evaluate_signal(
        "obv_divergence",
        _today_price_high_obv_lags(),
        "TEST",
        {"lookback_bars": 8, "obv_lag_frac": 0.25, "stop_atr_mult": 0.20, "target_r_multiple": 1.5},
        eng.IntradayBacktestConfig(),
        _prior_down_day(),
        None,
    )
    assert sig is not None
    assert sig.strategy == "obv_divergence"


def _today_1m_price_high_obv_lags() -> pd.DataFrame:
    """16 one-minute bars: heavy rally, heavy selloff, light new high.
    Enough closed 1-minute bars for lookback=8; not enough closed 5-minute
    bars for the default chart."""
    idx = pd.date_range("2024-06-04 09:30", periods=16, freq="1min")
    rows = []
    rows.extend(_minute_block(104.0, 106.5, 3000.0, n=5))
    rows.extend(_minute_block(106.5, 104.8, 4000.0, n=5))
    rows.extend(_minute_block(104.8, 107.2, 300.0, n=6))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_chart_minutes_1_fires_when_5m_cannot():
    bars = _today_1m_price_high_obv_lags()
    prior = _prior_down_day()
    assert obv.evaluate_obv_divergence(bars, prior) is None
    sig = obv.evaluate_obv_divergence(bars, prior, chart_minutes=1)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["chart_minutes"] == 1


def test_engine_forwards_chart_minutes():
    bars = _today_1m_price_high_obv_lags()
    params = {"lookback_bars": 8, "obv_lag_frac": 0.25, "stop_atr_mult": 0.20, "target_r_multiple": 1.5}
    assert eng._evaluate_signal(
        "obv_divergence", bars, "TEST", params, eng.IntradayBacktestConfig(), _prior_down_day(), None,
    ) is None
    sig = eng._evaluate_signal(
        "obv_divergence", bars, "TEST", params,
        eng.IntradayBacktestConfig(chart_minutes=1), _prior_down_day(), None,
    )
    assert sig is not None
    assert sig.context["chart_minutes"] == 1


def _shift_ohlc(bars: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = bars.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] + delta
    return out


def test_default_filters_recorded_in_context():
    sig = obv.evaluate_obv_divergence(_today_price_high_obv_lags(), _prior_down_day())
    assert sig is not None
    assert sig.context["require_location"] is True
    assert sig.context["require_obv_lag"] is True


def test_obv_lag_filter_blocks_then_fires_when_off():
    bars = _today_price_high_obv_lags()
    bars.loc[bars.index[-5:], "volume"] = 8000.0
    prior = _prior_down_day()
    assert obv.evaluate_obv_divergence(bars, prior) is None
    sig = obv.evaluate_obv_divergence(bars, prior, require_obv_lag=False)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["require_obv_lag"] is False


def test_location_filter_blocks_then_fires_when_off():
    bars = _shift_ohlc(_today_price_high_obv_lags(), -6.0)
    prior = _prior_down_day()
    assert obv.evaluate_obv_divergence(bars, prior) is None
    sig = obv.evaluate_obv_divergence(bars, prior, require_location=False)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["require_location"] is False


def test_location_off_does_not_block_when_profile_missing():
    prior = _prior_down_day()
    prior["volume"] = 0.0
    bars = _today_price_high_obv_lags()
    assert obv.evaluate_obv_divergence(bars, prior) is None
    sig = obv.evaluate_obv_divergence(bars, prior, require_location=False)
    assert sig is not None
    assert sig.context["prior_val"] is None
    assert sig.context["prior_vah"] is None


def test_obv_range_zero_does_not_block_when_lag_off():
    """Flat OBV (obv_range <= 0) is a lag-filter reject only."""
    bars = _today_price_high_obv_lags()
    bars["volume"] = 0.0
    prior = _prior_down_day()
    assert obv.evaluate_obv_divergence(bars, prior) is None
    sig = obv.evaluate_obv_divergence(bars, prior, require_obv_lag=False)
    assert sig is not None
    assert sig.context["require_obv_lag"] is False
    assert sig.context["obv_lag_frac_used"] is None


def test_engine_forwards_signal_filter_overrides():
    bars = _today_price_high_obv_lags()
    bars.loc[bars.index[-5:], "volume"] = 8000.0
    params = {"lookback_bars": 8, "obv_lag_frac": 0.25, "stop_atr_mult": 0.20, "target_r_multiple": 1.5}
    assert eng._evaluate_signal(
        "obv_divergence", bars, "TEST", params, eng.IntradayBacktestConfig(), _prior_down_day(), None,
    ) is None
    sig = eng._evaluate_signal(
        "obv_divergence", bars, "TEST", params,
        eng.IntradayBacktestConfig(
            signal_filter_overrides={"require_obv_lag": False, "chart_minutes": 99},
        ),
        _prior_down_day(), None,
    )
    assert sig is not None
    assert sig.context["require_obv_lag"] is False
    assert sig.context["chart_minutes"] == 5


def test_gex_snapshot_labels_regime_only():
    gex = GexSnapshot(
        symbol="QQQ", as_of="2024-06-04", source="synthetic", spot=450.0,
        net_gex=1.5e9, call_gex=2.0e9, put_gex=-0.5e9, regime="positive_gamma",
        call_wall=455.0, put_wall=440.0, gamma_flip=442.0,
    )
    sig = obv.evaluate_obv_divergence(
        _today_price_high_obv_lags(), _prior_down_day(), gex_snapshot=gex,
    )
    assert sig is not None
    assert sig.context["vol_regime"] == "positive_gamma"
    assert sig.context["tier"].endswith("_gex")
