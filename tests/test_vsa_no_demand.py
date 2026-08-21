"""Unit tests for vsa_no_demand — Williams/Coulling path of least resistance."""
from __future__ import annotations

import pandas as pd

from python.backtest import intraday_engine as eng
from python.microstructure.gex import GexSnapshot
from python.microstructure.signals import vsa_no_demand as vsa


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


def _today_no_selling_pressure() -> pd.DataFrame:
    """Drift below prior VAL, two heavy down bars, then a narrow light
    down-bar that holds the close, then a confirm that does not make a new low."""
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    for i in range(20):
        px = 104.0 - i * 0.08
        rows.append((px, px + 0.03, px - 0.03, px, 900.0))
    for i in range(10):
        px = 102.4 - i * 0.04
        rows.append((px, px + 0.06, px - 0.10, px - 0.08, 2800.0))
    rows.extend([
        (102.00, 102.04, 101.88, 101.98, 350.0),
        (101.98, 102.02, 101.90, 101.97, 350.0),
        (101.97, 102.01, 101.89, 101.96, 350.0),
        (101.96, 102.00, 101.88, 101.95, 350.0),
        (101.95, 102.02, 101.87, 101.99, 350.0),
    ])
    rows.extend([
        (101.99, 102.08, 101.90, 102.04, 600.0),
        (102.04, 102.10, 101.96, 102.06, 600.0),
        (102.06, 102.12, 102.00, 102.08, 600.0),
        (102.08, 102.14, 102.02, 102.10, 600.0),
        (102.10, 102.16, 102.04, 102.12, 600.0),
    ])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def _today_no_demand() -> pd.DataFrame:
    """Rally above prior VAH, two heavy up bars, then a narrow light
    up-bar, then a confirm that does not make a new high."""
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    for i in range(20):
        px = 104.0 + i * 0.08
        rows.append((px, px + 0.03, px - 0.03, px, 900.0))
    for i in range(10):
        px = 105.6 + i * 0.04
        rows.append((px, px + 0.10, px - 0.06, px + 0.08, 2800.0))
    rows.extend([
        (106.00, 106.12, 105.98, 106.08, 350.0),
        (106.08, 106.14, 106.02, 106.10, 350.0),
        (106.10, 106.16, 106.04, 106.12, 350.0),
        (106.12, 106.18, 106.06, 106.14, 350.0),
        (106.14, 106.20, 106.08, 106.16, 350.0),
    ])
    rows.extend([
        (106.16, 106.18, 106.02, 106.06, 600.0),
        (106.06, 106.10, 106.00, 106.04, 600.0),
        (106.04, 106.08, 105.98, 106.02, 600.0),
        (106.02, 106.06, 105.96, 106.00, 600.0),
        (106.00, 106.04, 105.94, 105.98, 600.0),
    ])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_fires_long_on_no_selling_pressure():
    sig = vsa.evaluate_vsa_no_demand(_today_no_selling_pressure(), _prior_up_day(), symbol="TEST")
    assert sig is not None
    assert sig.direction == "long"
    assert sig.strategy == "vsa_no_demand"
    assert sig.context["pattern"] == "no_selling_pressure"
    assert sig.stop_price < sig.entry_price


def test_fires_short_on_no_demand():
    sig = vsa.evaluate_vsa_no_demand(_today_no_demand(), _prior_down_day(), symbol="TEST")
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["pattern"] == "no_demand"
    assert sig.stop_price > sig.entry_price


def test_no_signal_when_setup_volume_is_not_lighter():
    bars = _today_no_selling_pressure()
    bars.loc[bars.index[30:35], "volume"] = 4000.0
    assert vsa.evaluate_vsa_no_demand(bars, _prior_up_day()) is None


def test_no_signal_without_prior_day():
    assert vsa.evaluate_vsa_no_demand(_today_no_selling_pressure(), None) is None


def test_no_signal_mid_five_minute_bin():
    assert vsa.evaluate_vsa_no_demand(_today_no_selling_pressure().iloc[:-1], _prior_up_day()) is None


def test_no_lookahead():
    bars = _today_no_selling_pressure()
    prior = _prior_up_day()
    sig = vsa.evaluate_vsa_no_demand(bars, prior)
    later = bars.copy()
    later.loc[bars.index[-1] + pd.Timedelta(minutes=1)] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    sig2 = vsa.evaluate_vsa_no_demand(later.iloc[: len(bars)], prior)
    assert sig is not None and sig2 is not None
    assert sig.entry_price == sig2.entry_price
    assert sig.stop_price == sig2.stop_price


def test_dispatches_through_intraday_engine():
    sig = eng._evaluate_signal(
        "vsa_no_demand",
        _today_no_selling_pressure(),
        "TEST",
        {"spread_atr_max": 0.55, "vol_lookback": 2, "stop_atr_mult": 0.20, "target_r_multiple": 1.5},
        eng.IntradayBacktestConfig(),
        _prior_up_day(),
        None,
    )
    assert sig is not None
    assert sig.strategy == "vsa_no_demand"


def _today_1m_no_selling_pressure() -> pd.DataFrame:
    """Enough 1-minute bars for ATR, then a narrow light down-bar and a
    confirm that does not make a new low — fires on 1m, not on 5m (too few
    closed 5-minute bars)."""
    idx = pd.date_range("2024-06-04 09:30", periods=16, freq="1min")
    rows = []
    for i in range(14):
        px = 102.4 - i * 0.05
        rows.append((px, px + 0.08, px - 0.10, px - 0.07, 2200.0))
    rows.append((101.72, 101.73, 101.66, 101.70, 160.0))
    rows.append((101.70, 101.78, 101.67, 101.75, 700.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_chart_minutes_1_fires_when_5m_cannot():
    bars = _today_1m_no_selling_pressure()
    prior = _prior_up_day()
    assert vsa.evaluate_vsa_no_demand(bars, prior) is None
    sig = vsa.evaluate_vsa_no_demand(bars, prior, chart_minutes=1)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.context["chart_minutes"] == 1


def test_engine_forwards_chart_minutes():
    bars = _today_1m_no_selling_pressure()
    params = {"spread_atr_max": 0.55, "vol_lookback": 2, "stop_atr_mult": 0.20, "target_r_multiple": 1.5}
    assert eng._evaluate_signal(
        "vsa_no_demand", bars, "TEST", params, eng.IntradayBacktestConfig(), _prior_up_day(), None,
    ) is None
    sig = eng._evaluate_signal(
        "vsa_no_demand", bars, "TEST", params,
        eng.IntradayBacktestConfig(chart_minutes=1), _prior_up_day(), None,
    )
    assert sig is not None
    assert sig.context["chart_minutes"] == 1


def _shift_ohlc(bars: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = bars.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] + delta
    return out


def test_default_filters_recorded_in_context():
    sig = vsa.evaluate_vsa_no_demand(_today_no_selling_pressure(), _prior_up_day())
    assert sig is not None
    assert sig.context["require_location"] is True
    assert sig.context["require_confirm"] is True
    assert sig.context["require_volume"] is True


def test_volume_filter_blocks_then_fires_when_off():
    bars = _today_no_selling_pressure()
    bars.loc[bars.index[30:35], "volume"] = 4000.0
    prior = _prior_up_day()
    assert vsa.evaluate_vsa_no_demand(bars, prior) is None
    sig = vsa.evaluate_vsa_no_demand(bars, prior, require_volume=False)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.context["require_volume"] is False


def test_location_filter_blocks_then_fires_when_off():
    """Shift the long setup into the prior-session value area so VAH/VAL
    rejects it; bar type + volume + confirm are unchanged."""
    bars = _shift_ohlc(_today_no_selling_pressure(), 6.0)
    prior = _prior_up_day()
    assert vsa.evaluate_vsa_no_demand(bars, prior) is None
    sig = vsa.evaluate_vsa_no_demand(bars, prior, require_location=False)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.context["require_location"] is False


def test_location_off_does_not_block_when_profile_missing():
    prior = _prior_up_day()
    prior["volume"] = 0.0
    bars = _today_no_selling_pressure()
    assert vsa.evaluate_vsa_no_demand(bars, prior) is None
    sig = vsa.evaluate_vsa_no_demand(bars, prior, require_location=False)
    assert sig is not None
    assert sig.context["prior_val"] is None
    assert sig.context["prior_vah"] is None


def _today_confirm_breaks_but_last_is_setup() -> pd.DataFrame:
    """Same no-demand shape as `_today_no_demand`, but the last 5-minute
    bar is itself a valid narrow light up-bar that makes a new high —
    so confirm-ON rejects, confirm-OFF treats the last bar as setup."""
    idx = pd.date_range("2024-06-04 09:30", periods=40, freq="1min")
    rows = []
    for i in range(20):
        px = 104.0 + i * 0.08
        rows.append((px, px + 0.03, px - 0.03, px, 900.0))
    for i in range(10):
        px = 105.6 + i * 0.04
        rows.append((px, px + 0.10, px - 0.06, px + 0.08, 2800.0))
    rows.extend([
        (106.00, 106.12, 105.98, 106.08, 350.0),
        (106.08, 106.14, 106.02, 106.10, 350.0),
        (106.10, 106.16, 106.04, 106.12, 350.0),
        (106.12, 106.18, 106.06, 106.14, 350.0),
        (106.14, 106.20, 106.08, 106.16, 350.0),
    ])
    rows.extend([
        (106.16, 106.22, 106.12, 106.18, 200.0),
        (106.18, 106.24, 106.14, 106.20, 200.0),
        (106.20, 106.26, 106.16, 106.22, 200.0),
        (106.22, 106.28, 106.18, 106.24, 200.0),
        (106.24, 106.30, 106.20, 106.26, 200.0),
    ])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_confirm_filter_blocks_then_fires_when_off():
    bars = _today_confirm_breaks_but_last_is_setup()
    prior = _prior_down_day()
    assert vsa.evaluate_vsa_no_demand(bars, prior) is None
    sig = vsa.evaluate_vsa_no_demand(bars, prior, require_confirm=False)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["require_confirm"] is False
    assert sig.entry_price == float(bars["close"].iloc[-1])


def test_engine_forwards_signal_filter_overrides():
    bars = _today_no_selling_pressure()
    bars.loc[bars.index[30:35], "volume"] = 4000.0
    params = {"spread_atr_max": 0.55, "vol_lookback": 2, "stop_atr_mult": 0.20, "target_r_multiple": 1.5}
    assert eng._evaluate_signal(
        "vsa_no_demand", bars, "TEST", params, eng.IntradayBacktestConfig(), _prior_up_day(), None,
    ) is None
    sig = eng._evaluate_signal(
        "vsa_no_demand", bars, "TEST", params,
        eng.IntradayBacktestConfig(
            signal_filter_overrides={"require_volume": False, "chart_minutes": 99},
        ),
        _prior_up_day(), None,
    )
    assert sig is not None
    assert sig.context["require_volume"] is False
    assert sig.context["chart_minutes"] == 5


def test_gex_snapshot_labels_regime_only():
    gex = GexSnapshot(
        symbol="QQQ", as_of="2024-06-04", source="synthetic", spot=450.0,
        net_gex=1.5e9, call_gex=2.0e9, put_gex=-0.5e9, regime="positive_gamma",
        call_wall=455.0, put_wall=440.0, gamma_flip=442.0,
    )
    sig = vsa.evaluate_vsa_no_demand(
        _today_no_selling_pressure(), _prior_up_day(), gex_snapshot=gex,
    )
    assert sig is not None
    assert sig.context["vol_regime"] == "positive_gamma"
    assert sig.context["gex_source"] == "synthetic"
    assert sig.context["tier"].endswith("_gex")
