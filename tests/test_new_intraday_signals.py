"""
Tests for the NEW intraday signal hypotheses (backtests/reports/
new_signals_report.md): orb_vwap_regime, vwap_band_fade, vp_breakout; plus
absorption_breakout (backtests/reports/
absorption_breakout_investigation_report.md, added 2026-08-14). Same style
as tests/test_intraday_signals.py — synthetic bar sequences covering
trigger, no-trigger, and no-lookahead behavior for each signal, plus
engine-level plumbing checks for the regime gate."""
from __future__ import annotations

import pandas as pd
import pytest

from python.backtest import intraday_engine as eng
from python.microstructure import context
from python.microstructure.context import LiquidityLevels
from python.microstructure.signals import absorption_breakout as absb_mod
from python.microstructure.signals import orb_vwap_regime as ovr_mod
from python.microstructure.signals import vp_breakout as vpb_mod
from python.microstructure.signals import vwap_band_fade as vbf_mod


def _flat_bars(n: int, price: float = 100.0, start: str = "2024-06-04 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price, "volume": 1000.0,
    }, index=idx)


# ── orb_vwap_regime ──────────────────────────────────────────────────────────

def _orb_bars() -> pd.DataFrame:
    bars = _flat_bars(40, price=100.0, start="2024-06-04 09:30")
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0  # fresh breakout above OR high (101.0)
    return bars


def test_orb_vwap_regime_blocks_signal_on_non_trending_day():
    bars = _orb_bars()
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    sig = ovr_mod.evaluate_orb_vwap_regime(
        bars.iloc[:22], orange, vwap.iloc[:22], is_trending_day=False, vwap_side_filter=True,
    )
    assert sig is None


def test_orb_vwap_regime_passes_through_underlying_orb_vwap_signal_on_trending_day():
    bars = _orb_bars()
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    sig = ovr_mod.evaluate_orb_vwap_regime(
        bars.iloc[:22], orange, vwap.iloc[:22], is_trending_day=True, vwap_side_filter=True,
    )
    assert sig is not None
    assert sig.direction == "long"
    assert sig.strategy == "orb_vwap_regime"
    assert sig.context["regime_gated"] is True
    assert sig.context["breakout_dir"] == "long"  # underlying orb_vwap context preserved


def test_orb_vwap_regime_no_entry_inside_opening_range_regardless_of_regime():
    bars = _orb_bars()
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)
    sig = ovr_mod.evaluate_orb_vwap_regime(
        bars.iloc[:10], orange, vwap.iloc[:10], is_trending_day=True, vwap_side_filter=False,
    )
    assert sig is None  # still inside the 15-minute OR window


def test_daily_trending_flags_no_lookahead_and_defaults_false_before_warmup():
    days = pd.bdate_range("2024-01-02", periods=30)
    frames = []
    price = 100.0
    for d in days:
        idx = pd.date_range(f"{d.date()} 09:30", periods=5, freq="1min")
        price *= 1.01  # steady uptrend -> Bull label once labelable
        frames.append(pd.DataFrame(
            {"open": price, "high": price, "low": price, "close": price, "volume": 1000.0}, index=idx,
        ))
    bars = pd.concat(frames)

    flags = eng._daily_trending_flags(bars)
    session_dates = sorted(set(bars.index.normalize()))
    # First _REGIME_WINDOW+1 days have no trailing label yet -> not present
    # (caller defaults them to False), rather than a guessed True/False.
    for d in session_dates[: eng._REGIME_WINDOW + 1]:
        assert d not in flags
    # Once labelable, a persistent uptrend must read as trending (Bull).
    assert flags[session_dates[-1]] is True

    # No lookahead: flags for early/mid dates must be identical whether or
    # not later bars in `bars` are mutated into something wild.
    mutated = bars.copy()
    last_idx = mutated.index[-1]
    mutated.loc[last_idx, ["open", "high", "low", "close"]] = [1.0, 1.0, 1.0, 1.0]
    flags_mutated = eng._daily_trending_flags(mutated)
    mid_date = session_dates[25]
    assert flags[mid_date] == flags_mutated[mid_date]


def test_daily_trending_flags_flat_prices_never_trending():
    days = pd.bdate_range("2024-01-02", periods=25)
    frames = []
    for d in days:
        idx = pd.date_range(f"{d.date()} 09:30", periods=5, freq="1min")
        frames.append(pd.DataFrame(
            {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1000.0}, index=idx,
        ))
    bars = pd.concat(frames)
    flags = eng._daily_trending_flags(bars)
    assert flags and all(v is False for v in flags.values())


# ── vwap_band_fade ───────────────────────────────────────────────────────────

def _band_fade_bars(extension_high: float = 108.0, hold_price: float = 104.0) -> pd.DataFrame:
    bars = _flat_bars(20, price=100.0)
    bars.loc[bars.index[15], ["open", "high", "low", "close", "volume"]] = \
        [100.0, extension_high, 100.0, extension_high - 0.5, 5000.0]
    for i in (16, 17, 18):
        bars.loc[bars.index[i], ["open", "high", "low", "close", "volume"]] = \
            [hold_price, hold_price + 0.2, hold_price - 0.2, hold_price, 1000.0]
    return bars


def test_vwap_band_fade_fires_short_after_upper_extension_stalls():
    bars = _band_fade_bars()
    bands = context.vwap_bands(bars)
    atr_series = context.atr(bars, period=14)

    sig_before = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:18], bands.iloc[:18], atr_series.iloc[:18],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )
    assert sig_before is None  # only 2 bars of stall so far, not 3

    sig = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:19], bands.iloc[:19], atr_series.iloc[:19],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )
    assert sig is not None
    assert sig.strategy == "vwap_band_fade"
    assert sig.direction == "short"
    assert sig.stop_price > 108.0  # beyond the extension extreme
    assert sig.target_price == pytest.approx(sig.context["vwap"])

    sig_after = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:20], bands.iloc[:20], atr_series.iloc[:20],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )
    assert sig_after is None  # does not refire once the extreme rolls out of the window


def test_vwap_band_fade_fires_long_after_lower_extension_stalls():
    bars = _flat_bars(20, price=100.0)
    bars.loc[bars.index[15], ["open", "high", "low", "close", "volume"]] = \
        [100.0, 100.0, 92.0, 92.5, 5000.0]
    for i in (16, 17, 18):
        bars.loc[bars.index[i], ["open", "high", "low", "close", "volume"]] = \
            [96.0, 96.2, 95.8, 96.0, 1000.0]
    bands = context.vwap_bands(bars)
    atr_series = context.atr(bars, period=14)

    sig = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:19], bands.iloc[:19], atr_series.iloc[:19],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )
    assert sig is not None
    assert sig.direction == "long"
    assert sig.stop_price < 92.0


def test_vwap_band_fade_no_signal_when_new_extreme_breaks_the_stall():
    bars = _band_fade_bars()
    # Bar 17 makes a NEW high beyond bar 15's extension -> no genuine stall.
    bars.loc[bars.index[17], ["open", "high", "low", "close", "volume"]] = [104.0, 110.0, 104.0, 109.0, 1000.0]
    bands = context.vwap_bands(bars)
    atr_series = context.atr(bars, period=14)

    sig = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:19], bands.iloc[:19], atr_series.iloc[:19],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )
    assert sig is None


def test_vwap_band_fade_no_signal_when_band_not_pierced():
    # Baseline bars alternate +/-0.3 around 100 (nonzero pre-existing VWAP
    # sigma), then a modest new local high (100.6) that stays within
    # 2-sigma of VWAP — a real local extreme, but not an "extension"
    # by this signal's definition.
    idx = pd.date_range("2024-06-04 09:30", periods=20, freq="1min")
    rows = []
    for i in range(20):
        base = 100.0 + (0.3 if i % 2 == 0 else -0.3)
        rows.append({"open": base, "high": base + 0.05, "low": base - 0.05, "close": base, "volume": 1000.0})
    bars = pd.DataFrame(rows, index=idx)
    bars.loc[bars.index[15], ["open", "high", "low", "close", "volume"]] = [100.3, 100.6, 100.0, 100.5, 3000.0]
    for i in (16, 17, 18):
        bars.loc[bars.index[i], ["open", "high", "low", "close", "volume"]] = [100.4, 100.45, 100.35, 100.4, 1000.0]

    bands = context.vwap_bands(bars)
    atr_series = context.atr(bars, period=14)
    sig = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:19], bands.iloc[:19], atr_series.iloc[:19],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )
    assert sig is None


def test_vwap_band_fade_no_lookahead():
    bars = _band_fade_bars()
    bands = context.vwap_bands(bars)
    atr_series = context.atr(bars, period=14)
    sig_at_18 = vbf_mod.evaluate_vwap_band_fade(
        bars.iloc[:19], bands.iloc[:19], atr_series.iloc[:19],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )

    bars2 = bars.copy()
    idx20 = bars2.index[19]
    bars2.loc[idx20, ["open", "high", "low", "close", "volume"]] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    bands2 = context.vwap_bands(bars2)
    atr_series2 = context.atr(bars2, period=14)
    sig_at_18_again = vbf_mod.evaluate_vwap_band_fade(
        bars2.iloc[:19], bands2.iloc[:19], atr_series2.iloc[:19],
        band_sigma_mult=2.0, stall_bars=3, stop_atr_mult=0.5,
    )

    assert sig_at_18 is not None and sig_at_18_again is not None
    assert sig_at_18.entry_price == sig_at_18_again.entry_price
    assert sig_at_18.stop_price == sig_at_18_again.stop_price


# ── vp_breakout ──────────────────────────────────────────────────────────────

_VP_LEVELS = LiquidityLevels(ydh=120.0, ydl=80.0, round_levels=[105.0, 110.0, 115.0, 90.0, 85.0])


def _vp_bars(breakout_price: float = 106.0, breakout_volume: float = 20000.0, n_base: int = 35) -> pd.DataFrame:
    base = _flat_bars(n_base, price=100.0)
    extra = _flat_bars(3, price=breakout_price, start=base.index[-1] + pd.Timedelta(minutes=1))
    extra.loc[extra.index[0], "volume"] = breakout_volume
    return pd.concat([base, extra])


def test_vp_breakout_fires_long_on_confirmed_high_volume_breakout():
    bars = _vp_bars()
    atr_series = context.atr(bars, period=14)

    sig_before = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:36], _VP_LEVELS, atr_series.iloc[:36], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )
    assert sig_before is None  # only 1 bar held outside the value area so far

    sig = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:37], _VP_LEVELS, atr_series.iloc[:37], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )
    assert sig is not None
    assert sig.strategy == "vp_breakout"
    assert sig.direction == "long"
    assert sig.stop_price < sig.entry_price
    assert sig.target_price == 110.0  # nearest round level above 106, via context.target_resistance_levels

    sig_after = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:38], _VP_LEVELS, atr_series.iloc[:38], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )
    assert sig_after is None  # does not refire on the next bar


def test_vp_breakout_fires_short_on_confirmed_breakdown():
    bars = _vp_bars(breakout_price=94.0, breakout_volume=20000.0)
    atr_series = context.atr(bars, period=14)

    sig = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:37], _VP_LEVELS, atr_series.iloc[:37], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )
    assert sig is not None
    assert sig.direction == "short"
    assert sig.stop_price > sig.entry_price
    assert sig.target_price == 90.0  # nearest round level below 94


def test_vp_breakout_no_signal_without_volume_spike():
    bars = _vp_bars(breakout_volume=1000.0)  # no spike, normal volume throughout
    atr_series = context.atr(bars, period=14)
    sig = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:37], _VP_LEVELS, atr_series.iloc[:37], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )
    assert sig is None


def test_vp_breakout_no_signal_when_price_falls_back_inside_value_area():
    bars = _vp_bars()
    # Second confirm bar dips back inside the value area -> no genuine hold.
    bars.loc[bars.index[36], ["open", "high", "low", "close"]] = [100.0, 100.1, 99.5, 99.8]
    atr_series = context.atr(bars, period=14)
    sig = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:37], _VP_LEVELS, atr_series.iloc[:37], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )
    assert sig is None


def test_vp_breakout_no_lookahead():
    bars = _vp_bars()
    atr_series = context.atr(bars, period=14)
    sig_at_36 = vpb_mod.evaluate_vp_breakout(
        bars.iloc[:37], _VP_LEVELS, atr_series.iloc[:37], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )

    bars2 = bars.copy()
    bars2.loc[bars2.index[-1], ["open", "high", "low", "close", "volume"]] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    atr_series2 = context.atr(bars2, period=14)
    sig_at_36_again = vpb_mod.evaluate_vp_breakout(
        bars2.iloc[:37], _VP_LEVELS, atr_series2.iloc[:37], vol_mult=2.0, confirm_bars=2, stop_atr_mult=0.5,
    )

    assert sig_at_36 is not None and sig_at_36_again is not None
    assert sig_at_36.entry_price == sig_at_36_again.entry_price
    assert sig_at_36.stop_price == sig_at_36_again.stop_price


# ── absorption_breakout (l2_absorption's continuation variant — see that
#    module's docstring and backtests/reports/
#    absorption_breakout_investigation_report.md) ────────────────────────────

def _breakout_bars(n: int = 25, price: float = 100.0, band: float = 0.2) -> pd.DataFrame:
    idx = pd.date_range("2024-06-04 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "open": price, "high": price + band, "low": price - band, "close": price, "volume": 1000.0,
    }, index=idx)


def test_absorption_breakout_fires_long_on_confirmed_breakout_with_volume_spike():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    # "now" bar CLOSES beyond level_high (100.2, the prior 20 bars' rolling
    # high) on 10x volume -> continuation long, the OPPOSITE polarity of
    # l2_absorption's fade at the same level.
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]

    sig = absb_mod.evaluate_absorption_breakout(bars)
    assert sig is not None
    assert sig.strategy == "absorption_breakout"
    assert sig.direction == "long"
    assert sig.stop_price < sig.entry_price
    assert sig.stop_price < 100.2  # stop sits back INSIDE the broken level
    assert sig.context["tier"] == "bar_only_proxy_no_l2_confirmation"


def test_absorption_breakout_fires_short_on_confirmed_breakdown_with_volume_spike():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.0, 99.4, 99.5, 10000.0]

    sig = absb_mod.evaluate_absorption_breakout(bars)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.stop_price > sig.entry_price
    assert sig.stop_price > 99.8


def test_absorption_breakout_no_signal_without_volume_spike():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    # Same breakout shape as the long case, but NORMAL volume.
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 1000.0]

    assert absb_mod.evaluate_absorption_breakout(bars) is None


def test_absorption_breakout_no_signal_when_level_holds_l2_absorption_polarity():
    """The literal l2_absorption trigger shape (touch + close back on the
    DEFENDED side) must NOT fire this signal — the two are mutually
    exclusive by construction on the same bar."""
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.1, 99.8, 100.05, 10000.0]

    assert absb_mod.evaluate_absorption_breakout(bars) is None


def test_absorption_breakout_no_signal_with_too_few_bars():
    bars = _breakout_bars(10, price=100.0, band=0.2)
    assert absb_mod.evaluate_absorption_breakout(bars) is None


def test_absorption_breakout_clearance_filter_requires_min_atr_beyond_level():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    # Closes JUST beyond level_high (100.2 -> 100.21) on high volume: fires
    # at the default breakout_atr_mult=0.0 (literal close-through)...
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.25, 100.0, 100.21, 10000.0]
    assert absb_mod.evaluate_absorption_breakout(bars) is not None
    # ...but not once a minimum-clearance filter is switched on.
    assert absb_mod.evaluate_absorption_breakout(bars, breakout_atr_mult=1.0) is None


def test_absorption_breakout_r_multiple_target_is_off_by_default():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]

    default = absb_mod.evaluate_absorption_breakout(bars)
    assert default.target_price is None
    assert default.context["target_r_multiple"] is None

    targeted = absb_mod.evaluate_absorption_breakout(bars, target_r_multiple=2.0)
    risk = targeted.entry_price - targeted.stop_price
    assert risk > 0
    assert targeted.target_price == pytest.approx(targeted.entry_price + 2.0 * risk)


def test_absorption_breakout_no_lookahead():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]
    sig_at_24 = absb_mod.evaluate_absorption_breakout(bars.iloc[:25])

    bars2 = bars.copy()
    idx25 = bars.index[24] + pd.Timedelta(minutes=1)
    bars2.loc[idx25] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    sig_at_24_again = absb_mod.evaluate_absorption_breakout(bars2.iloc[:25])

    assert sig_at_24 is not None and sig_at_24_again is not None
    assert sig_at_24.entry_price == sig_at_24_again.entry_price
    assert sig_at_24.stop_price == sig_at_24_again.stop_price


def test_absorption_breakout_dispatches_through_intraday_engine():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]
    sig = eng._evaluate_signal(
        "absorption_breakout", bars, "TEST",
        {"volume_mult": 3.0, "breakout_atr_mult": 0.0, "stop_atr_mult": 0.5, "target_r_multiple": None},
        eng.IntradayBacktestConfig(chart_minutes=1), None, None,
    )
    assert sig is not None
    assert sig.strategy == "absorption_breakout"
    assert sig.symbol == "TEST"


# ── micro_stop_cents (round 2, 2026-08-14 — backtests/reports/
#    absorption_breakout_investigation_report.md's dated addendum): fixed
#    cents-past-the-level stop, ALTERNATIVE to (not stacked with)
#    stop_atr_mult ────────────────────────────────────────────────────────

def test_absorption_breakout_micro_stop_cents_overrides_atr_stop_for_long():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]

    atr_based = absb_mod.evaluate_absorption_breakout(bars, stop_atr_mult=0.5)
    micro = absb_mod.evaluate_absorption_breakout(bars, stop_atr_mult=0.5, micro_stop_cents=0.02)

    assert atr_based.stop_price != micro.stop_price
    # level_high is 100.2 (prior 20 bars' rolling high) -- micro stop must
    # sit EXACTLY 2 cents past it, ignoring stop_atr_mult entirely.
    assert micro.stop_price == pytest.approx(100.2 - 0.02)
    assert micro.context["micro_stop_cents"] == pytest.approx(0.02)
    assert atr_based.context["micro_stop_cents"] is None


def test_absorption_breakout_micro_stop_cents_overrides_atr_stop_for_short():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.0, 99.4, 99.5, 10000.0]

    micro = absb_mod.evaluate_absorption_breakout(bars, stop_atr_mult=0.5, micro_stop_cents=0.01)
    # level_low is 99.8 -- short's micro stop sits EXACTLY 1 cent past it
    # (above it, since a failed breakdown reclaiming the level invalidates
    # the short).
    assert micro.stop_price == pytest.approx(99.8 + 0.01)


def test_absorption_breakout_micro_stop_cents_defaults_to_none_unchanged_behavior():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]

    default = absb_mod.evaluate_absorption_breakout(bars)
    explicit_none = absb_mod.evaluate_absorption_breakout(bars, micro_stop_cents=None)
    assert default.stop_price == explicit_none.stop_price
    assert default.context["micro_stop_cents"] is None


def test_absorption_breakout_micro_stop_cents_dispatches_through_intraday_engine():
    bars = _breakout_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.6, 100.0, 100.5, 10000.0]
    sig = eng._evaluate_signal(
        "absorption_breakout", bars, "TEST",
        {"volume_mult": 3.0, "breakout_atr_mult": 0.0, "stop_atr_mult": 0.5,
         "target_r_multiple": None, "micro_stop_cents": 0.02},
        eng.IntradayBacktestConfig(chart_minutes=1), None, None,
    )
    assert sig is not None
    assert sig.stop_price == pytest.approx(100.2 - 0.02)


def _absorption_1m_for_5m_chart(n_closed: int = 110, fire_last_5m_bin: bool = True,
                                fire_last_1m: bool = False) -> pd.DataFrame:
    """n_closed 1m bars from 09:30. 110 bars = 22 closed 5m bins (09:30–11:19)."""
    idx = pd.date_range("2024-06-04 09:30", periods=n_closed, freq="1min")
    rows = []
    for i in range(n_closed):
        in_last_5m = i >= n_closed - 5
        if fire_last_5m_bin and in_last_5m:
            rows.append((100.0, 110.0, 100.0, 109.0, 50_000.0))
        elif fire_last_1m and i == n_closed - 1:
            rows.append((100.0, 110.0, 100.0, 109.0, 50_000.0))
        else:
            rows.append((100.0, 101.0, 99.0, 100.0, 1000.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_absorption_breakout_engine_resamples_chart_minutes_5_and_keeps_1m():
    """Default chart_minutes=5 evaluates closed 5m OHLCV; chart_minutes=1
    still sees the raw 1m prefix. An incomplete 5m bin must not fire."""
    params = {"volume_mult": 3.0, "breakout_atr_mult": 0.5, "stop_atr_mult": 0.5}
    cfg5 = eng.IntradayBacktestConfig()
    cfg1 = eng.IntradayBacktestConfig(chart_minutes=1)
    assert cfg5.chart_minutes == 5

    complete = _absorption_1m_for_5m_chart(110, fire_last_5m_bin=True)
    sig5 = eng._evaluate_signal("absorption_breakout", complete, "TEST", params, cfg5, None, None)
    assert sig5 is not None
    assert sig5.strategy == "absorption_breakout"

    incomplete = _absorption_1m_for_5m_chart(109, fire_last_1m=True, fire_last_5m_bin=False)
    assert eng._evaluate_signal("absorption_breakout", incomplete, "TEST", params, cfg5, None, None) is None
    # Same incomplete prefix still fires when the engine is asked for 1m.
    assert eng._evaluate_signal("absorption_breakout", incomplete, "TEST", params, cfg1, None, None) is not None


# ── context.py's shared target-selection helpers (used by sweep_reclaim AND
#    vp_breakout — see context.py's docstring on why they were pulled up) ──

def test_context_target_levels_exclude_eq_highs_and_lows():
    levels = LiquidityLevels(
        ydh=110.0, ydl=90.0, eq_highs=[100.2, 100.3], eq_lows=[99.7, 99.8], round_levels=[105.0, 95.0],
    )
    resistance = context.target_resistance_levels(levels, current_price=100.0)
    support = context.target_support_levels(levels, current_price=100.0)
    assert 100.2 not in resistance and 100.3 not in resistance
    assert 99.7 not in support and 99.8 not in support
    assert 110.0 in resistance and 105.0 in resistance
    assert 90.0 in support and 95.0 in support


def test_context_nearest_liquidity_target():
    assert context.nearest_liquidity_target([105.0, 110.0, 120.0], 100.0, "long") == 105.0
    assert context.nearest_liquidity_target([80.0, 90.0, 95.0], 100.0, "short") == 95.0
    assert context.nearest_liquidity_target([], 100.0, "long") is None
