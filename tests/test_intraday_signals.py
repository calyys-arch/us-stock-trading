"""
Signal module tests: synthetic bar sequences for each of sweep_reclaim,
fvg_retest, orb_vwap — including explicit no-lookahead checks (a signal
must never fire before the bar that actually completes its pattern, and
future bars beyond "now" must never influence what fires at "now")."""
from __future__ import annotations

import pandas as pd
import pytest

from python.microstructure import context
from python.microstructure.signals import l2_absorption as absorb_mod
from python.microstructure.signals import orb_vwap as orb_mod
from python.microstructure.signals import fvg_retest as fvg_mod
from python.microstructure.signals import sweep_reclaim as sweep_mod
from python.microstructure.context import LiquidityLevels, OpeningRange


def _flat_bars(n: int, price: float = 100.0, start: str = "2024-06-04 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price, "volume": 1000.0,
    }, index=idx)


# ── sweep_reclaim ────────────────────────────────────────────────────────────

def test_sweep_reclaim_fires_on_reclaim_bar():
    bars = _flat_bars(10, price=100.0)
    # bar[5]: sweeps above 105 resistance; bar[6]: closes back inside.
    bars.loc[bars.index[5], ["open", "high", "low", "close"]] = [100.0, 106.0, 100.0, 105.5]
    bars.loc[bars.index[6], ["open", "high", "low", "close"]] = [105.0, 105.2, 99.0, 99.5]
    atr_series = context.atr(bars, period=5)
    levels = LiquidityLevels(ydh=105.0)

    # No signal yet at the sweep bar itself is fine either way, but there
    # must be NO signal before the sweep has even happened.
    sig_before = sweep_mod.evaluate_sweep_reclaim(bars.iloc[:5], levels, atr_series.iloc[:5])
    assert sig_before is None

    sig = sweep_mod.evaluate_sweep_reclaim(bars.iloc[:7], levels, atr_series.iloc[:7])
    assert sig is not None
    assert sig.direction == "short"
    assert sig.strategy == "sweep_reclaim"
    assert sig.signal_time == bars.index[6]
    assert sig.stop_price > 106.0  # stop beyond the sweep extreme


def test_sweep_reclaim_support_side_long():
    bars = _flat_bars(10, price=100.0)
    bars.loc[bars.index[5], ["open", "high", "low", "close"]] = [100.0, 100.0, 94.0, 94.5]
    bars.loc[bars.index[6], ["open", "high", "low", "close"]] = [95.0, 100.5, 95.0, 100.5]
    atr_series = context.atr(bars, period=5)
    levels = LiquidityLevels(ydl=95.0)

    sig = sweep_mod.evaluate_sweep_reclaim(bars.iloc[:7], levels, atr_series.iloc[:7])
    assert sig is not None
    assert sig.direction == "long"
    assert sig.stop_price < 94.0


def test_sweep_reclaim_no_signal_without_sweep():
    bars = _flat_bars(10, price=100.0)
    atr_series = context.atr(bars, period=5)
    levels = LiquidityLevels(ydh=105.0, ydl=95.0)
    assert sweep_mod.evaluate_sweep_reclaim(bars, levels, atr_series) is None


def test_sweep_reclaim_future_bars_do_not_affect_earlier_signal_no_lookahead():
    bars = _flat_bars(10, price=100.0)
    bars.loc[bars.index[5], ["open", "high", "low", "close"]] = [100.0, 106.0, 100.0, 105.5]
    bars.loc[bars.index[6], ["open", "high", "low", "close"]] = [105.0, 105.2, 99.0, 99.5]
    atr_series = context.atr(bars, period=5)
    levels = LiquidityLevels(ydh=105.0)

    sig_at_6 = sweep_mod.evaluate_sweep_reclaim(bars.iloc[:7], levels, atr_series.iloc[:7])

    # Mutate a FUTURE bar (index 8) into something wild; the decision made
    # using data available through bar 6 must be unaffected.
    bars2 = bars.copy()
    bars2.loc[bars2.index[8], ["open", "high", "low", "close"]] = [500.0, 600.0, 400.0, 550.0]
    atr_series2 = context.atr(bars2, period=5)
    sig_at_6_again = sweep_mod.evaluate_sweep_reclaim(bars2.iloc[:7], levels, atr_series2.iloc[:7])

    assert sig_at_6 is not None and sig_at_6_again is not None
    assert sig_at_6.entry_price == sig_at_6_again.entry_price
    assert sig_at_6.stop_price == sig_at_6_again.stop_price


# ── fvg_retest ───────────────────────────────────────────────────────────────

def test_fvg_retest_fires_on_bullish_gap():
    bars = _flat_bars(25, price=100.0, start="2024-06-04 09:30")
    bars.loc[bars.index[21], ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.2]  # bar1
    bars.loc[bars.index[22], ["open", "high", "low", "close", "volume"]] = [100.2, 106.0, 100.1, 105.5, 50000.0]  # bar2 impulse
    bars.loc[bars.index[23], ["open", "high", "low", "close"]] = [105.5, 107.0, 105.2, 106.5]  # bar3, low(105.2) > bar1.high(100.5)

    sig_before = fvg_mod.evaluate_fvg_retest(bars.iloc[:22])
    assert sig_before is None

    sig = fvg_mod.evaluate_fvg_retest(bars.iloc[:24])
    assert sig is not None
    assert sig.direction == "long"
    assert sig.strategy == "fvg_retest"
    assert sig.order_type == "limit"
    assert 100.5 <= sig.entry_price <= 105.2
    assert sig.expiry_time > sig.signal_time


def test_fvg_retest_no_signal_without_volume_spike():
    bars = _flat_bars(25, price=100.0)
    bars.loc[bars.index[21], ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.2]
    bars.loc[bars.index[22], ["open", "high", "low", "close"]] = [100.2, 106.0, 100.1, 105.5]  # normal volume
    bars.loc[bars.index[23], ["open", "high", "low", "close"]] = [105.5, 107.0, 105.2, 106.5]
    assert fvg_mod.evaluate_fvg_retest(bars.iloc[:24]) is None


def test_fvg_retest_does_not_refire_on_stale_gap():
    bars = _flat_bars(25, price=100.0)
    bars.loc[bars.index[21], ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.2]
    bars.loc[bars.index[22], ["open", "high", "low", "close", "volume"]] = [100.2, 106.0, 100.1, 105.5, 50000.0]
    bars.loc[bars.index[23], ["open", "high", "low", "close"]] = [105.5, 107.0, 105.2, 106.5]
    sig_at_23 = fvg_mod.evaluate_fvg_retest(bars.iloc[:24])
    sig_at_24 = fvg_mod.evaluate_fvg_retest(bars.iloc[:25])  # one bar later — same gap, must not refire
    assert sig_at_23 is not None
    assert sig_at_24 is None


# ── orb_vwap ─────────────────────────────────────────────────────────────────

def _orb_bars() -> pd.DataFrame:
    bars = _flat_bars(40, price=100.0, start="2024-06-04 09:30")
    # OR window = first 15 bars (09:30-09:45): high 101, low 99.
    bars.loc[bars.index[5], "high"] = 101.0
    bars.loc[bars.index[6], "low"] = 99.0
    return bars


def test_orb_vwap_no_entry_inside_opening_range():
    bars = _orb_bars()
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)
    sig = orb_mod.evaluate_orb_vwap(bars.iloc[:10], orange, vwap.iloc[:10], vwap_side_filter=False)
    assert sig is None  # bar 10 is still inside the 15-minute OR window


def test_orb_vwap_fires_on_fresh_breakout_with_vwap_filter():
    bars = _orb_bars()
    bars.loc[bars.index[20], "close"] = 100.0  # keep below OR high leading in
    bars.loc[bars.index[21], "close"] = 102.0  # fresh breakout above OR high (101.0)
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    sig = orb_mod.evaluate_orb_vwap(bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=True)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.context["breakout_dir"] == "long"


def test_orb_vwap_does_not_refire_while_staying_above_or_high():
    bars = _orb_bars()
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0
    bars.loc[bars.index[22], "close"] = 103.0  # still above OR high — must NOT refire
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    sig_21 = orb_mod.evaluate_orb_vwap(bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False)
    sig_22 = orb_mod.evaluate_orb_vwap(bars.iloc[:23], orange, vwap.iloc[:23], vwap_side_filter=False)
    assert sig_21 is not None
    assert sig_22 is None


def test_orb_vwap_gap_trap_flips_direction():
    bars = _orb_bars()
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0  # breakout LONG
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    # Pre-market gap DOWN (today's open 100 < prior close 110) opposite the
    # long breakout -> trap rule flips the trade to short.
    sig = orb_mod.evaluate_orb_vwap(
        bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False, prior_close=110.0,
    )
    assert sig is not None
    assert sig.direction == "short"
    assert sig.context["trap_flag"] is True
    assert sig.context["breakout_dir"] == "long"


def test_orb_vwap_stop_is_always_on_the_adverse_side_of_entry():
    """Regression guard for the gap-trap inverted-stop defect
    (orb_vwap.py's CORRECTNESS FIX note): the trap rule flips `direction`,
    and the original stop assignment then landed on the FAVORABLE side of
    the entry, which intraday_engine._check_exit fires immediately as a
    profitable exit labelled "stop". Both the ordinary and the trap-faded
    case must place the stop where it can only lose money."""
    bars = _orb_bars()
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], ["high", "low", "close"]] = [102.5, 101.5, 102.0]  # breakout LONG
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    plain = orb_mod.evaluate_orb_vwap(bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False)
    assert plain.direction == "long"
    assert plain.stop_price < plain.entry_price
    assert plain.stop_price == pytest.approx(99.0)  # unchanged: the raw OR low

    trapped = orb_mod.evaluate_orb_vwap(
        bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False, prior_close=110.0,
    )
    assert trapped.direction == "short" and trapped.context["trap_flag"] is True
    assert trapped.stop_price > trapped.entry_price
    # Beyond the failed-breakout extreme (this bar's high 102.5), not the
    # opening-range high (101.0) which the break already ran through.
    assert trapped.stop_price == pytest.approx(102.5)


def test_orb_vwap_atr_buffer_widens_the_stop_and_is_off_by_default():
    bars = _orb_bars()
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], ["high", "low", "close"]] = [102.0, 101.5, 102.0]
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)
    atr = context.atr(bars)

    no_buffer = orb_mod.evaluate_orb_vwap(bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False)
    buffered = orb_mod.evaluate_orb_vwap(
        bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False,
        atr_series=atr.iloc[:22], stop_atr_buffer_mult=0.5,
    )
    atr_now = float(atr.iloc[21])
    assert atr_now > 0
    assert buffered.stop_price == pytest.approx(no_buffer.stop_price - 0.5 * atr_now)

    # A buffer requested without an ATR series is skipped, not guessed.
    no_atr = orb_mod.evaluate_orb_vwap(
        bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False, stop_atr_buffer_mult=0.5,
    )
    assert no_atr.stop_price == pytest.approx(no_buffer.stop_price)


def test_orb_vwap_r_multiple_target_is_off_by_default():
    bars = _orb_bars()
    bars.loc[bars.index[20], "close"] = 100.0
    bars.loc[bars.index[21], "close"] = 102.0
    orange = context.opening_range(bars, minutes=15)
    vwap = context.session_vwap(bars)

    default = orb_mod.evaluate_orb_vwap(bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False)
    assert default.target_price is None

    targeted = orb_mod.evaluate_orb_vwap(
        bars.iloc[:22], orange, vwap.iloc[:22], vwap_side_filter=False, target_r_multiple=2.0,
    )
    risk = targeted.entry_price - targeted.stop_price
    assert risk > 0
    assert targeted.target_price == pytest.approx(targeted.entry_price + 2.0 * risk)


# ── l2_absorption (S4, bar-only proxy — see module docstring) ──────────────

def _absorption_bars(n: int = 25, price: float = 100.0, band: float = 0.2) -> pd.DataFrame:
    idx = pd.date_range("2024-06-04 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "open": price, "high": price + band, "low": price - band, "close": price, "volume": 1000.0,
    }, index=idx)


def test_l2_absorption_fires_long_on_support_touch_with_volume_spike():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    # "now" bar dips to the level_low the prior 20 bars established (99.8),
    # on 10x volume, but closes back above it -> absorption at support.
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.1, 99.8, 100.05, 10000.0]

    sig = absorb_mod.evaluate_l2_absorption(bars)
    assert sig is not None
    assert sig.strategy == "l2_absorption"
    assert sig.direction == "long"
    assert sig.stop_price < 99.8
    assert sig.context["tier"] == "bar_only_proxy_no_l2_confirmation"


def test_l2_absorption_fires_short_on_resistance_touch_with_volume_spike():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.2, 99.9, 99.95, 10000.0]

    sig = absorb_mod.evaluate_l2_absorption(bars)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.stop_price > 100.2


def test_l2_absorption_no_signal_without_volume_spike():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    # Same touch-and-hold shape as the long case, but NORMAL volume.
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.1, 99.8, 100.05, 1000.0]

    assert absorb_mod.evaluate_l2_absorption(bars) is None


def test_l2_absorption_no_signal_when_level_is_cleanly_violated():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    # Heavy volume AND the level breaks cleanly (close well below level_low)
    # — a breakdown, not absorption.
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.0, 98.0, 98.2, 10000.0]

    assert absorb_mod.evaluate_l2_absorption(bars) is None


def test_l2_absorption_no_signal_with_too_few_bars():
    bars = _absorption_bars(10, price=100.0, band=0.2)
    assert absorb_mod.evaluate_l2_absorption(bars) is None


def test_l2_absorption_r_multiple_target_is_off_by_default():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.1, 99.8, 100.05, 10000.0]

    default = absorb_mod.evaluate_l2_absorption(bars)
    assert default.target_price is None
    assert default.context["target_r_multiple"] is None

    targeted = absorb_mod.evaluate_l2_absorption(bars, target_r_multiple=2.0)
    risk = targeted.entry_price - targeted.stop_price
    assert risk > 0
    assert targeted.target_price == pytest.approx(targeted.entry_price + 2.0 * risk)
    assert targeted.context["target_r_multiple"] == 2.0


def test_l2_absorption_r_multiple_target_short_direction():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.2, 99.9, 99.95, 10000.0]

    targeted = absorb_mod.evaluate_l2_absorption(bars, target_r_multiple=1.5)
    assert targeted.direction == "short"
    risk = targeted.stop_price - targeted.entry_price
    assert risk > 0
    assert targeted.target_price == pytest.approx(targeted.entry_price - 1.5 * risk)


def test_l2_absorption_future_bars_do_not_affect_earlier_signal_no_lookahead():
    bars = _absorption_bars(25, price=100.0, band=0.2)
    bars.loc[bars.index[24], ["open", "high", "low", "close", "volume"]] = [100.0, 100.1, 99.8, 100.05, 10000.0]
    sig_at_24 = absorb_mod.evaluate_l2_absorption(bars.iloc[:25])

    bars2 = bars.copy()
    # Extend with a wild future bar; must not change the decision already
    # made using data available through "now" (index 24).
    idx26 = bars.index[24] + pd.Timedelta(minutes=1)
    bars2.loc[idx26] = [500.0, 600.0, 400.0, 550.0, 999999.0]
    sig_at_24_again = absorb_mod.evaluate_l2_absorption(bars2.iloc[:25])

    assert sig_at_24 is not None and sig_at_24_again is not None
    assert sig_at_24.entry_price == sig_at_24_again.entry_price
    assert sig_at_24.stop_price == sig_at_24_again.stop_price
