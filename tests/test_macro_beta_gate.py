"""Tests for python/analytics/macro_beta_gate.py (Lever 1, absorption_breakout
round 3 — backtests/reports/absorption_breakout_investigation_report.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.analytics.macro_beta_gate import (
    LiveMacroGate,
    _ohlcv_cols,
    compute_macro_momentum,
    load_index_1m_bars,
    macro_gate_ok,
)


def _bars(closes: list[float], start: str = "2026-01-02 09:30:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                          "volume": [1000.0] * len(closes)}, index=idx)


def test_compute_macro_momentum_averages_across_symbols():
    # QQQ rises steadily; SPY flat; composite 1m momentum should be the
    # average of the two symbols' own 1-bar returns at each timestamp.
    qqq = _bars([100, 101, 102, 103, 104, 105, 106])
    spy = _bars([50, 50, 50, 50, 50, 50, 50])
    out = compute_macro_momentum({"QQQ": qqq, "SPY": spy})

    t = qqq.index[2]
    qqq_1m = 102 / 101 - 1.0
    spy_1m = 0.0
    assert out.loc[t, "mom_1m"] == pytest.approx((qqq_1m + spy_1m) / 2)


def test_compute_macro_momentum_no_lookahead():
    """mom_1m/mom_5m at row t must be unaffected by any bar strictly after t."""
    closes = [100, 101, 102, 103, 104, 105, 106, 107]
    qqq_a = _bars(closes)
    out_a = compute_macro_momentum({"QQQ": qqq_a})

    # Mutate everything AFTER index 4 and recompute — rows [0..4] must be identical.
    closes_b = list(closes)
    closes_b[5:] = [999.0, 999.0, 999.0]
    qqq_b = _bars(closes_b)
    out_b = compute_macro_momentum({"QQQ": qqq_b})

    pd.testing.assert_frame_equal(out_a.iloc[:5], out_b.iloc[:5])


def test_compute_macro_momentum_first_bars_are_nan():
    qqq = _bars([100, 101, 102, 103, 104, 105, 106])
    out = compute_macro_momentum({"QQQ": qqq}, momentum_bars=(1, 5))
    assert pd.isna(out["mom_1m"].iloc[0])
    assert out["mom_5m"].iloc[:5].isna().all()
    assert not pd.isna(out["mom_5m"].iloc[5])


def test_compute_macro_momentum_handles_misaligned_timestamps_leniently():
    """A timestamp present for only ONE of the symbols still yields a
    (single-symbol) value rather than being dropped or NaN'd out."""
    qqq = _bars([100, 101, 102, 103, 104, 105])
    spy = _bars([50, 50, 50, 50, 50, 50])
    spy_gapped = spy.drop(spy.index[3])  # SPY missing minute 3, QQQ has it

    out = compute_macro_momentum({"QQQ": qqq, "SPY": spy_gapped})
    t = qqq.index[3]
    assert t in out.index
    qqq_only_1m = 103 / 102 - 1.0
    assert out.loc[t, "mom_1m"] == pytest.approx(qqq_only_1m)


def test_compute_macro_momentum_raises_on_empty_input():
    with pytest.raises(ValueError):
        compute_macro_momentum({})


def test_compute_macro_momentum_does_not_leak_across_session_boundary():
    """Regression for the 2026-08-15 round-2 audit finding
    (backtests/reports/backtest_engine_audit_round2.md): the first `bars`
    bars of a NEW session must be NaN, never silently computed against the
    PREVIOUS session's closing bars (an overnight-gap return masquerading
    as an N-minute intraday momentum). Two sessions, closes chosen so the
    "cross-session" (buggy) and "session-scoped" (correct) answers are
    wildly different and therefore unmistakable."""
    day1 = pd.date_range("2026-01-02 09:30:00", periods=5, freq="1min")
    day2 = pd.date_range("2026-01-05 09:30:00", periods=5, freq="1min")
    closes = pd.Series(
        [100.0, 100.0, 100.0, 100.0, 100.0,   # day 1: flat
         500.0, 501.0, 502.0, 503.0, 504.0],  # day 2: huge gap up, then a mild drift
        index=day1.append(day2),
    )
    qqq = pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000.0] * len(closes),
    })
    out = compute_macro_momentum({"QQQ": qqq}, momentum_bars=(1, 5))

    first_bar_day2 = day2[0]
    # The buggy cross-session version would compute 500/100 - 1 = +4.0 for
    # mom_1m here (comparing day 2's open to day 1's LAST close) — instead
    # the very first bar of a session must have NO trailing bar at all
    # within that same session, so both momentum columns are NaN.
    assert pd.isna(out.loc[first_bar_day2, "mom_1m"])
    assert pd.isna(out.loc[first_bar_day2, "mom_5m"])

    third_bar_day2 = day2[2]
    # By the 3rd bar of day 2, mom_1m has exactly 1 same-session prior bar
    # to look back to (502/501 - 1), and mom_5m still cannot reach back 5
    # bars within this short session, so it stays NaN too.
    assert out.loc[third_bar_day2, "mom_1m"] == pytest.approx(502.0 / 501.0 - 1.0)
    assert pd.isna(out.loc[third_bar_day2, "mom_5m"])

    # Day 1 itself is unaffected by the fix (no prior session exists to
    # leak from) — same values as the original un-grouped shift would give.
    assert out.loc[day1[4], "mom_1m"] == pytest.approx(0.0)


# ── macro_gate_ok ────────────────────────────────────────────────────────────

def _momentum_df(rows: dict) -> pd.DataFrame:
    """rows: {timestamp_str: (mom_1m, mom_5m)}"""
    idx = pd.to_datetime(list(rows.keys()))
    data = list(rows.values())
    return pd.DataFrame(data, index=idx, columns=["mom_1m", "mom_5m"])


def test_macro_gate_ok_allows_long_when_both_momentum_non_negative():
    mm = _momentum_df({"2026-01-02 09:35:00": (0.001, 0.002)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:35:00")) is True


def test_macro_gate_ok_blocks_long_when_1m_momentum_negative():
    mm = _momentum_df({"2026-01-02 09:35:00": (-0.001, 0.002)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:35:00")) is False


def test_macro_gate_ok_blocks_long_when_5m_momentum_negative():
    mm = _momentum_df({"2026-01-02 09:35:00": (0.001, -0.0001)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:35:00")) is False


def test_macro_gate_ok_allows_short_when_both_momentum_non_positive():
    mm = _momentum_df({"2026-01-02 09:35:00": (-0.001, -0.002)})
    assert macro_gate_ok(mm, "short", pd.Timestamp("2026-01-02 09:35:00")) is True


def test_macro_gate_ok_blocks_short_when_momentum_positive():
    mm = _momentum_df({"2026-01-02 09:35:00": (0.001, -0.002)})
    assert macro_gate_ok(mm, "short", pd.Timestamp("2026-01-02 09:35:00")) is False


def test_macro_gate_ok_exact_zero_is_allowed_both_directions():
    mm = _momentum_df({"2026-01-02 09:35:00": (0.0, 0.0)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:35:00")) is True
    assert macro_gate_ok(mm, "short", pd.Timestamp("2026-01-02 09:35:00")) is True


def test_macro_gate_ok_uses_asof_within_one_decision_bin():
    """Last night: signal 12:00:00.196 vs a start-labeled 12:00 row, or
    a refresh that still only had 11:58. Exact membership failed closed.
    As-of must take the last closed minute within 5 minutes."""
    mm = _momentum_df({"2026-01-02 09:35:00": (0.001, 0.002)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:35:00.196")) is True
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:38:00")) is True
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:40:00")) is True


def test_macro_gate_ok_fails_closed_when_asof_too_stale():
    mm = _momentum_df({"2026-01-02 09:35:00": (0.001, 0.002)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:40:01")) is False
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 15:00:00")) is False


def test_macro_gate_ok_fails_closed_on_nan_momentum():
    mm = _momentum_df({"2026-01-02 09:35:00": (np.nan, 0.002)})
    assert macro_gate_ok(mm, "long", pd.Timestamp("2026-01-02 09:35:00")) is False


def test_macro_gate_ok_rejects_unknown_direction():
    mm = _momentum_df({"2026-01-02 09:35:00": (0.001, 0.002)})
    with pytest.raises(ValueError):
        macro_gate_ok(mm, "sideways", pd.Timestamp("2026-01-02 09:35:00"))


def test_ohlcv_cols_returns_frame_not_column_index():
    """Regression: load_index_1m_bars used to do df[_ohlcv_cols(df)],
    which TypeErrors when _ohlcv_cols already returns a DataFrame."""
    raw = _bars([100, 101, 102])
    raw["extra"] = 1.0
    sliced = _ohlcv_cols(raw)
    assert list(sliced.columns) == ["open", "high", "low", "close", "volume"]
    # The live seed path assigns the return value directly.
    assert sliced["close"].iloc[-1] == 102


def test_load_index_1m_bars_does_not_typeerror_on_cached_panel(tmp_path, monkeypatch):
    idx = pd.date_range("2026-08-14 09:30", periods=3, freq="1min")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [100.0, 101.0, 102.0], "volume": 1.0},
        index=pd.MultiIndex.from_product([["QQQ"], idx], names=["code", "datetime"]),
    )

    def _fake_panel(symbols, start, end, cache_dir=None):
        return df

    monkeypatch.setattr(
        "python.data.intraday_cache.get_cached_intraday_panel",
        _fake_panel,
    )
    out = load_index_1m_bars(symbols=("QQQ",), lookback_days=1, cache_dir=tmp_path, fetch_live=False)
    assert "QQQ" in out
    assert list(out["QQQ"].columns) == ["open", "high", "low", "close", "volume"]


def test_refresh_for_skipped_when_live_fetch_disabled(monkeypatch):
    gate = LiveMacroGate(live_fetch=False)
    called = {"n": 0}

    def _boom(**kwargs):
        called["n"] += 1
        raise AssertionError("must not fetch")

    monkeypatch.setattr("python.analytics.macro_beta_gate.load_index_1m_bars", _boom)
    gate.refresh_for(pd.Timestamp("2026-03-02 12:00:00"))
    assert called["n"] == 0
    assert gate.ok("long", pd.Timestamp("2026-03-02 12:00:00")) is False


def test_refresh_for_fetches_once_per_decision_minute(monkeypatch):
    closes = [100, 101, 102, 103, 104, 105, 106]
    idx = pd.date_range("2026-03-02 09:30:00", periods=len(closes), freq="1min")
    bars = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000.0},
        index=idx,
    )
    calls = {"n": 0}

    def _load(*, fetch_live=False, **kwargs):
        calls["n"] += 1
        assert fetch_live is True
        return {"QQQ": bars, "SPY": bars, "XLK": bars}

    monkeypatch.setattr("python.analytics.macro_beta_gate.load_index_1m_bars", _load)
    gate = LiveMacroGate(live_fetch=True)
    t = pd.Timestamp("2026-03-02 09:36:00.196")
    gate.refresh_for(t)
    gate.refresh_for(t)
    gate.refresh_for(pd.Timestamp("2026-03-02 09:36:30"))
    assert calls["n"] == 1
    assert gate.ok("long", t) is True
    gate.refresh_for(pd.Timestamp("2026-03-02 09:41:00"))
    assert calls["n"] == 2
