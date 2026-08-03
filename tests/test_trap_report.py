"""
trap_report wiring tests — confirm assess_one/build_trap_report actually
call through to the FINRA ATS Tier 2 flag (python/data/finra_ats.py) and
that a missing cache degrades to None (unavailable), never a crash or a
silent False.

All other evidence sources (news/filings/ticks caches) simply don't exist
for the throwaway "ZZZTEST" symbol used here, which is the correct way to
exercise the "cache-only, missing == unavailable" contract without needing
real fixture data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from python.signals import trap_report
from python.signals.trap_report import assess_one, build_trap_report


def _flat_ohlcv(n_days: int = 30, price: float = 50.0) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    close = price + rng.normal(0, 0.2, n_days)
    df = pd.DataFrame({
        "open": close, "close": close,
        "high": close + 0.3, "low": close - 0.3,
        "volume": np.full(n_days, 500_000.0),
    }, index=dates)
    return df


def test_assess_one_wires_finra_flag_to_event_flags(monkeypatch):
    monkeypatch.setattr(trap_report, "_finra_elevated_vs_baseline", lambda symbol, week: True)
    df = _flat_ohlcv()
    assessment = assess_one("ZZZTEST", df.index[-1], df, earnings_by_symbol={}, econ_dates=set(), signal_context="unit test")
    assert assessment.event_flags["dark_pool_participation_elevated"] is True


def test_assess_one_missing_finra_cache_is_none_not_false():
    """No monkeypatch — real python/data/finra_ats.py against a throwaway
    symbol with no data/finra_ats/ZZZTEST.jsonl cache on disk."""
    df = _flat_ohlcv()
    assessment = assess_one("ZZZTEST", df.index[-1], df, earnings_by_symbol={}, econ_dates=set(), signal_context="unit test")
    assert assessment.event_flags["dark_pool_participation_elevated"] is None


def test_build_trap_report_flags_row_on_finra_flag_alone(monkeypatch, tmp_path):
    """A row with no computable trap sub-scores (no tick/news/filing
    caches) but an elevated FINRA flag must still show up as flagged —
    build_trap_report's `_flagged` predicate ORs in any True event flag,
    not just the trap_score."""
    monkeypatch.setattr(trap_report, "_finra_elevated_vs_baseline", lambda symbol, week: True)
    df = _flat_ohlcv()
    day = df.index[-1]
    out_path = build_trap_report(
        signal_specs=[("ZZZTEST", day, "unit test signal")],
        panel_by_symbol={"ZZZTEST": df},
        start=df.index[0], end=day,
        data_label="unit test",
        out_path=tmp_path / "report.md",
    )
    text = out_path.read_text(encoding="utf-8")
    assert "ZZZTEST" in text
    assert "dark_pool_participation_elevated=yes" in text


def test_build_trap_report_no_finra_cache_shows_unknown(tmp_path):
    df = _flat_ohlcv()
    day = df.index[-1]
    # Flat, quiet OHLCV won't trip any trap sub-score and there's no FINRA
    # cache for ZZZTEST -> nothing to flag, row shouldn't appear in the list.
    out_path = build_trap_report(
        signal_specs=[("ZZZTEST", day, "unit test signal")],
        panel_by_symbol={"ZZZTEST": df},
        start=df.index[0], end=day,
        data_label="unit test",
        out_path=tmp_path / "report.md",
    )
    text = out_path.read_text(encoding="utf-8")
    assert "(none)" in text
