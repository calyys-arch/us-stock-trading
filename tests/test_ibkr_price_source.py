"""
ibkr_price_source unit tests (no IB Gateway): durationStr/endDateTime
formatting, broker.yaml connection-setting defaults, and — the important
integration seam — price_cache falling back to yfinance WITH an honest
source label when IBKR raises IbkrHistoricalUnavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.data.ibkr_price_source import (
    IbkrHistoricalUnavailable,
    _duration_str,
    load_connection_settings,
)


def test_duration_str_days_and_years():
    assert _duration_str(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")).endswith(" D")
    assert _duration_str(pd.Timestamp("2018-01-01"), pd.Timestamp("2025-01-01")) == "8 Y"
    assert _duration_str(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")) == "6 D"


def test_fetch_ibkr_daily_bars_uses_blank_end_date_for_adjusted_last(monkeypatch):
    """Regression test: IB rejects reqHistoricalData with a non-blank
    endDateTime when whatToShow=ADJUSTED_LAST ('Error validating request...
    End date not supported with adjusted last'), and — worse — ib_async
    never raises for this particular error, so the synchronous call hangs
    forever instead of failing loudly. endDateTime must always be blank
    ("" = up to now) for ADJUSTED_LAST bars; the caller trims the result
    to [start, end] itself."""
    import ib_async

    from python.data import ibkr_price_source

    calls: dict = {}

    class FakeBar:
        def __init__(self, date):
            self.date = date
            self.open = self.high = self.low = self.close = 100.0
            self.volume = 1000

    class FakeIB:
        def connect(self, *a, **kw):
            pass

        def isConnected(self):
            return True

        def disconnect(self):
            pass

        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, contract, **kwargs):
            calls["kwargs"] = kwargs
            return [FakeBar(pd.Timestamp("2024-01-02"))]

    monkeypatch.setattr(ib_async, "IB", FakeIB)
    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())

    panel, _flags = ibkr_price_source.fetch_ibkr_daily_bars(["AAA"], "2024-01-01", "2024-01-02")

    assert calls["kwargs"]["endDateTime"] == ""
    assert len(panel) == 1


class _AlwaysReadyLimiter:
    """Stub for _get_intraday_rate_limiter — the real one is a process-wide
    singleton at 0.1 req/s (one token every ~10s), so back-to-back test
    calls would otherwise stall for real wall-clock seconds waiting for a
    refill that has nothing to do with what these tests are checking."""

    def try_acquire(self) -> bool:
        return True


def test_fetch_ibkr_intraday_month_handles_tz_aware_bar_dates(monkeypatch):
    """Regression test: real IB Gateway responses to reqHistoricalData for
    intraday bar sizes come back with ib_async parsing bar.date into a
    tz-AWARE Timestamp (US/Eastern) — unlike the tz-naive dates daily bars
    get. Before this fix, comparing that tz-aware index against the
    tz-naive month_start/month_end bounds raised
    "Invalid comparison between dtype=datetime64[us, US/Eastern] and
    Timestamp" on every single real call (caught only by manually running
    scripts/backfill_intraday.py against a live IB Gateway — no mocked
    test exercised this path with a tz-aware bar.date)."""
    from python.data import ibkr_price_source
    from python.data.ibkr_price_source import fetch_ibkr_intraday_month

    monkeypatch.setattr(ibkr_price_source, "_get_intraday_rate_limiter", lambda: _AlwaysReadyLimiter())

    class FakeBar:
        def __init__(self, date):
            self.date = date
            self.open = self.high = self.low = self.close = 100.0
            self.volume = 1000

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, contract, **kwargs):
            tz_aware_dates = pd.date_range("2025-07-01 09:30:00", periods=3, freq="1min", tz="US/Eastern")
            return [FakeBar(d) for d in tz_aware_dates]

    df = fetch_ibkr_intraday_month(FakeIB(), "AAPL", pd.Timestamp("2025-07-01"))

    assert not df.empty
    assert df.index.tz is None
    assert list(df.index) == list(pd.date_range("2025-07-01 09:30:00", periods=3, freq="1min"))


def test_fetch_ibkr_intraday_month_returns_empty_frame_for_no_bars(monkeypatch):
    from python.data import ibkr_price_source
    from python.data.ibkr_price_source import fetch_ibkr_intraday_month

    monkeypatch.setattr(ibkr_price_source, "_get_intraday_rate_limiter", lambda: _AlwaysReadyLimiter())

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, contract, **kwargs):
            return []

    df = fetch_ibkr_intraday_month(FakeIB(), "AAPL", pd.Timestamp("2025-07-01"))
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_load_connection_settings_defaults(tmp_path):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("ibkr:\n  host: 10.0.0.5\n  feed_port: 7497\n", encoding="utf-8")
    settings = load_connection_settings(cfg)
    assert settings == {"host": "10.0.0.5", "port": 7497, "client_id": 31}


def test_repo_broker_yaml_has_distinct_client_ids():
    """historical/tick-capture clientIds must differ from the live feed/
    broker ids — IB disconnects the older session on a clientId collision."""
    import yaml

    with open("configs/broker.yaml", encoding="utf-8") as f:
        ibkr = yaml.safe_load(f)["ibkr"]
    ids = [ibkr["feed_client_id"], ibkr["broker_client_id"],
           ibkr["historical_client_id"], ibkr["tick_capture_client_id"]]
    assert len(set(ids)) == len(ids)


def test_price_cache_falls_back_to_yfinance_with_honest_label(tmp_path, monkeypatch):
    """IBKR unavailable -> _fetch_remote must fall back to yfinance AND the
    cache meta must record 'yfinance' as the actual source (never silently
    pretending the data came from IB)."""
    from python.data import price_cache

    def _ibkr_raises(symbols, start, end, config_path):
        raise IbkrHistoricalUnavailable("gateway not running")

    def _fake_yf(symbols, start, end):
        dates = pd.bdate_range(start, end)
        frames = []
        for code in symbols:
            close = 100 + np.arange(len(dates), dtype=float)
            df = pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1e6}, index=dates)
            df.index.name = "date"
            df["adv_20d_dollars"] = 1e8
            df["code"] = code
            frames.append(df.reset_index().set_index(["date", "code"]))
        return pd.concat(frames).sort_index(), {}

    import python.data.ibkr_price_source as ibkr_mod
    import python.simulation.hist_data_us as yf_mod

    monkeypatch.setattr(ibkr_mod, "fetch_ibkr_daily_bars", _ibkr_raises)
    monkeypatch.setattr(yf_mod, "build_price_panel", _fake_yf)

    broker_cfg = tmp_path / "broker.yaml"
    broker_cfg.write_text("historical_data_source: ibkr\n", encoding="utf-8")

    panel, _flags, meta = price_cache.get_cached_price_panel(
        ["AAA"], "2024-01-02", "2024-03-01",
        cache_dir=tmp_path / "cache", broker_config_path=broker_cfg)
    assert meta["fetched_source"] == "yfinance"
    assert meta["sources"] == {"yfinance": ["AAA"]}
    assert len(panel) > 0


def test_price_cache_respects_yfinance_only_setting(tmp_path, monkeypatch):
    """historical_data_source: yfinance must never even try IBKR."""
    from python.data import price_cache

    def _ibkr_must_not_be_called(*args, **kwargs):
        raise AssertionError("IBKR path must not be used when historical_data_source=yfinance")

    def _fake_yf(symbols, start, end):
        dates = pd.bdate_range(start, end)
        df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                           "volume": 1e6, "adv_20d_dollars": 1e6, "code": symbols[0]},
                          index=dates)
        df.index.name = "date"
        return df.reset_index().set_index(["date", "code"]), {}

    import python.data.ibkr_price_source as ibkr_mod
    import python.simulation.hist_data_us as yf_mod

    monkeypatch.setattr(ibkr_mod, "fetch_ibkr_daily_bars", _ibkr_must_not_be_called)
    monkeypatch.setattr(yf_mod, "build_price_panel", _fake_yf)

    broker_cfg = tmp_path / "broker.yaml"
    broker_cfg.write_text("historical_data_source: yfinance\n", encoding="utf-8")
    _panel, _flags, meta = price_cache.get_cached_price_panel(
        ["BBB"], "2024-01-02", "2024-02-01",
        cache_dir=tmp_path / "cache", broker_config_path=broker_cfg)
    assert meta["fetched_source"] == "yfinance"
