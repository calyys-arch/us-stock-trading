"""
dashboard chart endpoints — /api/chart/{symbol}?interval=1m and
/api/chart/{symbol}/context. Both read from python/data/intraday_cache.py's
get_cached_intraday_panel, which we monkeypatch (no real cache/network
needed) — matching this repo's convention of never hitting IB Gateway in
unit tests.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.app import app
import python.data.intraday_cache as intraday_cache_module


def _synthetic_panel(symbol: str = "AAPL") -> pd.DataFrame:
    """Two sessions, 30 bars/day, with a deliberate volume spike so the
    volume profile POC/VAH/VAL land somewhere non-trivial, and a bullish
    FVG (bars 20/21/22) on day 2 so the signals scan has something to find."""
    frames = []
    for day, price in [("2025-06-02", 100.0), ("2025-06-03", 101.0)]:
        idx = pd.date_range(f"{day} 09:30", periods=30, freq="1min")
        volumes = [1_000.0] * 30
        volumes[10] = 50_000.0  # spike -> obvious POC bin
        df = pd.DataFrame({
            "open": price, "high": price + 0.3, "low": price - 0.3, "close": price, "volume": volumes,
        }, index=idx)
        if day == "2025-06-03":
            df.iloc[20] = [price, price + 0.2, price - 0.2, price + 0.1, 1_000.0]
            df.iloc[21] = [price + 0.1, price + 6.0, price, price + 5.5, 50_000.0]
            df.iloc[22] = [price + 5.5, price + 7.0, price + 5.2, price + 6.5, 1_000.0]
        df["code"] = symbol
        df.index.name = "ts"
        frames.append(df.reset_index().set_index(["ts", "code"]))
    return pd.concat(frames).sort_index()


@pytest.fixture()
def client(monkeypatch):
    panel = _synthetic_panel("AAPL")

    def _fake_get_cached_intraday_panel(symbols, start, end, cache_dir=None):
        symbol = symbols[0].upper()
        if symbol != "AAPL":
            raise RuntimeError(f"no cached 1-minute data for {symbol}")
        return panel

    monkeypatch.setattr(intraday_cache_module, "get_cached_intraday_panel", _fake_get_cached_intraday_panel)
    return TestClient(app)


def test_chart_1m_returns_cached_bars(client):
    resp = client.get("/api/chart/AAPL", params={"interval": "1m", "days": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["interval"] == "1m"
    assert body["symbol"] == "AAPL"
    assert len(body["dates"]) == 60  # 2 sessions x 30 bars
    assert len(body["close"]) == len(body["dates"])


def test_chart_1m_404_when_no_cached_data(client):
    resp = client.get("/api/chart/ZZZZ", params={"interval": "1m"})
    assert resp.status_code == 404
    assert "backfill_intraday" in resp.json()["detail"]


def test_chart_invalid_interval_rejected(client):
    # "5m"/"15m" are valid (resampled on the fly from the 1-minute cache —
    # see get_symbol_chart's docstring); "3m" isn't one of the supported
    # bar sizes and must still be rejected.
    resp = client.get("/api/chart/AAPL", params={"interval": "3m"})
    assert resp.status_code == 400


def test_chart_context_defaults_to_latest_session(client):
    resp = client.get("/api/chart/AAPL/context")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert body["date"] == "2025-06-03"  # latest of the two cached sessions
    assert body["available_dates"] == ["2025-06-02", "2025-06-03"]
    # prior-day YDH/YDL should be populated from the 2025-06-02 session
    assert body["liquidity"]["ydh"] == pytest.approx(100.3, abs=1e-6)
    assert body["liquidity"]["ydl"] == pytest.approx(99.7, abs=1e-6)
    # VWAP series should be aligned 1:1 with the requested session's bars
    assert len(body["vwap"]["dates"]) == 30
    assert len(body["vwap"]["vwap"]) == 30
    # the deliberate volume spike should make bin_volume non-empty
    assert body["volume_profile"]["poc"] is not None
    assert sum(body["volume_profile"]["bin_volume"]) > 0
    assert body["opening_range"]["high"] is not None
    # the deliberate FVG on this session should surface as an fvg_retest signal
    fvg_hits = [s for s in body["signals"] if s["strategy"] == "fvg_retest"]
    assert len(fvg_hits) >= 1
    assert fvg_hits[0]["direction"] == "long"


def test_chart_context_explicit_date(client):
    resp = client.get("/api/chart/AAPL/context", params={"date": "2025-06-02"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == "2025-06-02"
    # no prior cached session before 06-02 -> no YDH/YDL
    assert body["liquidity"]["ydh"] is None


def test_chart_context_unknown_date_404(client):
    resp = client.get("/api/chart/AAPL/context", params={"date": "2030-01-01"})
    assert resp.status_code == 404


def test_chart_context_404_when_no_cached_data(client):
    resp = client.get("/api/chart/ZZZZ/context")
    assert resp.status_code == 404
