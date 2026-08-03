"""
/api/regime/{symbol} — report-only Markov regime diagnostic endpoint.
Monkeypatches python.data.price_cache.get_cached_price_panel (same
convention as tests/test_dashboard_chart_api.py) so no real cache/network
is touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.app import app
import python.data.price_cache as price_cache_module


def _synthetic_daily_panel(symbol: str = "AAPL", n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": 1_000_000.0, "adv_20d_dollars": close * 1_000_000.0,
    }, index=idx)
    df.index.name = "date"
    df["code"] = symbol
    return df.reset_index().set_index(["date", "code"])


@pytest.fixture()
def client(monkeypatch):
    panel = _synthetic_daily_panel("AAPL")

    def _fake_get_cached_price_panel(symbols, start, end, refresh=False, cache_dir=None, broker_config_path=None):
        symbol = symbols[0].upper()
        if symbol != "AAPL":
            raise RuntimeError(f"no cached daily data for {symbol}")
        return panel, {}, {"sources": {"cache": [symbol]}}

    monkeypatch.setattr(price_cache_module, "get_cached_price_panel", _fake_get_cached_price_panel)
    return TestClient(app)


def test_regime_endpoint_returns_full_report(client):
    resp = client.get("/api/regime/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert body["current_state"] in ("Bear", "Sideways", "Bull")
    assert set(body["transition_matrix"]) == {"Bear", "Sideways", "Bull"}
    for row in body["transition_matrix"].values():
        assert pytest.approx(sum(row.values()), abs=1e-4) == 1.0
    assert pytest.approx(sum(body["stationary_distribution"].values()), abs=1e-4) == 1.0
    assert len(body["recent_history"]) > 0
    # naive_backtest may be None or a dict depending on synthetic data length vs min_train;
    # 400 bdays > min_train(252)+30 default, so it should be populated here.
    assert body["naive_backtest"] is not None
    assert "NOT a validated strategy" in body["naive_backtest"]["note"]


def test_regime_endpoint_404_for_unknown_symbol(client):
    resp = client.get("/api/regime/ZZZZ")
    assert resp.status_code == 404


def test_regime_endpoint_rejects_invalid_ticker_format(client):
    resp = client.get("/api/regime/not-a-ticker!")
    assert resp.status_code == 400


def test_regime_endpoint_custom_window_and_threshold(client):
    resp = client.get("/api/regime/AAPL", params={"window": 10, "threshold": 0.05, "years": 2})
    assert resp.status_code == 200
    assert resp.json()["window"] == 10
    assert resp.json()["threshold"] == 0.05
