"""
GET /api/signal_journal/today — backend for docs/microstructure_pivot_plan.md
§7's "今日訊號" dashboard panel. Monkeypatches
python.microstructure.signal_journal.SignalJournal's output_dir (via
dashboard.app's imported class/function) so no real data/signal_journal/
directory is touched.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dashboard.app as dashboard_app_module
from dashboard.app import app
from python.microstructure.signal_journal import SignalJournal


@pytest.fixture()
def client(tmp_path, monkeypatch):
    fixed_today = dt.date(2026, 8, 6)
    monkeypatch.setattr(dashboard_app_module, "today_et_date", lambda: fixed_today)
    # dashboard.app's endpoint calls `SignalJournal()` with no args (its
    # own default output_dir) — redirect the NAME it resolves inside that
    # module to always point at tmp_path, same convention as
    # tests/test_dashboard_regime_api.py patching price_cache_module's
    # imported function rather than reaching into fastapi internals.
    monkeypatch.setattr(dashboard_app_module, "SignalJournal", lambda: SignalJournal(output_dir=tmp_path))
    return TestClient(app), tmp_path, fixed_today


def test_empty_journal_returns_empty_list_not_error(client):
    test_client, _tmp_path, fixed_today = client
    resp = test_client.get("/api/signal_journal/today")
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == fixed_today.isoformat()
    assert body["signals"] == []
    assert body["count"] == 0


def test_returns_todays_recorded_signals(client):
    test_client, tmp_path, fixed_today = client
    journal = SignalJournal(output_dir=tmp_path)

    from python.core.types import QualifiedMicroOrder
    from python.microstructure.signals import MicroSignal

    signal = MicroSignal(
        symbol="AAPL", strategy="orb_vwap", direction="long",
        signal_time=pd.Timestamp("2026-08-06 10:00:00"),
        entry_price=100.0, stop_price=98.0, target_price=104.0,
        context={"vwap_distance_atr": 0.3},
    )
    order = QualifiedMicroOrder(
        raw=signal, qty=10, entry_limit_price=100.05, stop_price=98.0,
        stop_limit_price=97.9, target_price=104.0, gross_notional=1000.5,
        approved=True, rejection_reason=None,
    )
    journal.record(signal, order)

    resp = test_client.get("/api/signal_journal/today")
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == fixed_today.isoformat()
    assert body["count"] == 1
    assert body["signals"][0]["symbol"] == "AAPL"
    assert body["signals"][0]["strategy"] == "orb_vwap"
    assert body["signals"][0]["risk_passed"] is True
    assert body["signals"][0]["outcome"]["status"] == "pending"


def test_does_not_leak_other_days_signals(client):
    test_client, tmp_path, _fixed_today = client
    journal = SignalJournal(output_dir=tmp_path)

    from python.core.types import QualifiedMicroOrder
    from python.microstructure.signals import MicroSignal

    other_day_signal = MicroSignal(
        symbol="MSFT", strategy="sweep_reclaim", direction="short",
        signal_time=pd.Timestamp("2026-08-05 10:00:00"),
        entry_price=300.0, stop_price=305.0, target_price=290.0,
    )
    other_day_order = QualifiedMicroOrder(
        raw=other_day_signal, qty=5, entry_limit_price=299.9, stop_price=305.0,
        stop_limit_price=305.1, target_price=290.0, gross_notional=1500.0,
        approved=True, rejection_reason=None,
    )
    journal.record(other_day_signal, other_day_order)

    resp = test_client.get("/api/signal_journal/today")
    assert resp.status_code == 200
    assert resp.json()["signals"] == []
