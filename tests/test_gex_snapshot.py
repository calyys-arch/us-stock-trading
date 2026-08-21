"""Naive GEX computation + cache round-trip."""
from __future__ import annotations

from datetime import date

from python.data.gex_cache import load_gex_env, load_gex_snapshot, save_gex_snapshot
from python.microstructure.gex import compute_naive_gex, regime_from_net


def _call_heavy_chain(as_of: date = date(2026, 8, 18)) -> list[dict]:
    """ATM puts + a call wall at 110. Enough OI/IV that gamma is defined."""
    expiry = date(2026, 8, 22)
    calls = [
        {"strike": 100.0, "open_interest": 200, "gamma": 0.01},
        {"strike": 110.0, "open_interest": 8000, "gamma": 0.02},
        {"strike": 120.0, "open_interest": 100, "gamma": 0.005},
    ]
    puts = [
        {"strike": 90.0, "open_interest": 400, "gamma": 0.01},
        {"strike": 100.0, "open_interest": 3000, "gamma": 0.02},
        {"strike": 110.0, "open_interest": 200, "gamma": 0.005},
    ]
    return [{"expiry": expiry, "calls": calls, "puts": puts}]


def test_regime_from_net_sign():
    assert regime_from_net(1.0) == "positive_gamma"
    assert regime_from_net(-1.0) == "negative_gamma"
    assert regime_from_net(0.0) == "neutral"


def test_compute_naive_gex_walls_and_regime():
    snap = compute_naive_gex("QQQ", 100.0, _call_heavy_chain(), as_of=date(2026, 8, 18), source="synthetic")
    assert snap is not None
    assert snap.symbol == "QQQ"
    assert snap.call_wall == 110.0
    assert snap.put_wall == 100.0
    assert snap.call_gex > 0
    assert snap.put_gex < 0
    assert snap.regime in {"positive_gamma", "negative_gamma", "neutral"}
    assert "2026-08-22" in snap.expiries_used


def test_missing_or_empty_chain_is_none():
    assert compute_naive_gex("QQQ", 0.0, _call_heavy_chain()) is None
    assert compute_naive_gex("QQQ", 100.0, []) is None
    assert compute_naive_gex("QQQ", 100.0, [{
        "expiry": date(2026, 8, 22),
        "calls": [{"strike": 100.0, "open_interest": 0, "implied_volatility": 0.2}],
        "puts": [],
    }], as_of=date(2026, 8, 18)) is None


def test_save_and_load_roundtrip(tmp_path):
    snap = compute_naive_gex("QQQ", 100.0, _call_heavy_chain(), as_of=date(2026, 8, 18), source="synthetic")
    path = save_gex_snapshot(snap, cache_dir=tmp_path)
    assert path.exists()
    loaded = load_gex_snapshot("QQQ", "2026-08-18", cache_dir=tmp_path)
    assert loaded is not None
    assert loaded.net_gex == snap.net_gex
    assert loaded.call_wall == snap.call_wall
    assert loaded.regime == snap.regime


def test_load_missing_is_none(tmp_path):
    assert load_gex_snapshot("QQQ", "2026-08-18", cache_dir=tmp_path) is None
    assert load_gex_env("AAPL", "2026-08-18", cache_dir=tmp_path) is None
