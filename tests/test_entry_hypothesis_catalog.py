"""15 entry hypotheses catalog — independent gates, mixed chart kinds."""
from __future__ import annotations

from pathlib import Path

from python.analytics.entry_hypothesis_import import collect_existing_cells
from python.analytics.volume_route_policy import (
    CANONICAL_GATES,
    list_combinations,
    list_routes,
    load_catalog,
)

ENTRY = Path("configs/entry_hypothesis_tests.yaml")


def test_entry_catalog_has_fifteen_hypotheses():
    catalog = load_catalog(ENTRY)
    routes = {r.name: r for r in list_routes(catalog)}
    assert len(routes) == 15
    assert catalog["combination_mode"] == "single_gates"
    assert [c.name for c in list_combinations(catalog)] == [
        "wfo", "dd", "trades", "sample", "pf", "mc", "stress",
    ]
    assert routes["pairs_trading"].kind == "daily"
    assert routes["pairs_trading"].charts == (0,)
    assert routes["vsa_no_demand"].charts == (1, 5, 15)
    assert routes["orb_vwap"].kind == "intraday_native_1m"
    assert routes["orb_vwap"].charts == (1,)
    assert routes["absorption_breakout"].charts == (1, 5, 15)
    assert routes["daily_range_breakout"].kind == "daily"


def test_import_existing_official_cells():
    cells = collect_existing_cells()
    expected = {
        "pairs_trading_daily",
        "xsection_mean_reversion_daily",
        "daily_range_breakout_daily",
        "vsa_no_demand_5m",
        "obv_divergence_5m",
        "auction_reclaim_5m",
        "absorption_breakout_1m",
        "l2_absorption_1m",
        "sweep_reclaim_1m",
        "fvg_retest_1m",
        "orb_vwap_1m",
        "orb_vwap_regime_1m",
        "vwap_band_fade_1m",
        "vp_breakout_1m",
    }
    assert expected <= set(cells)
    vsa = cells["vsa_no_demand_5m"]
    assert vsa["route_gates"]["cost_adjusted_profit_factor"] is True
    assert vsa["route_gates"]["stress_slippage_1.5x_pf_ge_1"] is False
    assert set(vsa["route_gates"]) == set(CANONICAL_GATES)
    assert cells["pairs_trading_daily"]["route_gates"]["has_oos_trades"] is True
    assert cells["pairs_trading_daily"]["decision"] == "NO-GO"


def test_daily_verdict_treats_chart_zero_as_daily():
    from scripts.run_entry_hypothesis_gates import _verdict

    rows = [
        {
            "route": "pairs_trading",
            "chart_minutes": 0,
            "combination": "dd",
            "decision": "GO",
        }
    ]
    assert _verdict(rows, "pairs_trading", 0, "dd") == "PASS"
