"""VSA / OBV route strategies — one independent test per gate."""
from __future__ import annotations

import pytest

from python.analytics.volume_route_policy import (
    CANONICAL_GATES,
    GATE_SHORT,
    collect_route_gates,
    combination_name,
    list_combinations,
    list_routes,
    list_strategies,
    load_catalog,
    score_all,
    single_gate_combinations,
    strategy_id,
    time_stop_for,
)


def _all_pass() -> dict[str, bool]:
    return {name: True for name in CANONICAL_GATES}


def test_catalog_is_seven_single_gates():
    routes = {r.name: r for r in list_routes()}
    assert set(routes) == {"vsa", "obv"}
    assert routes["vsa"].signal == "vsa_no_demand"
    assert routes["obv"].signal == "obv_divergence"
    assert routes["vsa"].charts == (1, 5, 15)
    combos = list_combinations()
    assert len(combos) == 7
    assert [c.name for c in combos] == list(GATE_SHORT.values())
    assert all(len(c.gates) == 1 for c in combos)
    strategies = list_strategies()
    assert len(strategies) == 2 * 3 * 7
    ids = [s.strategy_id for s in strategies]
    assert len(ids) == len(set(ids))
    assert "vsa_5m_wfo" in ids
    assert "obv_15m_stress" in ids


def test_catalog_gate_names_match_canonical():
    catalog = load_catalog()
    assert catalog["combination_mode"] == "single_gates"
    assert tuple(catalog["gate_names"]) == CANONICAL_GATES


def test_time_stop_matches_chart_minutes_rule():
    assert time_stop_for(1) == 10
    assert time_stop_for(5) == 10
    assert time_stop_for(15) == 30


def test_collect_route_gates_merges_hard_soft_and_stress_alias():
    hard = {
        "oos_drawdown_within_limit": True,
        "has_oos_trades": True,
        "cost_adjusted_profit_factor": True,
        "stress_slippage_1.5x_pf_ge_1": False,
    }
    soft = {
        "wfo_go": False,
        "min_trades_per_oos_fold": True,
        "edge_profit_factor": True,
        "monte_carlo_p5_sharpe": False,
    }
    vector = collect_route_gates(hard, soft)
    assert set(vector) == set(CANONICAL_GATES)
    assert vector["wfo_go"] is False
    assert "edge_profit_factor" not in vector
    aliased = collect_route_gates(
        {"stress_slippage_1.5x_net_positive": True, "has_oos_trades": True},
        {},
    )
    assert aliased["stress_slippage_1.5x_pf_ge_1"] is True


def test_each_gate_is_scored_alone():
    vector = {name: False for name in CANONICAL_GATES}
    vector["wfo_go"] = True
    vector["oos_drawdown_within_limit"] = True
    vector["monte_carlo_p5_sharpe"] = True
    scores = {s.strategy_id: s for s in score_all(vector, route="vsa", chart_minutes=5)}
    assert len(scores) == 7
    assert scores["vsa_5m_wfo"].decision == "GO"
    assert scores["vsa_5m_dd"].decision == "GO"
    assert scores["vsa_5m_mc"].decision == "GO"
    assert scores["vsa_5m_trades"].decision == "NO-GO"
    assert scores["vsa_5m_sample"].decision == "NO-GO"
    assert scores["vsa_5m_pf"].decision == "NO-GO"
    assert scores["vsa_5m_stress"].decision == "NO-GO"


def test_score_cells_emits_seven_rows():
    from scripts.run_volume_route_strategies import _score_cells

    vector = _all_pass()
    vector["wfo_go"] = False
    cells = {
        "vsa_5m": {
            "route": "vsa",
            "signal": "vsa_no_demand",
            "chart_minutes": 5,
            "route_gates": vector,
        }
    }
    rows = _score_cells(cells, load_catalog())
    assert len(rows) == 7
    by_id = {r["strategy_id"]: r for r in rows}
    assert by_id["vsa_5m_wfo"]["decision"] == "NO-GO"
    assert by_id["vsa_5m_dd"]["decision"] == "GO"
    assert by_id["vsa_5m_trades"]["decision"] == "GO"


def test_unknown_gate_is_rejected():
    catalog = load_catalog()
    catalog = {**catalog, "gate_names": list(CANONICAL_GATES) + ["not_a_gate"]}
    with pytest.raises(ValueError, match="unknown gates"):
        list_combinations(catalog)
    assert combination_name(("wfo_go",)) == "wfo"
    assert strategy_id("obv", 1, "wfo") == "obv_1m_wfo"
    assert len(single_gate_combinations()) == 7
