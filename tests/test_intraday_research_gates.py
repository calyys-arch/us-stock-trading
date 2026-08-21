"""Official intraday research GO: survival AND, Monte Carlo is a warning."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_intraday_backtest import _pf_clears, assemble_intraday_gates


def _assemble(*, stress_pf_ok=True, survival_pf_ok=True, mc_ok=True, wfo_go=True,
              min_trades_ok=True, edge_pf_ok=True, oos_drawdown_ok=True,
              has_oos_trades=True):
    return assemble_intraday_gates(
        wfo_go=wfo_go,
        oos_drawdown_ok=oos_drawdown_ok,
        has_oos_trades=has_oos_trades,
        min_trades_ok=min_trades_ok,
        survival_pf_ok=survival_pf_ok,
        edge_pf_ok=edge_pf_ok,
        mc_ok=mc_ok,
        stress_mult=1.5,
        stress_pf_ok=stress_pf_ok,
    )


def test_monte_carlo_does_not_flip_intraday_go():
    hard, soft = _assemble(mc_ok=False, wfo_go=False, min_trades_ok=False, edge_pf_ok=False)
    assert all(hard.values())
    assert "monte_carlo_p5_sharpe" not in hard
    assert soft["monte_carlo_p5_sharpe"] is False
    assert soft["wfo_go"] is False
    assert soft["min_trades_per_oos_fold"] is False
    assert soft["edge_profit_factor"] is False


def test_survival_pf_failure_is_nogo():
    hard, _soft = _assemble(survival_pf_ok=False)
    assert hard["cost_adjusted_profit_factor"] is False
    assert not all(hard.values())


def test_stress_gate_is_pf_not_net_pnl():
    hard, _soft = _assemble(stress_pf_ok=False)
    assert "stress_slippage_1.5x_pf_ge_1" in hard
    assert not any(name.endswith("net_positive") for name in hard)
    assert hard["stress_slippage_1.5x_pf_ge_1"] is False
    assert not all(hard.values())


def test_drawdown_and_has_trades_remain_hard():
    hard_dd, _ = _assemble(oos_drawdown_ok=False)
    hard_tr, _ = _assemble(has_oos_trades=False)
    assert hard_dd["oos_drawdown_within_limit"] is False
    assert hard_tr["has_oos_trades"] is False
    assert not all(hard_dd.values())
    assert not all(hard_tr.values())


def test_pf_clears_inf_and_rejects_nan():
    assert _pf_clears(float("inf"), 1.0) is True
    assert _pf_clears(1.0, 1.0) is True
    assert _pf_clears(0.99, 1.0) is False
    assert _pf_clears(float("nan"), 1.0) is False
    assert _pf_clears(None, 1.0) is False
