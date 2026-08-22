"""Official intraday research GO: consistency AND, only sample-size and the
1.1 edge PF bar are warnings."""
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


def test_wfo_and_monte_carlo_are_hard():
    """Promoted 2026-08-22. These two are the only checks that test whether an
    edge holds up CONSISTENTLY rather than merely survives, so neither may be
    demoted back to a warning without reopening the hole recorded in
    backtests/reports/sizing_wfo.md."""
    hard_mc, _ = _assemble(mc_ok=False)
    hard_wfo, _ = _assemble(wfo_go=False)
    assert hard_mc["monte_carlo_p5_sharpe"] is False
    assert hard_wfo["wfo_go"] is False
    assert not all(hard_mc.values())
    assert not all(hard_wfo.values())


def test_only_sample_size_and_edge_pf_are_warnings():
    hard, soft = _assemble(min_trades_ok=False, edge_pf_ok=False)
    assert all(hard.values())
    assert soft == {"min_trades_per_oos_fold": False, "edge_profit_factor": False}


def test_shrinking_size_cannot_buy_a_go_on_the_in_sample_gate_alone():
    """The concrete failure this gate set was rewritten to prevent.

    auction_reclaim_5m at 0.75x position size cleared every one of the four
    original hard gates — the swing vote being the stress gate, the lone
    in-sample member (it replays the chosen params over the window they were
    chosen on) — and was handed a GO on 1-of-8 passing WFO folds, mean OOS
    Sharpe -6.95 and Monte Carlo p5 -2.22. Shrinking a position shrinks the
    measurement noise around an expectancy; it cannot flip that expectancy's
    sign, so the decision rule must not be satisfiable that way.
    """
    hard, _ = _assemble(
        oos_drawdown_ok=True, has_oos_trades=True, survival_pf_ok=True,
        stress_pf_ok=True,      # what 0.75x sizing bought
        wfo_go=False, mc_ok=False,  # what it did not move
    )
    assert not all(hard.values())


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
