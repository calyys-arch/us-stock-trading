"""Market-conditional live gate catalog — combinations of any length."""
from __future__ import annotations

import pandas as pd

from python.analytics.gate_policy import (
    FAMILY_CONTINUATION,
    FAMILY_MEAN_REVERSION,
    REGIME_MR,
    REGIME_TREND,
    REGIME_UNDECIDED,
    VOL_HIGH,
    VOL_LOW,
    VOL_NORMAL,
    classify_volatility,
    evaluate,
    live_order_permitted,
    list_situations,
    regime_from_pairs_gate,
    research_requires_all_gates,
    select_situation,
)


def test_research_go_does_not_require_all_seven_gates():
    assert research_requires_all_gates() is False


def test_catalog_has_variable_length_combinations():
    rows = list_situations()
    names = [row["name"] for row in rows]
    assert "research_full" not in names
    lengths = {len(row["gates"]) for row in rows}
    assert 1 in lengths
    assert 7 not in lengths
    assert len(lengths) >= 3


def test_unmatched_situation_is_refusal_not_research_go():
    """Continuation on an MR tape (or MR on a trend tape) used to fall
    through to research_full and pretend paper entry was research GO."""
    miss = evaluate("absorption_breakout", REGIME_MR, VOL_NORMAL)
    assert miss.allowed is False
    assert miss.gate_set == "no_matching_situation"
    assert "research" not in miss.reason
    assert select_situation(REGIME_MR, VOL_NORMAL, FAMILY_CONTINUATION) is None

    pairs_on_trend = evaluate("pairs_trading", REGIME_TREND, VOL_NORMAL)
    assert pairs_on_trend.allowed is False
    assert pairs_on_trend.gate_set == "no_matching_situation"
    assert "research" not in pairs_on_trend.reason


def test_crisis_is_one_gate_when_vol_high():
    s = select_situation(REGIME_TREND, VOL_HIGH, FAMILY_CONTINUATION)
    assert s is not None
    assert s.name == "crisis"
    assert s.gates == ("max_oos_drawdown",)


def test_quiet_trend_is_five_gates():
    s = select_situation(REGIME_TREND, VOL_LOW, FAMILY_CONTINUATION)
    assert s is not None
    assert s.name == "quiet_trend"
    assert len(s.gates) == 5


def test_trend_normal_is_four_gates():
    s = select_situation(REGIME_TREND, VOL_NORMAL, FAMILY_CONTINUATION)
    assert s is not None
    assert s.name == "trend_normal"
    assert len(s.gates) == 4


def test_mr_normal_is_five_gates():
    s = select_situation(REGIME_MR, VOL_NORMAL, FAMILY_MEAN_REVERSION)
    assert s is not None
    assert s.name == "mr_high_evidence"
    assert len(s.gates) == 5


def test_undecided_tape_blocks():
    assert evaluate("absorption_breakout", REGIME_UNDECIDED).allowed is False
    assert evaluate("pairs_trading", REGIME_UNDECIDED).allowed is False


def test_absorption_allowed_in_normal_trend():
    d = evaluate("absorption_breakout", REGIME_TREND, VOL_NORMAL)
    assert d.allowed is True
    assert d.gate_set == "trend_normal"
    assert len(d.required) == 4


def test_absorption_blocked_in_quiet_trend_by_sharpe():
    d = evaluate("absorption_breakout", REGIME_TREND, VOL_LOW)
    assert d.gate_set == "quiet_trend"
    assert d.allowed is False
    assert d.results["min_oos_sharpe"] is False


def test_pf_below_one_fails_when_pf_is_in_the_set():
    cards = {
        "absorption_breakout": {
            REGIME_TREND: {
                "cost_adjusted_profit_factor": 0.38,
                "max_oos_drawdown": 0.05,
                "stress_slippage_2x_net_positive": True,
                "has_oos_trades": True,
            }
        }
    }
    d = evaluate("absorption_breakout", REGIME_TREND, VOL_NORMAL, scorecards=cards)
    assert d.allowed is False
    assert d.results["cost_adjusted_profit_factor"] is False


def test_crisis_allows_if_only_drawdown_clears():
    cards = {
        "absorption_breakout": {
            REGIME_TREND: {
                "cost_adjusted_profit_factor": 0.38,
                "max_oos_drawdown": 0.05,
                "stress_slippage_2x_net_positive": False,
                "has_oos_trades": True,
            }
        }
    }
    d = evaluate("absorption_breakout", REGIME_TREND, VOL_HIGH, scorecards=cards)
    assert d.gate_set == "crisis"
    assert len(d.required) == 1
    assert d.allowed is True


def test_pairs_allowed_in_mr_regime():
    ok, reason = live_order_permitted("pairs_trading", REGIME_MR, VOL_NORMAL)
    assert ok is True, reason


def test_regime_from_pairs_gate():
    assert regime_from_pairs_gate(True) == REGIME_MR
    assert regime_from_pairs_gate(False) == REGIME_TREND
    assert regime_from_pairs_gate(None) == REGIME_UNDECIDED


def test_classify_volatility_high_vs_low():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    prices = [100.0]
    for i in range(1, 400):
        prices.append(prices[-1] * (1.01 if i % 2 == 0 else 0.99))
    s = pd.Series(prices, index=idx)

    quiet = s.copy()
    last = float(quiet.iloc[-21])
    for i in range(20):
        quiet.iloc[-20 + i] = last * (1.0 + 0.0001 * (1 if i % 2 == 0 else -1))
    assert classify_volatility(quiet) == VOL_LOW

    wild = s.copy()
    last = float(wild.iloc[-21])
    for i in range(20):
        last = last * (1.08 if i % 2 == 0 else 0.92)
        wild.iloc[-20 + i] = last
    assert classify_volatility(wild) == VOL_HIGH
