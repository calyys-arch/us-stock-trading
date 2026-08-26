"""Commission is not proportional to notional, and the backtest assumed it was.

These pin the published IBKR Pro Fixed schedule -- $0.005/share, $1.00 floor,
1% cap with the cap overriding the floor -- because every conclusion about how
many instruments this strategy can afford to hold rests on it.
"""
from __future__ import annotations

import math

import pytest

from python.portfolio.broker_costs import (
    MIN_PER_ORDER, basket_commission, commission, commission_bps,
    min_notional_for_cost_ceiling,
)


def test_the_published_footnote_example_reproduces_exactly():
    """IBKR's own worked example: 10 shares of a $0.20 stock is charged $0.02.

    Per-share gives $0.05, the floor says $1.00, and the 1% cap of $0.02 beats
    both. Getting the precedence backwards would overstate small-order cost 50x.
    """
    assert commission(10, 0.20) == pytest.approx(0.02)


def test_the_floor_binds_across_the_whole_retail_range():
    # A $817 position -- what $100k across 120 instruments produces.
    assert commission(8.17, 100.0) == pytest.approx(MIN_PER_ORDER)
    # Still binding at $8,000: 80 shares * 0.005 = $0.40 < $1.00.
    assert commission(80, 100.0) == pytest.approx(MIN_PER_ORDER)
    # Stops binding at 200 shares, where per-share reaches exactly $1.00.
    assert commission(200, 100.0) == pytest.approx(1.00)
    assert commission(400, 100.0) == pytest.approx(2.00)


def test_the_same_floor_is_a_different_cost_depending_on_order_size():
    """Why the no-trade band has to be a dollar amount, not a percentage."""
    assert commission_bps(8.17, 100.0) == pytest.approx(12.24, abs=0.05)
    assert commission_bps(1.0, 100.0) == pytest.approx(100.0, abs=0.5)
    # Under $100 of notional the cap takes over and the ratio pins at 100bps.
    assert commission_bps(0.5, 100.0) == pytest.approx(100.0, abs=0.5)
    # Big enough that the per-share rate takes over: 0.005/100 = 0.5bps.
    assert commission_bps(2000, 100.0) == pytest.approx(0.5, abs=0.01)


def test_a_cheap_share_price_cannot_be_traded_cheaply_at_any_size():
    """In the proportional regime the ratio is per_share/price -- independent of
    order size and inversely proportional to price. So a low-priced name is
    expensive to trade no matter how much of it you buy."""
    assert commission_bps(1_000_000, 2.0) == pytest.approx(25.0, abs=0.1)
    assert commission_bps(1_000_000, 10.0) == pytest.approx(5.0, abs=0.05)
    # At $2 a share nothing clears a 10bps ceiling, at any size.
    assert math.isinf(min_notional_for_cost_ceiling(2.0, ceiling_bps=10))
    # At $100 the proportional rate is 0.5bps, so the ceiling is reachable and
    # the binding constraint is the floor: $1.00 / 10bps = $1,000.
    assert min_notional_for_cost_ceiling(100.0, ceiling_bps=10) \
        == pytest.approx(1_000.0)


def test_the_band_scales_the_way_the_floor_does():
    assert min_notional_for_cost_ceiling(100.0, ceiling_bps=20) \
        == pytest.approx(500.0)
    assert min_notional_for_cost_ceiling(100.0, ceiling_bps=5) \
        == pytest.approx(2_000.0)


def test_a_basket_pays_the_floor_once_per_order_not_once_per_basket():
    """The reason 120 instruments cost 120 dollars to touch."""
    orders = {f"S{i}": (8.17, 100.0) for i in range(120)}
    assert basket_commission(orders) == pytest.approx(120.0)
    # The same money in 8 orders instead of 120.
    fewer = {f"S{i}": (122.5, 100.0) for i in range(8)}
    assert basket_commission(fewer) == pytest.approx(8.0)


def test_zero_and_degenerate_orders_cost_nothing():
    assert commission(0, 100.0) == 0.0
    assert commission(10, 0.0) == 0.0
    assert commission(-10, 100.0) == commission(10, 100.0), "sign must not matter"


def test_commission_free_routing_is_representable():
    assert commission(8.17, 100.0, per_share=0.0, minimum=0.0) == 0.0
    assert min_notional_for_cost_ceiling(100.0, 10, per_share=0.0,
                                         minimum=0.0) == 0.0
