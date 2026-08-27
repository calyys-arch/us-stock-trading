"""Commission is not proportional to notional, and the two brokers on the table
disagree about the one thing that matters most at retail size.

These pin both published schedules, because every conclusion about whether it
is worth maintaining constituent weights rests on which account executes.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from python.portfolio.broker_costs import (
    FUTU_HK_FIXED, IBKR_PRO_FIXED, NO_COMMISSION, basket_commission,
    commission, commission_bps, min_notional_for_cost_ceiling,
)

IBKR, FUTU = IBKR_PRO_FIXED, FUTU_HK_FIXED


def test_the_published_ibkr_footnote_example_reproduces_exactly():
    """IBKR's own worked example: 10 shares of a $0.20 stock is charged $0.02.

    Per-share gives $0.05, the floor says $1.00, and the 1% cap of $0.02 beats
    both. Getting the precedence backwards would overstate small-order cost 50x.
    """
    assert commission(10, 0.20, IBKR) == pytest.approx(0.02)


def test_futu_resolves_the_same_conflict_the_opposite_way():
    """"若與每筆最低收費標準衝突，以最低收費標準為準" -- the floor wins.

    Same trade, same two limbs, opposite answer, and this single difference is
    what decides whether a rebalance is affordable.
    """
    assert commission(10, 0.20, FUTU) == pytest.approx(1.99 + 0.003 * 10)


def test_futu_stacks_two_minimums_so_its_floor_is_twice_ibkrs():
    # An $817 position, what $100k across 120 instruments produces.
    assert commission(8.17, 100.0, IBKR) == pytest.approx(1.00)
    # 0.99 commission + 1.00 platform + 8.17 * 0.003 settlement.
    assert commission(8.17, 100.0, FUTU) == pytest.approx(1.99 + 0.02451)


def test_the_gap_that_decides_whether_rebalancing_is_worth_it():
    """A $25 drift correction: capped at IBKR, floored at Futu, 8x apart."""
    shares, price = 0.25, 100.0
    assert commission(shares, price, IBKR) == pytest.approx(0.25)
    assert commission(shares, price, FUTU) == pytest.approx(1.99 + 0.00075)
    assert commission_bps(shares, price, FUTU) > \
        7 * commission_bps(shares, price, IBKR)


def test_each_floor_stops_binding_where_its_per_share_rate_overtakes_it():
    assert commission(200, 100.0, IBKR) == pytest.approx(1.00)
    assert commission(400, 100.0, IBKR) == pytest.approx(2.00)
    # Futu: commission crosses at 202 shares, platform at 200.
    assert commission(400, 100.0, FUTU) == pytest.approx(
        0.0049 * 400 + 0.005 * 400 + 0.003 * 400)


def test_the_same_floor_is_a_different_cost_depending_on_order_size():
    """Why a no-trade band has to be a dollar amount, not a percentage drift."""
    assert commission_bps(8.17, 100.0, IBKR) == pytest.approx(12.24, abs=0.05)
    assert commission_bps(1.0, 100.0, IBKR) == pytest.approx(100.0, abs=0.5)
    # Big enough that the per-share rate takes over: 0.005/100 = 0.5bps.
    assert commission_bps(2000, 100.0, IBKR) == pytest.approx(0.5, abs=0.01)


def test_a_cheap_share_price_cannot_be_traded_cheaply_at_any_size():
    """In the proportional regime the ratio is per_share/price -- independent of
    order size and inversely proportional to price."""
    assert commission_bps(1_000_000, 2.0, IBKR) == pytest.approx(25.0, abs=0.1)
    assert commission_bps(1_000_000, 10.0, IBKR) == pytest.approx(5.0, abs=0.05)
    assert math.isinf(min_notional_for_cost_ceiling(2.0, 10, IBKR))


def test_the_no_trade_band_is_where_the_charge_falls_to_the_ceiling():
    # IBKR: $1.00 floor / 10bps = $1,000.
    assert min_notional_for_cost_ceiling(100.0, 10, IBKR) \
        == pytest.approx(1_000.0, rel=1e-3)
    assert min_notional_for_cost_ceiling(100.0, 20, IBKR) \
        == pytest.approx(500.0, rel=1e-3)
    # Futu: a $1.99 floor plus settlement needs roughly twice the order to
    # clear the same ceiling.
    futu_band = min_notional_for_cost_ceiling(100.0, 10, FUTU)
    assert futu_band > 2 * min_notional_for_cost_ceiling(100.0, 10, IBKR)


def test_the_band_actually_delivers_the_ceiling_it_promises():
    """The bisection has to be right at the boundary, not just monotone."""
    for schedule in (IBKR, FUTU):
        for price in (12.0, 100.0, 480.0):
            for ceiling in (5.0, 10.0, 50.0):
                band = min_notional_for_cost_ceiling(price, ceiling, schedule)
                if math.isinf(band):
                    continue
                at = commission_bps(band / price, price, schedule)
                assert at <= ceiling + 1e-6, (schedule.name, price, ceiling)
                just_under = commission_bps(band * 0.98 / price, price,
                                            schedule)
                assert just_under > ceiling, "band must be the true boundary"


def test_a_basket_pays_the_floor_once_per_order_not_once_per_basket():
    """The reason 120 instruments cost real money to touch."""
    orders = {f"S{i}": (8.17, 100.0) for i in range(120)}
    assert basket_commission(orders, IBKR) == pytest.approx(120.0)
    assert basket_commission(orders, FUTU) == pytest.approx(120 * 2.01451)
    # The same money in 8 orders instead of 120.
    fewer = {f"S{i}": (122.5, 100.0) for i in range(8)}
    assert basket_commission(fewer, IBKR) == pytest.approx(8.0)


def test_zero_and_degenerate_orders_cost_nothing():
    for schedule in (IBKR, FUTU):
        assert commission(0, 100.0, schedule) == 0.0
        assert commission(10, 0.0, schedule) == 0.0
        assert commission(-10, 100.0, schedule) == commission(10, 100.0,
                                                              schedule)


def test_commission_free_routing_is_representable():
    assert commission(8.17, 100.0, NO_COMMISSION) == 0.0
    assert min_notional_for_cost_ceiling(100.0, 10, NO_COMMISSION) == 0.0


def test_the_configured_broker_is_the_one_the_ledger_actually_charges():
    """The ledger and the order sheet each load the schedule independently, so
    nothing structurally stops them charging different firms. Pin them to the
    config and to each other."""
    import scripts.paper_track_v1 as ledger
    import scripts.strategy_v1_positions as sheet
    from python.portfolio.broker_costs import load_schedule

    configured = load_schedule()
    assert configured.name == ledger.load_schedule().name
    assert configured.name == sheet.load_schedule().name


def test_an_unknown_broker_is_refused_rather_than_quietly_defaulted():
    """Silently falling back to IBKR would understate a Futu account's costs by
    a factor of eight on small orders."""
    from python.portfolio.broker_costs import load_schedule

    cfg = tmp = Path(__file__).parent / "_broker_probe.yaml"
    cfg.write_text("broker: definitely-not-a-broker\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not one of"):
            load_schedule(cfg)
        cfg.write_text("broker: futu\n", encoding="utf-8")
        assert load_schedule(cfg) is FUTU
    finally:
        tmp.unlink()
