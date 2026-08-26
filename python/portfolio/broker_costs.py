"""What a broker actually charges, as opposed to a flat basis-point assumption.

The backtests in this repo charge ONE_WAY_COST_BPS = 4bps on notional traded.
That is a reasonable model for spread and impact, and it is completely wrong
about commission, because commission is not proportional to notional. IBKR Pro
Fixed charges per SHARE with a floor and a ceiling:

    USD 0.005 per share
    minimum USD 1.00 per order
    maximum 1% of trade value, and the cap overrides the floor

(interactivebrokers.com/en/pricing/commissions-stocks.php, verified 2026-08-26.
US-listed ETFs are charged identically to stocks -- there is no separate ETF
schedule. The one cost ETFs have that stocks do not, the expense ratio, needs
no modelling here because it is deducted from NAV daily and is therefore
already inside any return computed from prices.)

The floor is what matters at retail size, and it matters enormously. A flat
4bps model says a $817 order costs $0.33; the floor says $1.00, three times as
much. Spread that over a 120-instrument book rebalanced monthly and the gap
becomes the difference between a strategy that clears its costs and one that
does not.

The cap is what matters for small adjustments, which is exactly what a
rebalance produces: a $50 top-up costs $0.50, not $1.00. Any rule that decides
whether a small trade is worth making has to price it through the cap, not the
floor, or it will refuse trades that are actually cheap.
"""
from __future__ import annotations

PER_SHARE = 0.005
MIN_PER_ORDER = 1.00
MAX_FRACTION_OF_VALUE = 0.01

# IBKR Lite is commission-free on US stocks but is US-residents-only and routes
# for payment order flow, paying in spread instead of commission. Modelled as
# zero commission on the understanding that the 4bps spread assumption then
# carries the whole cost.
LITE_PER_SHARE = 0.0


def commission(shares: float, price: float, per_share: float = PER_SHARE,
               minimum: float = MIN_PER_ORDER) -> float:
    """Commission for one order under IBKR Pro Fixed.

    The cap is applied last and deliberately overrides the floor, matching the
    published footnote: 10 shares of a $0.20 stock is charged $0.02, not the
    $1.00 minimum, because 10 * 0.20 * 1% = 0.02 is smaller.
    """
    shares, price = abs(float(shares)), abs(float(price))
    if shares <= 0 or price <= 0:
        return 0.0
    notional = shares * price
    if per_share <= 0 and minimum <= 0:
        return 0.0
    return min(max(minimum, shares * per_share),
               notional * MAX_FRACTION_OF_VALUE)


def commission_bps(shares: float, price: float, **kw) -> float:
    """Commission as basis points of the notional traded.

    Useful because the answer is wildly non-constant: the same $1.00 floor is
    12bps on an $817 order and 200bps on a $50 one, which is why a no-trade
    band has to be expressed in dollars rather than in a percentage drift.
    """
    notional = abs(float(shares)) * abs(float(price))
    if notional <= 0:
        return 0.0
    return commission(shares, price, **kw) / notional * 10_000


def min_notional_for_cost_ceiling(price: float, ceiling_bps: float,
                                  per_share: float = PER_SHARE,
                                  minimum: float = MIN_PER_ORDER) -> float:
    """Smallest order whose commission stays within `ceiling_bps` of its value.

    This is the no-trade band. Below this notional an order cannot clear the
    ceiling no matter how much you want to make it, so the honest response is
    to leave the position alone until drift accumulates.

    Two regimes, and the boundary between them is where the floor stops binding:

      While the per-share charge is under the floor, commission is a flat
          `minimum`, so the ratio is minimum/notional and the band is simply
          minimum / ceiling. At $1.00 and 10bps that is $1,000 -- a threshold
          most retail rebalances never reach, which is the real finding.
      Once the per-share charge exceeds the floor, commission scales with
          notional and the ratio becomes per_share/price -- independent of size,
          and inversely proportional to share price. That works out to 0.5bps on
          a $100 share, 5bps on a $10 one and 25bps on a $2 one, so a
          low-priced name is expensive to trade at ANY size and no order clears
          a tight ceiling.

    The practical upshot is that the proportional regime is nearly free next to
    the 4bps spread assumption, and essentially all of the commission problem at
    retail size is the floor.
    """
    price = abs(float(price))
    if price <= 0 or ceiling_bps <= 0:
        return float("inf")
    ceiling = ceiling_bps / 10_000.0
    if per_share <= 0 and minimum <= 0:
        return 0.0
    # Proportional regime: ratio is per_share / price regardless of size.
    proportional = per_share / price
    if proportional > ceiling:
        return float("inf")
    from_floor = minimum / ceiling
    # Below `minimum / per_share * price` the floor binds; above it the
    # proportional rate does, and we already know that rate clears.
    floor_binds_below = minimum / per_share * price if per_share > 0 else \
        float("inf")
    return min(from_floor, floor_binds_below) if from_floor <= floor_binds_below \
        else floor_binds_below


def basket_commission(orders: dict[str, tuple[float, float]], **kw) -> float:
    """Total commission for a basket, as {symbol: (shares, price)}.

    Summed per order rather than on aggregate notional, because the floor
    applies once per order and that is the entire point.
    """
    return sum(commission(shares, price, **kw)
               for shares, price in orders.values())
