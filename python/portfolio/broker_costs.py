"""What a broker actually charges, as opposed to a flat basis-point assumption.

The backtests in this repo charge ONE_WAY_COST_BPS = 4bps on notional traded.
That is a reasonable model for spread and impact, and it is completely wrong
about commission, because commission is not proportional to notional. It is
billed per ORDER, with a floor, and the floor is what dominates at retail size.

TWO BROKERS, AND THE DIFFERENCE BETWEEN THEM IS NOT THE HEADLINE RATE. Per share
IBKR and Futu are within a hundredth of a cent of each other. What separates
them is how the per-order minimum interacts with the percentage cap:

    IBKR   the 1% cap OVERRIDES the floor. Their own worked example: 10 shares
           of a $0.20 stock is charged $0.02, not the $1.00 minimum.
    Futu   the floor OVERRIDES the 0.5% cap -- "若與每筆最低收費標準衝突，
           以最低收費標準為準".

For a full position that distinction is invisible. For the small orders a
monthly rebalance produces it is everything: a $25 correction costs $0.25 at
IBKR and $1.99 at Futu, eight times more. Any conclusion about whether it is
worth maintaining constituent weights depends entirely on which of these the
account actually trades through, so the schedule is a parameter rather than a
constant.

Futu also stacks two separate per-order minimums -- commission $0.99 and
platform fee $1.00 -- so its effective floor is $1.99 against IBKR's $1.00.

WHAT IS MODELLED AND WHAT IS NOT. Commission, platform fee and the per-share
settlement fee are modelled. FINRA's TAF is not: it is $0.000195 a share on
sells only, which on this book is under two cents a rebalance. The SEC fee was
abolished on 2025-05-14 and is zero. ETFs are charged as stocks at both
brokers; an ETF's expense ratio needs no modelling here because it is deducted
from NAV daily and is therefore already inside any return computed from prices.

Rates verified 2026-08-26 against interactivebrokers.com/en/pricing and
futuhk.com/support/topic2_283.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FeeSchedule:
    """One broker's per-order charge.

    `components` is a tuple of (per_share_rate, per_order_minimum) pairs, one
    per line item the broker bills separately. They are kept separate rather
    than summed because each carries its own minimum, and the rates cross their
    respective floors at slightly different share counts.

    `cap_overrides_floor` is the whole reason this is a dataclass and not two
    numbers. It decides which side wins when a tiny order's percentage cap
    falls below the per-order minimum, and it is the difference between a
    rebalance being nearly free and being unaffordable.
    """

    name: str
    components: tuple[tuple[float, float], ...]
    cap_fraction: float
    cap_overrides_floor: bool
    # Third-party per-share charges with no minimum of their own, so they scale
    # cleanly and are never capped.
    third_party_per_share: float = 0.0
    note: str = ""

    @property
    def min_per_order(self) -> float:
        return sum(minimum for _rate, minimum in self.components)

    @property
    def per_share(self) -> float:
        return sum(rate for rate, _minimum in self.components) \
            + self.third_party_per_share

    def charge(self, shares: float, price: float) -> float:
        shares, price = abs(float(shares)), abs(float(price))
        if shares <= 0 or price <= 0:
            return 0.0
        notional = shares * price
        billed = sum(max(minimum, rate * shares)
                     for rate, minimum in self.components)
        if self.cap_fraction > 0:
            capped = min(billed, notional * self.cap_fraction)
            billed = capped if self.cap_overrides_floor \
                else max(capped, self.min_per_order)
        return billed + self.third_party_per_share * shares

    def min_notional_for_ceiling(self, price: float,
                                 ceiling_bps: float) -> float:
        """Smallest order whose total charge stays within `ceiling_bps` of the
        value it moves. This is the no-trade band.

        Solved by bisection rather than algebra. The closed form has to case
        split on which component's floor binds, whether the cap is active and
        which way `cap_overrides_floor` points, and an earlier attempt to write
        it out got the IBKR case right and would have got Futu wrong. The
        charge is monotone in size over the region that matters, so searching
        is both shorter and harder to get wrong.
        """
        price = abs(float(price))
        if price <= 0 or ceiling_bps <= 0:
            return float("inf")
        ceiling = ceiling_bps / 10_000.0
        if self.per_share <= 0 and self.min_per_order <= 0:
            return 0.0

        def ratio(notional: float) -> float:
            return self.charge(notional / price, price) / notional

        # Above this the per-share rate alone decides, and it only improves
        # with size up to that point, so if it fails here it fails everywhere.
        ceiling_at_scale = self.per_share / price
        if ceiling_at_scale > ceiling:
            return float("inf")

        lo, hi = 1e-6, max(price, 1.0)
        while ratio(hi) > ceiling:
            hi *= 2
            if hi > 1e12:
                return float("inf")
        for _ in range(200):
            mid = (lo + hi) / 2
            if ratio(mid) > ceiling:
                lo = mid
            else:
                hi = mid
        return hi


IBKR_PRO_FIXED = FeeSchedule(
    name="IBKR Pro Fixed",
    components=((0.005, 1.00),),
    cap_fraction=0.01,
    cap_overrides_floor=True,
    note="USD 0.005/share, min USD 1.00, max 1% of trade value (cap wins)",
)

FUTU_HK_FIXED = FeeSchedule(
    name="Futu Securities (HK), fixed platform plan",
    # Commission and platform fee are billed separately, each with its own
    # minimum, so the effective floor is $1.99 rather than $1.00.
    components=((0.0049, 0.99), (0.005, 1.00)),
    cap_fraction=0.005,
    cap_overrides_floor=False,
    third_party_per_share=0.003,  # settlement fee, both sides
    note="0.0049+0.005/share, min 0.99+1.00, cap 0.5% but the floor wins; "
         "settlement 0.003/share. TAF (sell, 0.000195/share) omitted as "
         "immaterial; SEC fee abolished 2025-05-14.",
)

# Commission-free routing. IBKR Lite is US-residents-only and pays for order
# flow, so the cost moves into the spread rather than disappearing; the 4bps
# spread assumption then carries the whole bill.
NO_COMMISSION = FeeSchedule(name="none", components=(), cap_fraction=0.0,
                            cap_overrides_floor=True)

SCHEDULES = {"ibkr": IBKR_PRO_FIXED, "futu": FUTU_HK_FIXED,
             "none": NO_COMMISSION}

BROKER_CONFIG_PATH = Path("configs/broker.yaml")


def load_schedule(config_path: str | Path = BROKER_CONFIG_PATH) -> FeeSchedule:
    """Whose fees to charge, from configs/broker.yaml's `broker:` key.

    Deliberately not defaulted to whichever gateway happens to be running.
    The gateway supplies prices; the fee schedule has to match the account that
    would actually fill the order, and on this machine those are two different
    firms -- the IB Gateway session is a paper account (DU-prefixed) that
    cannot execute, while the Futu account is funded.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            name = str((yaml.safe_load(f) or {}).get("broker", "ibkr")).lower()
    except FileNotFoundError:
        name = "ibkr"
    if name not in SCHEDULES:
        raise ValueError(f"configs/broker.yaml broker: {name!r} is not one of "
                         f"{sorted(SCHEDULES)}")
    return SCHEDULES[name]


def commission(shares: float, price: float,
               schedule: FeeSchedule = IBKR_PRO_FIXED) -> float:
    return schedule.charge(shares, price)


def commission_bps(shares: float, price: float,
                   schedule: FeeSchedule = IBKR_PRO_FIXED) -> float:
    """Charge as basis points of the notional traded.

    Useful because the answer is wildly non-constant: at IBKR the same $1.00
    floor is 12bps on an $817 order and 100bps on a $50 one.
    """
    notional = abs(float(shares)) * abs(float(price))
    if notional <= 0:
        return 0.0
    return schedule.charge(shares, price) / notional * 10_000


def min_notional_for_cost_ceiling(price: float, ceiling_bps: float,
                                  schedule: FeeSchedule = IBKR_PRO_FIXED
                                  ) -> float:
    return schedule.min_notional_for_ceiling(price, ceiling_bps)


def basket_commission(orders: dict[str, tuple[float, float]],
                      schedule: FeeSchedule = IBKR_PRO_FIXED) -> float:
    """Total charge for a basket, as {symbol: (shares, price)}.

    Summed per order rather than on aggregate notional, because the floor
    applies once per order and that is the entire point.
    """
    return sum(schedule.charge(shares, price)
               for shares, price in orders.values())
