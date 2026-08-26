"""A share-level simulation of Strategy V1, so commission can be charged honestly.

WHY THIS EXISTS. Every V1 result so far was computed on a weight-space return
series: buckets held at 25% and constituents at equal weight, reweighted every
day, charged 4bps of notional traded. That is a fine model of spread but it
cannot represent commission at all, because commission is per ORDER -- it
depends on how many separate trades you place and how big each one is, and a
weight-space series has neither.

Worse, the two cannot be bolted together. Estimating commission and subtracting
it from a daily-reweighted return series charges for a schedule while crediting
the returns of a different one. If a no-trade band suppresses an order, the
weights genuinely drift and the return series genuinely changes; you cannot get
that from a subtraction.

So this holds actual share counts, lets them drift between rebalances, and
charges each order what the broker would charge. The output is an equity curve
that already has commission inside it.

WHAT IT DELIBERATELY SHARES WITH THE LIVE LEDGER. The exposure band, the
monthly constituent reset and the bucket weighting are the same functions the
order sheet and paper ledger call. This is the backtest OF the ledger, not a
parallel idealisation of it, which is the only way the forward record can be
compared to anything.

WHAT IT STILL ASSUMES. Fills at the close with no slippage beyond the modelled
spread, no borrow, no dividend timing, no taxes. Volatility for the exposure
path is estimated from the weight-space series rather than from the drifting
book, which is a mild inconsistency retained on purpose: the live ledger sizes
its brake the same way, from the strategy's nominal return history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from python.portfolio.broker_costs import commission, min_notional_for_cost_ceiling
from python.portfolio.strategy_v1 import COLLAPSE, band_exposure, target_weights


def _rebalance_days(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """First trading day of each month."""
    frame = pd.Series(index, index=index)
    return set(frame.groupby([index.year, index.month]).first().to_numpy())


def _concentration(holdings: dict[str, float], prices: pd.Series,
                   bucket_of: dict[str, str]) -> dict[str, float]:
    """Largest single position and the bucket split, as shares of the book.

    Recorded every day because the interesting configurations barely trade:
    a no-trade band tight enough to skip the monthly reset lets weights drift
    for years, and whether that drift quietly concentrates the book into
    whatever won is not visible in return or drawdown until it reverses.
    """
    values = {s: sh * float(prices.get(s, 0.0)) for s, sh in holdings.items()}
    total = sum(values.values())
    if total <= 0:
        return {}
    out = {"max_weight": max(values.values()) / total}
    for symbol, value in values.items():
        key = "w_" + COLLAPSE[bucket_of[symbol]]
        out[key] = out.get(key, 0.0) + value / total
    return out


def simulate(
    panel: pd.DataFrame,
    bucket_of: dict[str, str],
    exposure: pd.Series,
    capital: float,
    *,
    scheme: str = "bucket",
    cost_ceiling_bps: float | None = None,
    spread_bps_one_way: float = 4.0,
    per_share: float = 0.005,
    min_per_order: float = 1.00,
    fractional: bool = False,
    exposure_band: float = 0.10,
    charge_costs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the book forward one day at a time, charging every order.

    `cost_ceiling_bps` is the no-trade band: an order whose commission would
    exceed that fraction of the value it moves is not placed, and the position
    is left to drift until the gap is worth closing. `None` places every order.

    `charge_costs=False` still prices every order and still lets the band reject
    the small ones, but does not deduct the money. That separation exists to
    answer a specific question: a band improves the result partly by saving fees
    and partly by letting weights drift, and the two have very different
    standing -- the fee saving is arithmetic, the drift is one window's luck.
    Zeroing `per_share` and `min_per_order` cannot answer it, because the band's
    own threshold is derived from those numbers, so setting them to zero
    silently removes the band as well and makes both arms the same run.

    Returns the daily rows (equity, invested, concentration) and a per-rebalance
    trade log.
    """
    marks = panel.ffill()
    days = [d for d in marks.index if d in exposure.index
            and np.isfinite(exposure.loc[d])]
    if not days:
        return pd.DataFrame(columns=["equity", "invested"]), pd.DataFrame()

    resets = _rebalance_days(pd.DatetimeIndex(days))
    holdings: dict[str, float] = {}
    cash = float(capital)
    applied: float | None = None
    # The account exists with its cash before the first trade. Recording that
    # opening row is what puts the entry cost inside the return series: without
    # it the curve starts after entry and pct_change silently drops the one
    # trade that is guaranteed to happen.
    opening = days[0] - pd.Timedelta(days=1)
    rows: dict[pd.Timestamp, dict[str, float]] = {
        opening: {"equity": float(capital), "invested": 0.0}}
    log: list[dict] = []

    for day in days:
        prices = marks.loc[day]
        held_value = sum(sh * float(prices.get(s, 0.0))
                         for s, sh in holdings.items())
        equity = cash + held_value

        want, moved = band_exposure(float(exposure.loc[day]), applied,
                                    exposure_band)
        # An exposure change and a monthly reset both mean placing orders; a
        # quiet day in the middle of a month means the book simply drifts.
        if not (moved or applied is None or day in resets):
            rows[day] = {"equity": equity, "invested": held_value,
                         **_concentration(holdings, prices, bucket_of)}
            continue

        priceable = [s for s in marks.columns if float(prices.get(s, 0.0)) > 0]
        weights = target_weights(priceable, bucket_of, scheme)
        budget = equity * want

        # The band governs correcting drift, never changing exposure. Cutting
        # exposure is the strategy's only defence against a drawdown, and the
        # orders it produces are small -- trimming every position by a tenth --
        # so any notional floor blocks exactly the trades that matter most. An
        # earlier version applied the band to all orders and cost 2.8 points of
        # CAGR on the full basket by disabling the brake.
        ceiling = None if (moved or applied is None) else cost_ceiling_bps

        orders: dict[str, float] = {}
        for symbol, weight in weights.items():
            price = float(prices[symbol])
            target = budget * weight / price
            if not fractional:
                target = float(int(target))
            delta = target - holdings.get(symbol, 0.0)
            notional = abs(delta) * price
            if notional <= 0:
                continue
            if ceiling is not None:
                floor = min_notional_for_cost_ceiling(
                    price, ceiling, per_share=per_share, minimum=min_per_order)
                if notional < floor:
                    continue
            orders[symbol] = delta

        # Liquidate anything that fell out of the priceable set entirely.
        for symbol in list(holdings):
            if symbol not in weights and holdings[symbol] != 0:
                price = float(prices.get(symbol, 0.0))
                if price > 0:
                    orders[symbol] = -holdings[symbol]

        def price_orders(book: dict[str, float]) -> float:
            total = 0.0
            for symbol, delta in book.items():
                if delta == 0:
                    continue
                px = float(prices[symbol])
                total += commission(delta, px, per_share=per_share,
                                    minimum=min_per_order)
                total += abs(delta) * px * spread_bps_one_way / 10_000.0
            return total

        priced_cost = price_orders(orders)
        cost = priced_cost if charge_costs else 0.0

        # Never borrow to pay for the trade: if the buys plus their costs
        # overrun available cash, scale the buys back proportionally.
        proceeds = sum(-d * float(prices[s]) for s, d in orders.items() if d < 0)
        buys = sum(d * float(prices[s]) for s, d in orders.items() if d > 0)
        available = cash + proceeds - cost
        if buys > available and buys > 0:
            scale = max(0.0, available / buys)
            for symbol in list(orders):
                if orders[symbol] > 0:
                    orders[symbol] *= scale
                    if not fractional:
                        orders[symbol] = float(int(orders[symbol]))
            priced_cost = price_orders(orders)
            cost = priced_cost if charge_costs else 0.0

        traded = 0.0
        for symbol, delta in orders.items():
            if delta == 0:
                continue
            price = float(prices[symbol])
            cash -= delta * price
            traded += abs(delta) * price
            holdings[symbol] = holdings.get(symbol, 0.0) + delta
            if abs(holdings[symbol]) < 1e-12:
                del holdings[symbol]
        cash -= cost

        applied = want
        log.append({
            "date": day, "exposure": want, "equity": equity,
            "n_orders": sum(1 for d in orders.values() if d != 0),
            "traded_notional": traded, "cost": cost,
            "cost_priced": priced_cost, "cash": cash,
            "reason": ("enter" if len(log) == 0
                       else "exposure" if moved else "monthly"),
        })
        after = sum(sh * float(prices.get(s, 0.0)) for s, sh in holdings.items())
        rows[day] = {"equity": cash + after, "invested": after,
                     **_concentration(holdings, prices, bucket_of)}

    daily = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return daily, pd.DataFrame(log)


def returns_of(daily: pd.DataFrame) -> pd.Series:
    return daily["equity"].pct_change().dropna()


def invested_fraction(daily: pd.DataFrame) -> float:
    """Mean share of the account actually holding something.

    A no-trade band tight enough to reject every order at a given account size
    leaves the book unbuilt, which shows up as a flat near-zero return that
    looks like a strategy result but is really a refusal to trade. This is how
    that case is told apart from a real one.
    """
    if daily.empty:
        return 0.0
    return float((daily["invested"] / daily["equity"]).mean())
