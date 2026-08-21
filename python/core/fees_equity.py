"""
US equity transaction cost model — replaces forex-trading's fees.py
(pip_size, quote_to_hkd, forex_swap_cost — none of which apply here).

Components (Chan Ch.5 "Execution Frictions" + current SEC/FINRA/IBKR
schedules; all rates are configurable constants below since regulatory fee
schedules change periodically — see comments for source/vintage):

  1. Broker commission — IBKR US stocks tiered pricing: $0.005/share,
     $1.00 minimum per order, capped at 1% of trade value (IBKR published
     schedule, 2026 vintage).
  2. SEC Section 31 fee — SELL-side only, currently $8.00 per $1,000,000 of
     proceeds (i.e. 0.0008%), rounded up to the nearest cent per SEC's
     published fee rate (adjusted periodically by the SEC; verify current
     rate before going live).
  3. FINRA Trading Activity Fee (TAF) — SELL-side only, $0.000166/share,
     capped at $8.30 per trade (FINRA schedule, 2026 vintage).
  4. Short-borrow cost — for short positions only: annualized borrow rate
     (from ReferenceData / broker locate quote) * notional * holding_days/360.
  5. Slippage / market impact — modeled as a function of order size relative
     to ADV (square-root market impact model, Chan Ch.5 p.85-90 style):
     impact_bps = impact_coefficient * sqrt(order_notional / adv_dollars).
     This is a MODEL, not a guarantee — backtest reports must show the
     sensitivity of results to this coefficient (see chan_guards tests).
  6. Bid-ask half-spread — crossing the spread costs roughly half the quoted
     spread per fill, paid on BOTH the entry and the exit. Priced in bps of
     traded notional, the same unit
     python/backtest/intraday_engine.py's `half_spread_bps` uses and the same
     unit scripts/calibrate_slippage_spreads.py calibrates from real captured
     L2 depth (backtests/reports/calibrated_spreads.json). Defaults to 0.0
     here so pre-existing callers keep their previous (spread-free) behavior
     unchanged; callers that model spread must pass it EXPLICITLY and state
     the assumed value in their report.

All rates below are named constants (not magic numbers) so they can be
updated in one place when regulatory schedules change.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Regulatory / broker fee constants (verify before going live) ────────────
IBKR_PER_SHARE_COMMISSION = 0.005
IBKR_MIN_COMMISSION_PER_ORDER = 1.00
IBKR_MAX_COMMISSION_PCT_OF_TRADE = 0.01

SEC_SECTION_31_FEE_RATE = 8.00 / 1_000_000.0    # per $ of SELL proceeds
FINRA_TAF_PER_SHARE = 0.000166                   # SELL-side only
FINRA_TAF_MAX_PER_TRADE = 8.30

DEFAULT_ANNUAL_BORROW_RATE_EASY_TO_BORROW = 0.0075  # 0.75%/yr for liquid names
DEFAULT_MARKET_IMPACT_COEFFICIENT_BPS = 10.0        # bps per sqrt(participation)


@dataclass
class TradeCostBreakdown:
    commission: float
    sec_fee: float
    finra_taf: float
    borrow_cost: float
    market_impact: float
    half_spread: float = 0.0

    @property
    def total(self) -> float:
        return (self.commission + self.sec_fee + self.finra_taf
                + self.borrow_cost + self.market_impact + self.half_spread)


def commission(shares: int, price: float) -> float:
    shares = abs(shares)
    if shares == 0:
        return 0.0
    raw = shares * IBKR_PER_SHARE_COMMISSION
    trade_value = shares * price
    capped = min(raw, trade_value * IBKR_MAX_COMMISSION_PCT_OF_TRADE)
    return max(capped, IBKR_MIN_COMMISSION_PER_ORDER)


def sec_section_31_fee(shares: int, price: float, side: str) -> float:
    if side != "sell":
        return 0.0
    proceeds = abs(shares) * price
    return round(proceeds * SEC_SECTION_31_FEE_RATE, 2)


def finra_taf(shares: int, side: str) -> float:
    if side != "sell":
        return 0.0
    return min(abs(shares) * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX_PER_TRADE)


def short_borrow_cost(
    notional: float,
    holding_days: float,
    annual_rate: float = DEFAULT_ANNUAL_BORROW_RATE_EASY_TO_BORROW,
) -> float:
    if notional <= 0 or holding_days <= 0:
        return 0.0
    return notional * annual_rate * (holding_days / 360.0)


def market_impact(
    order_notional: float,
    adv_dollars: float,
    impact_coefficient_bps: float = DEFAULT_MARKET_IMPACT_COEFFICIENT_BPS,
) -> float:
    """Square-root market-impact model. Returns the estimated impact cost in
    USD. If `adv_dollars` is unknown/zero, returns 0.0 and callers should
    treat that as "impact unmodeled" rather than "impact is zero" (surface
    this distinction in reports — see backtests/reports/us_equity_health_check.md)."""
    if adv_dollars <= 0 or order_notional <= 0:
        return 0.0
    participation = order_notional / adv_dollars
    impact_bps = impact_coefficient_bps * (participation ** 0.5)
    return order_notional * (impact_bps / 10_000.0)


def half_spread_cost(notional: float, half_spread_bps: float) -> float:
    """Cost of crossing half the quoted bid-ask spread on `notional` of
    traded value. Zero `half_spread_bps` means "spread not modeled" — that
    is a real modeling gap, not a zero-cost market, and callers should say
    so in their report rather than leaving it implicit."""
    if notional <= 0 or half_spread_bps <= 0:
        return 0.0
    return notional * (half_spread_bps / 10_000.0)


def round_trip_cost(
    shares: int,
    entry_price: float,
    exit_price: float,
    is_short: bool = False,
    holding_days: float = 0.0,
    adv_dollars: float = 0.0,
    annual_borrow_rate: float = DEFAULT_ANNUAL_BORROW_RATE_EASY_TO_BORROW,
    half_spread_bps: float = 0.0,
) -> TradeCostBreakdown:
    """Full round-trip cost for one position: entry commission+fees, exit
    commission+fees, borrow cost (if short), market impact on both legs, and
    (when `half_spread_bps` is supplied) the bid-ask half-spread paid on both
    legs. `is_short` determines which leg is the SELL leg for SEC/FINRA fee
    purposes (open-short = sell first, close-short = buy to cover)."""
    entry_side = "sell" if is_short else "buy"
    exit_side = "buy" if is_short else "sell"

    entry_notional = abs(shares) * entry_price
    exit_notional = abs(shares) * exit_price

    total_commission = commission(shares, entry_price) + commission(shares, exit_price)
    total_sec = sec_section_31_fee(shares, entry_price, entry_side) + sec_section_31_fee(shares, exit_price, exit_side)
    total_taf = finra_taf(shares, entry_side) + finra_taf(shares, exit_side)
    borrow = short_borrow_cost(entry_notional, holding_days, annual_borrow_rate) if is_short else 0.0
    impact = market_impact(entry_notional, adv_dollars) + market_impact(exit_notional, adv_dollars)
    spread = (half_spread_cost(entry_notional, half_spread_bps)
              + half_spread_cost(exit_notional, half_spread_bps))

    return TradeCostBreakdown(
        commission=total_commission,
        sec_fee=total_sec,
        finra_taf=total_taf,
        borrow_cost=borrow,
        market_impact=impact,
        half_spread=spread,
    )
