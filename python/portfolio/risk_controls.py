"""Risk management controls for Strategy V1.

This module provides dynamic risk adjustments on top of the base volatility
targeting logic. The base strategy already uses realized volatility to scale
exposure; these controls add market-regime awareness and position-level limits.

DYNAMIC EXPOSURE ADJUSTMENT. The base 15% volatility target assumes normal market
conditions. During stress periods (high VIX), even a 15%-vol portfolio can suffer
outsized drawdowns due to correlation spikes and gap risk. This module scales
exposure down when VIX is elevated:

    VIX < 20:  normal exposure (as calculated by vol target)
    VIX 20-30: reduce by 20%
    VIX 30-40: reduce by 40%
    VIX > 40:  reduce by 60%

The thresholds are based on historical VIX quartiles and GFC stress test results.

CONCENTRATION LIMITS. The bucket-equal weighting gives each bucket 25%, and within
the US equity bucket (4 buckets total, so 25% of book), single names can drift to
large weights due to price appreciation. Limits:

    Single name: max 5% of book (after rebalancing)
    Single bucket: max 35% of book (allows 10pp drift from 25% target)

LIMIT ORDER LOGIC. Market orders pay the full spread (~4bps). Limit orders can
save 1-2 bps by sitting at the midpoint, but risk non-fills. For Strategy V1's
monthly rebalancing, the trade-off favors limits:
  - Rebalance orders are patient (not time-critical)
  - Non-fills after 1 day can be retried as market orders
  - Estimated savings: 1.5 bps per filled limit = ~$15 per $100k rebalance

This module provides the limit-order pricing and fill simulation logic.
"""
from typing import Literal
import pandas as pd
import numpy as np


# VIX-based exposure scaling
VIX_BRACKETS = [
    (0, 20, 1.00),   # Normal: no adjustment
    (20, 30, 0.80),  # Elevated: -20%
    (30, 40, 0.60),  # High: -40%
    (40, 999, 0.40), # Extreme: -60%
]


def vix_exposure_scalar(vix_level: float) -> float:
    """Return exposure scalar (0.4 to 1.0) based on VIX level.
    
    Args:
        vix_level: Current VIX index value (e.g., 18.5)
    
    Returns:
        Multiplier for target exposure. 1.0 = no adjustment, 0.4 = 60% reduction.
    
    Examples:
        >>> vix_exposure_scalar(15.0)
        1.0
        >>> vix_exposure_scalar(25.0)
        0.8
        >>> vix_exposure_scalar(45.0)
        0.4
    """
    for low, high, scalar in VIX_BRACKETS:
        if low <= vix_level < high:
            return scalar
    return 1.0  # Fallback for NaN or out-of-bounds


def adjust_exposure_for_vix(
    base_exposure: pd.Series,
    vix: pd.Series,
) -> pd.Series:
    """Apply VIX-based scaling to a base exposure series.
    
    Args:
        base_exposure: Target exposure from volatility targeting (0.0 to 1.0)
        vix: VIX index values, aligned to same dates as base_exposure
    
    Returns:
        Adjusted exposure series (always <= base_exposure)
    
    Example:
        If base exposure is 0.95 and VIX is 35, adjusted exposure is 0.95 * 0.6 = 0.57
    """
    scalars = vix.apply(vix_exposure_scalar)
    return (base_exposure * scalars).clip(upper=1.0)


def check_concentration(
    holdings: dict[str, float],
    prices: pd.Series,
    bucket_of: dict[str, str],
    max_single_name_pct: float = 0.05,
    max_bucket_pct: float = 0.35,
) -> dict:
    """Check if current holdings violate concentration limits.
    
    Args:
        holdings: Current share counts {symbol: shares}
        prices: Current prices {symbol: price}
        bucket_of: Bucket classification {symbol: bucket_name}
        max_single_name_pct: Maximum weight for any single name (default 5%)
        max_bucket_pct: Maximum weight for any bucket (default 35%)
    
    Returns:
        {
            "within_limits": bool,
            "violators": list of (symbol, weight, limit) for any breaches,
            "bucket_weights": dict of {bucket: weight},
            "max_single_name": (symbol, weight),
            "max_bucket": (bucket, weight),
        }
    """
    # Calculate position values
    values = {sym: holdings.get(sym, 0) * prices.get(sym, np.nan)
              for sym in holdings}
    total = sum(values.values())
    
    if total == 0:
        return {
            "within_limits": True,
            "violators": [],
            "bucket_weights": {},
            "max_single_name": (None, 0.0),
            "max_bucket": (None, 0.0),
        }
    
    # Single name weights
    name_weights = {sym: val / total for sym, val in values.items()}
    
    # Bucket weights
    bucket_values = {}
    for sym, val in values.items():
        bucket = bucket_of.get(sym, "unknown")
        bucket_values[bucket] = bucket_values.get(bucket, 0.0) + val
    bucket_weights = {b: v / total for b, v in bucket_values.items()}
    
    # Find violations
    violators = []
    for sym, w in name_weights.items():
        if w > max_single_name_pct:
            violators.append((sym, w, max_single_name_pct, "single_name"))
    
    for bucket, w in bucket_weights.items():
        if w > max_bucket_pct:
            violators.append((bucket, w, max_bucket_pct, "bucket"))
    
    max_name = max(name_weights.items(), key=lambda x: x[1], default=(None, 0.0))
    max_bucket = max(bucket_weights.items(), key=lambda x: x[1], default=(None, 0.0))
    
    return {
        "within_limits": len(violators) == 0,
        "violators": violators,
        "bucket_weights": bucket_weights,
        "max_single_name": max_name,
        "max_bucket": max_bucket,
    }


def limit_order_price(
    side: Literal["buy", "sell"],
    mid_price: float,
    spread_bps: float = 4.0,
    aggression: float = 0.5,
) -> float:
    """Calculate limit order price as a fraction through the spread.
    
    Args:
        side: "buy" or "sell"
        mid_price: Current mid price (or last trade)
        spread_bps: Typical bid-ask spread in basis points (default 4bps)
        aggression: How far through the spread to sit (0.0 = passive, 1.0 = market)
                    0.5 = sit at midpoint
    
    Returns:
        Limit price
    
    Example:
        If mid = $100, spread = 4bps (0.04%), aggression = 0.5:
          buy:  $100 * (1 - 0.04%/2 * 0.5) = $99.98
          sell: $100 * (1 + 0.04%/2 * 0.5) = $100.02
    """
    half_spread = (spread_bps / 10_000.0) / 2.0
    
    if side == "buy":
        # Buy at mid - (1-aggression)*half_spread
        # aggression=0 -> buy at bid, aggression=1 -> buy at mid, aggression=0.5 -> in between
        return mid_price * (1 - half_spread * (1 - aggression))
    else:  # sell
        return mid_price * (1 + half_spread * (1 - aggression))


def simulate_limit_fills(
    orders: list[tuple[str, Literal["buy", "sell"], float, float]],
    price_range: pd.DataFrame,
    aggression: float = 0.5,
) -> dict:
    """Simulate whether limit orders would have filled given a price range.
    
    Args:
        orders: List of (symbol, side, limit_price, shares)
        price_range: DataFrame with columns ['high', 'low'] for each symbol during the period
        aggression: Aggression level used to set the limit (needed to infer fill probability)
    
    Returns:
        {
            "filled": list of (symbol, side, shares),
            "unfilled": list of (symbol, side, shares),
            "fill_rate": float (fraction filled),
        }
    
    Fill logic:
        - Buy order at price P fills if intraday low <= P
        - Sell order at price P fills if intraday high >= P
    
    This is optimistic (assumes you get filled at your limit if price touches it).
    Real fills depend on order book depth and timing.
    """
    filled = []
    unfilled = []
    
    for symbol, side, limit_price, shares in orders:
        if symbol not in price_range.index:
            unfilled.append((symbol, side, shares))
            continue
        
        if side == "buy":
            if price_range.loc[symbol, "low"] <= limit_price:
                filled.append((symbol, side, shares))
            else:
                unfilled.append((symbol, side, shares))
        else:  # sell
            if price_range.loc[symbol, "high"] >= limit_price:
                filled.append((symbol, side, shares))
            else:
                unfilled.append((symbol, side, shares))
    
    fill_rate = len(filled) / len(orders) if orders else 1.0
    
    return {
        "filled": filled,
        "unfilled": unfilled,
        "fill_rate": fill_rate,
    }
