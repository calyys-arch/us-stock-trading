"""Tests for portfolio risk controls."""
import pytest
import pandas as pd
import numpy as np
from python.portfolio.risk_controls import (
    vix_exposure_scalar,
    adjust_exposure_for_vix,
    check_concentration,
    limit_order_price,
    simulate_limit_fills,
)


def test_vix_scalar_returns_1_for_normal_markets():
    assert vix_exposure_scalar(15.0) == 1.0
    assert vix_exposure_scalar(19.9) == 1.0


def test_vix_scalar_reduces_exposure_in_stress():
    assert vix_exposure_scalar(25.0) == 0.8
    assert vix_exposure_scalar(35.0) == 0.6
    assert vix_exposure_scalar(50.0) == 0.4


def test_vix_adjustment_never_exceeds_base_exposure():
    base = pd.Series([0.95, 0.80, 0.70], index=[1, 2, 3])
    vix = pd.Series([15.0, 25.0, 45.0], index=[1, 2, 3])
    
    adjusted = adjust_exposure_for_vix(base, vix)
    
    assert adjusted.iloc[0] == pytest.approx(0.95)  # 0.95 * 1.0
    assert adjusted.iloc[1] == pytest.approx(0.64)  # 0.80 * 0.8
    assert adjusted.iloc[2] == pytest.approx(0.28)  # 0.70 * 0.4


def test_concentration_check_passes_for_balanced_book():
    # 20 names, each 5% of book, spread across 4 buckets (5 names per bucket = 25% per bucket)
    holdings = {f"STOCK_{i}": 1.0 for i in range(20)}
    prices = pd.Series({f"STOCK_{i}": 100.0 for i in range(20)})  # Each $100
    bucket_of = {}
    buckets = ["equity", "bonds", "commodities", "intl"]
    for i in range(20):
        bucket_of[f"STOCK_{i}"] = buckets[i % 4]
    
    result = check_concentration(holdings, prices, bucket_of, 
                                max_single_name_pct=0.05, max_bucket_pct=0.35)
    
    # Each name is exactly 5% (at limit), each bucket is 25% (well under 35%)
    assert result["within_limits"] is True
    assert len(result["violators"]) == 0


def test_concentration_check_flags_oversized_name():
    # One name is 60% of the book
    holdings = {"NVDA": 100.0, "TLT": 10.0}
    prices = pd.Series({"NVDA": 150.0, "TLT": 100.0})  # NVDA = $15k, TLT = $1k
    bucket_of = {"NVDA": "equity", "TLT": "bonds"}
    
    result = check_concentration(holdings, prices, bucket_of, max_single_name_pct=0.05)
    
    assert result["within_limits"] is False
    assert any(v[0] == "NVDA" for v in result["violators"])
    assert result["max_single_name"][0] == "NVDA"
    assert result["max_single_name"][1] > 0.05


def test_concentration_check_flags_oversized_bucket():
    holdings = {f"STOCK_{i}": 1.0 for i in range(10)}
    prices = pd.Series({f"STOCK_{i}": 100.0 for i in range(10)})
    bucket_of = {f"STOCK_{i}": "equity" for i in range(10)}
    
    # All 10 stocks are in equity bucket = 100% of book, violates 35% limit
    result = check_concentration(holdings, prices, bucket_of, max_bucket_pct=0.35)
    
    assert result["within_limits"] is False
    assert any(v[0] == "equity" and v[3] == "bucket" for v in result["violators"])


def test_limit_order_at_midpoint():
    # aggression=0.5 means sit halfway between bid and mid (save 25% of the spread)
    buy = limit_order_price("buy", 100.0, spread_bps=4.0, aggression=0.5)
    sell = limit_order_price("sell", 100.0, spread_bps=4.0, aggression=0.5)
    
    # spread = 4bps = 0.0004, half = 0.0002
    # buy: mid * (1 - half_spread * (1 - aggression))
    #    = 100 * (1 - 0.0002 * 0.5) = 100 * 0.9999 = 99.99
    # sell: mid * (1 + half_spread * (1 - aggression))
    #     = 100 * (1 + 0.0002 * 0.5) = 100 * 1.0001 = 100.01
    assert buy == pytest.approx(99.99, rel=1e-4)
    assert sell == pytest.approx(100.01, rel=1e-4)


def test_limit_order_at_market_price():
    # aggression=1.0 means cross the spread (effectively a market order)
    buy = limit_order_price("buy", 100.0, spread_bps=4.0, aggression=1.0)
    sell = limit_order_price("sell", 100.0, spread_bps=4.0, aggression=1.0)
    
    # Should be very close to mid (crossing the spread)
    assert buy == pytest.approx(100.0, rel=1e-6)
    assert sell == pytest.approx(100.0, rel=1e-6)


def test_limit_fills_when_price_touches_limit():
    orders = [
        ("SPY", "buy", 499.0, 10.0),   # Buy at 499
        ("TLT", "sell", 101.0, 10.0),  # Sell at 101
    ]
    
    price_range = pd.DataFrame({
        "high": [502.0, 102.0],
        "low": [498.0, 100.0],
    }, index=["SPY", "TLT"])
    
    result = simulate_limit_fills(orders, price_range)
    
    # SPY touched 498, below our 499 limit -> filled
    # TLT touched 102, above our 101 limit -> filled
    assert result["fill_rate"] == 1.0
    assert len(result["filled"]) == 2


def test_limit_doesnt_fill_when_price_misses():
    orders = [
        ("SPY", "buy", 495.0, 10.0),   # Want to buy at 495
        ("TLT", "sell", 105.0, 10.0),  # Want to sell at 105
    ]
    
    price_range = pd.DataFrame({
        "high": [502.0, 102.0],
        "low": [498.0, 100.0],
    }, index=["SPY", "TLT"])
    
    result = simulate_limit_fills(orders, price_range)
    
    # SPY only went to 498, didn't touch 495 -> unfilled
    # TLT only went to 102, didn't touch 105 -> unfilled
    assert result["fill_rate"] == 0.0
    assert len(result["unfilled"]) == 2


def test_concentration_with_empty_holdings():
    result = check_concentration({}, pd.Series(), {})
    assert result["within_limits"] is True
    assert result["max_single_name"] == (None, 0.0)
