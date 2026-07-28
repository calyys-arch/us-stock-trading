"""
CADF (Cointegrating Augmented Dickey-Fuller) test + OLS hedge ratio.

Implements Chan Ch.5-7 pairs-trading methodology exactly:

1. OLS regress y = code_a's log price on x = code_b's log price to get the
   hedge ratio beta: `y = beta * x + spread`. Log prices (not raw prices) are
   used so that `beta` is a stationary, dimensionless hedge ratio robust to
   both names having different price levels (Chan p.98, p.127).
2. Run an Augmented Dickey-Fuller test on the OLS residual (the spread). If
   the spread is stationary (ADF rejects the unit-root null), the pair is
   cointegrated — Chan calls this the "CADF" test (Engle-Granger two-step
   method) and explicitly prefers it over the eigenvector/Johansen approach
   for exactly two instruments (simpler, same result for n=2).

Look-ahead-bias note: `test_pair()` must only ever be called with a
`lookback` window of prices STRICTLY BEFORE the trading decision. The hedge
ratio and cointegration critical values are estimated in-sample and then
held fixed for the out-of-sample trading window — re-estimating beta on
each new bar using data through "today" would leak future information into
the day's signal (Chan p.130, "in-sample" vs "out-of-sample" distinction).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from ..core.types import CointegrationResult
from .half_life import ornstein_uhlenbeck_half_life


def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS y = alpha + beta*x + residual. Returns (beta, residual_series)."""
    x_with_const = add_constant(x)
    model = OLS(y, x_with_const).fit()
    beta = float(model.params[1])
    residuals = np.asarray(model.resid)
    return beta, residuals


def test_pair(
    code_a: str,
    code_b: str,
    prices_a: pd.Series,
    prices_b: pd.Series,
    computed_at: datetime | None = None,
) -> CointegrationResult:
    """Run the full CADF test for one candidate pair.

    Parameters
    ----------
    prices_a, prices_b : aligned daily close-price series (same index,
        strictly in-sample / lookback-window data only). Must have >= 60
        observations for the ADF test to be meaningful (Chan recommends
        >= 1 year of daily data for the initial cointegration screen).
    """
    if len(prices_a) != len(prices_b):
        raise ValueError("test_pair: prices_a and prices_b must be aligned (same length)")
    n = len(prices_a)
    if n < 60:
        raise ValueError(f"test_pair: need >= 60 observations, got {n}")

    log_a = np.log(prices_a.to_numpy(dtype=float))
    log_b = np.log(prices_b.to_numpy(dtype=float))

    beta, residuals = _ols_hedge_ratio(log_a, log_b)

    # ADF test on the spread (residual). autolag="AIC" is the standard choice
    # per Chan's worked examples and the broader cointegration literature.
    adf_stat, _p_value, _usedlag, _nobs, crit_values, _icbest = adfuller(
        residuals, autolag="AIC"
    )

    spread_mean = float(np.mean(residuals))
    spread_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

    half_life = ornstein_uhlenbeck_half_life(residuals)

    return CointegrationResult(
        code_a=code_a,
        code_b=code_b,
        hedge_ratio=beta,
        cadf_tstat=float(adf_stat),
        cadf_crit_1pct=float(crit_values["1%"]),
        cadf_crit_5pct=float(crit_values["5%"]),
        cadf_crit_10pct=float(crit_values["10%"]),
        is_cointegrated_5pct=float(adf_stat) < float(crit_values["5%"]),
        half_life_days=half_life,
        spread_mean=spread_mean,
        spread_std=spread_std,
        computed_at=computed_at or datetime.utcnow(),
        lookback_days=n,
    )


def current_spread(price_a: float, price_b: float, hedge_ratio: float) -> float:
    """Spread value (in log-price space) for live/backtest z-score computation.
    Must use the SAME transform (log price, OLS residual convention) as
    test_pair() above, or the z-score will be computed against the wrong
    scale."""
    return float(np.log(price_a) - hedge_ratio * np.log(price_b))


def spread_z_score(spread: float, spread_mean: float, spread_std: float) -> float:
    if spread_std <= 0:
        return 0.0
    return (spread - spread_mean) / spread_std
