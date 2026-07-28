"""
Ornstein-Uhlenbeck half-life estimation (Chan p.52-54, p.99, p.131).

Model: dy(t) = lambda * (y(t) - mu) * dt + dW(t)

Discretized: y(t) - y(t-1) = lambda * y(t-1) + const + noise
(the OLS intercept absorbs `-lambda*mu`, so it can be dropped for half-life
purposes — only lambda, the mean-reversion speed, is needed).

Half-life = ln(2) / -lambda

Chan explicitly prefers this over ad-hoc "count bars until price crosses the
mean" because it is a well-defined estimator that uses the ENTIRE spread
series rather than just a handful of crossing events, and it directly gives
a natural holding-period / lookback-window parameter for the trading
strategy (e.g. use the half-life as an approximate reversion timescale when
sizing the rolling window for zscore computation, per Chan Ch.7 pairs
example).
"""
from __future__ import annotations

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def ornstein_uhlenbeck_half_life(spread: np.ndarray) -> float:
    """Return the O-U half-life in the same time unit as the input series'
    sampling interval (e.g. days, if `spread` is a daily series). Returns
    -1.0 if the series is not mean-reverting (lambda >= 0), which callers
    must treat as "not tradeable" (see CointegrationResult.is_tradeable)."""
    spread = np.asarray(spread, dtype=float)
    if len(spread) < 3:
        return -1.0

    y_lag = spread[:-1]
    y_diff = spread[1:] - spread[:-1]

    x = add_constant(y_lag)
    model = OLS(y_diff, x).fit()
    lam = float(model.params[1])  # coefficient on y_lag

    if lam >= 0:
        return -1.0  # non-mean-reverting (random walk or explosive)

    half_life = -np.log(2.0) / lam
    return float(half_life)
