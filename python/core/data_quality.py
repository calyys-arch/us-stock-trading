"""
Data-quality guards for historical/live price series.

The forex-trading timestamp-unit bug (docs/lessons_from_forex_trading.md #3)
went undetected for an entire research cycle partly because nothing
automatically flagged the resulting distorted holding-time distribution.
This module's 4-sigma extreme-move check is a similarly cheap, general
tripwire: any daily return more than `sigma_threshold` standard deviations
from the series' own rolling mean is very likely a bad print (split not
applied, decimal error, stale/duplicated quote) rather than a genuine market
move, and should be surfaced for manual review before a backtest run is
trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def flag_extreme_moves(
    returns: pd.Series,
    sigma_threshold: float = 4.0,
    rolling_window: int = 60,
) -> pd.Series:
    """Return a boolean Series (same index as `returns`) that is True where
    the return deviates more than `sigma_threshold` rolling standard
    deviations from the rolling mean. The first `rolling_window` points
    (insufficient history) are never flagged."""
    rolling_mean = returns.rolling(rolling_window, min_periods=rolling_window).mean()
    rolling_std = returns.rolling(rolling_window, min_periods=rolling_window).std(ddof=1)
    z = (returns - rolling_mean) / rolling_std.replace(0, np.nan)
    flagged = z.abs() > sigma_threshold
    return flagged.fillna(False)


def quality_report(prices: pd.Series, sigma_threshold: float = 4.0) -> dict:
    """Summarize data-quality findings for one instrument's price series.
    Intended for use in scripts/run_backtest.py pre-flight checks and
    docs/us_equity_health_check.md generation."""
    returns = prices.pct_change().dropna()
    flagged = flag_extreme_moves(returns, sigma_threshold=sigma_threshold)
    n_flagged = int(flagged.sum())
    return {
        "n_observations": int(len(prices)),
        "n_extreme_moves_flagged": n_flagged,
        "extreme_move_dates": [str(d) for d in flagged[flagged].index.tolist()],
        "max_abs_return": float(returns.abs().max()) if len(returns) else 0.0,
        "n_zero_or_negative_prices": int((prices <= 0).sum()),
        "n_duplicated_consecutive_prices": int((prices.diff() == 0).sum()),
    }
