"""
Chan Ch.3 data-snooping-bias guards, made mechanically checkable.

Chan's own rule of thumb (p.34-35): the number of historical data points
needed to reliably validate a strategy grows with the number of free
parameters it has — a strategy with `k` free parameters needs roughly
`252 * k` trading days (i.e. k YEARS of daily data) of out-of-sample or
walk-forward-tested history before its backtest Sharpe ratio can be trusted
at all; fewer than that and you are almost certainly fitting noise.

This module turns that rule into two checkable functions used by
tests/test_chan_guards.py and scripts/run_backtest.py's pre-flight checks:

  - count_free_parameters(): parse a strategy's config block and count
    genuine TUNABLE parameters (excluding housekeeping keys like `enabled`,
    `auto_execute`, `allow_overnight`).
  - sufficient_sample_size(): the 252*k day-count check itself.

MAX_FREE_PARAMETERS = 5 is this project's hard ceiling (both MVP strategies
are designed to use 3), matching architecture-rules.mdc.
"""
from __future__ import annotations

MAX_FREE_PARAMETERS = 5

# Keys that configure BEHAVIOR/HOUSEKEEPING/OPERATIONAL settings, not a
# statistically-fit free parameter of the strategy's signal-generation
# formula (Chan's parameter-count discipline is about the number of knobs
# that could be curve-fit to historical data, not every operational dial):
#   - enabled/auto_execute/allow_overnight/flatten_buffer_minutes: pure
#     on/off or scheduling housekeeping.
#   - coint_lookback_days/revalidate_every_days: the DATA WINDOW used to
#     estimate cointegration, not a signal threshold — Chan uses "as much
#     history as available, revalidated periodically" rather than optimizing
#     this window's length against returns.
#   - notional_per_leg: position SIZING (capital allocation), governed by
#     Kelly (python/core/kelly.py) + RiskEngine caps, not a fit parameter of
#     the entry/exit signal itself.
_NON_PARAMETER_KEYS = {
    "enabled", "auto_execute", "allow_overnight", "flatten_buffer_minutes",
    "coint_lookback_days", "revalidate_every_days", "notional_per_leg",
}


def count_free_parameters(strategy_config: dict) -> int:
    return sum(1 for k in strategy_config if k not in _NON_PARAMETER_KEYS)


def check_max_parameters(strategy_config: dict, max_allowed: int = MAX_FREE_PARAMETERS) -> tuple[bool, int]:
    n = count_free_parameters(strategy_config)
    return n <= max_allowed, n


def sufficient_sample_size(total_trading_days: int, num_free_parameters: int, days_per_parameter: int = 252) -> bool:
    """Chan's rule of thumb: >= 252 trading days of tested history PER free
    parameter. Returns False if the backtest/validation window is too short
    to trust the result."""
    required = days_per_parameter * max(num_free_parameters, 1)
    return total_trading_days >= required


def required_days(num_free_parameters: int, days_per_parameter: int = 252) -> int:
    return days_per_parameter * max(num_free_parameters, 1)
