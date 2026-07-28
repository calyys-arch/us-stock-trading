"""
PortfolioStrategy — cross-sectional strategy interface.

This is the interface Strategy B (daily cross-sectional mean reversion,
Chan Ch.3 Example 3.7/3.8) implements. It is evaluated ONCE PER DAY across
the whole universe (not once per instrument per tick like BaseStrategy),
because the signal for any one stock ("is it in the bottom decile of prior-
day returns?") is only meaningful relative to the rest of the universe.

Look-ahead-bias discipline (Chan Ch.3 p.51-52): `evaluate()` receives only
`as_of_data`, a DataFrame indexed by (date, code) that the caller must have
already truncated to data available strictly BEFORE the trading decision
(i.e. prior trading day's close). The strategy must not be given, and must
never reach for, same-day or future data when computing target weights.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from ..types import PortfolioTarget


class PortfolioStrategy(ABC):
    name: str = "portfolio_base"
    max_free_parameters: int = 5

    @abstractmethod
    def evaluate(
        self,
        as_of_data: pd.DataFrame,
        as_of: datetime,
        universe: list[str],
    ) -> PortfolioTarget:
        """Compute target weights for `as_of` using only data known before it.

        Parameters
        ----------
        as_of_data : DataFrame indexed by (date, code) with at least a
            'close' column (and any other columns the strategy needs, e.g.
            'sector', 'market_cap', 'adv_20d_dollars'). MUST contain no rows
            for `as_of` or later — enforced by the caller
            (backtest/vector_engine.py or the live daily scheduler), but
            strategies should still defensively assert this in tests
            (see tests/test_lookahead_bias.py).
        as_of : the trading day these weights will be executed on (at the
            open, per Chan Example 3.8).
        universe : list of eligible codes for this day (already filtered for
            price >= $5, no earnings today, has locate if short, etc. by the
            universe builder / risk engine — NOT this strategy's job).

        Returns
        -------
        PortfolioTarget with `weights` summing to at most 1.0 gross exposure
        per side (risk engine re-checks and can further scale down).
        """
        ...
