"""
Strategy B — Daily Cross-Sectional Mean Reversion (Chan Ch.3, "Are Stock
Returns Mean Reverting?", the Khandani-Lo-style contrarian strategy,
pp.46-55 and worked Examples 3.7/3.8).

Idea: on any given day, stocks that underperformed the cross-sectional
average yesterday tend to partially revert (outperform) today, and vice
versa. This is deliberately NOT a per-instrument model — the signal for one
stock ("did it underperform?") is only defined relative to the rest of the
universe on the same day, which is exactly why this needs the
PortfolioStrategy interface (python/core/strategies/portfolio_base.py)
instead of BaseStrategy.

Chan's exact weighting formula (p.47, eq. 3.7), used unmodified here:

    w_i(t) = -( r_i(t-1) - r_bar(t-1) ) / C(t-1)

where r_i(t-1) is stock i's return over the lookback window ending at the
PRIOR close, r_bar(t-1) is the cross-sectional average return that same day,
and C(t-1) = sum_i | r_i(t-1) - r_bar(t-1) | is a normalizer that makes the
portfolio's gross leverage exactly 1 (before the gross_leverage_target
scale-up and any RiskEngine caps are applied downstream). This construction
is automatically DOLLAR-NEUTRAL (sum of w_i = 0) because it is proportional
to a demeaned quantity.

Strategy is INTRADAY ONLY per the user's confirmed constraint: entered at
today's open (right after the prior day's return is fully known — no
look-ahead) and flattened by the close the same day (enforced by
execution_gateway's EOD flatten, not by this class).

Free parameters (Chan Ch.3 discipline: keep this <= 5):
  1. lookback_days   (return window, Chan's simplest example uses 1 day)
  2. gross_leverage_target (e.g. 1.0 = fully invested both sides)
  3. min_universe_size (skip a day entirely if too few eligible names —
     avoids a noisy/degenerate cross-section on illiquid days)
That's 3; well under the 5-parameter budget.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..types import PortfolioTarget
from .portfolio_base import PortfolioStrategy


class CrossSectionalMeanReversionStrategy(PortfolioStrategy):
    name = "xsection_mean_reversion"

    def __init__(
        self,
        lookback_days: int = 1,
        gross_leverage_target: float = 1.0,
        min_universe_size: int = 20,
    ) -> None:
        self.lookback_days = lookback_days
        self.gross_leverage_target = gross_leverage_target
        self.min_universe_size = min_universe_size

    def evaluate(
        self,
        as_of_data: pd.DataFrame,
        as_of: datetime,
        universe: list[str],
    ) -> PortfolioTarget:
        """`as_of_data` must be indexed by (date, code) with a 'close' column
        and contain ONLY rows strictly before `as_of` (caller-enforced — see
        tests/test_lookahead_bias.py). We use the last `lookback_days + 1`
        closes per code to compute the trailing return ending at the most
        recent available close (i.e. the prior trading day's close relative
        to `as_of`).

        Performance note: `as_of_data` can span the ENTIRE history up to
        `as_of` (it grows every day the vectorized backtester walks
        forward). Looping `.xs(code, level=1)` per code against that
        ever-growing frame is O(days * codes * history) and was measured to
        turn a ~1000-day / 40-name demo backtest into a 15+ minute run. We
        instead trim to just the last `lookback_days + 1` calendar dates
        BEFORE doing any per-code work, then do one vectorized `unstack`
        so the whole cross-section is computed with array ops, not a
        Python loop of pandas cross-sections.
        """
        eligible_set = set(universe)
        all_dates = as_of_data.index.get_level_values(0)
        unique_dates = np.unique(all_dates)
        needed = self.lookback_days + 1
        if len(unique_dates) < needed:
            return PortfolioTarget(
                strategy=self.name, as_of=as_of, weights={},
                metadata={"skipped": "insufficient_return_history", "n_names": 0},
            )
        recent_dates = unique_dates[-needed:]
        recent = as_of_data.loc[all_dates >= recent_dates[0]]

        wide = recent["close"].unstack(level=1)
        wide = wide.reindex(index=recent_dates)
        cols = [c for c in wide.columns if c in eligible_set]
        eligible = cols
        if len(eligible) < self.min_universe_size:
            return PortfolioTarget(
                strategy=self.name, as_of=as_of, weights={},
                metadata={"skipped": "universe_too_small", "n_eligible": len(eligible)},
            )
        wide = wide[cols]

        p_now = wide.iloc[-1]
        p_then = wide.iloc[0]
        valid_mask = p_now.notna() & p_then.notna() & (p_then > 0)
        returns = ((p_now[valid_mask] / p_then[valid_mask]) - 1.0).to_dict()

        if len(returns) < self.min_universe_size:
            return PortfolioTarget(
                strategy=self.name, as_of=as_of, weights={},
                metadata={"skipped": "insufficient_return_history", "n_names": len(returns)},
            )

        codes = list(returns.keys())
        r = np.array([returns[c] for c in codes])
        r_bar = float(np.mean(r))
        deviations = r - r_bar
        normalizer = float(np.sum(np.abs(deviations)))

        if normalizer <= 0:
            return PortfolioTarget(
                strategy=self.name, as_of=as_of, weights={},
                metadata={"skipped": "zero_dispersion"},
            )

        raw_weights = -deviations / normalizer  # Chan eq. 3.7; sums to 0 by construction
        scaled_weights = raw_weights * self.gross_leverage_target

        weights = {code: float(w) for code, w in zip(codes, scaled_weights)}

        return PortfolioTarget(
            strategy=self.name,
            as_of=as_of,
            weights=weights,
            metadata={
                "n_names": len(weights),
                "cross_sectional_mean_return": r_bar,
                "gross_leverage": float(np.sum(np.abs(scaled_weights))),
                "net_exposure": float(np.sum(scaled_weights)),
            },
        )
