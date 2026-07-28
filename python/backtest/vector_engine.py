"""
Vectorized pandas backtester for PortfolioStrategy (cross-sectional, daily
rebalance) implementations — i.e. Strategy B.

Chan Example 3.8 execution assumption, followed exactly here: weights are
computed from data available through day t-1's close, the position is
entered at day t's OPEN, and — because this strategy is INTRADAY ONLY per
the user's explicit constraint — flattened at day t's CLOSE the same day.
This is deliberately different from python/backtest/engine.py (the
event-driven tick-replay engine used for the overnight-capable pairs
strategy), because a daily cross-sectional rebalance has no meaningful
tick-level state to replay.

Anti-look-ahead-bias design (Chan Ch.3 blind spot #1, and the specific
mistake this project was warned against, see docs/lessons_from_forex_trading.md):
  - `_as_of_data()` truncates the panel to `< as_of_date` (STRICT less-than)
    before ever calling `strategy.evaluate()`.
  - Entry/exit prices used for P&L come from day t's OWN open/close, which is
    legitimate: the strategy decided the day-t POSITION using ONLY
    information through day t-1's close, then that position's fill price is
    whatever day t's actual open turns out to be (this is standard "trade
    at next bar's open" backtesting practice, not a look-ahead violation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.fees_equity import commission, market_impact
from ..core.strategies.portfolio_base import PortfolioStrategy
from ..core.types import PortfolioTarget

log = logging.getLogger(__name__)


@dataclass
class VectorBacktestResult:
    daily_returns: pd.Series             # net-of-cost portfolio return per day
    daily_gross_returns: pd.Series
    daily_costs: pd.Series
    targets_by_day: dict = field(default_factory=dict)   # {date: PortfolioTarget}
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def sharpe_annualized(self) -> float:
        if self.daily_returns.std(ddof=1) == 0 or len(self.daily_returns) < 2:
            return 0.0
        return float(self.daily_returns.mean() / self.daily_returns.std(ddof=1) * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        curve = (1.0 + self.daily_returns).cumprod()
        running_max = curve.cummax()
        drawdown = curve / running_max - 1.0
        return float(drawdown.min()) if len(drawdown) else 0.0

    @property
    def cagr(self) -> float:
        n_days = len(self.daily_returns)
        if n_days == 0:
            return 0.0
        total_return = float((1.0 + self.daily_returns).prod())
        years = n_days / 252.0
        if years <= 0 or total_return <= 0:
            return 0.0
        return total_return ** (1.0 / years) - 1.0


def run_vector_backtest(
    strategy: PortfolioStrategy,
    ohlc_panel: pd.DataFrame,          # MultiIndex (date, code); columns: open, close, adv_20d_dollars
    universe_by_day: dict,             # {date: [codes]} — point-in-time eligible universe
    capital: float = 1_000_000.0,
    impact_coefficient_bps: float = 10.0,
) -> VectorBacktestResult:
    dates = sorted(universe_by_day.keys())
    required_cols = {"open", "close"}
    missing = required_cols - set(ohlc_panel.columns)
    if missing:
        raise ValueError(f"run_vector_backtest: ohlc_panel missing columns {missing}")

    gross_returns: dict = {}
    net_returns: dict = {}
    costs: dict = {}
    targets_by_day: dict = {}

    # `ohlc_panel` is assumed sorted by (date, code) — true for every builder
    # in this codebase (sp500_universe / hist_data_us / demo synthetic
    # panels all call `.sort_index()`). That lets us find the "< as_of"
    # boundary with a binary search instead of re-scanning the whole panel
    # on every single day of the walk-forward loop, which used to make a
    # ~1000-day backtest take 15+ minutes once combined with Reality
    # Check's dozens of re-runs.
    panel_dates = ohlc_panel.index.get_level_values(0).to_numpy()

    for as_of in dates:
        universe = universe_by_day[as_of]

        boundary = int(np.searchsorted(panel_dates, np.datetime64(as_of), side="left"))
        as_of_data = ohlc_panel.iloc[:boundary]
        if as_of_data.empty:
            continue

        target: PortfolioTarget = strategy.evaluate(as_of_data, as_of, universe)
        targets_by_day[as_of] = target

        if not target.weights:
            gross_returns[as_of] = 0.0
            net_returns[as_of] = 0.0
            costs[as_of] = 0.0
            continue

        try:
            day_slice = ohlc_panel.xs(as_of, level=0)
        except KeyError:
            gross_returns[as_of] = 0.0
            net_returns[as_of] = 0.0
            costs[as_of] = 0.0
            continue

        day_pnl = 0.0
        day_cost = 0.0
        for code, weight in target.weights.items():
            if code not in day_slice.index:
                continue
            row = day_slice.loc[code]
            open_px, close_px = float(row["open"]), float(row["close"])
            if open_px <= 0:
                continue

            notional = abs(weight) * capital
            shares = int(notional / open_px)
            if shares == 0:
                continue

            signed_shares = shares if weight > 0 else -shares
            leg_return = (close_px / open_px - 1.0) * np.sign(weight)
            day_pnl += abs(weight) * leg_return * capital

            adv = float(row.get("adv_20d_dollars", 0.0)) if "adv_20d_dollars" in day_slice.columns else 0.0
            entry_comm = commission(signed_shares, open_px)
            exit_comm = commission(signed_shares, close_px)
            impact = market_impact(notional, adv, impact_coefficient_bps) * 2  # entry + exit
            day_cost += entry_comm + exit_comm + impact

        gross_returns[as_of] = day_pnl / capital
        net_returns[as_of] = (day_pnl - day_cost) / capital
        costs[as_of] = day_cost / capital

    idx = pd.DatetimeIndex(sorted(gross_returns.keys()))
    gross_series = pd.Series([gross_returns[d] for d in idx], index=idx)
    net_series = pd.Series([net_returns[d] for d in idx], index=idx)
    cost_series = pd.Series([costs[d] for d in idx], index=idx)
    equity_curve = capital * (1.0 + net_series).cumprod()

    return VectorBacktestResult(
        daily_returns=net_series,
        daily_gross_returns=gross_series,
        daily_costs=cost_series,
        targets_by_day=targets_by_day,
        equity_curve=equity_curve,
    )
