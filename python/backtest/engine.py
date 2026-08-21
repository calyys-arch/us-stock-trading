"""
Event-driven backtest engine for Strategy A (cointegrated pairs trading).

Unlike forex-trading's tick-level BacktestEngine, this replays DAILY closes
(Chan's own pairs-trading worked examples, e.g. GLD/GDX Ch.7, use daily
bars — the O-U half-life for real equity/ETF pairs is measured in days, so
tick-level replay would add complexity without adding validity). Pairs
trading MAY carry positions overnight in this system (confirmed constraint),
so "one bar = one day" is the natural replay granularity here, in contrast
to backtest/vector_engine.py's intraday open-to-close-only assumption for
the cross-sectional strategy.

Anti-look-ahead-bias design:
  - Cointegration is (re-)estimated on a trailing `coint_lookback_days`
    window ending at day t-1's close, then HELD FIXED (hedge ratio, spread
    mean/std, half-life) while trading days t through the next
    revalidation point — mirrors python/stat/pair_scanner.py's in-sample /
    out-of-sample discipline (Chan p.130).
  - Entry/exit decisions on day t use day t's close price for the
    z-score check (this is a daily end-of-day rebalance backtest, not an
    intraday one) and are FILLED at day t's close — a simplification noted
    explicitly in the health-check report; a more realistic fill model
    would use day t+1's open, and that refinement is left for a later
    iteration once the MVP's overall pipeline is validated end-to-end.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.fees_equity import round_trip_cost
from ..core.pair_position_manager import PairPositionManager
from ..core.strategies.pairs_trading import PairsTradingStrategy
from ..core.types import QualifiedSpreadOrder, SpreadSide
from ..stat.cointegration import current_spread, spread_z_score, test_pair

log = logging.getLogger(__name__)


@dataclass
class PairTrade:
    code_a: str
    code_b: str
    entry_date: datetime
    exit_date: datetime
    side: str
    qty_a: int
    qty_b: int
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str


@dataclass
class PairsBacktestConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    coint_lookback_days: int = 252
    revalidate_every_days: int = 21
    notional_per_leg: float = 50_000.0
    half_life_multiplier_max_hold: float = 3.0
    min_half_life_days: float = 1.0
    max_half_life_days: float = 60.0


@dataclass
class PairsBacktestReport:
    trades: list = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)   # {date: pnl}

    @property
    def total_net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades)

    def daily_returns_series(self, capital: float) -> pd.Series:
        idx = pd.DatetimeIndex(sorted(self.daily_pnl.keys()))
        return pd.Series([self.daily_pnl[d] / capital for d in idx], index=idx)

    def to_dict(self, capital: float = 1_000_000.0) -> dict:
        returns = self.daily_returns_series(capital)
        sharpe = 0.0
        if len(returns) >= 2 and returns.std(ddof=1) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        equity = (1 + returns).cumprod() if len(returns) else pd.Series(dtype=float)
        max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
        return {
            "total_trades": len(self.trades),
            "total_net_pnl": self.total_net_pnl,
            "win_rate": self.win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }


def run_pairs_backtest(
    code_a: str,
    code_b: str,
    prices_a: pd.Series,
    prices_b: pd.Series,
    config: PairsBacktestConfig | None = None,
) -> PairsBacktestReport:
    """Backtest ONE candidate pair over its full aligned daily price history.

    For the full-universe scan-then-trade variant — point-in-time pair
    selection across `configs/pairs_universe.yaml`, a multi-pair portfolio
    with a concurrency cap, and bid-ask/impact costs this function does not
    charge — use `python/backtest/pairs_scan_engine.py` instead (opt-in via
    `scripts/run_pairs_scan_backtest.py`; results in
    `backtests/reports/pairs_scan_report.md`). This function's single-pair
    behavior is deliberately unchanged by that work.
    """
    cfg = config or PairsBacktestConfig()
    strategy = PairsTradingStrategy(entry_z=cfg.entry_z, exit_z=cfg.exit_z)
    pm = PairPositionManager(half_life_multiplier_max_hold=cfg.half_life_multiplier_max_hold)

    df = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
    df = df.sort_index()

    report = PairsBacktestReport()
    coint = None
    days_since_revalidation = 10 ** 9

    for i in range(cfg.coint_lookback_days, len(df)):
        today = df.index[i]
        price_a_today = float(df["a"].iloc[i])
        price_b_today = float(df["b"].iloc[i])
        report.daily_pnl.setdefault(today, 0.0)

        # ── (Re-)estimate cointegration on the trailing in-sample window,
        # using data strictly BEFORE today (no look-ahead). ────────────────
        if days_since_revalidation >= cfg.revalidate_every_days:
            window = df.iloc[i - cfg.coint_lookback_days : i]
            try:
                coint = test_pair(code_a, code_b, window["a"], window["b"], computed_at=today)
            except Exception:
                log.exception("run_pairs_backtest: test_pair failed at %s", today)
                coint = None
            days_since_revalidation = 0
        days_since_revalidation += 1

        # ── Mark-to-market open position, check exits ───────────────────────
        if pm.is_open(code_a, code_b):
            pos = pm.get(code_a, code_b)
            if coint is not None and coint.spread_std > 0:
                spread = current_spread(price_a_today, price_b_today, pos.hedge_ratio)
                z = spread_z_score(spread, coint.spread_mean, coint.spread_std)
            else:
                z = pos.entry_z  # no fresh coint estimate available — hold

            exits = pm.check_exits({(code_a, code_b): z}, today, cfg.exit_z)
            for exited_pos, reason in exits:
                closed = pm.close_position(code_a, code_b)
                is_short_leg_a = closed.side == SpreadSide.SHORT_SPREAD
                cost_a = round_trip_cost(
                    closed.qty_a, closed.entry_price_a, price_a_today, is_short=is_short_leg_a,
                    holding_days=closed.holding_days(today),
                ).total
                cost_b = round_trip_cost(
                    closed.qty_b, closed.entry_price_b, price_b_today, is_short=not is_short_leg_a,
                    holding_days=closed.holding_days(today),
                ).total
                gross = closed.unrealized_pnl(price_a_today, price_b_today)
                net = gross - cost_a - cost_b
                report.trades.append(PairTrade(
                    code_a=code_a, code_b=code_b,
                    entry_date=closed.entry_time, exit_date=today,
                    side=closed.side.value, qty_a=closed.qty_a, qty_b=closed.qty_b,
                    entry_price_a=closed.entry_price_a, entry_price_b=closed.entry_price_b,
                    exit_price_a=price_a_today, exit_price_b=price_b_today,
                    gross_pnl=gross, cost=cost_a + cost_b, net_pnl=net,
                    exit_reason=reason,
                ))
                report.daily_pnl[today] = report.daily_pnl.get(today, 0.0) + net

        # ── Evaluate entry (only if no position currently open) ─────────────
        if not pm.is_open(code_a, code_b) and coint is not None:
            if not (cfg.min_half_life_days <= coint.half_life_days <= cfg.max_half_life_days):
                continue
            signal = strategy.evaluate(coint, [], price_a_today, price_b_today, today)
            if signal is None:
                continue
            qty_a = int(cfg.notional_per_leg / price_a_today) if price_a_today > 0 else 0
            qty_b = int((cfg.notional_per_leg * abs(coint.hedge_ratio)) / price_b_today) if price_b_today > 0 else 0
            if qty_a <= 0 or qty_b <= 0:
                continue
            order = QualifiedSpreadOrder(
                raw=signal, qty_a=qty_a, qty_b=qty_b,
                gross_notional=qty_a * price_a_today + qty_b * price_b_today,
                estimated_cost=0.0, kelly_fraction_used=0.0, approved=True,
            )
            pm.open_position(order, price_a_today, price_b_today, today)

    return report
