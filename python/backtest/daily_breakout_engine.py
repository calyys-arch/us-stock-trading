"""
Event-driven single-symbol backtest engine for
`python/core/strategies/daily_range_breakout.py` (Track 2,
backtests/reports/alt_universe_frequency_exploration.md).

Mirrors `python/backtest/engine.py`'s `run_pairs_backtest` shape (a plain
day-by-day loop producing a list of trade records + a daily P&L dict) but for
ONE instrument at a time instead of a pair, and with an explicit T+1-OPEN
fill discipline (see module docstring of daily_range_breakout.py) instead of
same-bar-close fills.

Costs: full `python/core/fees_equity.round_trip_cost` — commission + SEC
Section 31 + FINRA TAF + short borrow (for short trades) + square-root market
impact (against a trailing 20-day dollar ADV computed from `bars` itself) +
bid-ask half-spread. `half_spread_bps` is a single flat number per call (the
caller picks it — calibrated per-symbol for the existing 20-symbol universe,
or a documented wide-spread assumption for an alternate universe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.fees_equity import round_trip_cost
from ..core.strategies.daily_range_breakout import evaluate_daily_breakout


@dataclass
class DailyBreakoutConfig:
    range_days: int = 20
    hold_days: int = 10
    stop_atr_mult: float = 2.0
    target_r_multiple: float | None = 3.0
    atr_days: int = 14
    notional_per_trade: float = 50_000.0
    half_spread_bps: float = 0.0
    annual_borrow_rate: float = 0.0075


@dataclass
class DailyBreakoutTrade:
    symbol: str
    direction: str
    entry_date: datetime
    exit_date: datetime
    qty: int
    entry_price: float
    exit_price: float
    stop: float
    target: float | None
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str
    holding_days: int


@dataclass
class DailyBreakoutReport:
    trades: list = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)

    @property
    def total_net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        wins = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        losses = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if losses > 0:
            return wins / losses
        return float("inf") if wins > 0 else 0.0

    @property
    def profit_factor_gross(self) -> float:
        wins = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        losses = abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl < 0))
        if losses > 0:
            return wins / losses
        return float("inf") if wins > 0 else 0.0

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
        reasons: dict[str, int] = {}
        for t in self.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        return {
            "total_trades": len(self.trades),
            "total_net_pnl": self.total_net_pnl,
            "gross_pnl": float(sum(t.gross_pnl for t in self.trades)),
            "total_cost": float(sum(t.cost for t in self.trades)),
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "profit_factor_gross": self.profit_factor_gross,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "exit_reasons": dict(sorted(reasons.items())),
        }


def run_daily_breakout_backtest(
    symbol: str,
    bars: pd.DataFrame,  # index=date, columns=[open, high, low, close, volume]
    config: DailyBreakoutConfig | None = None,
) -> DailyBreakoutReport:
    """Single-symbol daily-bar replay. `bars` must be sorted ascending by
    date with no duplicate dates. A signal decided from bars through day i's
    close fills at day i+1's OPEN (never day i's own close) — see
    daily_range_breakout.py's module docstring for why this differs from
    engine.py's same-bar-close-fill convention."""
    cfg = config or DailyBreakoutConfig()
    bars = bars.sort_index()
    dates = bars.index
    adv20 = (bars["close"] * bars["volume"]).rolling(20, min_periods=1).mean()

    report = DailyBreakoutReport()
    position: dict | None = None
    pending: dict | None = None

    warmup = cfg.range_days + cfg.atr_days + 1
    for i in range(len(bars)):
        today = dates[i]
        row = bars.iloc[i]
        report.daily_pnl.setdefault(today, 0.0)

        # 1. Fill a pending entry (decided at yesterday's close) at TODAY's open.
        if pending is not None and position is None:
            entry_price = float(row["open"])
            qty = int(cfg.notional_per_trade / entry_price) if entry_price > 0 else 0
            if qty > 0:
                position = {
                    "direction": pending["direction"],
                    "entry_date": today,
                    "entry_idx": i,
                    "qty": qty,
                    "entry_price": entry_price,
                    "stop": pending["stop"],
                    "target": pending["target"],
                }
            pending = None

        # 2. Check exits for an open position using TODAY's bar.
        if position is not None:
            direction = position["direction"]
            exit_price = None
            exit_reason = None
            if direction == "long":
                if float(row["low"]) <= position["stop"]:
                    exit_price, exit_reason = position["stop"], "stop"
                elif position["target"] is not None and float(row["high"]) >= position["target"]:
                    exit_price, exit_reason = position["target"], "target"
            else:
                if float(row["high"]) >= position["stop"]:
                    exit_price, exit_reason = position["stop"], "stop"
                elif position["target"] is not None and float(row["low"]) <= position["target"]:
                    exit_price, exit_reason = position["target"], "target"

            holding_days = i - position["entry_idx"]
            if exit_reason is None and holding_days >= cfg.hold_days:
                exit_price, exit_reason = float(row["close"]), "time"

            if exit_reason is not None:
                is_short = direction == "short"
                adv_dollars = float(adv20.iloc[i]) if not pd.isna(adv20.iloc[i]) else 0.0
                cost = round_trip_cost(
                    position["qty"], position["entry_price"], exit_price,
                    is_short=is_short, holding_days=max(holding_days, 1),
                    adv_dollars=adv_dollars, annual_borrow_rate=cfg.annual_borrow_rate,
                    half_spread_bps=cfg.half_spread_bps,
                ).total
                sign = 1.0 if direction == "long" else -1.0
                gross = sign * position["qty"] * (exit_price - position["entry_price"])
                net = gross - cost
                report.trades.append(DailyBreakoutTrade(
                    symbol=symbol, direction=direction,
                    entry_date=position["entry_date"], exit_date=today,
                    qty=position["qty"], entry_price=position["entry_price"], exit_price=exit_price,
                    stop=position["stop"], target=position["target"],
                    gross_pnl=gross, cost=cost, net_pnl=net,
                    exit_reason=exit_reason, holding_days=holding_days,
                ))
                report.daily_pnl[today] += net
                position = None

        # 3. Evaluate a NEW candidate signal from bars through TODAY's close
        #    (only when flat and nothing already pending fill tomorrow).
        if position is None and pending is None and i >= warmup:
            window = bars.iloc[: i + 1]
            sig = evaluate_daily_breakout(
                window, range_days=cfg.range_days, stop_atr_mult=cfg.stop_atr_mult,
                target_r_multiple=cfg.target_r_multiple, atr_days=cfg.atr_days,
            )
            if sig is not None:
                pending = sig

    return report
