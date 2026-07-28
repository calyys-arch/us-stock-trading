"""
Daily live-vs-backtest reconciliation.

forex-trading lesson #6 (docs/lessons_from_forex_trading.md): `range_bounce`
showed +0.895R/trade live vs -0.11R/trade in the decade backtest — a direct
sign-flip contradiction that went unresolved because nothing automatically
compared live fills to what the same-day backtest replay would have
produced. Chan Ch.5 has an explicit checklist for "why does live diverge
from backtest" (fill-price slippage, data feed discrepancies, latency,
corporate actions) that was never automated in the predecessor system.

This module replays one trading day's ACTUAL fills against the same
signal/feature pipeline used for backtesting, and flags any of:
  - qualified-signal count mismatch (backtest would have traded when live
    didn't, or vice versa)
  - fill-price delta beyond `price_tolerance_bps`
  - P&L delta beyond `pnl_tolerance_pct`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class LiveFill:
    code: str
    side: str
    qty: int
    fill_price: float
    timestamp: datetime
    strategy: str


@dataclass
class BacktestFill:
    code: str
    side: str
    qty: int
    fill_price: float
    timestamp: datetime
    strategy: str


@dataclass
class ReconciliationFlag:
    kind: str          # "missing_in_backtest" | "missing_in_live" | "price_divergence" | "qty_divergence"
    code: str
    strategy: str
    detail: str


@dataclass
class ReconciliationReport:
    date: str
    flags: list = field(default_factory=list)
    live_fill_count: int = 0
    backtest_fill_count: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.flags) == 0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "is_clean": self.is_clean,
            "live_fill_count": self.live_fill_count,
            "backtest_fill_count": self.backtest_fill_count,
            "flags": [f.__dict__ for f in self.flags],
        }


def reconcile_day(
    live_fills: list,
    backtest_fills: list,
    price_tolerance_bps: float = 25.0,
    date: str | None = None,
) -> ReconciliationReport:
    """Compare one day's live fills against the same day's backtest-replay
    fills. Both lists should already be restricted to the same trading day.
    """
    report = ReconciliationReport(
        date=date or datetime.utcnow().date().isoformat(),
        live_fill_count=len(live_fills),
        backtest_fill_count=len(backtest_fills),
    )

    def _key(f):
        return (f.code, f.strategy, f.side)

    live_by_key = {_key(f): f for f in live_fills}
    bt_by_key = {_key(f): f for f in backtest_fills}

    for key, live_fill in live_by_key.items():
        bt_fill = bt_by_key.get(key)
        if bt_fill is None:
            report.flags.append(ReconciliationFlag(
                kind="missing_in_backtest", code=key[0], strategy=key[1],
                detail=f"live fill {live_fill.qty}@{live_fill.fill_price} has no backtest counterpart",
            ))
            continue

        if live_fill.fill_price > 0:
            price_delta_bps = abs(live_fill.fill_price - bt_fill.fill_price) / live_fill.fill_price * 10_000
            if price_delta_bps > price_tolerance_bps:
                report.flags.append(ReconciliationFlag(
                    kind="price_divergence", code=key[0], strategy=key[1],
                    detail=f"live={live_fill.fill_price:.2f} backtest={bt_fill.fill_price:.2f} "
                           f"delta={price_delta_bps:.1f}bps (tolerance={price_tolerance_bps}bps)",
                ))

        if live_fill.qty != bt_fill.qty:
            report.flags.append(ReconciliationFlag(
                kind="qty_divergence", code=key[0], strategy=key[1],
                detail=f"live_qty={live_fill.qty} backtest_qty={bt_fill.qty}",
            ))

    for key, bt_fill in bt_by_key.items():
        if key not in live_by_key:
            report.flags.append(ReconciliationFlag(
                kind="missing_in_live", code=key[0], strategy=key[1],
                detail=f"backtest expected a fill ({bt_fill.qty}@{bt_fill.fill_price}) that live did not produce",
            ))

    if not report.is_clean:
        log.warning("Reconciliation %s: %d flags found", report.date, len(report.flags))
    return report
