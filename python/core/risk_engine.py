"""
Risk Engine — constraint-checking layer between strategy signals and
execution. Ported in spirit from forex-trading/python/core/risk_engine.py
(the "signal -> approved order" gate pattern), but ALL forex-specific guards
(non_forex_instrument, forex_only_gateway_block, _QUOTE_TO_USD, is_forex)
are removed and replaced with the US-equity constraint set from Chan Ch.4
and standard Reg T / FINRA rules:

  - Kelly-based position sizing (python/core/kelly.py) as the PRIMARY capital
    allocator — this engine does not decide "how much edge exists", it
    takes the Kelly-suggested fraction and CLIPS it against hard limits.
  - 1% of 20-day ADV per order (market-impact/liquidity gate, Chan p.85-90).
  - Sector concentration cap (avoid the whole book being one sector bet).
  - Pattern Day Trader (PDT) rule: an account under $25,000 equity may not
    execute more than 3 day trades in a rolling 5-business-day window
    (FINRA Rule 4210) — Strategy B (intraday cross-sectional) is a day-trade
    generator, so this is enforced here, not left to the broker to reject.
  - Reg T leverage caps: 4:1 intraday buying power, 2:1 overnight, for a
    PDT-qualified account; the overnight cap specifically matters because
    Strategy A (pairs) is allowed to hold overnight.
  - Short-sale locate requirement: no short leg is approved without
    `short_locate_available=True` on the snapshot (hard-to-borrow names are
    rejected outright rather than silently downsized).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .types import (
    MarketSnapshot,
    PortfolioTarget,
    QualifiedPortfolioOrder,
    QualifiedSpreadOrder,
    SpreadSide,
    SpreadSignal,
)

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_adv_participation_pct: float = 0.01       # 1% of 20d ADV per order (Chan p.85)
    max_sector_exposure_pct: float = 0.30          # 30% of gross capital per sector
    max_single_position_pct: float = 0.10          # 10% of gross capital per name
    pdt_equity_threshold: float = 25_000.0
    max_day_trades_rolling_5d: int = 3
    reg_t_intraday_leverage: float = 4.0
    reg_t_overnight_leverage: float = 2.0
    kelly_multiplier: float = 0.5                  # half-Kelly (Chan p.70)
    require_short_locate: bool = True
    min_price: float = 5.0                         # Chan Ch.5: exclude sub-$5 stocks


class PDTTracker:
    """Tracks day trades in a rolling 5-BUSINESS-day window per FINRA 4210."""

    def __init__(self) -> None:
        self._day_trade_dates: deque[datetime] = deque()

    def record_day_trade(self, when: datetime) -> None:
        self._day_trade_dates.append(when)
        self._prune(when)

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=7)  # generous calendar buffer over 5 business days
        while self._day_trade_dates and self._day_trade_dates[0] < cutoff:
            self._day_trade_dates.popleft()

    def count_rolling_5d(self, now: datetime) -> int:
        self._prune(now)
        cutoff = now - timedelta(days=7)
        return sum(1 for d in self._day_trade_dates if d >= cutoff)

    def can_day_trade(self, now: datetime, account_equity: float, cfg: RiskConfig) -> bool:
        if account_equity >= cfg.pdt_equity_threshold:
            return True
        return self.count_rolling_5d(now) < cfg.max_day_trades_rolling_5d


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()
        self.pdt = PDTTracker()

    # ── Strategy A: pairs spread orders ─────────────────────────────────────

    def qualify_spread_order(
        self,
        signal: SpreadSignal,
        snapshot_a: MarketSnapshot,
        snapshot_b: MarketSnapshot,
        account_equity: float,
        kelly_fraction: float,
        now: Optional[datetime] = None,
    ) -> QualifiedSpreadOrder:
        now = now or signal.timestamp
        rejection: Optional[str] = None

        short_leg_snapshot = snapshot_a if signal.side == SpreadSide.SHORT_SPREAD else snapshot_b
        if self.cfg.require_short_locate and not short_leg_snapshot.short_locate_available:
            rejection = f"no_short_locate:{short_leg_snapshot.code}"
        elif snapshot_a.price < self.cfg.min_price or snapshot_b.price < self.cfg.min_price:
            rejection = "below_min_price"
        elif snapshot_a.is_halted or snapshot_b.is_halted:
            rejection = "leg_halted"
        elif not snapshot_a.is_regular_trading_hours or not snapshot_b.is_regular_trading_hours:
            rejection = "outside_rth"

        capped_fraction = max(-self.cfg.max_single_position_pct, min(self.cfg.max_single_position_pct, kelly_fraction))
        notional_target = abs(capped_fraction) * account_equity

        adv_cap_a = snapshot_a.adv_20d_dollars * self.cfg.max_adv_participation_pct
        adv_cap_b = snapshot_b.adv_20d_dollars * self.cfg.max_adv_participation_pct
        if adv_cap_a > 0:
            notional_target = min(notional_target, adv_cap_a)
        if adv_cap_b > 0:
            notional_target = min(notional_target, adv_cap_b / max(signal.hedge_ratio, 1e-6))

        qty_a = int(notional_target / snapshot_a.price) if snapshot_a.price > 0 else 0
        qty_b = int((notional_target * abs(signal.hedge_ratio)) / snapshot_b.price) if snapshot_b.price > 0 else 0

        if rejection is None and (qty_a <= 0 or qty_b <= 0):
            rejection = "size_rounded_to_zero"

        gross_notional = qty_a * snapshot_a.price + qty_b * snapshot_b.price

        return QualifiedSpreadOrder(
            raw=signal,
            qty_a=qty_a,
            qty_b=qty_b,
            gross_notional=gross_notional,
            estimated_cost=0.0,
            kelly_fraction_used=capped_fraction,
            approved=rejection is None,
            rejection_reason=rejection,
        )

    # ── Strategy B: cross-sectional portfolio target ────────────────────────

    def qualify_portfolio_order(
        self,
        target: PortfolioTarget,
        snapshots: dict,                 # {code: MarketSnapshot}
        account_equity: float,
        now: Optional[datetime] = None,
    ) -> QualifiedPortfolioOrder:
        now = now or target.as_of

        if account_equity < self.cfg.pdt_equity_threshold:
            trades_today = len(target.weights)
            allowed = max(0, self.cfg.max_day_trades_rolling_5d - self.pdt.count_rolling_5d(now))
            if trades_today > allowed:
                return QualifiedPortfolioOrder(
                    raw=target, target_shares={}, rejected_codes={"*": "pdt_limit_exceeded"},
                    approved=False,
                )

        sector_exposure: dict[str, float] = {}
        target_shares: dict[str, int] = {}
        rejected: dict[str, str] = {}
        gross_notional = 0.0

        for code, weight in target.weights.items():
            snap = snapshots.get(code)
            if snap is None:
                rejected[code] = "no_snapshot"
                continue
            if not snap.is_tradeable:
                rejected[code] = "not_tradeable"
                continue
            if weight < 0 and self.cfg.require_short_locate and not snap.short_locate_available:
                rejected[code] = "no_short_locate"
                continue

            capped_weight = max(-self.cfg.max_single_position_pct, min(self.cfg.max_single_position_pct, weight))
            notional = abs(capped_weight) * account_equity

            adv_cap = snap.adv_20d_dollars * self.cfg.max_adv_participation_pct
            if adv_cap > 0:
                notional = min(notional, adv_cap)

            sector = snap.sector or "UNKNOWN"
            projected_sector_notional = sector_exposure.get(sector, 0.0) + notional
            sector_cap = self.cfg.max_sector_exposure_pct * account_equity
            if projected_sector_notional > sector_cap:
                notional = max(0.0, sector_cap - sector_exposure.get(sector, 0.0))
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + notional

            shares = int(notional / snap.price) if snap.price > 0 else 0
            if shares <= 0:
                rejected[code] = "size_rounded_to_zero_or_capped"
                continue

            signed_shares = shares if weight > 0 else -shares
            target_shares[code] = signed_shares
            gross_notional += shares * snap.price

        if target_shares:
            self.pdt.record_day_trade(now)

        return QualifiedPortfolioOrder(
            raw=target,
            target_shares=target_shares,
            rejected_codes=rejected,
            gross_notional=gross_notional,
            estimated_cost=0.0,
            kelly_fraction_used=self.cfg.kelly_multiplier,
            approved=bool(target_shares),
        )

    # ── Reg T leverage check (used by execution_gateway before submission) ──

    def check_leverage(
        self,
        gross_notional: float,
        account_equity: float,
        is_overnight: bool,
    ) -> bool:
        if account_equity <= 0:
            return False
        cap = self.cfg.reg_t_overnight_leverage if is_overnight else self.cfg.reg_t_intraday_leverage
        return (gross_notional / account_equity) <= cap
