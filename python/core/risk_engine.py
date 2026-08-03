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
  - Marketable-limit pricing (user decision, 2026-07-29): every qualify_*
    method computes a bounded limit price from snapshot.price +/-
    `limit_price_buffer_bps`, NEVER leaves price selection to a market
    order — "control exact execution price, avoid PFOF-routed market
    orders" is enforced end-to-end, with ExecutionGateway._submit_order as
    the final chokepoint that refuses to submit anything without one.
  - Daily loss kill-switch (`DailyLossTracker`) and event blackout windows
    (`python/core/event_blackout.py`) for the intraday microstructure
    signals (qualify_microstructure_order) — Chan's book has no equivalent
    for the daily strategies because they don't compound risk within a
    session the way 1-minute signals do.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from typing import Optional

from .types import (
    MarketSnapshot,
    PortfolioTarget,
    QualifiedMicroOrder,
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

    # ── Order-type / execution-price controls (user decision, 2026-07-29:
    # always Limit/Stop-Limit, never Market — control exact execution price,
    # avoid PFOF-routed market-order fills) ──────────────────────────────────
    limit_price_buffer_bps: float = 10.0           # 0.10% marketable-limit buffer, entries
    flatten_limit_buffer_bps: float = 25.0         # wider buffer for urgent flatten/exit orders
    stop_limit_buffer_bps: float = 15.0            # stop trigger -> limit cap offset (protective exits)

    # ── Intraday microstructure signal controls (qualify_microstructure_order) ─
    micro_risk_per_trade_pct: float = 0.01         # 1% account risk per trade (stop-distance sizing)
    max_intraday_notional_pct: float = 0.05        # per-position cap, pct of account equity
    max_open_micro_positions: int = 5
    max_daily_loss_pct: float = 0.02               # kill-switch: halt new micro entries for the day
    event_blackout_minutes: int = 30               # +/- window around earnings/8-K/econ events
    micro_cancel_after_seconds: int = 60           # auto-cancel an unfilled entry limit after this long


RISK_CONFIG_PATH = "configs/risk.yaml"


def load_risk_config(path: str = RISK_CONFIG_PATH) -> RiskConfig:
    """RiskConfig built from configs/risk.yaml — the loader that actually
    makes that file's keys load-bearing (forex lesson #2: a config key with
    no reading code is worse than no config key at all). Unknown keys in
    the yaml (e.g. `max_gross_leverage`, `min_gross_to_cost_ratio` — read
    elsewhere, not by this dataclass) are ignored here rather than raising,
    since this loader's only job is RiskConfig's own fields. Falls back to
    RiskConfig()'s hardcoded defaults (with a warning) if the file is
    missing or unparseable — a live/paper run must never silently run with
    a HALF-loaded risk config."""
    import yaml

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        log.warning("load_risk_config: could not read %s (%s) — using hardcoded RiskConfig() defaults", path, exc)
        return RiskConfig()

    known_fields = {f.name for f in fields(RiskConfig)}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    ignored = set(raw) - known_fields
    if ignored:
        log.debug("load_risk_config: ignoring %s keys not on RiskConfig (%s)", path, sorted(ignored))
    return RiskConfig(**filtered)


def marketable_limit_price(price: float, side: str, buffer_bps: float) -> float:
    """A limit price that is aggressive enough to have a high fill
    probability against the current price while still capping the WORST
    price the order can execute at — the whole point of "never a market
    order": buy limits sit `buffer_bps` ABOVE price, sell limits sit
    `buffer_bps` BELOW it. Returns 0.0 (never a negative/zero-buffer price)
    when `price` isn't positive, which callers must treat as "no valid
    limit price" (see QualifiedSpreadOrder/QualifiedPortfolioOrder
    docstrings) rather than silently falling back to a market order."""
    if price <= 0:
        return 0.0
    buffer = buffer_bps / 10_000.0
    is_buy = side.lower() == "buy"
    return round(price * (1 + buffer) if is_buy else price * (1 - buffer), 2)


class DailyLossTracker:
    """Tracks realized+unrealized P&L for the CURRENT trading session only
    (resets on date change, not a rolling window like PDTTracker) — the
    daily-loss kill-switch for intraday microstructure signals. Chan's book
    has no equivalent for the daily strategies (they don't compound
    intraday risk the way repeated 1-minute entries do), so this is scoped
    to qualify_microstructure_order only."""

    def __init__(self) -> None:
        self._session_date: Optional[date] = None
        self._pnl: float = 0.0

    def _roll_session(self, now: datetime) -> None:
        today = now.date()
        if self._session_date != today:
            self._session_date = today
            self._pnl = 0.0

    def record_pnl(self, now: datetime, pnl_delta: float) -> None:
        self._roll_session(now)
        self._pnl += pnl_delta

    def session_pnl(self, now: datetime) -> float:
        self._roll_session(now)
        return self._pnl

    def kill_switch_triggered(self, now: datetime, account_equity: float, cfg: RiskConfig) -> bool:
        """True once today's loss exceeds `cfg.max_daily_loss_pct` of
        `account_equity` — callers (qualify_microstructure_order) must
        reject every NEW entry once this is true, though existing positions
        may still be closed (that's ExecutionGateway.flatten_*'s job, not
        this check's)."""
        if account_equity <= 0:
            return False
        return self.session_pnl(now) <= -abs(cfg.max_daily_loss_pct) * account_equity


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
        self.daily_loss = DailyLossTracker()

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
            limit_price_a=marketable_limit_price(snapshot_a.price, signal.entry_side_a, self.cfg.limit_price_buffer_bps),
            limit_price_b=marketable_limit_price(snapshot_b.price, signal.entry_side_b, self.cfg.limit_price_buffer_bps),
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
        limit_prices: dict[str, float] = {}
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
            limit_prices[code] = marketable_limit_price(
                snap.price, "buy" if weight > 0 else "sell", self.cfg.limit_price_buffer_bps,
            )
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
            limit_prices=limit_prices,
        )

    # ── Microstructure signals (python/microstructure/signals.MicroSignal) ──

    def qualify_microstructure_order(
        self,
        signal,                          # MicroSignal (loosely typed — see QualifiedMicroOrder)
        snapshot: MarketSnapshot,
        account_equity: float,
        open_micro_positions: int,
        event_blackout: Optional[bool] = None,
        now: Optional[datetime] = None,
    ) -> QualifiedMicroOrder:
        """Position-size + qualify one intraday MicroSignal. Unlike
        qualify_spread_order/qualify_portfolio_order, this ALWAYS attaches a
        protective stop-limit spec (stop_price/stop_limit_price) because
        every microstructure signal carries its own stop by construction
        (python/microstructure/signals.MicroSignal.stop_price) — there is no
        equivalent "spread-based exit" concept to fall back on here.

        `event_blackout`: True/False from python/core/event_blackout.py's
        is_event_blackout(); None is treated as "evidence unavailable" and
        does NOT block the order (same three-valued-evidence convention as
        python/signals/trap_detector.py, but here the caller is expected to
        have actually checked — passing None because you forgot to call the
        checker is a caller bug, not a legitimate "unknown" state)."""
        now = now or signal.signal_time
        rejection: Optional[str] = None

        if self.daily_loss.kill_switch_triggered(now, account_equity, self.cfg):
            rejection = "daily_loss_kill_switch"
        elif event_blackout:
            rejection = "event_blackout"
        elif open_micro_positions >= self.cfg.max_open_micro_positions:
            rejection = "max_open_micro_positions"
        elif not snapshot.is_tradeable:
            rejection = "snapshot_not_tradeable"
        elif not snapshot.is_regular_trading_hours:
            rejection = "outside_rth"

        entry_price = signal.entry_price
        stop_price = signal.stop_price
        stop_dist = abs(entry_price - stop_price)
        qty = 0
        if stop_dist > 0 and entry_price > 0 and account_equity > 0:
            risk_dollars = account_equity * self.cfg.micro_risk_per_trade_pct
            shares_by_risk = risk_dollars / stop_dist
            shares_by_notional = (account_equity * self.cfg.max_intraday_notional_pct) / entry_price
            qty = max(int(min(shares_by_risk, shares_by_notional)), 0)

        if rejection is None and qty <= 0:
            rejection = "size_rounded_to_zero"

        entry_side = "buy" if signal.direction == "long" else "sell"
        exit_side = "sell" if signal.direction == "long" else "buy"
        entry_limit = marketable_limit_price(entry_price, entry_side, self.cfg.limit_price_buffer_bps)
        # The protective stop's LIMIT cap sits a further buffer beyond the
        # trigger, in the direction the stop is already moving against us —
        # a stop-limit with limit==stop can fail to fill entirely in a fast
        # move, which defeats the point of a protective stop.
        stop_limit = marketable_limit_price(stop_price, exit_side, self.cfg.stop_limit_buffer_bps)

        return QualifiedMicroOrder(
            raw=signal,
            qty=qty,
            entry_limit_price=entry_limit,
            stop_price=stop_price,
            stop_limit_price=stop_limit,
            target_price=signal.target_price,
            cancel_after_seconds=self.cfg.micro_cancel_after_seconds,
            gross_notional=qty * entry_price,
            approved=rejection is None,
            rejection_reason=rejection,
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
