"""
Live daily-bar pairs scheduler — scanned-universe variant, regime-gated.

This is the live half of `python/backtest/pairs_scan_engine.py` for the
2026-08-15 paper-forward experiment. Frozen config is the 2022 GO
low-frequency variant (`entry_z=4.0`, `exit_z=0.5`,
`half_life_multiplier_max_hold=3.0`) from
`backtests/reports/regime_generalization_report.md` §2b — not a new grid.

Regime gate: `python/analytics/trend_efficiency_gate.live_entry_allowed`
on a broad-market proxy (SPY). When the gate says "persistent trend /
not mean-reversion-friendly", NO NEW entries are emitted; existing
positions still exit via the existing z-reversion / stale-timeout rules
(never trap the book). This is regime-conditional automation, NOT a
claim that pairs is GO in 2026. See
`backtests/reports/pairs_regime_live_protocol.md`.

Paper-only: publishes `qualified_spread_order` onto the same bus
ExecutionGateway already consumes. Tiny notional comes from
`configs/strategy.yaml` `paper_notional_per_leg` (and the RiskEngine
`paper_max_notional_usd` cap). Never a market order.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from python.analytics.trend_efficiency_gate import live_entry_allowed
from python.backtest.pairs_scan_engine import (
    PairsScanConfig,
    candidate_pairs_from_buckets,
    load_pairs_universe,
    pairs_buckets,
    select_active_pairs,
)
from python.core.bus import MessageBus
from python.core.calendar import is_regular_trading_hours
from python.core.pair_position_manager import PairPositionManager
from python.core.risk_engine import RiskEngine, marketable_limit_price
from python.core.strategies.pairs_trading import PairsTradingStrategy
from python.core.types import (
    CointegrationResult,
    MarketSnapshot,
    QualifiedSpreadOrder,
    SpreadSide,
    SpreadSignal,
)
from python.stat.cointegration import current_spread, spread_z_score
from python.stat.pair_scanner import scan

log = logging.getLogger(__name__)

REGIME_PROXY_SYMBOL = "SPY"


def _naive_day(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


def _price_snapshot(code: str, price: float, now: datetime, adv: float = 0.0) -> MarketSnapshot:
    return MarketSnapshot(
        code=code, price=price, volume_today=0, turnover_today=0.0, vwap=price,
        atr14=1.0, atr5=1.0, rsi14=50.0, ema8=price, ema20=price,
        bb_upper=price + 1, bb_mid=price, bb_lower=price - 1,
        vol_ma20=0.0, vol_ratio=1.0, bid_ask_spread_pct=0.001,
        timestamp=now, adv_20d_dollars=adv, short_locate_available=True,
        is_regular_trading_hours=True,
    )


class LivePairsScheduler:
    """One evaluation cycle = exits first, then (iff regime gate open)
    scanned-universe entries, until `max_concurrent_pairs`."""

    def __init__(
        self,
        bus: MessageBus,
        risk_engine: RiskEngine,
        strategy_cfg: dict,
        get_account_equity: Callable[[], float],
        close_panel: Optional[pd.DataFrame] = None,
        regime_close: Optional[pd.Series] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        paper_notional_per_leg: Optional[float] = None,
    ) -> None:
        self._bus = bus
        self._risk = risk_engine
        self._get_equity = get_account_equity
        self._now_fn = now_fn or datetime.utcnow
        self._cfg = PairsScanConfig(
            entry_z=float(strategy_cfg.get("entry_z", 4.0)),
            exit_z=float(strategy_cfg.get("exit_z", 0.5)),
            coint_lookback_days=int(strategy_cfg.get("coint_lookback_days", 252)),
            revalidate_every_days=int(strategy_cfg.get("revalidate_every_days", 21)),
            notional_per_leg=float(
                paper_notional_per_leg
                if paper_notional_per_leg is not None
                else strategy_cfg.get("paper_notional_per_leg", strategy_cfg.get("notional_per_leg", 3000.0))
            ),
            half_life_multiplier_max_hold=float(strategy_cfg.get("half_life_multiplier_max_hold", 3.0)),
            min_half_life_days=float(strategy_cfg.get("min_half_life_days", 1.0)),
            max_half_life_days=float(strategy_cfg.get("max_half_life_days", 60.0)),
        )
        self._strategy = PairsTradingStrategy(entry_z=self._cfg.entry_z, exit_z=self._cfg.exit_z)
        self._pm = PairPositionManager(
            half_life_multiplier_max_hold=self._cfg.half_life_multiplier_max_hold,
        )
        self._close_panel = close_panel
        self._regime_close = regime_close
        self._active: list[CointegrationResult] = []
        self._latest_by_pair: dict[tuple[str, str], CointegrationResult] = {}
        self._last_scan_as_of: Optional[pd.Timestamp] = None
        self._last_eval_date: Optional[object] = None
        self.regime_gate_open: bool = False
        self.regime_gate_as_of: Optional[str] = None
        self.regime_gate_reason: str = "not_yet_evaluated"

    @property
    def position_manager(self) -> PairPositionManager:
        return self._pm

    @property
    def regime_close(self):
        return self._regime_close

    def set_regime_close(self, regime_close: pd.Series) -> None:
        """Set the SPY (or other proxy) daily series without wiping the
        pairs price panel. Used so the live tape/gate catalog can classify
        even when the 66-ETF panel is still downloading."""
        self._regime_close = regime_close

    def set_panels(self, close_panel: pd.DataFrame, regime_close: Optional[pd.Series] = None) -> None:
        self._close_panel = close_panel
        if regime_close is not None:
            self._regime_close = regime_close
        elif close_panel is not None and REGIME_PROXY_SYMBOL in close_panel.columns:
            self._regime_close = close_panel[REGIME_PROXY_SYMBOL]

    def refresh_scan(self, as_of) -> None:
        """Point-in-time cointegration scan: lookback window ends strictly
        before `as_of` (exclusive), matching pairs_scan_engine.build_scan_schedule."""
        if self._close_panel is None or self._close_panel.empty:
            self._active = []
            return
        as_of_ts = _naive_day(as_of)
        hist = self._close_panel.loc[self._close_panel.index < as_of_ts]
        if len(hist) < 60:
            self._active = []
            return
        universe = load_pairs_universe()
        candidates = candidate_pairs_from_buckets(pairs_buckets(universe))
        results = scan(
            candidates, hist,
            lookback_days=self._cfg.coint_lookback_days,
            as_of=as_of_ts.to_pydatetime(),
            min_half_life_days=0.0,
            max_half_life_days=float("inf"),
        )
        self._active = select_active_pairs(results, self._cfg)
        for r in self._active:
            self._latest_by_pair[(r.code_a, r.code_b)] = r
        self._last_scan_as_of = as_of_ts

    def _maybe_rescan(self, as_of) -> None:
        as_of_ts = _naive_day(as_of)
        if self._last_scan_as_of is None:
            self.refresh_scan(as_of_ts)
            return
        last = _naive_day(self._last_scan_as_of)
        elapsed = (as_of_ts - last).days
        if elapsed >= self._cfg.revalidate_every_days:
            self.refresh_scan(as_of_ts)

    def evaluate_regime_gate(self, as_of) -> bool:
        if self._regime_close is None or self._regime_close.dropna().empty:
            self.regime_gate_open = False
            self.regime_gate_as_of = str(pd.Timestamp(as_of).date())
            self.regime_gate_reason = "missing_regime_proxy"
            return False
        allowed = live_entry_allowed(self._regime_close, as_of=as_of)
        self.regime_gate_open = bool(allowed)
        self.regime_gate_as_of = str(pd.Timestamp(as_of).date())
        self.regime_gate_reason = "mean_reversion_friendly" if allowed else "trend_gate_closed"
        return self.regime_gate_open

    async def evaluate_once(self, now: Optional[datetime] = None) -> dict:
        """Run one daily cycle. Safe to call repeatedly: at most one
        entry-evaluation per calendar date. Exits are always considered."""
        now = now or self._now_fn()
        result = {"exits": 0, "entries": 0, "gate_open": False, "skipped": None}
        if not is_regular_trading_hours(now):
            result["skipped"] = "outside_rth"
            return result
        if self._close_panel is None or self._close_panel.empty:
            result["skipped"] = "no_price_panel"
            return result

        today = _naive_day(now)
        last_bar = self._close_panel.index[-1]
        row = self._close_panel.loc[last_bar]
        self._maybe_rescan(today)
        gate_open = self.evaluate_regime_gate(today)
        result["gate_open"] = gate_open

        result["exits"] = await self._process_exits(row, now)

        if self._last_eval_date == today.date():
            result["skipped"] = "already_evaluated_today"
            return result
        self._last_eval_date = today.date()

        if not gate_open:
            result["skipped"] = self.regime_gate_reason
            return result

        result["entries"] = await self._process_entries(row, now)
        return result

    async def _process_exits(self, row: pd.Series, now: datetime) -> int:
        z_by_pair: dict[tuple[str, str], float] = {}
        for pos in list(self._pm.open_positions):
            key = (pos.code_a, pos.code_b)
            price_a, price_b = row.get(pos.code_a), row.get(pos.code_b)
            if price_a is None or price_b is None or pd.isna(price_a) or pd.isna(price_b):
                continue
            est = self._latest_by_pair.get(key)
            if est is not None and est.spread_std > 0:
                spread = current_spread(float(price_a), float(price_b), pos.hedge_ratio)
                z_by_pair[key] = spread_z_score(spread, est.spread_mean, est.spread_std)
            else:
                z_by_pair[key] = pos.entry_z

        n = 0
        for closed_pos, reason in self._pm.check_exits(z_by_pair, now, self._cfg.exit_z):
            closed = self._pm.close_position(closed_pos.code_a, closed_pos.code_b)
            if closed is None:
                continue
            price_a, price_b = row.get(closed.code_a), row.get(closed.code_b)
            if price_a is None or price_b is None or pd.isna(price_a) or pd.isna(price_b):
                continue
            exit_side = (
                SpreadSide.SHORT_SPREAD if closed.side == SpreadSide.LONG_SPREAD
                else SpreadSide.LONG_SPREAD
            )
            signal = SpreadSignal(
                id=f"exit-{closed.pair_key}",
                strategy="pairs_trading",
                code_a=closed.code_a, code_b=closed.code_b,
                hedge_ratio=closed.hedge_ratio, side=exit_side,
                z_score=z_by_pair.get((closed.code_a, closed.code_b), 0.0),
                entry_z_threshold=self._cfg.entry_z,
                exit_z_threshold=self._cfg.exit_z,
                spread_mean=0.0, half_life_days=closed.half_life_days,
                confidence=1.0, timestamp=now,
                metadata={"exit_reason": reason},
            )
            snap_a = _price_snapshot(closed.code_a, float(price_a), now)
            snap_b = _price_snapshot(closed.code_b, float(price_b), now)
            order = QualifiedSpreadOrder(
                raw=signal, qty_a=closed.qty_a, qty_b=closed.qty_b,
                gross_notional=closed.qty_a * float(price_a) + closed.qty_b * float(price_b),
                estimated_cost=0.0, kelly_fraction_used=0.0, approved=True,
                limit_price_a=marketable_limit_price(
                    float(price_a), signal.entry_side_a, self._risk.cfg.flatten_limit_buffer_bps,
                ),
                limit_price_b=marketable_limit_price(
                    float(price_b), signal.entry_side_b, self._risk.cfg.flatten_limit_buffer_bps,
                ),
            )
            await self._bus.publish("qualified_spread_order", order)
            n += 1
        return n

    async def _process_entries(self, row: pd.Series, now: datetime) -> int:
        n = 0
        n_open = len(self._pm.open_positions)
        equity = 0.0
        try:
            equity = float(self._get_equity() or 0.0)
        except Exception:
            log.exception("LivePairsScheduler: get_account_equity failed — treating as 0")
        for coint in self._active:
            if n_open >= self._cfg.max_concurrent_pairs:
                break
            key = (coint.code_a, coint.code_b)
            if self._pm.is_open(*key):
                continue
            price_a, price_b = row.get(coint.code_a), row.get(coint.code_b)
            if price_a is None or price_b is None or pd.isna(price_a) or pd.isna(price_b):
                continue
            price_a, price_b = float(price_a), float(price_b)
            if price_a <= 0 or price_b <= 0:
                continue
            signal = self._strategy.evaluate(coint, [], price_a, price_b, now)
            if signal is None:
                continue
            snap_a = _price_snapshot(coint.code_a, price_a, now)
            snap_b = _price_snapshot(coint.code_b, price_b, now)
            kelly = (self._cfg.notional_per_leg / equity) if equity > 0 else 0.0
            order = self._risk.qualify_spread_order(
                signal, snap_a, snap_b, account_equity=equity, kelly_fraction=kelly, now=now,
            )
            if not order.approved:
                log.info(
                    "LivePairsScheduler: entry rejected %s/%s (%s)",
                    coint.code_a, coint.code_b, order.rejection_reason,
                )
                await self._bus.publish("qualified_spread_order", order)
                continue
            await self._bus.publish("qualified_spread_order", order)
            self._pm.open_position(order, price_a, price_b, now)
            n_open += 1
            n += 1
        return n
