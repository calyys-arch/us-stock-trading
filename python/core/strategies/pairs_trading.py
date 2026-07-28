"""
Strategy A — Cointegrated Pairs Trading (Chan Ch.7, pp.126-133).

Entry: |z_score| >= entry_z (spread has deviated `entry_z` standard
deviations from its OLS/O-U-estimated mean).
Exit: handled by PairPositionManager.check_exits() (z-score reversion or
stale-timeout) — deliberately NOT part of this class, because exits apply
to already-open positions tracked centrally, not to a fresh per-bar
evaluate() call.

Free parameters across the full pairs-trading pipeline (Chan Ch.3
discipline: keep this <= 5 — see python/backtest/param_guard.py for the
exact accounting, which excludes pure data-window/sizing/housekeeping
settings such as coint_lookback_days and notional_per_leg):
  1. entry_z (this class)
  2. exit_z (this class; consumed by PairPositionManager for exits)
  3. half_life_multiplier_max_hold (PairPositionManager's stale-timeout rule)
  4. min_half_life_days (pair_scanner tradeability screen)
  5. max_half_life_days (pair_scanner tradeability screen)
That's exactly 5 — AT the parameter budget, not under it. Any new knob
requires removing one of these first.

This strategy MAY hold positions overnight — Chan's own pairs examples
(GLD/GDX, GLD/USO) have half-lives of days to weeks, not minutes, and
forcing an intraday-only exit would truncate the reversion before it
completes (confirmed with the user: pairs trading is explicitly allowed to
carry positions overnight, unlike Strategy B below).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ...core.types import CointegrationResult, SpreadSide, SpreadSignal
from .base import PairsStrategy


class PairsTradingStrategy(PairsStrategy):
    name = "pairs_trading"

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        self.entry_z = entry_z
        self.exit_z = exit_z

    def evaluate(
        self,
        coint: CointegrationResult,
        spread_series: list,
        current_price_a: float,
        current_price_b: float,
        timestamp: datetime,
    ) -> Optional[SpreadSignal]:
        if not coint.is_tradeable:
            return None
        if coint.spread_std <= 0:
            return None

        from ...stat.cointegration import current_spread, spread_z_score

        spread = current_spread(current_price_a, current_price_b, coint.hedge_ratio)
        z = spread_z_score(spread, coint.spread_mean, coint.spread_std)

        if z <= -self.entry_z:
            # spread is far BELOW its mean -> expect it to rise -> long the
            # spread (long code_a, short hedge_ratio*code_b)
            confidence = min(1.0, abs(z) / (self.entry_z * 2))
            return self._signal(
                coint, SpreadSide.LONG_SPREAD, z, self.entry_z, self.exit_z,
                confidence, timestamp,
                meta={"cadf_tstat": coint.cadf_tstat, "half_life_days": coint.half_life_days},
            )
        if z >= self.entry_z:
            confidence = min(1.0, abs(z) / (self.entry_z * 2))
            return self._signal(
                coint, SpreadSide.SHORT_SPREAD, z, self.entry_z, self.exit_z,
                confidence, timestamp,
                meta={"cadf_tstat": coint.cadf_tstat, "half_life_days": coint.half_life_days},
            )
        return None
