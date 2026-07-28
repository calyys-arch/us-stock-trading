"""
Abstract base classes for the two strategy shapes this system uses.

BaseStrategy — single-instrument, per-snapshot (ported pattern from
forex-trading/python/core/strategies/base.py). Kept for architecture parity
and any future momentum-overlay strategy; the pairs-trading strategy below
is deliberately NOT a BaseStrategy because a pair signal has two legs.

PairsStrategy — evaluates ONE pair (code_a, code_b) using a
CointegrationResult + a rolling spread series, emits Optional[SpreadSignal].

PortfolioStrategy — evaluates the WHOLE universe once per day, emits a
PortfolioTarget (target weight vector). This does not fit the per-snapshot
evaluate() shape at all — using it for a cross-sectional strategy was an
earlier design mistake and would silently prevent it from ever cross-
referencing other instruments.

All three are pure functions of their inputs — no I/O, no broker access,
no Greycat/DB access (per architecture-rules.mdc).
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ..types import (
    CointegrationResult,
    MarketSnapshot,
    MarketState,
    PortfolioTarget,
    RawSignal,
    SpreadSignal,
)


class BaseStrategy(ABC):
    """Single-instrument, per-snapshot strategy (not used by MVP strategies,
    kept for architecture parity / future momentum overlays)."""

    name: str = "base"
    timeframe: int | None = None

    @abstractmethod
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        market: MarketState,
    ) -> Optional[RawSignal]:
        """Return a RawSignal if entry conditions are met, else None."""
        ...

    def primary_candles(self, snap: MarketSnapshot) -> list:
        by_tf = getattr(snap, "candles_by_tf", None) or {}
        if self.timeframe is not None and self.timeframe in by_tf:
            return by_tf[self.timeframe]
        return snap.candles_primary

    def _signal(
        self,
        snapshot: MarketSnapshot,
        side: str,
        entry: float,
        stop: float,
        target: float,
        confidence: float,
        meta: Optional[dict] = None,
    ) -> RawSignal:
        return RawSignal(
            id=str(uuid.uuid4()),
            code=snapshot.code,
            strategy=self.name,
            side=side,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            metadata=meta or {},
        )


class PairsStrategy(ABC):
    """One instance evaluates ONE cointegrated pair.

    A PairScanner (python/stat/pair_scanner.py) is responsible for finding
    and periodically re-validating candidate pairs; this class only decides
    entry/exit given an already-validated CointegrationResult and the
    current spread.
    """

    name: str = "pairs_base"
    # Chan Ch.3 parameter-count discipline: see
    # python/core/strategies/pairs_trading.py module docstring for the exact
    # 5-parameter accounting across the full pairs pipeline (entry_z, exit_z,
    # half_life_multiplier_max_hold, min/max_half_life_days).
    max_free_parameters: int = 5

    @abstractmethod
    def evaluate(
        self,
        coint: CointegrationResult,
        spread_series: list,     # list[float], newest last, computed by caller
        current_price_a: float,
        current_price_b: float,
        timestamp: datetime,
    ) -> Optional[SpreadSignal]:
        """Return a SpreadSignal if entry/exit conditions are met, else None."""
        ...

    def _signal(
        self,
        coint: CointegrationResult,
        side,
        z_score: float,
        entry_z: float,
        exit_z: float,
        confidence: float,
        timestamp: datetime,
        meta: Optional[dict] = None,
    ) -> SpreadSignal:
        return SpreadSignal(
            id=str(uuid.uuid4()),
            strategy=self.name,
            code_a=coint.code_a,
            code_b=coint.code_b,
            hedge_ratio=coint.hedge_ratio,
            side=side,
            z_score=z_score,
            entry_z_threshold=entry_z,
            exit_z_threshold=exit_z,
            spread_mean=coint.spread_mean,
            half_life_days=coint.half_life_days,
            confidence=confidence,
            timestamp=timestamp,
            metadata=meta or {},
        )
