"""
PairPositionManager — tracks open pairs-trading positions (two legs, one
hedge ratio, one spread-based exit). This does NOT submit orders (that is
execution_gateway.py's job per architecture-rules.mdc); it is the
authoritative in-memory record of "what pair positions are currently open"
that both the strategy (to avoid re-entering an already-open pair) and the
execution gateway (to know what to flatten and when) consult.

Exit logic (Chan Ch.7, p.132-133):
  - Take-profit-ish: |z_score| crosses back through `exit_z_threshold`
    (the spread reverted toward its mean) — the position is closed at a
    PROFIT if entry z and current z have the correct relative sign.
  - Chan explicitly argues AGAINST a stop-loss for a mean-reversion pair
    trade: if the spread has moved further against you, the reversion
    thesis says the SAME trade is now more attractively priced, not less.
    Instead, a stale-timeout exit close(s the position if it has not
    reverted within `max_holding_days` (a multiple of the pair's estimated
    half-life; Chan suggests this catches "the cointegration relationship
    has broken down" cases that a symmetric z-score exit alone would miss).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from .types import QualifiedSpreadOrder, SpreadSide

log = logging.getLogger(__name__)


@dataclass
class OpenPairPosition:
    pair_key: str            # f"{code_a}|{code_b}"
    code_a: str
    code_b: str
    hedge_ratio: float
    side: SpreadSide
    qty_a: int
    qty_b: int
    entry_price_a: float
    entry_price_b: float
    entry_z: float
    entry_time: datetime
    half_life_days: float
    max_holding_days: float
    strategy: str
    metadata: dict = field(default_factory=dict)

    def holding_days(self, now: datetime) -> float:
        return (now - self.entry_time).total_seconds() / 86400.0

    def is_stale(self, now: datetime) -> bool:
        return self.holding_days(now) >= self.max_holding_days

    def unrealized_pnl(self, price_a: float, price_b: float) -> float:
        leg_a_sign = 1 if self.side == SpreadSide.LONG_SPREAD else -1
        leg_b_sign = -leg_a_sign
        pnl_a = leg_a_sign * (price_a - self.entry_price_a) * self.qty_a
        pnl_b = leg_b_sign * (price_b - self.entry_price_b) * self.qty_b
        return pnl_a + pnl_b


class PairPositionManager:
    def __init__(self, half_life_multiplier_max_hold: float = 3.0) -> None:
        self._open: dict[str, OpenPairPosition] = {}
        self._half_life_multiplier = half_life_multiplier_max_hold

    @staticmethod
    def _key(code_a: str, code_b: str) -> str:
        return f"{code_a}|{code_b}"

    def is_open(self, code_a: str, code_b: str) -> bool:
        return self._key(code_a, code_b) in self._open

    def get(self, code_a: str, code_b: str) -> OpenPairPosition | None:
        return self._open.get(self._key(code_a, code_b))

    def open_position(
        self,
        order: QualifiedSpreadOrder,
        fill_price_a: float,
        fill_price_b: float,
        entry_time: datetime,
    ) -> OpenPairPosition:
        raw = order.raw
        key = self._key(raw.code_a, raw.code_b)
        if key in self._open:
            raise ValueError(f"PairPositionManager: pair {key} already open")

        max_hold = max(raw.half_life_days, 1.0) * self._half_life_multiplier
        pos = OpenPairPosition(
            pair_key=key,
            code_a=raw.code_a,
            code_b=raw.code_b,
            hedge_ratio=raw.hedge_ratio,
            side=raw.side,
            qty_a=order.qty_a,
            qty_b=order.qty_b,
            entry_price_a=fill_price_a,
            entry_price_b=fill_price_b,
            entry_z=raw.z_score,
            entry_time=entry_time,
            half_life_days=raw.half_life_days,
            max_holding_days=max_hold,
            strategy=raw.strategy,
        )
        self._open[key] = pos
        log.info(
            "pair OPEN %s side=%s qty_a=%d qty_b=%d entry_z=%.2f max_hold=%.1fd",
            key, raw.side.value, order.qty_a, order.qty_b, raw.z_score, max_hold,
        )
        return pos

    def close_position(self, code_a: str, code_b: str) -> OpenPairPosition | None:
        key = self._key(code_a, code_b)
        pos = self._open.pop(key, None)
        if pos is not None:
            log.info("pair CLOSE %s", key)
        return pos

    def check_exits(
        self,
        current_z_by_pair: dict,   # {(code_a, code_b): current_z_score}
        now: datetime,
        exit_z_threshold: float,
    ) -> list[tuple[OpenPairPosition, str]]:
        """Returns [(position, reason)] for every pair that should be closed
        NOW — either z-score reversion or holding-period staleness. Reason is
        one of "z_reversion" | "stale_timeout". Never returns "stop_loss" —
        this strategy family does not use price-based stops (see module
        docstring)."""
        to_close: list[tuple[OpenPairPosition, str]] = []
        for pos in list(self._open.values()):
            z = current_z_by_pair.get((pos.code_a, pos.code_b))
            if z is not None:
                reverted = (
                    (pos.side == SpreadSide.LONG_SPREAD and z >= -exit_z_threshold)
                    or (pos.side == SpreadSide.SHORT_SPREAD and z <= exit_z_threshold)
                )
                if reverted:
                    to_close.append((pos, "z_reversion"))
                    continue
            if pos.is_stale(now):
                to_close.append((pos, "stale_timeout"))
        return to_close

    @property
    def open_positions(self) -> list[OpenPairPosition]:
        return list(self._open.values())
