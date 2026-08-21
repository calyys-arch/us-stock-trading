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

OPT-IN EXIT ABLATIONS (2026-08-13) — all DEFAULT OFF
----------------------------------------------------
`backtests/reports/pairs_scan_report.md` measured the exit mix on the
scanned universe and found **91-96% of positions exit on the stale timeout,
not on z-reversion**: the O-U half-life describes its own estimation window
well and predicts actual reversion time badly, so the timeout fires at an
arbitrary point rather than at a considered one. Three exit rules were added
to test whether that is fixable. Every one of them is off by default, and
with the defaults `check_exits` is byte-for-byte the rule it always was
(pinned by tests/test_pairs_exit_rules.py::
test_defaults_reproduce_the_legacy_exit_rule_exactly).

  1. `reestimate_half_life()` — re-derive `max_holding_days` from a FRESH
     half-life estimate while the position is open, instead of freezing it
     at open time from an estimate that may already have been up to
     `revalidate_every_days` old. Changes how `half_life_multiplier_max_hold`
     is APPLIED, not what it is: no new parameter.
  2. `broken_pairs` — exit immediately when the pair stops passing the
     existing `is_tradeable` screen, rather than waiting out a timeout on a
     relationship the scan no longer believes in. A boolean design option,
     not a tunable threshold. Reason: "coint_breakdown".
  3. `stop_z` — exit when the spread blows THROUGH the entry point rather
     than reverting. **This directly contradicts the Chan-derived design
     stated above and is the one change here that is a policy decision, not
     a mechanical one. It requires explicit human sign-off before being
     enabled anywhere, independently of what the numbers say.** Reason:
     "stop_loss".

The measured result of all three is in `pairs_scan_report.md` §2026-08-13.
Summary: none of them rescues the strategy, which is why they all remain
off. They are kept, wired and tested rather than deleted so that the
negative result is reproducible.
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

    def reestimate_half_life(self, code_a: str, code_b: str, half_life_days: float) -> bool:
        """Re-derive an OPEN position's stale-timeout budget from a fresh
        half-life estimate. Returns False if the pair is not open.

        Opt-in ablation 1 (see module docstring). `max_holding_days` is
        normally frozen at open time from the half-life the pair-selection
        scan produced, which by then can be up to `revalidate_every_days`
        stale and is never revised however long the position is held. This
        recomputes the SAME quantity — `half_life_multiplier * half_life` —
        from the current estimate, so the remaining allowance
        (`max_holding_days - holding_days`) tracks the latest read on how
        fast this spread actually reverts.

        Staleness is still measured from `entry_time`, so a downward
        revision can leave the remaining allowance at or below zero and the
        position exits on the next `check_exits`. That is the intended
        behavior, not an edge case: "this spread reverts faster than I
        thought, so I have already held it long enough" is exactly the
        information the fixed timeout throws away.

        The caller is responsible for only ever passing an estimate that was
        computable at the current bar — this method has no notion of time and
        cannot enforce that.
        """
        pos = self._open.get(self._key(code_a, code_b))
        if pos is None:
            return False
        pos.half_life_days = float(half_life_days)
        pos.max_holding_days = max(float(half_life_days), 1.0) * self._half_life_multiplier
        return True

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
        stop_z: float | None = None,
        broken_pairs: set | None = None,
        stale_timeout_enabled: bool = True,
    ) -> list[tuple[OpenPairPosition, str]]:
        """Returns [(position, reason)] for every pair that should be closed
        NOW. Reason is one of "z_reversion" | "stop_loss" | "coint_breakdown"
        | "stale_timeout".

        With the default arguments this is exactly the original rule —
        z-reversion or holding-period staleness, and never "stop_loss",
        because this strategy family's documented design has no price-based
        stop. The three optional arguments are the opt-in ablations described
        in the module docstring; `stop_z` in particular is a POLICY change,
        not a mechanical one, and must not be enabled without human sign-off.

        Precedence, when more than one rule fires on the same bar:
          z_reversion > stop_loss > coint_breakdown > stale_timeout
        Reversion outranks everything because it is the profitable exit the
        whole thesis is about — a pair that reverted AND simultaneously fell
        out of the tradeable set should book the reversion, not be relabelled
        a breakdown. The remaining three all close at the same price on the
        same bar, so their order changes only the attribution, never the P&L.

        `stop_z` is an ABSOLUTE |z| level (the engine derives it as
        `entry_z * stop_z_multiple`), so this class stays ignorant of
        `entry_z`. It fires only when the spread is beyond that level AND
        strictly beyond its own entry z — without the second condition a
        position entered at z = -4.0 under a stop at 3.0 would be stopped out
        on its first bar, which measures the entry distribution rather than
        the "kept widening after I entered" behavior the stop is meant to
        catch.
        """
        broken = broken_pairs or ()
        to_close: list[tuple[OpenPairPosition, str]] = []
        for pos in list(self._open.values()):
            key = (pos.code_a, pos.code_b)
            z = current_z_by_pair.get(key)
            if z is not None:
                reverted = (
                    (pos.side == SpreadSide.LONG_SPREAD and z >= -exit_z_threshold)
                    or (pos.side == SpreadSide.SHORT_SPREAD and z <= exit_z_threshold)
                )
                if reverted:
                    to_close.append((pos, "z_reversion"))
                    continue
                if stop_z is not None and abs(z) > abs(pos.entry_z):
                    widened = (
                        (pos.side == SpreadSide.LONG_SPREAD and z <= -abs(stop_z))
                        or (pos.side == SpreadSide.SHORT_SPREAD and z >= abs(stop_z))
                    )
                    if widened:
                        to_close.append((pos, "stop_loss"))
                        continue
            if key in broken:
                to_close.append((pos, "coint_breakdown"))
                continue
            if stale_timeout_enabled and pos.is_stale(now):
                to_close.append((pos, "stale_timeout"))
        return to_close

    @property
    def open_positions(self) -> list[OpenPairPosition]:
        return list(self._open.values())
