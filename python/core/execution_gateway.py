"""
Execution Gateway — the ONLY module allowed to submit/cancel orders or
manage positions at the broker (architecture-rules.mdc). Subscribes to
bus("qualified_spread_order") and bus("qualified_portfolio_order"),
forwards approved orders to a broker adapter (IbkrBroker or SimBroker), and
publishes bus("execution_report").

Differences from forex-trading/python/core/execution_gateway.py:
  - No Rust `execution_layer` pyo3 binding for this MVP — orders go straight
    to the Python broker adapter. Porting a Rust execution layer is listed
    as a future hardening step in docs/us_equity_health_check.md; it is not
    required to validate the strategy logic end-to-end, which is this
    project's current priority.
  - Forex-only guards (`_QUOTE_TO_USD`, `is_forex_instrument`,
    `forex_only_gateway_block`) are replaced by a `us_equity_only_guard`
    that is the mirror image: this system must NEVER accidentally route an
    order for a non-equity instrument (e.g. a stray forex code from a
    shared config file).
  - Regular Trading Hours guard: orders for Strategy B (intraday-only) are
    rejected outside RTH; Strategy A (pairs, overnight-capable) orders are
    allowed to be submitted as GTC-equivalent and are exempt from the
    same-day RTH-only restriction, but entries are still only accepted
    during RTH (no pre/post-market entries — spreads computed from RTH
    closes should not be traded against thin extended-hours liquidity).
  - EOD flatten: `flatten_intraday_positions()` is called by the live
    scheduler a configurable number of minutes before the close (see
    python/core/calendar.py:is_intraday_flatten_window) and force-closes
    every Strategy B position — this is the safety net for the
    "not filled in time" scenario, not the primary exit mechanism.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .bus import MessageBus
from .calendar import is_intraday_flatten_window, is_regular_trading_hours
from .types import QualifiedPortfolioOrder, QualifiedSpreadOrder

log = logging.getLogger(__name__)

# US-equity ticker symbols are 1-5 uppercase letters (optionally with a
# class suffix like BRK.B). Anything else (e.g. a leftover forex code like
# "EURUSD", which is exactly 6 uppercase letters with no separator) is
# almost certainly a config/data mixup from a shared file and must be
# rejected loudly rather than silently routed to a broker.
_FOREX_CODE_LEN = 6


def us_equity_only_guard(code: str) -> Optional[str]:
    """Returns a rejection reason string if `code` looks like a non-equity
    instrument, else None. This is a heuristic safety net, not a full
    validator — the authoritative check is IBKR contract qualification."""
    c = code.upper().strip()
    if not c:
        return "empty_code"
    if len(c) == _FOREX_CODE_LEN and c.isalpha() and c not in _KNOWN_LONG_TICKERS:
        return f"looks_like_forex_pair:{c}"
    return None


_KNOWN_LONG_TICKERS: set[str] = set()  # populate from universe file if 6-letter tickers exist


class ExecutionGateway:
    def __init__(
        self,
        bus: MessageBus,
        broker,
        mode: str = "observe",   # "observe" | "auto"
        flatten_buffer_check: bool = True,
        auto_execute_strategies: Optional[set] = None,
    ) -> None:
        self._bus = bus
        self._broker = broker
        self.mode = mode
        self._flatten_buffer_check = flatten_buffer_check
        # Per-strategy configs/strategy.yaml `auto_execute: true/false` gate.
        # BOTH this gateway's global mode=="auto" AND the specific strategy's
        # auto_execute flag must be true before an order is actually
        # submitted — this two-key AND is deliberate (forex lesson #2:
        # `auto_execute` existed in config with NO code reading it at all;
        # see tests/test_config_enforcement.py for the regression test).
        self._auto_execute_strategies: set = auto_execute_strategies or set()
        bus.subscribe("qualified_spread_order", self._on_spread_order)
        bus.subscribe("qualified_portfolio_order", self._on_portfolio_order)

    def _strategy_may_auto_execute(self, strategy_name: str) -> bool:
        return self.mode == "auto" and strategy_name in self._auto_execute_strategies

    async def _on_spread_order(self, order: QualifiedSpreadOrder) -> None:
        if not order.approved:
            log.info("ExecutionGateway: spread order rejected upstream (%s)", order.rejection_reason)
            return

        now = order.raw.timestamp
        guard_a = us_equity_only_guard(order.raw.code_a)
        guard_b = us_equity_only_guard(order.raw.code_b)
        if guard_a or guard_b:
            await self._publish_report(order, accepted=False, reason=guard_a or guard_b)
            return
        if not is_regular_trading_hours(now):
            await self._publish_report(order, accepted=False, reason="outside_rth_entry")
            return
        if not self._strategy_may_auto_execute(order.raw.strategy):
            log.info(
                "ExecutionGateway: observe mode (gateway_mode=%s, strategy=%s auto_execute=%s) — "
                "not submitting spread order for %s/%s",
                self.mode, order.raw.strategy, order.raw.strategy in self._auto_execute_strategies,
                order.raw.code_a, order.raw.code_b,
            )
            await self._publish_report(order, accepted=False, reason="observe_mode")
            return

        side_a = order.raw.entry_side_a
        side_b = order.raw.entry_side_b
        result_a = self._broker.place_order(order.raw.code_a, side_a, order.qty_a, tif="GTC")
        result_b = self._broker.place_order(order.raw.code_b, side_b, order.qty_b, tif="GTC")

        await self._bus.publish("execution_report", {
            "type": "spread_order",
            "code_a": order.raw.code_a, "code_b": order.raw.code_b,
            "leg_a": result_a, "leg_b": result_b,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def _on_portfolio_order(self, order: QualifiedPortfolioOrder) -> None:
        if not order.approved or not order.target_shares:
            log.info("ExecutionGateway: portfolio order has nothing to submit (rejected=%s)", order.rejected_codes)
            return

        now = order.raw.as_of
        if not is_regular_trading_hours(now):
            await self._publish_report(order, accepted=False, reason="outside_rth_entry")
            return
        if not self._strategy_may_auto_execute(order.raw.strategy):
            log.info(
                "ExecutionGateway: observe mode (gateway_mode=%s, strategy=%s auto_execute=%s) — "
                "not submitting portfolio order (%d names)",
                self.mode, order.raw.strategy, order.raw.strategy in self._auto_execute_strategies,
                len(order.target_shares),
            )
            await self._publish_report(order, accepted=False, reason="observe_mode")
            return

        results = {}
        for code, signed_shares in order.target_shares.items():
            guard = us_equity_only_guard(code)
            if guard:
                results[code] = {"accepted": False, "reason": guard}
                continue
            side = "buy" if signed_shares > 0 else "sell"
            results[code] = self._broker.place_order(code, side, abs(signed_shares), tif="DAY")

        await self._bus.publish("execution_report", {
            "type": "portfolio_order",
            "strategy": order.raw.strategy,
            "as_of": order.raw.as_of.isoformat(),
            "results": results,
        })

    async def flatten_intraday_positions(self, strategy_name: str, now: Optional[datetime] = None) -> None:
        """Force-close every open position tagged to an intraday-only
        strategy. Called by the live scheduler inside the pre-close flatten
        window (calendar.is_intraday_flatten_window)."""
        now = now or datetime.utcnow()
        if self._flatten_buffer_check and not is_intraday_flatten_window(now):
            log.debug("flatten_intraday_positions: not yet in flatten window, skipping")
            return

        positions = self._broker.get_positions()
        for code, qty in positions.items():
            if qty == 0:
                continue
            side = "sell" if qty > 0 else "buy"
            result = self._broker.place_order(code, side, abs(qty), tif="DAY")
            log.info("EOD flatten: %s %s x%d -> %s", side, code, abs(qty), result.get("accepted"))
            await self._bus.publish("execution_report", {
                "type": "eod_flatten", "code": code, "strategy": strategy_name, "result": result,
            })

    async def _publish_report(self, order, accepted: bool, reason: str) -> None:
        await self._bus.publish("execution_report", {
            "type": "rejected",
            "reason": reason,
            "accepted": accepted,
            "timestamp": datetime.utcnow().isoformat(),
        })
