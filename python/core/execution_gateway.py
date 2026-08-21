"""
Execution Gateway — the ONLY module allowed to submit/cancel orders or
manage positions at the broker (architecture-rules.mdc). Subscribes to
bus("qualified_spread_order") and bus("qualified_portfolio_order"),
forwards approved orders to a broker adapter (IbkrBroker or SimBroker), and
publishes bus("execution_report").

Differences from forex-trading/python/core/execution_gateway.py:
  - No Rust `execution_layer` pyo3 binding for this MVP — orders go straight
    to the Python broker adapter. Porting a Rust execution layer is listed
    as a future hardening step in backtests/reports/us_equity_health_check.md; it is not
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
  - NEVER a market order (user decision, 2026-07-29 — "control exact
    execution price, avoid PFOF-routed market-order fills"): every path
    that reaches the broker goes through `_submit_order`, the ONE
    chokepoint in this class that actually calls `broker.place_order`. It
    hard-rejects (never silently falls back to a market order) whenever it
    doesn't have a valid limit price to submit. `SimBroker`/`IbkrBroker`
    still technically accept order_type="market" as a low-level capability
    (direct/unit-test callers), but nothing in this class's own call paths
    ever passes that — see tests/test_never_market_orders.py.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from .bus import MessageBus
from .calendar import is_intraday_flatten_window, is_regular_trading_hours
from .types import QualifiedMicroOrder, QualifiedPortfolioOrder, QualifiedSpreadOrder

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
        price_lookup: Optional[Callable[[str], float]] = None,
        risk_config=None,
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
        # Market-conditional live gate catalog (python/analytics/gate_policy.py).
        # Off by default so unit tests that only check auto_execute keep
        # working; EngineRuntime enables it for paper sessions.
        self._gate_policy_enabled: bool = False
        self._market_regime: str = "undecided"
        self._market_vol: str = "unknown"
        # flatten_intraday_positions/_close_position have no RiskEngine
        # (they act on already-open broker positions, not a fresh
        # signal->snapshot pipeline) so they need their OWN price source to
        # build a bounded limit order. None = "no price cache wired up yet"
        # -> those methods log a warning and skip rather than ever falling
        # back to a market order (see their docstrings).
        self._price_lookup = price_lookup
        from .risk_engine import load_risk_config

        # Default reads configs/risk.yaml (flatten/limit buffer bps etc.) —
        # NOT a bare RiskConfig() — so those keys are actually load-bearing,
        # not dead config (forex lesson #2).
        self._risk_cfg = risk_config or load_risk_config()
        self._cancel_tasks: dict[str, asyncio.Task] = {}
        bus.subscribe("qualified_spread_order", self._on_spread_order)
        bus.subscribe("qualified_portfolio_order", self._on_portfolio_order)
        bus.subscribe("qualified_micro_order", self._on_microstructure_order)

    def _strategy_may_auto_execute(self, strategy_name: str) -> bool:
        return self.mode == "auto" and strategy_name in self._auto_execute_strategies

    def enable_gate_policy(self, enabled: bool = True) -> None:
        self._gate_policy_enabled = bool(enabled)

    def set_market_regime(self, regime: str) -> None:
        self._market_regime = str(regime)

    def set_market_vol(self, vol: str) -> None:
        self._market_vol = str(vol)

    def set_market_features(self, regime: str, vol: str) -> None:
        self._market_regime = str(regime)
        self._market_vol = str(vol)

    def _entry_allowed_by_gate_policy(self, strategy_name: str) -> tuple[bool, str]:
        if not self._gate_policy_enabled:
            return True, "gate_policy_disabled"
        from python.analytics.gate_policy import live_order_permitted
        return live_order_permitted(strategy_name, self._market_regime, self._market_vol)

    def set_mode(self, mode: str) -> None:
        """Runtime toggle for the gateway-level auto/observe gate — used by
        dashboard/engine_bridge.py's "Start Auto Trading" control. Note this
        is only HALF of the two-key AND (see __init__ docstring): a strategy
        still needs to be present in `_auto_execute_strategies` for its
        orders to actually go out."""
        if mode not in ("observe", "auto"):
            raise ValueError(f"ExecutionGateway.set_mode: invalid mode {mode!r}")
        log.warning("ExecutionGateway: mode %s -> %s", self.mode, mode)
        self.mode = mode

    def set_auto_execute_strategies(self, strategies: set) -> None:
        """Runtime override of the per-strategy auto-execute allowlist (the
        other half of the two-key AND). Used by the dashboard's "Start Auto
        Trading" button to arm real order submission for a session without
        editing configs/strategy.yaml on disk."""
        self._auto_execute_strategies = set(strategies)

    def set_broker(self, broker) -> None:
        """Swap the broker adapter at runtime — used by
        dashboard/engine_bridge.py to switch between SimBroker and
        IbkrBroker on Start/Stop without tearing down and re-subscribing
        this gateway's bus handlers. This is the ONLY sanctioned way to
        change the broker post-construction (architecture-rules.mdc: no
        other module may call broker APIs directly, but wiring which
        adapter this gateway forwards to is this class's own concern)."""
        self._broker = broker

    def set_price_lookup(self, price_lookup: Optional[Callable[[str], float]]) -> None:
        """Wire (or clear) the price source flatten_intraday_positions/
        _close_position use to build a bounded limit order. Until something
        calls this (e.g. dashboard/engine_bridge.py from a live snapshot
        cache), flatten calls degrade to "skip + warn", never a market
        order — see those methods' docstrings."""
        self._price_lookup = price_lookup

    def _submit_order(
        self,
        code: str,
        side: str,
        qty: int,
        limit_price: float,
        order_type: str = "limit",
        stop_price: float = 0.0,
        tif: str = "DAY",
    ) -> dict:
        """The ONLY place in this class that calls `self._broker.place_order`
        — every other method in this file must route through here. Hard-
        rejects (never a silent market-order fallback) whenever it doesn't
        have a valid price for the requested order_type. `order_type`
        "market" is refused outright even if a caller passes it by mistake
        — see module docstring's "NEVER a market order" section."""
        if order_type == "market":
            log.error("ExecutionGateway._submit_order: refusing a market order for %s %s x%d "
                      "(never submitted — see module docstring)", side, code, qty)
            return {"accepted": False, "order_id": None, "reason": "market_orders_disabled",
                    "filled_qty": 0, "avg_fill_price": 0.0}
        if order_type == "limit" and limit_price <= 0:
            return {"accepted": False, "order_id": None, "reason": "no_valid_limit_price",
                    "filled_qty": 0, "avg_fill_price": 0.0}
        if order_type == "stop_limit" and (limit_price <= 0 or stop_price <= 0):
            return {"accepted": False, "order_id": None, "reason": "no_valid_stop_limit_price",
                    "filled_qty": 0, "avg_fill_price": 0.0}
        return self._broker.place_order(
            code, side, qty, limit_price=limit_price, order_type=order_type, stop_price=stop_price, tif=tif,
        )

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
        policy_ok, policy_reason = self._entry_allowed_by_gate_policy(order.raw.strategy)
        if not policy_ok:
            log.info("ExecutionGateway: gate_policy blocked %s (%s)", order.raw.strategy, policy_reason)
            await self._publish_report(order, accepted=False, reason=f"gate_policy:{policy_reason}")
            return

        side_a = order.raw.entry_side_a
        side_b = order.raw.entry_side_b
        result_a = self._submit_order(order.raw.code_a, side_a, order.qty_a, order.limit_price_a, tif="GTC")
        result_b = self._submit_order(order.raw.code_b, side_b, order.qty_b, order.limit_price_b, tif="GTC")

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
        policy_ok, policy_reason = self._entry_allowed_by_gate_policy(order.raw.strategy)
        if not policy_ok:
            log.info("ExecutionGateway: gate_policy blocked %s (%s)", order.raw.strategy, policy_reason)
            await self._publish_report(order, accepted=False, reason=f"gate_policy:{policy_reason}")
            return

        results = {}
        for code, signed_shares in order.target_shares.items():
            guard = us_equity_only_guard(code)
            if guard:
                results[code] = {"accepted": False, "reason": guard}
                continue
            side = "buy" if signed_shares > 0 else "sell"
            limit_price = order.limit_prices.get(code, 0.0)
            results[code] = self._submit_order(code, side, abs(signed_shares), limit_price, tif="DAY")

        await self._bus.publish("execution_report", {
            "type": "portfolio_order",
            "strategy": order.raw.strategy,
            "as_of": order.raw.as_of.isoformat(),
            "results": results,
        })

    def _flatten_limit_price(self, code: str, side: str) -> float:
        """Bounded limit price for an urgent flatten/exit, from whatever
        `self._price_lookup` returns (see set_price_lookup's docstring) —
        wider buffer than a fresh entry (`flatten_limit_buffer_bps` >
        `limit_price_buffer_bps`) because flattening is urgency-driven, but
        STILL bounded, never unbounded like a market order. Returns 0.0
        (== "no valid price, caller must skip") when no price_lookup is
        wired up or it returns a non-positive price for `code`."""
        if self._price_lookup is None:
            return 0.0
        try:
            price = self._price_lookup(code)
        except Exception as exc:
            log.warning("ExecutionGateway._flatten_limit_price: price_lookup(%s) failed (%s)", code, exc)
            return 0.0
        if not price or price <= 0:
            return 0.0
        from .risk_engine import marketable_limit_price

        return marketable_limit_price(price, side, self._risk_cfg.flatten_limit_buffer_bps)

    async def _on_microstructure_order(self, order: QualifiedMicroOrder) -> None:
        """Entry is ALWAYS a limit order; the protective exit is ALWAYS a
        stop-limit (never a plain stop, which is itself a market order once
        triggered) — see RiskEngine.qualify_microstructure_order, which
        computes both prices from the signal's own stop/target. A take-
        profit target, if the signal supplied one, goes out as a third
        limit order. `cancel_after_seconds` schedules this entry's own
        cancellation if it's still unfilled after that long (a resting
        limit order with no adverse-price guarantee is not something we
        want sitting open indefinitely on a fast-moving intraday signal)."""
        if not order.approved:
            log.info("ExecutionGateway: micro order rejected upstream (%s)", order.rejection_reason)
            await self._publish_report(order, accepted=False, reason=order.rejection_reason or "rejected_upstream")
            return

        symbol = order.raw.symbol
        guard = us_equity_only_guard(symbol)
        if guard:
            await self._publish_report(order, accepted=False, reason=guard)
            return
        if not is_regular_trading_hours(order.raw.signal_time):
            await self._publish_report(order, accepted=False, reason="outside_rth_entry")
            return
        if not self._strategy_may_auto_execute(order.raw.strategy):
            log.info("ExecutionGateway: observe mode (gateway_mode=%s, strategy=%s) — "
                     "not submitting micro order for %s", self.mode, order.raw.strategy, symbol)
            await self._publish_report(order, accepted=False, reason="observe_mode")
            return
        policy_ok, policy_reason = self._entry_allowed_by_gate_policy(order.raw.strategy)
        if not policy_ok:
            log.info("ExecutionGateway: gate_policy blocked %s (%s)", order.raw.strategy, policy_reason)
            await self._publish_report(order, accepted=False, reason=f"gate_policy:{policy_reason}")
            return

        entry_side = "buy" if order.raw.direction == "long" else "sell"
        exit_side = "sell" if order.raw.direction == "long" else "buy"

        entry_result = self._submit_order(symbol, entry_side, order.qty, order.entry_limit_price, tif="DAY")
        exit_result = self._submit_order(
            symbol, exit_side, order.qty, order.stop_limit_price,
            order_type="stop_limit", stop_price=order.stop_price, tif="DAY",
        )
        target_result = None
        if order.target_price:
            target_result = self._submit_order(symbol, exit_side, order.qty, order.target_price, tif="DAY")

        if order.cancel_after_seconds and entry_result.get("order_id") and entry_result.get("filled_qty", 0) == 0:
            self._schedule_cancel(entry_result["order_id"], order.cancel_after_seconds)

        await self._bus.publish("execution_report", {
            "type": "micro_order",
            "symbol": symbol,
            "strategy": order.raw.strategy,
            "entry": entry_result,
            "protective_stop": exit_result,
            "target": target_result,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _schedule_cancel(self, order_id: str, delay_seconds: int) -> None:
        async def _cancel_later():
            await asyncio.sleep(delay_seconds)
            try:
                cancelled = self._broker.cancel_order(order_id)
                log.info("ExecutionGateway: auto-cancel order_id=%s after %ds -> %s",
                         order_id, delay_seconds, cancelled)
            except Exception as exc:
                log.warning("ExecutionGateway: auto-cancel order_id=%s failed (%s)", order_id, exc)
            finally:
                self._cancel_tasks.pop(order_id, None)

        self._cancel_tasks[order_id] = asyncio.ensure_future(_cancel_later())

    async def flatten_intraday_positions(self, strategy_name: str, now: Optional[datetime] = None) -> None:
        """Force-close every open position tagged to an intraday-only
        strategy. Called by the live scheduler inside the pre-close flatten
        window (calendar.is_intraday_flatten_window).

        NEVER a market order: if no price_lookup is wired up (see
        set_price_lookup), a position is SKIPPED — logged loudly as a
        warning — rather than force-closed at whatever price a market order
        would get. This is a deliberate trade-off (see module docstring):
        guaranteeing flatten completion by accepting unbounded execution
        price was explicitly rejected in favor of bounded, controlled
        exits."""
        now = now or datetime.utcnow()
        if self._flatten_buffer_check and not is_intraday_flatten_window(now):
            log.debug("flatten_intraday_positions: not yet in flatten window, skipping")
            return

        positions = self._broker.get_positions()
        for code, qty in positions.items():
            if qty == 0:
                continue
            side = "sell" if qty > 0 else "buy"
            limit_price = self._flatten_limit_price(code, side)
            if limit_price <= 0:
                log.warning("EOD flatten: no price available for %s — SKIPPING rather than "
                            "submitting a market order (wire ExecutionGateway.set_price_lookup)", code)
                continue
            result = self._submit_order(code, side, abs(qty), limit_price, tif="DAY")
            log.info("EOD flatten: %s %s x%d @ %.2f -> %s", side, code, abs(qty), limit_price, result.get("accepted"))
            await self._bus.publish("execution_report", {
                "type": "eod_flatten", "code": code, "strategy": strategy_name, "result": result,
            })

    async def _close_position(self, code: str, qty: int, report_type: str) -> dict:
        side = "sell" if qty > 0 else "buy"
        limit_price = self._flatten_limit_price(code, side)
        if limit_price <= 0:
            log.warning("FLATTEN (%s): no price available for %s — SKIPPING rather than submitting "
                        "a market order (wire ExecutionGateway.set_price_lookup)", report_type, code)
            result = {"accepted": False, "order_id": None, "reason": "no_price_lookup_configured",
                      "filled_qty": 0, "avg_fill_price": 0.0}
        else:
            result = self._submit_order(code, side, abs(qty), limit_price, tif="DAY")
        log.warning("FLATTEN (%s): %s %s x%d -> accepted=%s", report_type, side, code, abs(qty), result.get("accepted"))
        entry = {"code": code, "side": side, "qty": abs(qty), "result": result}
        await self._bus.publish("execution_report", {"type": report_type, **entry})
        return entry

    async def flatten_position(self, code: str) -> Optional[dict]:
        """Close a single open position immediately ("Exit" button next to
        one symbol in the dashboard's Positions panel) — independent of
        gateway mode/time-of-day, same as emergency_flatten_all but scoped
        to one code. Returns None if there is nothing open for that code."""
        qty = self._broker.get_positions().get(code, 0)
        if qty == 0:
            return None
        return await self._close_position(code, qty, "manual_flatten")

    async def emergency_flatten_all(self) -> list[dict]:
        """Immediately market-close EVERY open position at the broker,
        regardless of strategy, time-of-day, or gateway mode (this bypasses
        both the observe/auto gate and the EOD flatten-window check —
        unlike flatten_intraday_positions, this is a manual "get me flat
        right now" panic button, not a scheduled safety net). Returns one
        result dict per position closed, for the dashboard to display."""
        positions = self._broker.get_positions()
        results = []
        for code, qty in positions.items():
            if qty == 0:
                continue
            results.append(await self._close_position(code, qty, "emergency_flatten"))
        return results

    async def _publish_report(self, order, accepted: bool, reason: str) -> None:
        await self._bus.publish("execution_report", {
            "type": "rejected",
            "reason": reason,
            "accepted": accepted,
            "timestamp": datetime.utcnow().isoformat(),
        })
