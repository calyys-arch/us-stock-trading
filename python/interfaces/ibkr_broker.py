"""
IBKR broker adapter for US equities (paper or live).

Adapted from forex-trading/python/interfaces/ibkr_broker.py. The threading
architecture (dedicated background asyncio event loop bridged via
run_coroutine_threadsafe, so the rest of the system can call this
synchronously) is asset-agnostic and kept as-is. Forex-specific order
pathologies documented in the original (IDEALPRO odd-lot rejections, the
15-working-orders-per-side cap that caused a real incident) are NOT known
issues for US-equity SMART-routed orders, so that defensive complexity is
intentionally NOT ported — this module is deliberately simpler. If similar
pathologies are observed in production for equities, port the equivalent
guard from the forex version rather than re-inventing it.

Contracts: `Stock(symbol, "SMART", "USD")`.

Client IDs: IbkrFeed uses client_id=11 (data), IbkrBroker uses client_id=21
(orders) — must differ, matching the forex convention.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)

_FILL_TIMEOUT = 15.0
_RECONNECT_TIMEOUT = 10.0


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class IbkrBroker:
    """Synchronous IBKR broker adapter for US equities.

    All public methods are blocking and thread-safe; internally they run a
    coroutine on a dedicated background event loop.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 21) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._lock = threading.Lock()
        self._connect_lock: Optional[asyncio.Lock] = None
        self._ib = None
        self._connected = False
        self.last_positions_fetch_ok: bool = True

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="ibkr-broker-loop")
        self._thread.start()

        # asyncio.Lock() must be created on the loop it will be used from.
        self._connect_lock = self._run(self._make_lock())
        self._connect()
        log.info(
            "IbkrBroker initialised (US equities) host=%s port=%d client_id=%d connected=%s",
            host, port, client_id, self._connected,
        )

    async def _make_lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    def _run(self, coro, timeout: float = 30.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _on_ib_error(self, reqId, errorCode, errorString, contract) -> None:
        log.warning("IbkrBroker: API error reqId=%s code=%s msg=%s contract=%s", reqId, errorCode, errorString, contract)

    async def _do_connect(self) -> None:
        async with self._connect_lock:
            if self._connected and self._ib and self._ib.isConnected():
                return
            from ib_async import IB

            ib = IB()
            ib.errorEvent += self._on_ib_error
            await ib.connectAsync(self._host, self._port, clientId=self._client_id)
            self._ib = ib
            self._connected = True
            log.info("IbkrBroker: connected to %s:%d", self._host, self._port)

    def _connect(self) -> None:
        try:
            self._run(self._do_connect(), timeout=_RECONNECT_TIMEOUT)
        except Exception as exc:
            log.error("IbkrBroker: connection failed: %s", exc)
            self._connected = False

    async def _ensure_connected_async(self) -> None:
        if not self._connected or (self._ib and not self._ib.isConnected()):
            log.warning("IbkrBroker: not connected — reconnecting…")
            try:
                await self._do_connect()
            except Exception as exc:
                log.error("IbkrBroker: reconnect failed: %s", exc)
                self._connected = False

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        code: str,
        side: str,
        qty: int,
        limit_price: float = 0.0,
        order_type: str = "market",
        tif: str = "DAY",
    ) -> dict:
        """Submit an order for a US equity. `order_type`: "market" | "limit".
        DAY time-in-force by default (matches Chan's intraday-flatten design
        for Strategy B; Strategy A's overnight holds use explicit GTC on the
        exit order instead of relying on TIF to survive past session close).

        Returns {"accepted": bool, "order_id": str|None, "reason": str|None,
        "filled_qty": int, "avg_fill_price": float}.
        """
        action = "BUY" if side.lower() == "buy" else "SELL"
        contract_holder: dict = {}

        async def _place_and_wait(order):
            trade = self._ib.placeOrder(contract_holder["contract"], order)
            deadline = self._loop.time() + _FILL_TIMEOUT
            while self._loop.time() < deadline:
                await asyncio.sleep(0.25)
                status = trade.orderStatus.status
                if status in ("Filled", "Cancelled", "Inactive"):
                    break
            return trade

        async def _submit():
            from ib_async import Stock, MarketOrder, LimitOrder

            await self._ensure_connected_async()
            contract = Stock(code.upper(), "SMART", "USD")
            await self._ib.qualifyContractsAsync(contract)
            contract_holder["contract"] = contract

            if order_type == "limit" and limit_price > 0:
                order = LimitOrder(action, qty, round(limit_price, 2))
            else:
                order = MarketOrder(action, qty)
            order.tif = tif

            trade = await _place_and_wait(order)
            return trade

        try:
            with self._lock:
                trade = self._run(_submit(), timeout=_FILL_TIMEOUT + 10.0)
        except Exception as exc:
            log.error("IbkrBroker.place_order: %s %s x%d failed: %s", action, code, qty, exc)
            return {"accepted": False, "order_id": None, "reason": str(exc), "filled_qty": 0, "avg_fill_price": 0.0}

        status = trade.orderStatus.status
        filled = int(trade.orderStatus.filled or 0)
        avg_price = _safe_float(trade.orderStatus.avgFillPrice, 0.0)
        accepted = status == "Filled" or filled > 0

        return {
            "accepted": accepted,
            "order_id": str(trade.order.orderId),
            "reason": None if accepted else f"status={status}",
            "filled_qty": filled if action == "BUY" else -filled,
            "avg_fill_price": avg_price,
        }

    def cancel_order(self, order_id: str) -> bool:
        async def _cancel():
            await self._ensure_connected_async()
            for t in self._ib.openTrades():
                if str(t.order.orderId) == str(order_id):
                    self._ib.cancelOrder(t.order)
                    return True
            return False

        try:
            return bool(self._run(_cancel(), timeout=10.0))
        except Exception as exc:
            log.error("IbkrBroker.cancel_order: %s failed: %s", order_id, exc)
            return False

    def get_positions(self) -> dict:
        """Returns {code: signed_share_qty}."""
        async def _fetch():
            await self._ensure_connected_async()
            return self._ib.positions()

        try:
            raw = self._run(_fetch(), timeout=10.0)
            self.last_positions_fetch_ok = True
        except Exception as exc:
            log.error("IbkrBroker.get_positions failed: %s", exc)
            self.last_positions_fetch_ok = False
            return {}

        out: dict[str, int] = {}
        for p in raw:
            try:
                if p.contract.secType == "STK":
                    out[p.contract.symbol] = int(p.position)
            except Exception:
                continue
        return out

    def get_account_summary(self) -> dict:
        async def _fetch():
            await self._ensure_connected_async()
            return await self._ib.accountSummaryAsync()

        try:
            rows = self._run(_fetch(), timeout=10.0)
        except Exception as exc:
            log.error("IbkrBroker.get_account_summary failed: %s", exc)
            return {}

        out: dict[str, float] = {}
        for r in rows:
            if r.tag in ("NetLiquidation", "BuyingPower", "AvailableFunds", "GrossPositionValue"):
                out[r.tag] = _safe_float(r.value)
        return out

    def disconnect(self) -> None:
        try:
            if self._ib and self._ib.isConnected():
                self._run(self._async_disconnect(), timeout=5.0)
        except Exception:
            pass

    async def _async_disconnect(self) -> None:
        self._ib.disconnect()
        self._connected = False
