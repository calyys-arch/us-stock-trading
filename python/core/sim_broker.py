"""
SimBroker — in-memory fill simulator used when no live/paper broker is
configured (unit tests, dry-run mode, CI). Fills every order immediately at
the requested price with no slippage — realistic slippage/impact modeling
belongs in the backtest engines (python/core/fees_equity.py), not here;
this class exists purely so execution_gateway.py has something to call
during development without requiring IB Gateway to be running.
"""
from __future__ import annotations

import itertools
import logging

log = logging.getLogger(__name__)

_order_ids = itertools.count(1)


class SimBroker:
    def __init__(self) -> None:
        self._positions: dict[str, int] = {}

    def place_order(
        self,
        code: str,
        side: str,
        qty: int,
        limit_price: float = 0.0,
        order_type: str = "market",
        tif: str = "DAY",
    ) -> dict:
        order_id = str(next(_order_ids))
        signed = qty if side.lower() == "buy" else -qty
        self._positions[code] = self._positions.get(code, 0) + signed
        log.info("SimBroker: filled %s %s x%d @ %.2f (order_id=%s)", side, code, qty, limit_price, order_id)
        return {
            "accepted": True,
            "order_id": order_id,
            "reason": None,
            "filled_qty": signed,
            "avg_fill_price": limit_price,
        }

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self) -> dict:
        return dict(self._positions)

    def get_account_summary(self) -> dict:
        return {"NetLiquidation": 1_000_000.0, "BuyingPower": 4_000_000.0}

    def disconnect(self) -> None:
        pass
