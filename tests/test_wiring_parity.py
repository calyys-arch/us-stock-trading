"""
Wiring parity tests — forex-trading lesson #1
(docs/lessons_from_forex_trading.md): three forex strategies were silently
dead in live trading for months because the MarketSnapshot fields they
depended on (`has_volume_spike_60m`, `vol_ratio`, `breakouts_today`) were
never actually populated with meaningful values by DataEngine, even though
the types.py schema declared them.

This system's RiskEngine is the component that reads MarketSnapshot fields
directly (sector, adv_20d_dollars, short_locate_available, is_halted,
is_regular_trading_hours, price) to gate every order — if DataEngine ever
regresses to leaving one of these at its dataclass default, RiskEngine would
silently reject/approve orders based on wrong information. This test proves,
end-to-end through a real DataEngine instance, that every field RiskEngine
depends on is populated with the INJECTED reference-data value, not the
class default.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from python.core.bus import MessageBus
from python.core.data_engine import DataEngine, ReferenceData
from python.core.types import MarketSnapshot, Tick

# Every field RiskEngine.qualify_spread_order / qualify_portfolio_order reads
# directly off a MarketSnapshot. If RiskEngine grows to read a new field,
# add it here — that is the enforcement mechanism (a reviewer/CI failure
# forces this list to be kept honest).
RISK_ENGINE_SNAPSHOT_FIELDS = [
    "price",
    "sector",
    "adv_20d_dollars",
    "short_locate_available",
    "is_halted",
    "is_hard_to_borrow",
    "is_regular_trading_hours",
]

# Default (dataclass-default) values for each field above — if the built
# snapshot still equals this default, the field was never actually wired up.
_DEFAULTS = {
    "price": 0.0,
    "sector": "",
    "adv_20d_dollars": 0.0,
    "short_locate_available": True,   # default True; injected test value is False
    "is_halted": False,
    "is_hard_to_borrow": False,       # default False; injected test value is True
    "is_regular_trading_hours": None,  # depends on wall-clock; checked separately
}


def _feed_ticks_and_get_snapshot(reference_data: ReferenceData) -> MarketSnapshot:
    bus = MessageBus()
    captured: list[MarketSnapshot] = []

    async def _capture(snap: MarketSnapshot) -> None:
        captured.append(snap)

    bus.subscribe("snapshot", _capture)
    engine = DataEngine(bus, reference_data=reference_data, snapshot_throttle_sec=0.0, primary_tf=1)

    async def _run():
        base_ts = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)  # 09:30 ET, RTH
        for i in range(5):
            # Advance by a full minute per tick so CandleBuilder(primary_tf=1)
            # closes a bar between ticks — otherwise all 5 ticks land in the
            # SAME in-progress 1-minute bucket and _build_snapshot's
            # `len(candles) < 2` guard never clears (candles never published).
            tick = Tick(
                code="TESTCO", price=123.45 + i * 0.01, volume=1000,
                bid=123.40, ask=123.50, timestamp=base_ts + timedelta(minutes=i), source="live",
            )
            await engine.process_tick(tick)
        await asyncio.sleep(0)  # let published tasks run

    asyncio.run(_run())
    assert captured, "DataEngine never published a snapshot — cannot test wiring"
    return captured[-1]


def test_risk_engine_fields_are_populated_with_injected_reference_data():
    """Every field RiskEngine depends on must reflect the INJECTED
    ReferenceData value, not the MarketSnapshot dataclass default."""
    ref = ReferenceData(
        sector=lambda code: "Information Technology",
        adv_20d_dollars=lambda code: 75_000_000.0,
        short_locate_available=lambda code: False,   # deliberately non-default
        is_hard_to_borrow=lambda code: True,          # deliberately non-default
    )
    snap = _feed_ticks_and_get_snapshot(ref)

    assert snap.price > 0.0
    assert snap.sector == "Information Technology"
    assert snap.adv_20d_dollars == 75_000_000.0
    assert snap.short_locate_available is False
    assert snap.is_hard_to_borrow is True
    # is_regular_trading_hours is calendar-derived, not reference-data —
    # just confirm it was actually SET (not left at some uninitialized value).
    assert isinstance(snap.is_regular_trading_hours, bool)


def test_risk_engine_field_list_stays_in_sync_with_source():
    """If RiskEngine starts reading a new MarketSnapshot attribute that is
    not in RISK_ENGINE_SNAPSHOT_FIELDS above, this test fails loudly instead
    of the gap going unnoticed (mirrors the forex incident where nobody
    noticed three fields were unwired for months)."""
    import inspect

    from python.core import risk_engine as risk_engine_module

    source = inspect.getsource(risk_engine_module)
    # Heuristic: find every "snapshot_a.<attr>" / "snapshot_b.<attr>" / "snap.<attr>" access.
    import re

    accessed = set(re.findall(r"(?:snapshot_a|snapshot_b|snap)\.([a-z_][a-z0-9_]*)", source))
    # Remove non-field method-like accesses that aren't dataclass fields.
    accessed -= {"is_tradeable"}  # property, composed of already-covered fields

    missing = accessed - set(RISK_ENGINE_SNAPSHOT_FIELDS)
    assert not missing, (
        f"risk_engine.py reads MarketSnapshot field(s) {missing} that are not "
        f"covered by RISK_ENGINE_SNAPSHOT_FIELDS in this test — update both "
        f"the test and confirm DataEngine actually populates them."
    )
