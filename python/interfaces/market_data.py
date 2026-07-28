"""
Market data adapters — abstract interface + simulated/CSV feeds for US
equities. Ported from forex-trading/python/interfaces/market_data.py; the
SimulatedFeed's volatility model is adjusted from forex spread-percentage
scaling to a simple equity-appropriate daily-vol-based random walk, and
default virtual session start is 09:30 ET (NYSE open) rather than a
timezone-naive forex 24h clock.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator

from ..core.types import Tick

log = logging.getLogger(__name__)


class MarketDataFeed(ABC):
    @abstractmethod
    async def stream(self) -> AsyncIterator[Tick]:
        """Async generator that yields Tick objects."""
        ...


class SimulatedFeed(MarketDataFeed):
    """Generates synthetic tick data for one or more equities. Used for
    smoke-testing the pipeline without a live broker connection."""

    def __init__(
        self,
        codes: list[str],
        base_prices: dict[str, float] | None = None,
        ticks_per_second: float = 5.0,
        duration_seconds: float = 60.0,
        random_seed: int = 42,
        virtual_start: datetime | None = None,
        virtual_tick_seconds: float = 20.0,
        annualized_vol: float = 0.30,
    ) -> None:
        self.codes = codes
        self.base_prices = base_prices or {c: 100.0 for c in codes}
        self.ticks_per_second = ticks_per_second
        self.duration_seconds = duration_seconds
        self.random_seed = random_seed
        self.virtual_tick_seconds = virtual_tick_seconds
        self.virtual_start = virtual_start or datetime(2026, 1, 2, 9, 30, 0)
        self.annualized_vol = annualized_vol
        random.seed(random_seed)

    async def stream(self) -> AsyncIterator[Tick]:
        interval = 1.0 / max(self.ticks_per_second, 0.001)
        end_time = datetime.utcnow() + timedelta(seconds=self.duration_seconds)
        prices = {c: self.base_prices.get(c, 100.0) for c in self.codes}
        virtual_ts = self.virtual_start
        virtual_step = timedelta(seconds=self.virtual_tick_seconds)

        # Per-tick sigma derived from annualized vol assuming ~390 trading
        # minutes/day and one tick == virtual_tick_seconds of elapsed time.
        seconds_per_year = 252 * 390 * 60
        while datetime.utcnow() < end_time:
            for code in self.codes:
                price = prices[code]
                sigma = price * self.annualized_vol * (self.virtual_tick_seconds / seconds_per_year) ** 0.5
                change = random.gauss(0, sigma)
                reversion = (self.base_prices[code] - price) * 0.001
                price = max(price + change + reversion, 0.01)
                prices[code] = price

                spread = max(price * 0.0005, 0.01)  # 5 bps typical equity spread
                volume = random.randint(100, 5_000)
                yield Tick(
                    code=code,
                    price=round(price, 2),
                    volume=volume,
                    bid=round(price - spread / 2, 2),
                    ask=round(price + spread / 2, 2),
                    timestamp=virtual_ts,
                )
            virtual_ts += virtual_step
            await asyncio.sleep(interval)


class CsvFeed(MarketDataFeed):
    """Replays ticks from a CSV file.
    Expected columns: code, price, volume, bid, ask, timestamp (ISO-8601)."""

    def __init__(self, path: str | Path, speed_multiplier: float = 1.0) -> None:
        self.path = Path(path)
        self.speed_multiplier = speed_multiplier

    async def stream(self) -> AsyncIterator[Tick]:
        ticks: list[Tick] = []
        with self.path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticks.append(Tick(
                    code=row["code"],
                    price=float(row["price"]),
                    volume=int(row["volume"]),
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                ))

        if not ticks:
            return

        for i, tick in enumerate(ticks):
            yield tick
            if i + 1 < len(ticks):
                gap = (ticks[i + 1].timestamp - tick.timestamp).total_seconds()
                await asyncio.sleep(max(gap / self.speed_multiplier, 0))
