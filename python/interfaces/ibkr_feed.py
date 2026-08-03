"""
Interactive Brokers feed for US equities — real bid/ask + real traded volume.

Adapted from forex-trading/python/interfaces/ibkr_feed.py. The single most
important difference from the forex version: US stocks have REAL traded
volume over IBKR (unlike Forex CASH contracts, which never send a volume
tick at all — see the forex module's now-corrected docstring history). This
means `ticker.volume` is meaningful here and volume-gated logic in
DataEngine/strategies is NOT permanently starved the way one forex strategy
was (see docs/lessons_from_forex_trading.md #1).

Setup
-----
1. Install IB Gateway or TWS, enable API (Configuration -> API -> Settings).
2. Ports: IB Gateway paper=4002, live=4001; TWS paper=7497, live=7496.
3. pip install ib_async  (ib_insync is archived/unmaintained).

Contracts
---------
Uses `Stock(symbol, "SMART", "USD")` — SMART routing across US equity
exchanges/ECNs, USD-denominated (this system is US-equity-only per
architecture-rules.mdc).

Regular Trading Hours
----------------------
`useRTH=True` is used for historical prewarm bars: Strategy B (cross-
sectional mean reversion) is explicitly RTH-only by construction (Chan's
example trades at the open and closes by the close), and pre/post-market
liquidity is thin enough that pre-market ticks would corrupt ADV/impact
estimates if mixed into intraday indicators.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from ..core.types import Tick
from .market_data import MarketDataFeed

log = logging.getLogger(__name__)


class IbkrFeed(MarketDataFeed):
    """Live US-equity price streaming via Interactive Brokers TWS/IB Gateway.

    Parameters
    ----------
    codes : list[str]
        Ticker symbols, e.g. ["AAPL", "MSFT"].
    host, port, client_id : IBKR socket connection parameters.
    poll_interval : seconds between each price-refresh poll.
    reconnect_delay : initial reconnect wait; doubles on repeated failure (max 60s).
    """

    def __init__(
        self,
        codes: list[str],
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 11,
        poll_interval: float = 2.0,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.codes = [c.upper() for c in codes]
        self._host = host
        self._port = port
        self._client_id = client_id
        self._poll_interval = poll_interval
        self._reconnect_delay = reconnect_delay
        log.info(
            "IbkrFeed initialised (US equities) host=%s port=%d client_id=%d codes=%s",
            host, port, client_id, self.codes,
        )

    class _IbkrFatalError(Exception):
        """Non-retryable error (bad credentials, wrong port, etc.)."""

    async def stream(self) -> AsyncIterator[Tick]:
        delay = self._reconnect_delay
        while True:
            try:
                async for tick in self._stream_once():
                    delay = self._reconnect_delay
                    yield tick
            except asyncio.CancelledError:
                log.info("IbkrFeed: stream cancelled")
                return
            except IbkrFeed._IbkrFatalError as exc:
                log.error("IbkrFeed: fatal error — %s", exc)
                return
            except Exception as exc:
                log.warning("IbkrFeed: connection lost (%s) — reconnecting in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)

    async def _stream_once(self) -> AsyncIterator[Tick]:
        try:
            from ib_async import IB, Stock  # type: ignore[import]
        except ImportError as exc:
            raise IbkrFeed._IbkrFatalError("ib_async not installed. Run: pip install ib_async") from exc

        ib = IB()
        contracts: list[tuple[str, object]] = []
        try:
            log.info("IbkrFeed: connecting to %s:%d client_id=%d", self._host, self._port, self._client_id)
            await ib.connectAsync(self._host, self._port, clientId=self._client_id)
            log.info("IbkrFeed: connected")
        except Exception as exc:
            raise ConnectionError(
                f"IbkrFeed: could not connect to {self._host}:{self._port} — {exc}. "
                "Is TWS/IB Gateway running with API enabled?"
            ) from exc

        try:
            ib.reqMarketDataType(3)  # delayed data fallback when no live subscription

            for code in self.codes:
                try:
                    contract = Stock(code, "SMART", "USD")
                    qualified = await ib.qualifyContractsAsync(contract)
                    if qualified:
                        contracts.append((code, qualified[0]))
                        log.info("IbkrFeed: qualified %s", code)
                    else:
                        log.warning("IbkrFeed: could not qualify %s — skipping", code)
                except Exception as exc:
                    log.warning("IbkrFeed: qualify error for %s: %s", code, exc)

            if not contracts:
                raise IbkrFeed._IbkrFatalError("IbkrFeed: no contracts could be qualified")

            tickers = {}
            for code, contract in contracts:
                ticker = ib.reqMktData(contract, genericTickList="", snapshot=False)
                tickers[code] = ticker
                log.info("IbkrFeed: subscribed to %s", code)

            log.info("IbkrFeed: streaming %d symbols", len(tickers))

            loop_count = 0
            yield_count = 0
            HEARTBEAT_INTERVAL = 30.0
            STALE_TIMEOUT = 90.0
            last_heartbeat = time.monotonic()
            last_price_ts = time.monotonic()

            while ib.isConnected():
                loop_count += 1
                skipped = 0
                yielded_this_poll = 0
                for code, ticker in tickers.items():
                    bid = getattr(ticker, "bid", None)
                    ask = getattr(ticker, "ask", None)
                    last = getattr(ticker, "last", None)
                    vol = getattr(ticker, "volume", None)

                    if bid is None or ask is None or math.isnan(bid) or math.isnan(ask) or bid <= 0 or ask <= 0:
                        # Fall back to last trade price with a synthetic tiny spread
                        # if the quote itself hasn't populated yet but a trade has.
                        if last is not None and not math.isnan(last) and last > 0:
                            bid, ask = last * 0.9999, last * 1.0001
                        else:
                            skipped += 1
                            continue

                    tick_vol = 1
                    if vol is not None and not math.isnan(vol) and vol > 0:
                        tick_vol = int(vol)

                    price = round((bid + ask) / 2.0, 2)

                    yield Tick(
                        code=code,
                        price=price,
                        volume=tick_vol,
                        bid=round(bid, 2),
                        ask=round(ask, 2),
                        timestamp=datetime.now(timezone.utc),
                        quote_ready=True,
                        source="ibkr_live",
                    )
                    yield_count += 1
                    yielded_this_poll += 1

                if yielded_this_poll > 0:
                    last_price_ts = time.monotonic()

                if loop_count <= 5 or loop_count % 20 == 0:
                    log.info(
                        "IbkrFeed: poll#%d yielded=%d skipped=%d connected=%s",
                        loop_count, yield_count, skipped, ib.isConnected(),
                    )

                now_mono = time.monotonic()
                if now_mono - last_heartbeat >= HEARTBEAT_INTERVAL:
                    last_heartbeat = now_mono
                    try:
                        await asyncio.wait_for(ib.reqCurrentTimeAsync(), timeout=10.0)
                    except Exception as hb_exc:
                        raise ConnectionError(f"IbkrFeed: heartbeat timeout — {hb_exc}")

                stale_secs = now_mono - last_price_ts
                if stale_secs >= STALE_TIMEOUT and loop_count > 10:
                    raise ConnectionError(f"IbkrFeed: no prices for {stale_secs:.0f}s")

                await asyncio.sleep(self._poll_interval)

            raise ConnectionError("IbkrFeed: IB connection dropped")

        finally:
            for _code, contract in contracts:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
            try:
                if ib.isConnected():
                    ib.disconnect()
                    log.info("IbkrFeed: disconnected")
            except Exception:
                pass

    def fetch_recent_bars_sync(
        self,
        code: str,
        bar_size: str = "1 day",
        duration: str = "2 Y",
        use_rth: bool = True,
        end_datetime: str = "",
    ):
        """Fetch recent historical daily/intraday bars for research/prewarm.
        Returns a list of ib_async BarData objects (caller converts to
        Candle/Tick as needed) — kept close to the raw IBKR response so
        callers can access adjusted close, volume, barCount, etc. directly.

        `end_datetime`: IB-format end anchor ("YYYYMMDD HH:MM:SS US/Eastern"
        or an ib_async-accepted datetime); empty string means "now". Without
        this parameter the method could only fetch windows ending at the
        present moment, which made it useless for arbitrary historical
        [start, end] backtest ranges (python/data/ibkr_price_source.py)."""
        try:
            from ib_async import IB, Stock  # type: ignore[import]
        except ImportError:
            log.error("IbkrFeed.fetch_recent_bars_sync: ib_async not installed")
            return []

        ib = IB()
        bars = []
        try:
            ib.connect(self._host, self._port, clientId=self._client_id + 99)
            contract = Stock(code, "SMART", "USD")
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="ADJUSTED_LAST",
                useRTH=use_rth,
                formatDate=1,
            )
            log.info("IbkrFeed.fetch_recent_bars_sync: %s -> %d bars", code, len(bars))
        except Exception as exc:
            log.warning("IbkrFeed.fetch_recent_bars_sync error for %s: %s", code, exc)
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass
        return bars
