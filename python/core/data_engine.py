"""
Data Engine — tick -> candles -> indicators -> MarketSnapshot -> bus("snapshot").

Ported from forex-trading/python/core/data_engine.py. CandleBuilder and
IndicatorEngine are fully generic and kept essentially unchanged. The
DataEngine class itself drops all forex-specific enrichment (pip size,
forex session, is_forex_instrument) and adds:

  - Regular Trading Hours gating via python/core/calendar.py (NYSE holidays/
    early closes/DST — forex-trading had no such concept, it ran 24/5).
  - US-equity reference-data enrichment (sector, market_cap, ADV, earnings-
    today, short-locate) via injected lookup callables, since DataEngine
    itself must stay a pure tick-processing component with no DB/network
    access (architecture-rules.mdc).

No Greycat access here (this repo has no Greycat dependency at all — MVP
strategies use tabular statistics, not a feature graph).
"""
from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Optional

from .bus import MessageBus
from .calendar import is_regular_trading_hours
from .types import Candle, MarketSnapshot, Tick

log = logging.getLogger(__name__)

TIMEFRAMES = (1, 5, 15, 60)
MAX_TICKS_PER_STOCK = 20_000
CANDLE_HISTORY = 500


# ── Candle builder (asset-agnostic; unchanged from forex-trading) ────────────

class CandleBuilder:
    """Accumulates ticks for one instrument/timeframe into closed Candle objects."""

    def __init__(self, code: str, timeframe_minutes: int) -> None:
        self.code = code
        self.tf = timeframe_minutes
        self._current: Optional[dict] = None
        self._history: deque[Candle] = deque(maxlen=CANDLE_HISTORY)

    def push(self, tick: Tick) -> Optional[Candle]:
        bucket = self._bucket(tick.timestamp)
        if self._current is None or self._current["bucket"] != bucket:
            closed = self._close()
            self._current = {
                "bucket": bucket,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.volume,
                "turnover": tick.price * tick.volume,
                "timestamp": tick.timestamp,
            }
            return closed

        c = self._current
        c["high"] = max(c["high"], tick.price)
        c["low"] = min(c["low"], tick.price)
        c["close"] = tick.price
        c["volume"] += tick.volume
        c["turnover"] += tick.price * tick.volume
        return None

    def last(self, n: int) -> list[Candle]:
        closed = list(self._history)
        if self._current is not None:
            c = self._current
            open_candle = Candle(
                code=self.code,
                timeframe=f"{self.tf}m",
                open=c["open"], high=c["high"],
                low=c["low"], close=c["close"],
                volume=c["volume"], turnover=c["turnover"],
                timestamp=c["timestamp"],
            )
            closed = closed + [open_candle]
        return closed[-n:]

    def _bucket(self, ts: datetime) -> int:
        # Absolute minutes since epoch (date-aware): avoids merging candles
        # across day/weekend boundaries that share a minute-of-day bucket.
        abs_minutes = ts.toordinal() * 1440 + ts.hour * 60 + ts.minute
        return (abs_minutes // self.tf) * self.tf

    def _close(self) -> Optional[Candle]:
        if self._current is None:
            return None
        c = self._current
        candle = Candle(
            code=self.code,
            timeframe=f"{self.tf}m",
            open=c["open"], high=c["high"], low=c["low"], close=c["close"],
            volume=c["volume"], turnover=c["turnover"], timestamp=c["timestamp"],
        )
        self._history.append(candle)
        return candle


# ── Indicator engine (asset-agnostic; unchanged from forex-trading) ──────────

class IndicatorEngine:
    """Stateless indicator calculations operating on plain lists."""

    @staticmethod
    def vwap(ticks: list[Tick]) -> float:
        total_vol = sum(t.volume for t in ticks)
        if total_vol == 0:
            return ticks[-1].price if ticks else 0.0
        total_pv = sum(
            (((t.bid + t.ask) / 2.0) if (getattr(t, "quote_ready", False) and t.bid > 0 and t.ask > 0) else t.price)
            * t.volume
            for t in ticks
        )
        return total_pv / total_vol

    @staticmethod
    def ema(closes: list[float], period: int) -> float:
        if not closes:
            return 0.0
        if len(closes) < period:
            return statistics.mean(closes)
        k = 2.0 / (period + 1)
        val = statistics.mean(closes[:period])
        for price in closes[period:]:
            val = price * k + val * (1 - k)
        return val

    @staticmethod
    def atr(candles: list[Candle], period: int) -> float:
        if len(candles) < 2:
            return 0.0
        trs = [
            max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i - 1].close),
                abs(candles[i].low - candles[i - 1].close),
            )
            for i in range(1, len(candles))
        ]
        return statistics.mean(trs[-period:]) if trs else 0.0

    @staticmethod
    def rsi(closes: list[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(-period, 0)]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]
        avg_gain = statistics.mean(gains) if gains else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0
        if avg_loss == 0:
            return 100.0
        return 100 - 100 / (1 + avg_gain / avg_loss)

    @staticmethod
    def bollinger(closes: list[float], period: int = 20, mult: float = 2.0) -> tuple[float, float, float]:
        if not closes:
            return 0.0, 0.0, 0.0
        window = closes[-period:]
        mid = statistics.mean(window)
        std = statistics.stdev(window) if len(window) > 1 else 0.0
        return mid + mult * std, mid, mid - mult * std


# ── Bad-tick filter (unchanged threshold logic from forex-trading) ───────────
# Gap opens (first live tick of a session) are EXEMPT — legitimate overnight
# news gaps on individual equities regularly exceed this threshold.
_BAD_TICK_INTRADAY_MAX_DEVIATION = 0.08
_BAD_TICK_RECENT_WINDOW = 20


class ReferenceData:
    """Injectable US-equity reference-data lookups. DataEngine stays a pure
    tick-processing component; anything requiring a DB/network call (sector
    classification, market cap, ADV, earnings calendar, short-locate) is
    provided by the caller (engine_bridge) as plain callables so this class
    remains trivially unit-testable and architecture-rules-compliant."""

    def __init__(
        self,
        sector: Callable[[str], str] | None = None,
        market_cap: Callable[[str], float] | None = None,
        shares_outstanding: Callable[[str], float] | None = None,
        adv_20d_dollars: Callable[[str], float] | None = None,
        is_earnings_today: Callable[[str], bool] | None = None,
        short_locate_available: Callable[[str], bool] | None = None,
        is_hard_to_borrow: Callable[[str], bool] | None = None,
        prev_close_adjusted: Callable[[str], float] | None = None,
    ) -> None:
        self.sector = sector or (lambda code: "")
        self.market_cap = market_cap or (lambda code: 0.0)
        self.shares_outstanding = shares_outstanding or (lambda code: 0.0)
        self.adv_20d_dollars = adv_20d_dollars or (lambda code: 0.0)
        self.is_earnings_today = is_earnings_today or (lambda code: False)
        self.short_locate_available = short_locate_available or (lambda code: True)
        self.is_hard_to_borrow = is_hard_to_borrow or (lambda code: False)
        self.prev_close_adjusted = prev_close_adjusted or (lambda code: 0.0)


class DataEngine:
    """Consumes raw Tick objects; produces MarketSnapshot objects on bus("snapshot")."""

    _SNAPSHOT_THROTTLE_SEC = 0.2

    def __init__(
        self,
        bus: MessageBus,
        reference_data: ReferenceData | None = None,
        snapshot_throttle_sec: float | None = None,
        news_event_checker: Callable[[str], bool] | None = None,
        primary_tf: int = 5,
    ) -> None:
        self._bus = bus
        self._ref = reference_data or ReferenceData()
        self._snapshot_throttle_sec = (
            self._SNAPSHOT_THROTTLE_SEC if snapshot_throttle_sec is None else max(0.0, snapshot_throttle_sec)
        )
        self._primary_tf: int = primary_tf if primary_tf in TIMEFRAMES else 5
        self._news_event_checker = news_event_checker
        self._ind = IndicatorEngine()
        self._ticks: dict[str, list[Tick]] = defaultdict(list)
        self._builders: dict[str, dict[int, CandleBuilder]] = defaultdict(
            lambda: {tf: CandleBuilder("", tf) for tf in TIMEFRAMES}
        )
        self._last_snapshot_ts: dict[str, float] = defaultdict(float)
        self._recent_prices: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_BAD_TICK_RECENT_WINDOW)
        )
        self._seen_first_live_tick: dict[str, bool] = {}
        self._vwap_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=120))
        self._atr5_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    async def process_tick(self, tick: Tick) -> None:
        code = tick.code

        is_prewarm = tick.source == "prewarm"
        is_bootstrap = tick.source == "bootstrap_snapshot"
        is_first_live = not self._seen_first_live_tick.get(code, False)

        if not is_prewarm and not is_bootstrap:
            if is_first_live:
                self._seen_first_live_tick[code] = True
            elif self._recent_prices[code]:
                prices = sorted(self._recent_prices[code])
                median = prices[len(prices) // 2]
                if median > 0:
                    deviation = abs(tick.price - median) / median
                    if deviation > _BAD_TICK_INTRADAY_MAX_DEVIATION:
                        log.warning(
                            "BAD TICK rejected %s: price=%.3f median=%.3f deviation=%.1f%%",
                            code, tick.price, median, deviation * 100,
                        )
                        return

        if tick.price > 0 and not is_prewarm:
            self._recent_prices[code].append(tick.price)

        if not self._ticks[code]:
            self._builders[code] = {tf: CandleBuilder(code, tf) for tf in TIMEFRAMES}

        ticks = self._ticks[code]
        ticks.append(tick)
        if len(ticks) > MAX_TICKS_PER_STOCK:
            del ticks[: len(ticks) - MAX_TICKS_PER_STOCK]

        for tf in TIMEFRAMES:
            self._builders[code][tf].push(tick)

        now = time.monotonic()
        if self._snapshot_throttle_sec > 0 and (
            now - self._last_snapshot_ts[code] < self._snapshot_throttle_sec
        ):
            return
        self._last_snapshot_ts[code] = now

        snap = self._build_snapshot(tick)
        if snap is not None:
            await self._bus.publish("snapshot", snap)

    def _build_snapshot(self, tick: Tick) -> Optional[MarketSnapshot]:
        code = tick.code
        candles = self._builders[code][self._primary_tf].last(50)
        if len(candles) < 2:
            return None

        candles_1m = self._builders[code][1].last(60)
        candles_primary = list(candles)
        candles_by_tf = {tf: self._builders[code][tf].last(60) for tf in TIMEFRAMES}

        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        bb_upper, bb_mid, bb_lower = self._ind.bollinger(closes)
        vol_ma20 = statistics.mean(vols[-20:]) if len(vols) >= 20 else statistics.mean(vols)

        if len(vols) >= 2:
            closed_vols = vols[:-1]
            closed_ma = statistics.mean(closed_vols[-20:]) if len(closed_vols) >= 20 else statistics.mean(closed_vols)
            vol_ratio = closed_vols[-1] / closed_ma if closed_ma > 0 else 1.0
        else:
            vol_ratio = 1.0

        current_vwap = self._ind.vwap(self._ticks[code][-500:])

        now_sec = time.monotonic()
        self._vwap_history[code].append((now_sec, current_vwap))
        vwap_slope_15m = 0.0
        hist = self._vwap_history[code]
        if len(hist) >= 2:
            slope_window_sec = 15 * 60
            old_t, old_v = hist[0]
            for t, v in hist:
                if now_sec - t <= slope_window_sec:
                    old_t, old_v = t, v
                    break
            elapsed = now_sec - old_t
            if elapsed > 30 and old_v > 0:
                vwap_slope_15m = (current_vwap - old_v) / old_v / elapsed

        has_news = self._news_event_checker(code) if self._news_event_checker else False

        has_vol_spike = False
        if len(candles_1m) >= 20:
            vols_1m = [c.volume for c in candles_1m]
            ma20_1m = statistics.mean(vols_1m[-20:])
            if ma20_1m > 0:
                has_vol_spike = max(vols_1m) > 5.0 * ma20_1m

        atr5_val = self._ind.atr(candles, 5)
        self._atr5_history[code].append(atr5_val)
        atr5_spike_ratio = 1.0
        buf = self._atr5_history[code]
        if len(buf) >= 5:
            median = statistics.median(buf)
            if median > 0:
                atr5_spike_ratio = round(atr5_val / median, 3)

        return MarketSnapshot(
            code=code,
            price=tick.price,
            volume_today=sum(c.volume for c in candles),
            turnover_today=sum(c.turnover for c in candles),
            vwap=current_vwap,
            atr14=self._ind.atr(candles, 14),
            atr5=atr5_val,
            rsi14=self._ind.rsi(closes, 14),
            ema8=self._ind.ema(closes, 8),
            ema20=self._ind.ema(closes, 20),
            bb_upper=bb_upper,
            bb_mid=bb_mid,
            bb_lower=bb_lower,
            vol_ma20=vol_ma20,
            vol_ratio=vol_ratio,
            bid_ask_spread_pct=(tick.ask - tick.bid) / tick.price if tick.price > 0 else 0.0,
            timestamp=tick.timestamp,
            quote_ready=getattr(tick, "quote_ready", True),
            rsi_ready=len(closes) >= 15,
            vwap_slope_15m=vwap_slope_15m,
            candles_1m=candles_1m,
            candles_primary=candles_primary,
            candles_by_tf=candles_by_tf,
            has_news_event=has_news,
            has_volume_spike_60m=has_vol_spike,
            atr5_spike_ratio=atr5_spike_ratio,
            # ── US-equity reference data (injected, not fetched here) ────────
            sector=self._ref.sector(code),
            market_cap=self._ref.market_cap(code),
            shares_outstanding=self._ref.shares_outstanding(code),
            adv_20d_dollars=self._ref.adv_20d_dollars(code),
            is_earnings_today=self._ref.is_earnings_today(code),
            short_locate_available=self._ref.short_locate_available(code),
            is_hard_to_borrow=self._ref.is_hard_to_borrow(code),
            prev_close_adjusted=self._ref.prev_close_adjusted(code),
            is_regular_trading_hours=is_regular_trading_hours(tick.timestamp),
        )
