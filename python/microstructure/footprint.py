"""
5-minute footprint bars from captured trade prints.

Creamer's confirmation is two footprint candles (volume-at-price + delta):
absorption = aggressive flow at the extreme that is not rewarded; the next
candle must fail higher (long) / lower (short) with a dominance shift.
He cites a 400% bid/ask imbalance as the visual that "lights up".

This module aggregates `data/ticks/` prints into 5-minute bins aligned to
the same session origin as auction_reclaim._resample_closed_5m. Side:

  - Futu `ticker_direction` BUY/SELL is used when present (tape side).
  - NEUTRAL / missing / IBKR AllLast (no side) fall back to the Lee (1991)
    tick rule — same choice as trap_detector.order_flow_imbalance_score,
    and for the same reason (no synchronized quote for Lee-Ready).

A bin with fewer than `_MIN_CLASSIFIED` sided prints is `complete=False`
so the caller can fall back to the bar-only wick proxy instead of treating
a thin tape as a real footprint. Structural constants (not free params):
imbalance 4.0, extreme band 20% of the 5-minute range, min 20 prints.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_BIN_MINUTES = 5
_IMBALANCE = 4.0
_EXTREME_FRAC = 0.20
_MIN_CLASSIFIED = 20


@dataclass(frozen=True)
class FootprintBar:
    start: pd.Timestamp
    buy_volume: float
    sell_volume: float
    classified_trades: int
    complete: bool
    low: float
    high: float
    extreme_low_delta: float
    extreme_high_delta: float

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance_ratio(self) -> float:
        weaker = min(self.buy_volume, self.sell_volume)
        stronger = max(self.buy_volume, self.sell_volume)
        if weaker <= 0:
            return float("inf") if stronger > 0 else 0.0
        return stronger / weaker


def classify_sides(ticks: pd.DataFrame) -> pd.DataFrame:
    """Add a `side` column: 'buy' / 'sell' / None. Does not mutate input."""
    if ticks is None or ticks.empty:
        return ticks.iloc[0:0] if ticks is not None else pd.DataFrame()
    out = ticks.copy()
    sides: list[str | None] = []
    last_price: float | None = None
    last_side: str | None = None
    prices = out["price"].to_numpy(dtype=float)
    directions = (
        out["ticker_direction"].astype(str).str.upper()
        if "ticker_direction" in out.columns
        else [""] * len(out)
    )
    for i, price in enumerate(prices):
        tape = directions.iloc[i] if hasattr(directions, "iloc") else directions[i]
        side = _tape_side(tape)
        if side is None:
            if last_price is not None:
                if price > last_price:
                    side = "buy"
                elif price < last_price:
                    side = "sell"
                else:
                    side = last_side
        if side is not None:
            last_side = side
        last_price = price
        sides.append(side)
    out["side"] = pd.Series(sides, index=out.index, dtype=object)
    return out


def _tape_side(raw: str) -> str | None:
    if raw in {"BUY", "B", "1", "TICKER_DIRECTION.BUY"}:
        return "buy"
    if raw in {"SELL", "S", "2", "TICKER_DIRECTION.SELL"}:
        return "sell"
    return None


def footprint_5m(
    ticks: pd.DataFrame,
    origin: pd.Timestamp,
    bin_minutes: int = _BIN_MINUTES,
) -> dict[pd.Timestamp, FootprintBar]:
    """Map 5-minute bin start → FootprintBar. `origin` must be the same
    session-open timestamp the 1-minute bars use (typically 09:30 ET)."""
    if ticks is None or ticks.empty:
        return {}
    classified = classify_sides(ticks)
    if classified.empty:
        return {}
    if "time" in classified.columns:
        times = pd.DatetimeIndex(classified["time"])
    else:
        times = pd.DatetimeIndex(classified.index)
    origin = pd.Timestamp(origin)
    elapsed_min = ((times - origin).total_seconds() // 60).astype(int)
    bin_idx = elapsed_min // bin_minutes
    valid = elapsed_min >= 0
    classified = classified.loc[valid].copy()
    classified["_bin"] = bin_idx[valid]
    classified["_bin_start"] = origin + pd.to_timedelta(
        classified["_bin"].to_numpy() * bin_minutes, unit="min",
    )

    out: dict[pd.Timestamp, FootprintBar] = {}
    for start, grp in classified.groupby("_bin_start", sort=True):
        sided = grp[grp["side"].isin(("buy", "sell"))]
        buy = float(sided.loc[sided["side"] == "buy", "size"].sum())
        sell = float(sided.loc[sided["side"] == "sell", "size"].sum())
        n = int(len(sided))
        if grp.empty:
            continue
        lo = float(grp["price"].min())
        hi = float(grp["price"].max())
        rng = hi - lo
        if rng > 0:
            low_cut = lo + _EXTREME_FRAC * rng
            high_cut = hi - _EXTREME_FRAC * rng
            low_band = sided[sided["price"] <= low_cut]
            high_band = sided[sided["price"] >= high_cut]
            low_delta = float(low_band.loc[low_band["side"] == "buy", "size"].sum()) - float(
                low_band.loc[low_band["side"] == "sell", "size"].sum()
            )
            high_delta = float(high_band.loc[high_band["side"] == "buy", "size"].sum()) - float(
                high_band.loc[high_band["side"] == "sell", "size"].sum()
            )
        else:
            low_delta = buy - sell
            high_delta = buy - sell
        start_ts = pd.Timestamp(start)
        out[start_ts] = FootprintBar(
            start=start_ts,
            buy_volume=buy,
            sell_volume=sell,
            classified_trades=n,
            complete=n >= _MIN_CLASSIFIED,
            low=lo,
            high=hi,
            extreme_low_delta=low_delta,
            extreme_high_delta=high_delta,
        )
    return out


def probe_absorbed(bar: FootprintBar, side: str) -> bool:
    """Aggressive flow at the extreme that was not a one-sided runaway.
    Long: sellers printed at the lows (negative delta in the low band).
    Short: buyers printed at the highs."""
    if not bar.complete:
        return False
    if side == "long":
        return bar.extreme_low_delta < 0
    return bar.extreme_high_delta > 0


def confirm_dominance(bar: FootprintBar, side: str, imbalance: float = _IMBALANCE) -> bool:
    """Second candle: the reclaiming side takes over (delta flip or 400%
    imbalance). Absorption alone is not a reversal — Creamer's own words."""
    if not bar.complete:
        return False
    if side == "long":
        if bar.delta > 0:
            return True
        return bar.sell_volume > 0 and bar.buy_volume / bar.sell_volume >= imbalance
    if bar.delta < 0:
        return True
    return bar.buy_volume > 0 and bar.sell_volume / bar.buy_volume >= imbalance
