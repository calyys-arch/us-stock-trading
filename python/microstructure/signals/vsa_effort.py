"""
NEW signal (2026-08-18) — VSA effort-without-result (Wyckoff / Williams).

What was actually read, vs what was not:

- Richard D. Wyckoff, *Studies in Tape Reading* (1910, public domain,
  published as Rollo Tape): Chapter V "Volumes and Their Significance"
  was read in full from the 1910 text. Last sale is history; the live
  market is bid/ask plus which side the size is on. Large volume with
  poor price response is resistance or manipulation — do not trust the
  print. After a climax, size switching from the up side to the down
  side (or the reverse) marks the turn. Volume is always judged
  relative to that stock's own recent activity.
- Tom Williams, *Master the Markets* / *The Undeclared Secrets That
  Drive the Stock Market*: still in copyright. Those books were NOT
  ingested cover-to-cover. This module uses only the publicly named
  VSA bar types that descend from the Wyckoff chapter above
  (stopping volume, no-supply / no-demand test, upthrust / spring)
  as labels on the same effort-vs-result idea.
- Later books listed in chat (O'Neil, Granville, Dormeier, Dalton)
  were not ingested in full. They are not coded here.

Mechanical reduction (NOT a claim this is Williams' discretionary tape):

- Chart: closed 5-minute bars, same causality as auction_reclaim
  (`_TRADE_MINUTES = 5`). 1-minute cache is the resample source.
- Effort bar: relative volume >= effort_vol_mult AND close fails to
  reward the push (sellers print a low but close in the upper half;
  buyers print a high but close in the lower half).
- Test bar: the next closed 5-minute bar revisits that extreme on
  LIGHT volume (<= test_vol_mult of the same baseline) and does not
  make a new extreme. That is the "no supply" / "no demand" test —
  the fake print is not being followed.
- Location: the failed extreme must sit outside the prior session's
  value area (below VAL for a long spring, above VAH for a short
  upthrust). Structural, not a free parameter.
- GEX: optional environment label from python/microstructure/gex.py,
  same loader as auction_reclaim. Positive gamma is the natural fade
  regime; missing file → no invented dealer gamma. Not a free param
  and not a hard veto.
- Session: full RTH (Wyckoff is not a 90-minute NY-open rule). The
  engine still flattens before the close.

Free parameters (4, under param_guard.py's ceiling of 5):
  effort_vol_mult, test_vol_mult, stop_atr_mult, target_r_multiple.

Research-only: configs/strategy.yaml auto_execute stays false. This
signal is NOT in LIVE_SIGNALS.
"""
from __future__ import annotations

from datetime import time

import pandas as pd

from .. import context as ctx
from ..gex import GexSnapshot, resolve_gex_env
from . import MicroSignal

_TRADE_MINUTES = 5
_MIN_TRADE_BARS = 8
_VOLUME_LOOKBACK = 6
_ATR_PERIOD = 14
_VALUE_AREA_PCT = 0.70
_N_BINS = 30
_CLOSE_HELD = 0.50
_SESSION_END = time(16, 0)


def _resample_closed(bars: pd.DataFrame, minutes: int = _TRADE_MINUTES) -> pd.DataFrame:
    if bars.empty or len(bars) < minutes:
        return bars.iloc[0:0]
    origin = bars.index[0]
    last = bars.index[-1]
    elapsed = int((last - origin).total_seconds() // 60)
    if elapsed < 0:
        return bars.iloc[0:0]
    last_offset = minutes - 1
    if elapsed % minutes != last_offset:
        cut = elapsed % minutes + 1
        closed = bars.iloc[:-cut]
    else:
        closed = bars
    if len(closed) < minutes:
        return bars.iloc[0:0]
    resampled = closed.resample(f"{minutes}min", origin=origin).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"},
    )
    return resampled.dropna(subset=["open"])


def _close_loc(bar: pd.Series) -> float | None:
    rng = float(bar["high"]) - float(bar["low"])
    if rng <= 0:
        return None
    return (float(bar["close"]) - float(bar["low"])) / rng


def _rel_volume(bar: pd.Series, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return float(bar["volume"]) / baseline


def evaluate_vsa_effort(
    bars: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None = None,
    symbol: str = "",
    effort_vol_mult: float = 2.0,
    test_vol_mult: float = 0.85,
    stop_atr_mult: float = 0.20,
    target_r_multiple: float = 1.5,
    gex_snapshot: GexSnapshot | dict | None = None,
    chart_minutes: int = _TRADE_MINUTES,
) -> MicroSignal | None:
    """Fires at the close of a completed chart bar when the
    previous bar was unrewarded effort and this bar is a
    light-volume test that does not make a new extreme.
    `chart_minutes` is the closed-bar size (default 5)."""
    minutes = int(chart_minutes)
    if minutes < 1:
        return None
    if prior_day_bars is None or prior_day_bars.empty or bars.empty:
        return None
    now_time = bars.index[-1]
    if now_time.time() >= _SESSION_END:
        return None
    elapsed = int((now_time - bars.index[0]).total_seconds() // 60)
    if elapsed < 0 or elapsed % minutes != (minutes - 1):
        return None

    prior_5 = _resample_closed(prior_day_bars, minutes=minutes)
    if prior_5.empty:
        return None
    prior_tp = (prior_5["high"] + prior_5["low"] + prior_5["close"]) / 3.0
    profile = ctx.volume_profile_from_arrays(
        prior_tp.to_numpy(), prior_5["volume"].to_numpy(dtype=float),
        n_bins=_N_BINS, value_area_pct=_VALUE_AREA_PCT,
    )
    if profile.val is None or profile.vah is None:
        return None

    five = _resample_closed(bars, minutes=minutes)
    if len(five) < _MIN_TRADE_BARS:
        return None
    if five.index[-1] + pd.Timedelta(minutes=minutes - 1) != now_time:
        return None

    effort = five.iloc[-2]
    test = five.iloc[-1]
    baseline = five["volume"].iloc[-(_VOLUME_LOOKBACK + 2):-2]
    baseline_vol = float(baseline.mean())
    effort_rel = _rel_volume(effort, baseline_vol)
    test_rel = _rel_volume(test, baseline_vol)
    if effort_rel is None or test_rel is None:
        return None
    if effort_rel < effort_vol_mult:
        return None
    if test_rel > test_vol_mult:
        return None

    effort_loc = _close_loc(effort)
    if effort_loc is None:
        return None

    long_effort = effort_loc >= _CLOSE_HELD and float(effort["low"]) < float(five["low"].iloc[-(_VOLUME_LOOKBACK + 2):-2].min())
    short_effort = effort_loc <= (1.0 - _CLOSE_HELD) and float(effort["high"]) > float(five["high"].iloc[-(_VOLUME_LOOKBACK + 2):-2].max())
    if long_effort and float(effort["low"]) <= float(profile.val):
        side = "long"
    elif short_effort and float(effort["high"]) >= float(profile.vah):
        side = "short"
    else:
        return None

    if side == "long":
        if float(test["low"]) < float(effort["low"]):
            return None
        if float(test["close"]) <= float(test["open"]) and float(test["close"]) <= (
            float(effort["low"]) + float(effort["high"])
        ) / 2.0:
            return None
    else:
        if float(test["high"]) > float(effort["high"]):
            return None
        if float(test["close"]) >= float(test["open"]) and float(test["close"]) >= (
            float(effort["low"]) + float(effort["high"])
        ) / 2.0:
            return None

    atr_src = pd.concat([prior_5, five])
    atr_series = ctx.atr(atr_src, period=_ATR_PERIOD)
    now_atr = float(atr_series.iloc[-1])
    if pd.isna(now_atr) or now_atr <= 0:
        return None

    entry = float(test["close"])
    if side == "long":
        stop = float(effort["low"]) - stop_atr_mult * now_atr
        if stop >= entry:
            return None
        target = entry + target_r_multiple * (entry - stop)
        pattern = "spring_no_supply"
    else:
        stop = float(effort["high"]) + stop_atr_mult * now_atr
        if stop <= entry:
            return None
        target = entry - target_r_multiple * (stop - entry)
        pattern = "upthrust_no_demand"

    env_gex, symbol_gex = resolve_gex_env(gex_snapshot)
    if env_gex is not None:
        vol_regime = env_gex.regime
        gex_source = env_gex.source
    else:
        vol_regime = "unknown"
        gex_source = "unavailable"
    walls = symbol_gex or env_gex
    gex_tag = "gex" if env_gex is not None else "no_gex"
    return MicroSignal(
        symbol=symbol,
        strategy="vsa_effort",
        direction=side,
        signal_time=now_time,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        order_type="next_open",
        context={
            "pattern": pattern,
            "effort_rel_volume": effort_rel,
            "test_rel_volume": test_rel,
            "effort_close_loc": effort_loc,
            "effort_low": float(effort["low"]),
            "effort_high": float(effort["high"]),
            "prior_val": float(profile.val),
            "prior_vah": float(profile.vah),
            "chart_minutes": minutes,
            "atr": now_atr,
            "vol_regime": vol_regime,
            "gex_source": gex_source,
            "gex_net": env_gex.net_gex if env_gex is not None else None,
            "gex_call_wall": walls.call_wall if walls is not None else None,
            "gex_put_wall": walls.put_wall if walls is not None else None,
            "tier": f"bar_only_{minutes}m_{gex_tag}",
            "target_r_multiple": target_r_multiple,
        },
    )
