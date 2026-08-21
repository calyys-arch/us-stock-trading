"""
NEW signal (2026-08-18) — On-Balance Volume lead/lag (Granville).

What was actually read:

- Joseph E. Granville, *Granville's New Key to Stock Market Profits*
  (Prentice-Hall, 1963): Chapter 3–4 (how to build OBV, the field
  theory, the OBV test) and the compiled "Nine OBV Buying Signals" /
  "Nine OBV Selling Signals" were read from the extracted text.
  The day-to-day rules used here are Granville's B-2 and S-2:
  buy when price sharply outpaces OBV on the downside; sell when
  price sharply outpaces OBV on the upside. Longer-term field-trend
  and B-9/S-9 line breaks are NOT coded — those need a daily OBV
  chart and a drawn field, which this 5-minute session engine does
  not have.
- Buff Dormeier, *The Volume Factor*: the VPCI definition
  (VPC = VWMA − SMA, then × VPR × VM) was read. VPCI is a
  confirmation oscillator, not an entry rule of its own; it is NOT
  coded here. Granville's raw OBV lead is the implementable daily
  idea that Dormeier later generalizes.
- This is NOT vsa_effort and NOT vsa_no_demand. Those compare one
  bar's volume to a local baseline. This compares the PATH of
  cumulative volume (OBV) to the path of price over a lookback.

Mechanical reduction:

- Chart: closed 5-minute bars, same causality as vsa_effort.
- Session OBV is summed from today's closed 5-minute bars only
  (no invented overnight print).
- Short: last close is the lookback high, but OBV is at least
  `obv_lag_frac` of the lookback OBV-range below its own high
  (price made the high; volume did not confirm). Location: at or
  above prior-session VAH.
- Long: last close is the lookback low, but OBV is at least
  `obv_lag_frac` of the range above its own low. Location: at or
  below prior-session VAL.
- GEX: optional environment label. Not a free parameter.
- Session: full RTH. The engine still flattens before the close.

Free parameters (4, under param_guard.py's ceiling of 5):
  lookback_bars, obv_lag_frac, stop_atr_mult, target_r_multiple.

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
_ATR_PERIOD = 14
_VALUE_AREA_PCT = 0.70
_N_BINS = 30
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


def _session_obv(five: pd.DataFrame) -> pd.Series:
    direction = five["close"].diff().fillna(0.0)
    signed = five["volume"].astype(float).copy()
    signed[direction < 0] = -signed[direction < 0]
    signed[direction == 0] = 0.0
    return signed.cumsum()


def evaluate_obv_divergence(
    bars: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None = None,
    symbol: str = "",
    lookback_bars: int = 8,
    obv_lag_frac: float = 0.25,
    stop_atr_mult: float = 0.20,
    target_r_multiple: float = 1.5,
    gex_snapshot: GexSnapshot | dict | None = None,
    chart_minutes: int = _TRADE_MINUTES,
    require_location: bool = True,
    require_obv_lag: bool = True,
) -> MicroSignal | None:
    """Fires at a closed chart bar when price makes a lookback
    extreme that On-Balance Volume does not confirm (Granville B-2 / S-2).
    `chart_minutes` is the closed-bar size (default 5). Not a WFO free parameter.

    `require_location` / `require_obv_lag` are research-only ablation
    switches (default True = current behavior). Not Chan free parameters."""
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

    n = int(lookback_bars)
    if n < 4:
        return None

    prior_5 = _resample_closed(prior_day_bars, minutes=minutes)
    if prior_5.empty:
        return None
    prior_tp = (prior_5["high"] + prior_5["low"] + prior_5["close"]) / 3.0
    profile = ctx.volume_profile_from_arrays(
        prior_tp.to_numpy(), prior_5["volume"].to_numpy(dtype=float),
        n_bins=_N_BINS, value_area_pct=_VALUE_AREA_PCT,
    )
    prior_val = float(profile.val) if profile.val is not None else None
    prior_vah = float(profile.vah) if profile.vah is not None else None
    if require_location and (prior_val is None or prior_vah is None):
        return None

    five = _resample_closed(bars, minutes=minutes)
    if len(five) < max(_MIN_TRADE_BARS, n):
        return None
    if five.index[-1] + pd.Timedelta(minutes=minutes - 1) != now_time:
        return None

    window = five.iloc[-n:]
    obv = _session_obv(five).iloc[-n:]
    last_close = float(window["close"].iloc[-1])
    last_obv = float(obv.iloc[-1])
    px_high = float(window["close"].max())
    px_low = float(window["close"].min())
    obv_high = float(obv.max())
    obv_low = float(obv.min())
    obv_range = obv_high - obv_low
    if require_obv_lag and obv_range <= 0:
        return None

    price_new_high = last_close >= px_high - 1e-12
    price_new_low = last_close <= px_low + 1e-12
    if require_obv_lag:
        lag = obv_lag_frac * obv_range
        obv_lags_high = last_obv <= (obv_high - lag)
        obv_lags_low = last_obv >= (obv_low + lag)
    else:
        obv_lags_high = True
        obv_lags_low = True

    loc_short_ok = (not require_location) or (
        prior_vah is not None and last_close >= prior_vah
    )
    loc_long_ok = (not require_location) or (
        prior_val is not None and last_close <= prior_val
    )

    if price_new_high and obv_lags_high and loc_short_ok:
        side = "short"
        pattern = "obv_s2_price_leads"
    elif price_new_low and obv_lags_low and loc_long_ok:
        side = "long"
        pattern = "obv_b2_price_leads"
    else:
        return None

    atr_src = pd.concat([prior_5, five])
    atr_series = ctx.atr(atr_src, period=_ATR_PERIOD)
    now_atr = float(atr_series.iloc[-1])
    if pd.isna(now_atr) or now_atr <= 0:
        return None

    last = five.iloc[-1]
    entry = last_close
    if side == "long":
        stop = float(last["low"]) - stop_atr_mult * now_atr
        if stop >= entry:
            return None
        target = entry + target_r_multiple * (entry - stop)
    else:
        stop = float(last["high"]) + stop_atr_mult * now_atr
        if stop <= entry:
            return None
        target = entry - target_r_multiple * (stop - entry)

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
        strategy="obv_divergence",
        direction=side,
        signal_time=now_time,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        order_type="next_open",
        context={
            "pattern": pattern,
            "obv_now": last_obv,
            "obv_high": obv_high,
            "obv_low": obv_low,
            "obv_lag_frac_used": (
                None if obv_range <= 0
                else ((obv_high - last_obv) / obv_range if side == "short"
                      else (last_obv - obv_low) / obv_range)
            ),
            "lookback_bars": n,
            "prior_val": prior_val,
            "prior_vah": prior_vah,
            "chart_minutes": minutes,
            "require_location": require_location,
            "require_obv_lag": require_obv_lag,
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
