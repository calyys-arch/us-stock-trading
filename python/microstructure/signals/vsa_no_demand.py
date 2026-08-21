"""
NEW signal (2026-08-18) — VSA path of least resistance (Williams / Coulling).

What was actually read:

- Tom Williams, *Master the Markets* (3rd ed., TradeGuider, 2005): read in
  full from the extracted text. The mechanical bar rules used here come
  from the footnotes under "What is Bullish & Bearish Volume", the
  "Path of Least Resistance" list, "How to Identify Lack of Demand",
  and the "Main Signs of Weakness / Strength" checklists. Williams'
  wording: weakness shows on up-bars with narrow spread and volume less
  than the previous two bars (no demand); strength shows on down-bars
  with narrow spread, volume less than the previous two bars, and the
  close in the middle or high of the bar (no selling pressure). A single
  bar is not enough — the next bar must fail to continue.
- Anna Coulling, *A Complete Guide to Volume Price Analysis* (2013):
  Chapters 1–6 plus the three-step (micro / macro / global) method were
  read. Principle 2 (patience: do not act on the first anomaly bar) and
  Principle 6 (validation vs anomaly) are why this signal waits for the
  next closed 5-minute bar. Coulling is explicit that software cannot
  replace judgement; this is a testable reduction, not her discretionary
  tape.
- Richard D. Wyckoff / Jack K. Hutson, *Stocks & Commodities* V.4:1
  (the file in the user's folder is this 4-page article, not the 1931
  correspondence course): read in full. Supply/demand and wave turning
  points only — no extra bar rule.
- This is NOT vsa_effort. That module looks for HIGH-volume effort
  without result plus a light test. This module looks for the ABSENCE
  of professional activity on a narrow bar (Williams' no-demand /
  no-selling-pressure), confirmed by the next bar's failure to follow.

Mechanical reduction:

- Chart: closed 5-minute bars, same causality as vsa_effort.
- Setup bar: narrow spread (<= spread_atr_max * ATR) AND volume strictly
  less than each of the previous `vol_lookback` bars.
- Short (no demand): setup is an up-bar at or above prior-session VAH;
  the next closed bar does not make a new high.
- Long (no selling pressure): setup is a down-bar closing in the upper
  half, at or below prior-session VAL; the next closed bar does not
  make a new low.
- GEX: optional environment label, same loader as vsa_effort. Not a
  free parameter and not a hard veto.
- Session: full RTH. The engine still flattens before the close.

Free parameters (4, under param_guard.py's ceiling of 5):
  spread_atr_max, vol_lookback, stop_atr_mult, target_r_multiple.

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


def evaluate_vsa_no_demand(
    bars: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None = None,
    symbol: str = "",
    spread_atr_max: float = 0.55,
    vol_lookback: int = 2,
    stop_atr_mult: float = 0.20,
    target_r_multiple: float = 1.5,
    gex_snapshot: GexSnapshot | dict | None = None,
    chart_minutes: int = _TRADE_MINUTES,
    require_location: bool = True,
    require_confirm: bool = True,
    require_volume: bool = True,
) -> MicroSignal | None:
    """Fires at the close of the confirmation bar after a
    narrow, low-activity setup bar that Williams would call no-demand
    (short) or no selling pressure (long). `chart_minutes` is the
    closed-bar size (default 5). Not a WFO free parameter.

    `require_location` / `require_confirm` / `require_volume` are
    research-only ablation switches (default True = current behavior).
    They are not Chan free parameters."""
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

    lookback = int(vol_lookback)
    if lookback < 1:
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
    extra = 3 if require_confirm else 2
    if len(five) < max(_MIN_TRADE_BARS, lookback + extra):
        return None
    if five.index[-1] + pd.Timedelta(minutes=minutes - 1) != now_time:
        return None

    if require_confirm:
        setup = five.iloc[-2]
        confirm = five.iloc[-1]
        prev = five.iloc[-(lookback + 2):-2]
    else:
        setup = five.iloc[-1]
        confirm = None
        prev = five.iloc[-(lookback + 1):-1]
    if len(prev) < lookback:
        return None
    setup_vol = float(setup["volume"])
    if require_volume and (setup_vol <= 0 or not (setup_vol < float(prev["volume"].min()))):
        return None

    atr_src = pd.concat([prior_5, five])
    atr_series = ctx.atr(atr_src, period=_ATR_PERIOD)
    now_atr = float(atr_series.iloc[-1])
    if pd.isna(now_atr) or now_atr <= 0:
        return None
    spread = float(setup["high"]) - float(setup["low"])
    if spread <= 0 or spread > spread_atr_max * now_atr:
        return None

    setup_loc = _close_loc(setup)
    if setup_loc is None:
        return None
    setup_up = float(setup["close"]) > float(setup["open"])
    setup_down = float(setup["close"]) < float(setup["open"])

    loc_short_ok = (not require_location) or (
        prior_vah is not None and float(setup["close"]) >= prior_vah
    )
    loc_long_ok = (not require_location) or (
        prior_val is not None and float(setup["close"]) <= prior_val
    )

    if setup_up and loc_short_ok:
        if require_confirm and float(confirm["high"]) > float(setup["high"]):
            return None
        side = "short"
        pattern = "no_demand"
    elif setup_down and setup_loc >= _CLOSE_HELD and loc_long_ok:
        if require_confirm and float(confirm["low"]) < float(setup["low"]):
            return None
        side = "long"
        pattern = "no_selling_pressure"
    else:
        return None

    entry = float(setup["close"]) if not require_confirm else float(confirm["close"])
    if side == "long":
        stop = float(setup["low"]) - stop_atr_mult * now_atr
        if stop >= entry:
            return None
        target = entry + target_r_multiple * (entry - stop)
    else:
        stop = float(setup["high"]) + stop_atr_mult * now_atr
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
    prev_mean = float(prev["volume"].mean())
    return MicroSignal(
        symbol=symbol,
        strategy="vsa_no_demand",
        direction=side,
        signal_time=now_time,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        order_type="next_open",
        context={
            "pattern": pattern,
            "setup_rel_volume": setup_vol / prev_mean if prev_mean else None,
            "setup_spread_atr": spread / now_atr,
            "setup_close_loc": setup_loc,
            "setup_low": float(setup["low"]),
            "setup_high": float(setup["high"]),
            "prior_val": prior_val,
            "prior_vah": prior_vah,
            "chart_minutes": minutes,
            "atr": now_atr,
            "require_location": require_location,
            "require_confirm": require_confirm,
            "require_volume": require_volume,
            "vol_regime": vol_regime,
            "gex_source": gex_source,
            "gex_net": env_gex.net_gex if env_gex is not None else None,
            "gex_call_wall": walls.call_wall if walls is not None else None,
            "gex_put_wall": walls.put_wall if walls is not None else None,
            "tier": f"bar_only_{minutes}m_{gex_tag}",
            "target_r_multiple": target_r_multiple,
        },
    )
