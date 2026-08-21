"""
NEW signal (2026-08-18) — Auction reclaim (Creamer-style).

Faithful, testable reduction of Christopher Creamer's publicly described
intraday process (auction-market environment → location at a Fibonacci
discount/premium OUTSIDE the prior session value area → 5-minute
absorption + dominance-shift confirmation). This is NOT a claim that the
module reproduces his discretionary tape-reading or Tanuki's exact levels.

What this module actually has, vs what Creamer described:

- Environment (whiteboard step 1, BEFORE the open, on each stock's
  1-hour chart): Creamer looks at the 1-hour chart of the name he is
  about to trade — before 09:30 — and asks whether value is being
  created higher, lower, or sideways (HH/HL vs LH/LL vs overlapping).
  That read is a PRE-OPEN snapshot of THAT stock. It is not a 3-session
  hourly volume-profile POC stack, and it is not recomputed from today's
  developing 1-hour bars. This module builds the 1-hour candles of the
  last two COMPLETED RTH sessions (including the short 15:30–16:00
  slot) and labels value-up / value-down / sideways from those two
  days' 1-hour highs and lows. Independently, the same 1-hour bars
  measure TRADER MOMENTUM: who won each hour (close vs that hour's
  open, volume-weighted) and whether participation is building or
  fading across the 1h sequence. Volume does not confirm price
  structure and price structure does not confirm volume — they are
  allowed to oppose. Sideways or fewer than two prior sessions →
  stand aside. GEX (volatility, not direction) is the other
  environment piece when a snapshot exists. He also glances at
  15-minute; that is not coded.
- Location: Fibonacci 0.705–0.886 retracement of the prior session range,
  required to sit OUTSIDE that session's Volume Profile value area
  (python/microstructure/context.py's bar-approximate POC/VAH/VAL).
- Chart: Creamer's decision chart is 3-minute or 5-minute, not 1-minute.
  This module locks `_TRADE_MINUTES = 5` (his confirmation / footprint
  candle). Location (prior-session value area), the two-bar probe/confirm,
  relative volume, and ATR are ALL computed on closed 5-minute bars.
  The 1-minute cache is only the raw feed we resample — it is not the
  trading chart. 3-minute is the other chart he mentioned; it is NOT a
  free parameter and is not gridded (switching it would be a different
  structural system). The intraday engine still walks 1-minute bars so
  a stop/target that traded inside the next 5-minute candle is filled
  honestly; that is execution granularity, not the signal clock.
- Confirmation: two consecutive CLOSED 5-minute bars. Probe = failed
  auction at the extreme; confirm = dominance shift that fails HIGHER
  (long) or LOWER (short). When captured trade prints cover BOTH closed
  5-minute bins (data/ticks/, Futu ticker_direction or Lee tick rule),
  footprint absorption + 400% imbalance is a HARD filter on top of the
  bar wick/shift. Incomplete / missing ticks fall back to the bar proxy
  and `context["tier"]` says so — never invent a ladder.
- GEX / dealer gamma: optional naive snapshot from
  python/microstructure/gex.py (typically QQQ, written by
  scripts/snapshot_gex.py). When present it REPLACES the prior-session
  range-efficiency label in `context["vol_regime"]` (positive_gamma /
  negative_gamma). It is environment, not a direction filter, and not a
  free parameter. Missing file → keep the compressed/expanded proxy.
- Session: first 90 minutes of RTH only (09:30–11:00 ET). MNQ's
  20,000-contract 5-minute floor is replaced by a relative-volume
  multiple on the equity 5-minute bar (stocks are not MNQ).

No lookahead: only `bars` up to and including "now" (`bars.index[-1]`),
the already-closed `prior_day_bars` session, and ticks with time <= now
(`python/backtest/tick_replay.ticks_up_to`). Fill timing is still
next-bar open, enforced by python/backtest/intraday_engine.py.

Free parameters (4, under param_guard.py's ceiling of 5):
  min_rel_volume, min_wick_frac, stop_atr_mult, target_r_multiple.

Structural constants (not gridded): 5-minute trading chart (not 1-minute),
Fibonacci 0.705 / 0.886, 90-minute session cut, 70% value-area,
prior-session close location thresholds, footprint imbalance 4.0 /
min 20 sided prints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import pandas as pd

from ...data.tick_cache import ticks_up_to
from .. import context as ctx
from .. import footprint as fp
from ..gex import GexSnapshot, resolve_gex_env
from . import MicroSignal

_FIB_INNER = 0.705
_FIB_OUTER = 0.886
_SESSION_END = time(11, 0)
_VALUE_AREA_PCT = 0.70
_N_BINS = 30
# Creamer's confirmation chart. Not a free parameter — 3-minute would be
# a different locked system, not a grid axis.
_TRADE_MINUTES = 5
_MIN_TRADE_BARS = 5
_VOLUME_LOOKBACK = 3
_ATR_PERIOD = 14
# Pre-open 1h chart: last two completed sessions (HH/HL vs LH/LL).
# Structural, not a free param. Never includes the session being traded.
_ENV_SESSIONS = 2
_ENV_HOUR_MINUTES = 60
# Trader momentum on the 1h chart. Structural, not a free param.
# Pressure is volume-weighted signed body of each 1h bar — not HH/HL.
_MOM_PRESSURE = 0.20
_MOM_BUILD = 1.15
_MOM_FADE = 0.85


def _resample_closed(bars: pd.DataFrame, minutes: int = _TRADE_MINUTES) -> pd.DataFrame:
    """OHLCV on the trading chart using only bins whose last 1-minute print
    is already visible. Incomplete current bin is dropped — that is the
    causality contract, not a lookahead shortcut."""
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


def _resample_closed_5m(bars: pd.DataFrame) -> pd.DataFrame:
    """Back-compat alias used by tests."""
    return _resample_closed(bars, minutes=5)


@dataclass(frozen=True)
class PreopenEnvironment:
    """Frozen 1-hour-chart read for ONE stock, using only sessions that
    already closed before today's open. `bias` is None when the 1h chart
    is sideways or there is not enough history to look."""
    structure: str
    bias: str | None
    asof: pd.Timestamp | None
    n_hourly_bars: int
    last_high: float | None
    last_low: float | None
    prev_high: float | None
    prev_low: float | None
    last_volume: float | None = None
    prev_volume: float | None = None
    volume_ratio: float | None = None
    last_vwap: float | None = None
    prev_vwap: float | None = None
    peak_hour: str | None = None
    peak_hour_share: float | None = None
    trader_side: str = "unknown"
    trader_pace: str = "unknown"
    trader_momentum: str = "unknown"
    trader_pressure: float | None = None
    pace_ratio: float | None = None
    hourly: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "structure": self.structure,
            "bias": self.bias,
            "asof": None if self.asof is None else self.asof.isoformat(),
            "n_hourly_bars": self.n_hourly_bars,
            "last_high": self.last_high,
            "last_low": self.last_low,
            "prev_high": self.prev_high,
            "prev_low": self.prev_low,
            "last_volume": self.last_volume,
            "prev_volume": self.prev_volume,
            "volume_ratio": self.volume_ratio,
            "last_vwap": self.last_vwap,
            "prev_vwap": self.prev_vwap,
            "peak_hour": self.peak_hour,
            "peak_hour_share": self.peak_hour_share,
            "trader_side": self.trader_side,
            "trader_pace": self.trader_pace,
            "trader_momentum": self.trader_momentum,
            "trader_pressure": self.trader_pressure,
            "pace_ratio": self.pace_ratio,
            "hourly": list(self.hourly),
        }


def _unknown_preopen() -> PreopenEnvironment:
    return PreopenEnvironment(
        structure="unknown", bias=None, asof=None, n_hourly_bars=0,
        last_high=None, last_low=None, prev_high=None, prev_low=None,
    )


def prior_rth_sessions(
    bars: pd.DataFrame,
    before: pd.Timestamp,
    n: int = _ENV_SESSIONS,
) -> list[pd.DataFrame]:
    """Completed RTH sessions strictly before `before` (the day about to
    open). This is the only history a pre-open 1h read may use."""
    if bars is None or bars.empty:
        return []
    cutoff = pd.Timestamp(before).normalize()
    dates = sorted({ts.normalize() for ts in bars.index})
    picked = [d for d in dates if d < cutoff][-n:]
    return [bars.loc[bars.index.normalize() == d] for d in picked]


def _completed_session_hourly(session: pd.DataFrame) -> pd.DataFrame:
    """1-hour chart of a COMPLETED session. The last RTH slot
    (15:30–16:00) is kept — the session is over, that candle is closed.
    `_resample_closed` would drop it because it is only 30 minutes."""
    if session is None or session.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    origin = session.index[0]
    hourly = session.resample(f"{_ENV_HOUR_MINUTES}min", origin=origin).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"},
    )
    return hourly.dropna(subset=["open"])


def _bar_pressure(row: pd.Series) -> float:
    """Who controlled this 1h bar: signed body / range. Independent of
    whether the session made a higher high vs the prior day."""
    rng = float(row["high"]) - float(row["low"])
    if rng <= 0:
        return 0.0
    return max(-1.0, min(1.0, (float(row["close"]) - float(row["open"])) / rng))


def _hourly_rows(hourly: pd.DataFrame, session_label: str) -> list[dict]:
    total = float(hourly["volume"].sum()) if not hourly.empty else 0.0
    rows = []
    for ts, row in hourly.iterrows():
        vol = float(row["volume"])
        rows.append({
            "time": ts.isoformat(),
            "session": session_label,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": vol,
            "volume_share": (vol / total) if total > 0 else 0.0,
            "pressure": _bar_pressure(row),
        })
    return rows


def _hourly_volume_read(hourly: pd.DataFrame) -> dict:
    if hourly.empty:
        return {"volume": 0.0, "vwap": None, "peak_time": None, "peak_share": None}
    vol = hourly["volume"].astype(float)
    total = float(vol.sum())
    tp = (hourly["high"] + hourly["low"] + hourly["close"]) / 3.0
    if total <= 0:
        return {"volume": 0.0, "vwap": None, "peak_time": None, "peak_share": None}
    peak_ts = vol.idxmax()
    return {
        "volume": total,
        "vwap": float((tp * vol).sum() / total),
        "peak_time": peak_ts.isoformat(),
        "peak_share": float(vol.max() / total),
    }


def _trader_momentum(hourly: pd.DataFrame) -> dict:
    """Trader momentum from the 1h sequence itself.

    Side = volume-weighted signed body of each 1h bar (who won the hour).
    Pace = later 1h volume vs earlier 1h volume (are traders stepping in
    or stepping away). Neither is derived from session HH/HL."""
    empty = {
        "trader_side": "unknown", "trader_pace": "unknown",
        "trader_momentum": "unknown", "trader_pressure": None, "pace_ratio": None,
    }
    if hourly is None or hourly.empty:
        return empty
    vol = hourly["volume"].astype(float)
    total = float(vol.sum())
    if total <= 0:
        return empty
    n = len(hourly)
    split = max(1, n // 2)
    # Side from the RECENT 1h bars only — current trader thrust, not a
    # two-day blend that can cancel a sold last session against a bought
    # prior session. Pace still uses the full sequence.
    recent = hourly.iloc[split:]
    recent_vol = recent["volume"].astype(float)
    recent_total = float(recent_vol.sum())
    if recent_total <= 0:
        return empty
    recent_pressure = recent.apply(_bar_pressure, axis=1)
    net = float((recent_pressure * recent_vol).sum() / recent_total)
    if net >= _MOM_PRESSURE:
        side = "buying"
    elif net <= -_MOM_PRESSURE:
        side = "selling"
    else:
        side = "rotational"
    earlier = float(vol.iloc[:split].mean())
    later = float(vol.iloc[split:].mean())
    pace_ratio = (later / earlier) if earlier > 0 else None
    if pace_ratio is None:
        pace = "unknown"
    elif pace_ratio >= _MOM_BUILD:
        pace = "building"
    elif pace_ratio <= _MOM_FADE:
        pace = "fading"
    else:
        pace = "flat"
    return {
        "trader_side": side,
        "trader_pace": pace,
        "trader_momentum": f"{side}_{pace}",
        "trader_pressure": net,
        "pace_ratio": pace_ratio,
    }


def preopen_1h_environment(prior_sessions: list[pd.DataFrame] | None) -> PreopenEnvironment:
    """Observe this stock's 1-hour chart before the open.

    Two independent reads, both frozen before 09:30, never looking at today:

    1. Price structure — last two sessions' 1h highs/lows (value-up /
       value-down / sideways).
    2. Trader momentum — who won the 1h bars and whether participation
       is building or fading. Not a confirmation of (1)."""
    sessions = [s for s in (prior_sessions or []) if s is not None and not s.empty]
    if len(sessions) < _ENV_SESSIONS:
        return _unknown_preopen()
    prev_h = _completed_session_hourly(sessions[-2])
    last_h = _completed_session_hourly(sessions[-1])
    if prev_h.empty or last_h.empty:
        return _unknown_preopen()
    prev_high = float(prev_h["high"].max())
    prev_low = float(prev_h["low"].min())
    last_high = float(last_h["high"].max())
    last_low = float(last_h["low"].min())
    if last_high > prev_high and last_low > prev_low:
        structure, bias = "value_up", "up"
    elif last_high < prev_high and last_low < prev_low:
        structure, bias = "value_down", "down"
    else:
        structure, bias = "sideways", None
    prev_vol = _hourly_volume_read(prev_h)
    last_vol = _hourly_volume_read(last_h)
    ratio = (
        last_vol["volume"] / prev_vol["volume"]
        if prev_vol["volume"] > 0 else None
    )
    stitched = pd.concat([prev_h, last_h])
    mom = _trader_momentum(stitched)
    hourly_rows = _hourly_rows(prev_h, "prev") + _hourly_rows(last_h, "last")
    return PreopenEnvironment(
        structure=structure,
        bias=bias,
        asof=last_h.index[-1],
        n_hourly_bars=len(prev_h) + len(last_h),
        last_high=last_high,
        last_low=last_low,
        prev_high=prev_high,
        prev_low=prev_low,
        last_volume=last_vol["volume"],
        prev_volume=prev_vol["volume"],
        volume_ratio=ratio,
        last_vwap=last_vol["vwap"],
        prev_vwap=prev_vol["vwap"],
        peak_hour=last_vol["peak_time"],
        peak_hour_share=last_vol["peak_share"],
        trader_side=mom["trader_side"],
        trader_pace=mom["trader_pace"],
        trader_momentum=mom["trader_momentum"],
        trader_pressure=mom["trader_pressure"],
        pace_ratio=mom["pace_ratio"],
        hourly=hourly_rows,
    )


def scan_universe_preopen(
    bars_by_symbol: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
) -> list[dict]:
    """Pre-open 1h environment for every symbol, using only RTH sessions
    that closed before `asof`. Report / dashboard observation — does not
    emit a trade."""
    cutoff = pd.Timestamp(asof)
    rows: list[dict] = []
    for symbol, bars in bars_by_symbol.items():
        env = preopen_1h_environment(prior_rth_sessions(bars, cutoff))
        row = env.to_dict()
        row["symbol"] = symbol
        rows.append(row)
    return rows


def _vol_regime(prior: pd.DataFrame) -> str:
    """Compressed vs expanded prior session — a GEX stand-in, not GEX."""
    op = float(prior["open"].iloc[0])
    cl = float(prior["close"].iloc[-1])
    hi = float(prior["high"].max())
    lo = float(prior["low"].min())
    rng = hi - lo
    if rng <= 0:
        return "unknown"
    efficiency = abs(cl - op) / rng
    return "expanded" if efficiency >= 0.60 else "compressed"


def _fib_zone(high: float, low: float, side: str) -> tuple[float, float]:
    rng = high - low
    if side == "long":
        inner = high - _FIB_INNER * rng
        outer = high - _FIB_OUTER * rng
        return outer, inner
    inner = low + _FIB_INNER * rng
    outer = low + _FIB_OUTER * rng
    return inner, outer


def _probe_absorbed(probe: pd.Series, side: str, min_wick_frac: float) -> bool:
    rng = float(probe["high"]) - float(probe["low"])
    if rng <= 0:
        return False
    if side == "long":
        wick = min(float(probe["open"]), float(probe["close"])) - float(probe["low"])
        # Closed on the low = sellers were rewarded; that is not absorption.
        if float(probe["close"]) <= float(probe["low"]) + 0.15 * rng:
            return False
    else:
        wick = float(probe["high"]) - max(float(probe["open"]), float(probe["close"]))
        if float(probe["close"]) >= float(probe["high"]) - 0.15 * rng:
            return False
    return wick / rng >= min_wick_frac


def _confirm_shift(probe: pd.Series, confirm: pd.Series, side: str) -> bool:
    if side == "long":
        if float(confirm["low"]) < float(probe["low"]):
            return False
        return float(confirm["close"]) > float(confirm["open"])
    if float(confirm["high"]) > float(probe["high"]):
        return False
    return float(confirm["close"]) < float(confirm["open"])


def evaluate_auction_reclaim(
    bars: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None = None,
    symbol: str = "",
    min_rel_volume: float = 1.2,
    min_wick_frac: float = 0.45,
    stop_atr_mult: float = 0.15,
    target_r_multiple: float = 1.5,
    gex_snapshot: GexSnapshot | dict | None = None,
    ticks_so_far: pd.DataFrame | None = None,
    prior_sessions: list[pd.DataFrame] | None = None,
    preopen_env: PreopenEnvironment | None = None,
    chart_minutes: int = _TRADE_MINUTES,
) -> MicroSignal | None:
    """Fires AT the close of a completed 5-minute bar (the last 1-minute
    print of that bin) when location + two-bar absorption/reclaim line up
    with the pre-open 1h environment. Only ever looks at `bars` up to
    "now", the already-finished `prior_day_bars` session, and ticks with
    time <= now. `gex_snapshot` / `ticks_so_far` / `preopen_env` are
    optional context — not free parameters, not in SIGNAL_PARAM_KEYS.
    `preopen_env` is the frozen pre-open 1h read; if omitted it is
    rebuilt from `prior_sessions` (still never from today)."""
    sessions = list(prior_sessions) if prior_sessions else []
    if prior_day_bars is not None and not prior_day_bars.empty:
        if not sessions:
            sessions = [prior_day_bars]
        prior_day_bars = sessions[-1]
    if prior_day_bars is None or prior_day_bars.empty or bars.empty:
        return None

    minutes = int(chart_minutes)
    if minutes < 1:
        return None
    now_time = bars.index[-1]
    if now_time.time() >= _SESSION_END:
        return None
    # Cheap trading-chart close gate BEFORE resampling. An N-minute bin that
    # started at bars.index[0] (session open) is only complete when the
    # elapsed whole minutes since that open are N-1, 2N-1, ...
    elapsed = int((now_time - bars.index[0]).total_seconds() // 60)
    if elapsed < 0 or elapsed % minutes != (minutes - 1):
        return None

    env = preopen_env if preopen_env is not None else preopen_1h_environment(sessions)
    if env.bias is None:
        return None
    bias = env.bias
    side = "long" if bias == "up" else "short"

    prior_high = float(prior_day_bars["high"].max())
    prior_low = float(prior_day_bars["low"].min())
    if prior_high <= prior_low:
        return None

    zone_lo, zone_hi = _fib_zone(prior_high, prior_low, side)
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
    # Creamer: if the fib pocket sits inside value, skip.
    if side == "long" and zone_hi > float(profile.val):
        return None
    if side == "short" and zone_lo < float(profile.vah):
        return None

    five = _resample_closed(bars, minutes=minutes)
    if len(five) < _MIN_TRADE_BARS:
        return None
    # Only decide on a trading-chart close, never mid-bin.
    if five.index[-1] + pd.Timedelta(minutes=minutes - 1) != now_time:
        return None

    probe = five.iloc[-2]
    confirm = five.iloc[-1]
    baseline = five["volume"].iloc[-(_VOLUME_LOOKBACK + 2):-2]
    baseline_vol = float(baseline.mean())
    if baseline_vol <= 0:
        return None
    if max(float(probe["volume"]), float(confirm["volume"])) < min_rel_volume * baseline_vol:
        return None

    loc_price = float(probe["low"]) if side == "long" else float(probe["high"])
    if not (zone_lo <= loc_price <= zone_hi):
        return None
    if side == "long" and loc_price > float(profile.val):
        return None
    if side == "short" and loc_price < float(profile.vah):
        return None

    if not _probe_absorbed(probe, side, min_wick_frac):
        return None
    if not _confirm_shift(probe, confirm, side):
        return None

    footprint_tier = "bar_only_5m_proxy"
    probe_fp = None
    confirm_fp = None
    visible_ticks = ticks_up_to(ticks_so_far, now_time)
    if visible_ticks is not None:
        fps = fp.footprint_5m(visible_ticks, origin=bars.index[0])
        probe_fp = fps.get(five.index[-2])
        confirm_fp = fps.get(five.index[-1])
        if probe_fp is not None and confirm_fp is not None and probe_fp.complete and confirm_fp.complete:
            if not fp.probe_absorbed(probe_fp, side):
                return None
            if not fp.confirm_dominance(confirm_fp, side):
                return None
            footprint_tier = "footprint_5m"
        elif probe_fp is not None or confirm_fp is not None:
            footprint_tier = "bar_only_5m_proxy_incomplete_footprint"

    # ATR on the 5-minute chart (prior session + today so far). Using the
    # 1-minute ATR here was the previous bug: stops sat a few cents off
    # the probe extreme and got run by 1-minute noise.
    atr_src = pd.concat([prior_5, five])
    atr_series = ctx.atr(atr_src, period=_ATR_PERIOD)
    now_atr = float(atr_series.iloc[-1])
    if pd.isna(now_atr) or now_atr <= 0:
        return None

    entry = float(confirm["close"])
    if side == "long":
        stop = float(probe["low"]) - stop_atr_mult * now_atr
        if stop >= entry:
            return None
        target = entry + target_r_multiple * (entry - stop)
    else:
        stop = float(probe["high"]) + stop_atr_mult * now_atr
        if stop <= entry:
            return None
        target = entry - target_r_multiple * (stop - entry)

    env_gex, symbol_gex = resolve_gex_env(gex_snapshot)
    if env_gex is not None:
        vol_regime = env_gex.regime
        gex_source = env_gex.source
    else:
        vol_regime = _vol_regime(prior_day_bars)
        gex_source = "unavailable_vol_regime_proxy"

    walls = symbol_gex or env_gex
    gex_tag = "gex" if env_gex is not None else "no_gex"
    return MicroSignal(
        symbol=symbol,
        strategy="auction_reclaim",
        direction=side,
        signal_time=now_time,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        order_type="next_open",
        context={
            "value_bias": bias,
            "environment_tf": "1h",
            "environment_when": "preopen",
            "environment_structure": env.structure,
            "environment_sessions": _ENV_SESSIONS,
            "preopen_trader_side": env.trader_side,
            "preopen_trader_pace": env.trader_pace,
            "preopen_trader_momentum": env.trader_momentum,
            "preopen_trader_pressure": env.trader_pressure,
            "preopen_pace_ratio": env.pace_ratio,
            "preopen_volume_ratio": env.volume_ratio,
            "preopen_last_volume": env.last_volume,
            "preopen_prev_volume": env.prev_volume,
            "preopen_last_vwap": env.last_vwap,
            "preopen_peak_hour": env.peak_hour,
            "vol_regime": vol_regime,
            "gex_source": gex_source,
            "gex_net": env_gex.net_gex if env_gex is not None else None,
            "gex_call_wall": walls.call_wall if walls is not None else None,
            "gex_put_wall": walls.put_wall if walls is not None else None,
            "gex_gamma_flip": walls.gamma_flip if walls is not None else None,
            "prior_val": float(profile.val),
            "prior_vah": float(profile.vah),
            "prior_poc": float(profile.poc) if profile.poc is not None else None,
            "fib_zone_lo": zone_lo,
            "fib_zone_hi": zone_hi,
            "probe_low": float(probe["low"]),
            "probe_high": float(probe["high"]),
            "chart_minutes": minutes,
            "atr_timeframe": f"{minutes}m",
            "atr": now_atr,
            "tier": f"{footprint_tier}_{gex_tag}",
            "target_r_multiple": target_r_multiple,
        },
    )
