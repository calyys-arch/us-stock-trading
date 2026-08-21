"""
Context Engine — per-session liquidity levels, VWAP family, volume profile,
and opening range, computed from 1-minute OHLCV bars.

This is pure, stateless computation over pandas DataFrames — no I/O, no IB
calls. Callers (python/backtest/intraday_engine.py for backtest, and
dashboard/app.py's /api/chart/{symbol}/context for live display) feed it
already-loaded bars (python/data/intraday_cache.get_cached_intraday_panel).

Bar index convention: a tz-naive DatetimeIndex in US/Eastern (the timezone
IB's reqHistoricalData returns with formatDate=1) — never UTC. Every
function in this module assumes that; there is no timezone conversion here.

Honest scope note: PMH/PML (pre-market high/low) require extended-hours
bars. python/data/intraday_cache's default backfill uses useRTH=True (RTH
only, matching the rest of this system's RTH-only convention —
python/interfaces/ibkr_feed.py docstring), so `premarket_bars` is an
OPTIONAL argument throughout this module; when omitted, PMH/PML come back
as None rather than being silently approximated from RTH bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

_SESSION_OPEN = time(9, 30)
_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# ─────────────────────────────────────────────────────────────────────────────
# ATR (separate from python/core/data_engine.py's Indicators.atr, which
# operates on the live engine's streaming Candle objects — this is a
# vectorized pandas version for batch/backtest use over a bar DataFrame).
# ─────────────────────────────────────────────────────────────────────────────

def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple (non-Wilder) rolling-mean ATR — deterministic and easy to
    reason about for parameter-grid sweeps (Chan discipline favors simple,
    auditable formulas over subtly different smoothing conventions)."""
    return true_range(bars).rolling(period, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# VWAP family
# ─────────────────────────────────────────────────────────────────────────────

def _typical_price(bars: pd.DataFrame) -> pd.Series:
    return (bars["high"] + bars["low"] + bars["close"]) / 3.0


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative volume-weighted average price from the first bar in
    `bars` (callers pass bars already sliced to one session, anchored at
    09:30 ET, to get the conventional "session VWAP")."""
    tp = _typical_price(bars)
    cum_pv = (tp * bars["volume"]).cumsum()
    cum_vol = bars["volume"].cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).rename("vwap")


def vwap_bands(bars: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP plus +/-1sigma and +/-2sigma bands, where sigma is the
    cumulative volume-weighted standard deviation of typical price around
    the running VWAP (the standard "VWAP bands" construction)."""
    tp = _typical_price(bars)
    vwap = session_vwap(bars)
    cum_vol = bars["volume"].cumsum().replace(0, np.nan)
    sq_dev = ((tp - vwap) ** 2) * bars["volume"]
    variance = sq_dev.cumsum() / cum_vol
    sigma = np.sqrt(variance.clip(lower=0))
    return pd.DataFrame({
        "vwap": vwap,
        "upper_1": vwap + sigma,
        "lower_1": vwap - sigma,
        "upper_2": vwap + 2 * sigma,
        "lower_2": vwap - 2 * sigma,
    })


def anchored_vwap(bars: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.Series:
    """VWAP anchored at `anchor_ts` — bars before the anchor get NaN, bars
    from the anchor onward get a fresh cumulative VWAP restarting at zero.
    Used for anchored-VWAP-from-open, anchored-VWAP-from-swing-high/low.
    Documented risk (docs/microstructure_pivot_plan.md #10): anchored VWAP
    should not be trusted across an ex-dividend gap (TRADES bars are not
    split/dividend-adjusted)."""
    out = pd.Series(index=bars.index, dtype=float)
    mask = bars.index >= anchor_ts
    out.loc[mask] = session_vwap(bars.loc[mask])
    out.loc[~mask] = np.nan
    return out.rename("anchored_vwap")


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity levels: YDH/YDL, PMH/PML, EQH/EQL, round numbers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquidityLevels:
    ydh: float | None = None
    ydl: float | None = None
    pmh: float | None = None
    pml: float | None = None
    eq_highs: list[float] = field(default_factory=list)
    eq_lows: list[float] = field(default_factory=list)
    round_levels: list[float] = field(default_factory=list)


def _round_step_for_price(price: float) -> float:
    """Adaptive round-number grid step by price magnitude — a $5 grid on a
    $9 stock would be meaningless, a $1 grid on a $600 stock is too fine."""
    if price >= 200:
        return 10.0
    if price >= 50:
        return 5.0
    if price >= 20:
        return 1.0
    return 0.5


def _round_levels_near(price: float, span: int = 3, step: float | None = None) -> list[float]:
    step = step or _round_step_for_price(price)
    base = round(price / step) * step
    return sorted({round(base + i * step, 2) for i in range(-span, span + 1)})


def _equal_extrema(
    series: pd.Series,
    lookback: int,
    tolerance: float,
    find_max: bool,
) -> list[float]:
    """Cluster the top-`lookback` local extrema within `tolerance` of each
    other into "equal high/low" levels (>=2 touches). Heuristic, not exact
    swing-point detection — documented as such; good enough for a proxy
    liquidity-pool feature, not a claim of precise market-maker intent."""
    if series.empty:
        return []
    recent = series.tail(lookback).dropna()
    if recent.empty:
        return []
    values = sorted(recent.tolist(), reverse=find_max)
    clusters: list[list[float]] = []
    for v in values:
        placed = False
        for cluster in clusters:
            if abs(cluster[0] - v) <= tolerance:
                cluster.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    levels = [float(np.mean(c)) for c in clusters if len(c) >= 2]
    return sorted(levels, reverse=find_max)


def target_resistance_levels(levels: LiquidityLevels, current_price: float) -> list[float]:
    """Candidate levels for a LONG position's profit TARGET (i.e. resistance
    above `current_price`) — deliberately narrower than the raw
    `LiquidityLevels.eq_highs` field: only YDH/PMH and the nearby
    round-number grid are eligible, `eq_highs` is excluded.

    Canonical, shared implementation (originally lived only in
    sweep_reclaim.py; pulled up here so every signal that needs an
    "opposite liquidity pool" TARGET — sweep_reclaim, vp_breakout — uses
    ONE definition instead of three subtly different copies). Investigated
    after the 2026-07-30 sweep_reclaim backtest report's catastrophic
    results: `eq_highs`/`eq_lows` are clustered from only the last
    `eq_lookback` (default 20) bars at a tight `eq_atr_mult` (default 0.1x
    ATR) tolerance, so they sit only a small fraction of an ATR from the
    current price — using them as a TARGET meant the nearest-level pick
    almost always landed on the closest micro-noise cluster instead of a
    genuine, structurally distant liquidity pool (empirically ~90% of
    sweep_reclaim signals ended up with a target CLOSER than the stop,
    median target/stop distance ratio ~0.18 — risking ~5x what was being
    targeted). `eq_highs`/`eq_lows` stay legitimate for SWEEP DETECTION
    (a same-window cluster is exactly what a sweep needs to pierce), just
    never for a TARGET."""
    out = [lvl for lvl in (levels.ydh, levels.pmh) if lvl is not None]
    out.extend(r for r in levels.round_levels if r > current_price)
    return out


def target_support_levels(levels: LiquidityLevels, current_price: float) -> list[float]:
    """Support-side counterpart of `target_resistance_levels` — see that
    function's docstring for why `eq_lows` is excluded here."""
    out = [lvl for lvl in (levels.ydl, levels.pml) if lvl is not None]
    out.extend(r for r in levels.round_levels if r < current_price)
    return out


def nearest_liquidity_target(levels: list[float], price: float, direction: str) -> float | None:
    """Nearest level on the side a `direction` trade is heading toward —
    used as a simple profit target (the nearest opposite/next liquidity
    pool). `direction="long"` looks for the nearest level ABOVE `price`;
    `direction="short"` looks for the nearest level BELOW `price`."""
    if not levels:
        return None
    if direction == "long":
        candidates = [lvl for lvl in levels if lvl > price]
        return min(candidates) if candidates else None
    candidates = [lvl for lvl in levels if lvl < price]
    return max(candidates) if candidates else None


def compute_liquidity_levels(
    bars_today: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None = None,
    premarket_bars: pd.DataFrame | None = None,
    eq_lookback: int = 20,
    eq_atr_mult: float = 0.1,
    include_eq_levels: bool = True,
) -> LiquidityLevels:
    """Compute YDH/YDL (from `prior_day_bars`), PMH/PML (from
    `premarket_bars`, None if not supplied), EQH/EQL (from `bars_today`'s
    last `eq_lookback` bars, tolerance = eq_atr_mult * ATR), and nearby
    round-number levels anchored to the latest close in `bars_today`.

    `include_eq_levels=False` skips the EQH/EQL computation (and the ATR
    call it requires) entirely, leaving `eq_highs`/`eq_lows` empty — this
    function is called fresh on every unheld bar by signals that need a
    live liquidity-level snapshot (sweep_reclaim.py, vp_breakout.py), and
    profiling showed the ATR + `_equal_extrema` clustering is ~85% of this
    function's cost. vp_breakout.py's `target_resistance_levels`/
    `target_support_levels` never read `eq_highs`/`eq_lows` (see those
    functions' docstring — deliberately excluded, not just unused), so
    vp_breakout passes `include_eq_levels=False` to skip work whose result
    it would discard anyway; sweep_reclaim.py (which DOES use eq_highs/
    eq_lows for sweep detection) leaves the default `True`. Purely a
    performance opt-out — YDH/YDL/PMH/PML/round_levels are bit-for-bit
    identical either way."""
    levels = LiquidityLevels()

    if prior_day_bars is not None and not prior_day_bars.empty:
        levels.ydh = float(prior_day_bars["high"].max())
        levels.ydl = float(prior_day_bars["low"].min())

    if premarket_bars is not None and not premarket_bars.empty:
        levels.pmh = float(premarket_bars["high"].max())
        levels.pml = float(premarket_bars["low"].min())

    if not bars_today.empty:
        if include_eq_levels:
            current_atr = float(atr(bars_today, period=14).iloc[-1])
            tolerance = max(current_atr * eq_atr_mult, 1e-6)
            levels.eq_highs = _equal_extrema(bars_today["high"], eq_lookback, tolerance, find_max=True)
            levels.eq_lows = _equal_extrema(bars_today["low"], eq_lookback, tolerance, find_max=False)
        levels.round_levels = _round_levels_near(float(bars_today["close"].iloc[-1]))

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# Volume profile (bar-approximated POC/VAH/VAL)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    poc: float | None = None       # point of control (highest-volume price bin)
    vah: float | None = None       # value area high
    val: float | None = None       # value area low
    bin_edges: list[float] = field(default_factory=list)
    bin_volume: list[float] = field(default_factory=list)


def volume_profile_from_arrays(
    tp: np.ndarray, volume: np.ndarray, n_bins: int = 30, value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Core POC/VAH/VAL computation, factored out of `volume_profile` below
    to accept raw numpy arrays directly. Identical math to `volume_profile`
    — `volume_profile(bars)` is a thin DataFrame-unpacking wrapper around
    this. Callers that recompute a volume profile on a GROWING bars-so-far
    prefix every bar (vp_breakout.py, evaluated once per unheld bar across a
    multi-month/20-symbol backtest) should call this directly with arrays
    they already hold, since going through a fresh DataFrame slice + pandas
    Series arithmetic (`_typical_price`) on every call adds pandas-indexing
    overhead that dominates the actual (cheap) bincount work at this scale."""
    if len(tp) == 0:
        return VolumeProfile()

    lo, hi = float(tp.min()), float(tp.max())
    if hi <= lo:
        return VolumeProfile(poc=lo, vah=lo, val=lo, bin_edges=[lo, hi], bin_volume=[float(volume.sum())])

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(tp, edges) - 1, 0, n_bins - 1)
    bin_volume = np.bincount(bin_idx, weights=volume.astype(float, copy=False), minlength=n_bins)

    total_vol = bin_volume.sum()
    if total_vol <= 0:
        return VolumeProfile(bin_edges=list(edges), bin_volume=list(bin_volume))

    poc_idx = int(np.argmax(bin_volume))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    target = total_vol * value_area_pct
    accumulated = bin_volume[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    n_bins_actual = len(bin_volume)
    while accumulated < target and (lo_idx > 0 or hi_idx < n_bins_actual - 1):
        expand_lo = bin_volume[lo_idx - 1] if lo_idx > 0 else -1.0
        expand_hi = bin_volume[hi_idx + 1] if hi_idx < n_bins_actual - 1 else -1.0
        if expand_hi >= expand_lo:
            hi_idx += 1
            accumulated += bin_volume[hi_idx]
        else:
            lo_idx -= 1
            accumulated += bin_volume[lo_idx]

    return VolumeProfile(
        poc=float(bin_centers[poc_idx]),
        vah=float(bin_centers[hi_idx]),
        val=float(bin_centers[lo_idx]),
        bin_edges=list(edges),
        bin_volume=list(bin_volume),
    )


def volume_profile(bars: pd.DataFrame, n_bins: int = 30, value_area_pct: float = 0.70) -> VolumeProfile:
    """Approximate intraday volume profile from 1-minute bars: each bar's
    volume is assigned entirely to the price bin containing its typical
    price (bar-level approximation, NOT tick-precision — honestly
    documented in docs/microstructure_pivot_plan.md §2 "Volume Profile").
    POC = bin with the most volume; VAH/VAL = the price range containing
    `value_area_pct` of total volume, expanded outward from the POC."""
    if bars.empty:
        return VolumeProfile()
    tp = _typical_price(bars).to_numpy()
    volume = bars["volume"].to_numpy(dtype=float)
    return volume_profile_from_arrays(tp, volume, n_bins=n_bins, value_area_pct=value_area_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Opening range
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpeningRange:
    high: float | None = None
    low: float | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


def opening_range(bars_today: pd.DataFrame, minutes: int = 15) -> OpeningRange:
    """High/low of the first `minutes` of the session. `bars_today` must
    already be sliced to one session starting at the session open (09:30
    ET) — this function does not know about the trading calendar itself
    (python/core/calendar.py owns that)."""
    if bars_today.empty:
        return OpeningRange()
    start = bars_today.index[0]
    end = start + pd.Timedelta(minutes=minutes)
    window = bars_today.loc[(bars_today.index >= start) & (bars_today.index < end)]
    if window.empty:
        return OpeningRange(start=start, end=end)
    return OpeningRange(
        high=float(window["high"].max()),
        low=float(window["low"].min()),
        start=start,
        end=end,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision-chart resample (1m raw feed → closed N-minute bars at 09:30 ET)
# Same closed-bin rule as vsa_no_demand / auction_reclaim (_resample_closed):
# a 09:30 bin is complete when the 09:34 1m bar has closed (elapsed minutes
# since 09:30 ≡ minutes-1 mod minutes). Origin is the session open, NOT the
# first 1m print that happened to arrive (mid-session scheduler start).
# Agg is the same OHLCV reduction as python/data/intraday_cache.resample_ohlcv.
# ─────────────────────────────────────────────────────────────────────────────

def session_open_origin(ts: pd.Timestamp) -> pd.Timestamp:
    """09:30 ET of `ts`'s calendar date (tz-naive, matching this module)."""
    t = pd.Timestamp(ts)
    return t.normalize() + pd.Timedelta(hours=_SESSION_OPEN.hour, minutes=_SESSION_OPEN.minute)


def session_bin_just_closed(bars_1m: pd.DataFrame, minutes: int) -> bool:
    """True iff the last 1m bar completes an N-minute bin aligned to 09:30 ET."""
    if bars_1m.empty or minutes < 1:
        return False
    if minutes == 1:
        return True
    origin = session_open_origin(bars_1m.index[-1])
    elapsed = int((bars_1m.index[-1] - origin).total_seconds() // 60)
    if elapsed < 0:
        return False
    return elapsed % minutes == (minutes - 1)


def closed_session_bars(bars_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Closed N-minute OHLCV aligned to 09:30 ET, built from 1-minute bars.

    Incomplete current bin is dropped. Bins are per calendar day so a
    window never spans the overnight gap. `minutes <= 1` returns `bars_1m`
    unchanged (the raw feed). Missing 1m prints inside a completed clock
    bin are allowed — completeness is the 09:30-aligned clock, not a
    requirement that all `minutes` prints were observed (a live scheduler
    that attached mid-session still waits for 09:34 / 09:39 / …).
    """
    if minutes <= 1:
        return bars_1m
    if bars_1m.empty:
        return bars_1m.iloc[0:0]
    freq = f"{int(minutes)}min"
    chunks: list[pd.DataFrame] = []
    for _, day_df in bars_1m.groupby(bars_1m.index.normalize()):
        origin = session_open_origin(day_df.index[0])
        last = day_df.index[-1]
        elapsed = int((last - origin).total_seconds() // 60)
        if elapsed < 0:
            continue
        last_offset = minutes - 1
        if elapsed % minutes != last_offset:
            cut = int(elapsed % minutes) + 1
            closed = day_df.iloc[:-cut]
        else:
            closed = day_df
        if closed.empty:
            continue
        resampled = closed.resample(freq, origin=origin).agg(_OHLCV_AGG)
        chunks.append(resampled.dropna(subset=["open"]))
    if not chunks:
        return bars_1m.iloc[0:0]
    return pd.concat(chunks).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Combined context state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextState:
    liquidity: LiquidityLevels
    vwap: pd.DataFrame        # columns: vwap, upper_1, lower_1, upper_2, lower_2
    volume_profile: VolumeProfile
    opening_range: OpeningRange
    atr14: pd.Series


def compute_context(
    bars_today: pd.DataFrame,
    prior_day_bars: pd.DataFrame | None = None,
    premarket_bars: pd.DataFrame | None = None,
    or_minutes: int = 15,
    eq_lookback: int = 20,
    eq_atr_mult: float = 0.1,
) -> ContextState:
    """One-call orchestrator: everything a signal module needs for a given
    session's bars-so-far. `bars_today` should be sliced up to "now" in a
    backtest event loop (never the full session — that would look ahead)."""
    return ContextState(
        liquidity=compute_liquidity_levels(
            bars_today, prior_day_bars, premarket_bars,
            eq_lookback=eq_lookback, eq_atr_mult=eq_atr_mult,
        ),
        vwap=vwap_bands(bars_today),
        volume_profile=volume_profile(bars_today),
        opening_range=opening_range(bars_today, minutes=or_minutes),
        atr14=atr(bars_today, period=14),
    )
