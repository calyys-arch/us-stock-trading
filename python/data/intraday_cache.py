"""
Local 1-minute bar cache — parquet, chunked by calendar month.

Why this is a SEPARATE module from python/data/price_cache.py (daily bars):
  - Volume: 20 symbols x 12 months x ~9,600 bars/month (RTH) is ~2.3M rows.
    A single CSV per symbol (price_cache's approach) would be slow to
    reload every WFO iteration; parquet-per-month keeps each file small,
    columnar-compressed, and lets a backtest touching only a few months
    avoid reading the rest.
  - Fetch granularity: IB caps 1-minute TRADES history to ~1 calendar month
    per request (python/data/ibkr_price_source.fetch_ibkr_intraday_month),
    so "coverage" here is inherently per-(symbol, month), not per-symbol
    requested-range like price_cache.
  - No yfinance fallback: yfinance's 1-minute data only covers the last ~7
    days, which is useless for WFO history. IB is the only source
    (docs/microstructure_pivot_plan.md — "只用 IB，不訂外部資料源"); when IB
    is unreachable this module raises loudly rather than silently
    substituting thin data.

Cache layout:
    data/history_1m/<SYMBOL>/<YYYY-MM>.parquet   — one month of RTH 1m bars
    data/history_1m/<SYMBOL>/_meta.json           — per-month fetch status

A month is "closed" (safe to cache permanently, never re-fetched) once it
is strictly before the current calendar month. The current (in-progress)
month is always re-fetched on request so today's bars stay fresh —
mirrors price_cache's `refresh` semantics but keyed off calendar time
instead of an explicit flag, since intraday backfill is expected to run as
a recurring incremental job (scripts/backfill_intraday.py), not a
one-shot fetch-and-forget.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/history_1m")
BROKER_CONFIG_PATH = Path("configs/broker.yaml")
_META_FILENAME = "_meta.json"

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _symbol_dir(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / symbol.upper()


def _month_parquet_path(cache_dir: Path, symbol: str, month_start: pd.Timestamp) -> Path:
    return _symbol_dir(cache_dir, symbol) / f"{month_start:%Y-%m}.parquet"


def _meta_path(cache_dir: Path, symbol: str) -> Path:
    return _symbol_dir(cache_dir, symbol) / _META_FILENAME


def _load_meta(cache_dir: Path, symbol: str) -> dict:
    path = _meta_path(cache_dir, symbol)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("intraday_cache: unreadable %s — treating as empty", path)
        return {}


def _save_meta(cache_dir: Path, symbol: str, meta: dict) -> None:
    _symbol_dir(cache_dir, symbol).mkdir(parents=True, exist_ok=True)
    _meta_path(cache_dir, symbol).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """First-of-month timestamps covering [start, end], inclusive."""
    start_month = pd.Timestamp(start).replace(day=1)
    end_month = pd.Timestamp(end).replace(day=1)
    return list(pd.date_range(start_month, end_month, freq="MS"))


def _is_month_closed(month_start: pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    if now.tz is not None:
        now = now.tz_localize(None)
    current_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    target_month = pd.Timestamp(month_start).replace(day=1)
    if target_month.tz is not None:
        target_month = target_month.tz_localize(None)
    return target_month < current_month


def is_month_cached(
    symbol: str,
    month_start: pd.Timestamp,
    cache_dir: str | Path = CACHE_DIR,
    now: pd.Timestamp | None = None,
) -> bool:
    """True iff this month's parquet exists on disk AND (the month is
    closed OR it was fetched very recently) — the current in-progress month
    is never considered "done" so incremental backfill runs keep refreshing
    it, matching the module's `refresh semantics keyed off calendar time`
    described in the module docstring."""
    cache_dir = Path(cache_dir)
    path = _month_parquet_path(cache_dir, symbol, month_start)
    if not path.exists():
        return False
    meta = _load_meta(cache_dir, symbol)
    key = f"{pd.Timestamp(month_start):%Y-%m}"
    if key not in meta:
        return False
    return _is_month_closed(month_start, now)


def backfill_symbol_months(
    symbol: str,
    months: list[pd.Timestamp],
    ib,
    cache_dir: str | Path = CACHE_DIR,
    force: bool = False,
    bar_size: str = "1 min",
) -> dict:
    """Fetch + cache every month in `months` for one symbol over an already
    open IB connection (see python.data.ibkr_price_source.open_ib_connection).
    Writes each month's parquet + updates the meta sidecar IMMEDIATELY after
    each successful fetch — not batched at the end — so a killed/interrupted
    backfill run can be resumed from where it left off (scripts/
    backfill_intraday.py re-invokes this and closed, already-cached months
    are skipped for free).

    Returns a summary dict: {"fetched": [...], "skipped": [...],
    "empty": [...], "failed": [...]}."""
    from .ibkr_price_source import fetch_ibkr_intraday_month

    cache_dir = Path(cache_dir)
    summary = {"fetched": [], "skipped": [], "empty": [], "failed": []}
    meta = _load_meta(cache_dir, symbol)

    for month_start in months:
        month_start = pd.Timestamp(month_start).replace(day=1)
        key = f"{month_start:%Y-%m}"

        if not force and is_month_cached(symbol, month_start, cache_dir):
            summary["skipped"].append(key)
            continue

        try:
            df = fetch_ibkr_intraday_month(ib, symbol, month_start, bar_size=bar_size)
        except Exception as exc:
            log.error("intraday_cache: %s %s failed — %s", symbol, key, exc)
            summary["failed"].append(key)
            continue

        _symbol_dir(cache_dir, symbol).mkdir(parents=True, exist_ok=True)
        if df.empty:
            summary["empty"].append(key)
        else:
            df[_OHLCV_COLUMNS].to_parquet(_month_parquet_path(cache_dir, symbol, month_start))
            summary["fetched"].append(key)

        meta[key] = {
            "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            "n_bars": int(len(df)),
            "closed": bool(_is_month_closed(month_start)),
        }
        _save_meta(cache_dir, symbol, meta)
        log.info("intraday_cache: %s %s -> %d bars", symbol, key, len(df))

    return summary


def get_cached_intraday_panel(
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = CACHE_DIR,
) -> pd.DataFrame:
    """Read cached 1-minute bars for `symbols` over [start, end] from disk
    ONLY — this function never fetches over the network (backfill is a
    separate, explicit step via scripts/backfill_intraday.py or
    backfill_symbol_months). Returns a long-form DataFrame indexed by
    (ts, code) with columns [open, high, low, close, volume], restricted
    to bars whose timestamp falls in [start, end].

    Raises RuntimeError if none of `symbols` have any cached data at all in
    range — mirrors price_cache.get_cached_price_panel's fail-loud contract
    (an empty backtest panel is a bug, not a valid "no data" result)."""
    cache_dir = Path(cache_dir)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    symbols = [s.upper() for s in symbols]

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for month_start in month_range(start_ts, end_ts):
            path = _month_parquet_path(cache_dir, symbol, month_start)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            if df.empty:
                continue
            df = df.copy()
            df["code"] = symbol
            df.index.name = "ts"
            frames.append(df.reset_index().set_index(["ts", "code"]))

    if not frames:
        raise RuntimeError(
            f"intraday_cache: no cached 1-minute data for any of {len(symbols)} symbols in "
            f"[{start_ts}, {end_ts}] — run scripts/backfill_intraday.py first"
        )

    return pd.concat(frames).sort_index()


def cached_symbol_coverage(
    symbols: list[str],
    cache_dir: str | Path = CACHE_DIR,
) -> dict[str, dict]:
    """Per-symbol summary of cached months and total bar count — used by
    scripts/backfill_intraday.py to report progress and by health-check
    style reporting to flag symbols with gaps."""
    cache_dir = Path(cache_dir)
    out: dict[str, dict] = {}
    for symbol in symbols:
        meta = _load_meta(cache_dir, symbol.upper())
        months = sorted(meta.keys())
        total_bars = sum(int(v.get("n_bars", 0)) for v in meta.values())
        out[symbol.upper()] = {
            "months_cached": months,
            "n_months": len(months),
            "total_bars": total_bars,
        }
    return out
