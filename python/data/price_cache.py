"""
Local daily-bar cache: fetch once, re-run backtests offline.

The self-improve loop (scripts/self_improve_loop.py) re-runs the same
[start, end] window many times (grid search x WFO folds x iterations);
re-hitting a network source on every run would be slow, rate-limited, and —
worse — could silently change the data between iterations, making
promotion decisions non-reproducible. This cache pins the data to disk:

  - one CSV per symbol under data/history/<SYMBOL>.csv (gitignored,
    regenerable runtime data — same class as data/ticks/ and data/depth/),
  - a _meta.json sidecar recording, per symbol, WHICH RANGE WAS REQUESTED
    and WHICH SOURCE served it. Coverage checks compare against the
    requested range (not the first/last bar present) so a symbol that
    IPO'd mid-range is correctly treated as "fully cached" rather than
    endlessly re-fetched.

Source policy (configs/broker.yaml `historical_data_source`):
  - "ibkr" (default): python/data/ibkr_price_source.py via IB Gateway.
    If IB is unreachable this logs a clear warning and falls back to
    yfinance (python/simulation/hist_data_us.py) — the run proceeds, and
    the returned meta records which source actually served the data so
    reports can label it honestly (never silently mixed).
  - "yfinance": skip IB entirely.

`refresh=True` forces a re-fetch of every requested symbol (use when you
want bars through the latest trading day).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from ..core.data_quality import quality_report

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/history")
BROKER_CONFIG_PATH = Path("configs/broker.yaml")
_META_FILENAME = "_meta.json"

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / _META_FILENAME


def _load_meta(cache_dir: Path) -> dict:
    path = _meta_path(cache_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("price_cache: unreadable %s — treating cache as empty", path)
        return {}


def _save_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(cache_dir).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _symbol_csv(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.upper()}.csv"


def _is_covered(meta_entry: dict | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Cache hit iff a PREVIOUS REQUEST's range contains [start, end].
    Requested-range (not bar-range) comparison deliberately: a symbol that
    IPO'd after `start` has no bars near `start`, yet its cache is complete
    for any sub-range of what was already requested."""
    if not meta_entry:
        return False
    try:
        req_start = pd.Timestamp(meta_entry["requested_start"])
        req_end = pd.Timestamp(meta_entry["requested_end"])
    except (KeyError, ValueError):
        return False
    return req_start <= start and req_end >= end


def _historical_source_setting(broker_config_path: str | Path) -> str:
    try:
        with open(broker_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("historical_data_source", "ibkr")).lower()
    except FileNotFoundError:
        return "ibkr"


def _fetch_remote(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    broker_config_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    """Fetch a (date, code) panel for `symbols`, returning (panel, source_name).
    IBKR first (unless configured off), yfinance as the explicit fallback."""
    source_setting = _historical_source_setting(broker_config_path)

    if source_setting == "ibkr":
        from .ibkr_price_source import IbkrHistoricalUnavailable, fetch_ibkr_daily_bars

        try:
            panel, _flags = fetch_ibkr_daily_bars(symbols, start, end, config_path=broker_config_path)
            return panel, "ibkr"
        except IbkrHistoricalUnavailable as exc:
            log.warning(
                "price_cache: IBKR historical source unavailable (%s) — "
                "FALLING BACK to yfinance for %d symbols. Reports will label this run's "
                "data source accordingly.", exc, len(symbols),
            )

    from ..simulation.hist_data_us import build_price_panel

    panel, _flags = build_price_panel(symbols, str(start.date()), str(end.date()))
    return panel, "yfinance"


def get_cached_price_panel(
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    refresh: bool = False,
    cache_dir: str | Path = CACHE_DIR,
    broker_config_path: str | Path = BROKER_CONFIG_PATH,
) -> tuple[pd.DataFrame, dict, dict]:
    """Return (panel, quality_flags, meta) for `symbols` over [start, end].

    panel: MultiIndex (date, code) DataFrame with columns
        [open, high, low, close, volume, adv_20d_dollars] — the same
        contract as hist_data_us.build_price_panel.
    quality_flags: {symbol: quality_report} for symbols with flagged data.
    meta: {"sources": {source_name: [symbols...]}, "from_cache": [...],
           "fetched": [...]} for honest report labeling.
    """
    cache_dir = Path(cache_dir)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    symbols = [s.upper() for s in symbols]

    disk_meta = _load_meta(cache_dir)
    to_fetch = [
        s for s in symbols
        if refresh or not _is_covered(disk_meta.get(s), start_ts, end_ts) or not _symbol_csv(cache_dir, s).exists()
    ]
    from_cache = [s for s in symbols if s not in to_fetch]

    fetched_source = None
    if to_fetch:
        log.info("price_cache: fetching %d/%d symbols (%s)%s",
                 len(to_fetch), len(symbols),
                 ", ".join(to_fetch[:8]) + ("..." if len(to_fetch) > 8 else ""),
                 " [refresh forced]" if refresh else "")
        fetched_panel, fetched_source = _fetch_remote(to_fetch, start_ts, end_ts, broker_config_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        fetched_codes = fetched_panel.index.get_level_values(1).unique()
        for symbol in to_fetch:
            if symbol not in fetched_codes:
                log.warning("price_cache: %s could not be fetched from %s — not cached",
                            symbol, fetched_source)
                continue
            df = fetched_panel.xs(symbol, level=1)[_OHLCV_COLUMNS]
            df.to_csv(_symbol_csv(cache_dir, symbol), index_label="date")
            disk_meta[symbol] = {
                "requested_start": str(start_ts.date()),
                "requested_end": str(end_ts.date()),
                "source": fetched_source,
                "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            }
        _save_meta(cache_dir, disk_meta)

    frames: list[pd.DataFrame] = []
    quality_flags: dict = {}
    sources: dict[str, list[str]] = {}
    for symbol in symbols:
        csv_path = _symbol_csv(cache_dir, symbol)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True).sort_index()
        # adv is computed on the FULL cached series before slicing so the
        # 20-day rolling window is identical no matter what sub-range a
        # caller asks for (reproducibility across loop iterations).
        df["adv_20d_dollars"] = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean()
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            continue

        report = quality_report(df["close"])
        if report["n_extreme_moves_flagged"] > 0 or report["n_zero_or_negative_prices"] > 0:
            quality_flags[symbol] = report

        source = (disk_meta.get(symbol) or {}).get("source", "unknown")
        sources.setdefault(source, []).append(symbol)

        df = df.copy()
        df["code"] = symbol
        df.index.name = "date"
        frames.append(df.reset_index().set_index(["date", "code"]))

    if not frames:
        raise RuntimeError(
            f"price_cache: no data available for any of {len(symbols)} symbols in "
            f"[{start_ts.date()}, {end_ts.date()}]"
        )

    panel = pd.concat(frames).sort_index()
    meta = {
        "sources": sources,
        "from_cache": sorted(from_cache),
        "fetched": sorted(to_fetch),
        "fetched_source": fetched_source,
    }
    return panel, quality_flags, meta


def first_available_dates(
    symbols: list[str],
    cache_dir: str | Path = CACHE_DIR,
) -> dict[str, pd.Timestamp]:
    """Earliest cached bar per symbol — used by scripts/refresh_universe.py
    to warn when a selected name's history starts after the backtest start
    (e.g. a recent IPO in today's top-20 list)."""
    cache_dir = Path(cache_dir)
    out: dict[str, pd.Timestamp] = {}
    for symbol in symbols:
        csv_path = _symbol_csv(cache_dir, symbol.upper())
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        if len(df):
            out[symbol.upper()] = df.index.min()
    return out
