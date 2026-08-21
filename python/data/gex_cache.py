"""
Persist / load naive GEX snapshots under data/gex/<SYMBOL>/<YYYYMMDD>.json.

Snapshots are as-of the capture day. There is no historical options chain
in this repo, so a 2025–2026 1-minute backtest will almost always see
None — callers must fail closed (skip the GEX overlay) rather than invent
dealer gamma. Going forward, scripts/snapshot_gex.py writes today's file.

Fetcher: yfinance option_chain (live / near-term only). IBKR greeks are
a better source when the Gateway session has options permissions; that
path is not wired here because this environment's IB session has been a
Demo account without live market data.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ..core.rate_limiter import RateLimitConfig, registry
from ..microstructure.gex import GexSnapshot, compute_naive_gex

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/gex")
_MARKET_SYMBOL = "QQQ"
_YFINANCE_LIMITER = "yfinance"


def _day_key(day) -> str:
    if isinstance(day, str):
        return day.replace("-", "")[:8]
    ts = pd.Timestamp(day)
    return f"{ts.year:04d}{ts.month:02d}{ts.day:02d}"


def snapshot_path(symbol: str, day, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / symbol.upper() / f"{_day_key(day)}.json"


def save_gex_snapshot(snap: GexSnapshot, cache_dir: Path = CACHE_DIR) -> Path:
    path = snapshot_path(snap.symbol, snap.as_of, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_gex_snapshot(symbol: str, day, cache_dir: Path = CACHE_DIR) -> GexSnapshot | None:
    path = snapshot_path(symbol, day, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return GexSnapshot.from_dict(raw)
    except Exception:
        log.warning("gex_cache: unreadable %s — treating as missing", path)
        return None


def load_gex_env(symbol: str, day, cache_dir: Path = CACHE_DIR) -> dict[str, GexSnapshot] | None:
    """Market (QQQ) + per-symbol snapshots for one session, or None if both
    are missing. Partial is OK — auction_reclaim uses whatever is present."""
    market = load_gex_snapshot(_MARKET_SYMBOL, day, cache_dir=cache_dir)
    own = load_gex_snapshot(symbol, day, cache_dir=cache_dir) if symbol.upper() != _MARKET_SYMBOL else market
    if market is None and own is None:
        return None
    return {"market": market, "symbol": own}


def _acquire_yfinance() -> None:
    limiter = registry.get(_YFINANCE_LIMITER, RateLimitConfig(requests_per_second=2.0))
    while not limiter.try_acquire():
        time.sleep(0.25)


def fetch_yfinance_gex(
    symbol: str,
    as_of: date | None = None,
    dte_max: int = 45,
) -> GexSnapshot | None:
    """Hit yfinance for the listed expiries within `dte_max` and compute a
    naive GEX snapshot. Network + rate-limit only; persist via
    save_gex_snapshot. Returns None on a missing chain / empty OI."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("gex_cache: yfinance not installed. Run: pip install yfinance") from exc

    as_of = as_of or date.today()
    _acquire_yfinance()
    ticker = yf.Ticker(symbol)
    spot = _yfinance_spot(ticker)
    if spot is None or spot <= 0:
        log.warning("gex_cache: no spot for %s", symbol)
        return None

    try:
        expiries = list(ticker.options or [])
    except Exception:
        log.exception("gex_cache: options calendar failed for %s", symbol)
        return None
    if not expiries:
        log.warning(
            "gex_cache: empty options calendar for %s (Yahoo often 401s the "
            "options crumb in this environment) — no snapshot written",
            symbol,
        )
        return None

    chains: list[dict] = []
    for exp in expiries:
        exp_date = _parse_iso_date(exp)
        if exp_date is None or (exp_date - as_of).days < 0 or (exp_date - as_of).days > dte_max:
            continue
        _acquire_yfinance()
        try:
            chain = ticker.option_chain(exp)
        except Exception:
            log.warning("gex_cache: option_chain(%s, %s) failed — skipping expiry", symbol, exp)
            continue
        chains.append({
            "expiry": exp_date,
            "calls": _rows_from_yf(chain.calls),
            "puts": _rows_from_yf(chain.puts),
        })

    return compute_naive_gex(symbol, spot, chains, as_of=as_of, source="yfinance", dte_max=dte_max)


def _yfinance_spot(ticker) -> float | None:
    try:
        fast = getattr(ticker, "fast_info", None)
        if fast is not None:
            for key in ("last_price", "lastPrice", "regular_market_price"):
                val = fast.get(key) if hasattr(fast, "get") else getattr(fast, key, None)
                if val is not None and float(val) > 0:
                    return float(val)
    except Exception:
        pass
    try:
        hist = ticker.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


def _rows_from_yf(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    rows = []
    for rec in frame.to_dict(orient="records"):
        rows.append({
            "strike": rec.get("strike"),
            "open_interest": rec.get("openInterest"),
            "implied_volatility": rec.get("impliedVolatility"),
        })
    return rows


def _parse_iso_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
