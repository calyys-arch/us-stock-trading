"""
yfinance historical bar loader for backtest/research.

Adjustment discipline (Chan Ch.3 blind spot #3, survivorship/adjustment
bias): yfinance's `auto_adjust=True` (the default since yfinance >= 0.2)
returns split- AND dividend-adjusted OHLC, which is exactly what
python/stat/cointegration.py and the cross-sectional return calculation
need — an un-adjusted 4:1 split would otherwise show up as a -75% "return"
that would corrupt both the cointegration hedge-ratio estimate and the
cross-sectional mean-reversion signal. This loader ALWAYS requests
`auto_adjust=True` explicitly (never relies on the caller's yfinance
version default) and runs the data-quality 4-sigma check (see
python/core/data_quality.py) on every symbol it loads, surfacing flagged
dates in the returned dict rather than silently proceeding.

Known limitation (documented, not hidden): yfinance's own long-history
adjusted data occasionally has small discrepancies vs a paid vendor
(CRSP/Polygon); see README.md "Known limitations (MVP)".
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from ..core.data_quality import quality_report
from ..core.rate_limiter import RateLimitConfig, registry

log = logging.getLogger(__name__)

_YFINANCE_RATE_LIMITER_NAME = "yfinance"


def _get_rate_limiter():
    return registry.get(_YFINANCE_RATE_LIMITER_NAME, RateLimitConfig(requests_per_second=2.0, daily_quota=None))


def fetch_daily_bars(
    symbol: str,
    start: str,
    end: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV bars for one symbol. Returns a DataFrame
    indexed by date with columns [open, high, low, close, volume].
    Raises RuntimeError if yfinance is not installed or the fetch fails
    after `max_retries` attempts."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("hist_data_us: yfinance not installed. Run: pip install yfinance") from exc

    limiter = _get_rate_limiter()
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        if not limiter.try_acquire():
            time.sleep(0.5)
            continue
        try:
            df = yf.download(
                symbol, start=start, end=end, auto_adjust=True,
                progress=False, threads=False,
            )
            if df.empty:
                raise RuntimeError(f"yfinance returned no data for {symbol} [{start}, {end}]")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as exc:
            last_exc = exc
            log.warning("hist_data_us: fetch attempt %d/%d failed for %s: %s", attempt + 1, max_retries, symbol, exc)
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"hist_data_us: failed to fetch {symbol} after {max_retries} attempts: {last_exc}")


def build_price_panel(
    symbols: list[str],
    start: str,
    end: str,
    sigma_threshold: float = 4.0,
) -> tuple[pd.DataFrame, dict]:
    """Fetch adjusted daily bars for many symbols and assemble into a single
    MultiIndex (date, code) DataFrame with columns [open, high, low, close,
    volume, adv_20d_dollars]. Returns (panel, quality_reports) where
    quality_reports = {symbol: quality_report_dict} for any symbol with >= 1
    flagged extreme move — callers should surface these in
    docs/us_equity_health_check.md rather than silently trusting the data.
    """
    frames = []
    quality_flags: dict = {}

    for symbol in symbols:
        try:
            df = fetch_daily_bars(symbol, start, end)
        except RuntimeError as exc:
            log.error("build_price_panel: skipping %s — %s", symbol, exc)
            continue

        report = quality_report(df["close"])
        if report["n_extreme_moves_flagged"] > 0 or report["n_zero_or_negative_prices"] > 0:
            quality_flags[symbol] = report

        df = df.copy()
        df["adv_20d_dollars"] = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean()
        df["code"] = symbol
        df.index.name = "date"
        frames.append(df.reset_index().set_index(["date", "code"]))

    if not frames:
        raise RuntimeError("build_price_panel: no symbols could be loaded")

    panel = pd.concat(frames).sort_index()
    return panel, quality_flags
