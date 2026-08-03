"""
IB Gateway historical daily-bar source for backtest/research.

Why this exists: scripts/run_backtest.py's research pipeline previously
fetched history exclusively from yfinance (python/simulation/hist_data_us.py).
IBKR's ADJUSTED_LAST daily bars carry the same OHLCV information but come
from the SAME data source the live execution path (ibkr_feed / ibkr_broker)
will see — using them for research removes one backtest-vs-live data-source
inconsistency (user-confirmed design decision, 2026-07-28). This module is
NOT the microstructure/tick capture layer (see
python/interfaces/ibkr_tick_capture.py for that); it only replaces the
"daily adjusted bars" role of hist_data_us.py.

Output contract: `fetch_ibkr_daily_bars` returns exactly the same
`(panel, quality_flags)` shape as `hist_data_us.build_price_panel` — a
MultiIndex (date, code) DataFrame with columns [open, high, low, close,
volume, adv_20d_dollars] plus a {symbol: quality_report} dict — so
downstream (price_cache / optimize / run_backtest) never needs to know
which source produced the data.

Pacing: IB's historical-data API enforces strict pacing rules (identical
requests within 15s are rejected; sustained bursts trigger multi-minute
lockouts around ~60 requests / 10 minutes). All requests here go through a
shared `ibkr_historical` token-bucket (python/core/rate_limiter.py) at a
deliberately conservative 0.1 req/s, and the whole batch reuses ONE
connection instead of reconnecting per symbol.

Failure mode: any condition that prevents fetching (ib_async missing, IB
Gateway not running, zero symbols loadable) raises
`IbkrHistoricalUnavailable`. Callers (python/data/price_cache.py) catch
this single exception type and fall back to yfinance with an explicit log —
never silently.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from ..core.data_quality import quality_report
from ..core.rate_limiter import RateLimitConfig, registry

log = logging.getLogger(__name__)

_IBKR_HIST_LIMITER_NAME = "ibkr_historical"
BROKER_CONFIG_PATH = Path("configs/broker.yaml")

# ~60 historical requests / 10 min is IB's documented soft ceiling; 0.1/s
# (one request every 10s, each symbol needs qualify + fetch) stays well under.
_REQUESTS_PER_SECOND = 0.1


class IbkrHistoricalUnavailable(RuntimeError):
    """IB Gateway unreachable / ib_async missing / nothing fetched — the
    caller should fall back to another source (and say so in its report)."""


def _get_rate_limiter():
    return registry.get(
        _IBKR_HIST_LIMITER_NAME,
        RateLimitConfig(requests_per_second=_REQUESTS_PER_SECOND, daily_quota=None),
    )


def load_connection_settings(config_path: str | Path = BROKER_CONFIG_PATH) -> dict:
    """Read host/port/client_id for historical requests from
    configs/broker.yaml. `historical_client_id` must differ from the
    feed/broker/tick-capture client ids — IB kicks the older session when
    two connections share a clientId."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ibkr = cfg.get("ibkr", {}) or {}
    return {
        "host": ibkr.get("host", "127.0.0.1"),
        "port": int(ibkr.get("feed_port", 4002)),
        "client_id": int(ibkr.get("historical_client_id", 31)),
    }


def _duration_str(start: pd.Timestamp, anchor: pd.Timestamp) -> str:
    """IB durationStr covering [start, anchor] with a small buffer. IB only
    accepts integer 'N D' / 'N W' / 'N M' / 'N Y' units; for anything
    beyond ~1 year, whole years are the reliable choice."""
    days = max((anchor - start).days, 1)
    if days < 360:
        return f"{days + 5} D"
    years = days // 365 + 1
    return f"{years} Y"


def fetch_ibkr_daily_bars(
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config_path: str | Path = BROKER_CONFIG_PATH,
    connect_timeout: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Batch-fetch ADJUSTED_LAST daily bars for `symbols` over [start, end]
    via ONE IB Gateway connection. Returns (panel, quality_flags) with the
    exact `hist_data_us.build_price_panel` output contract.

    Raises IbkrHistoricalUnavailable when IB cannot be used at all (caller
    falls back to yfinance); individual symbol failures are logged and
    skipped, mirroring build_price_panel's behavior."""
    try:
        from ib_async import IB, Stock  # type: ignore[import]
    except ImportError as exc:
        raise IbkrHistoricalUnavailable(
            "ib_async not installed — cannot fetch IBKR historical bars"
        ) from exc

    settings = load_connection_settings(config_path)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # IB rejects any non-blank endDateTime when whatToShow=ADJUSTED_LAST
    # ("End date not supported with adjusted last") — the request must
    # always anchor to "now". Duration is therefore sized off `now`, not
    # `end_ts`, so a call for an end date far in the past still requests
    # enough bars to cover it; the [start_ts, end_ts] filter below then
    # trims the extra bars fetched past the caller's requested end.
    now_ts = pd.Timestamp.now()
    duration = _duration_str(start_ts, max(end_ts, now_ts))
    limiter = _get_rate_limiter()

    ib = IB()
    try:
        ib.connect(
            settings["host"], settings["port"],
            clientId=settings["client_id"], timeout=connect_timeout,
        )
    except Exception as exc:
        raise IbkrHistoricalUnavailable(
            f"could not connect to IB Gateway at {settings['host']}:{settings['port']} "
            f"(clientId={settings['client_id']}): {exc}. Is IB Gateway/TWS running with API enabled?"
        ) from exc

    frames: list[pd.DataFrame] = []
    quality_flags: dict = {}
    try:
        for symbol in symbols:
            while not limiter.try_acquire():
                time.sleep(1.0)
            try:
                contract = Stock(symbol, "SMART", "USD")
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    log.warning("ibkr_price_source: could not qualify %s — skipping", symbol)
                    continue
                bars = ib.reqHistoricalData(
                    qualified[0],
                    endDateTime="",  # must be blank for whatToShow=ADJUSTED_LAST (IB API restriction)
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                    formatDate=1,
                )
            except Exception as exc:
                log.error("ibkr_price_source: fetch failed for %s — %s", symbol, exc)
                continue

            if not bars:
                log.warning("ibkr_price_source: no bars returned for %s — skipping", symbol)
                continue

            df = pd.DataFrame(
                {
                    "date": [pd.Timestamp(b.date) for b in bars],
                    "open": [float(b.open) for b in bars],
                    "high": [float(b.high) for b in bars],
                    "low": [float(b.low) for b in bars],
                    "close": [float(b.close) for b in bars],
                    "volume": [float(b.volume) for b in bars],
                }
            ).set_index("date").sort_index()
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            if df.empty:
                log.warning("ibkr_price_source: %s has no bars inside [%s, %s] — skipping",
                            symbol, start_ts.date(), end_ts.date())
                continue

            report = quality_report(df["close"])
            if report["n_extreme_moves_flagged"] > 0 or report["n_zero_or_negative_prices"] > 0:
                quality_flags[symbol] = report

            df["adv_20d_dollars"] = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean()
            df["code"] = symbol
            df.index.name = "date"
            frames.append(df.reset_index().set_index(["date", "code"]))
            log.info("ibkr_price_source: %s -> %d bars", symbol, len(df))
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass

    if not frames:
        raise IbkrHistoricalUnavailable("no symbols could be loaded via IBKR historical data")

    return pd.concat(frames).sort_index(), quality_flags


# ─────────────────────────────────────────────────────────────────────────────
# 1-minute TRADES bars (microstructure pivot — python/data/intraday_cache.py)
# ─────────────────────────────────────────────────────────────────────────────
#
# Deliberately separate from the daily ADJUSTED_LAST path above:
#   - whatToShow="TRADES" (not ADJUSTED_LAST): intraday bars are NOT
#     split/dividend-adjusted by IB. Overnight gaps across ex-div dates are
#     real price discontinuities in this data — callers must not treat them
#     as bad prints. Documented risk in docs/microstructure_pivot_plan.md #10;
#     acceptable because this system day-trades intraday signals and never
#     holds positions overnight.
#   - TRADES bars DO accept a non-blank endDateTime (unlike ADJUSTED_LAST),
#     which is exactly what makes month-by-month chunked backfill possible:
#     each request anchors to one calendar month's last instant.
#   - IB caps how much 1-minute history a SINGLE request may return to
#     roughly one calendar month (docs/microstructure_pivot_plan.md §3a) —
#     there is no "just ask for 2 years" option the way ADJUSTED_LAST allows.
#   - A separate rate-limiter bucket (own budget from the daily-bars one)
#     because intraday backfill runs as its own long-lived batch script
#     (scripts/backfill_intraday.py), typically NOT overlapping in time with
#     daily-bar research fetches — but kept equally conservative in case it
#     ever does.

_IBKR_INTRADAY_LIMITER_NAME = "ibkr_intraday_historical"
_INTRADAY_REQUESTS_PER_SECOND = 0.1  # same conservative ~60 req/10min IB budget


def _get_intraday_rate_limiter():
    return registry.get(
        _IBKR_INTRADAY_LIMITER_NAME,
        RateLimitConfig(requests_per_second=_INTRADAY_REQUESTS_PER_SECOND, daily_quota=None),
    )


def open_ib_connection(
    config_path: str | Path = BROKER_CONFIG_PATH,
    connect_timeout: float = 10.0,
    client_id_override: int | None = None,
):
    """Open ONE IB Gateway connection for a whole batch of historical
    requests (callers loop many (symbol, month) jobs over it instead of
    reconnecting per request — reconnecting per request would itself count
    against IB's pacing budget). Raises IbkrHistoricalUnavailable on any
    connection-level failure so callers can fail loudly rather than hang."""
    try:
        from ib_async import IB  # type: ignore[import]
    except ImportError as exc:
        raise IbkrHistoricalUnavailable(
            "ib_async not installed — cannot fetch IBKR historical bars"
        ) from exc

    settings = load_connection_settings(config_path)
    client_id = client_id_override if client_id_override is not None else settings["client_id"]
    ib = IB()
    try:
        ib.connect(settings["host"], settings["port"], clientId=client_id, timeout=connect_timeout)
    except Exception as exc:
        raise IbkrHistoricalUnavailable(
            f"could not connect to IB Gateway at {settings['host']}:{settings['port']} "
            f"(clientId={client_id}): {exc}. Is IB Gateway/TWS running with API enabled?"
        ) from exc
    return ib


def fetch_ibkr_intraday_month(
    ib,
    symbol: str,
    month_start: pd.Timestamp,
    bar_size: str = "1 min",
    use_rth: bool = True,
) -> pd.DataFrame:
    """Fetch ONE calendar month of `bar_size` TRADES bars for `symbol` over
    an already-open `ib` connection (see open_ib_connection). `month_start`
    must be the first day of the target month; the request anchors its
    endDateTime to the last instant of that month.

    Returns an EMPTY DataFrame (not an exception) when IB returns zero bars
    for the month — a valid outcome for a month before the symbol's IPO or a
    month with a data gap — so the cache can still record "checked, no data"
    and not retry it forever. Only raises IbkrHistoricalUnavailable for
    request-level failures (contract could not be qualified, request
    errored) since those ARE worth surfacing/retrying."""
    from ib_async import Stock  # type: ignore[import]

    month_start = pd.Timestamp(month_start).replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)
    end_dt_str = (month_end + pd.Timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y%m%d %H:%M:%S")

    limiter = _get_intraday_rate_limiter()
    while not limiter.try_acquire():
        time.sleep(1.0)

    try:
        contract = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise IbkrHistoricalUnavailable(f"could not qualify contract for {symbol}")
        bars = ib.reqHistoricalData(
            qualified[0],
            endDateTime=end_dt_str,
            durationStr="1 M",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=1,
        )
    except IbkrHistoricalUnavailable:
        raise
    except Exception as exc:
        raise IbkrHistoricalUnavailable(f"reqHistoricalData failed for {symbol} {month_start.date()}: {exc}") from exc

    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # ib_async parses intraday bar.date into a tz-aware Timestamp (US/Eastern,
    # per the "date-time attributes without explicit time zone" warning IB
    # emits — it defaults to the exchange's timezone) even though daily bars
    # come back as plain dates. Every consumer downstream (intraday_cache's
    # own month-boundary trim below, python/microstructure/context.py, the
    # backtester) works in tz-NAIVE US/Eastern wall-clock time, so normalize
    # here once rather than leaking a tz-aware index into the parquet cache.
    ts_index = pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars])
    if ts_index.tz is not None:
        ts_index = ts_index.tz_localize(None)
    df = pd.DataFrame(
        {
            "ts": ts_index,
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(b.volume) for b in bars],
        }
    ).set_index("ts").sort_index()
    # Trim to the target month only — IB occasionally returns a few bars
    # just outside the requested window near month boundaries.
    df = df.loc[(df.index >= month_start) & (df.index <= month_end + pd.Timedelta(days=1))]
    return df
