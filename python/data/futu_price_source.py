"""
Futu/Moomoo OpenD historical K-line source for backtest/research — 1-minute
bar backfill for symbols outside the fixed 20-symbol universe's IBKR-sourced
cache (python/data/ibkr_price_source.py + python/data/intraday_cache.py).

Why this exists (2026-08-15, absorption_breakout round 3 — see
backtests/reports/absorption_breakout_investigation_report.md's round-3
section for the full investigation): the macro/sector-beta-alignment filter
lever needs QQQ/SPY/XLK 1-minute history, but IB Gateway is unreachable in
this environment (round-2 finding: ConnectionRefusedError) and yfinance only
carries ~7 days of 1-minute data (also confirmed round-2, no fallback here).
The user's Futu OpenD gateway (already used for tick/L2 capture — see
python/interfaces/futu_tick_capture.py's own docstring for the account/
subscription background) turned out to ALSO expose historical K-line via
`OpenQuoteContext.request_history_kline`, and — verified empirically against
THIS environment's own live OpenD session before writing a single line of
backfill code (2026-08-15) — its quota model consumes exactly ONE point per
DISTINCT stock code ever queried (a 100-point budget on this account,
confirmed via `check_history_kline_quota`; a request pulling QQQ then AAPL
moved usage from 0 -> 2, listing both codes with their first-request
timestamp), NOT per request or per date range queried. A single request also
caps out at 1000 rows (confirmed: a QQQ 1-minute request spanning 2026-08-01
.. 2026-08-13 returned exactly 1000 rows plus a non-None `page_req_key`), but
paginating an ALREADY-unlocked code via that key costs no further quota. That
combination makes a full-history pull for a handful of new symbols both cheap
and safe on this account.

Bar-timestamp convention — READ BEFORE USING (verified empirically 2026-08-15
by diffing Futu- vs IBKR-sourced AAPL bars for 2026-07-31, a day already
cached under data/history_1m/AAPL/2026-07.parquet from IBKR): Futu labels
each 1-minute bar by its END time — the interval [09:30:00, 09:31:00) comes
back timestamped "09:31:00", and the LAST regular-session bar of the day is
timestamped "16:00:00". Every other 1-minute source already in this repo
(IBKR, via ibkr_price_source.py/intraday_cache.py) labels bars by their START
time instead (first bar "09:30:00", last bar "15:59:00"). This module shifts
every Futu timestamp back by exactly one minute before writing to
data/history_1m/, specifically to normalize onto that repo-wide START-time
convention — after the shift, OHLC values lined up with the IBKR-sourced
AAPL bars for the same wall-clock minute to within normal cross-vendor
tape-aggregation noise (e.g. Futu open 304.01 vs IBKR open 304.05 for the
same shifted minute — different consolidated-tape reconstructions, not a
bug). Do NOT skip or "correct" this shift: skipping it silently reintroduces
a 1-minute misalignment against every other symbol's cache, which would
corrupt any signal (like the macro-beta filter this module exists for) that
joins bars across symbols by timestamp.

Known data-quality caveat (documented, not "fixed"): the LAST bar of a
session (post-shift label 15:59:00) can show a different close/inflated
volume versus an IBKR-sourced bar for the same day, because Futu's feed
appears to fold the 16:00:00 closing-auction print into that final bar while
IBKR's TRADES bars do not (observed directly on AAPL 2026-07-31: open/high/
low matched almost exactly, close and volume did not — Futu close 308.91 /
volume 16.96M vs IBKR close 309.03 / volume 2.36M for the same nominal
15:59:00 bar). Immaterial to this module's actual use (a 1m/5m *momentum*
filter, not exact P&L replay), but real — never present a Futu-sourced final
bar of the day as an exact match to an IBKR-sourced one.

Cache layout: IDENTICAL to python/data/intraday_cache.py's own contract —
data/history_1m/<SYMBOL>/<YYYY-MM>.parquet + data/history_1m/<SYMBOL>/_meta.json
(this module reuses that module's own path/meta helpers directly, so a
reader never needs to know which broker originally supplied a given symbol's
cache). The meta sidecar additionally records `"source": "futu"` per month
(IBKR-sourced months never had that key — its absence means "ibkr", the
original implicit default) purely as a provenance breadcrumb; no reader
depends on it.

Failure mode: matches ibkr_price_source's fail-loud contract exactly —
FutuHistoricalUnavailable covers futu-api missing, OpenD unreachable/not
logged in, and quota/permission errors from request_history_kline itself.
Callers must stop and report, never silently substitute a different,
unvetted data source (same "no fallback for intraday history" policy
intraday_cache.py's own module docstring already documents for IBKR).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from .intraday_cache import (
    CACHE_DIR,
    _is_month_closed,
    _load_meta,
    _month_parquet_path,
    _save_meta,
    _symbol_dir,
)

log = logging.getLogger(__name__)

BROKER_CONFIG_PATH = Path("configs/broker.yaml")
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Futu's request_history_kline returns at most this many rows per call;
# larger ranges must be paginated via the returned page_req_key (see
# fetch_history_kline_range). Confirmed empirically 2026-08-15 (QQQ
# 1-minute, 2026-08-01..2026-08-13 returned exactly 1000 rows + a non-None
# page_req_key).
_MAX_ROWS_PER_REQUEST = 1000

# Futu labels each bar by its END time; every other 1-minute source in this
# repo (see module docstring) labels by START time. Subtracting this shift
# is what makes Futu-sourced bars line up with IBKR-sourced bars covering
# the same wall-clock interval.
BAR_LABEL_SHIFT = pd.Timedelta(minutes=1)


class FutuHistoricalUnavailable(RuntimeError):
    """OpenD unreachable / futu-api missing / quota exhausted / permission
    denied — callers should stop and report, never silently fall back to a
    different, unvetted data source."""


def load_connection_settings(config_path: str | Path = BROKER_CONFIG_PATH) -> dict:
    """Mirrors ibkr_price_source.load_connection_settings's shape, reading
    the `futu:` block instead of `ibkr:` (see configs/broker.yaml)."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    futu_cfg = cfg.get("futu", {}) or {}
    return {
        "host": futu_cfg.get("host", "127.0.0.1"),
        "port": int(futu_cfg.get("port", 11111)),
        "market_prefix": futu_cfg.get("market_prefix", "US"),
        "rsa_key_path": futu_cfg.get("rsa_key_path") or None,
    }


def open_futu_quote_context(
    config_path: str | Path = BROKER_CONFIG_PATH,
):
    """Open ONE OpenQuoteContext for a whole batch of historical-kline
    requests — same "one shared connection, not one per symbol" pattern as
    ibkr_price_source.open_ib_connection. Applies the RSA protocol-encryption
    settings from configs/broker.yaml's futu.rsa_key_path BEFORE connecting,
    exactly like python/interfaces/futu_tick_capture.py's `_run_session`.
    Raises FutuHistoricalUnavailable on any connection-level failure (missing
    package, OpenD unreachable, quote session not logged in) so callers fail
    loudly rather than hang."""
    try:
        from futu import OpenQuoteContext, SysConfig
    except ImportError as exc:
        raise FutuHistoricalUnavailable(
            "futu-api not installed — cannot fetch Futu historical bars"
        ) from exc

    settings = load_connection_settings(config_path)
    if settings["rsa_key_path"]:
        # Must match OpenD's "Encrypted Private Key" setting exactly — see
        # futu_tick_capture.py's "Protocol encryption" docstring note.
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file(settings["rsa_key_path"])

    try:
        ctx = OpenQuoteContext(host=settings["host"], port=settings["port"])
    except Exception as exc:
        raise FutuHistoricalUnavailable(
            f"could not connect to Futu OpenD at {settings['host']}:{settings['port']}: {exc}. "
            "Is OpenD running and logged in?"
        ) from exc

    ret, state = ctx.get_global_state()
    if ret != 0 or not (isinstance(state, dict) and state.get("qot_logined")):
        try:
            ctx.close()
        except Exception:
            pass
        raise FutuHistoricalUnavailable(f"Futu OpenD quote session not logged in: {state}")
    return ctx


def check_history_kline_quota(ctx) -> dict:
    """Returns {"used": int, "remaining": int, "detail": [...]}. ALWAYS call
    this with a small probe request before a real backfill (this repo's
    explicit "verify quota/subscription reality first" discipline for any
    new market-data source) — see module docstring's quota model."""
    ret, quota = ctx.get_history_kl_quota(get_detail=True)
    if ret != 0:
        raise FutuHistoricalUnavailable(f"get_history_kl_quota failed: {quota}")
    used, remaining, detail = quota
    return {"used": int(used), "remaining": int(remaining), "detail": list(detail or [])}


def _futu_code(symbol: str, market_prefix: str) -> str:
    return f"{market_prefix}.{symbol.upper()}"


def fetch_history_kline_range(
    ctx,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    market_prefix: str = "US",
    ktype: str | None = None,
) -> pd.DataFrame:
    """Fetch ALL 1-minute bars for `symbol` in [start, end] over an
    already-open `ctx` (see open_futu_quote_context), transparently
    paginating via page_req_key (only the FIRST request against a
    never-before-queried code costs quota — see module docstring). Returns a
    DataFrame indexed by `ts` (already shifted to the repo-wide START-time
    bar convention — see module docstring), columns
    [open, high, low, close, volume], sorted and de-duplicated by index.
    Returns an EMPTY DataFrame (not an exception) when Futu genuinely has no
    bars in range (e.g. before listing) — mirrors
    fetch_ibkr_intraday_month's "empty is a valid outcome" contract. Only
    raises FutuHistoricalUnavailable for request-level failures (bad code,
    quota/permission error)."""
    from futu import AuType, KLType, RET_OK

    ktype = ktype or KLType.K_1M
    code = _futu_code(symbol, market_prefix)
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    page_req_key = None
    while True:
        ret, data, next_page_req_key = ctx.request_history_kline(
            code, start=start_str, end=end_str, ktype=ktype, autype=AuType.NONE,
            max_count=_MAX_ROWS_PER_REQUEST, page_req_key=page_req_key,
        )
        if ret != RET_OK:
            raise FutuHistoricalUnavailable(f"request_history_kline failed for {code}: {data}")
        if data is not None and len(data) > 0:
            frames.append(data[["time_key", "open", "high", "low", "close", "volume"]].copy())
        if not next_page_req_key or next_page_req_key == page_req_key:
            break
        page_req_key = next_page_req_key

    if not frames:
        empty = pd.DataFrame(columns=_OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="ts"))
        return empty

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["time_key"]) - BAR_LABEL_SHIFT
    df = df.drop(columns=["time_key"]).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for col in _OHLCV_COLUMNS:
        df[col] = df[col].astype(float)
    return df[_OHLCV_COLUMNS]


def backfill_symbol_months(
    symbol: str,
    months: list[pd.Timestamp],
    ctx,
    cache_dir: str | Path = CACHE_DIR,
    force: bool = False,
    market_prefix: str = "US",
) -> dict:
    """Same summary-dict contract as intraday_cache.backfill_symbol_months
    ({"fetched": [...], "skipped": [...], "empty": [...], "failed": [...]}),
    Futu-sourced. Unlike the IBKR path (one request per calendar month — IB's
    own per-request cap), this fetches the WHOLE [months[0], months[-1]]
    range in a single paginated call (cheap on Futu's per-code quota model —
    see module docstring) and then SPLITS the result into the same per-month
    parquet files intraday_cache.py already uses, writing/updating the meta
    sidecar after each month so an interrupted run is still resumable
    exactly like the IBKR path."""
    cache_dir = Path(cache_dir)
    months = [pd.Timestamp(m).replace(day=1) for m in months]
    summary = {"fetched": [], "skipped": [], "empty": [], "failed": []}
    meta = _load_meta(cache_dir, symbol)

    pending: list[pd.Timestamp] = []
    for month_start in months:
        key = f"{month_start:%Y-%m}"
        already_cached = (
            not force
            and _month_parquet_path(cache_dir, symbol, month_start).exists()
            and key in meta
            and _is_month_closed(month_start)
        )
        if already_cached:
            summary["skipped"].append(key)
        else:
            pending.append(month_start)

    if not pending:
        return summary

    range_start = min(pending)
    range_end = max(pending) + pd.offsets.MonthEnd(1)
    try:
        full_df = fetch_history_kline_range(ctx, symbol, range_start, range_end, market_prefix=market_prefix)
    except FutuHistoricalUnavailable:
        summary["failed"].extend(f"{m:%Y-%m}" for m in pending)
        raise

    _symbol_dir(cache_dir, symbol).mkdir(parents=True, exist_ok=True)
    for month_start in pending:
        key = f"{month_start:%Y-%m}"
        month_end = month_start + pd.offsets.MonthEnd(1)
        month_df = full_df.loc[(full_df.index >= month_start) & (full_df.index <= month_end + pd.Timedelta(days=1))]

        if month_df.empty:
            summary["empty"].append(key)
        else:
            month_df[_OHLCV_COLUMNS].to_parquet(_month_parquet_path(cache_dir, symbol, month_start))
            summary["fetched"].append(key)

        meta[key] = {
            "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            "n_bars": int(len(month_df)),
            "closed": bool(_is_month_closed(month_start)),
            "source": "futu",
        }
        _save_meta(cache_dir, symbol, meta)
        log.info("futu_price_source: %s %s -> %d bars", symbol, key, len(month_df))

    return summary
