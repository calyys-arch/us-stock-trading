"""
FINRA OTC Transparency (ATS) weekly volume — free, public, no-key API
(https://developer.finra.org, Query API `otcMarket` dataset group). This is
"Tier 2" of the dark-pool-internalization diagnostic (user decision,
2026-07-29): a COARSE, weekly-lagged, symbol-level "how much of this
symbol's volume traded through ATSs (dark pools) this week" ratio — it
cannot be attributed to a specific day or signal (unlike the tick-level
`dark_pool_internalization_score` in python/signals/trap_detector.py, which
needs our OWN captured tick archive and is only available going forward).
Report-only, same as every other module under python/signals/.

Two distinct FINRA datasets, confirmed empirically 2026-07-29 (see
tests/test_finra_ats.py's fixtures for real captured response shapes):

  - `weeklySummary` ("recent"): filterable by `issueSymbolIdentifier`,
    returns one row PER (week, reporting ATS/MPID) — i.e. multiple rows per
    symbol-week, one per dark pool that reported volume in that name that
    week. Rows must be summed across MPID to get a symbol-week total. Not
    server-side sorted and paginates with overlapping windows in practice,
    so `fetch_all_recent_weeks` pages until no NEW week is discovered rather
    than trusting a fixed page count. Empirically covers roughly the last
    ~3-4 years, NOT the full 2018+ backtest window.
  - `weeklySummaryHistoric` (older weeks): the API rejects a symbol filter
    outright ("Invalid query field 'issueSymbolIdentifier'... only
    [weekStartDate, tierIdentifier, historicalMonth, historicalWeek]") — it
    can only be pulled ONE CALENDAR WEEK AT A TIME, for every reporting
    symbol market-wide, then filtered down to our universe client-side.
    This is why `fetch_historic_week` takes a single `week_start_date` and
    a `symbols` filter set, and why backfilling many years is a slow,
    resumable, opt-in operation (scripts/backfill_finra_ats.py) rather than
    something trap_report.py would ever call at report time.

Cache: data/finra_ats/<SYMBOL>.jsonl, one JSON object per (symbol, week),
deduped on `week_start_date`, sorted ascending. trap_report.py reads this
cache ONLY (no network at report time — same "reproducible once cached"
contract as every other evidence source in that module).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_BASE_URL = "https://api.finra.org/data/group/otcMarket/name"
RECENT_URL = f"{_BASE_URL}/weeklySummary"
HISTORIC_URL = f"{_BASE_URL}/weeklySummaryHistoric"

CACHE_DIR = Path("data/finra_ats")

_LIMITER_NAME = "finra_ats"
_REQUESTS_PER_SECOND = 2.0

_PAGE_LIMIT = 300
_MAX_PAGES_RECENT = 40  # 40 * 300 = 12,000 rows ceiling per symbol — generous
                        # given the ~3-4yr rolling window this dataset has in
                        # practice; stops earlier via the "no new week" check.


def _get_client(http_client=None):
    if http_client is not None:
        return http_client
    import httpx

    return httpx.Client(timeout=30.0, headers={"Accept": "application/json"})


def _get_limiter():
    from ..core.rate_limiter import RateLimitConfig, registry

    return registry.get(_LIMITER_NAME, RateLimitConfig(requests_per_second=_REQUESTS_PER_SECOND, daily_quota=None))


def _post(http_client, url: str, payload: dict, limiter=None) -> list[dict]:
    limiter = limiter or _get_limiter()
    while not limiter.try_acquire():
        time.sleep(0.1)
    resp = http_client.post(url, json=payload, headers={"Content-Type": "application/json"})
    if resp.status_code == 204:
        return []
    resp.raise_for_status()
    return resp.json() or []


def _symbol_cache_path(symbol: str, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{symbol.upper()}.jsonl"


def _aggregate_by_week(rows: list[dict]) -> dict[str, dict]:
    """Sum `totalWeeklyShareQuantity`/`totalWeeklyTradeCount` across every
    MPID row for the same `weekStartDate` — the raw API is per-(week, ATS),
    we want per-(week) totals."""
    out: dict[str, dict] = {}
    for row in rows:
        week = row.get("weekStartDate")
        if not week:
            continue
        agg = out.setdefault(week, {"week_start_date": week, "total_shares": 0, "total_trades": 0})
        agg["total_shares"] += int(row.get("totalWeeklyShareQuantity") or 0)
        agg["total_trades"] += int(row.get("totalWeeklyTradeCount") or 0)
    return out


def fetch_all_recent_weeks(symbol: str, http_client=None, limiter=None) -> list[dict]:
    """Every week `weeklySummary` currently has for `symbol` (empirically
    ~3-4 years, see module docstring), summed across reporting ATSs.
    Pages by offset until a page contributes NO week not already seen
    (the API doesn't sort results, so a fixed page count would silently
    miss data)."""
    client = _get_client(http_client)
    symbol = symbol.upper()
    all_rows: list[dict] = []
    seen_weeks: set[str] = set()
    offset = 0
    for _ in range(_MAX_PAGES_RECENT):
        payload = {
            "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": symbol},
            ],
            "limit": _PAGE_LIMIT,
            "offset": offset,
        }
        try:
            rows = _post(client, RECENT_URL, payload, limiter=limiter)
        except Exception as exc:
            log.warning("finra_ats: fetch_all_recent_weeks(%s) failed at offset %d (%s)", symbol, offset, exc)
            break
        if not rows:
            break
        page_weeks = {row.get("weekStartDate") for row in rows if row.get("weekStartDate")}
        # Accumulate raw rows (aggregation happens once at the end) so a
        # symbol-week whose MPID rows are split across pages is still
        # summed correctly, rather than overwritten page-by-page.
        all_rows.extend(rows)
        if page_weeks <= seen_weeks:
            # Every week on this page was already known — the rolling
            # window has been fully covered (further offsets just re-serve
            # the same weeks in a different MPID order).
            break
        seen_weeks |= page_weeks
        offset += _PAGE_LIMIT
    return sorted(_aggregate_by_week(all_rows).values(), key=lambda r: r["week_start_date"])


def fetch_historic_week(week_start_date: str, symbols: set[str], http_client=None, limiter=None) -> dict[str, dict]:
    """{symbol: {week_start_date, total_shares, total_trades}} for every
    symbol in `symbols` that reported ATS volume in the single calendar
    week `week_start_date` (YYYY-MM-DD, must be a Monday). Pulls the FULL
    market-wide page set for that week (the historic dataset cannot be
    filtered by symbol server-side — see module docstring) and filters
    client-side, so this is inherently slower than `fetch_all_recent_weeks`
    and meant for scripts/backfill_finra_ats.py, not report-time use."""
    client = _get_client(http_client)
    wanted = {s.upper() for s in symbols}
    rows_by_symbol: dict[str, list[dict]] = {}
    offset = 0
    while True:
        payload = {
            "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "weekStartDate", "fieldValue": week_start_date},
                {"compareType": "EQUAL", "fieldName": "tierIdentifier", "fieldValue": "NMS"},
            ],
            "limit": _PAGE_LIMIT,
            "offset": offset,
        }
        try:
            rows = _post(client, HISTORIC_URL, payload, limiter=limiter)
        except Exception as exc:
            log.warning("finra_ats: fetch_historic_week(%s) failed at offset %d (%s)", week_start_date, offset, exc)
            break
        if not rows:
            break
        for row in rows:
            sym = str(row.get("issueSymbolIdentifier", "")).upper()
            if sym in wanted:
                rows_by_symbol.setdefault(sym, []).append(row)
        if len(rows) < _PAGE_LIMIT:
            break
        offset += _PAGE_LIMIT

    return {sym: agg for sym, rows in rows_by_symbol.items() for agg in _aggregate_by_week(rows).values()}


def load_cached_weeks(symbol: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Cached weekly ATS rows for `symbol` as a DataFrame indexed by
    week_start_date (empty DataFrame, not None, when nothing is cached yet —
    callers use `.empty` rather than an is-None check)."""
    path = _symbol_cache_path(symbol, cache_dir)
    if not path.exists():
        return pd.DataFrame(columns=["week_start_date", "total_shares", "total_trades"]).set_index("week_start_date")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["week_start_date", "total_shares", "total_trades"]).set_index("week_start_date")
    df = pd.DataFrame(rows).drop_duplicates(subset="week_start_date", keep="last")
    df = df.sort_values("week_start_date").set_index("week_start_date")
    return df


def save_weeks(symbol: str, weeks: list[dict], cache_dir: Path = CACHE_DIR) -> Path:
    """Merge `weeks` into the symbol's cache (dedup by week_start_date, last
    write wins) and rewrite the file sorted ascending."""
    path = _symbol_cache_path(symbol, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cached_weeks(symbol, cache_dir).reset_index().to_dict("records")
    merged = {row["week_start_date"]: row for row in existing}
    for row in weeks:
        merged[row["week_start_date"]] = row
    ordered = sorted(merged.values(), key=lambda r: r["week_start_date"])
    with open(path, "w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row) + "\n")
    return path


def weekly_participation_ratio(symbol: str, week_start_date: str, week_total_volume: float, cache_dir: Path = CACHE_DIR) -> float | None:
    """This symbol-week's ATS share of `week_total_volume` (the caller
    supplies total volume — typically that week's sum of our own cached
    daily bars — since this dataset only reports the ATS-side numerator).
    None when that week isn't cached or `week_total_volume` is non-positive
    (evidence unavailable, never a 0)."""
    weeks = load_cached_weeks(symbol, cache_dir)
    if week_start_date not in weeks.index or week_total_volume <= 0:
        return None
    return float(weeks.loc[week_start_date, "total_shares"]) / float(week_total_volume)


def trailing_baseline_ratio(symbol: str, before_week: str, lookback_weeks: int = 12, cache_dir: Path = CACHE_DIR) -> float | None:
    """Median ATS *share count* (not a volume ratio — we don't have
    per-week total-market volume cached here) over the `lookback_weeks`
    cached weeks strictly before `before_week`, for `elevated_vs_baseline`'s
    "is this week's ATS activity unusually high FOR THIS SYMBOL" comparison.
    None when fewer than half of `lookback_weeks` are available."""
    weeks = load_cached_weeks(symbol, cache_dir)
    prior = weeks[weeks.index < before_week].tail(lookback_weeks)
    if len(prior) < max(2, lookback_weeks // 2):
        return None
    return float(prior["total_shares"].median())


def elevated_vs_baseline(symbol: str, week_start_date: str, lookback_weeks: int = 12, elevation_ratio: float = 1.5, cache_dir: Path = CACHE_DIR) -> bool | None:
    """True when `week_start_date`'s ATS share-count for `symbol` exceeds
    `elevation_ratio`x its own trailing median (symbol-relative, avoiding a
    single global magic-number baseline). None when either the week itself
    or a usable baseline isn't cached (evidence unavailable)."""
    weeks = load_cached_weeks(symbol, cache_dir)
    if week_start_date not in weeks.index:
        return None
    baseline = trailing_baseline_ratio(symbol, week_start_date, lookback_weeks, cache_dir)
    if baseline is None or baseline <= 0:
        return None
    return float(weeks.loc[week_start_date, "total_shares"]) >= elevation_ratio * baseline
