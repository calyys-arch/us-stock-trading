"""
SEC EDGAR 8-K filings client — the FULL-HISTORY material-event source for
the report-only signal-trap diagnostic layer.

Why EDGAR on top of Finnhub (user decision, 2026-07-28): Finnhub's free
tier only returns ~12 months of company news, so a 2018-2025 backtest has
no news evidence for most of its span. 8-K filings are the LEGALLY REQUIRED
disclosure of material corporate events (Item 2.02 results, 1.01 material
agreements, 5.02 officer departures, 7.01/8.01 Reg-FD/other events...),
they carry precise acceptance timestamps and item codes, EDGAR's history is
complete back far beyond 2018, and the API is free — it just requires a
declared User-Agent and polite request rates.

Data flow:
  - ticker -> CIK via https://www.sec.gov/files/company_tickers.json
    (cached to data/filings/company_tickers.json).
  - Historical backfill via https://data.sec.gov/submissions/CIK##########.json
    ("recent" block + paginated older files) filtered to form 8-K / 8-K/A.
  - Incremental refresh via the browse-edgar Atom feed (type=8-K), merged
    into the same per-symbol cache (data/filings/8k/<SYMBOL>.json).

Compliance: SEC fair-access policy wants a User-Agent identifying the
requester (set SEC_EDGAR_USER_AGENT in .env; a generic default is used
otherwise, with a one-time nag) and <= 10 req/s. We run a shared token
bucket at 5 req/s.

Failure mode matches finnhub_client.py: network/parse errors log a warning
and fall back to whatever the cache holds; three-valued evidence helpers
return None (UNKNOWN) when there is no cache at all — the trap detector
must never read "we couldn't check" as "no event happened".
"""
from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{filename}"
RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

FILINGS_CACHE_DIR = Path("data/filings")
_DEFAULT_USER_AGENT = "us-stock-trading-research (contact: set SEC_EDGAR_USER_AGENT in .env)"

_LIMITER_NAME = "sec_edgar"
_REQUESTS_PER_SECOND = 5.0

_8K_FORMS = {"8-K", "8-K/A"}


class EdgarClient:
    def __init__(
        self,
        http_client=None,
        cache_dir: str | Path = FILINGS_CACHE_DIR,
        user_agent: str | None = None,
    ) -> None:
        if user_agent is None:
            import os

            from dotenv import load_dotenv

            load_dotenv()
            user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
            if not user_agent:
                log.warning("EdgarClient: SEC_EDGAR_USER_AGENT not set — using a generic "
                            "User-Agent. SEC fair-access policy asks for a contact address; "
                            "set it in .env (see .env.example).")
                user_agent = _DEFAULT_USER_AGENT
        self._user_agent = user_agent

        if http_client is None:
            import httpx

            http_client = httpx.Client(timeout=30.0, headers={"User-Agent": user_agent},
                                       follow_redirects=True)
        self._client = http_client

        from ..core.rate_limiter import RateLimitConfig, registry

        self._limiter = registry.get(
            _LIMITER_NAME, RateLimitConfig(requests_per_second=_REQUESTS_PER_SECOND, daily_quota=None)
        )
        self._cache_dir = Path(cache_dir)
        self._ticker_map: dict[str, str] | None = None

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict | None = None):
        while not self._limiter.try_acquire():
            time.sleep(0.1)
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp

    def _symbol_cache_path(self, symbol: str) -> Path:
        return self._cache_dir / "8k" / f"{symbol.upper()}.json"

    @staticmethod
    def _read_json(path: Path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("EdgarClient: unreadable cache %s", path)
            return None

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    # ── ticker -> CIK ────────────────────────────────────────────────────────

    def ticker_to_cik(self, symbol: str, refresh: bool = False) -> str | None:
        """10-digit zero-padded CIK for `symbol`, or None if unknown.
        The full SEC ticker map is one request, cached to disk."""
        symbol = symbol.upper().replace(".", "-")
        if self._ticker_map is None or refresh:
            cache_path = self._cache_dir / "company_tickers.json"
            raw = None if refresh else self._read_json(cache_path)
            if raw is None:
                try:
                    raw = self._get(TICKER_MAP_URL).json()
                    self._write_json(cache_path, raw)
                except Exception as exc:
                    log.warning("EdgarClient: ticker map fetch failed (%s)", exc)
                    raw = self._read_json(cache_path)
            if raw is None:
                return None
            self._ticker_map = {
                str(row["ticker"]).upper(): f"{int(row['cik_str']):010d}"
                for row in raw.values()
            }
        return self._ticker_map.get(symbol)

    # ── submissions backfill ─────────────────────────────────────────────────

    @staticmethod
    def _filings_from_columnar(block: dict) -> list[dict]:
        """SEC submissions JSON stores filings as parallel arrays — zip the
        columns we need and keep only 8-K forms."""
        forms = block.get("form", [])
        out = []
        for i, form in enumerate(forms):
            if form not in _8K_FORMS:
                continue
            def col(name: str):
                values = block.get(name, [])
                return values[i] if i < len(values) else ""
            out.append({
                "accession": col("accessionNumber"),
                "form": form,
                "filing_date": col("filingDate"),
                "acceptance_datetime": col("acceptanceDateTime"),
                "items": col("items"),
            })
        return out

    def backfill_8k(self, symbol: str, since: date | None = None) -> list[dict]:
        """Full 8-K history for `symbol` via the submissions API (recent
        block + older paginated files, stopping once a page is entirely
        before `since`). Writes the per-symbol cache and returns the list."""
        symbol = symbol.upper()
        cik = self.ticker_to_cik(symbol)
        if cik is None:
            log.warning("EdgarClient: no CIK for %s — skipping", symbol)
            return []

        try:
            doc = self._get(SUBMISSIONS_URL.format(filename=f"CIK{cik}.json")).json()
        except Exception as exc:
            log.warning("EdgarClient: submissions fetch failed for %s (%s) — using cache", symbol, exc)
            cached = self._read_json(self._symbol_cache_path(symbol))
            return (cached or {}).get("filings", [])

        filings = self._filings_from_columnar(doc.get("filings", {}).get("recent", {}))
        for extra in doc.get("filings", {}).get("files", []):
            if since is not None and extra.get("filingTo", "9999") < since.isoformat():
                continue
            try:
                page = self._get(SUBMISSIONS_URL.format(filename=extra["name"])).json()
                filings.extend(self._filings_from_columnar(page))
            except Exception as exc:
                log.warning("EdgarClient: extra submissions page %s failed (%s)", extra.get("name"), exc)

        if since is not None:
            filings = [f for f in filings if f["filing_date"] >= since.isoformat()]
        filings.sort(key=lambda f: f["filing_date"])

        self._write_json(self._symbol_cache_path(symbol), {
            "symbol": symbol,
            "cik": cik,
            "filings": filings,
            "backfilled_since": since.isoformat() if since else "",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log.info("EdgarClient: %s — %d 8-K filings cached", symbol, len(filings))
        return filings

    # ── RSS incremental refresh ──────────────────────────────────────────────

    def fetch_8k_rss(self, symbol: str, count: int = 40) -> list[dict]:
        """Latest 8-K entries from the browse-edgar Atom feed (near-real-time,
        unlike the submissions JSON which can lag slightly). Returns the same
        row shape as backfill_8k minus acceptance_datetime/items granularity."""
        cik = self.ticker_to_cik(symbol)
        if cik is None:
            return []
        try:
            resp = self._get(RSS_URL, params={
                "action": "getcompany", "CIK": cik, "type": "8-K",
                "dateb": "", "owner": "include", "count": count, "output": "atom",
            })
            root = ET.fromstring(resp.content)
        except Exception as exc:
            log.warning("EdgarClient: 8-K RSS failed for %s (%s)", symbol, exc)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        out = []
        for entry in root.findall("atom:entry", ns):
            updated = entry.findtext("atom:updated", "", ns)
            entry_id = entry.findtext("atom:id", "", ns)
            accession = entry_id.rsplit("accession-number=", 1)[-1] if "accession-number=" in entry_id else entry_id
            category = entry.find("atom:category", ns)
            form = category.get("term", "8-K") if category is not None else "8-K"
            if form not in _8K_FORMS:
                continue
            out.append({
                "accession": accession,
                "form": form,
                "filing_date": updated[:10],
                "acceptance_datetime": updated,
                "items": "",
            })
        return out

    def refresh_8k(self, symbol: str) -> list[dict]:
        """Merge the latest RSS entries into the cached backfill (dedup on
        accession). Cheap incremental update for symbols already backfilled;
        falls back to a full backfill when no cache exists."""
        symbol = symbol.upper()
        cached = self._read_json(self._symbol_cache_path(symbol))
        if not cached:
            return self.backfill_8k(symbol)

        filings = {f["accession"]: f for f in cached.get("filings", [])}
        for row in self.fetch_8k_rss(symbol):
            filings.setdefault(row["accession"], row)
        merged = sorted(filings.values(), key=lambda f: f["filing_date"])
        cached["filings"] = merged
        cached["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_json(self._symbol_cache_path(symbol), cached)
        return merged

    # ── evidence lookups (trap detector / report annotator) ─────────────────

    def get_cached_8k(self, symbol: str) -> list[dict] | None:
        """Cached filings, or None when the symbol was never backfilled
        (None = UNKNOWN, not 'no filings')."""
        cached = self._read_json(self._symbol_cache_path(symbol.upper()))
        return None if cached is None else cached.get("filings", [])

    def has_8k_near(self, symbol: str, day: date, window_days: int = 1) -> bool | None:
        """True/False: an 8-K was filed within +/- `window_days` of `day`.
        None: no cache for the symbol (evidence unavailable)."""
        filings = self.get_cached_8k(symbol)
        if filings is None:
            return None
        import pandas as pd

        target = pd.Timestamp(day)
        for f in filings:
            if not f.get("filing_date"):
                continue
            if abs((pd.Timestamp(f["filing_date"]) - target).days) <= window_days:
                return True
        return False

    def eight_k_dates_by_symbol(self, symbols: list[str]) -> dict[str, set[str]]:
        """{symbol: {filing_date, ...}} from cache only (no network) — the
        trap-report annotator's lookup shape. Symbols without a cache are
        absent from the result (UNKNOWN)."""
        out: dict[str, set[str]] = {}
        for symbol in symbols:
            filings = self.get_cached_8k(symbol)
            if filings is None:
                continue
            out[symbol.upper()] = {f["filing_date"] for f in filings if f.get("filing_date")}
        return out
