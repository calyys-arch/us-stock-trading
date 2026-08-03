"""
Shared Wikipedia table-fetch helper for the point-in-time index-membership
builders (sp500_universe.py, nasdaq100_universe.py).

Wikipedia returns HTTP 403 to requests with no `User-Agent` header —
`pandas.read_html(url)`'s default urllib opener sends none — so this fetches
the page manually with a browser-like UA and hands the raw HTML to
`pd.read_html`. The HTML must be wrapped in a file-like object (`io.StringIO`)
rather than passed as `str`/`bytes` directly: `pd.read_html` otherwise
interprets a raw string/bytes payload as a filename or URL to fetch, not as
literal markup to parse, and raises a confusing `FileNotFoundError`.
"""
from __future__ import annotations

import io
from urllib.request import Request, urlopen

import pandas as pd

_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; us-stock-trading-research-bot/1.0)"}


def fetch_wiki_tables(url: str, timeout: float = 30.0) -> list[pd.DataFrame]:
    req = Request(url, headers=_REQUEST_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        html = resp.read()
    return pd.read_html(io.StringIO(html.decode("utf-8")))
