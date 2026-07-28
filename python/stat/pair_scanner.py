"""
Pair scanner — finds and periodically re-validates cointegrated pairs.

Chan Ch.7 guidance followed here:
  - Only test pairs within the SAME sector/industry as economically sensible
    candidates (Chan's own worked examples use same-industry pairs like
    GLD/GDX, or ETF/constituent baskets) — this also keeps the number of
    pairwise tests, and therefore data-snooping/multiple-testing risk, low.
    Testing every possible pair in a 500-name universe would run ~125,000
    tests; at a 5% significance level that is expected to produce roughly
    6,000 FALSE positives by chance alone (Chan p.129 explicitly warns about
    this multiple-comparison trap).
  - Re-validate cointegration periodically (e.g. every `revalidate_days`) on
    a ROLLING lookback window, not just once at strategy inception —
    cointegration relationships decay/break as company fundamentals change
    (M&A, index removal, capital structure changes).
"""
from __future__ import annotations

import itertools
import logging
from datetime import datetime

import pandas as pd

from ..core.types import CointegrationResult
from .cointegration import test_pair

log = logging.getLogger(__name__)


def candidate_pairs_by_sector(sector_map: dict[str, str]) -> list[tuple[str, str]]:
    """sector_map: {code: sector_name}. Returns all same-sector pairs (a, b)
    with a < b (deduplicated, order-independent)."""
    by_sector: dict[str, list[str]] = {}
    for code, sector in sector_map.items():
        by_sector.setdefault(sector, []).append(code)

    pairs: list[tuple[str, str]] = []
    for _sector, codes in by_sector.items():
        codes = sorted(codes)
        pairs.extend(itertools.combinations(codes, 2))
    return pairs


def scan(
    candidate_pairs: list[tuple[str, str]],
    price_panel: pd.DataFrame,   # columns=codes, index=date, values=close (adjusted)
    lookback_days: int = 252,
    as_of: datetime | None = None,
    min_half_life_days: float = 1.0,
    max_half_life_days: float = 60.0,
) -> list[CointegrationResult]:
    """Test every candidate pair over the trailing `lookback_days` window
    ending strictly before `as_of` (or the last row of `price_panel` if
    `as_of` is None — callers doing backtests MUST pre-truncate `price_panel`
    themselves; this function does not know "today" in a backtest replay).

    Returns only pairs passing CointegrationResult.is_tradeable AND whose
    half-life falls within [min_half_life_days, max_half_life_days] — Chan
    notes an extremely short half-life (< 1 day) is often a statistical
    artifact / noise fit rather than genuine slow-moving cointegration, and
    an extremely long one is impractical to trade before the relationship
    likely breaks down.
    """
    window = price_panel.tail(lookback_days)
    results: list[CointegrationResult] = []

    for code_a, code_b in candidate_pairs:
        if code_a not in window.columns or code_b not in window.columns:
            continue
        pair_df = window[[code_a, code_b]].dropna()
        if len(pair_df) < 60:
            continue
        try:
            result = test_pair(
                code_a, code_b,
                pair_df[code_a], pair_df[code_b],
                computed_at=as_of or datetime.utcnow(),
            )
        except Exception:
            log.exception("pair_scanner: test_pair failed for (%s, %s)", code_a, code_b)
            continue

        if not result.is_tradeable:
            continue
        if not (min_half_life_days <= result.half_life_days <= max_half_life_days):
            continue
        results.append(result)

    results.sort(key=lambda r: r.cadf_tstat)  # most negative t-stat = strongest cointegration
    return results
