"""
Combined candidate universe: point-in-time (S&P 500 UNION Nasdaq-100),
narrowed to the top-K most liquid names by TRAILING average dollar volume.

Rationale (user-confirmed design decision, 2026-07-28): S&P 500 alone misses
many of the most actively-traded US equities that retail/momentum traders
are actually in (Nasdaq-100 covers most of that gap — see
nasdaq100_universe.py's module docstring). But index membership itself is
still a committee-driven, slow-moving list, not a measure of "what's hot
right now" — so on top of the combined index pool, this module ranks by
ACTUAL trailing traded dollar volume and keeps only the top `top_k`.

Look-ahead-bias discipline (Chan Ch.3 — this is the same class of bug this
whole codebase is built to avoid, see tests/test_lookahead_bias.py):
`top_by_trailing_dollar_volume` ranks using ONLY the `lookback_days` window
ending STRICTLY BEFORE `as_of`. This is NOT "pull today's most-active-stocks
list and assume it applied historically too" (which — silently making a
2026 hot-stock list look like it was tradeable in 2019 — is exactly the kind
of survivorship/look-ahead bug Chan warns about); it is a mechanical re-rank
that can be computed identically at every historical `as_of` from data that
was genuinely available on that date. `price_panel` must already be
restricted by the caller to rows available at `as_of` (same contract as
CrossSectionalMeanReversionStrategy.evaluate()).

`lookback_days` / `top_k` are DATA-WINDOW / UNIVERSE-CONSTRUCTION settings,
not a strategy's statistically-fit free parameter (same classification
rationale as `coint_lookback_days` in python/backtest/param_guard.py's
_NON_PARAMETER_KEYS) — they do not count against either strategy's 5-
parameter Chan Ch.3 budget.

`rank_by_trailing_dollar_volume` / `band_by_trailing_dollar_volume` (added
2026-08-13, `backtests/reports/alt_universe_frequency_exploration.md`)
generalize the same ranking to return the FULL ordering, or an arbitrary
[start, end) rank BAND rather than only a top-K cutoff — used to select a
deliberately less-liquid-than-top-K slice (e.g. "not mega-cap, not
micro-cap") of a point-in-time candidate pool, mechanically and without ever
looking at any strategy's returns. `top_by_trailing_dollar_volume` itself is
unchanged.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import nasdaq100_universe, sp500_universe

DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_TOP_K = 150


def combined_index_membership(as_of: datetime) -> set[str]:
    """Point-in-time union of S&P 500 and Nasdaq-100 membership as of
    `as_of`. Fetches Wikipedia tables fresh on every call — callers doing
    many dates should use combined_universe_by_day instead."""
    sp500 = sp500_universe.sp500_point_in_time_membership(as_of)
    nasdaq100 = nasdaq100_universe.nasdaq100_point_in_time_membership(as_of)
    return sp500 | nasdaq100


def combined_universe_by_day(dates: list[datetime]) -> dict[datetime, list[str]]:
    """Point-in-time union of S&P 500 and Nasdaq-100 membership for every
    date in `dates` (Wikipedia tables fetched once per index, not once per
    date)."""
    sp500_by_day = sp500_universe.universe_by_day(dates)
    nasdaq100_by_day = nasdaq100_universe.universe_by_day(dates)
    return {
        d: sorted(set(sp500_by_day.get(d, [])) | set(nasdaq100_by_day.get(d, [])))
        for d in dates
    }


def top_by_trailing_dollar_volume(
    candidates: list[str],
    price_panel: pd.DataFrame,
    as_of: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    """Rank `candidates` by mean(close * volume) over the `lookback_days`
    window ending strictly before `as_of`, return the top `top_k` codes.

    `price_panel` must be a (date, code)-MultiIndex DataFrame with `close`
    and `volume` columns, already restricted by the caller to rows available
    at `as_of` (rows on/after `as_of` are also explicitly excluded here as a
    defense-in-depth check, mirroring xsection_mean_reversion.py's own
    trim-before-use pattern).
    """
    all_dates = price_panel.index.get_level_values(0)
    window = price_panel.loc[all_dates < pd.Timestamp(as_of)]
    if window.empty:
        return sorted(candidates)[:top_k]

    unique_dates = window.index.get_level_values(0).unique().sort_values()
    recent_dates = unique_dates[-lookback_days:]
    recent = window.loc[window.index.get_level_values(0).isin(recent_dates)]

    dollar_volume = (recent["close"] * recent["volume"]).groupby(level=1).mean()
    eligible = dollar_volume[dollar_volume.index.isin(candidates)]
    ranked = eligible.sort_values(ascending=False)
    return ranked.head(top_k).index.tolist()


def rank_by_trailing_dollar_volume(
    candidates: list[str],
    price_panel: pd.DataFrame,
    as_of: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.Series:
    """Same ranking as `top_by_trailing_dollar_volume` but returns the FULL
    descending-sorted Series (code -> mean dollar volume) instead of only
    the head. Needed for a liquidity-BAND selection (e.g. ranks 150-220)
    rather than a top-K cutoff — same strictly-before-`as_of` window, same
    look-ahead discipline, just no truncation."""
    all_dates = price_panel.index.get_level_values(0)
    window = price_panel.loc[all_dates < pd.Timestamp(as_of)]
    if window.empty:
        return pd.Series(dtype=float)

    unique_dates = window.index.get_level_values(0).unique().sort_values()
    recent_dates = unique_dates[-lookback_days:]
    recent = window.loc[window.index.get_level_values(0).isin(recent_dates)]

    dollar_volume = (recent["close"] * recent["volume"]).groupby(level=1).mean()
    eligible = dollar_volume[dollar_volume.index.isin(candidates)]
    return eligible.sort_values(ascending=False)


def band_by_trailing_dollar_volume(
    candidates: list[str],
    price_panel: pd.DataFrame,
    as_of: datetime,
    band_start_rank: int,
    band_end_rank: int,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[str]:
    """Return the codes at ranks [`band_start_rank`, `band_end_rank`)
    (1-indexed, most-liquid-first) of the trailing-dollar-volume ordering —
    a LIQUIDITY BAND rather than a top-K cutoff. Used to select a
    deliberately less-liquid-than-top-K slice (e.g. "not mega-cap, not
    micro-cap") of a point-in-time candidate pool, mechanically and without
    looking at any strategy's returns."""
    ranked = rank_by_trailing_dollar_volume(candidates, price_panel, as_of, lookback_days)
    return ranked.iloc[max(band_start_rank - 1, 0):band_end_rank].index.tolist()


def liquid_universe_by_day(
    dates: list[datetime],
    price_panel: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> dict[datetime, list[str]]:
    """End-to-end: point-in-time (S&P 500 UNION Nasdaq-100) candidate pool
    per day, narrowed to the top `top_k` names by trailing dollar volume
    using only data available as of that day. This is the universe
    `scripts/run_backtest.py`'s real-data path feeds to Strategy B."""
    candidates_by_day = combined_universe_by_day(dates)
    return {
        d: top_by_trailing_dollar_volume(candidates_by_day[d], price_panel, d, lookback_days, top_k)
        for d in dates
    }
