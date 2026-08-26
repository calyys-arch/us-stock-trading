"""The order sheet and the backtest must allocate identically.

The whole point of Strategy V1's bucketing is that US equity gets a decided
share rather than the share its ticker count implies. If `target_weights` ever
drifts back toward counting tickers, the sheet would quietly trade something
other than what risk_bucket_study.json measured, and no other test would
notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.portfolio.strategy_v1 import (
    BUCKETS, COLLAPSE, bucket_returns, bucket_weighted_gross, load_universe,
    target_weights,
)

# Deliberately lopsided: 4 US equity names against 1 of everything else, the
# same imbalance that gave the real 120-name list 69% US equity.
LOPSIDED = {
    "AAA": "us_single_name", "BBB": "us_sector_etf", "CCC": "us_broad_etf",
    "DDD": "us_single_name", "EEE": "intl_equity_etf", "FFF": "bond_etf",
    "GGG": "commodity_etf",
}


def test_bucket_weights_give_each_bucket_a_quarter_regardless_of_count():
    w = target_weights(sorted(LOPSIDED), LOPSIDED, "bucket")
    by_bucket: dict[str, float] = {}
    for symbol, weight in w.items():
        bucket = COLLAPSE[LOPSIDED[symbol]]
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + weight

    assert sum(w.values()) == pytest.approx(1.0)
    assert set(by_bucket) == set(BUCKETS)
    for bucket, share in by_bucket.items():
        assert share == pytest.approx(0.25), bucket
    # Four US names split one quarter; the lone bond ETF gets a whole quarter.
    assert w["AAA"] == pytest.approx(0.0625)
    assert w["FFF"] == pytest.approx(0.25)


def test_naive_weights_let_ticker_count_decide_which_is_the_bug_being_fixed():
    w = target_weights(sorted(LOPSIDED), LOPSIDED, "naive")
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1 / 7) for v in w.values())
    us = sum(v for s, v in w.items() if COLLAPSE[LOPSIDED[s]] == "us_equity")
    assert us == pytest.approx(4 / 7)


def test_frozen_universe_covers_every_bucket_and_maps_cleanly():
    symbols, bucket_of = load_universe()
    assert len(symbols) == len(set(symbols)) == len(bucket_of)
    assert all(bucket_of[s] in COLLAPSE for s in symbols)
    w = target_weights(symbols, bucket_of, "bucket")
    seen = {COLLAPSE[bucket_of[s]] for s in symbols}
    assert seen == set(BUCKETS)
    assert sum(w.values()) == pytest.approx(1.0)


def _panel(days: int = 400) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=days)
    cols = {}
    for symbol, bucket in LOPSIDED.items():
        # Bonds drift slowly, commodities thrash; the spread is what makes an
        # inverse-vol scheme pile into the calm bucket.
        vol = 0.002 if bucket == "bond_etf" else 0.02
        steps = rng.normal(0.0002, vol, days)
        cols[symbol] = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    return pd.DataFrame(cols), LOPSIDED


def test_equal_bucket_scheme_holds_its_weights_flat_and_pays_no_turnover():
    panel, bucket_of = _panel()
    net, weights = bucket_weighted_gross(panel, bucket_of, "equal", 60, 0.10, 4.0)
    assert not weights.empty
    assert (weights.nunique() == 1).all()
    assert weights.iloc[0].to_numpy() == pytest.approx(0.25)
    # Flat weights never trade, so the cost charge must be exactly zero.
    assert weights.diff().abs().sum(axis=1).fillna(0.0).sum() == pytest.approx(0.0)
    assert len(net) > 0


def test_inverse_vol_scheme_overweights_the_calm_bucket():
    panel, bucket_of = _panel()
    _, weights = bucket_weighted_gross(panel, bucket_of, "rp", 60, 0.10, 4.0)
    mean = weights.mean()
    assert mean["bonds"] > 0.5, "low-vol bucket should dominate under inverse vol"
    assert mean["bonds"] > mean["commodities"]
    assert mean.sum() == pytest.approx(1.0)


def test_an_unpriced_bucket_yields_nan_not_a_free_zero_return():
    """`sum(axis=1)` skips NaN, so an all-NaN row used to score as 0.0%.

    That is a real observation to every downstream statistic, and it understates
    exposure: the bucket contributes nothing while its weight stays at 25%.
    """
    panel, bucket_of = _panel()
    # Blank the whole bonds bucket for a stretch after the warmup.
    dark = panel.index[200:210]
    panel.loc[dark, "FFF"] = np.nan

    buckets = bucket_returns(panel, bucket_of)
    assert buckets.loc[dark, "bonds"].isna().all()

    net, _ = bucket_weighted_gross(panel, bucket_of, "equal", 60, 0.10, 4.0)
    assert not net.index.intersection(dark).size, \
        "days with a dark bucket must be dropped, not counted as zero"


def test_the_first_day_of_a_panel_is_dropped_rather_than_scored_as_zero():
    panel, bucket_of = _panel()
    buckets = bucket_returns(panel, bucket_of)
    # pct_change has nothing to difference against on row zero.
    assert buckets.iloc[0].isna().all()
    net, _ = bucket_weighted_gross(panel, bucket_of, "equal", 60, 0.10, 4.0)
    assert panel.index[0] not in net.index
    assert not (net == 0.0).any(), "no exactly-zero days should survive"


def test_bucket_allocation_never_peeks_at_the_day_it_sizes():
    """A spike must not change the weight used on the day it happens."""
    panel, bucket_of = _panel()
    spiked = panel.copy()
    day = 300
    spiked.iloc[day, spiked.columns.get_loc("GGG")] *= 1.6

    _, base = bucket_weighted_gross(panel, bucket_of, "rp", 60, 0.10, 4.0)
    _, after = bucket_weighted_gross(spiked, bucket_of, "rp", 60, 0.10, 4.0)
    stamp = panel.index[day]
    assert after.loc[stamp].to_numpy() == pytest.approx(base.loc[stamp].to_numpy())
