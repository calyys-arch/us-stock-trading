"""The share-level simulator now carries every executable claim about V1.

It replaced a weight-space series plus a commission estimate, an arrangement
that charged for one trading schedule while crediting the returns of another.
Since it is the thing that says what the strategy actually returns net of what
a broker charges, the invariants it must not break are pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.portfolio.share_level import (
    invested_fraction, returns_of, simulate,
)
from scripts.run_retail_cost_study import quotas

BUCKETS = {"AAA": "us_broad_etf", "BBB": "us_broad_etf",
           "CCC": "bond_etf", "DDD": "bond_etf"}


def _panel(days: int = 90, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame({s: np.full(days, price) for s in BUCKETS}, index=idx)


def _flat_exposure(panel: pd.DataFrame, level: float = 1.0) -> pd.Series:
    return pd.Series(level, index=panel.index)


def test_the_opening_cash_is_in_the_return_series():
    """Entry costs land on day one, and a curve that starts after entry loses
    them: pct_change drops its own first row. The opening row exists so the one
    trade guaranteed to happen is inside the Sharpe."""
    panel = _panel()
    daily, log = simulate(panel, BUCKETS, _flat_exposure(panel), 100_000.0)
    assert daily["equity"].iloc[0] == pytest.approx(100_000.0)
    assert daily.index[0] < panel.index[0], "opening row precedes first trade"
    entry = log.iloc[0]
    assert entry["reason"] == "enter"
    assert returns_of(daily).iloc[0] < 0, "entry cost must show as a loss"
    assert returns_of(daily).iloc[0] == pytest.approx(
        -entry["cost"] / 100_000.0, rel=0.05)


def test_a_full_exposure_book_never_borrows_to_pay_its_costs():
    """Exposure 1.0 means fully invested, not 1.0 plus whatever the fees were.
    Cash going negative would quietly lever the book."""
    panel = _panel()
    daily, log = simulate(panel, BUCKETS, _flat_exposure(panel, 1.0), 10_000.0)
    assert (log["cash"] >= -1e-6).all(), "cash went negative"
    assert (daily["invested"] <= daily["equity"] + 1e-6).all()


def test_the_band_never_blocks_an_exposure_change():
    """The regression that cost 2.8 points of CAGR.

    Cutting exposure produces small orders -- a tenth off each position -- so a
    notional floor blocks precisely the trades that defend against a drawdown.
    A band that applies to exposure moves silently disables the brake.
    """
    panel = _panel()
    exp = _flat_exposure(panel, 1.0).copy()
    exp.iloc[45:] = 0.5  # a brake event mid-window
    # A ceiling this tight rejects any order below $10,000 of notional.
    daily, log = simulate(panel, BUCKETS, exp, 100_000.0,
                          cost_ceiling_bps=1.0)
    brake = log[log["reason"] == "exposure"]
    assert not brake.empty, "the exposure move must produce a rebalance"
    assert brake.iloc[0]["n_orders"] == len(BUCKETS), \
        "every position must be trimmed despite the band"
    after = daily.loc[daily.index > brake.iloc[0]["date"]]
    held = (after["invested"] / after["equity"]).iloc[0]
    assert held == pytest.approx(0.5, abs=0.02), "exposure must actually fall"


def test_the_band_does_suppress_pure_drift_correction():
    """The other half of the same rule: with exposure pinned, a tight band
    should stop the monthly reset from placing tiny orders."""
    panel = _panel()
    exp = _flat_exposure(panel, 1.0)
    loose, _ = simulate(panel, BUCKETS, exp, 100_000.0)
    tight, tight_log = simulate(panel, BUCKETS, exp, 100_000.0,
                                cost_ceiling_bps=1.0)
    monthly = tight_log[tight_log["reason"] == "monthly"]
    assert (monthly["n_orders"] == 0).all(), "band should reject small resets"
    assert tight_log["cost"].sum() < loose["equity"].iloc[0] * 1.0


def test_integer_shares_leave_cash_idle_and_fractional_do_not():
    """The dominant retail drag, larger than commission at $100k."""
    panel = _panel(price=317.0)  # awkward price, so rounding bites
    exp = _flat_exposure(panel, 1.0)
    whole, _ = simulate(panel, BUCKETS, exp, 20_000.0, fractional=False)
    part, _ = simulate(panel, BUCKETS, exp, 20_000.0, fractional=True)
    assert invested_fraction(whole) < invested_fraction(part)
    assert invested_fraction(part) > 0.98


def test_commission_can_be_switched_off_for_attribution():
    panel = _panel()
    exp = _flat_exposure(panel)
    _, paid = simulate(panel, BUCKETS, exp, 100_000.0)
    _, free = simulate(panel, BUCKETS, exp, 100_000.0, per_share=0.0,
                       min_per_order=0.0, spread_bps_one_way=0.0)
    assert paid["cost"].sum() > 0
    assert free["cost"].sum() == pytest.approx(0.0)


def test_no_band_however_tight_can_stop_the_book_being_built():
    """Entry is exempt from the band for the same reason exposure changes are.

    A notional floor applied to the entry means a small account simply never
    invests, and the flat near-zero return that follows reads as a strategy
    result rather than a refusal to trade. Exempting entry removes that failure
    mode entirely, so the study no longer has to detect it.
    """
    panel = _panel()
    # $1,000 across four names is $250 a position, and this ceiling would
    # reject any order under roughly $1,000,000.
    daily, log = simulate(panel, BUCKETS, _flat_exposure(panel), 1_000.0,
                          cost_ceiling_bps=1.0, per_share=0.005,
                          min_per_order=100.0)
    assert log.iloc[0]["reason"] == "enter"
    assert log.iloc[0]["n_orders"] == len(BUCKETS)
    assert invested_fraction(daily) > 0.5


def test_rebalances_happen_monthly_not_daily():
    panel = _panel(days=250)
    _, log = simulate(panel, BUCKETS, _flat_exposure(panel), 100_000.0)
    monthly = log[log["reason"] == "monthly"]
    assert 10 <= len(monthly) <= 12, f"expected ~11 resets, got {len(monthly)}"


def test_quotas_fill_the_request_even_with_lopsided_buckets():
    """The bug that made a "120-name" basket hold 67.

    Asking four buckets for 30 each when two of them hold 10 and 11 returns 67
    names unless the shortfall is redistributed.
    """
    pools = {"us_equity": list(range(83)), "bonds": list(range(11)),
             "commodities": list(range(10)), "intl": list(range(16))}
    for n in (8, 12, 20, 40, 66, 120):
        got = quotas(n, pools)
        assert sum(got.values()) == n, f"n={n} produced {sum(got.values())}"
        for bucket, take in got.items():
            assert take <= len(pools[bucket])
    # Full size must take everything.
    assert quotas(120, pools) == {k: len(v) for k, v in pools.items()}


def test_asking_for_more_than_exists_caps_at_the_universe():
    pools = {"a": [1, 2], "b": [3]}
    assert sum(quotas(99, pools).values()) == 3
