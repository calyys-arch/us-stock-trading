"""The paper ledger has to be right before it is trusted for months.

Its whole value is being an honest forward record, so the failure modes that
matter are the quiet ones: double-trading a day that was already recorded,
skipping days when run weekly instead of daily, losing money to costs it never
charged, or manufacturing a track record by replaying history.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.paper_track_v1 as pt
from python.portfolio.strategy_v1 import COLLAPSE

UNIVERSE = {
    "USA": "us_single_name", "USB": "us_sector_etf", "USC": "us_broad_etf",
    "INT": "intl_equity_etf", "BND": "bond_etf", "CMD": "commodity_etf",
}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Point the module at a temp journal and a deterministic price panel."""
    monkeypatch.setattr(pt, "JOURNAL", tmp_path / "journal.jsonl")

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-01-01", periods=400)
    panel = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx))))
         for s in UNIVERSE},
        index=idx)
    monkeypatch.setattr(pt, "load_universe", lambda: (sorted(UNIVERSE), UNIVERSE))
    monkeypatch.setattr(pt, "load_prices", lambda symbols: panel)
    return panel


def _args(**kw):
    # no_update stays on: these tests must never reach the network.
    base = {"init": False, "capital": 100_000.0, "target_vol": 0.15,
            "report": False, "no_update": True}
    base.update(kw)
    return type("Args", (), base)()


def _entries():
    return [json.loads(l) for l in pt.JOURNAL.read_text().splitlines() if l.strip()]


def test_init_records_capital_and_takes_no_position(ledger):
    assert pt._run(_args(init=True)) == 0
    rows = _entries()
    assert len(rows) == 1
    assert rows[0]["action"] == "init"
    assert rows[0]["equity"] == 100_000.0
    assert rows[0]["holdings"] == {}
    # The record must start at the last priced day, not replay 400 days of it.
    assert rows[0]["date"] == str(ledger.index[-1].date())


def test_init_refuses_to_clobber_an_existing_ledger(ledger):
    pt._run(_args(init=True))
    before = pt.JOURNAL.read_text()
    assert pt._run(_args(init=True, capital=1.0)) == 1
    assert pt.JOURNAL.read_text() == before


def test_a_second_run_on_the_same_day_changes_nothing(ledger, monkeypatch):
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:320])
    pt._run(_args())
    after_first = pt.JOURNAL.read_text()
    assert pt._run(_args()) == 0
    assert pt.JOURNAL.read_text() == after_first, "re-run must be a no-op"


def test_running_weekly_fills_every_missing_day_rather_than_skipping(
        ledger, monkeypatch):
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:305])
    pt._run(_args())

    dates = [r["date"] for r in _entries() if r["action"] != "init"]
    assert dates == [str(d.date()) for d in ledger.index[300:305]]


def test_entering_charges_cost_and_leaves_equity_short_by_exactly_that(
        ledger, monkeypatch):
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:301])
    pt._run(_args())

    rows = _entries()
    entry = rows[-1]
    assert entry["action"] == "enter"
    assert entry["cost"] > 0, "entering the whole book cannot be free"
    # Equity right after entering is the prior equity less the cost paid.
    assert entry["equity"] == pytest.approx(100_000.0 - entry["cost"], abs=0.02)
    assert set(entry["holdings"]) == set(UNIVERSE)


def test_holdings_land_on_the_bucket_weights_not_the_ticker_count(
        ledger, monkeypatch):
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:301])
    pt._run(_args())

    entry = _entries()[-1]
    prices = ledger.iloc[300]
    value = {s: sh * prices[s] for s, sh in entry["holdings"].items()}
    invested = sum(value.values())
    by_bucket: dict[str, float] = {}
    for symbol, dollars in value.items():
        by_bucket.setdefault(COLLAPSE[UNIVERSE[symbol]], 0.0)
        by_bucket[COLLAPSE[UNIVERSE[symbol]]] += dollars

    assert len(by_bucket) == 4
    for bucket, dollars in by_bucket.items():
        assert dollars / invested == pytest.approx(0.25, abs=1e-6), bucket


def test_exposure_never_exceeds_one_so_the_ledger_cannot_borrow(
        ledger, monkeypatch):
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:250])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger)
    pt._run(_args())

    for row in _entries():
        if row["action"] == "init":
            continue
        assert row["exposure_held"] <= 1.0 + 1e-6
        if row["exposure_target"] is not None:
            assert row["exposure_target"] <= 1.0 + 1e-6
        assert row["cash"] >= -1e-6, "cash must never go negative"


def test_warmup_days_take_no_position_instead_of_guessing(ledger, monkeypatch):
    short = ledger.iloc[:10]
    monkeypatch.setattr(pt, "load_prices", lambda s: short)
    pt._run(_args(init=True))
    # Init lands on day 10; extend by a few days that still lack 60 days of
    # bucket returns for a volatility estimate.
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:14])
    pt._run(_args())

    for row in _entries()[1:]:
        assert row["action"] == "warmup"
        assert row["holdings"] == {}
        assert row["cost"] == 0.0
        assert row["equity"] == 100_000.0


def test_trade_to_carries_a_line_it_has_no_fresh_price_for(ledger):
    prices = pd.Series({"USA": 100.0, "BND": float("nan")})
    holdings = {"USA": 5.0, "BND": 7.0}
    weights = {"USA": 0.5, "BND": 0.5}
    new, cash, cost = pt._trade_to(holdings, prices, weights, 10_000.0, 1.0)
    # BND cannot be priced, so its 7 shares are carried rather than dumped at a
    # stale mark or silently dropped.
    assert new["BND"] == 7.0
    assert cost > 0
    # Just under the naive 50 shares, because the cost is reserved out of the
    # budget rather than overdrawing cash to pay it.
    assert new["USA"] < 50.0
    assert new["USA"] == pytest.approx(50.0 - cost / 2 / 100.0, abs=1e-3)


def test_paying_costs_never_overdraws_cash_at_full_exposure(ledger):
    """The regression that motivated reserving cost inside the budget."""
    prices = pd.Series({s: 100.0 for s in UNIVERSE})
    weights = {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}
    new, cash, cost = pt._trade_to({}, prices, weights, 10_000.0, 1.0)
    invested = sum(sh * 100.0 for sh in new.values())
    assert cost > 0
    assert cash >= -1e-6, "fully invested must not borrow to pay fees"
    assert invested + cost == pytest.approx(10_000.0, abs=0.01)
