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
from scripts.paper_track_v1 import BACKTEST, ROOT

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


def test_report_measures_returns_from_the_init_row_not_after_it(ledger, monkeypatch, capsys):
    """The entry cost belongs in Sharpe, not only in the headline total return.

    Building the equity path from the marks alone dropped the init-to-first-mark
    move, so total return counted the entry cost and every risk statistic did
    not -- two numbers quoted side by side off two different starting points.
    """
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:303])
    pt._run(_args())
    capsys.readouterr()

    rows = _entries()
    equity = [r["equity"] for r in rows]
    pt._report()
    out = capsys.readouterr().out

    # One return per gap between journal rows, including init -> first mark.
    assert f"交易日數        {len(rows) - 1}" in out
    entry_cost = next(r["cost"] for r in rows if r["action"] == "enter")
    assert entry_cost > 0
    assert equity[1] < equity[0], "the entry day must show the cost"


def test_the_monthly_rebalance_fires_across_a_year_boundary(ledger, monkeypatch):
    """A (year, month) comparison is right; comparing month alone would skip
    December to January and silently hold stale weights for a year."""
    # Start early enough that the 60-day volatility window is warm before the
    # year turns, so init can land in December and the run crosses into January.
    idx = pd.bdate_range("2024-09-02", periods=140)
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
         for s in UNIVERSE}, index=idx)
    init_at = panel.index.get_indexer([pd.Timestamp("2024-12-18")])[0]
    monkeypatch.setattr(pt, "load_prices", lambda s: panel.iloc[:init_at + 1])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: panel)
    pt._run(_args())

    rows = [r for r in _entries() if r["action"] == "rebalance"]
    months = {r["date"][:7] for r in rows}
    assert "2025-01" in months, \
        f"December to January did not trigger a rebalance; saw {sorted(months)}"


def test_the_quoted_backtest_benchmark_still_matches_its_source():
    """BACKTEST is a hand-copied number the ledger judges live results against.

    Nothing stopped it drifting from the report it cites -- and re-running the
    study without --wfo used to delete that section outright. Tie them together
    so a stale benchmark fails loudly instead of quietly flattering the record.
    """
    report = ROOT / BACKTEST["source"]
    if not report.exists():
        pytest.skip(f"{BACKTEST['source']} not generated in this checkout")
    payload = json.loads(report.read_text(encoding="utf-8"))
    section, arm = BACKTEST["key"].split(".")
    assert section in payload, \
        "walk-forward section missing; re-run the study with --wfo"
    wfo = payload[section][arm]
    assert wfo["oos_days"] == BACKTEST["oos_days"]
    assert wfo["oos_sharpe"] == pytest.approx(BACKTEST["sharpe"], abs=0.005)
    assert wfo["oos_profit_factor"] == pytest.approx(
        BACKTEST["profit_factor"], abs=0.005)
    assert wfo["oos_max_drawdown"] == pytest.approx(
        BACKTEST["max_drawdown"], abs=0.0005)
