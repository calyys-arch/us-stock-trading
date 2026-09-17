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


@pytest.mark.parametrize("now_et,expect,why", [
    ("2026-08-26 22:30", "2026-08-26", "after the close, today has printed"),
    ("2026-08-26 11:00", "2026-08-25", "mid-session, today is still partial"),
    ("2026-08-26 16:30", "2026-08-25", "just after the bell, let it settle"),
    ("2026-08-24 09:00", "2026-08-21", "Monday pre-open falls back to Friday"),
    ("2026-08-23 12:00", "2026-08-21", "Sunday falls back to Friday"),
])
def test_the_top_up_target_is_the_last_session_that_definitely_closed(
        now_et, expect, why):
    """Marking against a partial bar is unrepairable in an append-only journal,
    but the old rule overpaid for that safety: it asked for the local
    yesterday, and this machine's local date runs ahead of New York, so the
    ledger sat two sessions behind rather than one."""
    now = pd.Timestamp(now_et, tz="America/New_York")
    assert pt.last_closed_session(now) == pd.Timestamp(expect), why


def test_the_request_compensates_for_yfinances_exclusive_end(monkeypatch):
    """Asking for the session itself returns everything up to the day before
    it, which is how the second lost day crept in."""
    seen = {}
    monkeypatch.setattr(
        "python.data.price_cache.top_up_cached_panel",
        lambda symbols, end, **kw: seen.update(end=end) or {
            "topped_up": [], "already_current": [], "readjusted": [],
            "not_cached": [], "failed": []})
    pt._top_up()
    assert seen["end"] == pt.last_closed_session() + pd.Timedelta(days=1)


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


def test_an_unpriced_holding_is_carried_not_marked_at_zero(ledger, monkeypatch, capsys):
    """The defect that could have sold real shares against a fake equity number.

    `_equity` skipped any holding without a finite price, which values it at
    zero rather than at its last close. One commodity name is 2.5% of the book,
    so a single failed vendor fetch printed a phantom 2.5% loss that reversed
    the next day -- and if the band tripped on that day the whole book was
    resized against the understated equity.
    """
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))

    # CMD is the only commodity name, so bucket weighting gives it 25% of the
    # book on its own -- the largest single mark this panel can lose.
    blanked = ledger.copy()
    gap = blanked.index[301]
    blanked.loc[gap, "CMD"] = np.nan
    monkeypatch.setattr(pt, "load_prices", lambda s: blanked.iloc[:303])
    pt._run(_args())
    capsys.readouterr()

    rows = [r for r in _entries() if r["action"] != "init"]
    on_gap = next(r for r in rows if r["date"] == str(gap.date()))
    prior = max((r for r in rows if r["date"] < str(gap.date())),
                key=lambda r: r["date"])
    move = on_gap["equity"] / prior["equity"] - 1
    assert abs(move) < 0.05, \
        f"a carried mark cannot move equity by {move:.2%} on one missing price"


def test_equity_refuses_to_value_a_holding_it_has_no_mark_for():
    prices = pd.Series({s: 100.0 for s in UNIVERSE if s != "CMD"})
    with pytest.raises(ValueError, match="沒有可用價格卻持有"):
        pt._equity({"CMD": 10.0}, prices, 0.0)


def test_the_run_stops_rather_than_writing_a_mark_it_cannot_defend(
        ledger, monkeypatch, capsys):
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))

    blanked = ledger.copy()
    blanked.loc[blanked.index[301]:, "CMD"] = np.nan
    monkeypatch.setattr(pt, "load_prices", lambda s: blanked.iloc[:320])
    status = pt._run(_args())
    out = capsys.readouterr().out

    assert "停在這裡" in out
    dates = [r["date"] for r in _entries()]
    # Advanced up to the carry limit, then stopped instead of guessing.
    assert str(blanked.index[301 + pt.MAX_CARRY_DAYS].date()) not in dates
    # A halt partway through `pending` must not report success: this used to
    # fall through to `return 0` regardless of how the loop ended, so
    # paper_v1_daily.sh's `status=$?` check saw a false success and went on
    # to run --report / early-read as if every pending day had been written.
    assert status == 1


def test_a_single_day_move_past_the_catastrophe_threshold_halts_the_run(
        ledger, monkeypatch, capsys):
    """CATASTROPHE_DAY_RETURN is the unconditional tripwire: no history
    lookup, no day-count minimum, just "is this mark even plausible."""
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:301])
    pt._run(_args())  # enters the book

    spiked = ledger.copy()
    blow_up_day = spiked.index[303]
    # CMD is the book's only commodity name and therefore a whole 25%-weight
    # bucket on its own (see the bucket-weights test above); a 6x spike there
    # moves total equity by roughly +125% in one day regardless of what the
    # other three buckets do, far past the 10% threshold.
    spiked.loc[blow_up_day, "CMD"] *= 6.0
    monkeypatch.setattr(pt, "load_prices", lambda s: spiked.iloc[:305])
    status = pt._run(_args())
    out = capsys.readouterr().out

    assert status == 1
    assert "停在這裡" in out
    assert f"{pt.CATASTROPHE_DAY_RETURN:.0%}" in out
    dates = [r["date"] for r in _entries()]
    assert str(blow_up_day.date()) not in dates, \
        "the implausible mark must not be written to the append-only journal"


def test_ordinary_daily_moves_never_trip_the_catastrophe_threshold(
        ledger, monkeypatch):
    """The fixture's own random walk (1.2% daily vol per name, diversified
    across four equal buckets) must never come near a 10% single-day swing --
    otherwise the threshold would be firing on ordinary noise, not bugs."""
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger)
    status = pt._run(_args())
    assert status == 0
    assert len(_entries()) == 1 + (len(ledger) - 300)


def test_commission_is_charged_per_order_not_as_a_flat_rate_on_notional():
    """The ledger charged 4bps of notional and nothing per order.

    That models spread but is silent on the $1.00 minimum, which at retail size
    is most of the bill: entering 120 positions of $817 costs $120 in minimums
    against $39 of spread. Nothing pinned the magnitude, so swapping the cost
    model in broke no test.
    """
    prices = pd.Series({s: 100.0 for s in UNIVERSE})
    weights = {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}
    _, _, cost = pt._size_to({}, prices, weights, 6_000.0)
    spread_only = 6_000.0 * pt.ONE_WAY_COST_BPS / 10_000.0
    # Six orders of $1,000 each: 10 shares apiece, so the $1.00 floor binds.
    assert cost == pytest.approx(spread_only + 6 * 1.00, abs=0.01)
    assert cost > spread_only * 2, "per-order minimums must dominate at this size"


def test_the_drift_band_skips_the_orders_where_the_minimum_bites():
    """Where the band's saving actually comes from.

    A truly tiny order is nearly free in dollars, because the 1% cap overrides
    the $1.00 floor: a 10-cent trade costs a tenth of a cent. The money goes on
    mid-sized orders -- above $100 of notional, so the cap no longer protects,
    but under 200 shares, so the floor still binds. A $200 drift correction is
    charged the full $1.00, which is 50bps of the value it moved.
    """
    prices = pd.Series({s: 100.0 for s in UNIVERSE})
    weights = {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}
    held = {s: 10.0 for s in UNIVERSE}  # $1,000 a name
    # Target $1,200 a name: a $200 order, squarely in the floor's range.
    _, _, unbanded = pt._size_to(held, prices, weights, 7_200.0)
    spread = 6 * 200.0 * pt.ONE_WAY_COST_BPS / 10_000.0
    assert unbanded == pytest.approx(spread + 6 * 1.00, abs=0.01)

    new, _, banded = pt._size_to(held, prices, weights, 7_200.0,
                                 ceiling=pt.DRIFT_BAND_BPS)
    assert new == held, "a 50bps order must not be placed under a 10bps band"
    assert banded == 0.0


def test_a_trade_too_small_for_the_minimum_to_bite_is_priced_by_the_cap():
    prices = pd.Series({s: 100.0 for s in UNIVERSE})
    weights = {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}
    held = {s: 10.0 for s in UNIVERSE}
    # A 10-cent order a name: 1% of value beats the $1.00 floor.
    _, _, cost = pt._size_to(held, prices, weights, 6_000.6)
    spread = 6 * 0.10 * pt.ONE_WAY_COST_BPS / 10_000.0
    assert cost == pytest.approx(spread + 6 * 0.001, abs=1e-4)


def test_the_drift_band_still_places_an_order_large_enough_to_be_worth_it():
    prices = pd.Series({s: 100.0 for s in UNIVERSE})
    weights = {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}
    held = {s: 10.0 for s in UNIVERSE}
    # Doubling every position is a $1,000 order a name, well over the band.
    new, _, cost = pt._size_to(held, prices, weights, 12_000.0,
                               ceiling=pt.DRIFT_BAND_BPS)
    assert all(new[s] > held[s] for s in UNIVERSE)
    assert cost > 0


def test_entering_is_never_subject_to_the_drift_band(ledger, monkeypatch):
    """A notional floor applied to the entry means a small account never
    invests, and the flat return that follows looks like a strategy result."""
    monkeypatch.setattr(pt, "DRIFT_BAND_BPS", 1.0)  # rejects almost everything
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:300])
    pt._run(_args(init=True))
    monkeypatch.setattr(pt, "load_prices", lambda s: ledger.iloc[:301])
    pt._run(_args())
    entered = [e for e in _entries() if e["action"] == "enter"]
    assert entered, "the ledger must still take a position"
    assert entered[0]["exposure_held"] > 0.5


def test_an_exposure_change_is_never_subject_to_the_drift_band(ledger,
                                                              monkeypatch):
    """Cutting exposure trims every position by a fraction, so any notional
    floor blocks the strategy's only defence against a drawdown."""
    monkeypatch.setattr(pt, "DRIFT_BAND_BPS", 1.0)
    prices = pd.Series({s: 100.0 for s in UNIVERSE})
    weights = {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}
    held = {s: 100.0 for s in UNIVERSE}  # $60,000 invested
    # A brake event: exposure 1.0 -> 0.6. Unbanded, every line must move.
    new, cash, cost = pt._trade_to(held, prices, weights, 60_000.0, 0.6,
                                   ceiling=None)
    assert all(new[s] < held[s] for s in UNIVERSE), "brake must actually sell"
    assert cash > 0
