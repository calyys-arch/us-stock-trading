"""
Scanned-universe pairs trading (python/backtest/pairs_scan_engine.py).

The centre of gravity here is LOOK-AHEAD BIAS. Selecting pairs by scanning a
universe is only a valid experiment if each fold's pairs were chosen using
data that existed before that fold traded; scan the whole history first and
you have manufactured the result rather than measured it. Four tests below
exist purely to make that failure mode impossible to reintroduce silently:

  - test_scan_window_never_includes_as_of_or_later
  - test_scan_results_unchanged_by_future_price_mutation
  - test_replay_cannot_reach_a_future_scan
  - test_trades_before_cutoff_unchanged_by_future_price_mutation

Each is written so that the OBVIOUS way to break point-in-time discipline
(an inclusive slice bound, a schedule anchored to the window instead of the
global index, a replay that grabs the nearest scan rather than the latest
PAST one) turns it red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from python.backtest.pairs_scan_engine import (
    MAX_CONCURRENT_PAIRS,
    PairsScanConfig,
    _effective_scan_date,
    build_scan_schedule,
    candidate_pairs_from_buckets,
    load_pairs_universe,
    pairs_buckets,
    run_scan_backtest,
    scan_positions,
    select_active_pairs,
)
from python.core.fees_equity import round_trip_cost
from python.stat.cointegration import current_spread, spread_z_score
from python.stat.cointegration import test_pair as run_cadf_test   # aliased: pytest would
                                                                   # otherwise collect the
                                                                   # imported function itself

_LOOKBACK = 120
_REVALIDATE = 20


def _cointegrated_leg_pair(n: int, seed: int, phi: float = 0.92):
    """Genuinely cointegrated by construction — same generator shape as
    run_backtest._synthetic_pair / test_optimize._cointegrated_pair (a
    dominant shared random-walk trend plus an anti-correlated AR(1) spread
    with a multi-day half-life)."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + rng.normal(0, 0.15)
    log_a = 4.0 + 0.1 * common + 0.5 * spread + rng.normal(0, 0.02, n)
    log_b = 3.8 + 0.1 * common - 0.5 * spread + rng.normal(0, 0.02, n)
    return np.exp(log_a), np.exp(log_b)


def _panel(n_days: int = 600, n_pairs: int = 3, seed: int = 11):
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    cols = {}
    for i in range(n_pairs):
        a, b = _cointegrated_leg_pair(n_days, seed + i)
        cols[f"A{i}"], cols[f"B{i}"] = a, b
    close = pd.DataFrame(cols, index=dates)
    adv = pd.DataFrame(3.0e8, index=dates, columns=close.columns)
    return close, adv


def _all_pairs(close: pd.DataFrame) -> list[tuple[str, str]]:
    return candidate_pairs_from_buckets({"g": list(close.columns)})


def _schedule(close: pd.DataFrame, **kw):
    return build_scan_schedule(
        close, _all_pairs(close),
        lookback_days=kw.get("lookback_days", _LOOKBACK),
        revalidate_every_days=kw.get("revalidate_every_days", _REVALIDATE),
        checkpoint_path=kw.get("checkpoint_path"),
        progress_every=0,
    )


# ── z-score scale (regression guard for the intercept mismatch) ─────────────
#
# `test_pair` summarizes the spread and `current_spread` reconstructs it at
# trade time; if those two disagree about whether the OLS intercept is part of
# "the spread", every z-score is shifted by alpha/sigma. On real ETF price
# levels that shift is order 1e2, which pins |z| permanently above any
# `entry_z` and permanently outside any `exit_z` — the strategy then enters on
# the first eligible bar and can only ever exit on the stale timeout. These
# tests fail loudly if that mismatch is reintroduced on either side.

def _window_z(prices_a: pd.Series, prices_b: pd.Series) -> np.ndarray:
    result = run_cadf_test("A", "B", prices_a, prices_b)
    return np.array([
        spread_z_score(current_spread(a, b, result.hedge_ratio),
                       result.spread_mean, result.spread_std)
        for a, b in zip(prices_a, prices_b)
    ])


def test_z_score_is_centred_and_unit_scaled_in_sample():
    a, b = _cointegrated_leg_pair(300, seed=3)
    z = _window_z(pd.Series(a), pd.Series(b))
    assert z.mean() == pytest.approx(0.0, abs=1e-8)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("scale_a,scale_b", [(1.0, 1.0), (250.0, 25.0), (12.0, 900.0)])
def test_z_score_stays_order_one_at_any_price_level(scale_a, scale_b):
    """The failure mode is scale-dependent: it only bites when the two legs
    trade at very different absolute prices (DIA ~$250 vs SPY ~$265 already
    produced |z| ~ 245 on real data)."""
    a, b = _cointegrated_leg_pair(300, seed=5)
    z = _window_z(pd.Series(a * scale_a), pd.Series(b * scale_b))
    assert abs(z).max() < 6.0, "in-sample |z| should live in single digits, not hundreds"


def test_spread_mean_matches_the_traded_spread_definition():
    a, b = _cointegrated_leg_pair(300, seed=7)
    prices_a, prices_b = pd.Series(a * 250.0), pd.Series(b * 25.0)
    result = run_cadf_test("A", "B", prices_a, prices_b)
    realized = np.array([current_spread(x, y, result.hedge_ratio)
                         for x, y in zip(prices_a, prices_b)])
    assert result.spread_mean == pytest.approx(float(realized.mean()), abs=1e-10)
    assert result.spread_std == pytest.approx(float(realized.std(ddof=1)), abs=1e-10)


def test_positions_actually_exit_on_z_reversion():
    """End-to-end consequence of the above: with a working z-score, a
    mean-reverting spread must produce reversion exits, not a book that can
    only ever time out."""
    close, adv = _panel(n_days=600, n_pairs=3, seed=13)
    report = run_scan_backtest(close, adv, _schedule(close), PairsScanConfig())
    reasons = {t.exit_reason for t in report.trades}
    assert report.trades
    assert "z_reversion" in reasons, f"only saw exit reasons {reasons}"


# ── candidate universe ──────────────────────────────────────────────────────

def test_candidate_pairs_are_within_bucket_only():
    buckets = {"energy": ["XLE", "XOP", "OIH"], "bonds": ["TLT", "IEF"]}
    pairs = candidate_pairs_from_buckets(buckets)
    bucket_of = {c: b for b, codes in buckets.items() for c in codes}
    assert len(pairs) == 3 + 1                      # C(3,2) + C(2,2)
    for a, b in pairs:
        assert a < b, "pairs must be canonicalized (a < b) so (x, y) and (y, x) are one test"
        assert bucket_of[a] == bucket_of[b], f"cross-bucket pair leaked: {(a, b)}"


def test_pairs_buckets_ignores_metadata_strings():
    """Live seed used to flatten universe.values(), so computed_at /
    selection_rule were iterated as character 'symbols' (space, punctuation)."""
    universe = load_pairs_universe()
    buckets = pairs_buckets(universe)
    symbols = {c for codes in buckets.values() for c in codes}
    assert symbols
    assert all(isinstance(s, str) and s.isalpha() and s.isupper() for s in symbols)
    assert " " not in symbols
    assert "(" not in symbols


def test_repo_pairs_universe_is_well_formed():
    """configs/pairs_universe.yaml is a fixed, a-priori grouping — a symbol in
    two buckets would silently multiply the number of tests run on it."""
    universe = load_pairs_universe()
    buckets = universe["buckets"]
    seen: set[str] = set()
    for codes in buckets.values():
        assert len(codes) == len(set(codes))
        assert not (seen & set(codes)), "a symbol appears in more than one bucket"
        seen |= set(codes)
    pairs = candidate_pairs_from_buckets(buckets)
    expected = sum(len(c) * (len(c) - 1) // 2 for c in buckets.values())
    assert len(pairs) == expected


# ── scan cadence ────────────────────────────────────────────────────────────

def test_scan_positions_anchored_to_global_index():
    """Scan dates are a function of the FULL price index only. If they were
    derived from a backtest window's start instead, different walk-forward
    folds would rescan on different dates and each fold would effectively get
    its own (subtly different) selection schedule."""
    positions = scan_positions(1000, lookback_days=252, revalidate_every_days=21)
    assert positions[0] == 252
    assert all(p >= 252 for p in positions)
    assert all(b - a == 21 for a, b in zip(positions, positions[1:]))
    # A longer history only appends; it never moves an existing scan date.
    longer = scan_positions(1500, lookback_days=252, revalidate_every_days=21)
    assert longer[: len(positions)] == positions


def test_scan_positions_rejects_nonpositive_cadence():
    with pytest.raises(ValueError):
        scan_positions(100, lookback_days=10, revalidate_every_days=0)


# ── LOOK-AHEAD: the scan itself ─────────────────────────────────────────────

def test_scan_window_never_includes_as_of_or_later(monkeypatch):
    """Every cointegration test must be fitted on prices STRICTLY before the
    date whose trading it authorizes. An inclusive upper bound (`iloc[j - L:
    j + 1]`) would make this fail immediately."""
    close, _adv = _panel(n_days=400)
    import python.stat.pair_scanner as scanner_mod

    real_test_pair = scanner_mod.test_pair
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def recording_test_pair(code_a, code_b, prices_a, prices_b, computed_at=None):
        seen.append((pd.Timestamp(computed_at), prices_a.index.max()))
        return real_test_pair(code_a, code_b, prices_a, prices_b, computed_at=computed_at)

    monkeypatch.setattr(scanner_mod, "test_pair", recording_test_pair)
    _schedule(close)

    assert seen, "no cointegration test ran — the assertion below would be vacuous"
    for as_of, last_price_date in seen:
        assert last_price_date < as_of, (
            f"scan dated {as_of} was fitted on prices through {last_price_date} — "
            "that bar was not knowable when the scan authorized trading it"
        )


def test_scan_results_unchanged_by_future_price_mutation():
    """Corrupt every price on/after a cutoff. Scans dated at or before the
    cutoff must be bit-identical, because they never saw those rows."""
    close, _adv = _panel(n_days=400)
    baseline = _schedule(close)
    cutoff = sorted(baseline)[len(baseline) // 2]

    corrupted = close.copy()
    corrupted.loc[corrupted.index >= cutoff] *= 100.0
    mutated = _schedule(corrupted)

    checked = 0
    for as_of in sorted(baseline):
        if as_of > cutoff:
            continue
        base_r = {(r.code_a, r.code_b): r for r in baseline[as_of]}
        mut_r = {(r.code_a, r.code_b): r for r in mutated[as_of]}
        assert base_r.keys() == mut_r.keys(), f"eligible set changed at {as_of}"
        for key, r in base_r.items():
            assert r.cadf_tstat == pytest.approx(mut_r[key].cadf_tstat, abs=1e-12)
            assert r.hedge_ratio == pytest.approx(mut_r[key].hedge_ratio, abs=1e-12)
            assert r.spread_mean == pytest.approx(mut_r[key].spread_mean, abs=1e-12)
        checked += 1
    assert checked > 1


# ── LOOK-AHEAD: the replay ──────────────────────────────────────────────────

def test_effective_scan_date_is_the_latest_PAST_scan():
    scans = pd.DatetimeIndex(["2020-01-10", "2020-02-10", "2020-03-10"])
    assert _effective_scan_date(scans, pd.Timestamp("2020-01-09")) is None
    assert _effective_scan_date(scans, pd.Timestamp("2020-01-10")) == pd.Timestamp("2020-01-10")
    assert _effective_scan_date(scans, pd.Timestamp("2020-02-09")) == pd.Timestamp("2020-01-10")
    assert _effective_scan_date(scans, pd.Timestamp("2020-03-31")) == pd.Timestamp("2020-03-10")


def test_replay_cannot_reach_a_future_scan():
    """Inject a scan dated after the replay window that would, if visible,
    authorize a completely different pair set. Results must not budge — a
    `searchsorted(..., side="left")` or a nearest-scan lookup would fail."""
    close, adv = _panel(n_days=400)
    schedule = _schedule(close)
    baseline = run_scan_backtest(close, adv, schedule, PairsScanConfig())

    future_date = close.index[-1] + pd.Timedelta(days=30)
    poisoned = dict(schedule)
    poisoned[future_date] = list(schedule[sorted(schedule)[0]])
    poisoned_run = run_scan_backtest(close, adv, poisoned, PairsScanConfig())

    assert len(poisoned_run.trades) == len(baseline.trades)
    assert poisoned_run.total_net_pnl == pytest.approx(baseline.total_net_pnl, abs=1e-9)


def test_trades_before_cutoff_unchanged_by_future_price_mutation():
    """End-to-end: nothing that happens after date T may alter a trade that
    had already closed by T — not through the scan, not through the replay."""
    close, adv = _panel(n_days=500)
    schedule = _schedule(close)
    cfg = PairsScanConfig(coint_lookback_days=_LOOKBACK, revalidate_every_days=_REVALIDATE)
    baseline = run_scan_backtest(close, adv, schedule, cfg)
    assert baseline.trades, "need trades for this test to mean anything"

    cutoff = close.index[350]
    corrupted = close.copy()
    corrupted.loc[corrupted.index >= cutoff] *= 100.0
    mutated = run_scan_backtest(corrupted, adv, _schedule(corrupted), cfg)

    def before(report):
        return [(t.code_a, t.code_b, t.entry_date, t.exit_date, round(t.net_pnl, 6))
                for t in report.trades if pd.Timestamp(t.exit_date) < cutoff]

    assert before(baseline), "cutoff chosen badly — no closed trades before it"
    assert before(baseline) == before(mutated)


# ── selection rule ──────────────────────────────────────────────────────────

def test_selection_ranks_by_cadf_tstat_and_applies_half_life_band():
    close, _adv = _panel(n_days=400)
    schedule = _schedule(close)
    as_of = sorted(schedule)[-1]
    results = schedule[as_of]
    assert results

    cfg = PairsScanConfig(min_half_life_days=1.0, max_half_life_days=60.0)
    active = select_active_pairs(results, cfg)
    tstats = [r.cadf_tstat for r in active]
    assert tstats == sorted(tstats), "ranking must be ascending CADF t-stat (strongest first)"
    assert all(1.0 <= r.half_life_days <= 60.0 for r in active)

    # Tightening the band can only ever shrink the eligible set.
    narrow = select_active_pairs(results, PairsScanConfig(min_half_life_days=5.0, max_half_life_days=20.0))
    assert len(narrow) <= len(active)
    assert {(r.code_a, r.code_b) for r in narrow} <= {(r.code_a, r.code_b) for r in active}


def test_concurrent_pair_cap_is_respected():
    close, adv = _panel(n_days=500, n_pairs=4, seed=21)
    schedule = _schedule(close)
    capped = run_scan_backtest(close, adv, schedule, PairsScanConfig(max_concurrent_pairs=2))
    uncapped = run_scan_backtest(close, adv, schedule, PairsScanConfig(max_concurrent_pairs=MAX_CONCURRENT_PAIRS))
    assert len(capped.trades) <= len(uncapped.trades)

    # Reconstruct concurrency from the trade list: at most `cap` open at once.
    events = []
    for t in capped.trades:
        events.append((pd.Timestamp(t.entry_date), 1))
        events.append((pd.Timestamp(t.exit_date), -1))
    events.sort(key=lambda e: (e[0], e[1]))   # closes before opens on the same day
    open_count = peak = 0
    for _date, delta in events:
        open_count += delta
        peak = max(peak, open_count)
    assert peak <= 2


def test_scan_produces_far_more_trades_than_a_single_pair():
    """The entire point of the exercise: a scanned universe must lift the
    trade count out of the 8-trades-in-7-years regime that made the Monte
    Carlo gate uninformative for the single hardcoded pair."""
    close, adv = _panel(n_days=600, n_pairs=4, seed=31)
    schedule = _schedule(close)
    scanned = run_scan_backtest(close, adv, schedule, PairsScanConfig())

    single_pairs = [p for p in _all_pairs(close) if p == ("A0", "B0")]
    single_schedule = build_scan_schedule(
        close, single_pairs, lookback_days=_LOOKBACK,
        revalidate_every_days=_REVALIDATE, progress_every=0)
    single = run_scan_backtest(close, adv, single_schedule, PairsScanConfig())

    assert len(scanned.trades) > len(single.trades)
    assert len(scanned.pairs_traded) > 1


# ── checkpointing ───────────────────────────────────────────────────────────

def test_scan_checkpoint_resumes_without_recomputing(tmp_path, monkeypatch):
    close, _adv = _panel(n_days=400)
    ckpt = tmp_path / "scan.jsonl"
    first = _schedule(close, checkpoint_path=ckpt)
    assert ckpt.exists() and len(ckpt.read_text().splitlines()) == len(first)

    import python.backtest.pairs_scan_engine as engine_mod

    def explode(*_a, **_kw):
        raise AssertionError("resumed run recomputed a scan date already on disk")

    monkeypatch.setattr(engine_mod, "scan", explode)
    resumed = _schedule(close, checkpoint_path=ckpt)

    assert sorted(resumed) == sorted(first)
    for as_of in first:
        a = {(r.code_a, r.code_b): r.cadf_tstat for r in first[as_of]}
        b = {(r.code_a, r.code_b): r.cadf_tstat for r in resumed[as_of]}
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_scan_checkpoint_refuses_to_resume_a_different_scan_definition(tmp_path):
    """A stale cache is the quiet way a methodology fix gets silently undone —
    e.g. resuming results computed under the OLD spread definition after
    python/stat/cointegration.py was corrected. The fingerprint makes that a
    hard error instead."""
    close, _adv = _panel(n_days=400)
    ckpt = tmp_path / "scan.jsonl"
    _schedule(close, checkpoint_path=ckpt)

    with pytest.raises(ValueError, match="different scan definition"):
        _schedule(close, checkpoint_path=ckpt, revalidate_every_days=_REVALIDATE + 5)

    smaller = close.drop(columns=[close.columns[-1]])
    with pytest.raises(ValueError, match="different scan definition"):
        build_scan_schedule(
            smaller, candidate_pairs_from_buckets({"g": list(smaller.columns)}),
            lookback_days=_LOOKBACK, revalidate_every_days=_REVALIDATE,
            checkpoint_path=ckpt, progress_every=0)


# ── costs ───────────────────────────────────────────────────────────────────

def test_half_spread_defaults_to_zero_and_scales_linearly():
    """Default 0.0 keeps every pre-existing caller's behavior identical; when
    supplied, the half-spread is charged on BOTH fills."""
    kwargs = dict(shares=1000, entry_price=100.0, exit_price=100.0, adv_dollars=1e9)
    assert round_trip_cost(**kwargs).half_spread == 0.0
    three = round_trip_cost(**kwargs, half_spread_bps=3.0)
    six = round_trip_cost(**kwargs, half_spread_bps=6.0)
    # 1000 shares x $100 x 3bps x 2 fills = $60
    assert three.half_spread == pytest.approx(60.0)
    assert six.half_spread == pytest.approx(2 * three.half_spread)
    assert six.total > three.total > round_trip_cost(**kwargs).total


def test_wider_spread_assumption_only_ever_costs_more():
    close, adv = _panel(n_days=500, n_pairs=3, seed=41)
    schedule = _schedule(close)
    normal = run_scan_backtest(close, adv, schedule, PairsScanConfig(half_spread_bps=3.0))
    stressed = run_scan_backtest(close, adv, schedule, PairsScanConfig(half_spread_bps=6.0))
    assert len(normal.trades) == len(stressed.trades), "cost must not change WHICH trades fire"
    assert stressed.total_cost > normal.total_cost
    assert stressed.total_net_pnl < normal.total_net_pnl


def test_market_impact_is_actually_priced():
    """engine.run_pairs_backtest passes adv_dollars=0.0, i.e. impact
    unmodeled. The scan engine feeds real 20-day dollar ADV, so a thinner
    book must cost more."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=51)
    schedule = _schedule(close)
    deep = run_scan_backtest(close, adv, schedule, PairsScanConfig())
    thin = run_scan_backtest(close, adv * 0.001, schedule, PairsScanConfig())
    assert thin.total_cost > deep.total_cost


# ── entry_gate (regime_gate_report.md Phase 2 gating hook) ──────────────────

def test_entry_gate_none_is_byte_identical_to_omitting_it():
    """Default None must reproduce every existing caller's behavior exactly —
    this is an ADDITIVE parameter, not a behavior change."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=61)
    schedule = _schedule(close)
    a = run_scan_backtest(close, adv, schedule, PairsScanConfig())
    b = run_scan_backtest(close, adv, schedule, PairsScanConfig(), entry_gate=None)
    assert len(a.trades) == len(b.trades)
    assert a.total_net_pnl == pytest.approx(b.total_net_pnl, abs=1e-9)


def test_entry_gate_all_false_blocks_every_new_entry():
    close, adv = _panel(n_days=500, n_pairs=3, seed=62)
    schedule = _schedule(close)
    closed_gate = pd.Series(False, index=close.index)
    report = run_scan_backtest(close, adv, schedule, PairsScanConfig(), entry_gate=closed_gate)
    assert len(report.trades) == 0
    assert report.open_at_end == 0


def test_entry_gate_missing_day_defaults_closed_not_open():
    """A day absent from the gate Series (e.g. a symbol/date the classifier
    never labeled) must fail CLOSED, matching
    trend_efficiency_gate.shifted_entry_gate's own fail-closed default for
    undecided rows — an accidentally sparse gate must never silently trade
    as if ungated."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=63)
    schedule = _schedule(close)
    empty_gate = pd.Series(dtype=bool)  # no index entries at all
    report = run_scan_backtest(close, adv, schedule, PairsScanConfig(), entry_gate=empty_gate)
    assert len(report.trades) == 0


def test_entry_gate_blocks_only_new_entries_not_existing_exits():
    """Open a position while the gate is ON, then flip the gate OFF for the
    rest of the run: that position must still be free to exit normally (the
    gate is an entry filter, never a forced liquidation) — same documented
    behavior as the pre-existing max_concurrent_pairs cap."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=64)
    schedule = _schedule(close)
    ungated = run_scan_backtest(close, adv, schedule, PairsScanConfig())
    assert ungated.trades, "need at least one trade for this test to mean anything"
    first_entry = min(pd.Timestamp(t.entry_date) for t in ungated.trades)

    # Gate ON only for the single day of the first entry, OFF every day after.
    gate = pd.Series(False, index=close.index)
    gate.loc[gate.index <= first_entry] = True
    gated = run_scan_backtest(close, adv, schedule, PairsScanConfig(), entry_gate=gate)

    gated_entries = sorted(pd.Timestamp(t.entry_date) for t in gated.trades)
    assert gated_entries, "the one pre-gate-close entry should still have fired"
    assert all(d <= first_entry for d in gated_entries), \
        "no entry should have opened after the gate closed"
    # That one position must still have been able to exit (booked as a trade,
    # not stuck open forever just because the gate later closed).
    assert any(pd.Timestamp(t.entry_date) <= first_entry for t in gated.trades)


# ── parameter discipline ────────────────────────────────────────────────────

def test_scan_adds_no_free_parameter_to_the_strategy_config():
    """The scan must not smuggle knobs into configs/strategy.yaml — that block
    is already exactly at the Chan Ch.3 ceiling of 5 (see
    tests/test_chan_guards.py). Scan structure lives in code constants and in
    configs/pairs_universe.yaml (data, like configs/universe.yaml)."""
    from python.backtest.param_guard import MAX_FREE_PARAMETERS, check_max_parameters

    with open("configs/strategy.yaml", encoding="utf-8") as f:
        strategy_cfg = yaml.safe_load(f)
    ok, n = check_max_parameters(strategy_cfg["pairs_trading"])
    assert ok and n == MAX_FREE_PARAMETERS

    scan_only_keys = {"max_concurrent_pairs", "half_spread_bps",
                      "candidate_universe", "pairs_universe"}
    assert not (set(strategy_cfg["pairs_trading"]) & scan_only_keys)
    assert "pairs_universe" not in strategy_cfg

    # And the two structural constants are pinned, so a future edit that
    # starts tuning them is a visible, reviewed diff rather than config creep.
    from python.backtest.pairs_scan_engine import DEFAULT_HALF_SPREAD_BPS

    assert MAX_CONCURRENT_PAIRS == 10
    assert DEFAULT_HALF_SPREAD_BPS == 3.0


def test_param_grid_never_grids_a_scan_constant():
    """configs/param_grids.yaml may only vary genuine signal thresholds."""
    from python.backtest.optimize import load_param_grid

    gridded = {k for combo in load_param_grid("pairs_trading") for k in combo}
    assert gridded == {"entry_z", "exit_z", "half_life_multiplier_max_hold"}
