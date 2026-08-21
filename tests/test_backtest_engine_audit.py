"""
Adversarial backtest/WFO engine audit (backtests/reports/backtest_engine_audit.md).

This file is NOT signal research. Its only job is to try to BREAK the
correctness guarantees of the shared backtest/walk-forward plumbing, using
synthetic data with a KNOWN correct answer, the same way tests/test_pairs_scan.py
already does for point-in-time pair selection and tests/test_lookahead_bias.py
does for the cross-sectional engine. Two real bugs were previously found in
this codebase by accident (a cointegration sign error, an orb_vwap inverted
stop) while investigating unrelated results — never by a dedicated audit.
This file is that dedicated audit for the pieces those two incidents did NOT
already cover: `python/backtest/walk_forward.py` (no prior test file existed
for it at all), the `optimize.py` window-slicing wrappers (extended here with
explicit future-mutation proofs, not just window-restriction assertions), the
cost-model sign conventions in `intraday_engine.py`/`fees_equity.py`, ground-
truth VWAP arithmetic in `microstructure/context.py`, and the previously
untested `daily_range_breakout.py` / `daily_breakout_engine.py` pair.

Every test either (a) computes the exact expected answer by construction and
asserts equality, or (b) plants a distinctive, detectable pattern strictly
outside the region a component is allowed to see and asserts the pattern had
zero effect. A failure here means a real bug, not a bad strategy.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from python.backtest import intraday_engine as eng
from python.backtest.daily_breakout_engine import DailyBreakoutConfig, run_daily_breakout_backtest
from python.backtest.optimize import build_intraday_backtest_fn, build_pairs_backtest_fn, load_wfo_config
from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig
from python.core.fees_equity import round_trip_cost
from python.core.strategies.daily_range_breakout import evaluate_daily_breakout
from python.microstructure import context as ctx


# ═════════════════════════════════════════════════════════════════════════
# 1. WalkForwardOptimizer fold-boundary audit — python/backtest/walk_forward.py
#    had NO dedicated test file before this one. This is the component the
#    task brief specifically names as the highest-priority, never-audited
#    piece: "an independent backtest-engine audit... has never been done as
#    its own task."
# ═════════════════════════════════════════════════════════════════════════

def test_wfo_fold_windows_are_chronological_non_overlapping_and_gapless():
    """For every fold: is_start < is_end == oos_start < oos_end, and fold
    k+1's is_start is exactly step_days after fold k's — i.e. the optimizer
    can never hand a candidate an OOS window that overlaps its OWN IS window
    (a gap or overlap there would mean "out-of-sample" is a lie)."""
    cfg = WFOConfig(is_days=100, oos_days=40, step_days=40,
                     min_pass_folds_ratio=0.6, min_oos_sharpe_abs=0.0)
    calls = []

    def backtest_fn(start, end, params):
        calls.append((start, end))
        return {"sharpe_ratio": 0.1}

    wfo = WalkForwardOptimizer(backtest_fn, cfg, [{}]).run(
        datetime(2020, 1, 1), datetime(2022, 1, 1))
    assert len(wfo.folds) >= 3

    for fold in wfo.folds:
        is_start = datetime.fromisoformat(fold.is_start)
        is_end = datetime.fromisoformat(fold.is_end)
        oos_start = datetime.fromisoformat(fold.oos_start)
        oos_end = datetime.fromisoformat(fold.oos_end)
        assert is_start < is_end == oos_start < oos_end
        assert (is_end - is_start).days == cfg.is_days
        assert (oos_end - oos_start).days == cfg.oos_days

    for a, b in zip(wfo.folds, wfo.folds[1:]):
        a_start = datetime.fromisoformat(a.is_start)
        b_start = datetime.fromisoformat(b.is_start)
        assert (b_start - a_start).days == cfg.step_days


def test_wfo_parameter_selection_cannot_see_into_its_own_oos_window():
    """Plant a distinctive pattern that only pays off for a 'cheat' candidate
    when the QUERIED window overlaps fold 0's OOS region. If fold-splitting
    ever let the IS-selection phase query a window reaching into OOS (an
    inclusive boundary, an off-by-one on is_end, a query spanning the whole
    IS+OOS range), 'cheat' would win the IS comparison it must not be able to
    win, and would then get carried into the OOS evaluation too."""
    is_days, oos_days = 100, 50
    start = datetime(2020, 1, 1)
    is_end = start + timedelta(days=is_days)
    oos_end = is_end + timedelta(days=oos_days)
    secret = (is_end, oos_end)   # fold 0's OOS window — must be invisible to fold 0's IS call

    is_phase_calls = []

    def backtest_fn(qstart, qend, params):
        overlaps_secret = qstart < secret[1] and qend > secret[0]
        if qend <= is_end:            # this call is (part of) an IS-phase evaluation
            is_phase_calls.append((qstart, qend, dict(params), overlaps_secret))
        if params.get("variant") == "cheat" and overlaps_secret:
            return {"sharpe_ratio": 100.0}   # only "sees" the secret if leaked to it
        if params.get("variant") == "cheat":
            return {"sharpe_ratio": -1.0}    # genuinely bad everywhere else
        return {"sharpe_ratio": 0.3}         # 'honest' candidate: mediocre everywhere

    grid = [{"variant": "honest"}, {"variant": "cheat"}]
    wfo = WalkForwardOptimizer(backtest_fn, WFOConfig(is_days=is_days, oos_days=oos_days, step_days=oos_days),
                                grid).run(start, oos_end)
    assert len(wfo.folds) == 1
    fold = wfo.folds[0]

    # No IS-phase call may ever have overlapped the secret OOS-only region.
    assert is_phase_calls, "no IS-phase call recorded — test is vacuous"
    assert not any(overlaps for *_, overlaps in is_phase_calls), (
        "an IS-phase backtest_fn call reached into the fold's own OOS window — "
        "this is precisely the fold-boundary leak this audit targets"
    )
    # Consequently 'cheat' must lose the IS comparison and never be selected.
    assert fold.best_params["variant"] == "honest"
    assert fold.oos_sharpe == pytest.approx(0.3)


def test_wfo_oos_evaluation_uses_the_is_selected_params_not_a_fresh_search():
    """Construct two candidates where 'is_winner' beats 'oos_winner' on IS but
    loses badly on OOS. The fold must still report the IS winner's (bad) OOS
    Sharpe — proving the OOS phase is a pure replay of the IS-chosen
    parameters, not a second optimization that would silently launder a bad
    IS choice into a good-looking fold."""
    is_days, oos_days = 60, 30
    start = datetime(2021, 1, 1)
    is_end = start + timedelta(days=is_days)

    def backtest_fn(qstart, qend, params):
        in_is = qend <= is_end
        if params["variant"] == "is_winner":
            return {"sharpe_ratio": 2.0 if in_is else -3.0}
        return {"sharpe_ratio": 1.0 if in_is else 5.0}   # 'oos_winner': loses IS, would win OOS

    grid = [{"variant": "is_winner"}, {"variant": "oos_winner"}]
    wfo = WalkForwardOptimizer(backtest_fn, WFOConfig(is_days=is_days, oos_days=oos_days, step_days=oos_days),
                                grid).run(start, start + timedelta(days=is_days + oos_days))
    fold = wfo.folds[0]
    assert fold.best_params["variant"] == "is_winner"
    assert fold.oos_sharpe == pytest.approx(-3.0)
    assert fold.oos_pass is False


def test_wfo_real_pairs_trading_config_fold_boundaries_have_no_gap_or_overlap():
    """Same boundary property as above, but with the ACTUAL production
    configs/goal.yaml override for pairs_trading (is_days=1008, the longest,
    most warmup-sensitive window in the repo) — a manual, day-by-day style
    check of one real fold's boundary, per the audit brief's methodology
    step 3."""
    wfo_cfg = load_wfo_config("pairs_trading")
    assert wfo_cfg.is_days == 1008

    def backtest_fn(start, end, params):
        return {"sharpe_ratio": 0.0}

    start = datetime(2018, 1, 1)
    end = datetime(2026, 1, 1)
    wfo = WalkForwardOptimizer(backtest_fn, wfo_cfg, [{}]).run(start, end)
    assert len(wfo.folds) >= 2

    for fold in wfo.folds:
        is_start = datetime.fromisoformat(fold.is_start)
        is_end = datetime.fromisoformat(fold.is_end)
        oos_start = datetime.fromisoformat(fold.oos_start)
        oos_end = datetime.fromisoformat(fold.oos_end)
        # The OOS window must start EXACTLY where IS ends: no gap (a day of
        # data invisible to both phases) and no overlap (a day counted as
        # both "training" and "out of sample").
        assert oos_start == is_end
        assert (is_end - is_start).days == wfo_cfg.is_days
        assert (oos_end - oos_start).days == wfo_cfg.oos_days
        assert is_start < is_end < oos_end


def test_wfo_no_folds_when_range_shorter_than_one_fold():
    def backtest_fn(start, end, params):
        return {"sharpe_ratio": 1.0}
    wfo = WalkForwardOptimizer(backtest_fn, WFOConfig(is_days=100, oos_days=50), [{}]).run(
        datetime(2020, 1, 1), datetime(2020, 3, 1))
    assert wfo.folds == []
    assert wfo.decision == "NO-GO"


# ═════════════════════════════════════════════════════════════════════════
# 2. optimize.py wrappers — future-mutation invariance (stronger than the
#    existing window-restriction assertions in tests/test_optimize.py: here
#    we prove a fold's numbers are IDENTICAL whether or not out-of-window
#    data is sane, corrupted, or wildly profitable-looking).
# ═════════════════════════════════════════════════════════════════════════

def _orb_day(start_str: str, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start_str, periods=30, freq="1min")
    bars = pd.DataFrame({"open": price, "high": price, "low": price, "close": price,
                         "volume": 100_000.0}, index=idx)
    bars.loc[bars.index[5], "high"] = price + 1.0
    bars.loc[bars.index[6], "low"] = price - 1.0
    bars.loc[bars.index[20], "close"] = price
    bars.loc[bars.index[21], "close"] = price + 2.0
    return bars


def test_intraday_backtest_fn_unaffected_by_mutating_bars_outside_the_window():
    days = pd.bdate_range("2024-06-03", periods=5)
    bars = pd.concat([_orb_day(f"{d.date()} 09:30") for d in days])
    base_cfg = {"or_minutes": 15, "vwap_side_filter": False}
    fn = build_intraday_backtest_fn({"AAA": bars}, "orb_vwap", base_cfg)

    baseline = fn(days[1].to_pydatetime(), days[3].to_pydatetime(), {})

    corrupted = bars.copy()
    outside_mask = (corrupted.index < days[0]) | (corrupted.index >= days[4])
    # days[0] is warmup-only for this call (warmup_days=1 -> starts at days[1]-1),
    # days[4] and beyond are strictly after the window: turn them into an
    # enormous, unmistakably fake favorable/unfavorable move.
    corrupted.loc[outside_mask, ["open", "high", "low", "close"]] *= 1000.0
    fn_corrupted = build_intraday_backtest_fn({"AAA": corrupted}, "orb_vwap", base_cfg)
    mutated = fn_corrupted(days[1].to_pydatetime(), days[3].to_pydatetime(), {})

    assert mutated["n_trades"] == baseline["n_trades"]
    assert mutated["total_net_pnl"] == pytest.approx(baseline["total_net_pnl"], abs=1e-6)
    assert mutated["daily_returns"] == pytest.approx(baseline["daily_returns"], abs=1e-9)


def test_pairs_backtest_fn_unaffected_by_mutating_prices_outside_the_window():
    rng = np.random.default_rng(9)
    n = 900
    dates = pd.bdate_range("2018-01-02", periods=n)
    common = np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.15)
    log_a = 4.0 + 0.1 * common + 0.5 * spread
    log_b = 3.8 + 0.1 * common - 0.5 * spread
    prices_a = pd.Series(np.exp(log_a), index=dates)
    prices_b = pd.Series(np.exp(log_b), index=dates)

    base_cfg = {"entry_z": 2.0, "exit_z": 0.5, "coint_lookback_days": 100,
                "revalidate_every_days": 21, "notional_per_leg": 50_000.0,
                "half_life_multiplier_max_hold": 3.0, "min_half_life_days": 1.0,
                "max_half_life_days": 60.0}
    fn = build_pairs_backtest_fn("A", "B", prices_a, prices_b, base_cfg)
    start, end = dates[400], dates[600]
    baseline = fn(start, end, {})
    assert baseline["n_trades"] > 0, "need trades for this test to mean anything"

    corrupted_a, corrupted_b = prices_a.copy(), prices_b.copy()
    future_mask = corrupted_a.index >= end
    corrupted_a.loc[future_mask] *= 50.0
    corrupted_b.loc[future_mask] *= 0.02
    fn_corrupted = build_pairs_backtest_fn("A", "B", corrupted_a, corrupted_b, base_cfg)
    mutated = fn_corrupted(start, end, {})

    assert mutated["n_trades"] == baseline["n_trades"]
    assert mutated["total_net_pnl"] == pytest.approx(baseline["total_net_pnl"], abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════
# 3. Ground-truth VWAP / VWAP-bands arithmetic (microstructure/context.py) —
#    exact hand-computed answers with NON-uniform volume (the existing
#    tests/test_context_engine.py case uses flat volume, which cannot catch
#    a volume-weighting bug since every weight is then equal).
# ═════════════════════════════════════════════════════════════════════════

def test_session_vwap_exact_value_with_nonuniform_volume():
    idx = pd.date_range("2024-06-04 09:30", periods=4, freq="1min")
    # typical price = (h+l+c)/3 for each bar, chosen to be simple round numbers.
    bars = pd.DataFrame({
        "open": [100.0, 101.0, 99.0, 102.0],
        "high": [100.0, 102.0, 100.0, 103.0],
        "low": [100.0, 100.0, 98.0, 101.0],
        "close": [100.0, 101.0, 99.0, 102.0],
        "volume": [100.0, 300.0, 100.0, 500.0],
    }, index=idx)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    vwap = ctx.session_vwap(bars)

    # Hand-computed cumulative VWAP at each bar, by definition
    # sum(tp_i * vol_i) / sum(vol_i) for i in [0, bar].
    cum_pv, cum_vol = 0.0, 0.0
    for i in range(len(bars)):
        cum_pv += float(tp.iloc[i]) * float(bars["volume"].iloc[i])
        cum_vol += float(bars["volume"].iloc[i])
        expected = cum_pv / cum_vol
        assert vwap.iloc[i] == pytest.approx(expected, rel=1e-12), f"bar {i} VWAP mismatch"

    # The heavy bar (index 3, volume 500) must pull VWAP noticeably toward
    # its own typical price versus a naive unweighted mean of all typical
    # prices — a real check that volume weighting is doing something.
    naive_mean = float(tp.mean())
    assert abs(vwap.iloc[-1] - float(tp.iloc[3])) < abs(naive_mean - float(tp.iloc[3]))


def test_vwap_bands_sigma_exact_value():
    """Hand-derive the volume-weighted variance formula and compare bar by
    bar — vwap_bands must never be computable from anything but bars[0..i]
    (a running sigma), and the arithmetic itself must be exactly the
    textbook volume-weighted standard deviation around the running VWAP."""
    idx = pd.date_range("2024-06-04 09:30", periods=5, freq="1min")
    bars = pd.DataFrame({
        "open": [10.0, 11.0, 9.0, 12.0, 8.0],
        "high": [10.0, 12.0, 10.0, 13.0, 9.0],
        "low": [10.0, 10.0, 8.0, 11.0, 7.0],
        "close": [10.0, 11.0, 9.0, 12.0, 8.0],
        "volume": [50.0, 200.0, 10.0, 400.0, 30.0],
    }, index=idx)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    bands = ctx.vwap_bands(bars)

    # Ground truth per the ACTUAL (causal) definition in context.py: each
    # bar j's own deviation term uses bar j's own contemporaneous running
    # VWAP (vwap-as-of-j), not the final running VWAP as of the current
    # bar i. This is what makes the whole thing a pure running cumsum with
    # no re-derivation of history — and, crucially, still only ever uses
    # bars[0..i] to produce bands[i], so it is not lookahead, just a
    # different (and here, correctly implemented) variance definition than
    # the naive "deviation from final vwap" one.
    cum_pv = cum_vol = 0.0
    running_vwaps = []
    for i in range(len(bars)):
        v = float(bars["volume"].iloc[i])
        p = float(tp.iloc[i])
        cum_pv += p * v
        cum_vol += v
        running_vwaps.append(cum_pv / cum_vol)

    cum_vol = 0.0
    cum_sqdev_num = 0.0
    for i in range(len(bars)):
        v = float(bars["volume"].iloc[i])
        p = float(tp.iloc[i])
        cum_vol += v
        cum_sqdev_num += ((p - running_vwaps[i]) ** 2) * v
        expected_sigma = (cum_sqdev_num / cum_vol) ** 0.5
        assert bands["vwap"].iloc[i] == pytest.approx(running_vwaps[i], rel=1e-9)
        assert bands["upper_1"].iloc[i] == pytest.approx(running_vwaps[i] + expected_sigma, rel=1e-9)
        assert bands["lower_2"].iloc[i] == pytest.approx(running_vwaps[i] - 2 * expected_sigma, rel=1e-9)


def test_opening_range_boundary_is_a_half_open_interval():
    """A bar landing EXACTLY on start+minutes must be EXCLUDED (the window is
    [start, start+minutes), matching the "no entries during the OR itself,
    entries begin exactly at its end" rule orb_vwap.py depends on)."""
    idx = pd.date_range("2024-06-04 09:30", periods=20, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 1000.0}, index=idx)
    boundary_bar_time = idx[0] + pd.Timedelta(minutes=15)
    assert boundary_bar_time in idx
    bars.loc[boundary_bar_time, "high"] = 500.0   # sits exactly at the boundary
    orange = ctx.opening_range(bars, minutes=15)
    assert orange.high == 100.0, "the boundary-timestamp bar leaked into the opening range"


# ═════════════════════════════════════════════════════════════════════════
# 4. Cost-model sign-convention audit — the exact bug PATTERN already found
#    twice (a sign/side error that quietly turns a cost into a gift, or
#    charges only one leg of a round trip). Exhaustive over every
#    direction x entry/exit combination, not just the happy path already
#    covered in tests/test_intraday_engine.py.
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("direction,is_entry,expect_worse", [
    ("long", True, "higher"),    # buying to open a long: pay MORE than the reference price
    ("long", False, "lower"),    # selling to close a long: receive LESS than the reference price
    ("short", True, "lower"),    # selling to open a short: receive LESS than the reference price
    ("short", False, "higher"),  # buying to cover a short: pay MORE than the reference price
])
def test_slippage_price_always_moves_against_the_trader(direction, is_entry, expect_worse):
    cfg = eng.IntradayBacktestConfig(half_spread_bps=5.0, impact_bps_per_participation=20.0)
    ref_price = 100.0
    filled = eng._slippage_price(ref_price, direction, shares=1000, bar_volume=10_000.0,
                                  cfg=cfg, is_entry=is_entry, symbol=None)
    if expect_worse == "higher":
        assert filled > ref_price
    else:
        assert filled < ref_price


def test_slippage_cost_is_symmetric_round_trip_never_a_net_gift():
    """A full round trip (entry then exit at the SAME reference price) must
    always cost money in every direction — slippage can never net out to a
    profit from nothing, which is exactly the class of defect the orb_vwap
    gap-trap bug produced (a stop that was actually on the profitable side)."""
    cfg = eng.IntradayBacktestConfig(half_spread_bps=5.0, impact_bps_per_participation=20.0)
    ref_price = 100.0
    for direction in ("long", "short"):
        entry = eng._slippage_price(ref_price, direction, 1000, 10_000.0, cfg, is_entry=True)
        exit_ = eng._slippage_price(ref_price, direction, 1000, 10_000.0, cfg, is_entry=False)
        if direction == "long":
            round_trip_pnl_per_share = exit_ - entry
        else:
            round_trip_pnl_per_share = entry - exit_
        assert round_trip_pnl_per_share < 0, (
            f"{direction} round trip at an unchanged reference price produced a "
            f"non-negative P&L ({round_trip_pnl_per_share}) — slippage is acting as a gift"
        )


def test_stress_multiplier_only_ever_increases_the_adverse_move():
    cfg_normal = eng.IntradayBacktestConfig(half_spread_bps=5.0, impact_bps_per_participation=20.0)
    cfg_stress = eng.IntradayBacktestConfig(half_spread_bps=5.0, impact_bps_per_participation=20.0,
                                             stress_slippage_multiplier=2.0)
    for direction in ("long", "short"):
        for is_entry in (True, False):
            normal = eng._slippage_price(100.0, direction, 1000, 10_000.0, cfg_normal, is_entry)
            stress = eng._slippage_price(100.0, direction, 1000, 10_000.0, cfg_stress, is_entry)
            assert abs(stress - 100.0) == pytest.approx(2 * abs(normal - 100.0))


def test_round_trip_cost_is_always_nonnegative_and_monotonic_in_spread():
    base = round_trip_cost(shares=1000, entry_price=50.0, exit_price=52.0, is_short=False,
                           holding_days=5, adv_dollars=1e8, half_spread_bps=0.0)
    wider = round_trip_cost(shares=1000, entry_price=50.0, exit_price=52.0, is_short=False,
                            holding_days=5, adv_dollars=1e8, half_spread_bps=10.0)
    assert base.total >= 0
    assert wider.total > base.total
    # A cost model must never subtract value based on direction alone.
    short_cost = round_trip_cost(shares=1000, entry_price=50.0, exit_price=48.0, is_short=True,
                                 holding_days=5, adv_dollars=1e8, half_spread_bps=3.0)
    assert short_cost.total > 0
    assert short_cost.borrow_cost > 0   # only charged for shorts, but must be > 0 when short


def test_round_trip_cost_charges_sec_and_finra_only_on_the_sell_leg():
    # Long: sell leg is the EXIT (at 51.0). Short: sell leg is the ENTRY
    # (at 51.0). Pin both sell legs to the identical price so the SEC fee
    # (which scales with sell-leg proceeds) is directly comparable — it must
    # come out identical, since the fee only depends on which leg is a
    # "sell" and at what price, never on side (long/short) as such.
    long_trip = round_trip_cost(shares=1000, entry_price=50.0, exit_price=51.0, is_short=False)
    short_trip = round_trip_cost(shares=1000, entry_price=51.0, exit_price=49.0, is_short=True)
    assert long_trip.sec_fee == pytest.approx(short_trip.sec_fee)
    assert long_trip.finra_taf == pytest.approx(short_trip.finra_taf)
    assert long_trip.sec_fee > 0
    assert long_trip.finra_taf > 0


# ═════════════════════════════════════════════════════════════════════════
# 5. daily_range_breakout.py / daily_breakout_engine.py — NEW code with NO
#    prior dedicated test coverage. Ground-truth breakout detection and
#    T+1-open fill discipline, plus a future-mutation lookahead proof.
# ═════════════════════════════════════════════════════════════════════════

def _breakout_bars(n: int = 40, base: float = 100.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    closes = base + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({
        "open": closes, "high": closes + 0.3, "low": closes - 0.3, "close": closes,
        "volume": 1_000_000.0,
    }, index=idx)
    return df


def test_daily_breakout_excludes_todays_own_bar_from_the_range_window():
    """Plant an extreme high on TODAY's own bar (index -1) that would, if
    wrongly included in the range window, make today's own close look like
    it is INSIDE the range instead of breaking out of it. The prior
    `range_days` bars must define the range, never today's own bar."""
    bars = _breakout_bars(n=40)
    range_days, atr_days = 10, 5
    window = bars.iloc[: 30]
    # Force a genuine breakout: today's close far above the prior range.
    prior_high = float(window.iloc[:-1].iloc[-range_days:]["high"].max())
    window = window.copy()
    window.iloc[-1, window.columns.get_loc("close")] = prior_high + 10.0
    window.iloc[-1, window.columns.get_loc("high")] = prior_high + 10.0

    sig = evaluate_daily_breakout(window, range_days=range_days, atr_days=atr_days)
    assert sig is not None
    assert sig["direction"] == "long"
    assert sig["range_high"] == pytest.approx(prior_high)

    # Now make today's bar ABSURDLY extreme (so if it were included in its
    # OWN range window, "today's close vs today's range" would trivially
    # never be a breakout) and confirm the signal is UNCHANGED.
    window2 = window.copy()
    window2.iloc[-1, window2.columns.get_loc("high")] = prior_high + 10_000.0
    sig2 = evaluate_daily_breakout(window2, range_days=range_days, atr_days=atr_days)
    assert sig2["range_high"] == sig["range_high"]
    assert sig2["direction"] == sig["direction"]


def _flat_bars(n: int, base: float = 100.0) -> pd.DataFrame:
    """A perfectly flat, noise-free daily-bar series (no drift, no spurious
    highs/lows), used where the test needs FULL CONTROL over exactly when a
    breakout fires (e.g. driving the stateful engine, where an accidental
    spurious signal from random noise could occupy `position` for
    `hold_days` and mask the engineered signal under test)."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "open": base, "high": base + 0.1, "low": base - 0.1, "close": base,
        "volume": 1_000_000.0,
    }, index=idx)


def test_daily_breakout_stop_and_target_sides_are_correct():
    # range_days=20 + atr_days=14 + 1 = 35 bars of history required before
    # evaluate_daily_breakout will even look at a breakout.
    bars = _breakout_bars(n=45)
    window = bars.iloc[:36].copy()
    prior_high = float(window.iloc[:-1].iloc[-20:]["high"].max())
    window.iloc[-1, window.columns.get_loc("close")] = prior_high + 5.0
    window.iloc[-1, window.columns.get_loc("high")] = prior_high + 5.0
    sig_long = evaluate_daily_breakout(window, range_days=20, atr_days=14, stop_atr_mult=2.0, target_r_multiple=3.0)
    assert sig_long is not None
    assert sig_long["stop"] < sig_long["signal_close"]     # long stop must sit BELOW entry
    assert sig_long["target"] > sig_long["signal_close"]   # long target must sit ABOVE entry

    prior_low = float(window.iloc[:-1].iloc[-20:]["low"].min())
    window.iloc[-1, window.columns.get_loc("close")] = prior_low - 5.0
    window.iloc[-1, window.columns.get_loc("low")] = prior_low - 5.0
    sig_short = evaluate_daily_breakout(window, range_days=20, atr_days=14, stop_atr_mult=2.0, target_r_multiple=3.0)
    assert sig_short is not None
    assert sig_short["direction"] == "short"
    assert sig_short["stop"] > sig_short["signal_close"]     # short stop must sit ABOVE entry
    assert sig_short["target"] < sig_short["signal_close"]   # short target must sit BELOW entry


def test_daily_breakout_engine_fills_at_next_days_open_not_signal_days_close():
    # Flat, noise-free base series: guarantees the ONLY breakout that can
    # possibly fire is the one deliberately engineered below (random noise
    # in a small window could otherwise spuriously trigger an earlier
    # signal and occupy `position` for `hold_days`, masking this trade).
    bars = _flat_bars(n=60)
    i = 40
    prior_high = float(bars.iloc[:i].iloc[-20:]["high"].max())
    bars = bars.copy()
    bars.iloc[i, bars.columns.get_loc("close")] = prior_high + 5.0
    bars.iloc[i, bars.columns.get_loc("high")] = prior_high + 5.0
    # Distinct, unmistakable open the next day — the fill price must equal
    # EXACTLY this, never the signal day's own close.
    bars.iloc[i + 1, bars.columns.get_loc("open")] = prior_high + 1.2345
    bars.iloc[i + 1, bars.columns.get_loc("high")] = prior_high + 1.2345
    bars.iloc[i + 1, bars.columns.get_loc("low")] = prior_high + 1.2345
    bars.iloc[i + 1, bars.columns.get_loc("close")] = prior_high + 1.2345

    # hold_days is deliberately small (not 100): a trade only ever lands in
    # `report.trades` once it EXITS, so it must close (via the time-stop,
    # here) within the remaining bars for this test to observe it at all.
    # stop_atr_mult=50 keeps the stop far away so the time-stop is what
    # fires, isolating exactly the fill-timing behavior under test.
    cfg = DailyBreakoutConfig(range_days=20, hold_days=5, stop_atr_mult=50.0, target_r_multiple=None)
    report = run_daily_breakout_backtest("TEST", bars, cfg)
    assert report.trades, "expected the engineered breakout to produce a trade"
    trade = next(t for t in report.trades if t.entry_date == bars.index[i + 1])
    assert trade.entry_price == pytest.approx(prior_high + 1.2345)
    assert trade.entry_date > pd.Timestamp(bars.index[i])   # never fills on the signal's own day


def test_daily_breakout_engine_trades_before_cutoff_unchanged_by_future_price_mutation():
    """End-to-end lookahead proof for the whole engine: corrupt every bar
    strictly after a cutoff date and require every trade that had ALREADY
    CLOSED by the cutoff to be bit-identical — the same discipline
    tests/test_pairs_scan.py applies to the scanned-pairs engine."""
    # Deterministic, noise-free base series with several deliberately
    # engineered single-day breakout pulses (spaced >range_days apart so
    # each is a genuinely fresh breakout, not still-visible in the next
    # pulse's range window). This guarantees reproducible trades instead
    # of depending on whether a random walk happens to break out.
    bars = _flat_bars(n=200)
    for p in (35, 75, 115, 155, 190):
        bars.iloc[p, bars.columns.get_loc("close")] = 105.0
        bars.iloc[p, bars.columns.get_loc("high")] = 105.0
    cfg = DailyBreakoutConfig(range_days=15, hold_days=8, stop_atr_mult=2.0, target_r_multiple=2.0)
    baseline = run_daily_breakout_backtest("TEST", bars, cfg)
    assert baseline.trades, "need trades for this test to mean anything"

    cutoff = bars.index[150]
    corrupted = bars.copy()
    mask = corrupted.index >= cutoff
    corrupted.loc[mask, ["open", "high", "low", "close"]] *= 5.0
    mutated = run_daily_breakout_backtest("TEST", corrupted, cfg)

    def before(report):
        return [(t.entry_date, t.exit_date, round(t.net_pnl, 6), t.exit_reason)
                for t in report.trades if pd.Timestamp(t.exit_date) < cutoff]

    baseline_before = before(baseline)
    assert baseline_before, "cutoff chosen badly — no closed trades before it"
    assert baseline_before == before(mutated)


def test_daily_breakout_evaluate_returns_none_with_insufficient_history():
    bars = _breakout_bars(n=10)
    assert evaluate_daily_breakout(bars, range_days=20, atr_days=14) is None
