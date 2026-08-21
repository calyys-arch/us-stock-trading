"""
Round-2 adversarial backtest/WFO engine audit
(backtests/reports/backtest_engine_audit_round2.md) — extends
tests/test_backtest_engine_audit.py with fresh ground-truth/mutation proofs
targeting code and questions that audit did NOT already cover: the
ambiguous-same-bar stop/target tie-break convention, cost-applied-exactly-
once-per-leg (no double counting across the slippage/commission split or
between the normal and 2x-stress paths), exact-boundary (>=/<=) behavior of
every gate function `python/backtest/optimize.py` exposes, and a sanity
re-derivation of the calibrated per-symbol half-spreads
(backtests/reports/calibrated_spreads.json) directly from the raw captured
L2 depth files this audit spot-checked by hand.

Every test either (a) computes the exact expected answer by construction
and asserts equality, or (b) plants a distinctive, detectable pattern and
asserts it had exactly the effect it should (no more, no less). A failure
here means a real bug, not a bad strategy — same discipline as the round-1
audit file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from python.backtest import intraday_engine as eng
from python.backtest.optimize import (
    check_drawdown_gate,
    check_has_trades_gate,
    check_min_trades_gate,
    check_profit_factor_gate,
)
from python.backtest.walk_forward import FoldResult, WFOResult


# ═════════════════════════════════════════════════════════════════════════
# 1. Ambiguous same-bar stop/target tie-break — priority item #2 in the
#    round-2 audit brief. When a single OHLC bar's range could plausibly
#    have hit BOTH the stop and the target within the same bar, confirm
#    _check_exit resolves this the SAME (conservative, stop-wins) way for
#    every direction — never a convention that happens to bias toward
#    stops for one side and targets for the other, which would be a
#    directional (not merely conservative) bias.
# ═════════════════════════════════════════════════════════════════════════

def _position(direction: str) -> eng.Position:
    entry = 100.0
    stop = 95.0 if direction == "long" else 105.0
    target = 110.0 if direction == "long" else 90.0
    return eng.Position(
        symbol="AAA", strategy="test", direction=direction,
        entry_time=pd.Timestamp("2024-01-02 09:31:00"), entry_price=entry,
        shares=100, stop_price=stop, target_price=target,
        entry_commission=1.0, signal_time=pd.Timestamp("2024-01-02 09:30:00"),
    )


@pytest.mark.parametrize("direction", ["long", "short"])
def test_ambiguous_bar_hitting_both_stop_and_target_always_resolves_to_stop(direction):
    """A wide bar whose [low, high] range engulfs BOTH the stop and the
    target must always be booked as a STOP exit, for both long and short —
    a single, uniform, conservative convention (never favorable-side-first
    for one direction and stop-first for the other, which would be a
    genuine directional bias rather than a uniformly conservative one)."""
    position = _position(direction)
    cfg = eng.IntradayBacktestConfig()
    # Bar range [85, 115] contains long's stop(95)/target(110) AND short's
    # stop(105)/target(90) — deliberately wide enough for both directions
    # so this one bar shape exercises the ambiguous case symmetrically.
    bar = pd.Series({"open": 100.0, "high": 115.0, "low": 85.0, "close": 100.0, "volume": 10_000.0})
    result = eng._check_exit(position, bar, elapsed_minutes=5.0, cfg=cfg)
    assert result is not None
    reason, price = result
    assert reason == "stop"
    assert price == position.stop_price


def test_ambiguous_bar_tie_break_is_identical_regardless_of_which_side_is_checked_first_in_code():
    """Sanity check that the uniform stop-wins convention is not an
    accident of long/short branch ordering: swap which side's stop/target
    would be hit first if the check order were reversed, and confirm the
    documented contract (stop checked before target, unconditionally)
    still produces a stop exit, not a target exit, in both cases."""
    cfg = eng.IntradayBacktestConfig()
    long_pos = _position("long")
    short_pos = _position("short")
    wide_bar = pd.Series({"open": 100.0, "high": 115.0, "low": 85.0, "close": 100.0, "volume": 10_000.0})

    long_reason, _ = eng._check_exit(long_pos, wide_bar, 5.0, cfg)
    short_reason, _ = eng._check_exit(short_pos, wide_bar, 5.0, cfg)
    assert long_reason == short_reason == "stop"


# ═════════════════════════════════════════════════════════════════════════
# 2. Cost applied EXACTLY once per leg — priority item #1's "trace one
#    single trade by hand" instruction, turned into a durable regression
#    test. Commission must be charged once at entry, once at exit, never
#    added a second time on top of the slippage-adjusted price; the 2x
#    slippage stress path must be a fully separate config, never leaking
#    into (or being leaked into by) the normal 1x run.
# ═════════════════════════════════════════════════════════════════════════

def _one_trade_session(breakout_price: float = 101.5) -> pd.DataFrame:
    idx = pd.date_range("2024-06-03 09:30", periods=20, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 100_000.0}, index=idx)
    bars.loc[bars.index[5], ["open", "high", "low", "close"]] = [100.0, breakout_price + 0.5, 100.0, breakout_price]
    for i in range(6, 12):
        bars.loc[bars.index[i], ["open", "high", "low", "close"]] = [breakout_price - 0.3 * (i - 5)] * 4
    return bars


def test_intraday_trade_cost_is_charged_exactly_once_per_leg():
    cfg = eng.IntradayBacktestConfig(half_spread_bps=5.0, impact_bps_per_participation=20.0,
                                      commission_per_share=0.005, min_commission=1.0)
    bars = _one_trade_session()
    report = eng.run_intraday_backtest({"AAA": bars}, "orb_vwap", {"or_minutes": 5, "vwap_side_filter": False}, cfg)
    assert report.trades, "need at least one trade for this test to mean anything"
    trade = report.trades[0]

    expected_commission_per_leg = max(cfg.commission_per_share * trade.shares, cfg.min_commission)
    # Costs must equal EXACTLY two commission legs — no third term (e.g. a
    # separately-line-itemed slippage cost) ever added on top; slippage is
    # ALREADY fully expressed inside entry_price/exit_price themselves.
    assert trade.costs == pytest.approx(2 * expected_commission_per_leg)

    # gross_pnl must be computed from the slippage-adjusted fill prices
    # (entry_price/exit_price), and net_pnl must equal gross_pnl minus
    # costs exactly once — not gross_pnl minus costs minus some second
    # slippage deduction.
    if trade.direction == "long":
        expected_gross = (trade.exit_price - trade.entry_price) * trade.shares
    else:
        expected_gross = (trade.entry_price - trade.exit_price) * trade.shares
    assert trade.gross_pnl == pytest.approx(expected_gross)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.costs)


def test_stress_slippage_multiplier_never_leaks_into_the_normal_1x_run():
    """build_intraday_backtest_fn (the path every WFO fold/IS-candidate/
    OOS-evaluation call goes through) must always use
    stress_slippage_multiplier == 1.0 unless the caller explicitly builds a
    stressed IntradayBacktestConfig — i.e. run_intraday_stress_test's 2x
    config can never accidentally become the config an ordinary gate
    evaluation uses, and vice versa."""
    from python.backtest.optimize import build_intraday_backtest_fn, run_intraday_stress_test

    bars = _one_trade_session()
    base_cfg = {"or_minutes": 5, "vwap_side_filter": False}
    fn = build_intraday_backtest_fn({"AAA": bars}, "orb_vwap", base_cfg)
    normal = fn(bars.index[0].to_pydatetime(), (bars.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime(), {})

    stressed = run_intraday_stress_test(
        {"AAA": bars}, "orb_vwap", base_cfg, {},
        bars.index[0].to_pydatetime(), (bars.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime(),
        stress_slippage_multiplier=2.0,
    )
    assert normal["n_trades"] == stressed["n_trades"] > 0
    # At 2x cost the same trades must be strictly worse (or, in the
    # vanishingly unlikely case slippage is zero, equal) — never BETTER,
    # and never identical when slippage is actually nonzero here.
    assert stressed["total_net_pnl"] < normal["total_net_pnl"]


# ═════════════════════════════════════════════════════════════════════════
# 3. Gate-threshold arithmetic — priority item #6. Exact-boundary (>=/<=)
#    behavior for every gate function optimize.py exposes, with especial
#    attention to a fold sitting EXACTLY at the threshold (must PASS, per
#    each gate's own docstring), not just comfortably above/below it.
# ═════════════════════════════════════════════════════════════════════════

def _wfo_result(oos_metrics_list: list[dict]) -> WFOResult:
    folds = [
        FoldResult(fold_idx=i, is_start="2020-01-01", is_end="2020-04-01",
                  oos_start="2020-04-01", oos_end="2020-05-01",
                  is_sharpe=0.0, oos_sharpe=0.0, oos_pass=True, oos_metrics=m)
        for i, m in enumerate(oos_metrics_list)
    ]
    return WFOResult(folds=folds, total_folds=len(folds), passing_folds=len(folds),
                     pass_ratio=1.0, decision="GO", config={})


def test_check_min_trades_gate_passes_at_exact_threshold():
    wfo = _wfo_result([{"n_trades": 100}, {"n_trades": 150}])
    assert check_min_trades_gate(wfo, min_trades=100) is True
    wfo_below = _wfo_result([{"n_trades": 99}, {"n_trades": 150}])
    assert check_min_trades_gate(wfo_below, min_trades=100) is False


def test_check_profit_factor_gate_passes_at_exact_threshold():
    wfo = _wfo_result([{"profit_factor": 1.3}, {"profit_factor": 2.0}])
    assert check_profit_factor_gate(wfo, min_profit_factor=1.3) is True
    wfo_below = _wfo_result([{"profit_factor": 1.2999999}, {"profit_factor": 2.0}])
    assert check_profit_factor_gate(wfo_below, min_profit_factor=1.3) is False


def test_check_drawdown_gate_passes_at_exact_threshold():
    wfo = _wfo_result([{"max_drawdown": -0.25}, {"max_drawdown": -0.10}])
    assert check_drawdown_gate(wfo, max_oos_drawdown=0.25) is True
    wfo_above = _wfo_result([{"max_drawdown": -0.2500001}, {"max_drawdown": -0.10}])
    assert check_drawdown_gate(wfo_above, max_oos_drawdown=0.25) is False


def test_check_has_trades_gate_requires_strictly_positive_trades_in_at_least_one_fold():
    wfo_zero = _wfo_result([{"n_trades": 0}, {"n_trades": 0}])
    assert check_has_trades_gate(wfo_zero) is False
    wfo_one = _wfo_result([{"n_trades": 0}, {"n_trades": 1}])
    assert check_has_trades_gate(wfo_one) is True


def test_check_min_trades_gate_false_on_empty_folds():
    empty = WFOResult(folds=[], total_folds=0, passing_folds=0, pass_ratio=0.0, decision="NO-GO", config={})
    assert check_min_trades_gate(empty, min_trades=100) is False
    assert check_has_trades_gate(empty) is False


# ═════════════════════════════════════════════════════════════════════════
# 4. Calibrated per-symbol spreads — priority item #1's "re-derive a couple
#    of spot-check numbers independently... yourself" instruction, turned
#    into a durable regression test: independently reconstruct the best-
#    bid/best-ask stream from the raw captured L2 depth files and confirm
#    it lands within a small tolerance of the checked-in calibration
#    output — this would catch a future re-run of the calibration script
#    silently drifting from what the raw depth data actually says.
# ═════════════════════════════════════════════════════════════════════════

_DEPTH_DIR = Path("data/depth")
_CALIBRATED_PATH = Path("backtests/reports/calibrated_spreads.json")


def _independent_median_half_spread_bps(symbol: str, day: str) -> float | None:
    """A deliberately SEPARATE re-implementation (not a call into
    scripts/calibrate_slippage_spreads.py) of the same best-bid/best-ask
    reconstruction, so this test cannot pass merely because it shares a
    bug with the script under test."""
    path = _DEPTH_DIR / symbol / f"{day}.jsonl"
    if not path.exists():
        return None
    best_bid = best_ask = None
    group_time = None
    samples: list[float] = []

    def flush() -> None:
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            return
        mid = (best_ask + best_bid) / 2.0
        samples.append(10_000.0 * (best_ask - best_bid) / 2.0 / mid)

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_time = row.get("time") or row.get("recorded_at")
            if group_time is not None and row_time != group_time:
                flush()
            group_time = row_time
            if row.get("position") != 0:
                continue
            side, op = row.get("side"), row.get("operation")
            price = float(row.get("price", 0.0) or 0.0)
            if op == 2:
                if side == 1:
                    best_bid = None
                elif side == 0:
                    best_ask = None
                continue
            if side == 1:
                best_bid = price
            elif side == 0:
                best_ask = price
    flush()
    if not samples:
        return None
    samples.sort()
    n = len(samples)
    return samples[n // 2] if n % 2 else (samples[n // 2 - 1] + samples[n // 2]) / 2.0


@pytest.mark.parametrize("symbol", ["AAPL", "STX", "NVDA"])
def test_calibrated_spreads_reproduce_independently_from_raw_depth(symbol):
    if not _CALIBRATED_PATH.exists():
        pytest.skip("calibrated_spreads.json not present in this checkout")
    calibrated = json.loads(_CALIBRATED_PATH.read_text(encoding="utf-8"))
    entry = calibrated.get("symbols", {}).get(symbol)
    if entry is None:
        pytest.skip(f"{symbol} not present in calibrated_spreads.json")
    days = entry["days"]
    if not all((_DEPTH_DIR / symbol / f"{d}.jsonl").exists() for d in days):
        pytest.skip(f"raw depth files for {symbol} not present in this checkout")

    per_day = [_independent_median_half_spread_bps(symbol, d) for d in days]
    per_day = [v for v in per_day if v is not None]
    assert per_day, "no independently-computable day for this symbol"
    independent_median = sorted(per_day)[len(per_day) // 2]

    # Not bit-for-bit identical (this test pools medians slightly
    # differently across days than the script's single-pass-over-all-days
    # median), but must land in the same ballpark — a real calibration bug
    # (wrong side swapped, unit error, off-by-10x, stale data) would blow
    # this tolerance by an order of magnitude, not a rounding amount.
    assert independent_median == pytest.approx(entry["median_bps"], rel=0.5)


# ═════════════════════════════════════════════════════════════════════════
# 5. `sufficient_sample_size` mis-wired as a HARD gate — priority item #7
#    ("confirm it is not being mis-wired anywhere as a hard gate"). Every
#    docstring that defines this check (param_guard.py's module docstring,
#    optimize.py::preflight_check's docstring) explicitly calls it a SOFT/
#    informational check: "warns (does not raise) ... a GO decision on a
#    too-short window is still evidence, just weaker evidence". scripts/
#    run_intraday_backtest.py and scripts/self_improve_loop.py both honor
#    this (print a WARNING, never fold it into a pass/fail gate dict). But
#    scripts/run_backtest.py's run_xsection()/run_pairs() DID fold it into
#    their `gates` dict, which feeds `overall_pass = all(gates.values())`
#    — silently turning a soft/informational check into a HARD rejection
#    for `xsection_mean_reversion`/`pairs_trading`'s health-check verdict.
#    This is exactly the systematic-over-pessimism failure mode item #7
#    asked to hunt for: a strategy with a genuinely sufficient walk-forward/
#    Monte-Carlo record could be marked NO-GO purely for having a technically
#    short raw sample window. FIXED: `sufficient_sample_size` is now
#    reported informationally (top-level key) but excluded from `gates`/
#    `overall_pass` in both functions, matching the documented contract.
#    (Note: re-deriving `backtests/reports/us_equity_health_check.md`'s one
#    checked-in real-data run confirms sample size was already PASSING there
#    — 1761 trading days >= the 1260 required for 5 free parameters — so
#    this bug did NOT flip that specific historical NO-GO verdict, which was
#    driven by `monte_carlo_p5_sharpe_nonneg` failing regardless. But it is a
#    genuine latent bug that could flip a FUTURE candidate's verdict, hence
#    this regression test.)
# ═════════════════════════════════════════════════════════════════════════


class _DemoArgs:
    """Minimal stand-in for argparse.Namespace: run_xsection/run_pairs only
    read `.demo` off `args` when `demo=True` (the `else` branch that reads
    `.start`/`.end`/`.pair_a`/`.pair_b`/`.refresh_data` is never reached)."""
    demo = True


def test_sufficient_sample_size_is_informational_not_a_hard_gate_in_run_xsection():
    from scripts.run_backtest import run_xsection

    result = run_xsection(_DemoArgs())
    assert "sufficient_sample_size" not in result["gates"], (
        "sufficient_sample_size is a documented SOFT/informational check "
        "(see param_guard.py / optimize.py::preflight_check docstrings) and "
        "must never be folded into the hard `gates`/`overall_pass` decision"
    )
    # Still surfaced for visibility, just not gating.
    assert "sufficient_sample_size" in result
    assert result["overall_pass"] == all(result["gates"].values())


def test_sufficient_sample_size_is_informational_not_a_hard_gate_in_run_pairs():
    from scripts.run_backtest import run_pairs

    result = run_pairs(_DemoArgs())
    assert "sufficient_sample_size" not in result["gates"], (
        "sufficient_sample_size is a documented SOFT/informational check "
        "and must never be folded into the hard `gates`/`overall_pass` decision"
    )
    assert "sufficient_sample_size" in result
    assert result["overall_pass"] == all(result["gates"].values())


def test_calibrated_spreads_are_plausible_for_mega_cap_liquid_names():
    """Sanity-check direction, not just magnitude: AAPL/MSFT/NVDA (this
    universe's most liquid mega-caps) must calibrate to a SMALL half-spread
    (sub-few-bps, tick-constrained), and must be smaller than at least one
    genuinely less-liquid name in the same universe — catching a future
    calibration bug that inflates costs uniformly (which would bias every
    signal toward systematic over-pessimism, the user's specific concern)
    or one that silently swaps a liquid/illiquid pair of symbols."""
    if not _CALIBRATED_PATH.exists():
        pytest.skip("calibrated_spreads.json not present in this checkout")
    symbols = json.loads(_CALIBRATED_PATH.read_text(encoding="utf-8")).get("symbols", {})
    for mega in ("AAPL", "MSFT", "NVDA"):
        if mega not in symbols:
            pytest.skip(f"{mega} not present in calibrated_spreads.json")
        assert symbols[mega]["median_bps"] < 3.0, (
            f"{mega} calibrated half-spread {symbols[mega]['median_bps']}bps looks implausibly "
            "wide for a mega-cap name — check for a units/parsing regression"
        )
    if "STX" in symbols and "AAPL" in symbols:
        assert symbols["STX"]["median_bps"] > symbols["AAPL"]["median_bps"], (
            "a less-liquid, higher-priced name (STX) calibrated TIGHTER than AAPL — "
            "check for a symbol mix-up in the calibration pipeline"
        )
