"""
trap_detector tests — constructed OHLCV bars with known shapes.

The single most important invariant: MISSING EVIDENCE IS None (UNKNOWN),
never 0 ("nothing suspicious") — for every sub-score and for the combined
score. The rest verifies each heuristic fires on its textbook pattern and
stays quiet on a plain trending bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.signals.trap_detector import (
    assess_signal_day,
    combine_trap_score,
    dark_pool_internalization_score,
    false_breakout_score,
    marking_the_close_score,
    order_book_churn_score,
    order_flow_imbalance_score,
    pinging_score,
    print_lag_score,
    short_distort_score,
    stop_hunt_score,
)


def _flat_ohlcv(n_days: int = 40, price: float = 100.0) -> pd.DataFrame:
    """Quiet 20-day range around `price` with mild noise and stable volume."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    close = price + rng.normal(0, 0.3, n_days)
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n_days),
        "close": close,
        "volume": np.full(n_days, 1_000_000.0),
    }, index=dates)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.4
    df["low"] = df[["open", "close"]].min(axis=1) - 0.4
    return df


def _set_bar(df: pd.DataFrame, day, o, h, l, c, volume=1_000_000.0) -> None:
    df.loc[day, ["open", "high", "low", "close", "volume"]] = [o, h, l, c, volume]


# ── false breakout ───────────────────────────────────────────────────────────

def test_bull_trap_scores_high():
    df = _flat_ohlcv()
    day = df.index[-1]
    # prior 20-day high ~100.7; pierce to 104 on 3x volume, close back at 99.
    _set_bar(df, day, o=100.0, h=104.0, l=98.5, c=99.0, volume=3_000_000.0)
    score = false_breakout_score(df, day)
    assert score is not None and score > 0.5


def test_plain_inside_bar_scores_zero():
    df = _flat_ohlcv()
    day = df.index[-1]
    _set_bar(df, day, o=100.0, h=100.3, l=99.7, c=100.1)
    assert false_breakout_score(df, day) == 0.0


def test_genuine_breakout_that_holds_scores_zero():
    df = _flat_ohlcv()
    day = df.index[-1]
    _set_bar(df, day, o=100.0, h=104.0, l=99.8, c=103.8)  # closes ABOVE the range
    assert false_breakout_score(df, day) == 0.0


def test_missing_ohlc_columns_returns_none():
    df = _flat_ohlcv().drop(columns=["high", "low", "volume"])
    assert false_breakout_score(df, df.index[-1]) is None
    assert stop_hunt_score(df, df.index[-1]) is None


def test_insufficient_history_returns_none():
    df = _flat_ohlcv(n_days=10)
    assert false_breakout_score(df, df.index[-1]) is None


# ── stop hunt ────────────────────────────────────────────────────────────────

def test_stop_run_wick_scores_high():
    df = _flat_ohlcv()
    day = df.index[-1]
    # sweep well below the prior 20-day low (~99.0) then close back up:
    _set_bar(df, day, o=100.0, h=100.4, l=96.0, c=100.2, volume=2_500_000.0)
    score = stop_hunt_score(df, day)
    assert score is not None and score > 0.5


def test_ordinary_down_day_scores_zero():
    df = _flat_ohlcv()
    day = df.index[-1]
    _set_bar(df, day, o=100.2, h=100.4, l=99.8, c=99.9)
    assert stop_hunt_score(df, day) == 0.0


# ── marking the close / order book churn (microstructure) ──────────────────

def test_marking_the_close_needs_tick_data():
    assert marking_the_close_score(None, pd.Timestamp("2024-06-03 16:00")) is None


def test_marking_the_close_flags_late_concentration():
    session_close = pd.Timestamp("2024-06-03 16:00")
    times = pd.date_range("2024-06-03 09:30", "2024-06-03 15:49", freq="1min")
    early = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0})
    late_times = pd.date_range("2024-06-03 15:51", "2024-06-03 15:59", freq="1min")
    late = pd.DataFrame({"time": late_times,
                         "price": np.linspace(100.0, 103.0, len(late_times)),
                         "size": 20_000.0})
    trades = pd.concat([early, late], ignore_index=True)
    score = marking_the_close_score(trades, session_close)
    assert score is not None and score > 0.5

    uniform = early.copy()
    assert marking_the_close_score(uniform, session_close) is not None
    assert marking_the_close_score(uniform, session_close) < 0.2


def test_order_book_churn_needs_depth_data():
    assert order_book_churn_score(None, n_trades=100) is None


def test_order_book_churn_flags_cancel_heavy_off_touch_book():
    rng = np.random.default_rng(4)
    n = 2000
    churny = pd.DataFrame({
        "operation": rng.choice([0, 2], n),          # only inserts/deletes, no trades keeping up
        "side": np.ones(n),                            # all one side
        "position": rng.integers(3, 10, n),            # away from the touch
        "size": np.full(n, 5000.0),
    })
    score = churny_score = order_book_churn_score(churny, n_trades=10)
    assert score is not None and churny_score > 0.6

    calm = pd.DataFrame({
        "operation": rng.choice([0, 1, 2], n),
        "side": rng.choice([0, 1], n),
        "position": rng.integers(0, 3, n),             # at/near the touch
        "size": rng.uniform(100, 200, n),
    })
    calm_score = order_book_churn_score(calm, n_trades=1000)
    assert calm_score is not None and calm_score < churny_score


# ── pinging (Tier 1) ─────────────────────────────────────────────────────────

def test_pinging_needs_tick_data():
    assert pinging_score(None) is None


def test_pinging_flags_burst_of_small_still_prints():
    rng = np.random.default_rng(7)
    n = 500
    # Baseline session: mostly round-lot-plus prints, normal price drift.
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="30s")
    prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    baseline = pd.DataFrame({"time": times, "price": prices, "size": rng.uniform(150, 500, n)})

    # Inject a tight burst of 12 tiny (<=100 share) prints within 60s, price
    # essentially frozen — the pinging footprint.
    burst_times = pd.date_range("2024-06-03 11:00:00", periods=12, freq="5s")
    burst = pd.DataFrame({"time": burst_times, "price": 105.0 + rng.normal(0, 0.001, 12), "size": 50.0})

    trades = pd.concat([baseline, burst], ignore_index=True)
    score = pinging_score(trades)
    assert score is not None and score > 0.5


def test_pinging_quiet_session_scores_zero():
    rng = np.random.default_rng(3)
    n = 300
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    prices = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    trades = pd.DataFrame({"time": times, "price": prices, "size": rng.uniform(200, 800, n)})
    assert pinging_score(trades) == 0.0


# ── dark pool internalization (Tier 1) ──────────────────────────────────────

def test_dark_pool_internalization_needs_exchange_column():
    trades_no_exchange = pd.DataFrame({"time": pd.date_range("2024-06-03", periods=60, freq="1min"),
                                        "price": 100.0, "size": 100.0})
    assert dark_pool_internalization_score(None) is None
    assert dark_pool_internalization_score(trades_no_exchange) is None


def test_dark_pool_internalization_flags_high_off_exchange_share():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    # 70% off-exchange (TRF) prints vs the ~40% market-wide baseline.
    exchanges = ["TRF"] * 140 + ["NASDAQ"] * 60
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0, "exchange": exchanges})
    score = dark_pool_internalization_score(trades)
    assert score is not None and score > 0.3


def test_dark_pool_internalization_baseline_share_scores_zero():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    exchanges = ["TRF"] * 80 + ["NASDAQ"] * 120  # 40% off-exchange == baseline
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0, "exchange": exchanges})
    assert dark_pool_internalization_score(trades) == 0.0


# ── print lag (late/out-of-sequence prints) ─────────────────────────────────

def test_print_lag_needs_special_conditions_column():
    trades_no_conditions = pd.DataFrame({"time": pd.date_range("2024-06-03", periods=60, freq="1min"),
                                          "price": 100.0, "size": 100.0})
    assert print_lag_score(None) is None
    assert print_lag_score(trades_no_conditions) is None


def test_print_lag_flags_high_late_print_share():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    # 30% late/out-of-sequence codes vs the 5% baseline.
    conditions = ["L"] * 60 + [""] * 140
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0, "special_conditions": conditions})
    score = print_lag_score(trades)
    assert score is not None and score > 0.2


def test_print_lag_baseline_share_scores_zero():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    conditions = ["Z"] * 10 + [""] * 190  # 5% == baseline
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0, "special_conditions": conditions})
    assert print_lag_score(trades) == 0.0


def test_print_lag_quiet_session_scores_zero():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0, "special_conditions": [""] * n})
    assert print_lag_score(trades) == 0.0


def test_print_lag_handles_multi_code_strings():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    # Some prints carry multiple space-separated condition codes; only ONE
    # needs to be a late/OOS code for that print to count.
    conditions = ["F L"] * 60 + [""] * 140
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 100.0, "special_conditions": conditions})
    score = print_lag_score(trades)
    assert score is not None and score > 0.2


# ── order flow imbalance (tick rule) ────────────────────────────────────────

def test_order_flow_imbalance_needs_tick_data():
    trades_too_short = pd.DataFrame({"time": pd.date_range("2024-06-03", periods=10, freq="1min"),
                                      "price": 100.0, "size": 100.0})
    assert order_flow_imbalance_score(None) is None
    assert order_flow_imbalance_score(trades_too_short) is None


def test_order_flow_imbalance_missing_columns_returns_none():
    trades_no_price = pd.DataFrame({"time": pd.date_range("2024-06-03", periods=60, freq="1min"),
                                     "size": 100.0})
    assert order_flow_imbalance_score(trades_no_price) is None


def test_order_flow_imbalance_flags_one_sided_uptick_flow():
    n = 100
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    # Price ticks up on every single print (all buy-initiated under the tick
    # rule) with large size -> aggressor volume should be ~100% buy-side.
    prices = 100.0 + np.arange(n) * 0.01
    trades = pd.DataFrame({"time": times, "price": prices, "size": 1000.0})
    score = order_flow_imbalance_score(trades)
    assert score is not None and score > 0.9


def test_order_flow_imbalance_balanced_alternating_ticks_scores_low():
    n = 200
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    # Alternating up/down ticks of EQUAL size -> buy volume ~= sell volume.
    prices = 100.0 + np.array([0.05 if i % 2 == 0 else -0.05 for i in range(n)]).cumsum()
    trades = pd.DataFrame({"time": times, "price": prices, "size": 500.0})
    score = order_flow_imbalance_score(trades)
    assert score is not None and score < 0.2


def test_order_flow_imbalance_all_flat_prices_scores_zero_not_none():
    n = 60
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    trades = pd.DataFrame({"time": times, "price": 100.0, "size": 200.0})  # no ticks at all
    assert order_flow_imbalance_score(trades) == 0.0


def test_order_flow_imbalance_zero_tick_inherits_last_direction():
    n = 60
    times = pd.date_range("2024-06-03 09:30", periods=n, freq="1min")
    # One up-tick to seed a buy direction, then a long run of unchanged
    # prints (zero-ticks) that must inherit "buy" rather than go unclassified.
    prices = [100.0, 100.5] + [100.5] * (n - 2)
    sizes = [100.0, 100.0] + [500.0] * (n - 2)
    trades = pd.DataFrame({"time": times, "price": prices, "size": sizes})
    score = order_flow_imbalance_score(trades)
    assert score is not None and score > 0.9  # nearly all classified volume ends up buy-side


# ── short & distort ──────────────────────────────────────────────────────────

def _crash_frame() -> tuple[pd.DataFrame, pd.Timestamp]:
    df = _flat_ohlcv()
    day = df.index[-1]
    _set_bar(df, day, o=99.5, h=99.6, l=88.0, c=88.5, volume=4_000_000.0)  # ~-11.5%
    return df, day


def test_short_distort_unknown_without_evidence():
    df, day = _crash_frame()
    assert short_distort_score(df, day, has_news=None, has_8k=None) is None
    assert short_distort_score(df, day, has_news=True, has_8k=None) is None


def test_short_distort_news_without_filing_scores_highest():
    df, day = _crash_frame()
    news_no_filing = short_distort_score(df, day, has_news=True, has_8k=False)
    filing = short_distort_score(df, day, has_news=True, has_8k=True)
    neither = short_distort_score(df, day, has_news=False, has_8k=False)
    assert news_no_filing > neither > filing
    assert news_no_filing > 0.4
    assert filing <= 0.1


def test_short_distort_quiet_day_scores_zero():
    df = _flat_ohlcv()
    assert short_distort_score(df, df.index[-1], has_news=True, has_8k=False) == 0.0


# ── aggregation ──────────────────────────────────────────────────────────────

def test_combine_all_none_is_none_not_zero():
    score, unavailable = combine_trap_score(
        {"a": None, "b": None})
    assert score is None
    assert unavailable == ["a", "b"]


def test_combine_averages_available_only():
    score, unavailable = combine_trap_score({"a": 0.8, "b": 0.4, "c": None})
    assert score == pytest.approx(0.6)
    assert unavailable == ["c"]


def test_assess_signal_day_end_to_end():
    df = _flat_ohlcv()
    day = df.index[-1]
    _set_bar(df, day, o=100.0, h=104.0, l=98.5, c=99.0, volume=3_000_000.0)
    assessment = assess_signal_day(
        "TEST", day, df, has_news=True, has_8k=False,
        event_flags={"signal": "unit test", "earnings_within_1d": False},
    )
    assert assessment.symbol == "TEST"
    assert assessment.components["false_breakout"] > 0.5
    assert "marking_the_close" in assessment.unavailable
    assert "order_book_churn" in assessment.unavailable
    assert "pinging" in assessment.unavailable
    assert "dark_pool_internalization" in assessment.unavailable
    assert "print_lag" in assessment.unavailable
    assert "order_flow_imbalance" in assessment.unavailable
    assert assessment.trap_score is not None
    assert assessment.event_flags["signal"] == "unit test"
