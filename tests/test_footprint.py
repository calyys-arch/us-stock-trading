"""5-minute footprint aggregation + causal tick replay."""
from __future__ import annotations

import pandas as pd

from python.backtest.tick_replay import ticks_up_to
from python.data.tick_cache import load_trade_ticks
from python.microstructure.footprint import classify_sides, confirm_dominance, footprint_5m, probe_absorbed


def _ticks(rows: list[tuple]) -> pd.DataFrame:
    """rows: (time, price, size, ticker_direction)."""
    return pd.DataFrame(rows, columns=["time", "price", "size", "ticker_direction"])


def test_futu_direction_beats_tick_rule():
    idx_t = pd.Timestamp("2026-08-17 09:30:00")
    ticks = _ticks([
        (idx_t, 100.0, 10.0, "BUY"),
        (idx_t + pd.Timedelta(seconds=1), 99.9, 8.0, "BUY"),  # downtick, but tape says BUY
        (idx_t + pd.Timedelta(seconds=2), 99.8, 5.0, "SELL"),
    ])
    sides = classify_sides(ticks)["side"].tolist()
    assert sides == ["buy", "buy", "sell"]


def test_lee_tick_rule_when_no_tape_side():
    idx_t = pd.Timestamp("2026-08-17 09:30:00")
    ticks = _ticks([
        (idx_t, 100.0, 10.0, ""),
        (idx_t + pd.Timedelta(seconds=1), 100.1, 8.0, ""),
        (idx_t + pd.Timedelta(seconds=2), 100.1, 5.0, ""),  # zero-tick inherits buy
        (idx_t + pd.Timedelta(seconds=3), 100.0, 4.0, ""),
    ])
    sides = classify_sides(ticks)["side"].tolist()
    assert sides[0] is None
    assert sides[1:] == ["buy", "buy", "sell"]


def test_footprint_5m_delta_and_extremes():
    origin = pd.Timestamp("2026-08-17 09:30:00")
    rows = []
    # 20 sells at the low of the first 5m bin, 5 buys at the high.
    for i in range(20):
        rows.append((origin + pd.Timedelta(seconds=i), 100.0, 10.0, "SELL"))
    for i in range(5):
        rows.append((origin + pd.Timedelta(seconds=30 + i), 101.0, 4.0, "BUY"))
    fps = footprint_5m(_ticks(rows), origin=origin)
    bar = fps[origin]
    assert bar.complete
    assert bar.sell_volume == 200.0
    assert bar.buy_volume == 20.0
    assert bar.delta == -180.0
    assert bar.extreme_low_delta < 0
    assert probe_absorbed(bar, "long")
    assert not confirm_dominance(bar, "long")


def test_confirm_dominance_400pct_imbalance():
    origin = pd.Timestamp("2026-08-17 09:30:00")
    rows = [(origin + pd.Timedelta(seconds=i), 100.0, 4.0, "BUY") for i in range(20)]
    rows += [(origin + pd.Timedelta(seconds=40 + i), 100.0, 1.0, "SELL") for i in range(5)]
    bar = footprint_5m(_ticks(rows), origin=origin)[origin]
    assert confirm_dominance(bar, "long")
    assert bar.imbalance_ratio >= 4.0


def test_incomplete_bin_is_not_a_real_footprint():
    origin = pd.Timestamp("2026-08-17 09:30:00")
    rows = [(origin + pd.Timedelta(seconds=i), 100.0, 1.0, "SELL") for i in range(5)]
    bar = footprint_5m(_ticks(rows), origin=origin)[origin]
    assert not bar.complete
    assert not probe_absorbed(bar, "long")


def test_ticks_up_to_drops_future_prints():
    now = pd.Timestamp("2026-08-17 09:34:00")
    ticks = _ticks([
        (pd.Timestamp("2026-08-17 09:30:00"), 100.0, 1.0, "BUY"),
        (pd.Timestamp("2026-08-17 09:34:00"), 100.1, 1.0, "BUY"),
        (pd.Timestamp("2026-08-17 09:35:00"), 999.0, 99.0, "SELL"),
    ])
    visible = ticks_up_to(ticks, now)
    assert visible is not None
    assert len(visible) == 2
    assert float(visible["price"].max()) < 200.0


def test_load_trade_ticks_skips_bidask(tmp_path):
    path = tmp_path / "AAPL" / "20260817.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"time": "2026-08-17T09:30:00", "price": 100.0, "size": 5.0, "ticker_direction": "BUY", "source": "futu"}\n'
        '{"time": "2026-08-17T09:30:01", "bid_price": 99.9, "ask_price": 100.1, "bid_size": 10, "ask_size": 10, "source": "ibkr"}\n',
        encoding="utf-8",
    )
    df = load_trade_ticks("AAPL", "2026-08-17", ticks_dir=tmp_path)
    assert df is not None
    assert len(df) == 1
    assert float(df.iloc[0]["price"]) == 100.0
