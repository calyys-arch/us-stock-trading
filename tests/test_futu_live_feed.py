"""
python/interfaces/futu_live_feed.py — file-tailing MarketDataFeed that
reconstructs live Ticks from scripts/capture_market_microstructure.py
--source futu's own data/ticks/ + data/depth/ JSONL output, without opening
a second Futu/OpenD connection. Uses real temp files with synthetic content
mimicking python/interfaces/futu_tick_capture.py's exact schema.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from python.interfaces.futu_live_feed import FutuLiveFeed

_FIXED_NOW = datetime(2026, 8, 6, 14, 0, 0, tzinfo=timezone.utc)  # -> day_key "20260806"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _trade_row(price: float, size: float, time: str = "2026-08-06T09:30:00.000000") -> dict:
    return {"time": time, "price": price, "size": size, "ticker_direction": "NEUTRAL",
            "tick_type": "AUTO_MATCH", "sequence": "1", "source": "futu"}


def _depth_row(side: int, position: int, price: float, size: float, operation: int = 0) -> dict:
    return {"time": "2026-08-06T13:30:00.000000+00:00", "position": position, "market_maker": "0",
            "operation": operation, "side": side, "price": price, "size": size, "source": "futu"}


def _feed(tmp_path: Path, codes: list[str], now: datetime = _FIXED_NOW, poll_interval: float = 0.05) -> FutuLiveFeed:
    return FutuLiveFeed(
        codes, ticks_dir=tmp_path / "ticks", depth_dir=tmp_path / "depth",
        poll_interval=poll_interval, clock=lambda: now,
    )


# ── (a) normal tick emission (trades + depth both present) ─────────────────

def test_normal_tick_emission_combines_trade_and_best_bid_ask(tmp_path):
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.0, size=100.0)])
    _write_jsonl(tmp_path / "depth" / "AAPL" / "20260806.jsonl", [
        _depth_row(side=1, position=0, price=199.9, size=50.0),   # best bid
        _depth_row(side=0, position=0, price=200.1, size=40.0),   # best ask
    ])
    feed = _feed(tmp_path, ["AAPL"])

    ticks = feed.poll_once()

    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.code == "AAPL"
    assert tick.price == 200.0
    assert tick.volume == 100
    assert tick.bid == 199.9
    assert tick.ask == 200.1
    assert tick.quote_ready is True
    assert tick.source == "futu_live"

    # Second poll with no new lines written -> no new tick (nothing changed).
    assert feed.poll_once() == []


def test_only_top_of_book_position_zero_is_tracked(tmp_path):
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.0, size=100.0)])
    _write_jsonl(tmp_path / "depth" / "AAPL" / "20260806.jsonl", [
        _depth_row(side=1, position=0, price=199.9, size=50.0),
        _depth_row(side=1, position=1, price=199.8, size=30.0),   # not top-of-book — ignored
        _depth_row(side=0, position=0, price=200.1, size=40.0),
    ])
    feed = _feed(tmp_path, ["AAPL"])

    tick = feed.poll_once()[0]
    assert tick.bid == 199.9
    assert tick.ask == 200.1


def test_delete_at_top_of_book_keeps_last_known_best(tmp_path):
    """Documented simplification: this feed ignores delete-at-position-0
    events rather than trying to infer the new best price — see module
    docstring's 'Best bid/ask reconstruction' section."""
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.0, size=100.0)])
    _write_jsonl(tmp_path / "depth" / "AAPL" / "20260806.jsonl", [
        _depth_row(side=1, position=0, price=199.9, size=50.0),
        _depth_row(side=0, position=0, price=200.1, size=40.0),
    ])
    feed = _feed(tmp_path, ["AAPL"])
    tick1 = feed.poll_once()[0]
    assert tick1.bid == 199.9
    assert tick1.quote_ready is True

    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.5, size=10.0)])
    _write_jsonl(tmp_path / "depth" / "AAPL" / "20260806.jsonl", [
        _depth_row(side=1, position=0, price=199.9, size=50.0, operation=2),  # delete
    ])
    tick2 = feed.poll_once()[0]
    assert tick2.bid == 199.9  # stale but retained, not reset to price
    assert tick2.ask == 200.1  # unaffected side untouched
    assert tick2.quote_ready is True


# ── (b) missing depth file -> degraded tick, no crash ──────────────────────

def test_missing_depth_file_degrades_to_price_as_bid_ask(tmp_path):
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=150.0, size=25.0)])
    feed = _feed(tmp_path, ["AAPL"])

    ticks = feed.poll_once()

    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.price == 150.0
    assert tick.bid == 150.0
    assert tick.ask == 150.0
    assert tick.quote_ready is False


def test_missing_depth_file_logs_once_not_per_poll(tmp_path, caplog):
    import logging
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=150.0, size=25.0)])
    feed = _feed(tmp_path, ["AAPL"])

    with caplog.at_level(logging.INFO, logger="python.interfaces.futu_live_feed"):
        feed.poll_once()
        feed.poll_once()
        feed.poll_once()

    missing_depth_logs = [r for r in caplog.records if "no depth file yet" in r.message]
    assert len(missing_depth_logs) == 1


# ── (c) trades file not existing yet -> no ticks, no crash ─────────────────

def test_missing_trades_file_emits_no_ticks(tmp_path):
    feed = _feed(tmp_path, ["AAPL"])
    assert feed.poll_once() == []


def test_missing_trades_file_logs_once_not_per_poll(tmp_path, caplog):
    import logging
    feed = _feed(tmp_path, ["AAPL"])

    with caplog.at_level(logging.INFO, logger="python.interfaces.futu_live_feed"):
        feed.poll_once()
        feed.poll_once()
        feed.poll_once()

    missing_trades_logs = [r for r in caplog.records if "no trades file yet" in r.message]
    assert len(missing_trades_logs) == 1


def test_one_missing_symbol_does_not_block_another(tmp_path):
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.0, size=100.0)])
    feed = _feed(tmp_path, ["AAPL", "MSFT"])  # MSFT has no file at all

    ticks = feed.poll_once()

    assert len(ticks) == 1
    assert ticks[0].code == "AAPL"


# ── (d) mid-stream rollover to a new (UTC) day ──────────────────────────────

def test_mid_stream_day_rollover_switches_to_new_days_file(tmp_path):
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.0, size=100.0)])
    _write_jsonl(tmp_path / "depth" / "AAPL" / "20260806.jsonl", [
        _depth_row(side=1, position=0, price=199.9, size=50.0),
        _depth_row(side=0, position=0, price=200.1, size=40.0),
    ])
    current_day = {"value": _FIXED_NOW}
    feed = FutuLiveFeed(
        ["AAPL"], ticks_dir=tmp_path / "ticks", depth_dir=tmp_path / "depth",
        poll_interval=0.05, clock=lambda: current_day["value"],
    )

    tick1 = feed.poll_once()[0]
    assert tick1.price == 200.0
    assert tick1.bid == 199.9
    assert tick1.ask == 200.1
    assert tick1.quote_ready is True

    # Roll over to the next UTC day, mimicking the capture script rotating
    # its own writer at midnight UTC (TickCaptureWriter.write's day_key).
    next_day = datetime(2026, 8, 7, 0, 5, 0, tzinfo=timezone.utc)
    current_day["value"] = next_day
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260807.jsonl", [_trade_row(price=210.0, size=5.0)])
    # Only the ASK side has posted a fresh position-0 event on day 2 so
    # far — no bid event yet. Without resetting best_bid/best_ask on day
    # rollover, day 1's stale bid=199.9 would incorrectly leak into day
    # 2's quote; this test proves that does NOT happen.
    _write_jsonl(tmp_path / "depth" / "AAPL" / "20260807.jsonl", [
        _depth_row(side=0, position=0, price=210.5, size=10.0),
    ])

    ticks2 = feed.poll_once()
    assert len(ticks2) == 1
    tick2 = ticks2[0]
    assert tick2.price == 210.0
    assert tick2.volume == 5
    assert feed._states["AAPL"].best_bid is None, "day-1's stale bid must not leak into day 2"
    assert feed._states["AAPL"].best_ask == 210.5
    # bid missing entirely on day 2 so far -> not a real quote yet, falls
    # back to trade price rather than mixing a stale bid with a fresh ask.
    assert tick2.bid == 210.0
    assert tick2.quote_ready is False


# ── misc / malformed-line resilience ────────────────────────────────────────

def test_malformed_json_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "ticks" / "AAPL" / "20260806.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write(json.dumps(_trade_row(price=200.0, size=100.0)) + "\n")

    feed = _feed(tmp_path, ["AAPL"])
    ticks = feed.poll_once()
    assert len(ticks) == 1
    assert ticks[0].price == 200.0


def test_incomplete_trailing_line_not_yet_consumed(tmp_path):
    """A partial write (no trailing newline yet) must not be parsed early —
    it should be picked up whole once the newline lands."""
    path = tmp_path / "ticks" / "AAPL" / "20260806.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_trade_row(price=200.0, size=100.0)))  # no trailing \n

    feed = _feed(tmp_path, ["AAPL"])
    assert feed.poll_once() == []

    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
    ticks = feed.poll_once()
    assert len(ticks) == 1
    assert ticks[0].price == 200.0


# ── async stream() wiring smoke test ─────────────────────────────────────

def test_stream_yields_ticks_from_polling(tmp_path):
    _write_jsonl(tmp_path / "ticks" / "AAPL" / "20260806.jsonl", [_trade_row(price=200.0, size=100.0)])
    feed = _feed(tmp_path, ["AAPL"], poll_interval=0.02)

    async def _run():
        collected = []
        async for tick in feed.stream():
            collected.append(tick)
            if len(collected) >= 1:
                break
        return collected

    collected = asyncio.run(_run())
    assert len(collected) == 1
    assert collected[0].code == "AAPL"
    assert collected[0].price == 200.0
