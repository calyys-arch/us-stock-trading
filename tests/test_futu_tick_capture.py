"""Tests for python/interfaces/futu_tick_capture.py.

Fakes out the `futu` SDK (same "no real gateway in CI" approach as
tests/test_ibkr_tick_capture.py) to drive FutuTickCapture.run() through
reconnect scenarios, and to verify the Ticker-push -> trades-row and
OrderBook-snapshot -> synthetic insert/update/delete-event translations
described in the module docstring.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from python.interfaces.futu_tick_capture import FutuTickCapture

RET_OK = 0
RET_ERROR = -1


class _FakeSubType:
    TICKER = "TICKER"
    ORDER_BOOK = "ORDER_BOOK"


class _FakeSession:
    ALL = "ALL"


class _FakeTickerHandlerBase:
    def on_recv_rsp(self, rsp_pb):
        return RET_OK, rsp_pb


class _FakeOrderBookHandlerBase:
    def on_recv_rsp(self, rsp_pb):
        return RET_OK, rsp_pb


class _FakeSysConfig:
    encrypt_calls: list[bool] = []
    rsa_file_calls: list[str] = []

    @classmethod
    def enable_proto_encrypt(cls, is_encrypt: bool) -> None:
        cls.encrypt_calls.append(is_encrypt)

    @classmethod
    def set_init_rsa_file(cls, path: str) -> None:
        cls.rsa_file_calls.append(path)


class _FakeOpenQuoteContext:
    """Scripted fake of futu.OpenQuoteContext. Each construction is one
    session attempt; class-level side-effect lists are indexed by attempt
    number (0-based) so tests can script a failure followed by recovery."""

    ticker_subscribe_side_effects: list | None = None
    orderbook_subscribe_side_effects: list | None = None
    global_state_side_effects: list | None = None
    instances: list["_FakeOpenQuoteContext"] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.handlers: list = []
        self.subscribed: dict[str, list[str]] = {}
        self.closed = False
        self._attempt_index = len(_FakeOpenQuoteContext.instances)
        _FakeOpenQuoteContext.instances.append(self)

    def set_handler(self, handler) -> None:
        self.handlers.append(handler)

    def _effect_for(self, effects: list | None, default):
        if effects is not None and self._attempt_index < len(effects):
            return effects[self._attempt_index]
        return default

    def subscribe(self, codes, subtypes, subscribe_push=True, session=None):
        if subtypes == [_FakeSubType.TICKER]:
            effect = self._effect_for(_FakeOpenQuoteContext.ticker_subscribe_side_effects, (RET_OK, None))
        else:
            effect = self._effect_for(_FakeOpenQuoteContext.orderbook_subscribe_side_effects, (RET_OK, None))
        ret, err = effect
        if ret == RET_OK:
            self.subscribed[subtypes[0]] = list(codes)
        return ret, err

    def get_global_state(self):
        return self._effect_for(
            _FakeOpenQuoteContext.global_state_side_effects, (RET_OK, {"qot_logined": True}),
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_ctx():
    _FakeOpenQuoteContext.ticker_subscribe_side_effects = None
    _FakeOpenQuoteContext.orderbook_subscribe_side_effects = None
    _FakeOpenQuoteContext.global_state_side_effects = None
    _FakeOpenQuoteContext.instances = []
    _FakeSysConfig.encrypt_calls = []
    _FakeSysConfig.rsa_file_calls = []
    yield
    _FakeOpenQuoteContext.ticker_subscribe_side_effects = None
    _FakeOpenQuoteContext.orderbook_subscribe_side_effects = None
    _FakeOpenQuoteContext.global_state_side_effects = None
    _FakeOpenQuoteContext.instances = []
    _FakeSysConfig.encrypt_calls = []
    _FakeSysConfig.rsa_file_calls = []


def _fake_futu_module() -> types.ModuleType:
    module = types.ModuleType("futu")
    module.OpenQuoteContext = _FakeOpenQuoteContext
    module.TickerHandlerBase = _FakeTickerHandlerBase
    module.OrderBookHandlerBase = _FakeOrderBookHandlerBase
    module.SubType = _FakeSubType
    module.Session = _FakeSession
    module.RET_OK = RET_OK
    module.SysConfig = _FakeSysConfig
    return module


def _make_capture(tmp_path: Path, **overrides) -> FutuTickCapture:
    kwargs = dict(
        symbols=["AAPL", "MSFT"],
        ticks_dir=tmp_path / "ticks",
        depth_dir=tmp_path / "depth",
        reconnect_delay=0.01,
        max_reconnect_delay=0.02,
    )
    kwargs.update(overrides)
    return FutuTickCapture(**kwargs)


def _run_capped(capture: FutuTickCapture, max_attempts: int) -> int:
    """Run capture.run() but force a stop once `max_attempts` session
    contexts have been created — every `time.sleep` call (inner
    health-check tick AND outer backoff) is intercepted."""

    def fake_sleep(_secs):
        if len(_FakeOpenQuoteContext.instances) >= max_attempts:
            capture._running = False

    with patch("python.interfaces.futu_tick_capture.time.sleep", fake_sleep):
        capture.run()
    return len(_FakeOpenQuoteContext.instances)


def test_single_successful_session_then_stops_cleanly(tmp_path):
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        attempts = _run_capped(capture, max_attempts=1)
    assert attempts == 1
    assert capture.event_counts == {"trades": 0, "depth": 0}
    assert _FakeOpenQuoteContext.instances[0].closed is True


def test_subscribes_to_ticker_and_order_book_for_all_symbols(tmp_path):
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        _run_capped(capture, max_attempts=1)
    ctx = _FakeOpenQuoteContext.instances[0]
    assert ctx.subscribed[_FakeSubType.TICKER] == ["US.AAPL", "US.MSFT"]
    assert ctx.subscribed[_FakeSubType.ORDER_BOOK] == ["US.AAPL", "US.MSFT"]


def test_ticker_subscribe_failure_triggers_reconnect_not_a_crash(tmp_path):
    _FakeOpenQuoteContext.ticker_subscribe_side_effects = [(RET_ERROR, "no permission"), (RET_OK, None)]
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        _run_capped(capture, max_attempts=2)  # must not raise
    assert len(_FakeOpenQuoteContext.instances) == 2


def test_order_book_subscribe_failure_continues_with_trades_only(tmp_path):
    """Order-book subscribe failing (e.g. quota exhausted) should not abort
    the whole session — trades keep flowing."""
    _FakeOpenQuoteContext.orderbook_subscribe_side_effects = [(RET_ERROR, "quota exceeded")]
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        attempts = _run_capped(capture, max_attempts=1)
    assert attempts == 1  # no reconnect triggered


def test_qot_logout_during_session_triggers_reconnect(tmp_path):
    _FakeOpenQuoteContext.global_state_side_effects = [(RET_OK, {"qot_logined": False}), (RET_OK, {"qot_logined": True})]
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        _run_capped(capture, max_attempts=2)  # must not raise
    assert len(_FakeOpenQuoteContext.instances) == 2


def test_backoff_delay_grows_with_attempt_and_is_capped(tmp_path):
    _FakeOpenQuoteContext.ticker_subscribe_side_effects = [
        (RET_ERROR, "a"), (RET_ERROR, "b"), (RET_OK, None),
    ]
    capture = _make_capture(tmp_path, reconnect_delay=10.0, max_reconnect_delay=15.0)

    seen_delays = []

    def fake_backoff_sleep(secs):
        seen_delays.append(secs)
        if len(seen_delays) >= 3:
            capture._running = False

    with patch.dict(sys.modules, {"futu": _fake_futu_module()}), \
            patch("python.interfaces.futu_tick_capture.time.sleep", fake_backoff_sleep):
        capture.run()

    assert seen_delays[0] == 10.0  # attempt 1 failed -> delay * 1
    assert seen_delays[1] == 15.0  # attempt 2 failed -> delay * 2, capped at max
    assert len(_FakeOpenQuoteContext.instances) == 3


def test_rsa_key_path_enables_protocol_encryption_before_connecting(tmp_path):
    capture = _make_capture(tmp_path, rsa_key_path="/tmp/conn_key.txt")
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        _run_capped(capture, max_attempts=1)
    assert _FakeSysConfig.encrypt_calls == [True]
    assert _FakeSysConfig.rsa_file_calls == ["/tmp/conn_key.txt"]


def test_no_rsa_key_path_leaves_encryption_untouched(tmp_path):
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        _run_capped(capture, max_attempts=1)
    assert _FakeSysConfig.encrypt_calls == []
    assert _FakeSysConfig.rsa_file_calls == []


def test_stop_closes_context_and_flips_running_false(tmp_path):
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        ctx = _FakeOpenQuoteContext(host="127.0.0.1", port=11111)
        capture._quote_ctx = ctx
        capture.stop()
    assert capture._running is False
    assert ctx.closed is True


# ── push handler translation tests (no run() involved) ──────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ticker_push_writes_trade_row_with_futu_specific_fields(tmp_path):
    capture = _make_capture(tmp_path, symbols=["AAPL"], rth_only=False)
    df = pd.DataFrame([{
        "code": "US.AAPL",
        "name": "Apple",
        "time": "2026-08-04 10:30:00.100",
        "price": 180.25,
        "volume": 9.0,
        "turnover": 1622.25,
        "ticker_direction": "BUY",
        "sequence": 123456789,
        "type": "ODD_LOT",
        "push_data_type": "REAL",
    }])
    capture._handle_ticker(df)

    assert capture.event_counts == {"trades": 1, "depth": 0}
    rows = _read_jsonl(tmp_path / "ticks" / "AAPL" / f"{pd.Timestamp.now(tz='UTC'):%Y%m%d}.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["price"] == 180.25
    assert row["size"] == 9.0
    assert row["ticker_direction"] == "BUY"
    assert row["source"] == "futu"
    # No exchange/special_conditions columns — Futu's US tick feed doesn't
    # expose venue/condition codes; trap_detector must see these as missing,
    # not zero.
    assert "exchange" not in row
    assert "special_conditions" not in row


def test_order_book_push_synthesizes_insert_then_update_then_delete_events(tmp_path):
    capture = _make_capture(tmp_path, symbols=["AAPL"], rth_only=False, depth_rows=3)

    # First snapshot: 2 bid levels, 1 ask level -> all inserts.
    capture._handle_order_book({
        "code": "US.AAPL",
        "Bid": [(180.20, 15, 3), (180.19, 1, 1)],
        "Ask": [(180.30, 100, 1)],
    })
    day_key = f"{pd.Timestamp.now(tz='UTC'):%Y%m%d}"
    rows = _read_jsonl(tmp_path / "depth" / "AAPL" / f"{day_key}.jsonl")
    assert len(rows) == 3
    assert all(r["operation"] == 0 for r in rows)  # insert
    assert capture.event_counts["depth"] == 3

    # Second snapshot: best bid price changes (update), second bid level
    # disappears (delete), ask unchanged (no event).
    capture._handle_order_book({
        "code": "US.AAPL",
        "Bid": [(180.21, 20, 2)],
        "Ask": [(180.30, 100, 1)],
    })
    rows = _read_jsonl(tmp_path / "depth" / "AAPL" / f"{day_key}.jsonl")
    assert len(rows) == 3 + 2  # one update (position 0 bid) + one delete (position 1 bid)
    new_rows = rows[3:]
    ops = sorted(r["operation"] for r in new_rows)
    assert ops == [1, 2]  # update, delete
    delete_row = next(r for r in new_rows if r["operation"] == 2)
    assert delete_row["side"] == 1  # bid
    assert delete_row["price"] == 180.19
