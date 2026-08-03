"""Tests for python/interfaces/ibkr_tick_capture.py's reconnect resilience.

Regression coverage for the 2026-07-29 incident: an overnight capture run
hung inside `qualifyContracts()` for ~2h (ib_async's default RequestTimeout
is 0 == wait forever) while IB Gateway's data farms were down, then crashed
with an unhandled ConnectionError once the peer closed the socket —
capturing zero data all night. These tests fake out `ib_async` so we can
drive `IbkrTickCapture.run()` through disconnect/retry scenarios without a
real IB Gateway.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from python.interfaces.ibkr_tick_capture import IbkrTickCapture


class _FakeEvent:
    """Stand-in for ib_async's `Event` (`+=` subscribes a handler)."""

    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _FakeContract:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


def _fake_stock(symbol: str, *_args, **_kwargs) -> _FakeContract:
    return _FakeContract(symbol)


class _FakeIB:
    """Scripted fake of ib_async.IB. Each new `IB()` construction inside
    run()'s retry loop is one "session attempt"; class-level lists index
    behaviour by attempt number so tests can script a failure followed by
    a recovery."""

    qualify_side_effects: list | None = None
    connect_side_effects: list | None = None
    max_sleep_iters = 1  # how many ib.sleep() calls before isConnected() flips False
    instances: list["_FakeIB"] = []

    def __init__(self) -> None:
        self.RequestTimeout = None
        self.pendingTickersEvent = _FakeEvent()
        self._connected = False
        self._sleep_calls = 0
        self._attempt_index = len(_FakeIB.instances)
        _FakeIB.instances.append(self)

    def connect(self, host, port, clientId, timeout):
        effects = _FakeIB.connect_side_effects
        if effects is not None and self._attempt_index < len(effects):
            effect = effects[self._attempt_index]
            if effect is not None:
                raise effect
        self._connected = True

    def qualifyContracts(self, contract):
        effects = _FakeIB.qualify_side_effects
        if effects is not None and self._attempt_index < len(effects):
            effect = effects[self._attempt_index]
            if isinstance(effect, Exception):
                raise effect
            if effect is False:
                return []
        return [contract]

    def reqTickByTickData(self, *_args, **_kwargs):
        pass

    def reqMktDepth(self, *_args, **_kwargs):
        pass

    def isConnected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False

    def sleep(self, _secs) -> None:
        self._sleep_calls += 1
        if self._sleep_calls >= _FakeIB.max_sleep_iters:
            self._connected = False


@pytest.fixture(autouse=True)
def _reset_fake_ib():
    _FakeIB.qualify_side_effects = None
    _FakeIB.connect_side_effects = None
    _FakeIB.max_sleep_iters = 1
    _FakeIB.instances = []
    yield
    _FakeIB.qualify_side_effects = None
    _FakeIB.connect_side_effects = None
    _FakeIB.instances = []


def _fake_ib_async_module() -> types.ModuleType:
    module = types.ModuleType("ib_async")
    module.IB = _FakeIB
    module.Stock = _fake_stock
    return module


def _make_capture(tmp_path: Path, **overrides) -> IbkrTickCapture:
    kwargs = dict(
        symbols=["AAPL", "MSFT"],
        ticks_dir=tmp_path / "ticks",
        depth_dir=tmp_path / "depth",
        reconnect_delay=0.01,
        max_reconnect_delay=0.02,
        request_timeout=1.0,
    )
    kwargs.update(overrides)
    return IbkrTickCapture(**kwargs)


def _run_capped(capture: IbkrTickCapture, max_attempts: int) -> int:
    """Run capture.run() but force a stop after `max_attempts` connect
    attempts (counted via the backoff `time.sleep` call between attempts),
    so tests don't spin forever when every scripted attempt "succeeds"
    (i.e. runs its session to completion rather than raising)."""
    calls = {"n": 0}

    def fake_backoff_sleep(_secs):
        calls["n"] += 1
        if calls["n"] >= max_attempts:
            capture._running = False

    with patch("python.interfaces.ibkr_tick_capture.time.sleep", fake_backoff_sleep):
        capture.run()
    return calls["n"]


def test_run_sets_bounded_request_timeout_instead_of_default_zero(tmp_path):
    """The 2026-07-29 incident happened because ib_async's default
    RequestTimeout (0 == wait forever) let qualifyContracts() hang for
    hours. run() must set a finite timeout on every session's IB instance."""
    capture = _make_capture(tmp_path, request_timeout=7.5)
    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}):
        _run_capped(capture, max_attempts=1)
    assert _FakeIB.instances[0].RequestTimeout == 7.5


def test_single_successful_session_then_stops_cleanly(tmp_path):
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}):
        attempts = _run_capped(capture, max_attempts=1)
    assert attempts == 1
    assert len(_FakeIB.instances) == 1
    assert capture.event_counts == {"trades": 0, "bidask": 0, "depth": 0}


def test_qualify_failure_triggers_reconnect_not_a_crash(tmp_path):
    """Regression for the exact incident: qualifyContracts() failing/timing
    out on every symbol must not raise out of run() — it should reconnect
    and eventually succeed once the data farm recovers."""
    _FakeIB.qualify_side_effects = [False, None]  # attempt 1: all fail; attempt 2: succeeds

    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}):
        _run_capped(capture, max_attempts=2)  # must not raise

    assert len(_FakeIB.instances) == 2, "expected exactly one reconnect attempt"


def test_qualify_raising_exception_on_one_symbol_does_not_abort_session(tmp_path):
    """A qualifyContracts() call that raises (e.g. a bounded-timeout
    TimeoutError) must be caught per-symbol so a still-broken farm on one
    session doesn't crash the process — it should simply reconnect."""
    _FakeIB.qualify_side_effects = [TimeoutError("boom"), None]

    capture = _make_capture(tmp_path, symbols=["AAPL"])
    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}):
        _run_capped(capture, max_attempts=2)  # must not raise

    assert len(_FakeIB.instances) == 2


def test_connect_failure_backs_off_and_reconnects(tmp_path):
    _FakeIB.connect_side_effects = [ConnectionError("refused"), None]

    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}):
        _run_capped(capture, max_attempts=2)  # must not raise

    assert len(_FakeIB.instances) == 2


def test_backoff_delay_grows_with_attempt_and_is_capped(tmp_path):
    _FakeIB.connect_side_effects = [ConnectionError("a"), ConnectionError("b"), None]
    capture = _make_capture(tmp_path, reconnect_delay=10.0, max_reconnect_delay=15.0)

    seen_delays = []

    def fake_backoff_sleep(secs):
        seen_delays.append(secs)
        if len(seen_delays) >= 3:
            capture._running = False

    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}), \
            patch("python.interfaces.ibkr_tick_capture.time.sleep", fake_backoff_sleep):
        capture.run()

    assert seen_delays[0] == 10.0  # attempt 1 failed -> delay * 1
    assert seen_delays[1] == 15.0  # attempt 2 failed -> delay * 2, capped at max
    assert len(_FakeIB.instances) == 3


def test_stop_disconnects_and_flips_running_false(tmp_path):
    capture = _make_capture(tmp_path)
    with patch.dict(sys.modules, {"ib_async": _fake_ib_async_module()}):
        fake_ib = _FakeIB()
        fake_ib.connect("127.0.0.1", 4002, 41, 10)
        capture._ib = fake_ib
        capture.stop()
    assert capture._running is False
    assert fake_ib.isConnected() is False
