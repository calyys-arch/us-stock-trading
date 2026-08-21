"""
python/microstructure/signal_journal.py — append-only JSONL journal for
every live MicroSignal that fires, approved or RiskEngine-rejected.
"""
from __future__ import annotations

import json

import pandas as pd

from python.core.types import QualifiedMicroOrder
from python.microstructure.signal_journal import SignalJournal
from python.microstructure.signals import MicroSignal


def _signal(**overrides) -> MicroSignal:
    base = dict(
        symbol="AAPL",
        strategy="fvg_retest",
        direction="long",
        signal_time=pd.Timestamp("2026-08-06 10:15:00"),
        entry_price=200.0,
        stop_price=198.0,
        target_price=205.0,
        context={"vwap_distance_atr": 0.4, "liquidity_level": "YDH", "gap_size": 1.2},
    )
    base.update(overrides)
    return MicroSignal(**base)


def _approved_order(signal: MicroSignal, **overrides) -> QualifiedMicroOrder:
    base = dict(
        raw=signal, qty=50, entry_limit_price=200.05, stop_price=198.0,
        stop_limit_price=197.9, target_price=205.0, gross_notional=10_002.5,
        approved=True, rejection_reason=None,
    )
    base.update(overrides)
    return QualifiedMicroOrder(**base)


def _rejected_order(signal: MicroSignal, reason: str = "observe_mode") -> QualifiedMicroOrder:
    return QualifiedMicroOrder(
        raw=signal, qty=0, entry_limit_price=0.0, stop_price=0.0, stop_limit_price=0.0,
        target_price=None, gross_notional=0.0, approved=False, rejection_reason=reason,
    )


# ── (a) full-context journaling ─────────────────────────────────────────────

def test_record_writes_full_triggering_context(tmp_path):
    journal = SignalJournal(output_dir=tmp_path)
    signal = _signal()
    order = _approved_order(signal)

    entry = journal.record(signal, order)

    assert entry["symbol"] == "AAPL"
    assert entry["strategy"] == "fvg_retest"
    assert entry["direction"] == "long"
    assert entry["entry_price"] == 200.0
    assert entry["stop_price"] == 198.0
    assert entry["target_price"] == 205.0
    assert entry["context"] == {"vwap_distance_atr": 0.4, "liquidity_level": "YDH", "gap_size": 1.2}
    assert entry["qualified_order"]["qty"] == 50
    assert entry["qualified_order"]["entry_limit_price"] == 200.05
    assert entry["outcome"] == {
        "status": "pending", "exit_price": None, "exit_time": None, "pnl": None, "pnl_pct": None,
    }

    on_disk = journal.read_day(signal.signal_time)
    assert len(on_disk) == 1
    assert on_disk[0]["symbol"] == "AAPL"

    path = tmp_path / "2026-08-06.jsonl"
    assert path.exists()
    with path.open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    assert lines[0]["symbol"] == "AAPL"


def test_record_is_append_only_across_multiple_signals(tmp_path):
    journal = SignalJournal(output_dir=tmp_path)
    sig1 = _signal(symbol="AAPL")
    sig2 = _signal(symbol="MSFT", signal_time=pd.Timestamp("2026-08-06 10:20:00"))

    journal.record(sig1, _approved_order(sig1))
    journal.record(sig2, _approved_order(sig2))

    rows = journal.read_day(pd.Timestamp("2026-08-06"))
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}


# ── (b) file rotation across a date boundary ────────────────────────────────

def test_file_rotates_by_signal_time_date(tmp_path):
    journal = SignalJournal(output_dir=tmp_path)
    day1_signal = _signal(signal_time=pd.Timestamp("2026-08-06 15:55:00"))
    day2_signal = _signal(signal_time=pd.Timestamp("2026-08-07 09:35:00"))

    journal.record(day1_signal, _approved_order(day1_signal))
    journal.record(day2_signal, _approved_order(day2_signal))

    assert (tmp_path / "2026-08-06.jsonl").exists()
    assert (tmp_path / "2026-08-07.jsonl").exists()

    day1_rows = journal.read_day(pd.Timestamp("2026-08-06"))
    day2_rows = journal.read_day(pd.Timestamp("2026-08-07"))
    assert len(day1_rows) == 1
    assert len(day2_rows) == 1
    assert day1_rows[0]["signal_time"].startswith("2026-08-06")
    assert day2_rows[0]["signal_time"].startswith("2026-08-07")

    # A day with no recorded signals reads back empty, not an error.
    assert journal.read_day(pd.Timestamp("2026-08-05")) == []


# ── (c) RiskEngine-filtered vs RiskEngine-passed flag ───────────────────────

def test_risk_passed_flag_true_for_approved_order(tmp_path):
    journal = SignalJournal(output_dir=tmp_path)
    signal = _signal()
    order = _approved_order(signal)

    entry = journal.record(signal, order)

    assert entry["risk_passed"] is True
    assert entry["rejection_reason"] is None


def test_risk_passed_flag_false_for_rejected_order():
    """Non-persisting variant of the fixture above: verifies the flag
    logic directly (record() also appends to disk, exercised by the
    tmp_path-based tests) — kept separate so the false-case assertion
    doesn't depend on tmp_path plumbing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        journal = SignalJournal(output_dir=tmp)
        signal = _signal()
        order = _rejected_order(signal, reason="event_blackout")

        entry = journal.record(signal, order)

        assert entry["risk_passed"] is False
        assert entry["rejection_reason"] == "event_blackout"


def test_both_flags_recorded_side_by_side_in_same_file(tmp_path):
    """Both a passing and a filtered-out signal on the same session date
    land in the same file, each correctly flagged — the core guarantee
    that makes the journal useful for a later backtest-vs-live diff."""
    journal = SignalJournal(output_dir=tmp_path)
    passing = _signal(symbol="AAPL")
    filtered = _signal(symbol="MSFT")

    journal.record(passing, _approved_order(passing))
    journal.record(filtered, _rejected_order(filtered, reason="max_open_micro_positions"))

    rows = journal.read_day(pd.Timestamp("2026-08-06"))
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["AAPL"]["risk_passed"] is True
    assert by_symbol["MSFT"]["risk_passed"] is False
    assert by_symbol["MSFT"]["rejection_reason"] == "max_open_micro_positions"


# ── misc resilience ──────────────────────────────────────────────────────

def test_read_day_skips_malformed_lines(tmp_path):
    journal = SignalJournal(output_dir=tmp_path)
    path = tmp_path / "2026-08-06.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write(json.dumps({"symbol": "AAPL"}) + "\n")

    rows = journal.read_day(pd.Timestamp("2026-08-06"))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"


def test_read_today_uses_current_et_date(monkeypatch, tmp_path):
    import datetime as dt_module

    import python.microstructure.signal_journal as sj_module

    fixed_utc = dt_module.datetime(2026, 8, 6, 14, 0, 0, tzinfo=dt_module.timezone.utc)  # 10:00 ET

    class _FixedDatetime(dt_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_utc

    monkeypatch.setattr(sj_module, "datetime", _FixedDatetime)

    journal = SignalJournal(output_dir=tmp_path)
    signal = _signal(signal_time=pd.Timestamp("2026-08-06 10:15:00"))
    journal.record(signal, _approved_order(signal))

    rows = journal.read_today()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
