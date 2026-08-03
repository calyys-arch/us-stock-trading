"""
intraday_cache tests — no network: backfill_symbol_months is driven with a
FakeIB double that returns synthetic 1-minute bars, matching the
tests/test_ibkr_price_source.py monkeypatching style.

Coverage:
  - fetch_ibkr_intraday_month builds the correct month-end endDateTime and
    trims bars to [month_start, month_end];
  - backfill writes one parquet per (symbol, month) and an updated meta
    sidecar after EACH month (resumability);
  - a closed month already cached is skipped on a second backfill call
    unless --force / force=True;
  - the CURRENT (open) month is never treated as "closed" so it keeps
    getting re-fetched;
  - get_cached_intraday_panel reads back exactly the cached range and
    raises when nothing is cached at all.
"""
from __future__ import annotations

import pandas as pd
import pytest

from python.data import intraday_cache


class _FakeBar:
    def __init__(self, ts: pd.Timestamp, price: float = 100.0, volume: float = 1000.0):
        self.date = ts
        self.open = self.high = self.low = self.close = price
        self.volume = volume


class _FakeIB:
    """Records reqHistoricalData calls; returns one synthetic bar/minute
    for the requested month unless `empty_for` names a month key to skip."""

    def __init__(self, empty_for: set[str] | None = None):
        self.calls: list[dict] = []
        self.empty_for = empty_for or set()
        self._connected = True

    def isConnected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False

    def qualifyContracts(self, contract):
        return [contract]

    def reqHistoricalData(self, contract, **kwargs):
        self.calls.append(kwargs)
        end_dt = pd.Timestamp(kwargs["endDateTime"])
        month_start = end_dt.replace(day=1, hour=0, minute=0, second=0)
        key = f"{month_start:%Y-%m}"
        if key in self.empty_for:
            return []
        # 3 synthetic bars near the start of the month — enough to test trimming/IO.
        return [
            _FakeBar(month_start + pd.Timedelta(minutes=i), price=100.0 + i)
            for i in range(3)
        ]


def test_fetch_ibkr_intraday_month_uses_month_end_anchor(monkeypatch):
    import ib_async

    from python.data import ibkr_price_source

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB()

    df = ibkr_price_source.fetch_ibkr_intraday_month(ib, "AAA", pd.Timestamp("2024-02-01"))

    assert len(ib.calls) == 1
    end_dt = pd.Timestamp(ib.calls[0]["endDateTime"])
    assert end_dt.month == 2 and end_dt.year == 2024 and end_dt.day == 29  # 2024 is a leap year
    assert ib.calls[0]["whatToShow"] == "TRADES"
    assert ib.calls[0]["durationStr"] == "1 M"
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_ibkr_intraday_month_empty_returns_empty_df(monkeypatch):
    import ib_async

    from python.data import ibkr_price_source

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB(empty_for={"2024-02"})

    df = ibkr_price_source.fetch_ibkr_intraday_month(ib, "AAA", pd.Timestamp("2024-02-01"))
    assert df.empty


def test_backfill_writes_parquet_and_meta_per_month(tmp_path, monkeypatch):
    import ib_async

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB()

    months = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    summary = intraday_cache.backfill_symbol_months("AAA", months, ib, cache_dir=tmp_path)

    assert summary["fetched"] == ["2024-01", "2024-02"]
    assert (tmp_path / "AAA" / "2024-01.parquet").exists()
    assert (tmp_path / "AAA" / "2024-02.parquet").exists()

    meta = intraday_cache._load_meta(tmp_path, "AAA")
    assert set(meta.keys()) == {"2024-01", "2024-02"}
    assert meta["2024-01"]["n_bars"] == 3


def test_backfill_skips_already_cached_closed_month(tmp_path, monkeypatch):
    import ib_async

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB()

    months = [pd.Timestamp("2024-01-01")]
    intraday_cache.backfill_symbol_months("AAA", months, ib, cache_dir=tmp_path)
    assert len(ib.calls) == 1

    # Second call: same (closed) month must be skipped, not re-fetched.
    summary = intraday_cache.backfill_symbol_months("AAA", months, ib, cache_dir=tmp_path)
    assert len(ib.calls) == 1
    assert summary["skipped"] == ["2024-01"]

    # --force overrides the skip.
    summary = intraday_cache.backfill_symbol_months("AAA", months, ib, cache_dir=tmp_path, force=True)
    assert len(ib.calls) == 2
    assert summary["fetched"] == ["2024-01"]


def test_current_month_is_never_closed_and_always_refetched(tmp_path, monkeypatch):
    import ib_async

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB()

    current_month = pd.Timestamp.now().replace(day=1)
    intraday_cache.backfill_symbol_months("AAA", [current_month], ib, cache_dir=tmp_path)
    assert len(ib.calls) == 1

    summary = intraday_cache.backfill_symbol_months("AAA", [current_month], ib, cache_dir=tmp_path)
    assert len(ib.calls) == 2
    assert summary["fetched"] == [f"{current_month:%Y-%m}"]


def test_empty_month_recorded_but_not_written_as_parquet(tmp_path, monkeypatch):
    import ib_async

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB(empty_for={"2024-03"})

    summary = intraday_cache.backfill_symbol_months(
        "AAA", [pd.Timestamp("2024-03-01")], ib, cache_dir=tmp_path,
    )
    assert summary["empty"] == ["2024-03"]
    assert not (tmp_path / "AAA" / "2024-03.parquet").exists()
    meta = intraday_cache._load_meta(tmp_path, "AAA")
    assert meta["2024-03"]["n_bars"] == 0


def test_get_cached_intraday_panel_reads_back_range(tmp_path, monkeypatch):
    import ib_async

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB()
    intraday_cache.backfill_symbol_months(
        "AAA", [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")], ib, cache_dir=tmp_path,
    )

    panel = intraday_cache.get_cached_intraday_panel(
        ["AAA"], "2024-01-01", "2024-02-28", cache_dir=tmp_path,
    )
    assert panel.index.names == ["ts", "code"]
    assert set(["open", "high", "low", "close", "volume"]) <= set(panel.columns)
    assert len(panel) == 6  # 3 bars x 2 months


def test_get_cached_intraday_panel_raises_when_nothing_cached(tmp_path):
    with pytest.raises(RuntimeError):
        intraday_cache.get_cached_intraday_panel(["ZZZ"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)


def test_month_range_inclusive_endpoints():
    months = intraday_cache.month_range(pd.Timestamp("2024-01-15"), pd.Timestamp("2024-03-05"))
    assert [f"{m:%Y-%m}" for m in months] == ["2024-01", "2024-02", "2024-03"]


def test_cached_symbol_coverage(tmp_path, monkeypatch):
    import ib_async

    monkeypatch.setattr(ib_async, "Stock", lambda *a, **kw: object())
    ib = _FakeIB()
    intraday_cache.backfill_symbol_months(
        "AAA", [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")], ib, cache_dir=tmp_path,
    )
    coverage = intraday_cache.cached_symbol_coverage(["AAA", "BBB"], cache_dir=tmp_path)
    assert coverage["AAA"]["n_months"] == 2
    assert coverage["AAA"]["total_bars"] == 6
    assert coverage["BBB"]["n_months"] == 0
