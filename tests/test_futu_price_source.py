"""Tests for python/data/futu_price_source.py (no live OpenD connection).

Fakes out the `futu` SDK (same approach as tests/test_futu_tick_capture.py)
to drive open_futu_quote_context / fetch_history_kline_range /
backfill_symbol_months without a real gateway, and verifies:
  - configs/broker.yaml's `futu:` block is read with the right defaults.
  - the RSA protocol-encryption dance happens before connecting, only when
    rsa_key_path is set (mirrors futu_tick_capture.py's own tests).
  - a not-logged-in quote session raises FutuHistoricalUnavailable rather
    than silently proceeding.
  - request_history_kline pagination via page_req_key is followed until
    exhausted, with only ONE quota-relevant "unlock" implied per code.
  - the empirically-verified END-time -> START-time bar shift (module
    docstring) is applied correctly, including the exact AAPL 2026-07-31
    regression values recorded in the docstring.
  - backfill_symbol_months splits one ranged fetch into the existing
    per-month parquet + meta.json cache layout, skips already-closed
    cached months (unless force=True), and records "source": "futu".
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from python.data import futu_price_source as fps
from python.data.futu_price_source import (
    FutuHistoricalUnavailable,
    backfill_symbol_months,
    check_history_kline_quota,
    fetch_history_kline_range,
    load_connection_settings,
    open_futu_quote_context,
)

RET_OK = 0
RET_ERROR = -1


class _FakeKLType:
    K_1M = "K_1M"


class _FakeAuType:
    NONE = "NONE"


class _FakeSysConfig:
    encrypt_calls: list[bool] = []
    rsa_file_calls: list[str] = []

    @classmethod
    def enable_proto_encrypt(cls, is_encrypt: bool) -> None:
        cls.encrypt_calls.append(is_encrypt)

    @classmethod
    def set_init_rsa_file(cls, path: str) -> None:
        cls.rsa_file_calls.append(path)


def _kline_df(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": "US.X", "name": "X", "time_key": t, "open": o, "high": h, "low": lo,
          "close": c, "volume": v, "pe_ratio": 0.0, "turnover_rate": 0.0, "turnover": 0.0,
          "change_rate": 0.0, "last_close": 0.0}
         for t, o, h, lo, c, v in rows]
    )


class _FakeOpenQuoteContext:
    """Scripted fake of futu.OpenQuoteContext's subset used by
    futu_price_source.py."""

    global_state_side_effects: list | None = None
    quota_side_effects: list | None = None
    # Each entry: dict mapping page_req_key (None for first) -> (ret, df, next_key)
    kline_pages: dict | None = None
    kline_calls: list = []
    construct_should_raise: bool = False

    def __init__(self, host: str, port: int) -> None:
        if _FakeOpenQuoteContext.construct_should_raise:
            raise ConnectionError("refused")
        self.host = host
        self.port = port
        self.closed = False

    def get_global_state(self):
        effects = _FakeOpenQuoteContext.global_state_side_effects
        if effects:
            return effects.pop(0)
        return (RET_OK, {"qot_logined": True})

    def get_history_kl_quota(self, get_detail=False):
        effects = _FakeOpenQuoteContext.quota_side_effects
        if effects:
            return effects.pop(0)
        return (RET_OK, (0, 100, []))

    def request_history_kline(self, code, start=None, end=None, ktype=None, autype=None,
                               max_count=None, page_req_key=None):
        _FakeOpenQuoteContext.kline_calls.append(
            {"code": code, "start": start, "end": end, "page_req_key": page_req_key}
        )
        pages = _FakeOpenQuoteContext.kline_pages or {}
        return pages[page_req_key]

    def close(self):
        self.closed = True


def _fake_futu_module() -> types.ModuleType:
    module = types.ModuleType("futu")
    module.OpenQuoteContext = _FakeOpenQuoteContext
    module.SysConfig = _FakeSysConfig
    module.KLType = _FakeKLType
    module.AuType = _FakeAuType
    module.RET_OK = RET_OK
    return module


@pytest.fixture(autouse=True)
def _reset_fake_ctx():
    _FakeOpenQuoteContext.global_state_side_effects = None
    _FakeOpenQuoteContext.quota_side_effects = None
    _FakeOpenQuoteContext.kline_pages = None
    _FakeOpenQuoteContext.kline_calls = []
    _FakeOpenQuoteContext.construct_should_raise = False
    _FakeSysConfig.encrypt_calls = []
    _FakeSysConfig.rsa_file_calls = []
    yield


# ── load_connection_settings ────────────────────────────────────────────────

def test_load_connection_settings_defaults(tmp_path):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("futu:\n  host: 10.0.0.9\n  port: 22222\n", encoding="utf-8")
    settings = load_connection_settings(cfg)
    assert settings == {
        "host": "10.0.0.9", "port": 22222, "market_prefix": "US", "rsa_key_path": None,
    }


def test_load_connection_settings_reads_repo_broker_yaml():
    settings = load_connection_settings()
    assert settings["port"] == 11111
    assert settings["market_prefix"] == "US"


# ── open_futu_quote_context ─────────────────────────────────────────────────

def test_open_futu_quote_context_applies_rsa_before_connecting(tmp_path):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("futu:\n  host: 127.0.0.1\n  port: 11111\n  rsa_key_path: /tmp/key.txt\n",
                    encoding="utf-8")
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        ctx = open_futu_quote_context(cfg)
    assert _FakeSysConfig.encrypt_calls == [True]
    assert _FakeSysConfig.rsa_file_calls == ["/tmp/key.txt"]
    assert isinstance(ctx, _FakeOpenQuoteContext)


def test_open_futu_quote_context_no_rsa_key_leaves_encryption_untouched(tmp_path):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("futu:\n  host: 127.0.0.1\n  port: 11111\n", encoding="utf-8")
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        open_futu_quote_context(cfg)
    assert _FakeSysConfig.encrypt_calls == []
    assert _FakeSysConfig.rsa_file_calls == []


def test_open_futu_quote_context_raises_when_not_logged_in(tmp_path):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("futu:\n  host: 127.0.0.1\n  port: 11111\n", encoding="utf-8")
    _FakeOpenQuoteContext.global_state_side_effects = [(RET_OK, {"qot_logined": False})]
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        with pytest.raises(FutuHistoricalUnavailable, match="not logged in"):
            open_futu_quote_context(cfg)


def test_open_futu_quote_context_raises_on_connection_failure(tmp_path):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("futu:\n  host: 127.0.0.1\n  port: 11111\n", encoding="utf-8")
    _FakeOpenQuoteContext.construct_should_raise = True
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        with pytest.raises(FutuHistoricalUnavailable, match="could not connect"):
            open_futu_quote_context(cfg)


def test_open_futu_quote_context_raises_when_futu_not_installed(tmp_path, monkeypatch):
    cfg = tmp_path / "broker.yaml"
    cfg.write_text("futu:\n  host: 127.0.0.1\n  port: 11111\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "futu", None)  # forces ImportError on `from futu import ...`
    with pytest.raises(FutuHistoricalUnavailable, match="not installed"):
        open_futu_quote_context(cfg)


# ── check_history_kline_quota ───────────────────────────────────────────────

def test_check_history_kline_quota_parses_tuple():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    _FakeOpenQuoteContext.quota_side_effects = [
        (RET_OK, (2, 98, [{"code": "US.QQQ"}, {"code": "US.AAPL"}])),
    ]
    result = check_history_kline_quota(ctx)
    assert result == {"used": 2, "remaining": 98, "detail": [{"code": "US.QQQ"}, {"code": "US.AAPL"}]}


def test_check_history_kline_quota_raises_on_error():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    _FakeOpenQuoteContext.quota_side_effects = [(RET_ERROR, "no permission")]
    with pytest.raises(FutuHistoricalUnavailable):
        check_history_kline_quota(ctx)


# ── fetch_history_kline_range: pagination + bar-label shift ────────────────

def test_fetch_history_kline_range_single_page_shifts_bar_labels():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    df = _kline_df([
        ("2026-07-31 09:31:00", 304.81, 306.90, 303.60, 304.05, 3767516.0),
        ("2026-07-31 09:32:00", 304.01, 305.015, 301.83, 301.93, 1704737.0),
    ])
    _FakeOpenQuoteContext.kline_pages = {None: (RET_OK, df, None)}

    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        out = fetch_history_kline_range(ctx, "AAPL", "2026-07-31", "2026-07-31")

    # Empirically-verified regression: Futu's end-labeled "09:31:00"/"09:32:00"
    # bars must become start-labeled "09:30:00"/"09:31:00" — matching the
    # IBKR-sourced convention already used by data/history_1m/.
    assert list(out.index) == [pd.Timestamp("2026-07-31 09:30:00"), pd.Timestamp("2026-07-31 09:31:00")]
    assert out.loc["2026-07-31 09:31:00", "close"] == 301.93
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_history_kline_range_paginates_until_key_exhausted():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    page1 = _kline_df([("2026-08-01 09:31:00", 1.0, 1.0, 1.0, 1.0, 100.0)])
    page2 = _kline_df([("2026-08-01 09:32:00", 2.0, 2.0, 2.0, 2.0, 200.0)])
    _FakeOpenQuoteContext.kline_pages = {
        None: (RET_OK, page1, b"page2key"),
        b"page2key": (RET_OK, page2, None),
    }

    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        out = fetch_history_kline_range(ctx, "QQQ", "2026-08-01", "2026-08-01")

    assert len(_FakeOpenQuoteContext.kline_calls) == 2
    assert _FakeOpenQuoteContext.kline_calls[0]["page_req_key"] is None
    assert _FakeOpenQuoteContext.kline_calls[1]["page_req_key"] == b"page2key"
    assert len(out) == 2


def test_fetch_history_kline_range_empty_result_is_not_an_error():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    _FakeOpenQuoteContext.kline_pages = {None: (RET_OK, _kline_df([]), None)}
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        out = fetch_history_kline_range(ctx, "QQQ", "2020-01-01", "2020-01-02")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_history_kline_range_raises_on_request_error():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    _FakeOpenQuoteContext.kline_pages = {None: (RET_ERROR, "quota exceeded / no subscription", None)}
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        with pytest.raises(FutuHistoricalUnavailable, match="quota exceeded"):
            fetch_history_kline_range(ctx, "QQQ", "2026-08-01", "2026-08-01")


def test_fetch_history_kline_range_uses_futu_code_convention():
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    _FakeOpenQuoteContext.kline_pages = {None: (RET_OK, _kline_df([]), None)}
    with patch.dict(sys.modules, {"futu": _fake_futu_module()}):
        fetch_history_kline_range(ctx, "spy", "2026-08-01", "2026-08-01", market_prefix="US")
    assert _FakeOpenQuoteContext.kline_calls[0]["code"] == "US.SPY"


# ── backfill_symbol_months ──────────────────────────────────────────────────

def test_backfill_symbol_months_writes_parquet_and_meta(tmp_path):
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    # Two bars in month 1, one bar in month 2, all pre-shifted to already be
    # start-labeled by using a fake fetch (module-level function is
    # monkeypatched below rather than exercising the raw futu pagination
    # path again — that is covered by the fetch_history_kline_range tests).
    fake_full_df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [10.0, 20.0, 30.0],
        },
        index=pd.DatetimeIndex(
            ["2026-05-15 09:30:00", "2026-05-15 09:31:00", "2026-06-01 09:30:00"], name="ts",
        ),
    )

    with patch.object(fps, "fetch_history_kline_range", return_value=fake_full_df) as mock_fetch:
        summary = backfill_symbol_months(
            "QQQ", [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")], ctx,
            cache_dir=tmp_path,
        )

    assert mock_fetch.call_count == 1  # ONE ranged fetch, not one per month
    assert summary["fetched"] == ["2026-05", "2026-06"]
    assert summary["skipped"] == []
    assert summary["empty"] == []

    may_df = pd.read_parquet(tmp_path / "QQQ" / "2026-05.parquet")
    assert len(may_df) == 2
    jun_df = pd.read_parquet(tmp_path / "QQQ" / "2026-06.parquet")
    assert len(jun_df) == 1

    import json
    meta = json.loads((tmp_path / "QQQ" / "_meta.json").read_text(encoding="utf-8"))
    assert meta["2026-05"]["source"] == "futu"
    assert meta["2026-05"]["n_bars"] == 2
    assert meta["2026-06"]["n_bars"] == 1


def test_backfill_symbol_months_skips_already_closed_cached_months(tmp_path):
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    symbol_dir = tmp_path / "QQQ"
    symbol_dir.mkdir(parents=True)
    old_df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex(["2020-01-02 09:30:00"], name="ts"),
    )
    old_df.to_parquet(symbol_dir / "2020-01.parquet")
    import json
    (symbol_dir / "_meta.json").write_text(
        json.dumps({"2020-01": {"fetched_at": "x", "n_bars": 1, "closed": True, "source": "futu"}}),
        encoding="utf-8",
    )

    with patch.object(fps, "fetch_history_kline_range") as mock_fetch:
        summary = backfill_symbol_months("QQQ", [pd.Timestamp("2020-01-01")], ctx, cache_dir=tmp_path)

    assert summary["skipped"] == ["2020-01"]
    mock_fetch.assert_not_called()


def test_backfill_symbol_months_force_refetches_closed_months(tmp_path):
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    symbol_dir = tmp_path / "QQQ"
    symbol_dir.mkdir(parents=True)
    old_df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex(["2020-01-02 09:30:00"], name="ts"),
    )
    old_df.to_parquet(symbol_dir / "2020-01.parquet")
    import json
    (symbol_dir / "_meta.json").write_text(
        json.dumps({"2020-01": {"fetched_at": "x", "n_bars": 1, "closed": True, "source": "futu"}}),
        encoding="utf-8",
    )
    fresh_df = pd.DataFrame(
        {"open": [9.0], "high": [9.0], "low": [9.0], "close": [9.0], "volume": [99.0]},
        index=pd.DatetimeIndex(["2020-01-03 09:30:00"], name="ts"),
    )

    with patch.object(fps, "fetch_history_kline_range", return_value=fresh_df) as mock_fetch:
        summary = backfill_symbol_months(
            "QQQ", [pd.Timestamp("2020-01-01")], ctx, cache_dir=tmp_path, force=True,
        )

    assert summary["fetched"] == ["2020-01"]
    mock_fetch.assert_called_once()
    new_df = pd.read_parquet(symbol_dir / "2020-01.parquet")
    assert new_df["volume"].iloc[0] == 99.0


def test_backfill_symbol_months_empty_month_recorded_not_written(tmp_path):
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    empty_df = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="ts"),
    )

    with patch.object(fps, "fetch_history_kline_range", return_value=empty_df):
        summary = backfill_symbol_months("QQQ", [pd.Timestamp("2019-01-01")], ctx, cache_dir=tmp_path)

    assert summary["empty"] == ["2019-01"]
    assert not (tmp_path / "QQQ" / "2019-01.parquet").exists()
    import json
    meta = json.loads((tmp_path / "QQQ" / "_meta.json").read_text(encoding="utf-8"))
    assert meta["2019-01"]["n_bars"] == 0


def test_backfill_symbol_months_propagates_failure_without_partial_writes(tmp_path):
    ctx = _FakeOpenQuoteContext("127.0.0.1", 11111)
    with patch.object(fps, "fetch_history_kline_range", side_effect=FutuHistoricalUnavailable("quota exhausted")):
        with pytest.raises(FutuHistoricalUnavailable):
            backfill_symbol_months("QQQ", [pd.Timestamp("2026-05-01")], ctx, cache_dir=tmp_path)
    assert not (tmp_path / "QQQ").exists()


# ── repo config sanity ───────────────────────────────────────────────────────

def test_repo_broker_yaml_has_futu_block_with_expected_defaults():
    settings = load_connection_settings()
    assert settings["host"] == "127.0.0.1"
    assert settings["port"] == 11111
    assert settings["market_prefix"] == "US"
