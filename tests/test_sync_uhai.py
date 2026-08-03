"""
scripts/sync_uhai.py — distillation + watermark + offline/online delivery.
All tests run inside a temp cwd (monkeypatch.chdir) so nothing touches this
repo's real backtests/data/configs, and reload the module's path constants
against that temp cwd since they're Path(...) literals evaluated at import
time.
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture()
def sync_uhai(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import scripts.sync_uhai as mod

    importlib.reload(mod)
    (tmp_path / "backtests" / "reports").mkdir(parents=True)
    (tmp_path / "backtests" / "logs").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    return mod


def _write_report(sync_uhai, generated_at="2026-07-29T00:00:00+00:00", decision="NO-GO"):
    payload = {
        "generated_at": generated_at,
        "results": [
            {
                "signal": "sweep_reclaim",
                "decision": decision,
                "window": "2025-06-01 .. 2025-12-01",
                "data_label": "demo",
                "wfo_pass_ratio": 0.4,
                "oos_sharpe_mean": -0.2,
                "gates": {"wfo": False, "monte_carlo": True},
                "candidate_params": {"sweep_min_atr": 0.15, "reclaim_bars": 3},
            },
        ],
    }
    sync_uhai.REPORT_JSON_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _write_universe(sync_uhai, computed_at="2026-07-28"):
    sync_uhai.UNIVERSE_CONFIG_PATH.write_text(
        f"fixed_universe:\n  symbols: [AAPL, MSFT]\n  top_n: 20\n"
        f"  ranking_metric: trailing_60d_avg_dollar_volume\n  computed_at: '{computed_at}'\n"
        f"  source_pool: manual\n",
        encoding="utf-8",
    )


def test_no_sources_yields_no_events(sync_uhai):
    events, _state = sync_uhai.collect_events(reset=False)
    assert events == []


def test_backtest_report_distills_result_and_param_events(sync_uhai):
    _write_report(sync_uhai)
    events, state = sync_uhai.collect_events(reset=False)
    predicates = {e["predicate"] for e in events}
    assert predicates == {"hasBacktestResult", "hasParameterSet"}
    result_event = next(e for e in events if e["predicate"] == "hasBacktestResult")
    assert result_event["subject"] == "signal:sweep_reclaim"
    obj = json.loads(result_event["object_json"])
    assert obj["decision"] == "NO-GO"
    assert state["last_report_generated_at"] == "2026-07-29T00:00:00+00:00"


def test_rerun_with_unchanged_report_yields_no_new_events(sync_uhai):
    _write_report(sync_uhai)
    events1, state1 = sync_uhai.collect_events(reset=False)
    sync_uhai._save_state(state1)
    assert len(events1) == 2

    events2, _state2 = sync_uhai.collect_events(reset=False)
    assert events2 == []


def test_new_report_run_after_watermark_is_synced_again(sync_uhai):
    _write_report(sync_uhai, generated_at="2026-07-29T00:00:00+00:00")
    events1, state1 = sync_uhai.collect_events(reset=False)
    sync_uhai._save_state(state1)

    _write_report(sync_uhai, generated_at="2026-07-30T00:00:00+00:00", decision="GO")
    events2, _state2 = sync_uhai.collect_events(reset=False)
    assert len(events2) == 2
    result_event = next(e for e in events2 if e["predicate"] == "hasBacktestResult")
    assert json.loads(result_event["object_json"])["decision"] == "GO"


def test_promotion_history_incremental_sync(sync_uhai):
    record1 = {"strategy": "pairs_trading", "decision": "REJECTED", "reason": "no improvement",
               "candidate_oos_sharpe": 0.1, "baseline_oos_sharpe": 0.3, "gates": {"wfo": True},
               "config_written": False, "timestamp": "2026-07-01T00:00:00Z"}
    sync_uhai.PROMOTION_HISTORY_PATH.write_text(json.dumps(record1) + "\n", encoding="utf-8")

    events1, state1 = sync_uhai.collect_events(reset=False)
    assert len(events1) == 1
    assert events1[0]["subject"] == "strategy:pairs_trading"
    sync_uhai._save_state(state1)

    record2 = {"strategy": "pairs_trading", "decision": "PROMOTED", "reason": "beat baseline",
               "candidate_oos_sharpe": 0.5, "baseline_oos_sharpe": 0.3, "gates": {"wfo": True},
               "config_written": True, "timestamp": "2026-07-02T00:00:00Z"}
    with sync_uhai.PROMOTION_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record2) + "\n")

    events2, _state2 = sync_uhai.collect_events(reset=False)
    assert len(events2) == 1
    assert json.loads(events2[0]["object_json"])["decision"] == "PROMOTED"


def test_universe_config_watermarked_by_computed_at(sync_uhai):
    _write_universe(sync_uhai, computed_at="2026-07-28")
    events1, state1 = sync_uhai.collect_events(reset=False)
    assert len(events1) == 1
    assert events1[0]["subject"] == "universe:fixed_universe"
    sync_uhai._save_state(state1)

    events2, _state2 = sync_uhai.collect_events(reset=False)
    assert events2 == []


def test_reset_flag_ignores_watermark(sync_uhai):
    _write_report(sync_uhai)
    events1, state1 = sync_uhai.collect_events(reset=False)
    sync_uhai._save_state(state1)
    assert len(events1) == 2

    events2, _state2 = sync_uhai.collect_events(reset=True)
    assert len(events2) == 2  # resynced despite unchanged generated_at


def test_offline_export_writes_jsonl(sync_uhai):
    _write_report(sync_uhai)
    events, _state = sync_uhai.collect_events(reset=False)
    out_path = sync_uhai._write_offline_export(events)
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(events)
    assert json.loads(lines[0])["subject"] == "signal:sweep_reclaim"


def test_push_to_greycat_failure_returns_false(sync_uhai, monkeypatch):
    import httpx

    def _raise_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _raise_post)
    ok = sync_uhai._push_to_greycat(
        [{"subject": "s", "predicate": "p", "object_json": "{}", "source": "test", "ts": "2026-07-29T00:00:00Z"}],
        "http://localhost:8080",
        None,
    )
    assert ok is False


def test_push_to_greycat_success_encodes_ts_as_epoch_micros(sync_uhai, monkeypatch):
    import httpx

    captured = {}

    class _FakeResponse:
        text = "2"

        def raise_for_status(self):
            pass

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", _fake_post)
    ok = sync_uhai._push_to_greycat(
        [{"subject": "s", "predicate": "p", "object_json": "{}", "source": "test", "ts": "2026-07-29T00:00:00+00:00"}],
        "http://localhost:8080/",
        "sometoken",
    )
    assert ok is True
    assert captured["url"] == "http://localhost:8080/microstructure::ingestMicroEvents"
    assert captured["headers"] == {"Authorization": "sometoken"}
    assert isinstance(captured["json"]["events"][0]["ts"], int)
    assert captured["json"]["events"][0]["ts"] > 0


def test_main_offline_mode_end_to_end(sync_uhai, capsys):
    _write_report(sync_uhai)
    sync_uhai.main([])
    out = capsys.readouterr().out
    assert "offline" in out
    exported = list(sync_uhai.SYNC_DIR.glob("*.jsonl"))
    assert len(exported) == 1
    assert sync_uhai.STATE_PATH.exists()


def test_main_prints_nothing_new_when_no_sources(sync_uhai, capsys):
    sync_uhai.main([])
    out = capsys.readouterr().out
    assert "nothing new" in out
