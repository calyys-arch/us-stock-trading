"""
promotion.py tests: the decision matrix (gates -> improvement -> promote),
the comment-preserving strategy.yaml write-back, refusal to write
new/forbidden keys, and the append-only JSONL history.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from python.backtest.promotion import (
    PromotionRecord,
    append_history,
    evaluate_and_promote,
    write_strategy_config,
)

_SAMPLE_CONFIG = """\
# Top-of-file comment that MUST survive a write-back.
pairs_trading:
  enabled: true
  auto_execute: false   # observe-only — never auto-written
  entry_z: 2.0          # entry threshold comment
  exit_z: 0.5
"""


@pytest.fixture
def config_path(tmp_path) -> Path:
    path = tmp_path / "strategy.yaml"
    path.write_text(_SAMPLE_CONFIG, encoding="utf-8")
    return path


def test_write_back_updates_values_and_preserves_comments(config_path):
    write_strategy_config("pairs_trading", {"entry_z": 2.5, "exit_z": 0.0}, path=config_path)
    text = config_path.read_text(encoding="utf-8")
    assert "Top-of-file comment" in text
    assert "entry threshold comment" in text
    assert "observe-only" in text
    cfg = yaml.safe_load(text)["pairs_trading"]
    assert cfg["entry_z"] == 2.5
    assert cfg["exit_z"] == 0.0
    assert cfg["auto_execute"] is False


def test_write_back_refuses_new_keys(config_path):
    with pytest.raises(ValueError, match="new keys"):
        write_strategy_config("pairs_trading", {"invented_knob": 1.0}, path=config_path)


def test_write_back_refuses_forbidden_keys(config_path):
    with pytest.raises(ValueError, match="forbidden"):
        write_strategy_config("pairs_trading", {"auto_execute": True}, path=config_path)


def test_write_back_unknown_strategy(config_path):
    with pytest.raises(KeyError):
        write_strategy_config("no_such_strategy", {"entry_z": 1.0}, path=config_path)


def _promote(config_path, history_path, **overrides):
    kwargs = dict(
        strategy_name="pairs_trading",
        candidate_params={"entry_z": 2.5},
        baseline_params={"entry_z": 2.0},
        candidate_oos_sharpe=1.0,
        baseline_oos_sharpe=0.5,
        gates={"wfo_go": True, "monte_carlo": True},
        wfo_summary={"decision": "GO", "total_folds": 4},
        min_improvement=0.0,
        write_config=True,
        config_path=config_path,
        history_path=history_path,
    )
    kwargs.update(overrides)
    return evaluate_and_promote(**kwargs)


def test_promoted_when_gates_pass_and_improves(config_path, tmp_path):
    history = tmp_path / "history.jsonl"
    record = _promote(config_path, history)
    assert record.decision == "PROMOTED"
    assert record.config_written is True
    assert yaml.safe_load(config_path.read_text())["pairs_trading"]["entry_z"] == 2.5


def test_rejected_on_failed_gate_even_with_better_sharpe(config_path, tmp_path):
    record = _promote(config_path, tmp_path / "h.jsonl",
                      gates={"wfo_go": False, "monte_carlo": True})
    assert record.decision == "REJECTED"
    assert "wfo_go" in record.reason
    assert record.config_written is False
    assert yaml.safe_load(config_path.read_text())["pairs_trading"]["entry_z"] == 2.0


def test_rejected_when_not_beating_incumbent(config_path, tmp_path):
    record = _promote(config_path, tmp_path / "h.jsonl",
                      candidate_oos_sharpe=0.5, baseline_oos_sharpe=0.5)
    assert record.decision == "REJECTED"
    assert record.config_written is False


def test_rejected_when_improvement_exactly_equals_min_improvement_boundary(config_path, tmp_path):
    """Regression for the 2026-08-15 round-2 audit
    (backtests/reports/backtest_engine_audit_round2.md): with a NON-zero
    `min_improvement`, an improvement EXACTLY AT the threshold must still
    be REJECTED (configs/goal.yaml's own comment on
    `live_promotion.min_oos_sharpe_improvement`: "any STRICT improvement
    qualifies" — a tie at the margin is not a strict improvement), and an
    improvement even one float ULP above it must PROMOTE. This is the
    off-by-one boundary check the audit brief specifically asked for."""
    exact = _promote(config_path, tmp_path / "exact.jsonl",
                      candidate_oos_sharpe=0.7, baseline_oos_sharpe=0.5, min_improvement=0.2)
    assert exact.decision == "REJECTED"

    just_above = _promote(config_path, tmp_path / "above.jsonl",
                          candidate_oos_sharpe=0.70001, baseline_oos_sharpe=0.5, min_improvement=0.2)
    assert just_above.decision == "PROMOTED"


def test_rejected_when_candidate_equals_baseline(config_path, tmp_path):
    record = _promote(config_path, tmp_path / "h.jsonl",
                      candidate_params={"entry_z": 2.0}, baseline_params={"entry_z": 2.0})
    assert record.decision == "REJECTED"
    assert "no change" in record.reason


def test_no_write_flag_promotes_without_touching_config(config_path, tmp_path):
    record = _promote(config_path, tmp_path / "h.jsonl", write_config=False)
    assert record.decision == "PROMOTED"
    assert record.config_written is False
    assert yaml.safe_load(config_path.read_text())["pairs_trading"]["entry_z"] == 2.0


def test_history_is_append_only_jsonl(config_path, tmp_path):
    history = tmp_path / "history.jsonl"
    _promote(config_path, history)
    _promote(config_path, history, gates={"wfo_go": False})
    lines = history.read_text().strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["decision"] == "PROMOTED"
    assert second["decision"] == "REJECTED"
    assert first["strategy"] == "pairs_trading"


def test_append_history_creates_parent_dirs(tmp_path):
    record = PromotionRecord(
        timestamp="t", strategy="s", decision="REJECTED", reason="r",
        candidate_params={}, baseline_params={}, candidate_oos_sharpe=0.0,
        baseline_oos_sharpe=0.0, gates={}, wfo_summary={},
    )
    path = tmp_path / "nested" / "dir" / "history.jsonl"
    append_history(record, path=path)
    assert path.exists()
