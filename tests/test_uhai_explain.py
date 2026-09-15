"""
Guards on python/uhai/explain.py's contract: an explanation must never
become a hard dependency, must never invent a verdict, and must always
return a usable string.
"""
from __future__ import annotations

import httpx
import pytest

from python.uhai import explain as uhai_explain

REJECTED_RECORD = {
    "timestamp": "2026-09-01T00:00:00+00:00",
    "strategy": "sweep_reclaim",
    "decision": "REJECTED",
    "reason": "gates failed: monte_carlo_p5_sharpe",
    "candidate_params": {"or_minutes": 15},
    "candidate_oos_sharpe": 0.42,
    "baseline_oos_sharpe": 0.58,
    "gates": {
        "wfo_go": True,
        "oos_drawdown_within_limit": True,
        "has_oos_trades": True,
        "monte_carlo_p5_sharpe": False,
    },
    "extra": {"market_regime": "Bull"},
}

PROMOTED_RECORD = {
    "timestamp": "2026-09-01T00:00:00+00:00",
    "strategy": "sweep_reclaim",
    "decision": "PROMOTED",
    "reason": "",
    "candidate_params": {"or_minutes": 30},
    "candidate_oos_sharpe": 0.9,
    "baseline_oos_sharpe": 0.5,
    "gates": {"wfo_go": True, "oos_drawdown_within_limit": True,
              "has_oos_trades": True, "monte_carlo_p5_sharpe": True},
    "extra": {},
}


# ── template fallback: always available, no network ──────────────────────────

def test_template_names_the_failed_gate():
    text = uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)
    assert "sweep_reclaim" in text
    assert "REJECTED" in text
    assert "monte_carlo_p5_sharpe" in text
    # Passing gates must not be listed among the failed ones.
    failed_segment = text.split("failed gate(s):", 1)[1].split("|", 1)[0]
    assert "wfo_go" not in failed_segment


def test_template_reports_all_gates_passed_when_none_failed():
    text = uhai_explain._template_explanation("sweep_reclaim", PROMOTED_RECORD)
    assert "all gates passed" in text
    assert "PROMOTED" in text


def test_template_includes_sharpe_comparison():
    text = uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)
    assert "0.420" in text
    assert "0.580" in text


def test_template_handles_missing_optional_fields_gracefully():
    bare = {"strategy": "x", "decision": "REJECTED"}
    text = uhai_explain._template_explanation("x", bare)
    assert "REJECTED" in text


# ── explain_decision: fallback triggers ───────────────────────────────────────

def test_explain_disable_env_forces_template(monkeypatch):
    monkeypatch.setenv("UHAI_EXPLAIN_DISABLE", "1")
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: (_ for _ in ()).throw(
        AssertionError("must not probe the sidecar when explanation is disabled")))
    result = uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)
    assert result == uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)


def test_no_sidecar_falls_back_to_template(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: False)
    result = uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)
    assert result == uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)


def test_generate_http_error_falls_back_to_template(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: True)

    def _refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _refuse)
    result = uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)
    assert result == uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)


def test_generate_empty_answer_falls_back_to_template(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: True)

    class _EmptyResponse:
        def raise_for_status(self): return None
        def json(self): return {"answer": "   ", "model_used": "gpt-5.5"}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _EmptyResponse())
    result = uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)
    assert result == uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)


def test_generate_non_dict_payload_falls_back_to_template(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: True)

    class _ListResponse:
        def raise_for_status(self): return None
        def json(self): return ["not", "a", "dict"]

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _ListResponse())
    result = uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)
    assert result == uhai_explain._template_explanation("sweep_reclaim", REJECTED_RECORD)


# ── explain_decision: happy path ──────────────────────────────────────────────

def test_generate_success_returns_llm_answer(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: True)

    captured = {}

    class _OkResponse:
        def raise_for_status(self): return None
        def json(self): return {"answer": "  The Monte Carlo gate failed.  ", "model_used": "gpt-5.5"}

    def _post(url, json, headers, timeout):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return _OkResponse()

    monkeypatch.setattr(httpx, "post", _post)
    result = uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)

    assert result == "The Monte Carlo gate failed."
    assert captured["url"].endswith("/generate")
    assert captured["json"]["model_id"] == "gpt-5.5"
    assert "monte_carlo_p5_sharpe" in captured["json"]["prompt"]
    # The prompt must not invent numbers -- it should only ever restate what
    # is already in the record.
    assert "0.42" in captured["json"]["prompt"]


def test_explain_model_env_var_overrides_default(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setenv("UHAI_EXPLAIN_MODEL", "local_llm")
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: True)

    captured = {}

    class _OkResponse:
        def raise_for_status(self): return None
        def json(self): return {"answer": "ok"}

    def _post(url, json, headers, timeout):  # noqa: A002
        captured["json"] = json
        return _OkResponse()

    monkeypatch.setattr(httpx, "post", _post)
    uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD)
    assert captured["json"]["model_id"] == "local_llm"


def test_explicit_model_id_overrides_env(monkeypatch):
    monkeypatch.delenv("UHAI_EXPLAIN_DISABLE", raising=False)
    monkeypatch.setenv("UHAI_EXPLAIN_MODEL", "local_llm")
    monkeypatch.setattr(uhai_explain, "is_uhai_sidecar", lambda: True)

    captured = {}

    class _OkResponse:
        def raise_for_status(self): return None
        def json(self): return {"answer": "ok"}

    def _post(url, json, headers, timeout):  # noqa: A002
        captured["json"] = json
        return _OkResponse()

    monkeypatch.setattr(httpx, "post", _post)
    uhai_explain.explain_decision("sweep_reclaim", REJECTED_RECORD, model_id="gpt-5.5")
    assert captured["json"]["model_id"] == "gpt-5.5"
