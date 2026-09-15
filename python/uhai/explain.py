"""
UHAI-backed explanation layer: turn one raw promotion_history.jsonl record
(a WFO/gate verdict) into a short, human-readable diagnosis.

Why this exists
----------------
docs/uhai_integration_review.md (2026-09-10, code-verified) concluded UHAI's
two genuine gaps relative to scripts/self_improve_loop.py were (1) smarter
search — already covered by python/uhai/client.py's suggest_params() — and
(2) cross-round memory + explanation. assess_degradation()/load_history()
already cover the memory half. This module is the explanation half: reading
"REJECTED: gates failed: monte_carlo_p5_sharpe" is unambiguous but terse;
this asks UHAI's model router (POST /generate, model_id="gpt-5.5" now that
the Universal-Hybrid-AI upstream enabled real cloud routing instead of a
hardcoded local_llm — see that repo's commit 011aa40) to restate the SAME
facts in one short paragraph a human can read without cross-referencing
gate names against param_guard.py.

Hard contract: this NEVER invents a verdict. The LLM is asked to restate
numbers already in the record, not to re-judge the strategy — a rejection
stays a rejection regardless of what prose comes back. See _build_prompt's
explicit "do not invent numbers" instruction and _template_explanation's
role as the ground truth this exists to make more readable, not replace.

Fail-safe by design, matching python/uhai/client.py's contract ("a missing
suggester must never stop an optimization run"): every failure mode below
falls back to _template_explanation() — a plain, LLM-free, always-available
rendering of the same record — rather than raising or returning None.
Callers can therefore always print *something* useful:
  - UHAI_EXPLAIN_DISABLE set                    -> template only
  - sidecar not running / not identified        -> template only
  - /generate times out, errors, or 4xx/5xx      -> template only
  - /generate replies with no usable `answer`    -> template only

Environment:
    UHAI_EXPLAIN_MODEL     model_id sent to /generate. Default "gpt-5.5".
                            Set to "local_llm" to avoid any cloud call.
    UHAI_EXPLAIN_DISABLE   any non-empty value forces the template path,
                            e.g. for CI or an offline research session.
    (UHAI_PYTHON_SVC_URL / SIDECAR_API_KEY are read via client.py, same as
    suggest_params().)
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from python.uhai.client import is_uhai_sidecar, sidecar_url

log = logging.getLogger(__name__)

EXPLAIN_TIMEOUT_SECONDS = 30.0
DEFAULT_EXPLAIN_MODEL = "gpt-5.5"
DEFAULT_MAX_TOKENS = 300


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("SIDECAR_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _template_explanation(strategy: str, record: dict[str, Any]) -> str:
    """Local, LLM-free rendering of a raw promotion_history.jsonl record.

    Always available (no network, no sidecar) — this is the ground truth
    the LLM path exists to make more readable, not a degraded substitute
    for it. Every explain_decision() failure path returns exactly this.
    """
    decision = record.get("decision", "UNKNOWN")
    gates = record.get("gates") or {}
    failed = sorted(name for name, ok in gates.items() if not ok)
    reason = record.get("reason", "")

    parts = [f"{strategy}: {decision}"]
    if failed:
        parts.append(f"failed gate(s): {', '.join(failed)}")
    elif gates:
        parts.append("all gates passed")
    if reason:
        parts.append(f"reason on file: {reason}")

    cand = record.get("candidate_oos_sharpe")
    base = record.get("baseline_oos_sharpe")
    if isinstance(cand, (int, float)) and isinstance(base, (int, float)):
        parts.append(f"candidate OOS Sharpe {cand:+.3f} vs baseline {base:+.3f}")

    regime = (record.get("extra") or {}).get("market_regime")
    if regime and regime != "unknown":
        parts.append(f"regime: {regime}")

    return " | ".join(parts)


def _build_prompt(strategy: str, record: dict[str, Any]) -> str:
    gates = record.get("gates") or {}
    gate_lines = "\n".join(
        f"  - {name}: {'PASS' if ok else 'FAIL'}" for name, ok in gates.items()
    ) or "  (no gate detail recorded)"

    return (
        "You are explaining one walk-forward-validation verdict from a "
        "quantitative trading research log to a researcher who already "
        "knows the methodology (WFO, Monte Carlo, Chan-discipline parameter "
        "limits) and wants a precise, non-hyped summary of THIS ONE "
        "decision — not general advice about trading strategies.\n\n"
        f"Strategy: {strategy}\n"
        f"Decision: {record.get('decision', '(unknown)')}\n"
        f"Reason on file: {record.get('reason') or '(none recorded)'}\n"
        f"Market regime during the tested window: "
        f"{(record.get('extra') or {}).get('market_regime', 'unknown')}\n"
        f"Candidate OOS Sharpe: {record.get('candidate_oos_sharpe')}\n"
        f"Baseline (currently-live) OOS Sharpe: {record.get('baseline_oos_sharpe')}\n"
        f"Candidate parameters tested: {record.get('candidate_params')}\n"
        f"Gate results:\n{gate_lines}\n\n"
        "In 3-4 sentences: (1) state which gate(s), if any, actually decided "
        "this outcome, (2) say whether the candidate looks like a genuine "
        "improvement, statistical noise, or an actively worse fit versus the "
        "baseline, and (3) if REJECTED, note briefly whether this looks like "
        "a dead end or worth revisiting with more data. Use ONLY the numbers "
        "given above — do not invent any. Do not recommend loosening gates "
        "or Chan-discipline parameter limits to force a pass."
    )


def explain_decision(
    strategy: str,
    record: dict[str, Any],
    *,
    model_id: str | None = None,
) -> str:
    """Human-readable explanation of one promotion_history.jsonl record.

    Always returns a non-empty string; never raises. See module docstring
    for the full list of fallback conditions.
    """
    if os.getenv("UHAI_EXPLAIN_DISABLE"):
        return _template_explanation(strategy, record)

    if not is_uhai_sidecar():
        log.info("UHAI: no sidecar at %s — using local template explanation", sidecar_url())
        return _template_explanation(strategy, record)

    model = model_id or os.getenv("UHAI_EXPLAIN_MODEL", DEFAULT_EXPLAIN_MODEL)
    prompt = _build_prompt(strategy, record)

    try:
        response = httpx.post(
            f"{sidecar_url()}/generate",
            json={
                "model_id": model,
                "prompt": prompt,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": 0.2,
            },
            headers=_headers(),
            timeout=EXPLAIN_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        log.warning("UHAI: /generate failed (%s) — using local template explanation", exc)
        return _template_explanation(strategy, record)
    except ValueError as exc:
        log.warning(
            "UHAI: /generate returned undecodable JSON (%s) — using local template explanation",
            exc,
        )
        return _template_explanation(strategy, record)

    if not isinstance(payload, dict):
        log.warning("UHAI: /generate returned a non-object payload — using local template explanation")
        return _template_explanation(strategy, record)

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        log.warning("UHAI: /generate returned no usable answer — using local template explanation")
        return _template_explanation(strategy, record)

    return answer.strip()
