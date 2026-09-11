"""
UHAI client: smarter parameter search on top of this repo's own history.

Two things UHAI adds to scripts/self_improve_loop.py, and where each lives:

  1. Smarter search — a Gaussian-Process/Expected-Improvement suggester,
     hosted by UHAI's Python sidecar (POST /optimize/suggest-params). The
     history it fits on is read LOCALLY from
     backtests/logs/promotion_history.jsonl (python/uhai/history.py), not
     fetched back out of GreyCat: GCL has no JSON parser, so the
     `object_json` payload sync_uhai.py writes into the triple store cannot
     be read back as structured parameters there. Going straight to the
     sidecar also removes a network hop and lets this work with nothing but
     the sidecar running.

  2. Cross-iteration memory — that same local history, plus degradation
     detection over it (`assess_degradation`). GreyCat keeps its role as the
     relationship-aware store of distilled findings (scripts/sync_uhai.py),
     which is what it is actually good at.

Everything here degrades to "no suggestions": the loop must run with the
sidecar offline, and it does — the caller falls back to the fixed grid.

Environment:
    UHAI_PYTHON_SVC_URL  UHAI sidecar base URL. Default http://127.0.0.1:8092.

        NOT 8082, which is UHAI's own sidecar default (see
        src/core/sidecar_client.gcl): in THIS repo 8082 is already the
        dashboard's port (scripts/start_dashboard.py, README.md, and
        scripts/watch_paper_session.py's hard-coded DASH_URL). Pointing a
        suggestion POST at that server would aim it at an API surface that
        includes /api/engine/start and /api/positions/flatten_all. Hence a
        distinct default port, a distinct variable name from the
        PYTHON_SVC_URL that UHAI's GCL reads, and the identity probe below.

        When running the sidecar for this repo, start it on 8092:
            PYTHON_SVC_PORT=8092 python python-svc/main.py

    SIDECAR_API_KEY      optional; sent as X-API-Key when set
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from python.uhai.history import HistoryRecord, load_history, summarize, to_sidecar_payload

log = logging.getLogger(__name__)

# 127.0.0.1, not localhost: see src/core/sidecar_client.gcl — resolving
# "localhost" to both ::1 and 127.0.0.1 doubles the cost of every
# sidecar-is-offline fallback, and that fallback is the normal case here.
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8092"
SUGGEST_TIMEOUT_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 3.0


def sidecar_url() -> str:
    return os.getenv("UHAI_PYTHON_SVC_URL") or DEFAULT_SIDECAR_URL


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("SIDECAR_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def is_uhai_sidecar(url: str | None = None) -> bool:
    """Confirm `url` is the UHAI sidecar and not some other local service.

    A port on localhost is not proof of identity — this repo runs its own
    FastAPI dashboard, and a suggestion POST that lands there would be aimed
    at trading controls. UHAI's /health reports its feature flags, so a reply
    carrying them is the cheap positive identification.
    """
    base = url or sidecar_url()
    try:
        response = httpx.get(f"{base}/health", timeout=PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return False

    if not isinstance(payload, dict):
        return False
    # /health on the sidecar returns a liveness flag plus the model feature
    # flags; the dashboard's routes have no such shape (its /health 404s).
    return "status" in payload or "features" in payload or "models" in payload


def bounds_from_grid_spec(grid_spec: dict[str, list]) -> dict[str, list[float]]:
    """configs/param_grids.yaml's per-strategy block -> {name: [low, high]}.

    Only numeric axes are returned. A boolean axis (e.g. orb_vwap's
    `vwap_side_filter`) has no continuous interior to interpolate, and
    handing it to a Real-valued Gaussian Process would produce meaningless
    fractional "suggestions" like 0.37; those axes stay grid-only.

    A single-valued axis is also dropped: low == high gives the optimizer a
    zero-width dimension, which skopt rejects outright.
    """
    bounds: dict[str, list[float]] = {}
    for name, values in (grid_spec or {}).items():
        if not isinstance(values, list) or not values:
            continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            continue
        low, high = float(min(values)), float(max(values))
        if low >= high:
            continue
        bounds[name] = [low, high]
    return bounds


# Decimal places kept on a continuous suggestion. The grids this fits on are
# coarse (three values per axis over a few years of bars); carrying a
# Gaussian Process's full float precision into configs/strategy.yaml would
# assert a resolution the data cannot support.
_SUGGESTION_DECIMALS = 3


def coerce_to_grid_types(
    suggestion: dict[str, float],
    grid_spec: dict[str, list],
) -> dict[str, float]:
    """Snap a continuous suggestion back onto each axis's real domain.

    An axis whose grid values are all integers is an integer axis —
    xsection_mean_reversion's `lookback_days` is a count of days, and a
    suggestion of 2.7 days is not a parameter the strategy can take. Those
    are rounded to int; everything else is rounded to `_SUGGESTION_DECIMALS`.
    """
    coerced: dict[str, float] = {}
    for name, value in suggestion.items():
        values = grid_spec.get(name)
        is_integer_axis = (
            isinstance(values, list)
            and bool(values)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in values)
        )
        if is_integer_axis:
            coerced[name] = int(round(float(value)))
        else:
            coerced[name] = round(float(value), _SUGGESTION_DECIMALS)
    return coerced


def suggest_params(
    strategy: str,
    current_params: dict,
    grid_spec: dict[str, list],
    *,
    n_suggestions: int = 5,
    include_synthetic_history: bool = False,
) -> list[dict[str, float]]:
    """Ask the sidecar for parameter sets worth testing next.

    Args:
        strategy: strategy name, used to select its own history.
        current_params: the incumbent config (passed through for context).
        grid_spec: this strategy's block from configs/param_grids.yaml.
            Its min/max form the search box, so a suggestion can never
            propose a value outside the range the grid's own author already
            sanctioned; its value types decide integer vs continuous axes.
        n_suggestions: how many points to ask for.
        include_synthetic_history: fit on --demo runs too. Only for demos.

    Returns:
        Parameter dicts, deduplicated and type-coerced. Empty on any
        failure — the caller keeps its grid.

    Raises:
        Nothing. Transport and protocol failures are logged and swallowed,
        because a missing suggester must never stop an optimization run.
    """
    grid_bounds = bounds_from_grid_spec(grid_spec)
    if not grid_bounds:
        log.info("UHAI: no continuous grid axes for %s — nothing to suggest over", strategy)
        return []

    # Identify the service before sending it anything. See this module's
    # docstring: an unidentified listener on a local port is not assumed to
    # be the sidecar.
    if not is_uhai_sidecar():
        log.info("UHAI: no sidecar at %s — fixed grid only", sidecar_url())
        return []

    history = load_history(
        strategy,
        include_synthetic=include_synthetic_history,
        param_keys=set(grid_bounds),
    )
    log.info("UHAI: %s history %s", strategy, summarize(history))

    body = {
        "strategy": strategy,
        "current_params": {
            k: v for k, v in current_params.items() if isinstance(v, (int, float))
        },
        "grid_bounds": grid_bounds,
        "history": to_sidecar_payload(history),
        "n_suggestions": n_suggestions,
    }

    try:
        response = httpx.post(
            f"{sidecar_url()}/optimize/suggest-params",
            json=body,
            headers=_headers(),
            timeout=SUGGEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        log.warning("UHAI: sidecar unavailable (%s) — falling back to the fixed grid", exc)
        return []
    except ValueError as exc:
        log.warning("UHAI: sidecar returned undecodable JSON (%s)", exc)
        return []

    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list):
        log.warning("UHAI: sidecar reply has no suggestions list")
        return []

    explanations = payload.get("explanations") or []

    # Coerce onto each axis's real domain, then drop duplicates that rounding
    # collapsed together. This is common on the GP path, not exceptional: once
    # the surrogate has localised a region it will happily propose several
    # points that agree to three decimals.
    out: list[dict[str, float]] = []
    kept_explanations: list[str] = []
    seen: set[tuple] = set()
    n_collapsed = 0

    for i, raw in enumerate(suggestions):
        if not isinstance(raw, dict) or not raw:
            continue
        if set(raw) != set(grid_bounds):
            log.warning("UHAI: suggestion has unexpected axes %s — dropping", sorted(raw))
            continue
        coerced = coerce_to_grid_types(raw, grid_spec)
        key = tuple(sorted(coerced.items()))
        if key in seen:
            n_collapsed += 1
            continue
        seen.add(key)
        out.append(coerced)
        kept_explanations.append(explanations[i] if i < len(explanations) else "")

    log.info(
        "UHAI: %d suggestion(s) via %s from %d history point(s)%s",
        len(out),
        payload.get("method", "unknown"),
        payload.get("history_count", 0),
        f" ({n_collapsed} more collapsed onto a duplicate when rounded)" if n_collapsed else "",
    )
    for suggestion, explanation in zip(out, kept_explanations):
        log.info("UHAI:   %s — %s", suggestion, explanation)

    return out


# ── cross-iteration memory: is the incumbent decaying? ───────────────────────

@dataclass(frozen=True)
class DegradationVerdict:
    """Whether the live parameter set is scoring worse than it used to."""

    is_degraded: bool
    recent_mean_sharpe: float
    earlier_mean_sharpe: float
    n_recent: int
    n_earlier: int
    reason: str


def assess_degradation(
    strategy: str,
    *,
    recent_runs: int = 3,
    min_earlier_runs: int = 3,
    decline_threshold: float = 0.20,
    include_synthetic_history: bool = False,
) -> DegradationVerdict:
    """Compare the LIVE parameters' recent re-measured OOS Sharpe against
    everything before.

    This reads `baseline_oos_sharpe` — the currently-deployed parameters,
    re-measured on each run's window — NOT `candidate_oos_sharpe`, which is
    whatever the search happened to try that run and, on a REJECTED run, was
    thrown away. A search that keeps trying (and rejecting) better-looking
    candidates says nothing about whether the strategy actually running
    right now is still working; only the baseline trend answers that.

    This is a *staleness signal*, not a verdict on the strategy: it says
    "the evidence behind today's live parameters is getting worse, re-run
    the loop", and the loop's own gates then decide whether anything
    actually changes. Deliberately conservative — it needs both a recent
    sample and an earlier one before it will claim anything.

    `decline_threshold` is a fractional drop (0.20 = the recent mean is 20%
    below the earlier mean). When the earlier mean is at or below zero a
    percentage is meaningless, so the comparison falls back to an absolute
    one.
    """
    all_history = load_history(strategy, include_synthetic=include_synthetic_history)
    # A malformed record's baseline_oos_sharpe is None (see history.py) —
    # exclude rather than let it corrupt the mean.
    history = [r for r in all_history if r.baseline_oos_sharpe is not None]

    if len(history) < recent_runs + min_earlier_runs:
        return DegradationVerdict(
            is_degraded=False,
            recent_mean_sharpe=0.0,
            earlier_mean_sharpe=0.0,
            n_recent=0,
            n_earlier=0,
            reason=(
                f"insufficient history: {len(history)} record(s) with a baseline Sharpe, "
                f"need {recent_runs + min_earlier_runs}"
            ),
        )

    recent = history[-recent_runs:]
    earlier = history[:-recent_runs]

    recent_mean = sum(r.baseline_oos_sharpe for r in recent) / len(recent)
    earlier_mean = sum(r.baseline_oos_sharpe for r in earlier) / len(earlier)

    if earlier_mean > 0:
        decline = (earlier_mean - recent_mean) / earlier_mean
        is_degraded = decline >= decline_threshold
        # A negative decline is an improvement; saying "-42% decline" reads as
        # a problem when it is the opposite.
        change = (
            f"{decline * 100:.1f}% below" if decline >= 0
            else f"{-decline * 100:.1f}% above"
        )
        reason = (
            f"recent mean OOS Sharpe {recent_mean:+.3f} vs earlier {earlier_mean:+.3f} "
            f"({change} earlier, flagged at {decline_threshold * 100:.0f}% below)"
        )
    else:
        # Nothing to erode from; only a further absolute drop is meaningful.
        is_degraded = recent_mean < earlier_mean
        reason = (
            f"recent mean OOS Sharpe {recent_mean:+.3f} vs earlier {earlier_mean:+.3f} "
            f"(earlier mean not positive — absolute comparison)"
        )

    return DegradationVerdict(
        is_degraded=is_degraded,
        recent_mean_sharpe=recent_mean,
        earlier_mean_sharpe=earlier_mean,
        n_recent=len(recent),
        n_earlier=len(earlier),
        reason=reason,
    )


def days_since_last_promotion(
    strategy: str,
    *,
    include_synthetic_history: bool = False,
) -> float | None:
    """Age in days of the newest PROMOTED record, or None if never promoted."""
    import time as _time

    history = load_history(strategy, include_synthetic=include_synthetic_history)
    promoted = [r for r in history if r.decision == "PROMOTED"]
    if not promoted:
        return None
    return (_time.time() - promoted[-1].timestamp) / 86400.0
