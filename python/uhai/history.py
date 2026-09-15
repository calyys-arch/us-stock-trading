"""
Cross-iteration memory: read this repo's own promotion history back as
training data for UHAI's parameter suggester.

The history already exists — python/backtest/promotion.py appends one JSON
object per decision to backtests/logs/promotion_history.jsonl, including the
parameter set tested and the OOS Sharpe it earned. That file IS the memory;
this module just makes it queryable.

Two filters matter and are not optional:

  - data_source: a --demo run records `"data_source": "synthetic"`. Feeding
    synthetic-data Sharpes into the suggester that proposes parameters for
    real money would be worse than having no memory at all, so real runs
    read real history only (`include_synthetic=False`, the default).

  - strategy: pairs_trading's entry_z means nothing to
    xsection_mean_reversion, and the two have disjoint parameter names.

Records are keyed on `candidate_params` (what was actually tested) and
`candidate_oos_sharpe` (what it scored), NOT on the promotion verdict: a
REJECTED candidate is still a valid observation of "this corner of the
parameter space scores about this much", which is exactly what a Gaussian
Process needs. Filtering to PROMOTED-only would leave the model blind to
every region known to be bad.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

PROMOTION_HISTORY_PATH = Path("backtests/logs/promotion_history.jsonl")

# Sources that describe synthetic/demo data rather than real market bars.
_SYNTHETIC_SOURCES = {"synthetic", "demo"}


@dataclass(frozen=True)
class HistoryRecord:
    """One past (parameters -> OOS Sharpe) observation.

    Carries TWO distinct Sharpe numbers per run, and callers must not
    confuse them:

      - `oos_sharpe` / `params` — the CANDIDATE the search tried that run
        (the WFO fold winner out of that iteration's grid). This is what the
        suggester trains on: "what did this parameter set score".
      - `baseline_oos_sharpe` / `baseline_params` — the ALREADY-LIVE
        parameters (whatever configs/strategy.yaml held at the time),
        re-measured on that same run's window. Every run computes this
        whether or not anything gets promoted, so its trend across runs is
        the one honest answer to "is the strategy I actually have deployed
        still working on recent data" — which the candidate trend is NOT: on
        a REJECTED run the candidate is something that was tried and thrown
        away, not what is live.
    """

    params: dict[str, float]
    oos_sharpe: float
    timestamp: float  # Unix seconds — what the sidecar's optimizer expects
    strategy: str
    data_source: str
    decision: str
    # None only for a malformed record missing this normally-always-present
    # field; degradation checks must skip such records rather than treat
    # None as zero.
    baseline_params: dict[str, float] | None = None
    baseline_oos_sharpe: float | None = None
    # Which regime the tested window mostly sat in, as recorded by
    # self_improve_loop.py. "unknown" for runs from before the tag existed and
    # for --demo runs, so any consumer must treat it as optional.
    market_regime: str = "unknown"

    @property
    def is_synthetic(self) -> bool:
        return self.data_source.lower() in _SYNTHETIC_SOURCES


def _parse_timestamp(raw: str) -> float | None:
    """ISO-8601 (as written by promotion.py) -> Unix seconds."""
    try:
        return datetime.fromisoformat(raw).timestamp()
    except (TypeError, ValueError):
        return None


def load_history(
    strategy: str,
    *,
    path: Path = PROMOTION_HISTORY_PATH,
    include_synthetic: bool = False,
    param_keys: set[str] | None = None,
) -> list[HistoryRecord]:
    """Read past parameter observations for one strategy.

    Args:
        strategy: strategy name, matched exactly against the record's field.
        path: history file; a missing file is an empty history, not an error
            (a fresh checkout has never run the loop).
        include_synthetic: keep --demo runs. Only ever true for demo/testing.
        param_keys: if given, drop records whose parameter set is not exactly
            these keys. Guards against a grid that has since changed shape —
            a record tested before a key existed cannot be placed in today's
            search space, and silently dropping the key would misattribute
            its Sharpe to the remaining dimensions.

    Returns:
        Records oldest-first.
    """
    if not path.exists():
        log.info("no promotion history at %s — suggester starts cold", path)
        return []

    records: list[HistoryRecord] = []
    skipped_synthetic = 0
    skipped_shape = 0

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            log.warning("%s:%d is not valid JSON — skipping", path, line_no)
            continue

        if entry.get("strategy") != strategy:
            continue

        params = entry.get("candidate_params")
        sharpe = entry.get("candidate_oos_sharpe")
        if not isinstance(params, dict) or not params or not isinstance(sharpe, (int, float)):
            continue

        # A non-finite Sharpe (no trades, zero return variance) carries no
        # information about the parameter space and would poison a GP fit.
        if not math.isfinite(float(sharpe)):
            continue

        ts = _parse_timestamp(entry.get("timestamp", ""))
        if ts is None:
            continue

        extra = entry.get("extra")

        # baseline_oos_sharpe/baseline_params are core promotion.py fields,
        # present on every well-formed record — but validated leniently
        # (None on absence/bad type) rather than by dropping the whole
        # record, so a hypothetical gap here never silently starves the
        # suggester of candidate-side training data it doesn't need this for.
        baseline_sharpe_raw = entry.get("baseline_oos_sharpe")
        baseline_oos_sharpe = (
            float(baseline_sharpe_raw)
            if isinstance(baseline_sharpe_raw, (int, float)) and math.isfinite(float(baseline_sharpe_raw))
            else None
        )
        baseline_params_raw = entry.get("baseline_params")
        baseline_params = (
            {k: float(v) for k, v in baseline_params_raw.items() if isinstance(v, (int, float))}
            if isinstance(baseline_params_raw, dict)
            else None
        )

        record = HistoryRecord(
            params={k: float(v) for k, v in params.items() if isinstance(v, (int, float))},
            oos_sharpe=float(sharpe),
            timestamp=ts,
            strategy=strategy,
            data_source=str(entry.get("data_source", "")),
            decision=str(entry.get("decision", "")),
            baseline_params=baseline_params,
            baseline_oos_sharpe=baseline_oos_sharpe,
            market_regime=str((extra or {}).get("market_regime", "unknown")),
        )

        if record.is_synthetic and not include_synthetic:
            skipped_synthetic += 1
            continue

        if param_keys is not None and set(record.params) != param_keys:
            skipped_shape += 1
            continue

        records.append(record)

    records.sort(key=lambda r: r.timestamp)

    if skipped_synthetic:
        log.info("dropped %d synthetic-data record(s) from %s history", skipped_synthetic, strategy)
    if skipped_shape:
        log.info("dropped %d record(s) whose parameter shape no longer matches the grid", skipped_shape)
    log.info("loaded %d usable history record(s) for %s", len(records), strategy)

    return records


def load_latest_raw_record(
    strategy: str,
    *,
    path: Path = PROMOTION_HISTORY_PATH,
    include_synthetic: bool = False,
) -> dict | None:
    """The most recent RAW promotion_history.jsonl entry for `strategy`,
    unparsed (unlike load_history/HistoryRecord, which keep only the fields
    the GP suggester needs and drop everything else — gates, reason,
    wfo_summary, extra).

    For explain.explain_decision(), which needs the full record (gate-by-gate
    pass/fail, the human-written `reason`, market_regime) to build a useful
    prompt, not just (params, sharpe). Returns None if there is no matching,
    well-formed record — a missing/empty history is "nothing to explain",
    not an error.

    Same synthetic-data guard as load_history: a --demo run's rejection
    reason is not informative about real trading and must not be surfaced by
    default as if it were.
    """
    if not path.exists():
        return None

    latest: dict | None = None
    latest_ts: float | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("strategy") != strategy:
            continue

        data_source = str(entry.get("data_source", ""))
        if data_source.lower() in _SYNTHETIC_SOURCES and not include_synthetic:
            continue

        ts = _parse_timestamp(entry.get("timestamp", ""))
        if ts is None:
            continue

        if latest_ts is None or ts >= latest_ts:
            latest, latest_ts = entry, ts

    return latest


def to_sidecar_payload(records: list[HistoryRecord]) -> list[dict]:
    """Shape records for POST /optimize/suggest-params."""
    return [
        {"params": r.params, "oos_sharpe": r.oos_sharpe, "timestamp": r.timestamp}
        for r in records
    ]


def summarize(records: list[HistoryRecord]) -> dict:
    """Small human-readable digest for logs and reports."""
    if not records:
        return {"count": 0}
    sharpes = [r.oos_sharpe for r in records]
    return {
        "count": len(records),
        "best_sharpe": max(sharpes),
        "worst_sharpe": min(sharpes),
        "oldest": datetime.fromtimestamp(records[0].timestamp).date().isoformat(),
        "newest": datetime.fromtimestamp(records[-1].timestamp).date().isoformat(),
        "promoted": sum(1 for r in records if r.decision == "PROMOTED"),
    }
