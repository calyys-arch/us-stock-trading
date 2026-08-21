"""
Promotion logic for the self-improve WFO loop: decide whether a candidate
parameter set earned the right to replace the current values in
configs/strategy.yaml, write it back if so, and record EVERY decision
(promoted or not) to an append-only JSONL history.

Promotion policy (user-confirmed, 2026-07-28 — "auto_write" option):
  1. ALL validation gates must pass (WFO GO + fold drawdown ceiling +
     has-trades + Monte Carlo p5; Reality Check where the caller supplies
     it). A candidate that fails ANY gate is rejected regardless of Sharpe.
  2. The candidate's mean OOS Sharpe must beat the CURRENT config's mean
     OOS Sharpe — measured on the SAME folds and data — by STRICTLY MORE
     than goal.yaml live_promotion.min_oos_sharpe_improvement (an
     improvement numerically EQUAL to the margin does NOT qualify — see
     that config key's own comment: "0.0 = any strict improvement
     qualifies", i.e. a tied/zero-improvement candidate must never
     promote). "New params won the grid search" is not enough; they must
     beat the incumbent out-of-sample or the incumbent stays.
  3. Only strategy PARAMETERS are ever written. auto_execute / enabled are
     NEVER touched by this module — going live remains a human decision
     (configs/strategy.yaml's observe-mode contract).

Config write-back uses ruamel.yaml round-trip mode so configs/strategy.yaml
keeps its comments and formatting — the file is the project's
parameter-discipline documentation, not just data, and a rewrite that
stripped its comments would destroy that.

History: backtests/logs/promotion_history.jsonl (append-only, one JSON object
per decision) is the machine-readable audit trail;
backtests/reports/self_improvement_log.md (written by
scripts/self_improve_loop.py) is the human-readable one. All backtest
reports/logs live under backtests/ (see README.md's directory layout) so
future runs have one place to look, separate from data/'s raw price/tick
caches.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STRATEGY_CONFIG_PATH = Path("configs/strategy.yaml")
HISTORY_PATH = Path("backtests/logs/promotion_history.jsonl")

# Keys that must never be auto-written, even if they somehow appear in a
# candidate dict — going live / enabling a strategy is a human decision.
_FORBIDDEN_WRITE_KEYS = {"enabled", "auto_execute"}


@dataclass
class PromotionRecord:
    timestamp: str
    strategy: str
    decision: str                 # "PROMOTED" | "REJECTED"
    reason: str
    candidate_params: dict
    baseline_params: dict
    candidate_oos_sharpe: float
    baseline_oos_sharpe: float
    gates: dict                   # {gate_name: bool}
    wfo_summary: dict             # pass_ratio / total_folds / decision
    universe_fingerprint: str = ""
    data_source: str = ""
    config_written: bool = False
    iteration: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_strategy_config(
    strategy_name: str,
    new_params: dict,
    path: str | Path = STRATEGY_CONFIG_PATH,
) -> None:
    """Round-trip-update ONLY `new_params`' keys inside the strategy's block,
    preserving all comments/ordering. Refuses to introduce new keys or touch
    forbidden ones — the config's schema is owned by humans."""
    from ruamel.yaml import YAML

    forbidden = set(new_params) & _FORBIDDEN_WRITE_KEYS
    if forbidden:
        raise ValueError(f"refusing to auto-write forbidden keys: {sorted(forbidden)}")

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    path = Path(path)
    doc = yaml_rt.load(path.read_text(encoding="utf-8"))
    if strategy_name not in doc:
        raise KeyError(f"{path} has no '{strategy_name}' block")
    block = doc[strategy_name]
    unknown = set(new_params) - set(block.keys())
    if unknown:
        raise ValueError(
            f"refusing to introduce new keys into {path}:{strategy_name}: {sorted(unknown)}"
        )
    for key, value in new_params.items():
        block[key] = value
    with open(path, "w", encoding="utf-8") as f:
        yaml_rt.dump(doc, f)
    log.info("promotion: wrote %s to %s:%s", new_params, path, strategy_name)


def append_history(record: PromotionRecord, path: str | Path = HISTORY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), default=str) + "\n")


def evaluate_and_promote(
    strategy_name: str,
    candidate_params: dict,
    baseline_params: dict,
    candidate_oos_sharpe: float,
    baseline_oos_sharpe: float,
    gates: dict,
    wfo_summary: dict,
    min_improvement: float = 0.0,
    write_config: bool = True,
    config_path: str | Path = STRATEGY_CONFIG_PATH,
    history_path: str | Path = HISTORY_PATH,
    universe_fingerprint: str = "",
    data_source: str = "",
    iteration: int = 0,
) -> PromotionRecord:
    """Apply the promotion policy and (when it passes and `write_config`)
    write the candidate back to configs/strategy.yaml. Always appends the
    decision to the JSONL history."""
    failed_gates = sorted(name for name, ok in gates.items() if not ok)
    improvement = candidate_oos_sharpe - baseline_oos_sharpe

    if failed_gates:
        decision, reason = "REJECTED", f"gates failed: {', '.join(failed_gates)}"
    elif candidate_params == baseline_params:
        decision, reason = "REJECTED", "candidate equals current config (no change to promote)"
    elif improvement <= min_improvement:
        # Deliberately "<=", not "<": verified during the 2026-08-15
        # round-2 audit (backtests/reports/backtest_engine_audit_round2.md)
        # against configs/goal.yaml's own comment on
        # `live_promotion.min_oos_sharpe_improvement` — "0.0 = any STRICT
        # improvement qualifies" — which is the authoritative statement of
        # intent for this boundary, and explicitly requires improvement >
        # min_improvement (a tie must NOT promote), not >=. This class's
        # OWN docstring above ("beat it by at least...") is the imprecise
        # one; worded more carefully there instead of changing this
        # comparison — see that docstring update in the same commit.
        decision, reason = "REJECTED", (
            f"OOS Sharpe improvement {improvement:+.3f} does not strictly beat incumbent by "
            f"more than {min_improvement:+.3f}"
        )
    else:
        decision, reason = "PROMOTED", (
            f"all gates passed; OOS Sharpe {candidate_oos_sharpe:+.3f} beats incumbent "
            f"{baseline_oos_sharpe:+.3f} by {improvement:+.3f}"
        )

    config_written = False
    if decision == "PROMOTED" and write_config:
        write_strategy_config(strategy_name, candidate_params, path=config_path)
        config_written = True

    record = PromotionRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        strategy=strategy_name,
        decision=decision,
        reason=reason,
        candidate_params=dict(candidate_params),
        baseline_params=dict(baseline_params),
        candidate_oos_sharpe=float(candidate_oos_sharpe),
        baseline_oos_sharpe=float(baseline_oos_sharpe),
        gates=dict(gates),
        wfo_summary=dict(wfo_summary),
        universe_fingerprint=universe_fingerprint,
        data_source=data_source,
        config_written=config_written,
        iteration=iteration,
    )
    append_history(record, path=history_path)
    log.info("promotion[%s]: %s — %s", strategy_name, decision, reason)
    return record
