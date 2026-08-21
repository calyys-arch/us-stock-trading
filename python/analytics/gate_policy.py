"""
Market-conditional LIVE order-eligibility policy.

This module only decides whether a paper/auto ENTRY may be submitted.
It is not research GO (`scripts/run_intraday_backtest.py`).

Combinations are not a fixed "N of 7" and not a random search over
subsets. `configs/goal.yaml` `gate_policy.situations` is an ordered
catalog: each situation names an explicit gate list. `select_situation`
picks the first row whose `when` matches the current market and whose
`family` matches the strategy. No match is a refusal — there is no
seven-gate / research-GO fallback. Adding a new market → combination
is a YAML edit.

Scorecards stay frozen in `configs/gate_scorecards.yaml` (regime-matched
historical windows). They are not re-tuned from the live tape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

GOAL_PATH = Path("configs/goal.yaml")
SCORECARD_PATH = Path("configs/gate_scorecards.yaml")

ALL_GATES: tuple[str, ...] = (
    "min_oos_sharpe",
    "max_oos_drawdown",
    "wfo_pass_ratio",
    "monte_carlo_p5",
    "reality_check",
    "cost_adjusted_profit_factor",
    "stress_slippage_2x_net_positive",
    "has_oos_trades",
)

REGIME_TREND = "trend_persistent"
REGIME_MR = "mean_reversion_friendly"
REGIME_UNDECIDED = "undecided"

VOL_HIGH = "high"
VOL_NORMAL = "normal"
VOL_LOW = "low"
VOL_UNKNOWN = "unknown"

FAMILY_CONTINUATION = "continuation"
FAMILY_MEAN_REVERSION = "mean_reversion"
FAMILY_ANY = "any"

STRATEGY_FAMILY: dict[str, str] = {
    "absorption_breakout": FAMILY_CONTINUATION,
    "pairs_trading": FAMILY_MEAN_REVERSION,
}

# Scorecards are stored under the tape window, not the situation name.
_TAPE_KEYS = frozenset({REGIME_TREND, REGIME_MR})


@dataclass(frozen=True)
class Situation:
    name: str
    gates: tuple[str, ...]
    family: str
    tape: str
    vol: str


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    regime: str
    vol: str
    gate_set: str
    required: tuple[str, ...]
    results: dict[str, bool]
    reason: str


def load_gate_policy(path: str | Path = GOAL_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        goal = yaml.safe_load(f) or {}
    return dict(goal.get("gate_policy") or {})


def load_scorecards(path: str | Path = SCORECARD_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def regime_from_pairs_gate(pairs_gate_open: bool | None) -> str:
    if pairs_gate_open is None:
        return REGIME_UNDECIDED
    return REGIME_MR if pairs_gate_open else REGIME_TREND


def classify_volatility(
    close: pd.Series | None,
    window: int = 20,
    reference: int = 252,
    high_mult: float = 1.5,
    low_mult: float = 0.7,
) -> str:
    """High / normal / low vs this series' own trailing vol median.
    Insufficient history → unknown (matches situations that omit `vol`
    or set it to any)."""
    if close is None or close.dropna().empty:
        return VOL_UNKNOWN
    s = close.dropna().astype(float)
    if len(s) < window + 2:
        return VOL_UNKNOWN
    vol = s.pct_change().rolling(window, min_periods=window).std()
    if len(s) < reference + window:
        latest = vol.dropna()
        if latest.empty:
            return VOL_UNKNOWN
        # No long reference yet — treat as normal, not a fake "high".
        return VOL_NORMAL
    med = vol.rolling(reference, min_periods=reference).median()
    v, m = vol.iloc[-1], med.iloc[-1]
    if pd.isna(v) or pd.isna(m) or m <= 0:
        return VOL_UNKNOWN
    if v > m * high_mult:
        return VOL_HIGH
    if v < m * low_mult:
        return VOL_LOW
    return VOL_NORMAL


def _as_list(value) -> list:
    if value is None or value == "any":
        return ["any"]
    if isinstance(value, (list, tuple)):
        return list(value) or ["any"]
    return [value]


def _when_matches(when: dict, tape: str, vol: str) -> bool:
    tapes = _as_list(when.get("tape", "any"))
    vols = _as_list(when.get("vol", "any"))
    tape_ok = "any" in tapes or tape in tapes
    vol_ok = "any" in vols or vol in vols or (vol == VOL_UNKNOWN and "any" in vols)
    return tape_ok and vol_ok


def _family_matches(want: str, family: str) -> bool:
    return want in (FAMILY_ANY, family)


def list_situations(policy: dict | None = None) -> list[dict]:
    policy = policy if policy is not None else load_gate_policy()
    return list(policy.get("situations") or [])


def select_situation(
    tape: str,
    vol: str,
    family: str,
    policy: dict | None = None,
) -> Situation | None:
    """First catalog row whose `when` matches (tape, vol) and whose
    `family` matches the strategy. Catalog order is the priority."""
    if tape not in _TAPE_KEYS:
        return None
    for row in list_situations(policy):
        if not _when_matches(row.get("when") or {}, tape, vol):
            continue
        if not _family_matches(str(row.get("family") or FAMILY_ANY), family):
            continue
        gates = tuple(row.get("gates") or ())
        if not gates:
            continue
        unknown = [g for g in gates if g not in ALL_GATES]
        if unknown:
            raise ValueError(f"unknown gates in situation {row.get('name')!r}: {unknown}")
        return Situation(
            name=str(row.get("name") or "unnamed"),
            gates=gates,
            family=str(row.get("family") or FAMILY_ANY),
            tape=tape,
            vol=vol,
        )
    return None


def active_gate_set(
    tape: str,
    family: str,
    vol: str = VOL_NORMAL,
    policy: dict | None = None,
) -> Situation | None:
    return select_situation(tape, vol, family, policy)


def _passes(name: str, scorecard: dict, policy: dict) -> bool:
    floors = policy.get("live_floors") or {}
    live_pf = float(floors.get("min_cost_adjusted_profit_factor", 1.0))
    max_dd = float(floors.get("max_oos_drawdown", 0.25))
    min_wfo = float(floors.get("min_pass_folds_ratio", 0.60))
    min_sharpe = float(floors.get("min_oos_sharpe", 0.5))
    min_mc = float(floors.get("min_p5_sharpe", 0.0))
    max_rc = float(floors.get("max_reality_check_p_value", 0.05))

    if name == "max_oos_drawdown":
        dd = scorecard.get("max_oos_drawdown")
        return dd is not None and abs(float(dd)) <= max_dd
    if name == "cost_adjusted_profit_factor":
        pf = scorecard.get("cost_adjusted_profit_factor")
        return pf is not None and float(pf) >= live_pf
    if name == "stress_slippage_2x_net_positive":
        return bool(scorecard.get("stress_slippage_2x_net_positive"))
    if name == "has_oos_trades":
        return bool(scorecard.get("has_oos_trades"))
    if name == "wfo_pass_ratio":
        ratio = scorecard.get("wfo_pass_ratio")
        return ratio is not None and float(ratio) >= min_wfo
    if name == "min_oos_sharpe":
        sharpe = scorecard.get("mean_oos_sharpe")
        return sharpe is not None and float(sharpe) >= min_sharpe
    if name == "monte_carlo_p5":
        p5 = scorecard.get("monte_carlo_p5_sharpe")
        return p5 is not None and float(p5) >= min_mc
    if name == "reality_check":
        p = scorecard.get("reality_check_p_value")
        return p is not None and float(p) <= max_rc
    return False


def evaluate(
    strategy: str,
    tape: str,
    vol: str = VOL_NORMAL,
    scorecards: dict | None = None,
    policy: dict | None = None,
) -> PolicyDecision:
    policy = policy if policy is not None else load_gate_policy()
    scorecards = scorecards if scorecards is not None else load_scorecards()
    family = STRATEGY_FAMILY.get(strategy)
    if family is None:
        return PolicyDecision(False, tape, vol, "unknown_strategy", (), {}, "unknown_strategy")
    if tape not in _TAPE_KEYS:
        return PolicyDecision(False, tape, vol, "undecided", (), {}, "regime_undecided")

    situation = select_situation(tape, vol, family, policy)
    if situation is None:
        return PolicyDecision(
            False, tape, vol, "no_matching_situation", (), {},
            f"no_situation_for_family_{family}_tape_{tape}_vol_{vol}",
        )

    card = (scorecards.get(strategy) or {}).get(tape) or {}
    if not card:
        return PolicyDecision(
            False, tape, vol, situation.name, situation.gates, {},
            "missing_regime_matched_scorecard",
        )

    results = {g: _passes(g, card, policy) for g in situation.gates}
    failed = [g for g, ok in results.items() if not ok]
    allowed = not failed
    reason = "ok" if allowed else "failed: " + ", ".join(failed)
    return PolicyDecision(allowed, tape, vol, situation.name, situation.gates, results, reason)


def live_order_permitted(
    strategy: str,
    tape: str,
    vol: str = VOL_NORMAL,
    scorecards: dict | None = None,
    policy: dict | None = None,
) -> tuple[bool, str]:
    decision = evaluate(strategy, tape, vol, scorecards=scorecards, policy=policy)
    return decision.allowed, decision.reason


def research_requires_all_gates(policy: dict | None = None) -> bool:
    policy = policy if policy is not None else load_gate_policy()
    return bool(policy.get("research_require_all_gates", True))


def summarize_active(
    tape: str,
    vol: str = VOL_NORMAL,
    strategies: Iterable[str] | None = None,
) -> dict:
    strategies = list(strategies or STRATEGY_FAMILY)
    out = []
    for name in strategies:
        d = evaluate(name, tape, vol)
        out.append({
            "strategy": name,
            "allowed": d.allowed,
            "gate_set": d.gate_set,
            "n_gates": len(d.required),
            "required": list(d.required),
            "results": d.results,
            "reason": d.reason,
        })
    return {"regime": tape, "vol": vol, "strategies": out}
