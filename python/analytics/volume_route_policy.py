"""VSA / OBV research-route strategies — one test per gate.

Each (route, chart) cell shares one WFO. `combination_mode: single_gates`
then scores each of the seven gates independently. That is 7 VSA tests
and 7 OBV tests, compared across 1m / 5m / 15m.

This module only scores an already-recorded gate vector. It does not
run WFO and it does not flip `auto_execute`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATALOG_PATH = Path("configs/volume_route_strategies.yaml")

CANONICAL_GATES: tuple[str, ...] = (
    "wfo_go",
    "oos_drawdown_within_limit",
    "has_oos_trades",
    "min_trades_per_oos_fold",
    "cost_adjusted_profit_factor",
    "monte_carlo_p5_sharpe",
    "stress_slippage_1.5x_pf_ge_1",
)

GATE_SHORT: dict[str, str] = {
    "wfo_go": "wfo",
    "oos_drawdown_within_limit": "dd",
    "has_oos_trades": "trades",
    "min_trades_per_oos_fold": "sample",
    "cost_adjusted_profit_factor": "pf",
    "monte_carlo_p5_sharpe": "mc",
    "stress_slippage_1.5x_pf_ge_1": "stress",
}

# User-facing alias from the older "net PnL > 0" wording. Official
# research GO already uses PF >= 1 at 1.5x costs.
_STRESS_ALIASES = {
    "stress_slippage_1.5x_net_positive": "stress_slippage_1.5x_pf_ge_1",
    "stress_slippage_2x_net_positive": "stress_slippage_1.5x_pf_ge_1",
}


@dataclass(frozen=True)
class Route:
    name: str
    signal: str
    charts: tuple[int, ...]
    kind: str = "intraday"


@dataclass(frozen=True)
class Combination:
    name: str
    description: str
    gates: tuple[str, ...]
    mask: int


@dataclass(frozen=True)
class RouteStrategy:
    strategy_id: str
    route: str
    signal: str
    chart_minutes: int
    combination: str
    gates: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class StrategyScore:
    strategy_id: str
    decision: str
    required: tuple[str, ...]
    results: dict[str, bool]
    failed: tuple[str, ...]


def time_stop_for(chart_minutes: int) -> int:
    """Same wall-clock stop as scripts/compare_chart_minutes.py."""
    return max(10, 2 * int(chart_minutes))


def load_catalog(path: str | Path = CATALOG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def canonicalize_gate(name: str) -> str:
    return _STRESS_ALIASES.get(name, name)


def combination_name(gates: tuple[str, ...]) -> str:
    if not gates:
        return "none"
    return "+".join(GATE_SHORT[g] for g in gates)


def collect_route_gates(hard: dict, soft: dict) -> dict[str, bool]:
    """Merge official hard + soft dicts into the seven-gate vector."""
    merged = {canonicalize_gate(k): bool(v) for k, v in {**soft, **hard}.items()}
    for key in list(merged):
        if key.startswith("stress_slippage_") and key.endswith("_pf_ge_1"):
            merged["stress_slippage_1.5x_pf_ge_1"] = bool(merged[key])
    return {name: bool(merged.get(name, False)) for name in CANONICAL_GATES}


def list_routes(catalog: dict | None = None) -> tuple[Route, ...]:
    catalog = catalog if catalog is not None else load_catalog()
    rows = []
    for name, raw in (catalog.get("routes") or {}).items():
        charts = tuple(int(c) for c in (raw.get("charts") or ()))
        if not charts:
            raise ValueError(f"route {name!r} has no charts")
        signal = str(raw.get("signal") or "")
        if not signal:
            raise ValueError(f"route {name!r} has no signal")
        rows.append(Route(
            name=str(name),
            signal=signal,
            charts=charts,
            kind=str(raw.get("kind") or "intraday"),
        ))
    if not rows:
        raise ValueError("volume_route_strategies.yaml has no routes")
    return tuple(rows)


def _gates_from_catalog(catalog: dict) -> tuple[str, ...]:
    raw = catalog.get("gate_names") or CANONICAL_GATES
    gates = tuple(canonicalize_gate(str(g)) for g in raw)
    unknown = [g for g in gates if g not in CANONICAL_GATES]
    if unknown:
        raise ValueError(f"unknown gates: {unknown}")
    if len(gates) != len(CANONICAL_GATES) or set(gates) != set(CANONICAL_GATES):
        raise ValueError("gate_names must be the seven canonical gates")
    return CANONICAL_GATES


def single_gate_combinations(gates: tuple[str, ...] = CANONICAL_GATES) -> tuple[Combination, ...]:
    return tuple(
        Combination(
            name=GATE_SHORT[name],
            description=name,
            gates=(name,),
            mask=1 << i,
        )
        for i, name in enumerate(gates)
    )


def list_combinations(catalog: dict | None = None) -> tuple[Combination, ...]:
    catalog = catalog if catalog is not None else load_catalog()
    mode = str(catalog.get("combination_mode") or "single_gates")
    _gates_from_catalog(catalog)
    if mode == "single_gates":
        return single_gate_combinations(CANONICAL_GATES)
    raise ValueError(f"unsupported combination_mode {mode!r}")


def strategy_id(route: str, chart_minutes: int, combination: str) -> str:
    return f"{route}_{int(chart_minutes)}m_{combination}"


def list_strategies(catalog: dict | None = None) -> tuple[RouteStrategy, ...]:
    catalog = catalog if catalog is not None else load_catalog()
    out: list[RouteStrategy] = []
    for route in list_routes(catalog):
        for minutes in route.charts:
            for combo in list_combinations(catalog):
                out.append(RouteStrategy(
                    strategy_id=strategy_id(route.name, minutes, combo.name),
                    route=route.name,
                    signal=route.signal,
                    chart_minutes=minutes,
                    combination=combo.name,
                    gates=combo.gates,
                    description=combo.description,
                ))
    return tuple(out)


def score_combination(
    gate_vector: dict[str, bool],
    combination: Combination,
    *,
    route: str,
    chart_minutes: int,
) -> StrategyScore:
    results = {}
    failed = []
    for name in combination.gates:
        key = canonicalize_gate(name)
        ok = bool(gate_vector.get(key, False))
        results[key] = ok
        if not ok:
            failed.append(key)
    sid = strategy_id(route, chart_minutes, combination.name)
    return StrategyScore(
        strategy_id=sid,
        decision="GO" if not failed else "NO-GO",
        required=combination.gates,
        results=results,
        failed=tuple(failed),
    )


def score_all(
    gate_vector: dict[str, bool],
    *,
    route: str,
    chart_minutes: int,
    catalog: dict | None = None,
) -> list[StrategyScore]:
    return [
        score_combination(gate_vector, combo, route=route, chart_minutes=chart_minutes)
        for combo in list_combinations(catalog)
    ]


def passing_gate_names(gate_vector: dict[str, bool]) -> tuple[str, ...]:
    return tuple(name for name in CANONICAL_GATES if gate_vector.get(name))
