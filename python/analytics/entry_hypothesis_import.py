"""Fill 15×7 cells from existing official reports. Does not run WFO."""

from __future__ import annotations

import json
from pathlib import Path

from python.analytics.volume_route_policy import CANONICAL_GATES, collect_route_gates

REPORTS = Path("backtests/reports")


def _pf_ok(metrics: dict | None, floor: float = 1.0) -> bool:
    if not metrics:
        return False
    try:
        return float(metrics.get("profit_factor") or 0.0) >= floor
    except (TypeError, ValueError):
        return False


def _from_intraday_result(route: str, minutes: int, raw: dict, source: str) -> dict:
    hard = dict(raw.get("gates") or {})
    soft = dict(raw.get("soft_gates") or {})
    stress = raw.get("stress_metrics") or {}
    if "profit_factor" in stress:
        hard["stress_slippage_1.5x_pf_ge_1"] = _pf_ok(stress)
    cell = dict(raw)
    cell["route"] = route
    cell["chart_minutes"] = minutes
    cell["imported_from"] = source
    cell["route_gates"] = collect_route_gates(hard, soft)
    return cell


def _from_a0(route: str, minutes: int, path: Path, signal: str) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["signal"] = signal
    raw["data_label"] = raw.get("data_label") or (
        f"imported {path} ({raw.get('window')}, {raw.get('cost_model')})"
    )
    return _from_intraday_result(route, minutes, raw, str(path))


def _daily(route: str, source: str, *, decision: str, gates: dict, metrics: dict, extra: dict) -> dict:
    vector = {name: bool(gates.get(name, False)) for name in CANONICAL_GATES}
    cell = {
        "signal": route,
        "route": route,
        "decision": decision,
        "chart_minutes": 0,
        "imported_from": source,
        "full_window_metrics": metrics,
        "gates": gates,
        "soft_gates": {},
        "route_gates": vector,
        "data_label": f"imported official daily evidence — {source}",
    }
    cell.update(extra)
    return cell


def collect_existing_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}

    vb = json.loads((REPORTS / "volume_book_signals_backtest_report.json").read_text(encoding="utf-8"))
    for row in vb.get("results") or []:
        sig = row.get("signal")
        if sig in ("vsa_no_demand", "obv_divergence"):
            cells[f"{sig}_5m"] = _from_intraday_result(sig, 5, row, "volume_book_signals_backtest_report.json")

    auction = json.loads((REPORTS / "auction_reclaim_backtest_report.json").read_text(encoding="utf-8"))
    for row in auction.get("results") or []:
        cells["auction_reclaim_5m"] = _from_intraday_result(
            "auction_reclaim", 5, row, "auction_reclaim_backtest_report.json",
        )

    slip = json.loads((REPORTS / "slippage_calibration_report.json").read_text(encoding="utf-8"))
    for name, blob in (slip.get("signals") or {}).items():
        payload = (blob or {}).get("new") or blob
        if not isinstance(payload, dict) or not payload.get("gates"):
            continue
        cells[f"{name}_1m"] = _from_intraday_result(
            name, 1, payload, f"slippage_calibration_report.json:{name}.new",
        )

    cells["l2_absorption_1m"] = _from_a0(
        "l2_absorption", 1,
        REPORTS / "_l2_absorption_validation" / "A0_grid_full20.json",
        "l2_absorption",
    )
    cells["absorption_breakout_1m"] = _from_a0(
        "absorption_breakout", 1,
        REPORTS / "_absorption_breakout_validation" / "A0_grid_full20.json",
        "absorption_breakout",
    )

    pairs = json.loads((REPORTS / "pairs_scan_report.json").read_text(encoding="utf-8"))
    dev = pairs.get("dev") or {}
    fw = dev.get("full_window") or {}
    stress = dev.get("stress_2x_spread") or {}
    mc = ((dev.get("monte_carlo") or {}).get("sharpe") or {})
    p_gates = dev.get("gates") or {}
    cells["pairs_trading_daily"] = _daily(
        "pairs_trading",
        "pairs_scan_report.json:dev",
        decision=str(dev.get("verdict") or "NO-GO"),
        gates={
            "wfo_go": bool(p_gates.get("wfo_go")),
            "oos_drawdown_within_limit": bool(p_gates.get("oos_drawdown_within_limit")),
            "has_oos_trades": bool(p_gates.get("has_oos_trades")),
            "min_trades_per_oos_fold": int(fw.get("n_trades") or 0) >= 40,
            "cost_adjusted_profit_factor": _pf_ok(fw),
            "monte_carlo_p5_sharpe": bool(p_gates.get("monte_carlo_p5_sharpe")),
            "stress_slippage_1.5x_pf_ge_1": _pf_ok(stress),
        },
        metrics=fw,
        extra={
            "window": str((dev.get("window") or "")),
            "wfo_pass_ratio": (dev.get("candidate_wfo") or {}).get("pass_ratio"),
            "mc_p5_sharpe": mc.get("p5"),
            "stress_metrics": stress,
        },
    )

    # Mega-cap xsection from the settled review numbers (self-improve / strategy_review).
    cells["xsection_mean_reversion_daily"] = _daily(
        "xsection_mean_reversion",
        "strategy_review_summary.md §2.2 + self_improvement_log",
        decision="NO-GO",
        gates={
            "wfo_go": False,
            "oos_drawdown_within_limit": True,
            "has_oos_trades": True,
            "min_trades_per_oos_fold": True,
            "cost_adjusted_profit_factor": False,
            "monte_carlo_p5_sharpe": False,
            "stress_slippage_1.5x_pf_ge_1": False,
        },
        metrics={"n_trades": None, "profit_factor": None, "note": "OOS Sharpe -0.492, MC p5 -0.957, WFO 31%"},
        extra={"wfo_pass_ratio": 0.31, "oos_sharpe_mean": -0.492, "mc_p5_sharpe": -0.957},
    )

    brk = json.loads((REPORTS / "track2_daily_breakout_results.json").read_text(encoding="utf-8"))
    wfo = brk.get("wfo") or {}
    mc = (brk.get("monte_carlo_pooled_oos") or {}).get("sharpe") or {}
    cells["daily_range_breakout_daily"] = _daily(
        "daily_range_breakout",
        "track2_daily_breakout_results.json",
        decision=str(wfo.get("decision") or "NO-GO"),
        gates={
            "wfo_go": False,
            "oos_drawdown_within_limit": int(brk.get("wfo_drawdown_gate_violations") or 0) == 0,
            "has_oos_trades": int(brk.get("n_pooled_oos_trades") or 0) > 0,
            "min_trades_per_oos_fold": int(brk.get("n_pooled_oos_trades") or 0) >= 40,
            "cost_adjusted_profit_factor": False,
            "monte_carlo_p5_sharpe": float(mc.get("p5") or -1) >= 0,
            "stress_slippage_1.5x_pf_ge_1": False,
        },
        metrics={
            "n_trades": brk.get("n_pooled_oos_trades"),
            "total_net_pnl": sum(
                ((f.get("oos_metrics") or {}).get("net_pnl") or 0)
                for f in (wfo.get("folds") or [])
            ),
        },
        extra={
            "wfo_folds": wfo.get("total_folds"),
            "wfo_pass_ratio": wfo.get("pass_ratio"),
            "oos_sharpe_mean": wfo.get("oos_sharpe_mean"),
            "mc_p5_sharpe": mc.get("p5"),
        },
    )
    return cells
