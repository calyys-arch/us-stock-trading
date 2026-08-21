"""
Phase 2 gated-strategy validation (regime_gate_report.md) — production
`xsection_mean_reversion` config, gated by the SAME
`python/analytics/trend_efficiency_gate.shifted_entry_gate` instance (same
`window=20`, `reference_window=252`, same SPY input) already used to gate
`pairs_trading` in `scripts/_regime_gate_phase2_pairs.py` — one classifier,
applied unchanged to a second, unrelated strategy family, which is itself
evidence against per-strategy curve-fitting of the classifier's parameters.

Gating mechanism: on a gate-OFF day, `universe_by_day[day] = []` is passed to
`run_vector_backtest` instead of the full fixed universe. Because
`CrossSectionalMeanReversionStrategy.evaluate` already returns zero weights
whenever `len(eligible) < min_universe_size` (see
`python/core/strategies/xsection_mean_reversion.py`), an empty universe list
deterministically produces a flat (zero-position) day with NO changes to the
strategy or engine code — the same "caller decides eligibility, strategy
code is unmodified" pattern `_regime_generalization_xsection.py` already
uses for its NBIS/SNDK exclusion, just applied per-day instead of for the
whole run.

Free-parameter accounting: `xsection_mean_reversion`'s own 3 parameters
(lookback_days, gross_leverage_target, min_universe_size — see that
strategy's own docstring) are UNCHANGED (production config,
`configs/strategy.yaml`, never re-tuned here) + the classifier's 2
(window, reference_window, already counted once against the pairs pipeline
and reused unchanged here) = 5, at but not over the ceiling for this
pipeline.

Usage:
    python scripts/_regime_gate_phase2_xsection.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
log = logging.getLogger("regime_gate_phase2_xsection")

from python.analytics.trend_efficiency_gate import DEFAULT_REFERENCE_WINDOW, DEFAULT_WINDOW, shifted_entry_gate
from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import load_wfo_config, max_oos_drawdown_threshold
from python.backtest.vector_engine import run_vector_backtest
from python.backtest.walk_forward import WalkForwardOptimizer
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy
from python.data.price_cache import get_cached_price_panel

CACHE_DIR = Path("backtests/reports/_regime_gate_phase2_cache")
REPORT_JSON = Path("backtests/reports/regime_gate_phase2_xsection.json")
FULL_START, FULL_END = "2018-01-01", "2026-08-01"
FETCH_START, FETCH_END = "2016-06-01", "2026-08-01"
_CAPITAL = 1_000_000.0
SKIP_FIRST_DAYS = 30
_SYMBOLS_NOT_YET_LISTED = {"NBIS", "SNDK"}  # same exclusion as _regime_generalization_xsection.py


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("checkpoint written: %s", path)


def _metrics(result, capital=_CAPITAL) -> dict:
    net = result.daily_returns
    gross = result.daily_gross_returns
    cost = result.daily_costs
    sharpe = 0.0
    if len(net) >= 2 and net.std(ddof=1) > 0:
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
    equity = (1.0 + net).cumprod() if len(net) else pd.Series(dtype=float)
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    n_active = int((net != 0).sum())
    gross_d, cost_d, net_d = gross * capital, cost * capital, net * capital

    def _pf(s):
        wins = float(s[s > 0].sum())
        losses = float(-s[s < 0].sum())
        return wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)

    return {
        "sharpe_ratio": sharpe, "max_drawdown": max_dd,
        "n_active_days": n_active, "n_trades": n_active, "n_days": int(len(net)),
        "total_net_pnl": float(net_d.sum()), "total_gross_pnl": float(gross_d.sum()),
        "total_cost": float(cost_d.sum()),
        "gross_to_cost_ratio": float(gross_d.sum()) / float(cost_d.sum()) if cost_d.sum() != 0 else float("inf"),
        "profit_factor_net_daily": _pf(net_d),
        "daily_returns": net.tolist(),
    }


def _strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "daily_returns"}


def _make_backtest_fn(panel, symbols, base_cfg, tradeable_dates, gate: pd.Series | None):
    def backtest_fn(start, end, params: dict) -> dict:
        merged = {**base_cfg, **params}
        strategy = CrossSectionalMeanReversionStrategy(
            lookback_days=merged["lookback_days"],
            gross_leverage_target=merged["gross_leverage_target"],
            min_universe_size=merged["min_universe_size"],
        )
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window_dates = [d for d in tradeable_dates if start_ts <= pd.Timestamp(d) < end_ts]
        if not window_dates:
            return _metrics_empty()
        universe_by_day = {}
        for d in window_dates:
            if gate is not None and not bool(gate.get(pd.Timestamp(d), False)):
                universe_by_day[d] = []
            else:
                universe_by_day[d] = list(symbols)
        result = run_vector_backtest(strategy, panel, universe_by_day, capital=_CAPITAL, half_spread_bps=None)
        return _metrics(result)
    return backtest_fn


def _metrics_empty() -> dict:
    return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "n_active_days": 0, "n_trades": 0,
            "n_days": 0, "total_net_pnl": 0.0, "total_gross_pnl": 0.0, "total_cost": 0.0,
            "gross_to_cost_ratio": float("inf"), "profit_factor_net_daily": 0.0, "daily_returns": []}


def main() -> None:
    with open("configs/universe.yaml", encoding="utf-8") as f:
        symbols = list(yaml.safe_load(f)["fixed_universe"]["symbols"])
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)["xsection_mean_reversion"]

    fetchable = [s for s in symbols if s.upper() not in _SYMBOLS_NOT_YET_LISTED]
    panel, _q, meta = get_cached_price_panel(fetchable + ["SPY"], FETCH_START, FETCH_END)
    log.info("panel loaded: sources=%s", meta["sources"])

    spy_close = panel.xs("SPY", level=1)["close"].sort_index()
    entry_gate = shifted_entry_gate(spy_close, window=DEFAULT_WINDOW, reference_window=DEFAULT_REFERENCE_WINDOW)
    log.info("entry_gate: %d/%d days ON", int(entry_gate.sum()), len(entry_gate))

    xsection_panel = panel[panel.index.get_level_values(1) != "SPY"]
    all_dates = sorted(xsection_panel.index.get_level_values(0).unique())
    tradeable_dates = all_dates[SKIP_FIRST_DAYS:]

    fn_ungated = _make_backtest_fn(xsection_panel, symbols, base_cfg, tradeable_dates, gate=None)
    fn_gated = _make_backtest_fn(xsection_panel, symbols, base_cfg, tradeable_dates, gate=entry_gate)

    wfo_cfg = load_wfo_config("xsection_mean_reversion")
    log.info("WFO config: %s", wfo_cfg)

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd_gate = max_oos_drawdown_threshold()
    min_cost_ratio = float(goal["cost_gate"]["min_gross_to_cost_ratio"])

    results = {}
    for label, fn in (("ungated", fn_ungated), ("gated", fn_gated)):
        ckpt = CACHE_DIR / f"xsection_{label}.json"
        if ckpt.exists():
            log.info("resuming [%s] from %s", label, ckpt)
            results[label] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue
        log.info("=== %s WFO over full history [%s, %s) ===", label, FULL_START, FULL_END)
        wfo = WalkForwardOptimizer(fn, wfo_cfg, [{}]).run(
            pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime())
        wfo.print_summary()

        oos_returns: list[float] = []
        for f in wfo.folds:
            oos_returns.extend(f.oos_metrics.get("daily_returns", []))
        mc_aggregate = MonteCarloValidator(n_sims=500).run(oos_returns)

        full_metrics = fn(pd.Timestamp(FULL_START).to_pydatetime(), pd.Timestamp(FULL_END).to_pydatetime(), {})
        mc_full = MonteCarloValidator(n_sims=500).run(full_metrics.get("daily_returns", []))
        gates = {
            "has_active_days": full_metrics["n_active_days"] > 0,
            "sharpe_above_min_oos_sharpe": full_metrics["sharpe_ratio"] >= min_oos_sharpe,
            "drawdown_within_limit": abs(full_metrics["max_drawdown"]) <= max_dd_gate,
            "monte_carlo_p5_sharpe": mc_full.sharpe.p5 >= min_p5,
            "cost_gate_gross_to_cost_ratio": full_metrics["gross_to_cost_ratio"] >= min_cost_ratio,
        }

        results[label] = {
            "config": "xsection_mean_reversion_production",
            "wfo": wfo.to_dict(),
            "aggregate_oos_monte_carlo": mc_aggregate.to_dict(),
            "aggregate_oos_n_days": len(oos_returns),
            "full_window_2018_2026": {
                "window": [FULL_START, FULL_END],
                "metrics": _strip(full_metrics),
                "monte_carlo": mc_full.to_dict(),
                "gates": gates,
                "verdict": "GO" if all(gates.values()) else "NO-GO",
            },
        }
        _write_json(ckpt, results[label])

    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Phase 2 gated-vs-ungated full-history (2018-2026, all regimes "
            "blended) WFO comparison for xsection_mean_reversion production "
            "config, gated by the SAME classifier instance (window=20, "
            "reference_window=252) already used for pairs_trading."
        ),
        "classifier": {"window": DEFAULT_WINDOW, "reference_window": DEFAULT_REFERENCE_WINDOW,
                        "n_days_gate_on": int(entry_gate.sum()), "n_days_total": len(entry_gate)},
        "results": results,
    }
    _write_json(REPORT_JSON, out)
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
