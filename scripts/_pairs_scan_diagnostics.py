"""
Supplementary diagnostics for backtests/reports/pairs_scan_report.md.

Report-only. Runs no parameter search and changes no gate: it answers the
"why" questions the gate table alone cannot — is the NO-GO a transaction-cost
problem or a thesis problem, how much of the eligible-pair count is expected
false positives, and what does the SAME scan produce at the incumbent
`entry_z: 2.0` (the apples-to-apples comparison against the old 8-trade
single-pair run).

Writes backtests/reports/_pairs_scan_cache/_diagnostics.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("pairs_scan_diagnostics")

from python.backtest.pairs_scan_engine import (
    PairsScanConfig,
    build_scan_schedule,
    candidate_pairs_from_buckets,
    load_pairs_universe,
    run_scan_backtest,
    select_active_pairs,
)
from python.data.price_cache import get_cached_price_panel

CACHE_DIR = Path("backtests/reports/_pairs_scan_cache")
OUT = CACHE_DIR / "_diagnostics.json"

WARMUP_START = "2016-06-01"
DEV = ("2018-01-01", "2024-01-01")
HOLDOUT = ("2024-01-01", "2026-08-01")


def _panels():
    universe = load_pairs_universe()
    symbols = sorted({s for codes in universe["buckets"].values() for s in codes})
    panel, _flags, _meta = get_cached_price_panel(symbols, WARMUP_START, HOLDOUT[1])
    close = panel["close"].unstack("code").sort_index()
    adv = panel["adv_20d_dollars"].unstack("code").sort_index()
    pairs = candidate_pairs_from_buckets(universe["buckets"])
    return close, adv, pairs


def _slice(close, adv, start, end, lookback):
    idx = close.index
    start_pos = int(idx.searchsorted(pd.Timestamp(start), side="left"))
    end_pos = int(idx.searchsorted(pd.Timestamp(end), side="left"))
    warm = max(0, start_pos - lookback)
    return close.iloc[warm:end_pos], adv.iloc[warm:end_pos]


def _window_stats(report, start, end) -> dict:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    trades = [t for t in report.trades if start_ts <= pd.Timestamp(t.exit_date) < end_ts]
    gross = sum(t.gross_pnl for t in trades)
    cost = sum(t.cost for t in trades)
    net = sum(t.net_pnl for t in trades)
    wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    gross_wins = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_losses = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
    hold = [(pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days for t in trades]
    return {
        "n_trades": len(trades),
        "n_distinct_pairs": len({(t.code_a, t.code_b) for t in trades}),
        "gross_pnl": gross,
        "total_cost": cost,
        "net_pnl": net,
        "cost_per_trade": (cost / len(trades)) if trades else 0.0,
        "gross_pnl_per_trade": (gross / len(trades)) if trades else 0.0,
        "profit_factor_after_costs": (wins / losses) if losses else None,
        "profit_factor_before_costs": (gross_wins / gross_losses) if gross_losses else None,
        "win_rate_after_costs": (sum(1 for t in trades if t.net_pnl > 0) / len(trades)) if trades else 0.0,
        "win_rate_before_costs": (sum(1 for t in trades if t.gross_pnl > 0) / len(trades)) if trades else 0.0,
        "median_holding_days": float(pd.Series(hold).median()) if hold else 0.0,
        "exit_reasons": pd.Series([t.exit_reason for t in trades]).value_counts().to_dict() if trades else {},
    }


def main() -> None:
    base_cfg = yaml.safe_load(Path("configs/strategy.yaml").read_text(encoding="utf-8"))["pairs_trading"]
    lookback = base_cfg["coint_lookback_days"]
    close, adv, pairs = _panels()
    schedule = build_scan_schedule(
        close, pairs, lookback_days=lookback,
        revalidate_every_days=base_cfg["revalidate_every_days"],
        checkpoint_path=CACHE_DIR / "scan_schedule.jsonl", progress_every=0)

    # ── Multiple-comparisons arithmetic on the scan itself ──────────────────
    eligible = {}
    for as_of, results in schedule.items():
        eligible[as_of] = len(select_active_pairs(results, PairsScanConfig(
            min_half_life_days=base_cfg["min_half_life_days"],
            max_half_life_days=base_cfg["max_half_life_days"])))
    counts = pd.Series(eligible)
    scan_stats = {
        "n_candidate_pairs_tested_per_scan": len(pairs),
        "n_scan_dates": len(schedule),
        "total_cointegration_tests": len(pairs) * len(schedule),
        "eligible_min": int(counts.min()), "eligible_median": float(counts.median()),
        "eligible_max": int(counts.max()), "eligible_mean": float(counts.mean()),
        "expected_false_positives_per_scan_at_5pct": 0.05 * len(pairs),
        "expected_false_positive_share_of_median_eligible":
            (0.05 * len(pairs)) / float(counts.median()) if counts.median() else None,
    }

    out = {"scan": scan_stats, "runs": {}}

    # ── Cost attribution + incumbent-parameter comparison ───────────────────
    final_params = json.loads((CACHE_DIR / "_checkpoint_dev.json").read_text(encoding="utf-8"))["final_params"]
    incumbent = {"entry_z": base_cfg["entry_z"], "exit_z": base_cfg["exit_z"],
                 "half_life_multiplier_max_hold": base_cfg["half_life_multiplier_max_hold"]}

    # Costs never change WHICH trades fire (exits are z-score/timeout driven,
    # tests/test_pairs_scan.py pins this), so the before/after-cost profit
    # factors inside each run are already the full cost-attribution story —
    # no separate zero-cost re-run is needed or would tell us anything new.
    variants = [
        ("dev_final_params", DEV, final_params, 3.0),
        ("dev_incumbent_params", DEV, incumbent, 3.0),
        ("holdout_final_params", HOLDOUT, final_params, 3.0),
        ("holdout_incumbent_params", HOLDOUT, incumbent, 3.0),
    ]
    for name, (start, end), params, spread in variants:
        cfg = PairsScanConfig(
            entry_z=params["entry_z"], exit_z=params["exit_z"],
            coint_lookback_days=lookback,
            revalidate_every_days=base_cfg["revalidate_every_days"],
            notional_per_leg=base_cfg["notional_per_leg"],
            half_life_multiplier_max_hold=params["half_life_multiplier_max_hold"],
            min_half_life_days=base_cfg["min_half_life_days"],
            max_half_life_days=base_cfg["max_half_life_days"],
            half_spread_bps=spread,
        )
        wc, wa = _slice(close, adv, start, end, lookback)
        report = run_scan_backtest(wc, wa, schedule, cfg)
        stats = _window_stats(report, start, end)
        stats["params"] = params
        stats["half_spread_bps"] = spread
        stats["window"] = [start, end]
        out["runs"][name] = stats
        log.info("%s: %s", name, {k: stats[k] for k in
                                  ("n_trades", "net_pnl", "profit_factor_after_costs",
                                   "profit_factor_before_costs")})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
