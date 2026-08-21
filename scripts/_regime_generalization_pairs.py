"""
Regime-generalization test — `pairs_trading`, two ALREADY-FIXED configurations
from prior rounds, run UNMODIFIED against calendar windows they have never
been evaluated against (2020 COVID crash, 2022 rate-hike bear, optionally
2018-2019).

THIS IS NOT A NEW TUNING ROUND. Both configurations below were selected
during the 2024-2026-window campaign and are frozen here exactly as reported:

  pairs_dynamic_halflife_exit
      `backtests/reports/pairs_scan_report.md` round two, ablation A1
      (dynamic half-life re-estimation on exit). final_params taken verbatim
      from `_pairs_scan_cache/_checkpoint_ablation_dev_A1_dynamic_half_life.json`
      (the dev-window's own last-fold-winner selection, NOT re-picked here):
      entry_z=2.5, exit_z=1.0, half_life_multiplier_max_hold=4.0,
      exit_rules={dynamic_half_life: True}. This is the only configuration in
      the whole campaign with gross PF > 1.0 from an exit-rule change (1.037
      on round two's own dev window with ITS OWN final_params — the 2.5/1.0/4.0
      params above, not round one's 2.5/0.0/2.0).

  pairs_lowfreq_entry_z_4
      `backtests/reports/alt_universe_frequency_exploration.md` §3.2 (Track
      2b). final_params verbatim from that report's holdout run (also pinned
      in `_checkpoint_track2_lowfreq_holdout.json`): entry_z=4.0, exit_z=0.5,
      half_life_multiplier_max_hold=3.0, baseline exit rule (no ablation).
      Best-ever full-window gross PF in the campaign (1.106).

Both reuse, VERBATIM and WITHOUT recomputation:
  - `configs/pairs_universe.yaml`'s 66-ETF, 6-bucket, within-bucket-only
    candidate universe (untouched).
  - The already-built point-in-time cointegration scan schedule
    (`backtests/reports/_pairs_scan_cache/scan_schedule.jsonl`), which
    already spans 2016-06-01..2026-07-31 — i.e. it ALREADY covers 2018-2022
    with the same point-in-time discipline verified in
    `pairs_scan_report.md` §8 and `holdout_methodology_dossier.md` §1.2. No
    new scan is run; this script only replays it over new date windows.
  - `python/backtest/pairs_scan_engine.run_scan_backtest` /
    `python/backtest/optimize.build_pairs_scan_backtest_fn` unmodified.
  - The same 3.0bps ETF half-spread assumption (`DEFAULT_HALF_SPREAD_BPS`)
    and the same mandatory 2x stress re-run (6.0bps), unchanged from every
    prior pairs-trading round. See the regime_generalization_report.md for
    the caveat that 2020's realized crash volatility means true intraday
    spreads were almost certainly wider than this calm-period-derived
    assumption.

No parameter is searched, gridded, or re-picked anywhere in this file.

Usage:
    python scripts/_regime_generalization_pairs.py               # all windows, all configs
    python scripts/_regime_generalization_pairs.py --window 2020_covid
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("python.core.pair_position_manager").setLevel(logging.WARNING)
log = logging.getLogger("regime_generalization_pairs")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import build_pairs_scan_backtest_fn, max_oos_drawdown_threshold
from python.backtest.pairs_scan_engine import (
    DEFAULT_HALF_SPREAD_BPS,
    MAX_CONCURRENT_PAIRS,
    STRESS_HALF_SPREAD_MULTIPLIER,
)
from run_pairs_scan_backtest import _load_schedule, _strategy_cfg, _strip, _write_json, load_panels

CACHE_DIR = Path("backtests/reports/_regime_generalization_cache")
REPORT_JSON = Path("backtests/reports/regime_generalization_pairs.json")

# ── Frozen configurations (do not edit without updating the report's
# provenance citation) ───────────────────────────────────────────────────────
FROZEN_CONFIGS: dict[str, dict] = {
    "pairs_dynamic_halflife_exit": {
        "final_params": {"entry_z": 2.5, "exit_z": 1.0, "half_life_multiplier_max_hold": 4.0},
        "exit_rules": {"dynamic_half_life": True},
        "source": (
            "pairs_scan_report.md round 2 (exit-rule ablations), variant A1 "
            "'dynamic_half_life'; final_params = dev-window last-fold winner, "
            "verbatim from _checkpoint_ablation_dev_A1_dynamic_half_life.json"
        ),
        "prior_result_2024_2026": {
            "window": "dev 2018-01-01..2024-01-01 (this variant was NOT sent to "
                      "the 2024-2026 holdout — A2 was, per the pre-declared "
                      "selection rule; A1's number below is its OWN dev-window "
                      "full-window run, the only number this exact config has)",
            "gross_pf": 1.037, "cost_adjusted_pf": 0.910, "sharpe": -0.296,
            "monte_carlo_p5_sharpe": -0.975, "net_pnl": -53117, "trades": 670,
        },
    },
    "pairs_lowfreq_entry_z_4": {
        "final_params": {"entry_z": 4.0, "exit_z": 0.5, "half_life_multiplier_max_hold": 3.0},
        "exit_rules": {},
        "source": (
            "alt_universe_frequency_exploration.md §3.2 (Track 2b), holdout "
            "final_params, verbatim from _checkpoint_track2_lowfreq_holdout.json"
        ),
        "prior_result_2024_2026": {
            "window": "holdout 2024-01-01..2026-08-01",
            "gross_pf": 0.747, "cost_adjusted_pf": 0.675, "sharpe": -0.963,
            "monte_carlo_p5_sharpe": -1.864, "net_pnl": -58233, "trades": 151,
        },
    },
}

# ── Test windows. All strictly inside the already-built 2016-06-01..2026-07-31
# scan schedule and price panel — no new scan or fetch is needed. ──────────
WINDOWS: dict[str, tuple[str, str]] = {
    "2018_2019_calmer": ("2018-01-01", "2020-01-01"),
    "2020_covid": ("2020-01-01", "2021-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
}


def _panel_args() -> argparse.Namespace:
    # Matches run_pairs_scan_backtest.py's CLI defaults exactly — same
    # warmup_start / holdout_end, so load_panels() reproduces the identical
    # close/adv panel and candidate-pair list already used by every prior
    # pairs-trading round.
    return argparse.Namespace(
        warmup_start="2016-06-01",
        start="2018-01-01",
        dev_end="2024-01-01",
        holdout_end="2026-08-01",
        refresh_data=False,
    )


def _run_one(name: str, cfg: dict, window_name: str, start: str, end: str,
             close, adv, schedule, base_cfg) -> dict:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=DEFAULT_HALF_SPREAD_BPS,
        max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
        exit_rules=cfg["exit_rules"],
    )
    log.info("[%s / %s] single run of %s over [%s, %s)",
              name, window_name, cfg["final_params"], start_ts.date(), end_ts.date())
    metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg["final_params"])
    mc = MonteCarloValidator(n_sims=500).run(metrics.get("daily_returns", []))

    stress_fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=DEFAULT_HALF_SPREAD_BPS * STRESS_HALF_SPREAD_MULTIPLIER,
        max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
        exit_rules=cfg["exit_rules"],
    )
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg["final_params"])

    goal = yaml.safe_load(Path("configs/goal.yaml").read_text(encoding="utf-8"))
    min_p5 = float(goal["monte_carlo"]["min_p5_sharpe"])
    min_oos_sharpe = float(goal["wfo"]["min_oos_sharpe"])
    max_dd = max_oos_drawdown_threshold()
    gates = {
        "has_trades": metrics["n_trades"] > 0,
        "sharpe_above_min_oos_sharpe": metrics["sharpe_ratio"] >= min_oos_sharpe,
        "drawdown_within_limit": abs(metrics["max_drawdown"]) <= max_dd,
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
    }
    return {
        "config": name,
        "config_source": cfg["source"],
        "final_params": cfg["final_params"],
        "exit_rules": cfg["exit_rules"],
        "window_name": window_name,
        "window": [str(start_ts.date()), str(end_ts.date())],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "half_spread_bps": DEFAULT_HALF_SPREAD_BPS,
        "metrics": _strip(metrics),
        "stress_2x_spread": _strip(stress),
        "monte_carlo": mc.to_dict(),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", choices=list(WINDOWS), default=None,
                        help="run only this window (default: all)")
    parser.add_argument("--config", choices=list(FROZEN_CONFIGS), default=None,
                        help="run only this config (default: both)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    close, adv, _universe, candidate_pairs, meta = load_panels(_panel_args())
    base_cfg = _strategy_cfg()
    schedule = _load_schedule(close, candidate_pairs, base_cfg)
    log.info("panel/schedule loaded: %s", meta)

    windows = {args.window: WINDOWS[args.window]} if args.window else WINDOWS
    configs = {args.config: FROZEN_CONFIGS[args.config]} if args.config else FROZEN_CONFIGS

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for cfg_name, cfg in configs.items():
        for window_name, (start, end) in windows.items():
            ckpt = CACHE_DIR / f"pairs_{cfg_name}_{window_name}.json"
            if ckpt.exists() and not args.force:
                log.info("resuming [%s / %s] from %s", cfg_name, window_name, ckpt)
                all_results[f"{cfg_name}__{window_name}"] = json.loads(ckpt.read_text(encoding="utf-8"))
                continue
            result = _run_one(cfg_name, cfg, window_name, start, end, close, adv, schedule, base_cfg)
            _write_json(ckpt, result)
            all_results[f"{cfg_name}__{window_name}"] = result

    _write_json(REPORT_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Regime-generalization test. Both configs are frozen from prior "
            "campaign rounds and re-run UNMODIFIED against new calendar "
            "windows; no parameter here was selected or tuned using any "
            "result in this file."
        ),
        "data": meta,
        "results": all_results,
    })
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
