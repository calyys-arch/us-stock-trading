"""
Phase 1 replication test (regime_gate_report.md) — PRIMARY window.

PRE-DECLARED HYPOTHESIS (written before this script was ever run, and before
any result from it was seen): 2008-01-01..2010-01-01 (the global financial
crisis: the 2008 credit crunch / Lehman collapse crash through the 2009
V-shaped recovery) is, by well-documented historical record predating this
task, an "acute stress, crash-then-recovery" regime sharing the two-way,
genuinely mean-reverting daily-move character of 2020's COVID crash
(`regime_generalization_report.md` §1b) rather than the one-directional
persistent-trend character of 2022 or 2024-2026. If `pairs_trading`'s
low-frequency `entry_z=4.0` config (or the dynamic-half-life config) shows the
SAME qualitative pattern here — meaningfully better than its 2024-2026/2018-19
results — that is independent replication. If not, the 2022 flip is most
likely small-sample noise, the direct analog of this campaign's four prior
opposite-direction false positives.

BOTH frozen configs below are copied VERBATIM from
`scripts/_regime_generalization_pairs.py`'s FROZEN_CONFIGS — not one
parameter is re-picked, re-tuned, or gridded here. Do not edit
`final_params`/`exit_rules` without updating the citation.

WHY A NEW FETCH + NEW SCAN SCHEDULE (not the existing
2016-06-01..2026-07-31 one): the existing point-in-time cointegration scan
schedule (`backtests/reports/_pairs_scan_cache/scan_schedule.jsonl`) does not
cover 2008-2009 at all. This script builds a SEPARATE schedule under its own
cache directory using the exact same, unmodified
`python/backtest/pairs_scan_engine.build_scan_schedule` /
`python/stat/pair_scanner.scan` — same lookback (252 trading days), same
revalidate cadence (21 trading days), same within-bucket-only candidate
pairs, same CADF/half-life screen — just fed 2006-2009 price data instead of
2016-2026 price data. No engine or scan code is modified.

DATA-AVAILABILITY EXCLUSION (checked BEFORE running, a data fact not a
result): 9 of the 66 `configs/pairs_universe.yaml` ETFs have zero price
history before 2008 per a direct yfinance fetch probe on 2026-08-14 (Yahoo
returns "no price data found" for the whole pre-listing range, so — exactly
like the xsection regime-generalization script's NBIS/SNDK handling — they
must be excluded from the fetch call itself): BND, EMB, GDXJ, HYG, JNK, MBB,
SIL, UNG, XLRE (inception dates 2009-2015). The remaining 57 symbols all have
data from 2006-06-22 or earlier, comfortably before this script's
`warmup_start` (2006-09-01) + 10-day tolerance, so
`run_pairs_scan_backtest.load_panels`'s existing too-short-history guard
drops nothing further.

Usage:
    python scripts/_regime_gate_phase1_gfc_pairs.py
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
log = logging.getLogger("regime_gate_phase1_gfc_pairs")

from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.optimize import build_pairs_scan_backtest_fn, max_oos_drawdown_threshold
from python.backtest.pairs_scan_engine import (
    DEFAULT_HALF_SPREAD_BPS,
    MAX_CONCURRENT_PAIRS,
    STRESS_HALF_SPREAD_MULTIPLIER,
    build_scan_schedule,
    candidate_pairs_from_buckets,
    load_pairs_universe,
)
from python.data.price_cache import get_cached_price_panel

CACHE_DIR = Path("backtests/reports/_regime_gate_phase1_cache")
SCAN_CHECKPOINT = CACHE_DIR / "gfc_scan_schedule.jsonl"
REPORT_JSON = Path("backtests/reports/regime_gate_phase1_gfc_pairs.json")

WARMUP_START = "2006-09-01"
FETCH_END = "2010-01-15"

# Empirically confirmed (this task, 2026-08-14) via a direct yfinance batch
# fetch probe: these 9 `pairs_universe.yaml` ETFs have ZERO price history
# before 2008 (real-world inception 2009-2015) and yfinance errors on the
# whole batch if they are left in the fetch call, exactly the failure mode
# `_regime_generalization_xsection.py` already documented for NBIS/SNDK.
_SYMBOLS_NOT_YET_LISTED_BY_2008 = {
    "BND", "EMB", "GDXJ", "HYG", "JNK", "MBB", "SIL", "UNG", "XLRE",
}

FROZEN_CONFIGS: dict[str, dict] = {
    "pairs_dynamic_halflife_exit": {
        "final_params": {"entry_z": 2.5, "exit_z": 1.0, "half_life_multiplier_max_hold": 4.0},
        "exit_rules": {"dynamic_half_life": True},
        "source": "verbatim from scripts/_regime_generalization_pairs.py FROZEN_CONFIGS "
                  "(pairs_scan_report.md round 2, ablation A1)",
    },
    "pairs_lowfreq_entry_z_4": {
        "final_params": {"entry_z": 4.0, "exit_z": 0.5, "half_life_multiplier_max_hold": 3.0},
        "exit_rules": {},
        "source": "verbatim from scripts/_regime_generalization_pairs.py FROZEN_CONFIGS "
                  "(alt_universe_frequency_exploration.md Track 2b holdout)",
    },
}

WINDOW_NAME = "2008_2009_gfc"
WINDOW = ("2008-01-01", "2010-01-01")


def _strategy_cfg() -> dict:
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["pairs_trading"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("checkpoint written: %s", path)


def _strip(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "daily_returns"}


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    universe = load_pairs_universe()
    all_symbols = sorted({s for codes in universe["buckets"].values() for s in codes})
    fetchable = [s for s in all_symbols if s not in _SYMBOLS_NOT_YET_LISTED_BY_2008]
    panel, quality_flags, meta = get_cached_price_panel(
        fetchable, WARMUP_START, FETCH_END,
        cache_dir="data/history_gfc2008",  # separate cache dir: never mixes with the
                                            # 2016-2026 panel already used by every other round
    )
    close = panel["close"].unstack("code").sort_index()
    adv = panel["adv_20d_dollars"].unstack("code").sort_index()

    warmup_ts = pd.Timestamp(WARMUP_START)
    tolerance = warmup_ts + pd.Timedelta(days=10)
    too_short = [c for c in close.columns if close[c].first_valid_index() is None
                 or close[c].first_valid_index() > tolerance]
    if too_short:
        log.warning("dropping %d symbols whose history starts after %s: %s",
                    len(too_short), tolerance.date(), sorted(too_short))
        close = close.drop(columns=too_short)
        adv = adv.drop(columns=too_short)

    buckets = {b: [s for s in codes if s in close.columns]
               for b, codes in universe["buckets"].items()}
    candidate_pairs = candidate_pairs_from_buckets(buckets)

    meta_out = {
        "sources": {k: len(v) for k, v in meta["sources"].items()},
        "n_symbols": int(close.shape[1]),
        "symbols_excluded_not_yet_listed": sorted(_SYMBOLS_NOT_YET_LISTED_BY_2008),
        "n_symbols_dropped_short_history_after_fetch": len(too_short),
        "symbols_dropped_short_history": sorted(too_short),
        "n_candidate_pairs": len(candidate_pairs),
        "bucket_sizes": {b: len(c) for b, c in buckets.items()},
        "first_date": str(close.index.min().date()),
        "last_date": str(close.index.max().date()),
        "n_trading_days": int(len(close)),
    }
    return close, adv, candidate_pairs, meta_out


def _run_one(name: str, cfg: dict, close, adv, schedule, base_cfg) -> dict:
    start_ts, end_ts = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    fn = build_pairs_scan_backtest_fn(
        close, adv, schedule, base_cfg,
        half_spread_bps=DEFAULT_HALF_SPREAD_BPS,
        max_concurrent_pairs=MAX_CONCURRENT_PAIRS,
        exit_rules=cfg["exit_rules"],
    )
    log.info("[%s] single run of %s over [%s, %s)", name, cfg["final_params"], start_ts.date(), end_ts.date())
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
        "window_name": WINDOW_NAME,
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    close, adv, candidate_pairs, meta = load_panels()
    base_cfg = _strategy_cfg()
    log.info("panel loaded: %s", meta)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    schedule = build_scan_schedule(
        close, candidate_pairs,
        lookback_days=base_cfg["coint_lookback_days"],
        revalidate_every_days=base_cfg["revalidate_every_days"],
        checkpoint_path=SCAN_CHECKPOINT,
    )
    passing = [len(v) for v in schedule.values()]
    scan_summary = {
        "n_scan_dates": len(schedule),
        "n_candidate_pairs": len(candidate_pairs),
        "eligible_pairs_per_scan_min": int(min(passing)) if passing else 0,
        "eligible_pairs_per_scan_median": float(pd.Series(passing).median()) if passing else 0.0,
        "eligible_pairs_per_scan_max": int(max(passing)) if passing else 0,
    }
    log.info("scan schedule: %s", scan_summary)

    all_results = {}
    for cfg_name, cfg in FROZEN_CONFIGS.items():
        ckpt = CACHE_DIR / f"gfc_{cfg_name}.json"
        if ckpt.exists() and not args.force:
            log.info("resuming [%s] from %s", cfg_name, ckpt)
            all_results[cfg_name] = json.loads(ckpt.read_text(encoding="utf-8"))
            continue
        result = _run_one(cfg_name, cfg, close, adv, schedule, base_cfg)
        _write_json(ckpt, result)
        all_results[cfg_name] = result

    _write_json(REPORT_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Phase 1 PRIMARY replication window (2008-2009 GFC). Both configs "
            "frozen verbatim from scripts/_regime_generalization_pairs.py; "
            "window declared BEFORE this script was run (see module docstring)."
        ),
        "data": meta,
        "scan_summary": scan_summary,
        "results": all_results,
    })
    log.info("done -> %s", REPORT_JSON)


if __name__ == "__main__":
    main()
