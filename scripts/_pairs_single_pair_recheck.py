"""
Re-run the ORIGINAL single-pair backtest (AMAT/LRCX, the configuration behind
`backtests/reports/us_equity_health_check.md`'s n_trades=8) now that
`python/stat/cointegration.py`'s spread-mean/intercept mismatch is fixed, and
compare against a deliberately re-broken control.

Purpose: establish whether the "8 trades in 7 years" figure — the entire basis
for `backtests/reports/strategy_review_summary.md` §2.1's rare-event
diagnosis — was a property of the strategy or an artifact of that bug. Report
only; writes backtests/reports/_pairs_scan_cache/_single_pair_recheck.json and
changes no config.
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
log = logging.getLogger("single_pair_recheck")

import python.stat.cointegration as coint_mod
from python.backtest.engine import PairsBacktestConfig, run_pairs_backtest
from python.data.price_cache import get_cached_price_panel

OUT = Path("backtests/reports/_pairs_scan_cache/_single_pair_recheck.json")
PAIR = ("AMAT", "LRCX")
WINDOW = ("2018-01-01", "2025-01-01")


def _summary(report, label: str) -> dict:
    reasons = pd.Series([t.exit_reason for t in report.trades]).value_counts().to_dict() if report.trades else {}
    d = report.to_dict()
    d.update({"label": label, "exit_reasons": reasons,
              "total_cost": sum(t.cost for t in report.trades),
              "gross_pnl": sum(t.gross_pnl for t in report.trades)})
    return d


def main() -> None:
    cfg_block = yaml.safe_load(Path("configs/strategy.yaml").read_text(encoding="utf-8"))["pairs_trading"]
    cfg = PairsBacktestConfig(
        entry_z=cfg_block["entry_z"], exit_z=cfg_block["exit_z"],
        coint_lookback_days=cfg_block["coint_lookback_days"],
        revalidate_every_days=cfg_block["revalidate_every_days"],
        notional_per_leg=cfg_block["notional_per_leg"],
        half_life_multiplier_max_hold=cfg_block["half_life_multiplier_max_hold"],
        min_half_life_days=cfg_block["min_half_life_days"],
        max_half_life_days=cfg_block["max_half_life_days"],
    )
    panel, _flags, meta = get_cached_price_panel(list(PAIR), *WINDOW)
    prices_a = panel.xs(PAIR[0], level=1)["close"]
    prices_b = panel.xs(PAIR[1], level=1)["close"]
    log.info("%s/%s: %d aligned daily bars (%s)", *PAIR, min(len(prices_a), len(prices_b)),
             "+".join(sorted(meta["sources"])))

    fixed = _summary(run_pairs_backtest(*PAIR, prices_a, prices_b, cfg), "fixed_spread_mean")

    # Control: reinstate the old behavior (spread_mean summarizing the OLS
    # RESIDUAL, i.e. ~0, while current_spread keeps returning alpha+residual).
    real_test_pair = coint_mod.test_pair

    def legacy_test_pair(code_a, code_b, pa, pb, computed_at=None):
        r = real_test_pair(code_a, code_b, pa, pb, computed_at=computed_at)
        r.spread_mean = 0.0
        return r

    import python.backtest.engine as engine_mod

    engine_mod.test_pair = legacy_test_pair
    try:
        legacy = _summary(run_pairs_backtest(*PAIR, prices_a, prices_b, cfg), "legacy_spread_mean_zero")
    finally:
        engine_mod.test_pair = real_test_pair

    out = {
        "pair": "/".join(PAIR),
        "window": list(WINDOW),
        "config": cfg.__dict__,
        "n_trading_days": int(min(len(prices_a), len(prices_b))),
        "data_source": "+".join(sorted(meta["sources"])),
        "legacy": legacy,
        "fixed": fixed,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("legacy: %s", legacy)
    log.info("fixed : %s", fixed)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
