#!/usr/bin/env python3
"""Stress-test the VIX overlay against the exact crisis it was designed for.

python/portfolio/risk_controls.py's module docstring justifies the VIX
brackets by name: "The thresholds are based on historical VIX quartiles and
GFC stress test results." docs/2026_09_10_deliverables.md then claims "GFC
回測顯示，VIX overlay 可將回撤從 -56% 降到 -30%". Neither claim was ever backed
by a run of the VIX overlay itself -- see backtests/reports/
v1_vix_validation_report.md for the full account. That -56% -> -30% number is
backtests/reports/vol_target_gfc_stress.json's "只留股票" (equity-only) arm,
UNSCALED -> VOL_TARGET, which contains no VIX logic at all.

This script runs the thing that was actually claimed: Strategy V1's real
bucket-equal + 15%-vol-target pipeline, extended with the VIX overlay,
against 2006-09-01 -> 2010-01-14 using the existing data/history_gfc2008
cache (57 ETFs) plus a freshly-fetched ^VIX series for the same window.

FOUR ARMS, same underlying bucket-weighted gross return series:
  unscaled     exposure = 1.0 throughout (the bucket_equal gross return
               series itself, no vol targeting, no VIX).
  vol_target   today's live mechanism: exposure_path(gross, 0.15%), the
               same function scripts/strategy_v1_positions.py and
               scripts/paper_track_v1.py use.
  vol_target + VIX overlay   the SAME base exposure above, multiplied by
               adjust_exposure_for_vix()'s bracket scalar (20/30/40 ->
               0.8/0.6/0.4), turnover cost recomputed on the adjusted series.
  constant_scale (control)   a FIXED exposure equal to the VIX arm's own
               realized mean exposure, held flat with no timing at all. The
               VIX overlay can only scale exposure DOWN, so part of any
               improvement over vol_target is mechanically just running
               lower average exposure -- "less exposure alone cuts
               drawdown" (run_vol_target_study.py's own stated discipline).
               vol_target+VIX only demonstrates real timing value if it
               beats this control, not just the vol_target arm.

The VIX_BRACKETS thresholds are NOT touched or tuned here -- this is a
one-shot test of the frozen rule, not a parameter search.

TWO UNIVERSES, same structure as run_vol_target_study.gfc_stress_test():
  full        all 57 cached ETFs split into V1's real 4 buckets (us_equity,
              intl_equity, bonds, commodities) via each ticker's fine-grained
              label in config/strategy_v1_universe.json (COLLAPSE'd the same
              way strategy_v1.bucket_weighted_gross does it internally).
  equity_only us_equity + intl_equity buckets only (bonds/commodities
              dropped), so the comparison is not partly "diversification
              into things that rallied in 2008" -- the arm that actually has
              a crash to survive, same reasoning as the existing GFC study.

Universe mapping: every one of the 57 tickers in data/history_gfc2008
already has a fine-grained bucket label in config/strategy_v1_universe.json
(they are ETFs already in that 120-name universe) -- verified by intersecting
load_universe()'s fine bucket_of with the GFC panel's columns; nothing here
is invented or hand-classified.

Usage:
    .venv/bin/python scripts/run_v1_vix_gfc_stress.py --save
    (needs network to fetch ^VIX 2006-09-01..2010-01-14; run with
    required_permissions ["network"] or ["full_network"])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python.portfolio.strategy_v1 import bucket_weighted_gross, load_universe, COLLAPSE  # noqa: E402
from python.portfolio.risk_controls import adjust_exposure_for_vix  # noqa: E402
from python.simulation.hist_data_us import fetch_daily_bars  # noqa: E402
from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from scripts.run_daily_baseline_gates import load_panel, _sharpe, _max_dd, _profit_factor  # noqa: E402
from scripts.run_vol_target_study import (  # noqa: E402
    exposure_path, apply_exposure, VOL_LOOKBACK, REBALANCE_BAND,
    ONE_WAY_COST_BPS, MAX_DD_LIMIT,
)

GFC_DIR = ROOT / "data/history_gfc2008"
VOL_TARGET = 0.15  # frozen live target; not fitted here
OUT_JSON = ROOT / "backtests/reports/v1_vix_gfc_stress.json"

# Same non-equity set as run_vol_target_study.gfc_stress_test, expressed as
# collapsed V1 buckets rather than an ad hoc ticker list: dropping these
# removes the 2008 Treasury/gold rally so the equity_only arm has an actual
# crash to survive.
EQUITY_ONLY_BUCKETS = {"us_equity", "intl_equity"}


def gfc_bucket_of() -> dict[str, str]:
    """Fine-grained bucket label for each ticker cached in data/history_gfc2008,
    taken directly from config/strategy_v1_universe.json (no hand mapping)."""
    all_symbols, all_bucket_of = load_universe()
    cached = sorted(p.stem for p in GFC_DIR.glob("*.csv"))
    missing = [s for s in cached if s not in all_bucket_of]
    if missing:
        raise RuntimeError(
            f"{len(missing)} GFC-cached ticker(s) have no bucket label in "
            f"config/strategy_v1_universe.json: {missing}")
    return {s: all_bucket_of[s] for s in cached}


def fetch_gfc_vix() -> pd.Series:
    print("  抓取 ^VIX 2006-09-01 -> 2010-01-14 ...")
    vix_df = fetch_daily_bars("^VIX", "2006-08-20", "2010-01-15")
    vix = vix_df["close"].rename("VIX")
    print(f"    ^VIX: {len(vix)} 日，{vix.index[0].date()} 到 {vix.index[-1].date()}")
    return vix


def _summ(net: pd.Series, sims: int) -> dict:
    mc = MonteCarloValidator(n_sims=sims, seed=42).run(
        [float(v) for v in net.tolist()])
    dd = _max_dd(net)
    return {
        "n_days": int(len(net)),
        "sharpe": _sharpe(net),
        "max_drawdown": dd,
        "profit_factor": _profit_factor(net),
        "mc_p5_sharpe": float(mc.sharpe.p5),
        "drawdown_gate_pass": dd >= MAX_DD_LIMIT,
    }


def run_universe(label: str, panel: pd.DataFrame, bucket_of: dict[str, str],
                 vix: pd.Series, sims: int) -> dict:
    print(f"\n== {label} ==  {panel.shape[1]} 檔 ETF，{len(panel)} 個交易日")

    gross, bucket_weights = bucket_weighted_gross(
        panel, bucket_of, "equal", VOL_LOOKBACK, REBALANCE_BAND, ONE_WAY_COST_BPS)
    print(f"   桶：{sorted(bucket_weights.columns)}")

    # --- arm 1: unscaled (exposure = 1.0 throughout) ---
    unscaled = gross

    # --- arm 2: vol_target only (today's live mechanism) ---
    base_exposure = exposure_path(gross, VOL_TARGET)
    net_vol_target = apply_exposure(gross, base_exposure)
    mean_exp_vt = float(base_exposure.reindex(net_vol_target.index).mean())

    # --- arm 3: vol_target + VIX overlay ---
    vix_aligned = vix.reindex(base_exposure.index, method="ffill")
    missing_vix = int(vix_aligned.isna().sum())
    adjusted_exposure = adjust_exposure_for_vix(base_exposure, vix_aligned)
    net_vix = apply_exposure(gross, adjusted_exposure)
    mean_exp_vix = float(adjusted_exposure.reindex(net_vix.index).mean())

    # --- arm 4 (control): constant exposure == VIX arm's own mean, no timing ---
    flat = pd.Series(mean_exp_vix, index=gross.index)
    net_constant = apply_exposure(gross, flat)

    row = {
        "n_symbols": int(panel.shape[1]),
        "buckets": sorted(bucket_weights.columns),
        "vix_missing_days_after_ffill": missing_vix,
        "unscaled": _summ(unscaled, sims) | {"mean_exposure": 1.0},
        "vol_target": _summ(net_vol_target, sims) | {"mean_exposure": mean_exp_vt},
        "vol_target_plus_vix": _summ(net_vix, sims) | {"mean_exposure": mean_exp_vix},
        "constant_scale_control": _summ(net_constant, sims) | {"mean_exposure": mean_exp_vix},
    }
    row["vix_vs_vol_target_only"] = {
        "dd_delta_pts": (row["vol_target_plus_vix"]["max_drawdown"]
                        - row["vol_target"]["max_drawdown"]) * 100,
        "sharpe_delta": (row["vol_target_plus_vix"]["sharpe"]
                        - row["vol_target"]["sharpe"]),
        "mc_p5_sharpe_delta": (row["vol_target_plus_vix"]["mc_p5_sharpe"]
                              - row["vol_target"]["mc_p5_sharpe"]),
    }
    row["vix_vs_constant_scale"] = {
        "dd_delta_pts": (row["vol_target_plus_vix"]["max_drawdown"]
                        - row["constant_scale_control"]["max_drawdown"]) * 100,
        "sharpe_delta": (row["vol_target_plus_vix"]["sharpe"]
                        - row["constant_scale_control"]["sharpe"]),
        "mc_p5_sharpe_delta": (row["vol_target_plus_vix"]["mc_p5_sharpe"]
                              - row["constant_scale_control"]["mc_p5_sharpe"]),
    }

    for arm_key, arm_name in (("unscaled", "unscaled        "),
                              ("vol_target", "vol_target      "),
                              ("vol_target_plus_vix", "vol_target+VIX  "),
                              ("constant_scale_control", "固定同曝險(控制)")):
        a = row[arm_key]
        print(f"   {arm_name}曝險 {a['mean_exposure']:.2f}  回撤 "
              f"{a['max_drawdown']:.1%}  Sharpe {a['sharpe']:.2f}  "
              f"MC p5 {a['mc_p5_sharpe']:+.2f}  "
              f"閘門 {'PASS' if a['drawdown_gate_pass'] else 'FAIL'}")
    d = row["vix_vs_vol_target_only"]
    print(f"   → VIX 疊加相對純 vol_target：回撤差 {d['dd_delta_pts']:+.1f} 個百分點，"
          f"Sharpe 差 {d['sharpe_delta']:+.2f}，MC p5 Sharpe 差 "
          f"{d['mc_p5_sharpe_delta']:+.2f}")
    dc = row["vix_vs_constant_scale"]
    print(f"   → VIX 疊加相對「同平均曝險但不擇時」：回撤差 {dc['dd_delta_pts']:+.1f} "
          f"個百分點，Sharpe 差 {dc['sharpe_delta']:+.2f}，MC p5 Sharpe 差 "
          f"{dc['mc_p5_sharpe_delta']:+.2f}"
          f"{'（擇時有貢獻）' if dc['dd_delta_pts'] > 0 and dc['sharpe_delta'] > 0 else '（擇時貢獻不明確，可能只是降曝險的功勞）'}")

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print("=== V1 + VIX 疊加 GFC 壓力測試 (2006-09-01 -> 2010-01-14) ===")

    full_panel = load_panel(min_rows=800, directory=GFC_DIR)
    bucket_of_fine = gfc_bucket_of()
    print(f"全部快取：{full_panel.shape[1]} 檔，{len(full_panel)} 個交易日 "
          f"({full_panel.index[0].date()} -> {full_panel.index[-1].date()})")

    equity_cols = [s for s in full_panel.columns
                  if COLLAPSE[bucket_of_fine[s]] in EQUITY_ONLY_BUCKETS]
    equity_panel = full_panel[equity_cols]
    print(f"股票子集（us_equity+intl_equity）：{equity_panel.shape[1]} 檔 "
          f"（剔除 {full_panel.shape[1] - equity_panel.shape[1]} 檔債券/商品）")
    print(f"凍結目標波動 = {VOL_TARGET:.0%}（沿用實盤設定，此處不擬合）")
    print(f"VIX 分級門檻/scalar 為既定規則（20/30/40 對應 0.8/0.6/0.4），此處不調整\n")

    vix = fetch_gfc_vix()

    results = {}
    results["full_57"] = run_universe(
        "全部 57 檔（V1 四個粗桶）", full_panel, bucket_of_fine, vix, args.sims)
    results["equity_only"] = run_universe(
        "只留股票（us_equity + intl_equity 兩桶）", equity_panel, bucket_of_fine,
        vix, args.sims)

    if args.save:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window": ["2006-09-01", "2010-01-14"],
            "frozen_target_vol": VOL_TARGET,
            "max_dd_limit": MAX_DD_LIMIT,
            "vix_brackets_note": ("VIX_BRACKETS thresholds/scalars "
                                 "(20/30/40 -> 0.8/0.6/0.4) are the frozen "
                                 "live rule; nothing fitted here"),
            "vol_lookback": VOL_LOOKBACK,
            "rebalance_band": REBALANCE_BAND,
            "one_way_cost_bps": ONE_WAY_COST_BPS,
            "universes": results,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {OUT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
