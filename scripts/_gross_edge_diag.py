"""
Generalized fast gross-vs-net diagnostic (NOT a WFO run) — one full-window
backtest pass at ZERO cost and one at REAL (engine default) cost, for any
signal against any cached universe, using that signal's CURRENT
configs/strategy.yaml defaults.

Built to answer a specific follow-up question after the sweep_reclaim
low-liquidity investigation: strategy_review_summary.md's "cost/frequency
mismatch (real marginal signal, but too many trades for cost to survive)"
bucket (sweep_reclaim, orb_vwap, orb_vwap_regime, fvg_retest) was
classified using each signal's NET (cost-adjusted) profit factor at the
time. sweep_reclaim's gross_PF (this investigation, low-liq universe)
turned out to be 0.805 -- BELOW 1.0 even before any cost -- which
contradicts "real marginal signal, cost is the only problem" for that one.
This script re-checks the other three (and re-checks sweep_reclaim) on the
ORIGINAL 20-symbol mega-cap universe (configs/universe.yaml) they were
actually classified against, isolating gross edge directly instead of
inferring it from a net number and a categorization label.

Usage:
    python scripts/_gross_edge_diag.py --signal orb_vwap
    python scripts/_gross_edge_diag.py --signal orb_vwap_regime
    python scripts/_gross_edge_diag.py --signal fvg_retest
    python scripts/_gross_edge_diag.py --signal sweep_reclaim --universe-config configs/universe.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from python.backtest.intraday_engine import (
    SIGNAL_PARAM_KEYS,
    IntradayBacktestConfig,
    run_intraday_backtest,
)
from python.data.fixed_universe import load_universe_config
from python.data.intraday_cache import get_cached_intraday_panel

START, END = "2025-08-01", "2026-07-01"
SIGNAL_WARMUP_DAYS = {"orb_vwap_regime": 35}


def load_bars(universe_path: str, warmup_days: int):
    universe_cfg = load_universe_config(path=universe_path)
    symbols = universe_cfg["symbols"]
    warmup_start = (pd.Timestamp(START) - pd.Timedelta(days=warmup_days)).date().isoformat()
    panel = get_cached_intraday_panel(symbols, warmup_start, END)
    out = {}
    for sym in symbols:
        if sym in panel.index.get_level_values("code"):
            out[sym] = panel.xs(sym, level="code").sort_index()
    return out, symbols


def summarize(label, report, capital, start_ts):
    trades = [t for t in report.trades if t.exit_time >= start_ts]
    n = len(trades)
    print(f"\n--- {label} ---")
    print(f"signals_emitted={report.signals_emitted} signals_filled={report.signals_filled} n_trades={n}")
    if n == 0:
        print("NO TRADES AT ALL in this window.")
        return
    gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_loss = sum(-t.gross_pnl for t in trades if t.gross_pnl < 0)
    net_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    net_loss = sum(-t.net_pnl for t in trades if t.net_pnl < 0)
    gross_pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    net_pf = (net_profit / net_loss) if net_loss > 0 else (float("inf") if net_profit > 0 else 0.0)
    total_net = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    n_sessions = len({t.entry_time.date() for t in trades}) or 1
    by_symbol = Counter(t.symbol for t in trades)
    print(f"win_rate={wins/n:.1%}  gross_PF={gross_pf:.3f}  net_PF={net_pf:.3f}  trades/session(all symbols)={n/n_sessions:.1f}")
    print(f"total_net_pnl=${total_net:,.0f} ({100*total_net/capital:.3f}% of ${capital:,.0f} capital)")
    print(f"trades per symbol: {dict(by_symbol)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", required=True)
    ap.add_argument("--universe-config", default="configs/universe.yaml")
    args = ap.parse_args()

    warmup_days = SIGNAL_WARMUP_DAYS.get(args.signal, 1)
    print(f"Loading cached 1m bars for {args.universe_config} ({args.signal}, warmup={warmup_days}d), {START}..{END} ...")
    bars_by_symbol, symbols = load_bars(args.universe_config, warmup_days)
    print(f"  {len(bars_by_symbol)}/{len(symbols)} symbols have cached bars")

    strategy_cfg = yaml.safe_load(open("configs/strategy.yaml", encoding="utf-8"))[args.signal]
    sig_keys = SIGNAL_PARAM_KEYS[args.signal]
    params = {k: strategy_cfg[k] for k in sig_keys if k in strategy_cfg}
    print(f"\n{args.signal} params (current configs/strategy.yaml defaults): {params}")

    capital = 1_000_000.0
    start_ts = pd.Timestamp(START)

    zero_cost_cfg = IntradayBacktestConfig(
        capital=capital, half_spread_bps=0.0, impact_bps_per_participation=0.0,
        commission_per_share=0.0, min_commission=0.0,
    )
    real_cost_cfg = IntradayBacktestConfig(capital=capital)

    print("\nRunning ZERO-COST pass...")
    report_zero = run_intraday_backtest(bars_by_symbol, args.signal, params, zero_cost_cfg)
    summarize("ZERO-COST (gross edge only)", report_zero, capital, start_ts)

    print("\nRunning REAL-COST pass (engine defaults: 2.0bps half-spread flat, 20bps impact)...")
    report_real = run_intraday_backtest(bars_by_symbol, args.signal, params, real_cost_cfg)
    summarize("REAL-COST", report_real, capital, start_ts)


if __name__ == "__main__":
    main()
