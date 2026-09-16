"""
Follow-up to _lowliq_diag_sweep_reclaim.py: the default params
(sweep_min_atr=0.15, stop_atr_mult=0.25) produced a whipsaw death-spiral on
the 5-name low-liquidity universe (17 trades/session/symbol, 21% win rate,
gross PF 0.805 even at ZERO cost). This script asks the follow-up question:
is that specific to the DEFAULT parameter values (i.e. "参数搬错地方" —
re-calibrate sweep_min_atr/stop_atr_mult for this liquidity tier's actual
noise level and it should behave), or does the sweep-and-reclaim LOGIC
itself stay broken across a much wider parameter range than
configs/param_grids.yaml's existing [0.1-0.2] / [0.15-0.35] grid ever
tried ("逻辑本身无效")?

Sweeps sweep_min_atr x stop_atr_mult over a MUCH wider range than the
existing param grid, at ZERO COST ONLY (isolates the pure directional/gross
question — if a combo cannot clear PF 1.0 even before any transaction cost,
no cost-model fix could ever rescue it, so there is no point spending the
extra runtime on the real-cost pass for combos that fail here).

Usage: python scripts/_lowliq_recalib_sweep_reclaim.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from python.backtest.intraday_engine import IntradayBacktestConfig, run_intraday_backtest
from python.data.fixed_universe import load_universe_config
from python.data.intraday_cache import get_cached_intraday_panel

UNIVERSE_PATH = "configs/alt_universe_lowliq.yaml"
START, END = "2025-08-01", "2026-07-01"
WARMUP_START = "2025-07-31"

# Existing configs/param_grids.yaml only ever tried sweep_min_atr in
# [0.1, 0.15, 0.2] and stop_atr_mult in [0.15, 0.25, 0.35] (all narrow,
# "loose trigger / tight stop" neighborhood) -- deliberately going much
# wider here than that grid ever covered.
SWEEP_MIN_ATR_GRID = [0.15, 0.3, 0.6, 1.0]
STOP_ATR_MULT_GRID = [0.25, 0.5, 1.0, 2.0]
RECLAIM_BARS = 3  # held fixed -- not the lever this test is about


def load_bars():
    universe_cfg = load_universe_config(path=UNIVERSE_PATH)
    symbols = universe_cfg["symbols"]
    panel = get_cached_intraday_panel(symbols, WARMUP_START, END)
    out = {}
    for sym in symbols:
        if sym in panel.index.get_level_values("code"):
            out[sym] = panel.xs(sym, level="code").sort_index()
    return out


def run_one(bars_by_symbol, sweep_min_atr, stop_atr_mult, cfg, start_ts):
    params = {"sweep_min_atr": sweep_min_atr, "reclaim_bars": RECLAIM_BARS, "stop_atr_mult": stop_atr_mult}
    report = run_intraday_backtest(bars_by_symbol, "sweep_reclaim", params, cfg)
    trades = [t for t in report.trades if t.exit_time >= start_ts]
    n = len(trades)
    if n == 0:
        return {"sweep_min_atr": sweep_min_atr, "stop_atr_mult": stop_atr_mult, "n_trades": 0,
                "win_rate": None, "gross_pf": None, "total_net_pnl": 0.0}
    gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_loss = sum(-t.gross_pnl for t in trades if t.gross_pnl < 0)
    gross_pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_net = sum(t.net_pnl for t in trades)
    n_sessions = len({t.entry_time.date() for t in trades}) or 1
    return {
        "sweep_min_atr": sweep_min_atr, "stop_atr_mult": stop_atr_mult, "n_trades": n,
        "trades_per_session_total": round(n / n_sessions, 1),
        "win_rate": round(wins / n, 3), "gross_pf": round(gross_pf, 3),
        "total_net_pnl": round(total_net, 0),
    }


def main():
    print(f"Loading cached 1m bars for {UNIVERSE_PATH} symbols, {WARMUP_START}..{END} ...")
    bars_by_symbol = load_bars()
    capital = 1_000_000.0
    zero_cost_cfg = IntradayBacktestConfig(
        capital=capital, half_spread_bps=0.0, impact_bps_per_participation=0.0,
        commission_per_share=0.0, min_commission=0.0,
    )
    start_ts = pd.Timestamp(START)

    print(f"\nGrid: sweep_min_atr in {SWEEP_MIN_ATR_GRID} x stop_atr_mult in {STOP_ATR_MULT_GRID} "
          f"(reclaim_bars fixed at {RECLAIM_BARS}), ZERO-COST only, {len(SWEEP_MIN_ATR_GRID)*len(STOP_ATR_MULT_GRID)} combos total.\n")
    print(f"{'sweep_min_atr':>14} {'stop_atr_mult':>14} {'n_trades':>9} {'trades/sess':>12} {'win_rate':>9} {'gross_PF':>9} {'total_net_pnl':>15}")

    results = []
    t0 = time.time()
    for sweep_min_atr in SWEEP_MIN_ATR_GRID:
        for stop_atr_mult in STOP_ATR_MULT_GRID:
            r = run_one(bars_by_symbol, sweep_min_atr, stop_atr_mult, zero_cost_cfg, start_ts)
            results.append(r)
            elapsed = time.time() - t0
            print(f"{r['sweep_min_atr']:>14} {r['stop_atr_mult']:>14} {r['n_trades']:>9} "
                  f"{r.get('trades_per_session_total', 0):>12} "
                  f"{('%.1f%%' % (100*r['win_rate']) if r['win_rate'] is not None else 'n/a'):>9} "
                  f"{(r['gross_pf'] if r['gross_pf'] is not None else float('nan')):>9} "
                  f"{r['total_net_pnl']:>15,.0f}   (elapsed {elapsed:.0f}s)", flush=True)

    best = max(results, key=lambda r: (r["gross_pf"] if r["gross_pf"] not in (None, float("inf")) else -1))
    print(f"\nBest gross_PF combo: {best}")
    n_above_1 = sum(1 for r in results if r["gross_pf"] and r["gross_pf"] > 1.0)
    print(f"Combos with gross_PF > 1.0 (i.e. would even be WORTH testing real-cost survival): {n_above_1} / {len(results)}")


if __name__ == "__main__":
    main()
