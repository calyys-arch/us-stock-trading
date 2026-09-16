"""
One-off fast diagnostic (NOT a WFO run) — answers "does sweep_reclaim even
have gross (pre-cost) edge on the 5-name low-liquidity universe" in minutes
instead of the hours a full WFO+MC+stress pipeline takes, by running the
CURRENT default sweep_reclaim params ONCE over the whole window at both
zero cost and real cost, instead of grid-searching + walk-forwarding.

This is the fast version of the diagnostic promised in chat: distinguishes
"signal never fires / too few trades to judge" vs "fires plenty but gross
edge is flat/negative" (signal design problem) vs "gross edge positive but
costs kill it" (cost model / this liquidity tier's spread problem).

Usage: python scripts/_lowliq_diag_sweep_reclaim.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from python.backtest.intraday_engine import IntradayBacktestConfig, run_intraday_backtest
from python.data.fixed_universe import load_universe_config
from python.data.intraday_cache import get_cached_intraday_panel

UNIVERSE_PATH = "configs/alt_universe_lowliq.yaml"
START, END = "2025-08-01", "2026-07-01"
WARMUP_START = "2025-07-31"


def load_bars():
    universe_cfg = load_universe_config(path=UNIVERSE_PATH)
    symbols = universe_cfg["symbols"]
    panel = get_cached_intraday_panel(symbols, WARMUP_START, END)
    out = {}
    for sym in symbols:
        if sym in panel.index.get_level_values("code"):
            out[sym] = panel.xs(sym, level="code").sort_index()
    return out


def summarize(label, report, capital):
    trades = report.trades
    n = len(trades)
    gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_loss = sum(-t.gross_pnl for t in trades if t.gross_pnl < 0)
    net_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    net_loss = sum(-t.net_pnl for t in trades if t.net_pnl < 0)
    gross_pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    net_pf = (net_profit / net_loss) if net_loss > 0 else (float("inf") if net_profit > 0 else 0.0)
    total_gross = sum(t.gross_pnl for t in trades)
    total_net = sum(t.net_pnl for t in trades)
    total_costs = sum(t.costs for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    win_rate = wins / n if n else 0.0
    by_symbol = Counter(t.symbol for t in trades)

    print(f"\n--- {label} ---")
    print(f"signals_emitted={report.signals_emitted} signals_filled={report.signals_filled} n_trades={n}")
    if n == 0:
        print("NO TRADES AT ALL in this window — cannot compute PF/win-rate.")
        return
    print(f"win_rate={win_rate:.1%}  gross_PF={gross_pf:.3f}  net_PF={net_pf:.3f}")
    print(f"total_gross_pnl=${total_gross:,.0f}  total_costs=${total_costs:,.0f}  total_net_pnl=${total_net:,.0f}")
    print(f"total_net_pnl as % of ${capital:,.0f} capital: {100*total_net/capital:.3f}%")
    print(f"trades per symbol: {dict(by_symbol)}")


def main():
    print(f"Loading cached 1m bars for {UNIVERSE_PATH} symbols, {WARMUP_START}..{END} ...")
    bars_by_symbol = load_bars()
    for sym, df in bars_by_symbol.items():
        print(f"  {sym}: {len(df)} bars, {df.index.normalize().nunique()} sessions")

    strategy_cfg = yaml.safe_load(open("configs/strategy.yaml", encoding="utf-8"))["sweep_reclaim"]
    params = {k: v for k, v in strategy_cfg.items() if k not in ("enabled", "auto_execute")}
    print(f"\nsweep_reclaim params (current configs/strategy.yaml defaults): {params}")

    # Only feed bars strictly inside [START, END) into the report, but keep
    # the 1-day warmup slice in bars_by_symbol for run_intraday_backtest's
    # YDH/YDL prior-day lookups (same contract as build_intraday_backtest_fn).
    capital = 1_000_000.0

    zero_cost_cfg = IntradayBacktestConfig(
        capital=capital, half_spread_bps=0.0, impact_bps_per_participation=0.0,
        commission_per_share=0.0, min_commission=0.0,
    )
    real_cost_cfg = IntradayBacktestConfig(capital=capital)  # engine defaults: 2.0bps half-spread, 20bps impact, etc.

    start_ts = pd.Timestamp(START)

    print("\nRunning ZERO-COST pass (isolates gross/pre-cost directional edge)...")
    report_zero = run_intraday_backtest(bars_by_symbol, "sweep_reclaim", params, zero_cost_cfg)
    report_zero.trades = [t for t in report_zero.trades if t.exit_time >= start_ts]
    summarize("ZERO-COST (gross edge only)", report_zero, capital)

    print("\nRunning REAL-COST pass (engine defaults: 2.0bps half-spread flat, 20bps impact)...")
    report_real = run_intraday_backtest(bars_by_symbol, "sweep_reclaim", params, real_cost_cfg)
    report_real.trades = [t for t in report_real.trades if t.exit_time >= start_ts]
    summarize("REAL-COST (what WFO actually sees)", report_real, capital)


if __name__ == "__main__":
    main()
