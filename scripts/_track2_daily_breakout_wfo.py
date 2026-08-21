"""
Track 2 (backtests/reports/alt_universe_frequency_exploration.md): WFO +
Monte Carlo + reserved holdout for the new daily-bar breakout signal
(python/core/strategies/daily_range_breakout.py,
python/backtest/daily_breakout_engine.py) on the SAME 20-symbol universe
(`configs/universe.yaml`) `orb_vwap` failed on — deliberately apples-to-
apples on universe, varying only FREQUENCY (1-minute -> daily bars), to
isolate whether frequency alone (not universe) was orb_vwap's problem.

Data: `data/history_altuni_daily20/*.csv` (yfinance daily OHLCV, fetched for
this exploration; see that dir's _meta.json). 17/20 symbols have full
2016-06-01..2026-07-31 history; NBIS (IPO 2024-10), PLTR (IPO 2020-09), SNDK
(spinoff 2025-02) have shorter real histories — included wherever they have
data (more trades = more statistical power), excluded implicitly wherever
they don't (an early WFO fold simply gets fewer symbols' worth of trades,
not zero — a genuine data-availability constraint stated here, not silently
worked around).

Cost model: python/core/fees_equity.round_trip_cost with a PER-SYMBOL
calibrated half-spread (bps), hardcoded below from
backtests/reports/slippage_calibration_report.md table (a), section (a) —
the only real captured-L2-depth-derived spread numbers this codebase has for
this universe. Two honest caveats carried into the report:
  1. That calibration is from 2 trading days in August 2026 (see the
     report's "Sample-size honesty note"), applied here as a constant across
     the ENTIRE 2016-2026 backtest window — i.e. TODAY's spread used as a
     proxy for 10 years of history. Equity spreads have structurally
     COMPRESSED over the last decade (more competition among market makers,
     faster systems), so this likely UNDER-states true historical cost in
     the earlier years of the window — a bias in the "makes the edge look
     better than it would have been" direction. Flagged, not corrected (no
     historical L2 capture exists to do better).
  2. It was calibrated for 1-MINUTE-bar execution context; applied here
     unchanged because a half-spread is paid identically regardless of
     holding period (this trade crosses the spread exactly twice, at entry
     and exit, same as any other round trip) — frequency of trading is what
     differs between this signal and orb_vwap, not the per-crossing cost.

Free params (<=5, Chan ceiling; shared with daily_range_breakout.py's own
docstring): range_days, hold_days, stop_atr_mult, target_r_multiple.

WFO fold config: reuses configs/goal.yaml's DEFAULT wfo block UNCHANGED
(is_days=504, oos_days=126, step_days=126, min_pass_folds_ratio=0.60) —
this signal has no per-signal override in goal.yaml (deliberately not added
one; see the exploration report for why touching goal.yaml was avoided).
min_oos_sharpe_abs=0.5 and max_oos_drawdown=0.25 are also goal.yaml's
defaults, applied as-is, unweakened.

Holdout: the WFO explores 2016-06-01 .. 2025-01-31 only. The last ~18
months, 2025-02-01 .. 2026-07-31, are NEVER touched by parameter search —
reserved and evaluated exactly once, at the end, using the single
configuration most consistently selected as "best IS candidate" across the
WFO folds (mode of best_params), per this task's explicit holdout
discipline.

Usage:
    python scripts/_track2_daily_breakout_wfo.py
Writes backtests/reports/track2_daily_breakout_results.json (checkpointed:
precomputed per-candidate trade lists are cached to
backtests/reports/track2_daily_breakout_trades_cache.json so a re-run does
not redo the (cheap but nonzero) backtest sweep).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from python.backtest.daily_breakout_engine import DailyBreakoutConfig, run_daily_breakout_backtest
from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig

DATA_DIR = Path("data/history_altuni_daily20")
TRADES_CACHE_PATH = Path("backtests/reports/track2_daily_breakout_trades_cache.json")
RESULTS_PATH = Path("backtests/reports/track2_daily_breakout_results.json")

# From backtests/reports/slippage_calibration_report.md table (a) "Calibrated
# median (bps)" column — see this module's docstring for the two caveats.
CALIBRATED_HALF_SPREAD_BPS: dict[str, float] = {
    "AAPL": 0.323, "AMAT": 3.742, "AMD": 1.830, "AVGO": 1.307, "GOOGL": 0.419,
    "INTC": 0.995, "LITE": 5.748, "LRCX": 4.193, "META": 1.289, "MRVL": 3.048,
    "MSFT": 0.707, "MU": 2.063, "NBIS": 4.627, "NVDA": 0.455, "ORCL": 2.076,
    "PLTR": 0.939, "QCOM": 1.889, "SNDK": 3.344, "STX": 6.588, "WDC": 4.426,
}

NOTIONAL_PER_TRADE = 50_000.0
PORTFOLIO_CAPITAL = NOTIONAL_PER_TRADE * len(CALIBRATED_HALF_SPREAD_BPS)  # $1,000,000

PARAM_GRID = [
    {"range_days": rd, "hold_days": hd, "stop_atr_mult": sa, "target_r_multiple": tr}
    for rd, hd, sa, tr in product([10, 20, 40], [5, 10, 20], [1.5, 2.5], [2.0, 3.0])
]

HOLDOUT_START = datetime(2025, 2, 1)
FULL_START = datetime(2016, 6, 1)
WFO_END = datetime(2025, 1, 31)   # WFO never sees data after this
FULL_END = datetime(2026, 7, 31)


def load_bars() -> dict[str, pd.DataFrame]:
    bars = {}
    for symbol in CALIBRATED_HALF_SPREAD_BPS:
        path = DATA_DIR / f"{symbol}.csv"
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.columns = [c.lower() for c in df.columns]
        bars[symbol] = df[["open", "high", "low", "close", "volume"]].sort_index()
    return bars


def _candidate_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def precompute_trades(bars: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    """One full-history backtest per (candidate, symbol); cached to disk.
    Returns {candidate_key: [trade_dict, ...]} pooled across all 20 symbols.
    Trade dicts carry entry_date/exit_date as ISO strings for JSON caching."""
    if TRADES_CACHE_PATH.exists():
        print(f"Loading cached trades from {TRADES_CACHE_PATH} ...")
        return json.loads(TRADES_CACHE_PATH.read_text())

    cache: dict[str, list[dict]] = {}
    for ci, params in enumerate(PARAM_GRID):
        key = _candidate_key(params)
        all_trades: list[dict] = []
        for symbol, df in bars.items():
            cfg = DailyBreakoutConfig(
                **params,
                notional_per_trade=NOTIONAL_PER_TRADE,
                half_spread_bps=CALIBRATED_HALF_SPREAD_BPS[symbol],
            )
            report = run_daily_breakout_backtest(symbol, df, cfg)
            for t in report.trades:
                d = asdict(t)
                d["entry_date"] = t.entry_date.isoformat()
                d["exit_date"] = t.exit_date.isoformat()
                all_trades.append(d)
        cache[key] = all_trades
        print(f"  [{ci + 1}/{len(PARAM_GRID)}] {params} -> {len(all_trades)} trades")

    TRADES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADES_CACHE_PATH.write_text(json.dumps(cache))
    return cache


def _window_metrics(trades: list[dict], start: datetime, end: datetime, capital: float) -> dict:
    """Metrics for trades ENTERED in [start, end) (fold-membership rule; see
    module docstring). P&L is booked on EXIT date (may fall after `end` for
    a multi-day hold entered near the boundary — the trade's economics
    belong to the fold that decided it, not the fold it happens to close
    in)."""
    sel = [t for t in trades
           if start.isoformat() <= t["entry_date"] < end.isoformat()]
    if not sel:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "n_trades": 0,
                "gross_pnl": 0.0, "total_cost": 0.0, "net_pnl": 0.0}

    daily_pnl: dict[str, float] = {}
    for t in sel:
        daily_pnl[t["exit_date"]] = daily_pnl.get(t["exit_date"], 0.0) + t["net_pnl"]

    all_days = pd.date_range(start, max(end, pd.Timestamp(max(daily_pnl))), freq="D")
    returns = pd.Series(0.0, index=all_days)
    for d, pnl in daily_pnl.items():
        ts = pd.Timestamp(d)
        if ts in returns.index:
            returns.loc[ts] = pnl / capital

    sharpe = 0.0
    if returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * (252 ** 0.5))
    equity = (1 + returns).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": abs(max_dd),
        "n_trades": len(sel),
        "gross_pnl": float(sum(t["gross_pnl"] for t in sel)),
        "total_cost": float(sum(t["cost"] for t in sel)),
        "net_pnl": float(sum(t["net_pnl"] for t in sel)),
    }


def main() -> None:
    bars = load_bars()
    for symbol, df in bars.items():
        print(f"{symbol}: {len(df)} bars, {df.index[0].date()} .. {df.index[-1].date()}")

    trades_by_candidate = precompute_trades(bars)

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        key = _candidate_key(params)
        return _window_metrics(trades_by_candidate[key], start, end, PORTFOLIO_CAPITAL)

    wfo_cfg = WFOConfig(is_days=504, oos_days=126, step_days=126,
                         min_pass_folds_ratio=0.60, min_oos_sharpe_abs=0.5)
    optimizer = WalkForwardOptimizer(backtest_fn, wfo_cfg, PARAM_GRID)
    wfo_result = optimizer.run(FULL_START, WFO_END)
    wfo_result.print_summary()

    dd_violations = [f for f in wfo_result.folds if f.oos_metrics.get("max_drawdown", 0.0) > 0.25]
    print(f"Folds violating max_oos_drawdown<=0.25 gate: {len(dd_violations)}/{len(wfo_result.folds)}")

    best_param_counter = Counter(json.dumps(f.best_params, sort_keys=True) for f in wfo_result.folds)
    holdout_params = json.loads(best_param_counter.most_common(1)[0][0]) if best_param_counter else PARAM_GRID[0]
    print(f"\nMost frequently IS-selected params across folds: {holdout_params} "
          f"(selected in {best_param_counter.most_common(1)[0][1]}/{len(wfo_result.folds)} folds)")

    pooled_oos_trades = []
    for f in wfo_result.folds:
        key = _candidate_key(f.best_params)
        pooled_oos_trades += [t for t in trades_by_candidate[key]
                               if f.oos_start <= t["entry_date"] < f.oos_end]
    mc = MonteCarloValidator(n_sims=1000, seed=42)
    mc_result = mc.run([t["net_pnl"] for t in pooled_oos_trades]) if pooled_oos_trades else None
    if mc_result:
        mc_result.print_summary()

    holdout_metrics = backtest_fn(HOLDOUT_START, FULL_END, holdout_params)
    print(f"\n== RESERVED HOLDOUT ({HOLDOUT_START.date()}..{FULL_END.date()}), params={holdout_params} ==")
    for k, v in holdout_metrics.items():
        print(f"  {k}: {v}")
    cost_ratio = (holdout_metrics["gross_pnl"] / holdout_metrics["total_cost"]
                  if holdout_metrics["total_cost"] > 0 else float("inf"))
    print(f"  gross/cost ratio: {cost_ratio:.3f} (gate >= 2.0)")

    # Stress test: 2x calibrated half-spread on the holdout window only,
    # same params — mirrors configs/goal.yaml intraday.stress_slippage_multiplier
    # discipline even though this signal isn't formally classified "intraday".
    stressed_trades: list[dict] = []
    for symbol, df in bars.items():
        cfg = DailyBreakoutConfig(
            **holdout_params, notional_per_trade=NOTIONAL_PER_TRADE,
            half_spread_bps=CALIBRATED_HALF_SPREAD_BPS[symbol] * 2.0,
        )
        window = df.loc[(df.index >= FULL_START) & (df.index < FULL_END)]
        report = run_daily_breakout_backtest(symbol, window, cfg)
        for t in report.trades:
            if HOLDOUT_START.isoformat() <= t.entry_date.isoformat() < FULL_END.isoformat():
                stressed_trades.append({"net_pnl": t.net_pnl, "gross_pnl": t.gross_pnl, "cost": t.cost})
    stress_net = sum(t["net_pnl"] for t in stressed_trades)
    stress_gross = sum(t["gross_pnl"] for t in stressed_trades)
    stress_cost = sum(t["cost"] for t in stressed_trades)
    print(f"\n== HOLDOUT STRESS (2x calibrated half-spread) ==")
    print(f"  n_trades={len(stressed_trades)} net_pnl={stress_net:.2f} "
          f"gross={stress_gross:.2f} cost={stress_cost:.2f}")

    out = {
        "wfo": wfo_result.to_dict(),
        "wfo_drawdown_gate_violations": len(dd_violations),
        "wfo_total_folds": len(wfo_result.folds),
        "holdout_params": holdout_params,
        "holdout_window": [HOLDOUT_START.isoformat(), FULL_END.isoformat()],
        "holdout_metrics": holdout_metrics,
        "holdout_cost_gate_ratio": cost_ratio,
        "holdout_stress_2x_spread": {
            "n_trades": len(stressed_trades), "net_pnl": stress_net,
            "gross_pnl": stress_gross, "cost": stress_cost,
        },
        "monte_carlo_pooled_oos": mc_result.to_dict() if mc_result else None,
        "n_pooled_oos_trades": len(pooled_oos_trades),
        "run_at": datetime.utcnow().isoformat(),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
