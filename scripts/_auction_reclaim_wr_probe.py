"""One-shot WR probe for auction_reclaim after GEX/footprint wiring.

Two windows, both executed (not estimated):

1) Footprint overlap: rebuild 1-minute RTH bars from captured ticks
   (2026-08-04 .. 2026-08-17) and walk only sessions whose TRUE prior
   session is available. Compares bar-only vs footprint-filtered on the
   same days / same params.

2) Historical bar-only: cached 1-minute panel 2025-08-01 .. 2026-07-01
   (no ticks, no GEX) — the same window as the official WFO, one param
   set, so we can read win rate rather than only fold Sharpe.

Usage:
    python scripts/_auction_reclaim_wr_probe.py
    python scripts/_auction_reclaim_wr_probe.py --skip-historical
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from python.backtest.intraday_engine import IntradayBacktestConfig, run_intraday_backtest, run_symbol_day
from python.data.fixed_universe import load_universe_config
from python.data.intraday_cache import get_cached_intraday_panel
from python.data.tick_cache import load_trade_ticks

TICK_DAYS = [
    pd.Timestamp("2026-08-04"),
    pd.Timestamp("2026-08-05"),
    pd.Timestamp("2026-08-06"),
    pd.Timestamp("2026-08-12"),
    pd.Timestamp("2026-08-13"),
    pd.Timestamp("2026-08-14"),
    pd.Timestamp("2026-08-17"),
]
# True previous RTH session. Missing prior → skip (do not use a stale day).
TRUE_PRIOR = {
    pd.Timestamp("2026-08-04"): pd.Timestamp("2026-07-31"),
    pd.Timestamp("2026-08-05"): pd.Timestamp("2026-08-04"),
    pd.Timestamp("2026-08-06"): pd.Timestamp("2026-08-05"),
    pd.Timestamp("2026-08-13"): pd.Timestamp("2026-08-12"),
    pd.Timestamp("2026-08-14"): pd.Timestamp("2026-08-13"),
    pd.Timestamp("2026-08-17"): pd.Timestamp("2026-08-14"),
}

PARAM_SETS = {
    "yaml_defaults": {
        "min_rel_volume": 1.2, "min_wick_frac": 0.45,
        "stop_atr_mult": 0.15, "target_r_multiple": 1.5,
    },
    "wfo_last_fold": {
        "min_rel_volume": 1.4, "min_wick_frac": 0.40,
        "stop_atr_mult": 0.10, "target_r_multiple": 1.5,
    },
}


def _rth_1m_from_ticks(ticks: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if ticks is None or ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    t = ticks.set_index("time").sort_index()
    start = day + pd.Timedelta(hours=9, minutes=30)
    end = day + pd.Timedelta(hours=15, minutes=59)
    t = t.loc[(t.index >= start) & (t.index <= end + pd.Timedelta(minutes=1))]
    if t.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ohlc = t["price"].resample("1min").ohlc()
    vol = t["size"].resample("1min").sum()
    bars = ohlc.join(vol.rename("volume"))
    full = pd.date_range(start, end, freq="1min")
    bars = bars.reindex(full)
    bars["close"] = bars["close"].ffill()
    bars["open"] = bars["open"].fillna(bars["close"])
    bars["high"] = bars["high"].fillna(bars["close"])
    bars["low"] = bars["low"].fillna(bars["close"])
    bars["volume"] = bars["volume"].fillna(0.0)
    return bars.dropna(subset=["open", "close"])


def _summarize(trades) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins_net": 0, "wr_net": None, "wins_gross": 0, "wr_gross": None,
                "net_pnl": 0.0, "gross_pnl": 0.0, "pf_net": None}
    wins_net = sum(1 for t in trades if t.net_pnl > 0)
    wins_gross = sum(1 for t in trades if t.gross_pnl > 0)
    gp = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gl = sum(-t.net_pnl for t in trades if t.net_pnl < 0)
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return {
        "n": n,
        "wins_net": wins_net,
        "wr_net": wins_net / n,
        "wins_gross": wins_gross,
        "wr_gross": wins_gross / n,
        "net_pnl": float(sum(t.net_pnl for t in trades)),
        "gross_pnl": float(sum(t.gross_pnl for t in trades)),
        "pf_net": pf,
    }


def _fmt(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    wr = s["wr_net"]
    wr_g = s["wr_gross"]
    pf = s["pf_net"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']}  WR_net={wr:.1%} ({s['wins_net']}/{s['n']})  "
            f"WR_gross={wr_g:.1%}  PF_net={pf_s}  "
            f"net=${s['net_pnl']:,.0f}")


def probe_footprint(symbols: list[str]) -> None:
    print("\n=== 1) Footprint overlap (tick-rebuilt 1m, true prior only) ===")
    print("tick days:", ", ".join(d.strftime("%Y-%m-%d") for d in TICK_DAYS))
    print("sessions with true prior:", ", ".join(d.strftime("%Y-%m-%d") for d in TRUE_PRIOR))

    july31_panel = get_cached_intraday_panel(symbols, "2026-07-31", "2026-07-31 16:00")
    prior_bars: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    for sym in symbols:
        if ("code" in july31_panel.index.names
                and sym in set(july31_panel.index.get_level_values("code"))):
            prior_bars[(sym, pd.Timestamp("2026-07-31"))] = (
                july31_panel.xs(sym, level="code").sort_index()
            )

    cfg = IntradayBacktestConfig()
    empty_ticks = pd.DataFrame(columns=["time", "price", "size"])
    session_cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]] = {}

    for day, prior_day in TRUE_PRIOR.items():
        for sym in symbols:
            prior = prior_bars.get((sym, prior_day))
            if prior is None or prior.empty:
                continue
            ticks = load_trade_ticks(sym, day)
            day_bars = _rth_1m_from_ticks(ticks, day)
            if day_bars.empty or len(day_bars) < 30:
                continue
            session_cache[(sym, day)] = (day_bars, ticks if ticks is not None else empty_ticks)
            prior_bars[(sym, day)] = day_bars
            print(f"  loaded {sym} {day.date()} ticks={0 if ticks is None else len(ticks)} bars={len(day_bars)}",
                  flush=True)

    print(f"usable sessions: {len(session_cache)}")

    for label, params in PARAM_SETS.items():
        bar_trades = []
        fp_trades = []
        for (sym, day), (day_bars, ticks) in session_cache.items():
            prior_day = TRUE_PRIOR[day]
            prior = prior_bars[(sym, prior_day)]
            prior_close = float(prior["close"].iloc[-1])
            t_bar, _, _ = run_symbol_day(
                sym, day_bars, prior, "auction_reclaim", params, cfg,
                prior_close=prior_close, day_ticks=empty_ticks,
            )
            t_fp, _, _ = run_symbol_day(
                sym, day_bars, prior, "auction_reclaim", params, cfg,
                prior_close=prior_close, day_ticks=ticks,
            )
            bar_trades.extend(t_bar)
            fp_trades.extend(t_fp)
        print(f"\n[{label}] sessions={len(session_cache)}")
        print(f"  bar-only : {_fmt(_summarize(bar_trades))}")
        print(f"  footprint: {_fmt(_summarize(fp_trades))}")
        if fp_trades:
            print("  footprint trades:")
            for t in fp_trades:
                mark = "WIN" if t.net_pnl > 0 else "LOSS"
                print(f"    {mark} {t.symbol} {t.direction} {t.entry_time} "
                      f"net=${t.net_pnl:,.0f} {t.exit_reason}")


def probe_historical(symbols: list[str]) -> None:
    print("\n=== 2) Historical bar-only 2025-08-01 .. 2026-07-01 ===")
    panel = get_cached_intraday_panel(symbols, "2025-08-01", "2026-07-01")
    bars_by_symbol = {}
    for sym in symbols:
        if sym in set(panel.index.get_level_values("code")):
            bars_by_symbol[sym] = panel.xs(sym, level="code").sort_index()
    cfg = IntradayBacktestConfig()
    for label, params in PARAM_SETS.items():
        report = run_intraday_backtest(bars_by_symbol, "auction_reclaim", params, cfg)
        s = _summarize(report.trades)
        print(f"[{label}] {_fmt(s)}  emitted={report.signals_emitted} filled={report.signals_filled}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-historical", action="store_true")
    parser.add_argument("--skip-footprint", action="store_true")
    args = parser.parse_args()

    with open("configs/strategy.yaml", encoding="utf-8") as f:
        yaml_ar = (yaml.safe_load(f) or {}).get("auction_reclaim", {})
    print("yaml auction_reclaim auto_execute=", yaml_ar.get("auto_execute"))
    symbols = load_universe_config()["symbols"]
    print("universe", len(symbols), "symbols")

    if not args.skip_footprint:
        probe_footprint(symbols)
    if not args.skip_historical:
        probe_historical(symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
