"""
Read-only diagnostic for the orb_vwap rescue investigation
(backtests/reports/orb_vwap_rescue_report.md).

Answers three questions the strategy review raised but never measured,
using the SAME engine/cost model the WFO pipeline uses — no new modelling,
no gate changes, nothing written back to any config:

  1. How many times per symbol per session does orb_vwap actually fire and
     get filled? (review §3.3's "~4-5 entries per symbol per day is
     implausible for a once-per-session opening-range event")
  2. What is the entry-ordinal profile of P&L — i.e. is the Nth entry of a
     session systematically worse than the 1st?
  3. What share of trades come from the gap-trap fade rule, and how do
     those trades exit? (the trap rule flips `direction` but NOT the
     `stop = OR low if long else OR high` assignment, which puts the stop
     on the FAVORABLE side of entry for every trap trade — see the report.)

Usage:
    python scripts/_orb_vwap_diagnostics.py --start 2025-08-01 --end 2025-11-01
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd  # noqa: E402

from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    run_symbol_day,
)
from python.microstructure import context as ctx  # noqa: E402
from python.microstructure.signals.orb_vwap import evaluate_orb_vwap  # noqa: E402

OUT_PATH = Path("backtests/reports/_orb_vwap_diagnostics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2025-11-01")
    ap.add_argument("--or-minutes", type=int, default=5)
    args = ap.parse_args()

    from python.data.fixed_universe import load_universe_config
    from python.data.intraday_cache import get_cached_intraday_panel

    symbols = load_universe_config()["symbols"]
    panel = get_cached_intraday_panel(symbols, args.start, args.end)

    cfg = IntradayBacktestConfig()
    params = {"or_minutes": args.or_minutes, "vwap_side_filter": True}

    per_session_fills: list[int] = []
    per_session_signals: list[int] = []
    ordinal_pnl: dict[int, list[float]] = defaultdict(list)
    exit_reason_pnl: dict[str, list[float]] = defaultdict(list)
    profitable_stop_pnl: list[float] = []
    losing_stop_pnl: list[float] = []
    inverted_stop_count = 0
    trap_signal_count = 0
    total_signals = 0

    for sym in symbols:
        if sym not in panel.index.get_level_values("code"):
            continue
        bars = panel.xs(sym, level="code").sort_index()
        prior_day_bars = None
        prior_close = None
        for d in sorted(set(bars.index.normalize())):
            day_bars = bars.loc[bars.index.normalize() == d]
            if len(day_bars) < 5:
                if not day_bars.empty:
                    prior_day_bars, prior_close = day_bars, float(day_bars["close"].iloc[-1])
                continue

            trades, emitted, filled = run_symbol_day(
                sym, day_bars, prior_day_bars, "orb_vwap", params, cfg, prior_close=prior_close,
            )
            per_session_fills.append(filled)
            per_session_signals.append(emitted)
            for i, t in enumerate(trades, start=1):
                ordinal_pnl[min(i, 6)].append(t.net_pnl)
                exit_reason_pnl[t.exit_reason].append(t.net_pnl)
                if t.exit_reason == "stop":
                    # A "stop" that MADE money can only happen when the stop
                    # price sat on the favorable side of the entry — the
                    # signature of the gap-trap rule's inverted stop.
                    (profitable_stop_pnl if t.net_pnl > 0 else losing_stop_pnl).append(t.net_pnl)

            # Independent (no fill simulation) scan of every bar's raw
            # signal so trap_flag / stop-side stats cover EVERY qualifying
            # break, not only the ones the state machine happened to be
            # free to take.
            flatten_cutoff = day_bars.index[-1] - pd.Timedelta(minutes=cfg.flatten_buffer_minutes)
            tb = day_bars.loc[day_bars.index <= flatten_cutoff]
            if len(tb) >= 2:
                vwap = ctx.session_vwap(tb)
                orange = ctx.opening_range(tb, minutes=args.or_minutes)
                for i in range(1, len(tb)):
                    sig = evaluate_orb_vwap(
                        tb.iloc[: i + 1], orange, vwap.iloc[: i + 1], symbol=sym,
                        prior_close=prior_close, **params,
                    )
                    if sig is None:
                        continue
                    total_signals += 1
                    if sig.context.get("trap_flag"):
                        trap_signal_count += 1
                    adverse_ok = (
                        sig.stop_price < sig.entry_price if sig.direction == "long"
                        else sig.stop_price > sig.entry_price
                    )
                    if not adverse_ok:
                        inverted_stop_count += 1

            prior_day_bars, prior_close = day_bars, float(day_bars["close"].iloc[-1])

    n_sessions = len(per_session_fills)
    out = {
        "window": f"{args.start} .. {args.end}",
        "or_minutes": args.or_minutes,
        "symbol_sessions": n_sessions,
        "total_fills": int(sum(per_session_fills)),
        "total_signals_emitted_by_state_machine": int(sum(per_session_signals)),
        "total_raw_signals_every_bar": total_signals,
        "mean_fills_per_symbol_session": (sum(per_session_fills) / n_sessions) if n_sessions else 0.0,
        "fills_per_session_histogram": dict(sorted(Counter(per_session_fills).items())),
        "raw_signals_trap_flagged": trap_signal_count,
        "raw_signals_with_stop_on_favorable_side": inverted_stop_count,
        "raw_signals_with_stop_on_favorable_side_pct": (
            100.0 * inverted_stop_count / total_signals if total_signals else 0.0
        ),
        "pnl_by_entry_ordinal": {
            str(k): {"n": len(v), "total": float(sum(v)), "mean": float(sum(v) / len(v)) if v else 0.0}
            for k, v in sorted(ordinal_pnl.items())
        },
        "pnl_by_exit_reason": {
            k: {"n": len(v), "total": float(sum(v)), "mean": float(sum(v) / len(v)) if v else 0.0}
            for k, v in sorted(exit_reason_pnl.items())
        },
        "stop_exits_that_made_money": {"n": len(profitable_stop_pnl),
                                        "total_pnl": float(sum(profitable_stop_pnl))},
        "stop_exits_that_lost_money": {"n": len(losing_stop_pnl),
                                        "total_pnl": float(sum(losing_stop_pnl))},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
