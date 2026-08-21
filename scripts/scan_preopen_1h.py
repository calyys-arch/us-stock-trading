"""Observe each universe stock's 1-hour environment before the open.

Builds the 1-hour chart from completed RTH sessions only (never today).
Two independent reads: price structure (HH/HL) and trader momentum
(who won the 1h bars, whether participation is building or fading).
Volume does not confirm price. Report-only — does not emit a trade.

Usage:
    python scripts/scan_preopen_1h.py
    python scripts/scan_preopen_1h.py --date 2026-08-18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from python.data.fixed_universe import load_universe_config
from python.data.intraday_cache import get_cached_intraday_panel, latest_cached_bar_time
from python.microstructure.signals.auction_reclaim import scan_universe_preopen


def _default_asof(symbols: list[str]) -> pd.Timestamp:
    """Day after the newest cached 1-minute bar — the next session we
    could actually observe. Wall-clock today is wrong when August cache
    is empty (n_bars: 0)."""
    latest = None
    for sym in symbols:
        ts = latest_cached_bar_time(sym)
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    if latest is None:
        raise RuntimeError("no cached 1-minute bars — run scripts/backfill_intraday.py first")
    return latest.normalize() + pd.Timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Session about to open (YYYY-MM-DD, default next day after latest cache)")
    args = parser.parse_args()

    symbols = load_universe_config()["symbols"]
    asof = pd.Timestamp(args.date).normalize() if args.date else _default_asof(symbols)
    panel = get_cached_intraday_panel(symbols, asof - pd.Timedelta(days=15), asof + pd.Timedelta(days=1))
    bars_by_symbol = {}
    for sym in symbols:
        if sym in set(panel.index.get_level_values("code")):
            bars_by_symbol[sym] = panel.xs(sym, level="code").sort_index()

    rows = scan_universe_preopen(bars_by_symbol, asof)
    print(f"pre-open 1h  structure vs trader momentum  asof={asof.date()}  n={len(rows)}")
    print(
        f"{'SYM':<6} {'STRUCTURE':<12} {'MOMENTUM':<20} "
        f"{'SIDE':<11} {'PACE':<9} {'PRESS':>6} {'PACE_R':>6} {'LAST_VOL':>12}"
    )
    opposed = 0
    for r in rows:
        opposed_row = (
            (r["structure"] == "value_up" and r["trader_side"] == "selling")
            or (r["structure"] == "value_down" and r["trader_side"] == "buying")
        )
        if opposed_row:
            opposed += 1
        print(
            f"{r['symbol']:<6} {r['structure']:<12} {r['trader_momentum']:<20} "
            f"{r['trader_side']:<11} {r['trader_pace']:<9} "
            f"{r['trader_pressure'] or 0:6.2f} {r['pace_ratio'] or 0:6.2f} "
            f"{r['last_volume'] or 0:12,.0f}"
        )
    counts = {}
    sides = {}
    paces = {}
    for r in rows:
        counts[r["structure"]] = counts.get(r["structure"], 0) + 1
        sides[r["trader_side"]] = sides.get(r["trader_side"], 0) + 1
        paces[r["trader_pace"]] = paces.get(r["trader_pace"], 0) + 1
    print("structure", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("traders  ", " ".join(f"{k}={v}" for k, v in sorted(sides.items())),
          "|", " ".join(f"{k}={v}" for k, v in sorted(paces.items())))
    print(f"structure vs traders opposed={opposed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
