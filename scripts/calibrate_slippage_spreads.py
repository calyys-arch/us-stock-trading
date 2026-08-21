"""
Calibrates per-symbol half-spread (in bps) from REAL captured Level-2 depth
data (data/depth/<SYMBOL>/<YYYYMMDD>.jsonl), replacing
python/backtest/intraday_engine.py's flat `half_spread_bps: float = 2.0`
placeholder with real numbers where we have them (docs/
microstructure_pivot_plan.md §4a's honest caveat: "half_spread 用該股近期
平均買賣價差...沒有就用保守常數" — this script computes that "該股近期平均
買賣價差" from data/depth/ instead of guessing).

Schema contract (python/interfaces/futu_tick_capture.py /
ibkr_tick_capture.py — READ, not guessed): each line in a depth JSONL file
is ONE order-book LEVEL event:
    {"recorded_at": ..., "time": <iso8601 UTC>, "position": <int>,
     "market_maker": ..., "operation": 0|1|2 (insert|update|delete),
     "side": 0=ask, 1=bid, "price": float, "size": float, "source": ...}
This is a DIFF stream (Futu's full-snapshot-per-push diffed by book
position — see futu_tick_capture.py's module docstring "APPROXIMATION"
note), not a repeated full snapshot. To reconstruct the best bid/ask at any
instant we only need to track `position == 0` events per side: an
insert/update at position 0 sets that side's current best price; a delete
at position 0 (the side's book emptying out entirely — rare for liquid
names) clears it until the next insert/update refills it.

Sampling method: rows sharing the exact same `time` value were written by
the SAME push callback invocation (one atomic order-book update, possibly
touching both sides) — see futu_tick_capture.py's `_handle_order_book`,
which writes bid-side rows then ask-side rows per push, all stamped with
one `now_iso`. We apply every row in a `time`-group to the running best-bid/
best-ask state, then take exactly ONE spread sample after the group is
fully applied (not one sample per row), avoiding the "one push updates
both sides -> counted twice, transiently mixing old/new" distortion a
per-row sampling approach would have. Samples where the book is crossed/
locked (ask <= bid) are dropped and counted separately (thin/transient
book states, not a real tradeable spread) rather than silently biasing the
median negative.

Time-of-day bucketing: open (09:30-10:00 ET), close (15:30-16:00 ET),
midday (everything else during RTH) — reported qualitatively per
docs/microstructure_pivot_plan.md's expectation that spreads are wider at
the open/close, not folded into a time-varying cost model (out of scope,
see task instructions).

Usage:
    python scripts/calibrate_slippage_spreads.py
    python scripts/calibrate_slippage_spreads.py --symbols AAPL,MSFT
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

DEPTH_DIR = Path("data/depth")
OUT_JSON = Path("backtests/reports/calibrated_spreads.json")
_ET = ZoneInfo("America/New_York")
_OPEN_END = dtime(10, 0)
_CLOSE_START = dtime(15, 30)

# Sanity thresholds (task instructions): flag rather than silently trust.
_MEGACAP_SUSPECT_BPS = 50.0  # a mega-cap median this wide almost certainly signals a parsing bug
MEGACAPS = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL"}


@dataclass
class SymbolSpreadStats:
    symbol: str
    n_days: int
    days: list[str]
    n_samples: int
    n_crossed_dropped: int
    median_bps: float
    mean_bps: float
    p25_bps: float
    p75_bps: float
    open_median_bps: float | None
    midday_median_bps: float | None
    close_median_bps: float | None
    suspect: bool
    suspect_reason: str | None = None


def _bucket(et_dt: datetime) -> str:
    t = et_dt.time()
    if t < _OPEN_END:
        return "open"
    if t >= _CLOSE_START:
        return "close"
    return "midday"


def _iter_symbol_files(symbol: str) -> list[Path]:
    d = DEPTH_DIR / symbol
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jsonl"))


def compute_symbol_spread(symbol: str) -> SymbolSpreadStats | None:
    files = _iter_symbol_files(symbol)
    if not files:
        return None

    samples_bps: list[float] = []
    bucket_samples: dict[str, list[float]] = {"open": [], "midday": [], "close": []}
    n_crossed = 0
    days: list[str] = []

    for path in files:
        days.append(path.stem)
        best_bid: float | None = None
        best_ask: float | None = None
        group_time: str | None = None

        def flush_sample() -> None:
            nonlocal n_crossed
            if best_bid is None or best_ask is None:
                return
            if best_ask <= best_bid:
                n_crossed += 1
                return
            mid = (best_ask + best_bid) / 2.0
            if mid <= 0:
                return
            half_spread_bps = 10_000.0 * (best_ask - best_bid) / 2.0 / mid
            samples_bps.append(half_spread_bps)
            if group_time is not None:
                try:
                    et_dt = datetime.fromisoformat(group_time).astimezone(_ET)
                    bucket_samples[_bucket(et_dt)].append(half_spread_bps)
                except ValueError:
                    pass

        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row_time = row.get("time") or row.get("recorded_at")
                if group_time is not None and row_time != group_time:
                    flush_sample()
                group_time = row_time

                if row.get("position") != 0:
                    continue
                side = row.get("side")
                op = row.get("operation")
                price = float(row.get("price", 0.0) or 0.0)
                if op == 2:  # delete: this side's book emptied out at the top
                    if side == 1:
                        best_bid = None
                    elif side == 0:
                        best_ask = None
                    continue
                if side == 1:
                    best_bid = price
                elif side == 0:
                    best_ask = price
        flush_sample()  # last group in the file

    if not samples_bps:
        return SymbolSpreadStats(
            symbol=symbol, n_days=len(days), days=days, n_samples=0, n_crossed_dropped=n_crossed,
            median_bps=float("nan"), mean_bps=float("nan"), p25_bps=float("nan"), p75_bps=float("nan"),
            open_median_bps=None, midday_median_bps=None, close_median_bps=None,
            suspect=True, suspect_reason="zero valid (non-crossed) samples",
        )

    sorted_samples = sorted(samples_bps)
    median_bps = statistics.median(sorted_samples)
    mean_bps = statistics.fmean(sorted_samples)
    p25_bps = sorted_samples[int(0.25 * (len(sorted_samples) - 1))]
    p75_bps = sorted_samples[int(0.75 * (len(sorted_samples) - 1))]

    def bucket_median(name: str) -> float | None:
        vals = bucket_samples[name]
        return statistics.median(vals) if vals else None

    suspect = False
    reason = None
    if symbol in MEGACAPS and median_bps >= _MEGACAP_SUSPECT_BPS:
        suspect = True
        reason = f"mega-cap median {median_bps:.2f}bps >= {_MEGACAP_SUSPECT_BPS}bps sanity ceiling — likely a parsing bug, not reality"
    elif median_bps <= 0:
        suspect = True
        reason = "non-positive median half-spread"
    elif median_bps > 200.0:
        suspect = True
        reason = f"median {median_bps:.2f}bps is implausibly wide for any of this universe's names"

    return SymbolSpreadStats(
        symbol=symbol, n_days=len(days), days=days, n_samples=len(samples_bps),
        n_crossed_dropped=n_crossed, median_bps=median_bps, mean_bps=mean_bps,
        p25_bps=p25_bps, p75_bps=p75_bps,
        open_median_bps=bucket_median("open"), midday_median_bps=bucket_median("midday"),
        close_median_bps=bucket_median("close"), suspect=suspect, suspect_reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=None, help="comma-separated symbol list (default: all under data/depth/)")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = sorted(p.name for p in DEPTH_DIR.iterdir() if p.is_dir()) if DEPTH_DIR.is_dir() else []

    results: dict[str, dict] = {}
    for sym in symbols:
        stats = compute_symbol_spread(sym)
        if stats is None:
            print(f"  {sym}: NO DEPTH DATA on disk — skipped")
            continue
        flag = " <<< SUSPECT: " + stats.suspect_reason if stats.suspect else ""
        print(f"  {sym}: median={stats.median_bps:.3f}bps mean={stats.mean_bps:.3f}bps "
              f"n_samples={stats.n_samples} n_days={stats.n_days} days={stats.days} "
              f"crossed_dropped={stats.n_crossed_dropped} "
              f"open={stats.open_median_bps}, midday={stats.midday_median_bps}, close={stats.close_median_bps}{flag}")
        results[sym] = {
            "median_bps": stats.median_bps, "mean_bps": stats.mean_bps,
            "p25_bps": stats.p25_bps, "p75_bps": stats.p75_bps,
            "n_samples": stats.n_samples, "n_days": stats.n_days, "days": stats.days,
            "n_crossed_dropped": stats.n_crossed_dropped,
            "open_median_bps": stats.open_median_bps, "midday_median_bps": stats.midday_median_bps,
            "close_median_bps": stats.close_median_bps,
            "suspect": stats.suspect, "suspect_reason": stats.suspect_reason,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "flat_baseline_bps": 2.0,
        "symbols": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote calibrated spreads for {len(results)} symbols to {out_path}")


if __name__ == "__main__":
    main()
