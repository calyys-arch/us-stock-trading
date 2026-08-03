"""
Long-running tick-by-tick + Level 2 depth archiver (report-only diagnostic
layer). Run this alongside IB Gateway during market hours to build the
local microstructure archive (trades/bidask -> data/ticks/<SYMBOL>/*.jsonl,
L2 depth -> data/depth/<SYMBOL>/*.jsonl) that python/signals/trap_detector.py's
tick/order-book heuristics analyze offline.

Ticks not captured are gone forever — the archive only covers days this
script actually ran, and the trap report marks order-book evidence for
other days as UNAVAILABLE rather than pretending to know.

Usage:
    python scripts/capture_market_microstructure.py                     # fixed universe
    python scripts/capture_market_microstructure.py --symbols AAPL,TSLA
    python scripts/capture_market_microstructure.py --max-depth-symbols 3 --include-extended-hours

Stop with Ctrl-C.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

from python.interfaces.ibkr_tick_capture import IbkrTickCapture


def _default_symbols() -> list[str]:
    try:
        from python.data.fixed_universe import load_universe_config

        return load_universe_config()["symbols"]
    except Exception as exc:
        print(f"NOTE: fixed universe unavailable ({exc}) — defaulting to XLE,XOP")
        return ["XLE", "XOP"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="",
                        help="comma-separated tickers (default: fixed universe from configs/universe.yaml)")
    parser.add_argument("--max-depth-symbols", type=int, default=3,
                        help="L2 depth subscriptions are capped by IB (typically 3 concurrent); "
                             "depth is requested for the FIRST N symbols only")
    parser.add_argument("--include-extended-hours", action="store_true",
                        help="also record pre/post-market events (default: RTH only)")
    parser.add_argument("--broker-config", default="configs/broker.yaml")
    args = parser.parse_args()

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else _default_symbols())

    with open(args.broker_config, encoding="utf-8") as f:
        broker = yaml.safe_load(f) or {}
    ibkr = broker.get("ibkr", {}) or {}

    capture = IbkrTickCapture(
        symbols=symbols,
        host=ibkr.get("host", "127.0.0.1"),
        port=int(ibkr.get("feed_port", 4002)),
        client_id=int(ibkr.get("tick_capture_client_id", 41)),
        max_depth_symbols=args.max_depth_symbols,
        rth_only=not args.include_extended_hours,
    )
    print(f"Capturing {len(symbols)} symbols -> data/ticks/ + data/depth/ (Ctrl-C to stop)")
    try:
        capture.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        capture.stop()


if __name__ == "__main__":
    main()
