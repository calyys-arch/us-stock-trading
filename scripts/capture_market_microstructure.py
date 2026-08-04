"""
Long-running tick-by-tick + Level 2 depth archiver (report-only diagnostic
layer). Run this during market hours to build the local microstructure
archive (trades -> data/ticks/<SYMBOL>/*.jsonl, L2 depth ->
data/depth/<SYMBOL>/*.jsonl) that python/signals/trap_detector.py's
tick/order-book heuristics analyze offline.

Two interchangeable sources, picked with --source (default: ibkr):
  ibkr  python/interfaces/ibkr_tick_capture.py  — needs IB Gateway/TWS
        running+logged in, AND a real-time market-data subscription on the
        connected account (a free-standing Demo account cannot get one —
        see configs/broker.yaml's futu: block comment for why this project
        hit that wall on 2026-08-04).
  futu  python/interfaces/futu_tick_capture.py  — needs Futu/Moomoo's OpenD
        gateway app running+logged in locally, with a funded account that
        has LV3 (or better) US-equity quote permission.

Ticks not captured are gone forever — the archive only covers days this
script actually ran, and the trap report marks order-book evidence for
other days as UNAVAILABLE rather than pretending to know.

Usage:
    python scripts/capture_market_microstructure.py                          # fixed universe, IB Gateway
    python scripts/capture_market_microstructure.py --source futu            # fixed universe, Futu OpenD
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


def _default_symbols() -> list[str]:
    try:
        from python.data.fixed_universe import load_universe_config

        return load_universe_config()["symbols"]
    except Exception as exc:
        print(f"NOTE: fixed universe unavailable ({exc}) — defaulting to XLE,XOP")
        return ["XLE", "XOP"]


def _build_ibkr_capture(symbols: list[str], broker: dict, args) -> object:
    from python.interfaces.ibkr_tick_capture import IbkrTickCapture

    ibkr = broker.get("ibkr", {}) or {}
    return IbkrTickCapture(
        symbols=symbols,
        host=ibkr.get("host", "127.0.0.1"),
        port=int(ibkr.get("feed_port", 4002)),
        client_id=int(ibkr.get("tick_capture_client_id", 41)),
        max_depth_symbols=args.max_depth_symbols,
        rth_only=not args.include_extended_hours,
    )


def _build_futu_capture(symbols: list[str], broker: dict, args) -> object:
    from python.interfaces.futu_tick_capture import FutuTickCapture

    futu = broker.get("futu", {}) or {}
    return FutuTickCapture(
        symbols=symbols,
        host=futu.get("host", "127.0.0.1"),
        port=int(futu.get("port", 11111)),
        market_prefix=futu.get("market_prefix", "US"),
        max_depth_symbols=args.max_depth_symbols,
        rth_only=not args.include_extended_hours,
        rsa_key_path=futu.get("rsa_key_path") or None,
    )


_BUILDERS = {"ibkr": _build_ibkr_capture, "futu": _build_futu_capture}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(_BUILDERS), default="ibkr",
                        help="which live gateway to capture from (default: ibkr)")
    parser.add_argument("--symbols", default="",
                        help="comma-separated tickers (default: fixed universe from configs/universe.yaml)")
    parser.add_argument("--max-depth-symbols", type=int, default=3,
                        help="cap on concurrent Level-2 depth subscriptions — IB typically allows ~3 "
                             "concurrent; Futu's quota is far higher, override this higher with --source futu "
                             "(e.g. --max-depth-symbols 20 for the full universe)")
    parser.add_argument("--include-extended-hours", action="store_true",
                        help="also record pre/post-market events (default: RTH only)")
    parser.add_argument("--broker-config", default="configs/broker.yaml")
    args = parser.parse_args()

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else _default_symbols())

    with open(args.broker_config, encoding="utf-8") as f:
        broker = yaml.safe_load(f) or {}

    capture = _BUILDERS[args.source](symbols, broker, args)
    print(f"Capturing {len(symbols)} symbols via {args.source} -> data/ticks/ + data/depth/ (Ctrl-C to stop)")
    try:
        capture.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        capture.stop()


if __name__ == "__main__":
    main()
