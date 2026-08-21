"""
Capture today's naive options GEX snapshot for QQQ (Creamer's market
regime) plus optional equity symbols.

Writes data/gex/<SYMBOL>/<YYYYMMDD>.json. There is no historical backfill
— Yahoo's option_chain is the live surface. Days without a file are
treated as GEX-unavailable by the signal (bar-proxy vol_regime).

Usage:
    python scripts/snapshot_gex.py                  # QQQ + fixed universe
    python scripts/snapshot_gex.py --symbols QQQ,AAPL
    python scripts/snapshot_gex.py --market-only    # QQQ only
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
log = logging.getLogger("snapshot_gex")

from python.data.gex_cache import fetch_yfinance_gex, save_gex_snapshot


def _default_symbols() -> list[str]:
    try:
        from python.data.fixed_universe import load_universe_config

        return ["QQQ"] + list(load_universe_config()["symbols"])
    except Exception as exc:
        log.warning("fixed universe unavailable (%s) — QQQ only", exc)
        return ["QQQ"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot naive options GEX (yfinance chain).")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Default: QQQ + fixed universe.")
    parser.add_argument("--market-only", action="store_true", help="QQQ only.")
    parser.add_argument("--dte-max", type=int, default=45, help="Max days to expiry (structural default 45).")
    args = parser.parse_args()

    if args.market_only:
        symbols = ["QQQ"]
    elif args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = [s.upper() for s in _default_symbols()]

    as_of = date.today()
    ok = 0
    for symbol in symbols:
        log.info("fetching GEX for %s", symbol)
        snap = fetch_yfinance_gex(symbol, as_of=as_of, dte_max=args.dte_max)
        if snap is None:
            log.warning("no GEX snapshot for %s (empty chain or no spot)", symbol)
            continue
        path = save_gex_snapshot(snap)
        ok += 1
        log.info(
            "%s regime=%s net_gex=%.3g call_wall=%s put_wall=%s flip=%s -> %s",
            symbol, snap.regime, snap.net_gex, snap.call_wall, snap.put_wall,
            snap.gamma_flip, path,
        )
    log.info("wrote %d/%d snapshots", ok, len(symbols))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
