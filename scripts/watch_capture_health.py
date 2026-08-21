"""Continuously verify the live Futu tick/L2 capture is actually producing data.

Unlike a plain `ps`/PID check (which can't tell a healthy capture from a
process that is alive but stuck in a crash-reconnect loop or has silently
stopped writing), this polls the newest tick + depth JSONL file per symbol
and confirms their byte size is *growing* while the market is in regular
trading hours. Prints one line per poll; lines starting with "ALERT" are
meant to be grepped/watched by the calling agent or a monitoring tool.

Usage:
    python scripts/watch_capture_health.py [--interval 300] [--stall-min 20]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time as time_mod
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.calendar import is_regular_trading_hours  # noqa: E402

WATCH_SYMBOLS = ["AAPL", "NVDA", "STX"]
PROC_PATTERN = "capture_market_microstructure.py --source futu"


def _proc_alive() -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", PROC_PATTERN], text=True)
        pids = [int(p) for p in out.split()]
        return pids[0] if pids else None
    except subprocess.CalledProcessError:
        return None


def _latest_file_size(root: str, symbol: str) -> tuple[str | None, int]:
    files = sorted(glob.glob(os.path.join(root, symbol, "*.jsonl")))
    if not files:
        return None, 0
    path = files[-1]
    return path, os.path.getsize(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls")
    ap.add_argument("--stall-min", type=int, default=20, help="minutes of no growth during RTH before ALERT")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tick_root = os.path.join(base, "data", "ticks")
    depth_root = os.path.join(base, "data", "depth")

    last_sizes: dict[str, int] = {}
    last_paths: dict[str, str | None] = {}
    last_growth_ts: dict[str, float] = {}

    print(f"watch_capture_health: polling every {args.interval}s, stall threshold {args.stall_min}min", flush=True)

    while True:
        now = datetime.now(timezone.utc)
        rth = is_regular_trading_hours(now)
        pid = _proc_alive()
        now_mono = time_mod.monotonic()

        status_bits = [f"pid={pid if pid else 'DEAD'}", f"rth={rth}"]
        alerts = []

        if pid is None:
            alerts.append("ALERT: capture process is not running")

        for symbol in WATCH_SYMBOLS:
            for label, root in (("tick", tick_root), ("depth", depth_root)):
                key = f"{label}:{symbol}"
                path, size = _latest_file_size(root, symbol)
                prev_size = last_sizes.get(key)
                prev_path = last_paths.get(key)
                # A new "latest" file (date rollover) is itself evidence of
                # active writing, even though its size (starting near 0) is
                # smaller than the previous day's fully-written file — don't
                # let that look like a stall.
                path_changed = key in last_paths and path != prev_path
                grew = path_changed or (prev_size is not None and size > prev_size)
                if grew or key not in last_growth_ts:
                    last_growth_ts[key] = now_mono
                last_sizes[key] = size
                last_paths[key] = path
                stall_min = (now_mono - last_growth_ts[key]) / 60.0
                status_bits.append(f"{key}={size}B(+{stall_min:.0f}m)")
                if rth and stall_min > args.stall_min:
                    alerts.append(
                        f"ALERT: {key} has not grown in {stall_min:.0f}min during regular trading hours "
                        f"(file={path})"
                    )

        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{ts} " + " ".join(status_bits), flush=True)
        for a in alerts:
            print(f"{ts} {a}", flush=True)

        time_mod.sleep(args.interval)


if __name__ == "__main__":
    main()
