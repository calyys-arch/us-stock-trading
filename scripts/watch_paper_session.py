"""Unattended paper-session watchdog: keep capture + dashboard alive and armed.

Does not loosen gates. If the dashboard process dies, it is restarted and
re-armed via the same Start (paper) + Start Auto Trading HTTP calls a human
would click. Capture is restarted only when the process is gone.

Usage:
    python scripts/watch_paper_session.py [--interval 120]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.calendar import is_regular_trading_hours  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH_URL = "http://127.0.0.1:8082"
CAPTURE_CMD = [
    sys.executable,
    os.path.join(ROOT, "scripts", "capture_market_microstructure.py"),
    "--source",
    "futu",
    "--max-depth-symbols",
    "20",
]
DASH_CMD = [
    sys.executable,
    os.path.join(ROOT, "scripts", "start_dashboard.py"),
    "--host",
    "127.0.0.1",
    "--port",
    "8082",
]
CAPTURE_PATTERN = "capture_market_microstructure.py --source futu"
DASH_PATTERN = "scripts/start_dashboard.py --host 127.0.0.1 --port 8082"


def _pgrep(pattern: str) -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
        pids = [int(p) for p in out.split() if p.strip()]
        return pids[0] if pids else None
    except subprocess.CalledProcessError:
        return None


def _get_json(path: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{DASH_URL}{path}", timeout=5) as resp:
            import json
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _post(path: str) -> bool:
    req = urllib.request.Request(f"{DASH_URL}{path}", method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _spawn(cmd: list[str], log_path: str) -> None:
    log = open(log_path, "ab")
    subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )


def _ensure_capture() -> str:
    if _pgrep(CAPTURE_PATTERN):
        return "capture_ok"
    _spawn(CAPTURE_CMD, "/tmp/futu_capture.log")
    return "capture_restarted"


def _ensure_dashboard_armed() -> str:
    state = _get_json("/api/state")
    if state is None:
        if not _pgrep(DASH_PATTERN):
            _spawn(DASH_CMD, "/tmp/paper_dashboard.log")
            time.sleep(2)
        return "dashboard_restarted"
    actions = []
    if not state.get("running"):
        if _post("/api/engine/start"):
            actions.append("started")
        time.sleep(1)
        state = _get_json("/api/state") or state
    if state.get("running") and state.get("mode") != "auto":
        if _post("/api/engine/auto/start"):
            actions.append("armed")
    return ",".join(actions) if actions else "dashboard_ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    print(f"watch_paper_session: interval={args.interval}s", flush=True)
    while True:
        now = datetime.now(timezone.utc)
        cap = _ensure_capture()
        dash = _ensure_dashboard_armed()
        state = _get_json("/api/state") or {}
        print(
            f"{now.isoformat()} rth={is_regular_trading_hours(now)} "
            f"{cap} {dash} running={state.get('running')} mode={state.get('mode')} "
            f"feed={state.get('futu_live_feed_active')} "
            f"armed={state.get('armed_strategies')} "
            f"gates={state.get('live_gate_regime')} "
            f"pairs={state.get('pairs_regime_gate_reason')}",
            flush=True,
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
