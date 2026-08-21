"""Run only the 15×7 cells that are still missing, 15m then 5m then 1m.

Does not --resume volume_route_strategies.json.
Does not start a 1m cell while another heavy WFO is still writing.
Official hard-AND is unchanged.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "scripts" / "run_entry_hypothesis_gates.py"
LIVE_5M = Path.home() / ".cursor/projects/Users-gseed-Desktop-WORKSPACE-us-stock-trading/terminals/120071.txt"

# Faster charts first. 1m last, and only after the in-flight VSA 5m job exits.
QUEUE = [
    ("vsa_effort", "15"),
    ("auction_reclaim", "15"),
    ("absorption_breakout", "15"),
    ("vsa_no_demand", "15"),
    ("obv_divergence", "15"),
    ("vsa_effort", "5"),
    ("absorption_breakout", "5"),
    ("vsa_effort", "1"),
    ("auction_reclaim", "1"),
    ("vsa_no_demand", "1"),
    ("obv_divergence", "1"),
]


def _live_5m_running() -> bool:
    if not LIVE_5M.exists():
        return False
    text = LIVE_5M.read_text(encoding="utf-8", errors="replace")
    return "\nexit_code:" not in text and "\n---\nexit_code:" not in text.split("running_for_ms")[-1]


def main() -> int:
    for name, chart in QUEUE:
        if chart == "1":
            while _live_5m_running():
                print("waiting for in-flight 5m VSA/OBV WFO before any 1m cell...", flush=True)
                time.sleep(60)
        print(f"\n######## queue {name} {chart}m ########", flush=True)
        cmd = [
            str(PY), "-u", str(RUNNER),
            "--hypothesis", name,
            "--chart", chart,
            "--resume",
            "--start", "2025-08-01",
            "--end", "2026-07-01",
        ]
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"FAILED {name} {chart}m rc={rc}", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
