"""
Merges the per-signal checkpoint JSONs written by
scripts/_resume_new_signals_validation.py into the final combined
backtests/reports/new_signals_report.md + .json — reuses
run_intraday_backtest.py's exact report-writing functions so the final
report is byte-for-byte in the same format as a normal (uninterrupted)
`--signal new` run would have produced.

Usage:
    python scripts/_merge_new_signals_checkpoints.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_intraday_backtest import (  # noqa: E402
    NEW_REPORT_JSON_PATH,
    NEW_SIGNALS,
    write_new_signals_report,
    write_report_json,
)

CHECKPOINT_DIR = Path("backtests/reports")


def _checkpoint_path(signal_name: str) -> Path:
    return CHECKPOINT_DIR / f"_checkpoint_{signal_name}.json"


def main() -> None:
    results = []
    for signal_name in NEW_SIGNALS:
        p = _checkpoint_path(signal_name)
        if not p.exists():
            raise SystemExit(f"missing checkpoint for {signal_name}: {p} — run "
                              f"scripts/_resume_new_signals_validation.py {signal_name} first")
        state = json.loads(p.read_text(encoding="utf-8"))
        r = dict(state["main"])
        if state.get("full_grid"):
            r["full_grid"] = state["full_grid"]
        elif state.get("full_grid_skipped_reason"):
            r["full_grid_skipped_reason"] = state["full_grid_skipped_reason"]
        results.append(r)

    out_path = write_new_signals_report(results)
    json_path = write_report_json(results, path=NEW_REPORT_JSON_PATH)
    print(f"Report written to {out_path} (machine-readable: {json_path})")
    for r in results:
        print(f"  {r['signal']}: {r['decision']}")


if __name__ == "__main__":
    main()
