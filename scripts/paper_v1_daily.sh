#!/bin/bash
# Advance the Strategy V1 paper ledger once, unattended.
#
# Wrapped rather than called directly from launchd because the price top-up
# takes about five minutes -- Yahoo serves roughly 2.7 seconds per ticker no
# matter how the request is batched, so 120 names cannot go faster from here.
# Nobody should sit through that, and nobody should have to remember to. What
# matters is that the record has no gaps; the script itself fills in any
# missing days, so a skipped run self-heals on the next one.
#
# Every run appends to logs/paper_v1.log, including failures. A silent cron job
# that has been broken for a month is worse than no cron job.

set -uo pipefail

# launchd jobs have a much lower open-file limit than interactive shells.
# 120 symbols = 120 CSV files to write, and pandas opens several internal
# files per to_csv call. Raise the soft limit to match the hard limit.
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

LOG="$REPO/logs/paper_v1.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "===================================================================="
    echo "run at $(date '+%Y-%m-%d %H:%M:%S %Z')"
} >>"$LOG"

if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "FAILED: no interpreter at .venv/bin/python" >>"$LOG"
    exit 1
fi

"$REPO/.venv/bin/python" scripts/paper_track_v1.py >>"$LOG" 2>&1
status=$?

if [ $status -ne 0 ]; then
    echo "FAILED with exit code $status" >>"$LOG"
else
    "$REPO/.venv/bin/python" scripts/paper_track_v1.py --report >>"$LOG" 2>&1
fi

exit $status
