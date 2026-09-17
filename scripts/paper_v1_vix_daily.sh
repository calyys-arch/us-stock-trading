#!/bin/bash
# Advance the Strategy V1 + VIX-overlay paper ledger once, unattended.
#
# This is the PARALLEL forward-test of adjust_exposure_for_vix() in
# python/portfolio/risk_controls.py: same universe, same volatility-target
# exposure logic as scripts/paper_v1_daily.sh's ledger, but the already-
# banded exposure is additionally scaled by that day's VIX bracket before
# trading. It writes to its OWN journal (data/paper_v1_vix/journal.jsonl)
# on its OWN 250-day clock, started separately via --init -- it never
# touches, resumes, or backfills the primary V1 ledger.
#
# --no-update: this job is scheduled at 07:10, ten minutes after
# com.usstock.paper-v1 (07:00), on the SAME machine, against the SAME
# data/history price cache. paper_v1_daily.sh's _top_up() already refreshed
# every cached symbol for today's session by the time this runs, and
# _top_up() takes ~5 minutes across 120 names -- there is nothing this job
# would gain from repeating it, so it reads the cache as-is. (Verified
# 2026-09-17: --no-update exists in scripts/paper_track_v1.py and does
# exactly this -- skips the _top_up() call only; every other code path,
# including the VIX fetch below, is unaffected.) If the 07:00 job is ever
# moved, delayed, or fails before finishing its top-up, this job's price
# marks are only as fresh as whatever the cache last had -- the ledger's own
# staleness guard (MAX_CARRY_DAYS) is what catches that, not this script.
#
# Every run appends to logs/paper_v1_vix.log, including failures.

set -uo pipefail

# Same open-file-limit fix as paper_v1_daily.sh: launchd's default soft limit
# is too low for 120 symbols' worth of CSV cache reads.
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

JOURNAL="data/paper_v1_vix/journal.jsonl"
LOG="$REPO/logs/paper_v1_vix.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "===================================================================="
    echo "run at $(date '+%Y-%m-%d %H:%M:%S %Z')"
} >>"$LOG"

if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "FAILED: no interpreter at .venv/bin/python" >>"$LOG"
    exit 1
fi

"$REPO/.venv/bin/python" scripts/paper_track_v1.py \
    --vix-overlay --journal-path "$JOURNAL" --no-update >>"$LOG" 2>&1
status=$?

if [ $status -ne 0 ]; then
    # Includes a mid-run halt (stale price, missing VIX data, or a single-day
    # move past CATASTROPHE_DAY_RETURN), not just an outright crash --
    # scripts/paper_track_v1.py returns non-zero in every one of those cases.
    echo "FAILED with exit code $status" >>"$LOG"
else
    "$REPO/.venv/bin/python" scripts/paper_track_v1.py \
        --report --journal-path "$JOURNAL" >>"$LOG" 2>&1
    # paper_v1_early_read.py is intentionally NOT called here: it rebuilds
    # the no-VIX historical exposure path and has no VIX-overlay equivalent.
    # Building one is out of scope for this ledger; --report above already
    # covers day-count-gated Sharpe/drawdown reporting for this journal.
fi

exit $status
