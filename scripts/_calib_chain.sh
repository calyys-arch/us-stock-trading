#!/bin/bash
# Waits for <signal>'s "old" phase checkpoint to exist (polling, since the
# old-phase recovery for sweep_reclaim/fvg_retest/orb_vwap can take hours
# and was launched separately/earlier), then automatically launches the
# "new" (calibrated) phase — so the whole 6-signal pipeline completes
# unattended without a human needing to manually chain the two phases.
set -euo pipefail
SIGNAL="$1"
cd "$(dirname "$0")/.."
source .venv/bin/activate
CKPT="backtests/reports/_checkpoint_calib_${SIGNAL}.json"

while true; do
  if [ -f "$CKPT" ] && python3 -c "import json,sys; d=json.load(open('$CKPT')); sys.exit(0 if d.get('old') is not None else 1)"; then
    break
  fi
  sleep 60
done

echo "[$(date)] old phase ready for $SIGNAL -- launching old_fixed + new (calibrated) phases" >> "backtests/reports/_calib_logs/${SIGNAL}_new.log"
python scripts/_calibration_validation.py "$SIGNAL" both >> "backtests/reports/_calib_logs/${SIGNAL}_new.log" 2>&1
