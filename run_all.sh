#!/bin/bash
# Run the full analysis pipeline (scripts 01-14 and 16) in order.
# Stops at the first error, so no step can use outputs left over from an older run.
# Script 00 (hole selection) is NOT included: run it once, then download the mineral logs.
#
# Usage (from the repo folder, with the venv active):
#   ./run_all.sh 2>&1 | tee outputs/run_log.txt
set -e
set -o pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

for s in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 16; do   # 15 is optional (needs raw TSG files + pytsg)
  f=$(ls pipeline/${s}_*.py)
  echo ""
  echo "=================== $f ==================="
  "$PY" "$f"
done

echo ""
echo "=== Done: all outputs and figures in outputs/ come from this run ($(date '+%Y-%m-%d %H:%M'))."
