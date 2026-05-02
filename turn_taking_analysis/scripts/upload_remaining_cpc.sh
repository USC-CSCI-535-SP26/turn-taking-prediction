#!/usr/bin/env bash
# Upload pre-built CPC tarballs to Rasika's Drive.
# Modified version of scripts/upload_cpc.sh — tar step is gone (tarballs were
# pre-built by Rasika and shipped via USB alongside this script, no manifest
# needed). Same rclone invocation, same throttling, same logging style.
# Idempotent — safe to kill and re-run.
#
# Assumes this script lives in the same folder as the .tar files (the folder
# Rasika handed over on USB). To run:
#   bash upload_remaining_cpc.sh
set -euo pipefail

REMOTE=seamless_cpc                                  # your rclone remote name
DRIVE_DEST="535_project_data/data/spliced/cpc_tar"   # path under Rasika's My Drive
LOG=/tmp/upload_remaining_cpc.log

# Source dir = directory this script lives in (works regardless of cwd).
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) === src=$SRC_DIR remote=$REMOTE dest=$DRIVE_DEST"

# --- Step 1: sanity-check the source dir ------------------------------------
n_tars=$(ls "$SRC_DIR"/*.tar 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date +%H:%M:%S)] found $n_tars tarballs in $SRC_DIR"
if [ "$n_tars" -eq 0 ]; then
  echo "ERROR: no .tar files found alongside this script in $SRC_DIR"
  echo "  This script expects to live in the same folder as the .tar files"
  echo "  Rasika handed over (cpc_tarballs_for_gera/). Copy it back into that"
  echo "  folder and re-run."
  exit 1
fi

# --- Step 2: upload via rclone (throttled, resumable, integrity-checked) ----
# Throttle params copied verbatim from upload_cpc.sh — chosen after Rasika's
# first run hit Drive's per-100s rate limit (burst freezes at 18% with default
# --transfers=8). 15 MiB/s × 4 transfers stays under the per-user quota and
# gives a steady, predictable rate (~10-15 min for 10.6 GiB).
#   --checksum             : hash-compare existing dest files (the 403 already
#                            on Drive will hash-match → skipped automatically;
#                            also catches partial uploads on resume)
#   --include "*.tar"      : only upload tarballs (skip README.md, etc.)
#   --transfers=4          : reduce concurrent pressure on Drive's quota
#   --tpslimit=4 --burst=4 : cap rclone's API calls/sec → no 403 quotaExceeded
#   --bwlimit=15M          : hard cap upload speed; trades peak for sustained
#   --log-file             : audit trail, replayable diagnostics
echo "[$(date +%H:%M:%S)] uploading to ${REMOTE}:${DRIVE_DEST}..."
rclone copy "$SRC_DIR" "${REMOTE}:${DRIVE_DEST}" \
  --include "*.tar" \
  --checksum \
  --transfers=4 --drive-chunk-size=64M \
  --tpslimit=4 --tpslimit-burst=4 \
  --bwlimit=15M \
  --progress --stats=15s --log-file="$LOG" --log-level INFO

# --- Step 3: report ---------------------------------------------------------
# No cleanup — tarballs are user-supplied source data, not script-generated
# artifacts. Safe to delete manually after Rasika confirms the upload.
echo "[$(date +%H:%M:%S)] upload complete — tarballs preserved at $SRC_DIR"
echo "=== $(date) === DONE"
echo ""
echo "Ask Rasika to verify from her end:"
echo "  rclone size ${REMOTE}:${DRIVE_DEST}"
echo "Expected: 457 objects (was 403 before this run; 54 added)."
