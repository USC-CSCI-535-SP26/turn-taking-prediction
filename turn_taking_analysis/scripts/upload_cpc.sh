#!/usr/bin/env bash
# Tar each interaction's CPC dirs, upload all 457 tarballs to Drive, clean up.
# Idempotent — safe to kill and re-run.
set -euo pipefail

REMOTE=seamless_cpc                                  # your rclone remote name
DRIVE_DEST="535_project_data/data/spliced/cpc_tar"   # path under My Drive
TAR_DIR=/tmp/cpc_tarballs
LOG=/tmp/upload_cpc.log

# Resolve project root from script location (works regardless of cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
cd "$PROJ"

mkdir -p "$TAR_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) === project=$PROJ remote=$REMOTE dest=$DRIVE_DEST"

# --- Step 1: tar each interaction (parallel, skip-if-exists) ----------------
echo "[$(date +%H:%M:%S)] tarring interactions..."
awk -F, 'NR>1 {print $11"\t"$12"\t"$3}' manifests/manifest.csv \
  | xargs -P 4 -L 1 bash -c '
      iid="$2"; a="$0"; b="$1"
      out="'"$TAR_DIR"'/${iid}.tar"
      if [ -s "$out" ]; then exit 0; fi
      tar -cf "$out" -C subset/cpc "$a" "$b"
    '
echo "[$(date +%H:%M:%S)] tar done: $(ls "$TAR_DIR" | wc -l | tr -d ' ') tarballs"

# --- Step 2: upload via rclone (parallel, resumable) ------------------------
echo "[$(date +%H:%M:%S)] uploading to ${REMOTE}:${DRIVE_DEST}..."
rclone copy "$TAR_DIR" "${REMOTE}:${DRIVE_DEST}" \
  --transfers=8 --drive-chunk-size=64M \
  --progress --stats=15s --log-level INFO

# --- Step 3: cleanup --------------------------------------------------------
echo "[$(date +%H:%M:%S)] cleaning up local tarballs..."
rm -rf "$TAR_DIR"
echo "=== $(date) === DONE"