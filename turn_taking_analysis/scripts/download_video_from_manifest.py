#!/usr/bin/env python3
"""
download_video_from_manifest.py

Download per-participant .mp4 files for every interaction listed in a
manifest CSV (as produced by build_manifest.py) and save them
under a flat directory.

Default layout
--------------
    turn_taking_analysis/
      manifests/manifest.csv         <-- input
      subset/
        video/
          V00_S0743_I00000483_P0885.mp4  <-- output
          V00_S0743_I00000483_P0888.mp4
          V01_S0325_I00000314_P1666.mp4
          ...

Size warning
------------
Seamless videos are UHD 4K at 30 FPS (paper §2.4.2, p. 13). A single
3-minute participant video can run 1–3 GB. For the default 24-dyad
POC manifest (48 participant videos) the total payload is likely in
the tens of GB — possibly ~100 GB worst case. Make sure the target
filesystem has headroom before running the full manifest. For initial
testing, use --filter-split val (8 videos).

The shared download_file helper reads the full response into memory
before writing to disk. For a 2 GB video this transiently uses ~2 GB
of RAM — inelegant but workable on a laptop. If memory becomes a
concern, adding a streaming-download variant to
scripts/download_annotated_interactions.py is the right fix.

Reuse of shared helpers
-----------------------
This script is a thin orchestration layer around helpers already in
csci535-project/scripts/download_annotated_interactions.py. Reused:

    build_s3_url()          — canonical S3 URL construction
    download_file()         — HTTP-to-disk writer with 403-as-missing semantics
    extract_interaction_id()— file_id -> interaction_id
    extract_participant_id()— file_id -> participant_id

The manifest's precomputed `video_url_a` / `video_url_b` columns are
NOT consumed; URLs are regenerated via build_s3_url so the canonical
URL scheme lives in one place.

Why a flat directory
--------------------
Every Seamless file_id uniquely encodes {vendor, session, interaction,
participant}. Nesting by interaction_id adds no information and
complicates downstream globbing for OpenFace extraction. Downstream
scripts can recover the interaction_id via extract_interaction_id()
or by joining back to the manifest on file_id.

Behavior
--------
- Reads (split, seamless_split, label, file_id_a, file_id_b, interaction_id)
  from the manifest. URLs are regenerated.
- For each row, queues both participant videos for download.
- Skips files that already exist and are nonempty unless --overwrite.
- Downloads concurrently via ThreadPoolExecutor. Default workers=2
  rather than 4 (video payloads are large; avoid saturating bandwidth
  and RAM).
- HTTP 403 is treated as "file not on S3" (consistent with the rest
  of csci535-project/scripts/), not as a hard failure.

Usage
-----
    # Default: download all videos from manifest.csv
    python download_video_from_manifest.py

    # Only the val split (staged debugging)
    python download_video_from_manifest.py --filter-split val

    # Print URLs without downloading
    python download_video_from_manifest.py --dry-run

    # Force re-download
    python download_video_from_manifest.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# -----------------------------------------------------------------------------
# Re-use helpers from csci535-project/scripts. This script lives at
# turn_taking_analysis/scripts/, so we resolve the sibling directory and
# insert it on sys.path — matching the pattern in build_poc_manifest.py
# and download_audio_from_manifest.py.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                 # .../csci535-project
SHARED_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (  # noqa: E402
    build_s3_url,
    download_file,
    extract_interaction_id,
    extract_participant_id,
)


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_MANIFEST_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "manifest.csv"
)
DEFAULT_OUTPUT_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "video"
)

# Category + extension are fixed for this script — we only fetch mp4s.
S3_CATEGORY = "video"
FILE_EXT = "mp4"

# Lower default than the audio script — videos are large, and 4 parallel
# streams can easily saturate a laptop NIC or fill RAM if download_file
# reads each full response into memory.
DEFAULT_NUM_WORKERS = 2


# =============================================================================
# Manifest parsing
# =============================================================================

REQUIRED_COLUMNS = {
    "split",              # our POC split: train/val/test
    "seamless_split",     # Seamless's own split: train/dev/test (for S3 URL)
    "label",              # "naturalistic" (for S3 URL)
    "interaction_id",     # for logging / cross-check
    "file_id_a",
    "file_id_b",
}


def load_manifest(manifest_path: str) -> list[dict]:
    """
    Load manifest.csv into a list of dict rows.

    Validates that every required column is present — surfaces schema
    drift immediately rather than letting rows silently produce None.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run build_manifest.py first."
        )

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: "
                f"{sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest {manifest_path} has no rows.")

    return rows


# =============================================================================
# Building the download task list
# =============================================================================

class DownloadTask:
    """One video to fetch: url + destination path + bookkeeping tags.

    Tags come from the manifest row (split, interaction_id) plus the
    shared extract_* helpers applied to file_id. Storing them on the
    task makes the per-line progress output readable without
    re-deriving anything downstream.
    """

    __slots__ = (
        "split", "interaction_id", "participant_id",
        "file_id", "url", "output_path",
    )

    def __init__(self, split: str, interaction_id: str, participant_id: str,
                 file_id: str, url: str, output_path: str):
        self.split = split
        self.interaction_id = interaction_id
        self.participant_id = participant_id
        self.file_id = file_id
        self.url = url
        self.output_path = output_path

    def __repr__(self) -> str:
        return f"DownloadTask({self.file_id})"


def build_tasks(
    rows: list[dict],
    output_dir: str,
    filter_split: str | None,
) -> list[DownloadTask]:
    """
    Expand manifest rows into one DownloadTask per participant video
    (two tasks per row).

    URLs are generated via build_s3_url from (label, seamless_split,
    file_id) — NOT read from the manifest's video_url_* columns — so
    the S3 URL scheme stays canonicalized in the shared helper.

    Tag fields (interaction_id, participant_id) are derived from
    file_id via the shared extract_* helpers to keep parsing logic in
    one place. interaction_id is cross-checked against the manifest
    column as a guard against accidental manifest edits.
    """
    tasks: list[DownloadTask] = []
    for row in rows:
        if filter_split and row["split"] != filter_split:
            continue

        label = row["label"]
        seamless_split = row["seamless_split"]

        for side in ("a", "b"):
            file_id = row[f"file_id_{side}"]
            if not file_id:
                print(f"  WARN: row {row['interaction_id']} has empty "
                      f"file_id_{side}; skipping.", file=sys.stderr)
                continue

            # Canonical URL construction via shared helper.
            url = build_s3_url(
                label=label,
                split=seamless_split,
                category=S3_CATEGORY,
                file_id=file_id,
                ext=FILE_EXT,
            )

            # Shared extractors + cross-check with manifest.
            derived_iid = extract_interaction_id(file_id)
            if derived_iid != row["interaction_id"]:
                print(f"  WARN: file_id {file_id} implies interaction "
                      f"{derived_iid} but manifest row says "
                      f"{row['interaction_id']}; using derived value.",
                      file=sys.stderr)

            output_path = os.path.join(output_dir, f"{file_id}.{FILE_EXT}")

            tasks.append(DownloadTask(
                split=row["split"],
                interaction_id=derived_iid,
                participant_id=extract_participant_id(file_id),
                file_id=file_id,
                url=url,
                output_path=output_path,
            ))

    tasks.sort(key=lambda t: (t.split, t.interaction_id, t.file_id))
    return tasks


# =============================================================================
# Execution
# =============================================================================

def already_have(output_path: str) -> bool:
    """True iff the file exists on disk and is nonempty. A 0-byte mp4
    from an interrupted prior run is treated as not-yet-downloaded.

    Note: we do NOT validate that the on-disk mp4 is playable — a
    partial download that happens to have nonzero size will be trusted.
    If this becomes a problem in practice, add ffprobe-based
    verification here or (more simply) always run with --overwrite
    after an interrupted session."""
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def run_task(task: DownloadTask, overwrite: bool, dry_run: bool) -> dict:
    """
    Execute a single DownloadTask. Actual HTTP work is delegated to
    the shared download_file helper (which handles HTTP-403-as-missing,
    parent-dir creation, and byte-accurate progress messages).

    Result status codes:
        "skipped"  — on disk already; not overwriting.
        "dry"      — --dry-run; URL printed but not fetched.
        "ok"       — downloaded successfully.
        "missing"  — S3 returned 403 (file not on S3).
        "failed"   — any other error; message carries detail.
    """
    if not overwrite and already_have(task.output_path):
        return {
            "task": task,
            "status": "skipped",
            "message": f"already on disk "
                       f"({os.path.getsize(task.output_path):,} bytes)",
        }

    if dry_run:
        return {"task": task, "status": "dry", "message": f"DRY: {task.url}"}

    # download_file is the shared helper: HTTP GET, writes all bytes
    # to disk, returns (success, message). 403 comes back with
    # "NOT FOUND" in the message per the helper's documented contract.
    success, msg = download_file(task.url, task.output_path)
    if success:
        return {"task": task, "status": "ok", "message": msg}
    if "NOT FOUND" in msg or "403" in msg:
        return {"task": task, "status": "missing", "message": msg}
    return {"task": task, "status": "failed", "message": msg}


def run_tasks(
    tasks: list[DownloadTask],
    num_workers: int,
    overwrite: bool,
    dry_run: bool,
) -> list[dict]:
    """Run all tasks concurrently and return the per-task results."""
    results: list[dict] = []

    # Pre-create output dir so --dry-run also leaves the directory
    # in place (and real runs don't race on mkdir from N workers).
    if tasks and not dry_run:
        os.makedirs(os.path.dirname(tasks[0].output_path), exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(run_task, t, overwrite, dry_run): t for t in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            task = result["task"]
            print(f"  [{result['status']:<7}] {task.split:<5} "
                  f"{task.file_id}  {result['message']}")
            results.append(result)

    results.sort(key=lambda r: (r["task"].split, r["task"].interaction_id,
                                r["task"].file_id))
    return results


# =============================================================================
# Summary
# =============================================================================

def print_summary(results: list[dict], elapsed_s: float, dry_run: bool) -> None:
    """Print counts by status + total downloaded bytes."""
    counts: dict[str, int] = {}
    bytes_downloaded = 0
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "ok":
            # download_file returns "OK (NNN bytes)"; parse back out.
            try:
                bytes_downloaded += int(
                    r["message"].split("(")[1].split(" ")[0].replace(",", "")
                )
            except (IndexError, ValueError):
                pass

    print("\n" + "=" * 64)
    print("DOWNLOAD SUMMARY" + ("  [DRY-RUN]" if dry_run else ""))
    print("=" * 64)
    print(f"  Total tasks:  {len(results)}")
    for status in ("ok", "skipped", "dry", "missing", "failed"):
        n = counts.get(status, 0)
        if n:
            print(f"  {status:<10} {n}")
    if bytes_downloaded:
        mib = bytes_downloaded / (1024 * 1024)
        gib = bytes_downloaded / (1024 ** 3)
        print(f"  Downloaded:   {bytes_downloaded:,} bytes "
              f"({mib:,.1f} MiB / {gib:,.2f} GiB)")
    print(f"  Elapsed:      {elapsed_s:,.1f} s")
    print("=" * 64)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download per-participant .mp4 files for every interaction "
            "in a POC manifest CSV. Reuses build_s3_url + download_file "
            "from csci535-project/scripts/download_annotated_interactions.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST_PATH,
        help="Path to the manifest CSV. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save .mp4 files into. Default: %(default)s",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of concurrent download threads. Default: %(default)s "
            "(conservative — videos are large)."
        ),
    )
    parser.add_argument(
        "--filter-split", choices=["train", "val", "test"], default=None,
        help=(
            "If set, only download rows whose manifest `split` column "
            "matches (useful for staged debugging: download val first, "
            "confirm everything works, then train/test)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Re-download files that already exist on disk. Default is "
            "to skip them (idempotent resume)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print URLs that would be downloaded without fetching anything.",
    )
    args = parser.parse_args()

    print(f"Loading manifest:  {args.manifest}")
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  {len(rows):,} interactions in manifest "
          f"({2 * len(rows):,} participant videos)")

    tasks = build_tasks(rows, args.output_dir, args.filter_split)
    if args.filter_split:
        print(f"  Filtered to split='{args.filter_split}': "
              f"{len(tasks):,} tasks")
    if not tasks:
        print("Nothing to download. Exiting.")
        return 0

    print(f"Output directory:  {args.output_dir}")
    print(f"Workers:           {args.num_workers}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")
    print(f"Starting {len(tasks):,} download task(s)...\n")
    print("NOTE: Seamless videos are UHD 4K; expect ~1–3 GB per file. "
          "The default 24-dyad POC manifest is likely tens of GB total.\n")

    start = time.time()
    results = run_tasks(
        tasks,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    elapsed = time.time() - start

    print_summary(results, elapsed, args.dry_run)

    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_missing = sum(1 for r in results if r["status"] == "missing")
    if n_failed:
        return 2
    if n_missing:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
