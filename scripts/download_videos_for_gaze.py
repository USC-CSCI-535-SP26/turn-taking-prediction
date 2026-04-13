#!/usr/bin/env python3
"""
download_videos_for_gaze.py

Downloads MP4 video files for participants in annotated_interactions/ that are
missing Imitator movement features (has_imitator_movement=0 in filelist.csv).
These are the ~46 interactions for which we need to run OpenFace to extract
gaze features instead.

Only downloads videos for participants that:
  1. Belong to an interaction already present in --input-dir
  2. Have has_imitator_movement=0 in filelist.csv
  3. Do NOT already have a .mp4 file in their participant directory

S3 URL pattern:
    https://dl.fbaipublicfiles.com/seamless_interaction/{label}/{split}/video/{file_id}.mp4

Downloaded files are saved as:
    {participant_dir}/{file_id}.mp4

Usage:
    python download_videos_for_gaze.py \\
        --input-dir annotated_interactions \\
        --repo-path /path/to/seamless_interaction

    # Dry run (show what would be downloaded):
    python download_videos_for_gaze.py \\
        --input-dir annotated_interactions \\
        --repo-path /path/to/seamless_interaction \\
        --dry-run
"""

import argparse
import csv
import os
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


S3_BASE_URL = "https://dl.fbaipublicfiles.com/seamless_interaction"


# =============================================================================
# Helpers
# =============================================================================

def load_filelist(repo_path: str) -> dict[str, dict]:
    """Load filelist.csv, keyed by file_id."""
    path = os.path.join(repo_path, "assets", "filelist.csv")
    if not os.path.exists(path):
        print(f"ERROR: filelist.csv not found at {path}")
        sys.exit(1)
    with open(path, newline="") as f:
        return {row["file_id"]: row for row in csv.DictReader(f)}


def discover_participants(input_dir: str) -> list[dict]:
    """
    Walk annotated_interactions/ and return all participants as dicts with:
        interaction_id, role, pid, dir, file_id
    """
    participants = []
    for interaction_id in sorted(os.listdir(input_dir)):
        idir = os.path.join(input_dir, interaction_id)
        if not (os.path.isdir(idir) and interaction_id.startswith("V")):
            continue
        for name in sorted(os.listdir(idir)):
            if not name.startswith("participant_"):
                continue
            pdir = os.path.join(idir, name)
            if not os.path.isdir(pdir):
                continue
            parts = name.split("_", 2)   # ["participant", "a"/"b", "P####"]
            role = parts[1]
            pid  = parts[2]
            participants.append({
                "interaction_id": interaction_id,
                "role": role,
                "pid": pid,
                "dir": pdir,
                "file_id": f"{interaction_id}_{pid}",
            })
    return participants


def already_has_mp4(pdir: str, file_id: str) -> bool:
    """Check whether an MP4 already exists in the participant directory."""
    expected = os.path.join(pdir, f"{file_id}.mp4")
    if os.path.exists(expected):
        return True
    # Also accept any .mp4 in the directory (in case of name variation)
    for fname in os.listdir(pdir):
        if fname.endswith(".mp4"):
            return True
    return False


def download_file(url: str, output_path: str) -> tuple[bool, str]:
    """Download a single file. Returns (success, message)."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as response:
            data = response.read()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        size_mb = len(data) / 1_048_576
        return True, f"OK ({size_mb:.1f} MB)"
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return False, "NOT FOUND (HTTP 403)"
        return False, f"HTTP ERROR {e.code}"
    except Exception as e:
        return False, f"ERROR: {e}"


# =============================================================================
# Main logic
# =============================================================================

def build_download_tasks(
    input_dir: str,
    filelist: dict[str, dict],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Identify which MP4s need to be downloaded.

    Returns:
        tasks            — list of dicts with keys: file_id, split, output_path
        skipped_has_mov  — file_ids skipped because they have imitator movement
        skipped_has_mp4  — file_ids skipped because MP4 already present
    """
    participants = discover_participants(input_dir)
    tasks = []
    skipped_has_movement = []
    skipped_has_mp4 = []
    not_in_filelist = []

    for p in participants:
        entry = filelist.get(p["file_id"])
        if entry is None:
            not_in_filelist.append(p["file_id"])
            continue

        # Skip if Imitator features exist — gaze already available via .npy
        if entry.get("has_imitator_movement") == "0":
            skipped_has_movement.append(p["file_id"])
            continue

        # Skip if MP4 already downloaded
        if already_has_mp4(p["dir"], p["file_id"]):
            skipped_has_mp4.append(p["file_id"])
            continue

        tasks.append({
            "file_id": p["file_id"],
            "split": entry["split"],
            "output_path": os.path.join(p["dir"], f"{p['file_id']}.mp4"),
        })

    if not_in_filelist:
        print(f"  WARNING: {len(not_in_filelist)} participant(s) not found in "
              f"filelist.csv — skipping")

    return tasks, skipped_has_movement, skipped_has_mp4


def run_downloads(
    tasks: list[dict],
    dry_run: bool = False,
    num_workers: int = 4,
) -> None:
    label = "naturalistic"

    def _do(task):
        url = f"{S3_BASE_URL}/{label}/{task['split']}/video/{task['file_id']}.mp4"
        fname = os.path.basename(task["output_path"])
        if dry_run:
            return fname, True, f"DRY RUN: {url}"
        success, msg = download_file(url, task["output_path"])
        return fname, success, msg

    print(f"\n{'='*60}")
    print(f"{'DRY RUN: ' if dry_run else ''}Downloading {len(tasks)} MP4 files "
          f"({num_workers} threads)...")
    print(f"{'='*60}\n")

    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_do, t): t for t in tasks}
        for future in as_completed(futures):
            fname, success, msg = future.result()
            print(f"  {fname}: {msg}")
            if success:
                ok += 1
            elif "NOT FOUND" in msg:
                skip += 1
            else:
                fail += 1

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Downloaded:    {ok}")
    print(f"  Not on server: {skip}")
    print(f"  Errors:        {fail}")
    print(f"{'='*60}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download MP4 video files for interactions missing Imitator gaze "
            "features, so OpenFace can be run on them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Path to the annotated_interactions/ directory.",
    )
    parser.add_argument(
        "--repo-path", required=True,
        help="Path to the cloned seamless_interaction repository (for filelist.csv).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel download threads. Default: 4.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: --input-dir not found: {args.input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Seamless Interaction: Download Videos for OpenFace Gaze")
    print("=" * 60)

    print(f"\nLoading filelist.csv...")
    filelist = load_filelist(args.repo_path)
    print(f"  {len(filelist)} entries loaded.")

    print(f"\nScanning {args.input_dir}...")
    tasks, skipped_mov, skipped_mp4 = build_download_tasks(args.input_dir, filelist)

    print(f"\n  Participants with Imitator features (skip):  {len(skipped_mov)}")
    print(f"  Participants with MP4 already present (skip): {len(skipped_mp4)}")
    print(f"  Participants needing MP4 download:            {len(tasks)}")

    # Summarise which interactions are affected
    affected_interactions = sorted(set(
        t["file_id"].rsplit("_", 1)[0] for t in tasks
    ))
    if affected_interactions:
        print(f"\n  {len(affected_interactions)} interactions need video download:")
        for iid in affected_interactions:
            print(f"    {iid}")

    if not tasks:
        print("\nNothing to download. Exiting.")
        return

    run_downloads(tasks, dry_run=args.dry_run, num_workers=args.num_workers)


if __name__ == "__main__":
    main()

