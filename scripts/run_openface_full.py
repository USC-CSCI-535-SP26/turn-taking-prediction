#!/usr/bin/env python3
"""
run_openface_full.py

Runs OpenFace FeatureExtraction on the FULL video for every participant
across all 100 interactions (both Imitator and V03). No clipping — OpenFace
processes each MP4 from start to finish.

Output per participant:
    {participant_dir}/openface_full/
        {pid}.csv         <- OpenFace CSV for the full interaction
        {pid}.hog         <- HOG features (can be ignored)

Usage:
    python run_openface_full.py \\
        --input-dir annotated_interactions \\
        --openface-bin ~/OpenFace/build/bin/FeatureExtraction

    # Skip participants that already have a full OpenFace CSV:
    python run_openface_full.py \\
        --input-dir annotated_interactions \\
        --openface-bin ~/OpenFace/build/bin/FeatureExtraction \\
        --skip-existing

    # Only process specific interactions:
    python run_openface_full.py \\
        --input-dir annotated_interactions \\
        --openface-bin ~/OpenFace/build/bin/FeatureExtraction \\
        --only V00_S1288_I00000099 V03_S1234_I00000001
"""

import argparse
import json
import os
import subprocess
import sys


# =============================================================================
# Helpers
# =============================================================================

def check_openface(openface_bin: str):
    if not os.path.isfile(openface_bin):
        raise RuntimeError(f"OpenFace binary not found: {openface_bin}")
    if not os.access(openface_bin, os.X_OK):
        raise RuntimeError(f"OpenFace binary not executable: {openface_bin}")


def get_participants(interaction_dir: str) -> list:
    """Return list of {role, pid, dir} for each participant."""
    participants = []
    for name in sorted(os.listdir(interaction_dir)):
        if not name.startswith("participant_"):
            continue
        pdir = os.path.join(interaction_dir, name)
        if not os.path.isdir(pdir):
            continue
        parts = name.split("_", 2)
        role = parts[1]
        pid  = parts[2]
        participants.append({
            "role": role,
            "pid":  pid,
            "dir":  pdir,
        })
    return participants


def find_mp4(pdir: str) -> str | None:
    """Find the MP4 file in a participant directory."""
    for fname in os.listdir(pdir):
        if fname.endswith(".mp4"):
            return os.path.join(pdir, fname)
    return None


def openface_full_done(pdir: str, pid: str) -> bool:
    """Return True if a full OpenFace CSV already exists for this participant."""
    csv_path = os.path.join(pdir, "openface_full", f"{pid}.csv")
    return os.path.exists(csv_path)


def run_openface(openface_bin: str, video_path: str, out_dir: str, pid: str) -> tuple:
    """
    Run OpenFace FeatureExtraction on a full video.
    Returns (success, message).
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        openface_bin,
        "-f", video_path,
        "-out_dir", out_dir,
        "-of", pid,
        "-gaze",
        "-pose",
        "-aus",
        "-2Dfp",
        "-3Dfp",
        "-wild",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, f"exit code {result.returncode}: {result.stderr[:200]}"
    csv_path = os.path.join(out_dir, f"{pid}.csv")
    if not os.path.exists(csv_path):
        return False, "CSV not produced"
    return True, "ok"


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenFace on the full video for all 100 interactions. "
            "Outputs CSVs to participant_dir/openface_full/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Path to the annotated_interactions/ directory.",
    )
    parser.add_argument(
        "--openface-bin", required=True,
        help="Path to OpenFace FeatureExtraction binary.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="INTERACTION_ID",
        help="Only process these specific interaction IDs (space-separated).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip participants that already have a full OpenFace CSV.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print OpenFace stderr output.",
    )
    parser.add_argument(
        "--split", type=int, choices=[1, 2], default=None,
        help="Process only half the interactions: 1 = first half, 2 = second half.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: --input-dir not found: {args.input_dir}")
        sys.exit(1)

    try:
        check_openface(args.openface_bin)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Seamless Interaction: Run OpenFace on Full Videos")
    print("=" * 60)
    print(f"  Input dir:       {args.input_dir}")
    print(f"  OpenFace binary: {args.openface_bin}")
    print(f"  Skip existing:   {args.skip_existing}")

    # Discover interactions
    all_interactions = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("V")
    ])

    if args.only:
        interactions = [i for i in all_interactions if i in args.only]
        missing = set(args.only) - set(interactions)
        if missing:
            print(f"  WARNING: interactions not found: {missing}")
    else:
        interactions = all_interactions

    # Apply split filter
    if args.split is not None:
        midpoint = len(interactions) // 2
        if args.split == 1:
            interactions = interactions[:midpoint]
            print(f"  Split 1: interactions 1–{midpoint} of {len(all_interactions)}")
        elif args.split == 2:
            interactions = interactions[midpoint:]
            print(f"  Split 2: interactions {midpoint+1}–{len(all_interactions)} of {len(all_interactions)}")

    print(f"\nFound {len(interactions)} interaction(s) to process.\n")

    total_processed = total_skipped = total_errors = 0
    all_errors = []

    for idx, name in enumerate(interactions, 1):
        idir = os.path.join(args.input_dir, name)
        print(f"[{idx}/{len(interactions)}] {name}")

        participants = get_participants(idir)
        if not participants:
            print(f"  No participants found — skipping")
            continue

        for p in participants:
            pid  = p["pid"]
            pdir = p["dir"]

            # Skip if already done
            if args.skip_existing and openface_full_done(pdir, pid):
                print(f"  {pid}: skip (already done)")
                total_skipped += 1
                continue

            mp4 = find_mp4(pdir)
            if mp4 is None:
                print(f"  {pid}: ERROR — no MP4 found in {pdir}")
                all_errors.append((name, pid, "no MP4 found"))
                total_errors += 1
                continue

            out_dir = os.path.join(pdir, "openface_full")
            print(f"  {pid}: running OpenFace on {os.path.basename(mp4)} ...")

            success, msg = run_openface(args.openface_bin, mp4, out_dir, pid)

            if success:
                print(f"  {pid}: done")
                total_processed += 1
            else:
                print(f"  {pid}: ERROR — {msg}")
                if args.verbose:
                    print(f"    {msg}")
                all_errors.append((name, pid, msg))
                total_errors += 1

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Processed : {total_processed}")
    print(f"  Skipped   : {total_skipped}")
    print(f"  Errors    : {total_errors}")
    print(f"{'='*60}")

    if all_errors:
        print("\nErrors:")
        for iname, pid, err in all_errors:
            print(f"  {iname} / {pid}: {err}")


if __name__ == "__main__":
    main()
