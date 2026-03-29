#!/usr/bin/env python3
"""
download_emotion_features.py

Downloads the 5 emotion-related movement features from Meta's S3 bucket for each
participant in each of the 100 annotated interactions, and saves them as a single
consolidated NPZ file (emotion_features.npz) in each participant's directory.

The 5 features downloaded per participant:
    1. movement/emotion_arousal     — continuous arousal intensity [-1, 1], shape (N_frames,)
    2. movement/EmotionArousalToken — quantized arousal (12 bins), shape (N_frames,)
    3. movement/emotion_valence     — continuous valence [-1, 1], shape (N_frames,)
    4. movement/EmotionValenceToken — quantized valence (12 bins), shape (N_frames,)
    5. movement/emotion_scores      — 8-category emotion scores, shape (N_frames, 8)

Each feature is stored on S3 as a separate .npy file at:
    https://dl.fbaipublicfiles.com/seamless_interaction/{label}/{split}/movement/{feature}/{file_id}.npy

This script reads the existing annotated_interactions/ directory structure (created by
download_annotated_interactions.py) to determine which interactions and participants to
process. It also reads filelist.csv from the seamless_interaction repo to look up the
split (train/dev/test) for each file_id, which is needed to construct the S3 URL.

Output:
    For each participant directory, creates:
        emotion_features.npz
    containing NumPy arrays keyed by feature name:
        - emotion_arousal
        - EmotionArousalToken
        - emotion_valence
        - EmotionValenceToken
        - emotion_scores

Note on has_imitator_movement:
    filelist.csv (in the seamless_interaction repo's assets/ directory) contains a column
    called has_imitator_movement with values "0" or "1". This flag indicates whether Meta
    successfully ran their Imitator face-tracking model on that participant's video and
    produced movement feature files (.npy). If the flag is "0", the emotion .npy files do
    not exist on S3 for that participant, so downloading them would fail with HTTP 403.
    This script checks the flag before attempting any downloads and skips participants
    whose flag is "0", reporting the count of skipped participants in the summary.

Requirements:
    - Python 3.8+
    - numpy
    - The annotated_interactions/ directory (from download_annotated_interactions.py)
    - The seamless_interaction GitHub repo (for filelist.csv to look up splits)

Usage:
    python download_emotion_features.py \
        --repo-path /path/to/seamless_interaction \
        --interactions-dir ./annotated_interactions

    Optional arguments:
        --num-workers   Number of parallel download threads (default: 4)
        --dry-run       Print what would be downloaded without actually downloading
        --overwrite     Re-download even if emotion_features.npz already exists
"""

import argparse
import csv
import io
import os
import re
import sys
import tempfile
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# =============================================================================
# Constants
# =============================================================================

S3_BASE_URL = "https://dl.fbaipublicfiles.com/seamless_interaction"

# The 5 emotion features to download, as they appear in the S3 directory structure.
# Each corresponds to a subdirectory under movement/ on S3.
EMOTION_FEATURES = [
    "emotion_arousal",
    "EmotionArousalToken",
    "emotion_valence",
    "EmotionValenceToken",
    "emotion_scores",
]


# =============================================================================
# Helpers
# =============================================================================

def load_filelist_lookup(repo_path: str) -> dict:
    """
    Load filelist.csv and return a dict mapping file_id -> row dict.

    We need this to look up the split (train/dev/test) and label (naturalistic)
    for each file_id, since the S3 URL requires both.
    """
    filelist_path = os.path.join(repo_path, "assets", "filelist.csv")
    if not os.path.exists(filelist_path):
        print(f"ERROR: filelist.csv not found at {filelist_path}")
        print(f"Make sure --repo-path points to the cloned seamless_interaction repository.")
        sys.exit(1)

    with open(filelist_path, newline="") as f:
        return {row["file_id"]: row for row in csv.DictReader(f)}


def extract_file_id_from_participant_dir(interaction_id: str, participant_dir_name: str) -> str:
    """
    Reconstruct the file_id from the interaction ID and participant directory name.

    participant_dir_name is like "participant_a_P0737" -> participant ID is "P0737"
    file_id is like "V00_S1132_I00000333_P0737"
    """
    # Extract participant ID (everything after "participant_a_" or "participant_b_")
    pid = participant_dir_name.split("_", 2)[2]  # "participant_a_P0737" -> "P0737"
    return f"{interaction_id}_{pid}"


def download_npy(url: str) -> np.ndarray:
    """
    Download a .npy file from a URL and return it as a NumPy array.

    Returns None if the file doesn't exist (HTTP 403/404).
    Raises on other errors.
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()

        # Load the .npy data from the bytes in memory
        return np.load(io.BytesIO(data), allow_pickle=False)

    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def download_and_save_emotion_features(
    file_id: str,
    label: str,
    split: str,
    output_dir: str,
    dry_run: bool = False,
) -> tuple:
    """
    Download all 5 emotion features for one participant and save as emotion_features.npz.

    Args:
        file_id: e.g., "V00_S1132_I00000333_P0737"
        label: "naturalistic" or "improvised"
        split: "train", "dev", or "test"
        output_dir: Participant directory to save emotion_features.npz in.
        dry_run: If True, print URLs without downloading.

    Returns:
        Tuple of (file_id, success: bool, message: str, features_found: list)
    """
    output_path = os.path.join(output_dir, "emotion_features.npz")

    if dry_run:
        urls = []
        for feature in EMOTION_FEATURES:
            url = f"{S3_BASE_URL}/{label}/{split}/movement/{feature}/{file_id}.npy"
            urls.append(f"  {feature}: {url}")
        return (file_id, True, "DRY RUN:\n" + "\n".join(urls), EMOTION_FEATURES)

    # Download each feature
    arrays = {}
    features_found = []
    features_missing = []

    for feature in EMOTION_FEATURES:
        url = f"{S3_BASE_URL}/{label}/{split}/movement/{feature}/{file_id}.npy"
        try:
            arr = download_npy(url)
            if arr is not None:
                arrays[feature] = arr
                features_found.append(feature)
            else:
                features_missing.append(feature)
        except Exception as e:
            features_missing.append(feature)
            return (file_id, False, f"ERROR downloading {feature}: {e}", features_found)

    if not arrays:
        return (file_id, False, "No emotion features available on S3", features_found)

    if features_missing:
        msg_parts = [f"Partial: {len(features_found)}/5 features found"]
        msg_parts.append(f"Missing: {', '.join(features_missing)}")
        msg = "; ".join(msg_parts)
    else:
        # Report shapes for verification
        shape_info = ", ".join(
            f"{k}: {v.shape}" for k, v in arrays.items()
        )
        msg = f"OK — 5/5 features ({shape_info})"

    # Save all downloaded arrays into a single NPZ file
    np.savez_compressed(output_path, **arrays)

    return (file_id, True, msg, features_found)


# =============================================================================
# Main orchestration
# =============================================================================

def discover_participants(interactions_dir: str) -> list:
    """
    Walk the annotated_interactions/ directory and discover all participant directories.

    Returns:
        List of tuples: (interaction_id, participant_dir_name, full_participant_path)
    """
    participants = []
    interactions_path = Path(interactions_dir)

    if not interactions_path.exists():
        print(f"ERROR: Interactions directory not found: {interactions_dir}")
        sys.exit(1)

    for interaction_dir in sorted(interactions_path.iterdir()):
        if not interaction_dir.is_dir():
            continue

        interaction_id = interaction_dir.name

        for item in sorted(interaction_dir.iterdir()):
            if item.is_dir() and item.name.startswith("participant_"):
                participants.append((interaction_id, item.name, str(item)))

    return participants


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download emotion features from S3 for all participants in "
            "annotated_interactions/ and save as emotion_features.npz."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the cloned seamless_interaction GitHub repository.",
    )
    parser.add_argument(
        "--interactions-dir",
        default="./annotated_interactions",
        help="Path to the annotated_interactions/ directory. Default: ./annotated_interactions",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download threads. Default: 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if emotion_features.npz already exists.",
    )
    args = parser.parse_args()

    # Validate inputs
    print("=" * 60)
    print("Download Emotion Features for Annotated Interactions")
    print("=" * 60)

    # Load filelist.csv for split/label lookup
    print("\nLoading filelist.csv...")
    filelist_lookup = load_filelist_lookup(args.repo_path)
    print(f"  Loaded {len(filelist_lookup)} file entries")

    # Discover all participant directories
    print(f"\nScanning {args.interactions_dir}...")
    participants = discover_participants(args.interactions_dir)
    print(f"  Found {len(participants)} participant directories across "
          f"{len(set(p[0] for p in participants))} interactions")

    # Build download tasks
    tasks = []
    skipped_existing = 0
    skipped_no_movement = 0

    for interaction_id, participant_dir_name, participant_path in participants:
        # Check if already downloaded
        output_path = os.path.join(participant_path, "emotion_features.npz")
        if os.path.exists(output_path) and not args.overwrite:
            skipped_existing += 1
            continue

        # Reconstruct file_id and look up split/label
        file_id = extract_file_id_from_participant_dir(interaction_id, participant_dir_name)

        if file_id not in filelist_lookup:
            print(f"  WARNING: {file_id} not found in filelist.csv, skipping")
            continue

        row = filelist_lookup[file_id]
        label = row["label"]
        split = row["split"]

        # Check if imitator movement features are available
        if row.get("has_imitator_movement", "0") != "1":
            skipped_no_movement += 1
            continue

        tasks.append((file_id, label, split, participant_path))

    print(f"\n  Tasks to process: {len(tasks)}")
    print(f"  Skipped (already exists): {skipped_existing}")
    print(f"  Skipped (no movement features): {skipped_no_movement}")

    if not tasks:
        print("\nNothing to download. Done.")
        return

    # Execute downloads
    print(f"\n{'=' * 60}")
    print(f"Downloading emotion features ({len(tasks)} participants, {args.num_workers} threads)...")
    print(f"{'=' * 60}\n")

    success_count = 0
    partial_count = 0
    fail_count = 0

    def _do_download(task):
        fid, label, split, out_dir = task
        return download_and_save_emotion_features(fid, label, split, out_dir, dry_run=args.dry_run)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_do_download, task): task for task in tasks}

        for future in as_completed(futures):
            file_id, success, msg, features_found = future.result()
            status = "OK" if success and len(features_found) == 5 else "PARTIAL" if success else "FAIL"

            if success and len(features_found) == 5:
                success_count += 1
            elif success:
                partial_count += 1
            else:
                fail_count += 1

            print(f"  [{status}] {file_id}: {msg}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Full success (5/5 features): {success_count}")
    print(f"  Partial (some features):     {partial_count}")
    print(f"  Failed:                      {fail_count}")
    print(f"  Skipped (already existed):   {skipped_existing}")
    print(f"  Skipped (no movement data):  {skipped_no_movement}")
    print(f"  Output: emotion_features.npz in each participant directory")


if __name__ == "__main__":
    main()
