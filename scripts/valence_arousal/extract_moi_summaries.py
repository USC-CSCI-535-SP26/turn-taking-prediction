#!/usr/bin/env python3
"""
extract_moi_summaries.py

For each interaction in annotated_interactions/, reads the 3P-IS annotation files
(which define Moments of Interest) and the emotion_features.npz files for both
participants, then produces a single JSON file per interaction containing one record
per MOI with frame indices, mean valence/arousal, and mean Ekman emotion scores for
both the annotated and non-annotated participant.

Index alignment
---------------
Emotion features are sampled at 30 Hz. 3P-IS timestamps (start_ts, end_ts) are
integers in seconds measured from the start of the interaction. To convert:

    start_idx = start_ts * 30   (inclusive)
    end_idx   = end_ts   * 30   (exclusive, following Python slice convention)

Both participants in the same interaction have emotion feature arrays of identical
length N (verified empirically across the dataset), so the same indices can be used
to slice either participant's arrays. The script clamps end_idx to min(end_ts * 30, N)
in case an MOI extends slightly past the end of the feature array.

Ekman emotion scores
--------------------
The emotion_scores array has shape (N, 8). The 8 columns correspond to:
    0: neutral, 1: happy, 2: sad, 3: surprise,
    4: fear, 5: disgust, 6: anger, 7: contempt

These are raw logits from the Imitator model's expression encoder, NOT softmax
probabilities. Values can be negative and can exceed 1. The mean across frames
is still a valid summary statistic for comparing MOIs, but the values should not
be interpreted as probabilities or expected to sum to 1.

Output
------
For each interaction, writes:
    {interaction}/interaction/moi_emotion_summaries.json

Each entry in the JSON array represents one MOI and contains:
    - annotated_participant        participant ID whose 3P-IS file this came from
    - non_annotated_participant    the other participant in the interaction
    - interaction_id               interaction directory name (e.g., V00_S1132_I00000333)
    - imitator_emotion_features_present   true if emotion_features.npz exists for both participants
    - annotation                   free-text 3P-IS annotation
    - start_ts                     MOI start time in seconds (integer)
    - end_ts                       MOI end time in seconds (integer)
    - start_idx                    first frame index at 30 Hz (inclusive)
    - end_idx                      last frame index at 30 Hz (exclusive)
    - n_frames                     number of frames in the window
    - annotated_mean_valence       mean emotion_valence over the window (annotated participant)
    - annotated_mean_arousal       mean emotion_arousal over the window (annotated participant)
    - annotated_mean_emotion_scores   dict of 8 Ekman category means (annotated participant)
    - non_annotated_mean_valence   mean emotion_valence over the window (non-annotated participant)
    - non_annotated_mean_arousal   mean emotion_arousal over the window (non-annotated participant)
    - non_annotated_mean_emotion_scores   dict of 8 Ekman category means (non-annotated participant)

Fields are null when the corresponding participant's emotion_features.npz is missing.

Usage
-----
    python extract_moi_summaries.py --interactions-dir ./annotated_interactions

    Optional arguments:
        --both-required   Only process interactions where both participants have
                          emotion_features.npz (54 of 100). Default: process all.
        --overwrite       Overwrite existing moi_emotion_summaries.json files.
        --dry-run         Print what would be processed without writing any files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# Constants
# =============================================================================

FRAME_RATE = 30  # Emotion features are sampled at 30 Hz

# The 8 columns of emotion_scores, in order. These are raw logits from the
# Imitator expression encoder (not softmax probabilities).
EKMAN_CATEGORIES = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]


# =============================================================================
# Helpers
# =============================================================================

def load_3p_is(participant_dir: str, file_id: str) -> list:
    """
    Load the 3P-IS JSONL file for a participant.

    Returns a list of dicts, each with keys: annotation, start_ts, end_ts.
    Returns an empty list if the file doesn't exist.
    """
    path = os.path.join(participant_dir, f"3P-IS_{file_id}.json")
    if not os.path.exists(path):
        return []

    annotations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations


def load_emotion_features(participant_dir: str) -> dict | None:
    """
    Load emotion_features.npz for a participant.

    Returns a dict of arrays keyed by feature name, or None if the file
    doesn't exist.
    """
    path = os.path.join(participant_dir, "emotion_features.npz")
    if not os.path.exists(path):
        return None

    npz = np.load(path, allow_pickle=False)
    return {key: npz[key] for key in npz.files}


def compute_moi_summary(
    features: dict | None,
    start_ts: int,
    end_ts: int,
) -> dict:
    """
    Compute summary statistics for one MOI window from one participant's
    emotion features.

    Args:
        features: Dict of numpy arrays from emotion_features.npz, or None
                  if the participant has no emotion features.
        start_ts: MOI start time in seconds.
        end_ts:   MOI end time in seconds.

    Returns:
        Dict with keys: mean_valence, mean_arousal, mean_emotion_scores,
        start_idx, end_idx, n_frames.  All values are None if features is None.
    """
    if features is None:
        return {
            "mean_valence": None,
            "mean_arousal": None,
            "mean_emotion_scores": None,
        }

    # Convert timestamps to frame indices
    start_idx = start_ts * FRAME_RATE
    end_idx = end_ts * FRAME_RATE

    # Clamp end_idx to array length
    n_frames_available = features["emotion_valence"].shape[0]
    end_idx = min(end_idx, n_frames_available)

    # Guard against empty or invalid slices
    if start_idx >= end_idx:
        return {
            "mean_valence": None,
            "mean_arousal": None,
            "mean_emotion_scores": None,
        }

    # Slice and compute means
    valence_slice = features["emotion_valence"][start_idx:end_idx]
    arousal_slice = features["emotion_arousal"][start_idx:end_idx]
    scores_slice = features["emotion_scores"][start_idx:end_idx]

    mean_valence = float(np.mean(valence_slice))
    mean_arousal = float(np.mean(arousal_slice))

    # Mean across frames for each of the 8 Ekman categories
    mean_scores = np.mean(scores_slice, axis=0)  # shape (8,)
    mean_emotion_scores = {
        category: round(float(mean_scores[i]), 6)
        for i, category in enumerate(EKMAN_CATEGORIES)
    }

    return {
        "mean_valence": round(mean_valence, 6),
        "mean_arousal": round(mean_arousal, 6),
        "mean_emotion_scores": mean_emotion_scores,
    }


def extract_participant_id(participant_dir_name: str) -> str:
    """
    Extract participant ID from directory name.

    "participant_a_P0737" -> "P0737"
    """
    return participant_dir_name.split("_", 2)[2]


def reconstruct_file_id(interaction_id: str, participant_id: str) -> str:
    """
    Reconstruct the file_id from interaction ID and participant ID.

    "V00_S1132_I00000333" + "P0737" -> "V00_S1132_I00000333_P0737"
    """
    return f"{interaction_id}_{participant_id}"


# =============================================================================
# Per-interaction processing
# =============================================================================

def process_interaction(
    interaction_dir: str,
    interaction_id: str,
    both_required: bool,
    dry_run: bool,
    overwrite: bool,
) -> tuple:
    """
    Process one interaction: load 3P-IS annotations and emotion features for
    both participants, compute per-MOI summaries, and write the output JSON.

    Returns:
        Tuple of (interaction_id, status: str, n_mois: int, message: str)
        where status is "OK" or "SKIP".
    """
    output_path = os.path.join(interaction_dir, "interaction", "moi_emotion_summaries.json")

    # Check for existing output
    if os.path.exists(output_path) and not overwrite:
        return (interaction_id, "SKIP", 0, "already exists (use --overwrite)")

    # Discover participant directories
    participant_dirs = sorted([
        d for d in Path(interaction_dir).iterdir()
        if d.is_dir() and d.name.startswith("participant_")
    ])

    if len(participant_dirs) != 2:
        return (interaction_id, "SKIP", 0,
                f"expected 2 participant dirs, found {len(participant_dirs)}")

    # Build participant info
    participants = {}
    for pdir in participant_dirs:
        role = pdir.name.split("_")[1]  # "a" or "b"
        pid = extract_participant_id(pdir.name)
        file_id = reconstruct_file_id(interaction_id, pid)
        participants[role] = {
            "id": pid,
            "dir": str(pdir),
            "dir_name": pdir.name,
            "file_id": file_id,
        }

    # Load emotion features for both participants
    for role in participants:
        participants[role]["features"] = load_emotion_features(participants[role]["dir"])

    a_has = participants["a"]["features"] is not None
    b_has = participants["b"]["features"] is not None
    both_have = a_has and b_has

    if both_required and not both_have:
        missing = []
        if not a_has:
            missing.append(participants["a"]["id"])
        if not b_has:
            missing.append(participants["b"]["id"])
        return (interaction_id, "SKIP", 0,
                f"--both-required: missing emotion features for {', '.join(missing)}")

    # Load 3P-IS annotations for both participants
    all_mois = []
    for role in ("a", "b"):
        p = participants[role]
        other_role = "b" if role == "a" else "a"
        other_p = participants[other_role]

        annotations = load_3p_is(p["dir"], p["file_id"])
        for ann in annotations:
            all_mois.append({
                "annotated_role": role,
                "non_annotated_role": other_role,
                "annotated_participant": p["id"],
                "non_annotated_participant": other_p["id"],
                "annotation": ann["annotation"],
                "start_ts": ann["start_ts"],
                "end_ts": ann["end_ts"],
            })

    # Sort by start time
    all_mois.sort(key=lambda x: x["start_ts"])

    if not all_mois:
        return (interaction_id, "SKIP", 0, "no 3P-IS annotations found")

    if dry_run:
        return (interaction_id, "DRY_RUN", len(all_mois),
                f"{len(all_mois)} MOIs would be processed")

    # Compute summaries for each MOI
    output_records = []
    for moi in all_mois:
        start_ts = moi["start_ts"]
        end_ts = moi["end_ts"]

        # Compute frame indices (same for both participants)
        start_idx = start_ts * FRAME_RATE
        end_idx = end_ts * FRAME_RATE

        # Clamp end_idx using whichever participant has features
        annotated_features = participants[moi["annotated_role"]]["features"]
        non_annotated_features = participants[moi["non_annotated_role"]]["features"]

        # Determine actual end_idx clamped to array bounds
        clamped_end_idx = end_idx
        if annotated_features is not None:
            n = annotated_features["emotion_valence"].shape[0]
            clamped_end_idx = min(clamped_end_idx, n)
        if non_annotated_features is not None:
            n = non_annotated_features["emotion_valence"].shape[0]
            clamped_end_idx = min(clamped_end_idx, n)

        n_frames = max(0, clamped_end_idx - start_idx)

        # Compute summaries for each participant
        ann_summary = compute_moi_summary(annotated_features, start_ts, end_ts)
        non_ann_summary = compute_moi_summary(non_annotated_features, start_ts, end_ts)

        record = {
            "interaction_id": interaction_id,
            "annotated_participant": moi["annotated_participant"],
            "non_annotated_participant": moi["non_annotated_participant"],
            "imitator_emotion_features_present": both_have,
            "annotation": moi["annotation"],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_idx": start_idx,
            "end_idx": clamped_end_idx,
            "n_frames": n_frames,
            "annotated_mean_valence": ann_summary["mean_valence"],
            "annotated_mean_arousal": ann_summary["mean_arousal"],
            "annotated_mean_emotion_scores": ann_summary["mean_emotion_scores"],
            "non_annotated_mean_valence": non_ann_summary["mean_valence"],
            "non_annotated_mean_arousal": non_ann_summary["mean_arousal"],
            "non_annotated_mean_emotion_scores": non_ann_summary["mean_emotion_scores"],
        }
        output_records.append(record)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_records, f, indent=2)

    detail = f"{len(output_records)} MOIs"
    if not both_have:
        detail += " (no emotion features)"

    return (interaction_id, "OK", len(output_records), detail)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-MOI emotion summaries for all interactions in "
            "annotated_interactions/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--interactions-dir",
        default="./annotated_interactions",
        help="Path to the annotated_interactions/ directory. Default: ./annotated_interactions",
    )
    parser.add_argument(
        "--both-required",
        action="store_true",
        help=(
            "Only process interactions where both participants have "
            "emotion_features.npz. Default: process all interactions."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing moi_emotion_summaries.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without writing any files.",
    )
    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("Extract Per-MOI Emotion Summaries")
    print("=" * 60)

    interactions_path = Path(args.interactions_dir)
    if not interactions_path.exists():
        print(f"\nERROR: Interactions directory not found: {args.interactions_dir}")
        sys.exit(1)

    # Discover interaction directories
    interaction_dirs = sorted([
        d for d in interactions_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"\nFound {len(interaction_dirs)} interaction directories")
    if args.both_required:
        print("  --both-required: will skip interactions missing emotion features")
    if args.dry_run:
        print("  --dry-run: no files will be written")

    # Process each interaction
    ok_count = 0
    skip_count = 0
    total_mois = 0

    print(f"\n{'=' * 60}")
    print("Processing interactions...")
    print(f"{'=' * 60}\n")

    for idir in interaction_dirs:
        interaction_id, status, n_mois, msg = process_interaction(
            str(idir),
            idir.name,
            both_required=args.both_required,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

        if status in ("OK", "DRY_RUN"):
            ok_count += 1
            total_mois += n_mois
        else:
            skip_count += 1

        print(f"  [{status:>7}] {interaction_id}: {msg}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Interactions processed: {ok_count}")
    print(f"  Interactions skipped:   {skip_count}")
    print(f"  Total MOI records:      {total_mois}")
    print(f"  Output: interaction/moi_emotion_summaries.json per interaction")


if __name__ == "__main__":
    main()
