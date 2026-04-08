#!/usr/bin/env python3
"""
build_moi_table.py

Traverses all moi_emotion_summaries.json files in annotated_interactions/ and
produces a single flat CSV with one row per MOI.

Output columns:
    interaction_id, annotated_participant, non_annotated_participant,
    emotion_features_present, annotation,
    start_ts, end_ts, start_idx, end_idx, n_frames,
    annotated_mean_valence, annotated_mean_arousal,
    non_annotated_mean_valence, non_annotated_mean_arousal

Emotion scores (8 Ekman categories per participant) are excluded from the CSV
to keep it readable in a spreadsheet. Use the per-interaction JSON files or
the raw emotion_features.npz if you need them.

Usage:
    python build_moi_table.py --interactions-dir ./annotated_interactions

    Optional arguments:
        --output       Output CSV path. Default: ./moi_valence_arousal_table.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


CSV_COLUMNS = [
    "interaction_id",
    "annotated_participant",
    "non_annotated_participant",
    "emotion_features_present",
    "annotation",
    "start_ts",
    "end_ts",
    "start_idx",
    "end_idx",
    "n_frames",
    "annotated_mean_valence",
    "annotated_mean_arousal",
    "non_annotated_mean_valence",
    "non_annotated_mean_arousal",
]

# Map from JSON field name to CSV column name
FIELD_RENAMES = {
    "imitator_emotion_features_present": "emotion_features_present",
}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate moi_emotion_summaries.json files into a single CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--interactions-dir",
        default="./annotated_interactions",
        help="Path to the annotated_interactions/ directory. Default: ./annotated_interactions",
    )
    parser.add_argument(
        "--output",
        default="./moi_valence_arousal_table.csv",
        help="Output CSV file path. Default: ./moi_valence_arousal_table.csv",
    )
    args = parser.parse_args()

    interactions_path = Path(args.interactions_dir)
    if not interactions_path.exists():
        print(f"ERROR: Interactions directory not found: {args.interactions_dir}")
        sys.exit(1)

    # Discover all moi_emotion_summaries.json files
    interaction_dirs = sorted([
        d for d in interactions_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    rows = []
    found_count = 0
    missing_count = 0

    for idir in interaction_dirs:
        json_path = idir / "interaction" / "moi_emotion_summaries.json"
        if not json_path.exists():
            missing_count += 1
            continue

        found_count += 1
        with open(json_path) as f:
            mois = json.load(f)

        for moi in mois:
            row = {}
            for col in CSV_COLUMNS:
                # Check if this column is a rename of a JSON field
                json_key = next(
                    (k for k, v in FIELD_RENAMES.items() if v == col),
                    col,
                )
                row[col] = moi.get(json_key)
            rows.append(row)

    if not rows:
        print("No MOI records found. Have you run extract_moi_summaries.py?")
        sys.exit(1)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} MOI records from {found_count} interactions to {args.output}")
    if missing_count:
        print(f"  ({missing_count} interactions had no moi_emotion_summaries.json)")


if __name__ == "__main__":
    main()
