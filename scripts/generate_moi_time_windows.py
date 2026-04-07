#!/usr/bin/env python3
"""
generate_moi_time_windows.py

For each interaction in annotated_interactions/, reads the 3P-IS annotation files
for both participants, determines who is speaking during each Moment of Interest
(MOI) using timestamps_by_turn.json, and creates a JSON file with pre-MOI, MOI,
and post-MOI time windows.

Output: {interaction}/interaction/pre_and_post_moi/time_windows_pre_post.json

Usage:
    python generate_moi_time_windows.py --input-dir /path/to/annotated_interactions --window 5

    --window: seconds before/after each MOI for the pre/post windows (default: 15)
"""

import argparse
import json
import os
import sys


def load_turns(interaction_dir):
    """Load timestamps_by_turn.json for speaker lookup."""
    path = os.path.join(interaction_dir, "interaction", "timestamps_by_turn.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def determine_speaker(start, end, turns):
    """
    Determine who is speaking during a time window [start, end] using turn data.

    Returns the participant_id of the speaker with the most overlap, or None if
    both participants have equal overlap or no one is speaking.
    """
    overlap_by_pid = {}

    for turn in turns:
        # Calculate overlap between [start, end] and [turn.start, turn.end]
        overlap_start = max(start, turn["start"])
        overlap_end = min(end, turn["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > 0:
            pid = turn["participant_id"]
            overlap_by_pid[pid] = overlap_by_pid.get(pid, 0) + overlap

    if not overlap_by_pid:
        return None

    # Sort by overlap descending
    ranked = sorted(overlap_by_pid.items(), key=lambda x: -x[1])

    if len(ranked) == 1:
        return ranked[0][0]

    # Two speakers: return the one with more overlap, or None if equal
    if ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return None


def load_session_relationship(interaction_dir):
    """Load session_relationship.json, return (relationship, relationship_detail) or (None, None)."""
    path = os.path.join(interaction_dir, "interaction", "session_relationship.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get("relationship"), data.get("relationship_detail")


def process_interaction(interaction_dir, window_seconds):
    """
    Process one interaction: read 3P-IS files for both participants,
    look up speakers, and write time_windows_pre_post.json.
    """
    turns = load_turns(interaction_dir)
    if turns is None:
        return None, "missing timestamps_by_turn.json"

    relationship, relationship_detail = load_session_relationship(interaction_dir)

    # Find participant directories and their 3P-IS files
    entries = []

    for pname in sorted(os.listdir(interaction_dir)):
        if not pname.startswith("participant_"):
            continue
        pdir = os.path.join(interaction_dir, pname)
        if not os.path.isdir(pdir):
            continue

        pid = pname.split("_", 2)[2]

        # Find 3P-IS file
        for fname in os.listdir(pdir):
            if fname.startswith("3P-IS_") and fname.endswith(".json"):
                fpath = os.path.join(pdir, fname)
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ann = json.loads(line)

                        start_moi = ann["start_ts"]
                        end_moi = ann["end_ts"]

                        speaker = determine_speaker(start_moi, end_moi, turns)

                        entries.append({
                            "event_speaker": speaker,
                            "annotated_participant": pid,
                            "start_pre_moi": start_moi - window_seconds,
                            "end_pre_moi": start_moi,
                            "start_moi": start_moi,
                            "end_moi": end_moi,
                            "start_post_moi": end_moi,
                            "end_post_moi": end_moi + window_seconds,
                            "annotation": ann.get("annotation"),
                            "relationship": relationship,
                            "relationship_detail": relationship_detail,
                        })

    # Sort chronologically by MOI start time
    entries.sort(key=lambda e: e["start_moi"])

    # Write output
    out_dir = os.path.join(interaction_dir, "interaction", "pre_and_post_moi")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "time_windows_pre_post.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    return len(entries), None


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre/post MOI time windows from 3P-IS annotations."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Path to the annotated_interactions/ directory.",
    )
    parser.add_argument(
        "--window", type=float, default=15.0,
        help="Seconds before/after each MOI for pre/post windows. Default: 15.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Directory not found: {args.input_dir}")
        sys.exit(1)

    interactions = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("V")
    ])

    print(f"Processing {len(interactions)} interactions (window={args.window}s)...\n")

    total_entries = 0
    errors = []

    for i, name in enumerate(interactions, 1):
        idir = os.path.join(args.input_dir, name)
        count, error = process_interaction(idir, args.window)

        if error:
            errors.append((name, error))
            print(f"  [{i}/{len(interactions)}] {name}: ERROR - {error}")
        else:
            total_entries += count
            print(f"  [{i}/{len(interactions)}] {name}: {count} entries")

    print(f"\nDone. {len(interactions) - len(errors)}/{len(interactions)} succeeded, "
          f"{total_entries} total entries written.")
    if errors:
        for name, err in errors:
            print(f"  ERROR: {name}: {err}")


if __name__ == "__main__":
    main()
