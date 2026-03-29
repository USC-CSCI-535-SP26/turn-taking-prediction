#!/usr/bin/env python3
"""
generate_moi_turns.py

For each interaction in annotated_interactions/, identifies the non-annotated
participant's nearest turn before and after each Moment of Interest (MOI) using
timestamps_by_turn.json.

Pre turn:  last turn of the non-annotated participant that starts before start_moi.
Post turn: first turn of the non-annotated participant that ends after end_moi.

Overlap flags indicate whether the pre/post turn overlaps with the MOI window
itself (NOT the annotated participant's turn).

Output: {interaction}/interaction/pre_and_post_moi/turns_pre_post.json

Usage:
    python generate_moi_turns.py --input-dir /path/to/annotated_interactions
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
    Determine who is speaking during [start, end] using turn data.
    Returns participant_id with most overlap, or None if tied/no overlap.
    """
    overlap_by_pid = {}
    for turn in turns:
        overlap_start = max(start, turn["start"])
        overlap_end = min(end, turn["end"])
        overlap = max(0, overlap_end - overlap_start)
        if overlap > 0:
            pid = turn["participant_id"]
            overlap_by_pid[pid] = overlap_by_pid.get(pid, 0) + overlap

    if not overlap_by_pid:
        return None

    ranked = sorted(overlap_by_pid.items(), key=lambda x: -x[1])
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return None


def get_participant_ids(interaction_dir):
    """Return sorted list of participant IDs from directory names."""
    pids = []
    for name in sorted(os.listdir(interaction_dir)):
        if name.startswith("participant_") and os.path.isdir(
            os.path.join(interaction_dir, name)
        ):
            pids.append(name.split("_", 2)[2])
    return pids


def find_pre_post_turns(start_moi, end_moi, non_ann_turns):
    """
    Find the nearest pre and post turns for the non-annotated participant.

    Pre:  last turn whose start < start_moi
    Post: first turn whose end > end_moi

    Returns (pre_turn_or_None, pre_overlap, post_turn_or_None, post_overlap).
    """
    pre_turn = None
    for t in non_ann_turns:
        if t["start"] < start_moi:
            pre_turn = t  # keeps updating; last one wins (turns are chronological)
        else:
            break

    post_turn = None
    for t in non_ann_turns:
        if t["end"] > end_moi:
            post_turn = t
            break

    # Overlap with the MOI window itself
    pre_overlap = pre_turn is not None and pre_turn["end"] > start_moi
    post_overlap = post_turn is not None and post_turn["start"] < end_moi

    return pre_turn, pre_overlap, post_turn, post_overlap


def process_interaction(interaction_dir):
    """
    Process one interaction: read 3P-IS files, find non-annotated participant's
    nearest pre/post turns around each MOI, and write turns_pre_post.json.
    """
    turns = load_turns(interaction_dir)
    if turns is None:
        return None, "missing timestamps_by_turn.json"

    pids = get_participant_ids(interaction_dir)
    if len(pids) != 2:
        return None, f"expected 2 participants, found {len(pids)}"

    # Index turns by participant
    turns_by_pid = {pid: [] for pid in pids}
    for t in turns:
        pid = t["participant_id"]
        if pid in turns_by_pid:
            turns_by_pid[pid].append(t)

    entries = []

    for pname in sorted(os.listdir(interaction_dir)):
        if not pname.startswith("participant_"):
            continue
        pdir = os.path.join(interaction_dir, pname)
        if not os.path.isdir(pdir):
            continue

        annotated_pid = pname.split("_", 2)[2]
        non_annotated_pid = [p for p in pids if p != annotated_pid][0]
        non_ann_turns = turns_by_pid[non_annotated_pid]

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

                        pre_turn, pre_overlap, post_turn, post_overlap = (
                            find_pre_post_turns(start_moi, end_moi, non_ann_turns)
                        )

                        entry = {
                            "event_speaker": speaker,
                            "annotated_participant": annotated_pid,
                            "non_annotated_participant": non_annotated_pid,
                            "start_pre_moi": pre_turn["start"] if pre_turn else None,
                            "end_pre_moi": pre_turn["end"] if pre_turn else None,
                            "pre_overlap": pre_overlap,
                            "start_moi": start_moi,
                            "end_moi": end_moi,
                            "start_post_moi": post_turn["start"] if post_turn else None,
                            "end_post_moi": post_turn["end"] if post_turn else None,
                            "post_overlap": post_overlap,
                        }
                        entries.append(entry)

    entries.sort(key=lambda e: e["start_moi"])

    out_dir = os.path.join(interaction_dir, "interaction", "pre_and_post_moi")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "turns_pre_post.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    return len(entries), None


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre/post MOI turn windows from non-annotated participant's turns."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the annotated_interactions/ directory.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Directory not found: {args.input_dir}")
        sys.exit(1)

    interactions = sorted(
        [
            d
            for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("V")
        ]
    )

    print(f"Processing {len(interactions)} interactions...\n")

    total_entries = 0
    errors = []

    for i, name in enumerate(interactions, 1):
        idir = os.path.join(args.input_dir, name)
        count, error = process_interaction(idir)

        if error:
            errors.append((name, error))
            print(f"  [{i}/{len(interactions)}] {name}: ERROR - {error}")
        else:
            total_entries += count
            print(f"  [{i}/{len(interactions)}] {name}: {count} entries")

    print(
        f"\nDone. {len(interactions) - len(errors)}/{len(interactions)} succeeded, "
        f"{total_entries} total entries written."
    )
    if errors:
        for name, err in errors:
            print(f"  ERROR: {name}: {err}")


if __name__ == "__main__":
    main()
