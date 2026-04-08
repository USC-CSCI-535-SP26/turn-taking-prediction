#!/usr/bin/env python3
"""
Generate timestamps_by_turn.json for each interaction in annotated_interactions/.

For each interaction, reads both participants' VAD JSONL files, merges consecutive
same-participant VAD segments into "turns" (a turn boundary occurs only when the
other participant speaks), then interleaves them chronologically and flags overlap.
"""

import json
import os
import sys


def load_vad(path):
    """Load VAD JSONL file, return list of {"start": float, "end": float}."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def segments_overlap(start, end, other_segments):
    """Check if any segment in other_segments overlaps with [start, end]."""
    for seg in other_segments:
        if seg["start"] < end and seg["end"] > start:
            return True
    return False


def merge_into_turns(segments_a, segments_b):
    """
    Merge each participant's VAD segments into turns. A turn for participant X
    is a maximal sequence of X's consecutive VAD segments where no gap between
    them contains speech from the other participant Y.

    A gap between X's segment_i (end) and segment_i+1 (start) "contains Y speech"
    if any of Y's segments overlaps with [segment_i.end, segment_i+1.start].

    Returns list of turns sorted by start time, each:
        {"participant_id": str, "start": float, "end": float, "overlapping": bool}
    """
    def _merge_segments(my_segs, other_segs, pid):
        """Merge one participant's VAD segments into turns."""
        if not my_segs:
            return []

        turns = []
        turn_start = my_segs[0]["start"]
        turn_end = my_segs[0]["end"]

        for i in range(1, len(my_segs)):
            gap_start = turn_end
            gap_end = my_segs[i]["start"]

            # Check if the other participant speaks in this gap
            gap_has_other = False
            if gap_end > gap_start:
                for seg in other_segs:
                    # Does this other-participant segment overlap with the gap?
                    if seg["start"] < gap_end and seg["end"] > gap_start:
                        gap_has_other = True
                        break

            if gap_has_other:
                # End current turn, start new one
                turns.append({"participant_id": pid, "start": turn_start, "end": turn_end})
                turn_start = my_segs[i]["start"]
                turn_end = my_segs[i]["end"]
            else:
                # Extend current turn
                turn_end = my_segs[i]["end"]

        # Don't forget the last turn
        turns.append({"participant_id": pid, "start": turn_start, "end": turn_end})
        return turns

    turns_a = _merge_segments(segments_a, segments_b, "A")
    turns_b = _merge_segments(segments_b, segments_a, "B")

    # Combine and sort chronologically by start time
    all_turns = turns_a + turns_b
    all_turns.sort(key=lambda t: t["start"])

    # Now check overlap: for each turn, does the OTHER participant have any
    # VAD segment overlapping with [turn.start, turn.end]?
    for turn in all_turns:
        other_segs = segments_b if turn["participant_id"] == "A" else segments_a
        turn["overlapping"] = segments_overlap(turn["start"], turn["end"], other_segs)

    return all_turns


def process_interaction(interaction_dir):
    """
    Process one interaction directory. Find both participant dirs, load their
    VAD files, merge into turns, write timestamps_by_turn.json.
    """
    # Find participant directories
    participant_dirs = {}
    for name in sorted(os.listdir(interaction_dir)):
        if name.startswith("participant_") and os.path.isdir(os.path.join(interaction_dir, name)):
            # Extract participant ID: "participant_a_P0737" -> "P0737"
            pid = name.split("_", 2)[2]
            role = name.split("_")[1]  # "a" or "b"
            participant_dirs[role] = {
                "pid": pid,
                "dir": os.path.join(interaction_dir, name),
            }

    if "a" not in participant_dirs or "b" not in participant_dirs:
        return None, "Missing participant directories"

    # Find and load VAD files
    vad_a = None
    vad_b = None
    for role, info in participant_dirs.items():
        for fname in os.listdir(info["dir"]):
            if fname.startswith("vad_") and fname.endswith(".jsonl"):
                vad_path = os.path.join(info["dir"], fname)
                if role == "a":
                    vad_a = load_vad(vad_path)
                else:
                    vad_b = load_vad(vad_path)

    if vad_a is None or vad_b is None:
        return None, "Missing VAD file(s)"

    # Merge into turns
    turns = merge_into_turns(vad_a, vad_b)

    # Replace "A"/"B" with actual participant IDs
    pid_map = {"A": participant_dirs["a"]["pid"], "B": participant_dirs["b"]["pid"]}
    for turn in turns:
        turn["participant_id"] = pid_map[turn["participant_id"]]

    # Write to interaction/timestamps_by_turn.json
    output_dir = os.path.join(interaction_dir, "interaction")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "timestamps_by_turn.json")
    with open(output_path, "w") as f:
        json.dump(turns, f, indent=2)

    return len(turns), None


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    interactions = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("V")
    ])

    print(f"Processing {len(interactions)} interactions...\n")

    total_turns = 0
    errors = []

    for i, name in enumerate(interactions, 1):
        interaction_dir = os.path.join(input_dir, name)
        num_turns, error = process_interaction(interaction_dir)

        if error:
            errors.append((name, error))
            print(f"  [{i}/{len(interactions)}] {name}: ERROR - {error}")
        else:
            total_turns += num_turns
            print(f"  [{i}/{len(interactions)}] {name}: {num_turns} turns")

    print(f"\nDone. {len(interactions) - len(errors)}/{len(interactions)} succeeded, "
          f"{total_turns} total turns written.")
    if errors:
        print(f"\nErrors:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
