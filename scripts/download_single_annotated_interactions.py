#!/usr/bin/env python3
"""
download_single_annotated_interactions.py

Downloads the 343 naturalistic interactions from the Seamless Interaction dataset
where exactly ONE of the two participants has third-party (3P) annotations. For
each interaction, downloads interaction-level metadata, per-participant data
(3P annotations for the annotated participant, VAD for both), and generates
derived files (turn structure, pre/post MOI windows, video viewer).

This script mirrors the directory structure of both_annotated_interactions/
created by download_annotated_interactions.py, with these differences:
    - No emotion_features.npz (not downloaded)
    - No moi_emotion_summaries.json (depends on emotion features)
    - No valence/valence_engaged fields in pre_and_post_moi files
    - video_viewer.html is generated in each interaction/ directory
    - Only the annotated participant has 3P-IS/3P-R/3P-V files

Requirements:
    - Python 3.8+
    - The seamless_interaction GitHub repo must be cloned locally, with the
      assets/ directory containing filelist.csv, interactions.csv,
      participants.csv, and relationships.csv.
    - Internet access to download from Meta's public S3 bucket at
      https://dl.fbaipublicfiles.com/seamless_interaction/

Usage:
    python download_single_annotated_interactions.py --repo-path /path/to/seamless_interaction

    Optional arguments:
        --output-dir    Directory to create single_annotated_interactions/ in (default: ..)
        --include-wav   Also download per-participant audio (.wav) files
        --num-workers   Number of parallel download threads (default: 4)
        --dry-run       Print what would be downloaded without actually downloading
        --window        Seconds before/after each MOI for pre/post windows (default: 15)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Imports from sibling scripts
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_annotated_interactions import (
    load_filelist,
    extract_interaction_id,
    extract_participant_id,
    build_interaction_metadata,
    load_interactions_csv,
    load_relationships_csv,
    load_participants_csv,
    download_participant_files,
    generate_video_viewer,
)
from generate_timestamps_by_turn import (
    process_interaction as generate_timestamps_by_turn,
)
from generate_moi_time_windows import (
    process_interaction as generate_time_windows,
)
from generate_moi_turns import (
    process_interaction as generate_turns_pre_post,
)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the 343 naturalistic interactions from the Seamless Interaction "
            "dataset where exactly one participant has third-party (3P) annotations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help=(
            "Path to the cloned seamless_interaction GitHub repository. "
            "Must contain the assets/ directory with filelist.csv, "
            "interactions.csv, participants.csv, and relationships.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="..",
        help=(
            "Directory to create single_annotated_interactions/ in. "
            "Default: parent of scripts/ (i.e. the project root)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download threads. Default: 4.",
    )
    parser.add_argument(
        "--include-wav",
        action="store_true",
        help="Also download per-participant audio (.wav) files. Excluded by default.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=15.0,
        help="Seconds before/after each MOI for pre/post time windows. Default: 15.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    # Validate repo path
    assets_dir = os.path.join(args.repo_path, "assets")
    required_files = ["filelist.csv", "interactions.csv", "participants.csv", "relationships.csv"]
    for fname in required_files:
        if not os.path.exists(os.path.join(assets_dir, fname)):
            print(f"ERROR: {fname} not found in {assets_dir}")
            print(f"Make sure --repo-path points to the cloned seamless_interaction repository.")
            sys.exit(1)

    print("=" * 60)
    print("Seamless Interaction: Download Single-Annotated Interactions")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Identify the 343 single-annotated naturalistic interactions
    # -------------------------------------------------------------------------
    print("\nStep 1: Identifying target interactions from filelist.csv...")
    filelist_rows = load_filelist(args.repo_path)
    print(f"  Total file entries in filelist.csv: {len(filelist_rows)}")

    # Group ALL naturalistic entries by interaction ID
    nat = [r for r in filelist_rows if r["label"] == "naturalistic"]
    interactions_by_id = defaultdict(list)
    for r in nat:
        iid = extract_interaction_id(r["file_id"])
        interactions_by_id[iid].append(r)

    # Find interactions where exactly 1 of 2 participants has 3P annotations
    single_annotated = {}
    for iid, entries in interactions_by_id.items():
        if len(entries) != 2:
            continue
        annotated = [e for e in entries if e["has_annotation_3p"] == "1"]
        if len(annotated) == 1:
            file_ids = sorted([e["file_id"] for e in entries])
            single_annotated[iid] = {
                "file_ids": file_ids,
                "annotated_file_id": annotated[0]["file_id"],
            }

    print(f"  Naturalistic files with 3P annotations: {sum(1 for r in nat if r['has_annotation_3p'] == '1')}")
    print(f"  Interactions with exactly ONE participant annotated: {len(single_annotated)}")

    if len(single_annotated) != 343:
        print(f"  WARNING: Expected 343 interactions, found {len(single_annotated)}.")
        print(f"  This may indicate a dataset version change.")

    # -------------------------------------------------------------------------
    # Step 2: Create directories, write metadata, queue downloads
    # -------------------------------------------------------------------------
    print(f"\nStep 2: Creating directories and writing metadata...")

    interactions_lookup = load_interactions_csv(args.repo_path)
    relationships_lookup = load_relationships_csv(args.repo_path)
    participants_lookup = load_participants_csv(args.repo_path)

    filelist_by_id = {row["file_id"]: row for row in filelist_rows}

    root_dir = os.path.join(args.output_dir, "single_annotated_interactions")
    os.makedirs(root_dir, exist_ok=True)

    total = len(single_annotated)
    download_tasks = []

    for idx, (interaction_id, info) in enumerate(sorted(single_annotated.items()), 1):
        file_ids = info["file_ids"]
        annotated_fid = info["annotated_file_id"]

        # Determine participant_a and participant_b (alphabetical by participant ID)
        pid_a = extract_participant_id(file_ids[0])
        pid_b = extract_participant_id(file_ids[1])

        # Create directory structure
        interaction_dir = os.path.join(root_dir, interaction_id)
        meta_dir = os.path.join(interaction_dir, "interaction")
        dir_a = os.path.join(interaction_dir, f"participant_a_{pid_a}")
        dir_b = os.path.join(interaction_dir, f"participant_b_{pid_b}")

        for d in [meta_dir, dir_a, dir_b]:
            os.makedirs(d, exist_ok=True)

        # Write interaction-level metadata
        metadata = build_interaction_metadata(
            interaction_id, file_ids, filelist_rows,
            interactions_lookup, relationships_lookup, participants_lookup,
        )

        for filename, data in [
            ("interaction_metadata.json", metadata["interaction_metadata"]),
            ("session_relationship.json", metadata["session_relationship"]),
            ("participants_metadata.json", metadata["participants_metadata"]),
            ("filelist_entries.json", metadata["filelist_entries"]),
        ]:
            filepath = os.path.join(meta_dir, filename)
            if not args.dry_run:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)

        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] metadata written")

        # Queue participant downloads
        for fid in file_ids:
            pid = extract_participant_id(fid)
            split = filelist_by_id[fid]["split"]
            is_annotated = (fid == annotated_fid)
            pdir = dir_a if fid == file_ids[0] else dir_b
            plabel = f"participant_a_{pid}" if fid == file_ids[0] else f"participant_b_{pid}"

            download_tasks.append((
                fid, "naturalistic", split, pdir,
                args.include_wav, is_annotated, f"  {plabel}",
            ))

    # -------------------------------------------------------------------------
    # Step 3: Download participant files in parallel
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Step 3: Downloading participant files ({len(download_tasks)} participants, {args.num_workers} threads)...")
    print(f"{'=' * 60}\n")

    success_count = 0
    fail_count = 0
    skip_count = 0

    def _do_download(task):
        fid, label, split, out_dir, inc_wav, is_annotated, prefix = task
        results = download_participant_files(
            fid, label, split, out_dir,
            include_wav=inc_wav,
            include_annotations=is_annotated,
            dry_run=args.dry_run,
        )
        return prefix, results

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_do_download, task): task for task in download_tasks}

        for future in as_completed(futures):
            prefix, results = future.result()
            for filename, success, msg in results:
                if success:
                    success_count += 1
                elif "NOT FOUND" in msg:
                    skip_count += 1
                else:
                    fail_count += 1
                print(f"{prefix}/{filename}: {msg}")

    print(f"\n  Files OK:      {success_count}")
    print(f"  Files skipped: {skip_count} (not available on server)")
    print(f"  Files failed:  {fail_count}")

    if args.dry_run:
        print("\nDry run — skipping derived file generation.")
        return

    # -------------------------------------------------------------------------
    # Step 4: Generate derived files for each interaction
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Step 4: Generating derived files (turns, MOI windows, video viewers)...")
    print(f"{'=' * 60}\n")

    turns_ok = 0
    tw_ok = 0
    tp_ok = 0
    vv_ok = 0
    errors = []

    for idx, interaction_id in enumerate(sorted(single_annotated.keys()), 1):
        interaction_dir = os.path.join(root_dir, interaction_id)

        # 4a. timestamps_by_turn.json (from VAD files)
        count, err = generate_timestamps_by_turn(interaction_dir)
        if err:
            errors.append((interaction_id, "timestamps_by_turn", err))
        else:
            turns_ok += 1

        # 4b. pre_and_post_moi/time_windows_pre_post.json
        count, err = generate_time_windows(interaction_dir, args.window)
        if err:
            errors.append((interaction_id, "time_windows", err))
        else:
            tw_ok += 1

        # 4c. pre_and_post_moi/turns_pre_post.json
        count, err = generate_turns_pre_post(interaction_dir)
        if err:
            errors.append((interaction_id, "turns_pre_post", err))
        else:
            tp_ok += 1

        # 4d. video_viewer.html
        try:
            generate_video_viewer(interaction_id, interactions_dir=root_dir)
            vv_ok += 1
        except Exception as e:
            errors.append((interaction_id, "video_viewer", str(e)))

        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] derived files generated")

    print(f"\n  timestamps_by_turn:    {turns_ok}/{total} OK")
    print(f"  time_windows_pre_post: {tw_ok}/{total} OK")
    print(f"  turns_pre_post:        {tp_ok}/{total} OK")
    print(f"  video_viewer.html:     {vv_ok}/{total} OK")

    if errors:
        print(f"\n  {len(errors)} error(s):")
        for iid, step, err in errors[:20]:
            print(f"    {iid} [{step}]: {err}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Interactions:  {total}")
    print(f"  Output dir:    {root_dir}")


if __name__ == "__main__":
    main()
