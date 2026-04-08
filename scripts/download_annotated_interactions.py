#!/usr/bin/env python3
"""
download_annotated_interactions.py

Downloads the 100 naturalistic interactions from the Seamless Interaction dataset
where BOTH participants have third-party (3P) annotations. For each interaction,
downloads interaction-level metadata and per-participant data (3P annotations
and voice activity detection). Audio (.wav) files are excluded by default and
can be included with the --include-wav flag.

Requirements:
    - Python 3.8+
    - The seamless_interaction GitHub repo must be cloned locally, with the
      assets/ directory containing filelist.csv, interactions.csv,
      participants.csv, and relationships.csv.
    - Internet access to download from Meta's public S3 bucket at
      https://dl.fbaipublicfiles.com/seamless_interaction/

Usage:
    python download_annotated_interactions.py --repo-path /path/to/seamless_interaction

    Optional arguments:
        --output-dir    Directory to create annotated_interactions/ in (default: current directory)
        --include-wav   Also download per-participant audio (.wav) files
        --num-workers   Number of parallel download threads (default: 4)
        --dry-run       Print what would be downloaded without actually downloading

The script creates the following directory structure:

    annotated_interactions/
    ├── V00_S1288_I00000099/
    │   ├── interaction/
    │   │   ├── interaction_metadata.json    (prompt text, IPC codes, interaction type)
    │   │   ├── session_relationship.json    (stranger/familiar + detail)
    │   │   ├── participants_metadata.json   (Big Five personality scores)
    │   │   └── filelist_entries.json        (annotation/movement flags from filelist.csv)
    │   ├── participant_a_P0071/
    │   │   ├── 3P-IS_V00_S1288_I00000099_P0071.json   (perceived internal state)
    │   │   ├── 3P-R_V00_S1288_I00000099_P0071.json    (perceived behavior rationale)
    │   │   ├── 3P-V_V00_S1288_I00000099_P0071.json    (visual element description)
    │   │   ├── V00_S1288_I00000099_P0071.wav           (only with --include-wav)
    │   │   └── vad_V00_S1288_I00000099_P0071.jsonl     (Silero voice activity detection)
    │   └── participant_b_P1119/
    │       ├── 3P-IS_V00_S1288_I00000099_P1119.json
    │       ├── 3P-R_V00_S1288_I00000099_P1119.json
    │       ├── 3P-V_V00_S1288_I00000099_P1119.json
    │       ├── V00_S1288_I00000099_P1119.wav           (only with --include-wav)
    │       └── vad_V00_S1288_I00000099_P1119.jsonl
    └── ... (99 more interactions)

Note on participant_a vs participant_b naming:
    The Seamless dataset's interactions.csv defines participant_a and participant_b
    prompt roles, but does NOT provide a mapping from those roles to specific
    participant IDs (file_ids). We therefore assign participant_a and participant_b
    alphabetically by participant ID (lower ID = participant_a). This is an arbitrary
    but deterministic convention — it does NOT correspond to the prompt assignment
    in interactions.csv.

Note on the annotation JSON files:
    The official seamless_interaction Python package has a bug in fs.py
    (method _wget_download_from_s3, ~line 997) where it calls json.load() on
    annotation files that are actually JSONL format (one JSON object per line).
    This causes a JSONDecodeError: "Extra data: line 2 column 1". This script
    avoids that bug by downloading annotation files directly via HTTP requests
    and saving the raw content without attempting to parse it as single JSON.
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

# Base URL for Meta's public S3 bucket hosting the Seamless Interaction dataset.
# All downloadable files (audio, video, features, annotations, metadata) are
# served from this URL with no authentication required.
S3_BASE_URL = "https://dl.fbaipublicfiles.com/seamless_interaction"


# =============================================================================
# Step 1: Identify the 100 target interactions from filelist.csv
# =============================================================================

def load_filelist(repo_path: str) -> list[dict]:
    """
    Load filelist.csv from the seamless_interaction repo's assets/ directory.

    Each row represents one participant-interaction pair (a "file") with columns:
        - file_id:              e.g., "V00_S1288_I00000099_P1119"
        - label:                "naturalistic" or "improvised"
        - split:                "train", "dev", or "test"
        - batch_idx:            integer batch index for bulk downloads
        - archive_idx:          integer archive index within the batch
        - has_imitator_movement: "0" or "1" — whether imitator face features exist
        - has_annotation_1p:    "0" or "1" — whether first-person annotations exist
        - has_annotation_3p:    "0" or "1" — whether third-person annotations exist

    Returns:
        List of dicts, one per row in filelist.csv.
    """
    filelist_path = os.path.join(repo_path, "assets", "filelist.csv")
    if not os.path.exists(filelist_path):
        print(f"ERROR: filelist.csv not found at {filelist_path}")
        print(f"Make sure --repo-path points to the cloned seamless_interaction repository.")
        sys.exit(1)

    with open(filelist_path, newline="") as f:
        return list(csv.DictReader(f))


def extract_interaction_id(file_id: str) -> str:
    """
    Extract the interaction-level identifier from a participant-level file_id.

    File IDs have the format: V{vendor}_S{session}_I{prompt_hash}_P{participant}
    For example: "V00_S1288_I00000099_P1119"

    The interaction ID is everything except the participant suffix:
    "V00_S1288_I00000099"

    This uniquely identifies a specific interaction instance (a particular dyad
    doing a particular prompt in a particular session). Note that the I-number
    alone is just the prompt hash — many different dyads across different sessions
    share the same prompt.

    Args:
        file_id: Full participant-level file ID string.

    Returns:
        Interaction-level ID string (without participant suffix).
    """
    # Split off the last _P{id} segment. We use rsplit with maxsplit=1
    # to handle the case correctly — the participant ID is always the last
    # underscore-delimited segment.
    return file_id.rsplit("_", 1)[0]


def extract_participant_id(file_id: str) -> str:
    """
    Extract the participant ID (e.g., "P1119") from a file_id.

    Args:
        file_id: Full participant-level file ID string.

    Returns:
        Participant ID string (e.g., "P1119" or "P0844A").
    """
    return file_id.rsplit("_", 1)[1]


def extract_prompt_hash(interaction_id: str) -> str:
    """
    Extract the prompt hash (I-number) from an interaction ID.

    For example: "V00_S1288_I00000099" -> "00000099"

    This is used to look up interaction metadata in interactions.csv,
    where the key column is 'prompt_hash'.

    Args:
        interaction_id: Interaction-level ID string.

    Returns:
        Prompt hash string (zero-padded 8-digit number).
    """
    match = re.search(r"I(\d+)", interaction_id)
    if not match:
        raise ValueError(f"Could not extract prompt hash from {interaction_id}")
    return match.group(1)


def extract_vendor_and_session(interaction_id: str) -> tuple[str, str]:
    """
    Extract vendor ID and session number from an interaction ID.

    For example: "V00_S1288_I00000099" -> ("V00", "1288")

    Used to look up session-level relationship data in relationships.csv.

    Args:
        interaction_id: Interaction-level ID string.

    Returns:
        Tuple of (vendor_id, session_id) strings.
    """
    parts = interaction_id.split("_")
    vendor = parts[0]            # "V00"
    session = parts[1][1:]       # Strip "S" prefix -> "1288"
    return vendor, session


def find_both_annotated_interactions(filelist_rows: list[dict]) -> dict[str, list[str]]:
    """
    Filter filelist.csv to find naturalistic interactions where BOTH participants
    have third-party (3P) annotations.

    Filtering logic:
        1. Select rows where label == "naturalistic" (exclude improvised)
        2. Select rows where has_annotation_3p == "1"
        3. Group these rows by interaction ID (file_id without participant suffix)
        4. Keep only interactions that have exactly 2 annotated file entries
           (i.e., both participants in the dyad were annotated)

    In the full dataset (as of the version used in development):
        - 93,620 naturalistic file entries total
        - 543 of those have has_annotation_3p == "1"
        - These 543 span 443 unique interactions
        - 100 of those 443 interactions have both participants annotated (200 file entries)
        - 343 have only one participant annotated (343 file entries)
        - 200 + 343 = 543 ✓

    Args:
        filelist_rows: List of dicts from filelist.csv.

    Returns:
        Dict mapping interaction_id -> [file_id_1, file_id_2] for the 100
        both-annotated interactions. File IDs are sorted alphabetically so
        the ordering is deterministic.
    """
    # Step 1 & 2: Filter for naturalistic + 3P annotated
    nat_3p = [
        row for row in filelist_rows
        if row["label"] == "naturalistic" and row["has_annotation_3p"] == "1"
    ]
    print(f"  Naturalistic files with 3P annotations: {len(nat_3p)}")

    # Step 3: Group by interaction ID
    interaction_files = defaultdict(list)
    for row in nat_3p:
        iid = extract_interaction_id(row["file_id"])
        interaction_files[iid].append(row["file_id"])

    print(f"  Unique interactions with >= 1 annotated participant: {len(interaction_files)}")

    # Step 4: Keep only those with exactly 2 annotated participants
    both_annotated = {}
    for iid, file_ids in interaction_files.items():
        if len(file_ids) == 2:
            # Sort alphabetically for deterministic participant_a / participant_b assignment
            both_annotated[iid] = sorted(file_ids)

    print(f"  Interactions with BOTH participants annotated: {len(both_annotated)}")

    if len(both_annotated) != 100:
        print(f"  WARNING: Expected 100 interactions, found {len(both_annotated)}.")
        print(f"  This may indicate a dataset version change.")

    return both_annotated


# =============================================================================
# Step 2: Build interaction-level metadata from the CSV files
# =============================================================================

def load_interactions_csv(repo_path: str) -> dict[str, dict]:
    """
    Load interactions.csv, keyed by prompt_hash.

    Contains prompt text, IPC octant codes, and interaction type for each
    unique prompt used in the dataset.

    Returns:
        Dict mapping prompt_hash -> row dict.
    """
    path = os.path.join(repo_path, "assets", "interactions.csv")
    with open(path, newline="") as f:
        return {row["prompt_hash"]: row for row in csv.DictReader(f)}


def load_relationships_csv(repo_path: str) -> dict[str, dict]:
    """
    Load relationships.csv, keyed by "vendor_id|session_id".

    Contains relationship type (stranger/familiar) and detail (friends,
    coworkers, siblings, etc.) for each session.

    Returns:
        Dict mapping "vendor_id|session_id" -> row dict.
    """
    path = os.path.join(repo_path, "assets", "relationships.csv")
    with open(path, newline="") as f:
        return {
            f"{row['vendor_id']}|{row['session_id']}": row
            for row in csv.DictReader(f)
        }


def load_participants_csv(repo_path: str) -> dict[str, dict]:
    """
    Load participants.csv, keyed by "vendor_id|participant_id".

    Contains Big Five personality raw scores (extraversion, agreeableness,
    conscientiousness, neuroticism, openness). Many entries are "Undisclosed".

    Returns:
        Dict mapping "vendor_id|participant_id" -> row dict.
    """
    path = os.path.join(repo_path, "assets", "participants.csv")
    with open(path, newline="") as f:
        return {
            f"{row['vendor_id']}|{row['participant_id']}": row
            for row in csv.DictReader(f)
        }


def build_interaction_metadata(
    interaction_id: str,
    file_ids: list[str],
    filelist_rows: list[dict],
    interactions_lookup: dict,
    relationships_lookup: dict,
    participants_lookup: dict,
) -> dict[str, dict]:
    """
    Assemble all interaction-level metadata files for one interaction.

    Produces four JSON-serializable dicts corresponding to the four files
    saved in the interaction/ subdirectory:

    1. interaction_metadata.json:
       Prompt text for both participants, IPC octant code, interaction type.
       Sourced from interactions.csv using the prompt hash (I-number).

    2. session_relationship.json:
       Whether the dyad members are strangers or familiar, plus detail.
       Sourced from relationships.csv using vendor + session ID.

    3. participants_metadata.json:
       Big Five personality scores for both participants.
       Sourced from participants.csv using vendor + participant ID.

    4. filelist_entries.json:
       The raw filelist.csv rows for both participants in this interaction,
       including annotation flags, movement availability, split, and batch info.

    Args:
        interaction_id: e.g., "V00_S1288_I00000099"
        file_ids: List of 2 file IDs for this interaction.
        filelist_rows: Full filelist.csv data (for filelist entries lookup).
        interactions_lookup: interactions.csv keyed by prompt_hash.
        relationships_lookup: relationships.csv keyed by "vendor|session".
        participants_lookup: participants.csv keyed by "vendor|participant_id".

    Returns:
        Dict with keys "interaction_metadata", "session_relationship",
        "participants_metadata", "filelist_entries", each mapping to a
        JSON-serializable object.
    """
    result = {}

    # 1. Interaction metadata (from interactions.csv)
    prompt_hash = extract_prompt_hash(interaction_id)
    result["interaction_metadata"] = interactions_lookup.get(prompt_hash, {})

    # 2. Session relationship (from relationships.csv)
    vendor, session = extract_vendor_and_session(interaction_id)
    rel_key = f"{vendor}|{session}"
    result["session_relationship"] = relationships_lookup.get(rel_key, {})

    # 3. Participants metadata (from participants.csv)
    participants = {}
    for fid in file_ids:
        pid = extract_participant_id(fid)
        # participants.csv uses numeric participant_id without "P" prefix,
        # and the vendor_id without "V" prefix in some cases. We need to
        # handle both formats. The vendor in participants.csv may be "00"
        # instead of "V00".
        pid_num = pid.lstrip("P").lstrip("0") or "0"
        # Try multiple lookup key formats
        for v in [vendor, vendor.lstrip("V")]:
            for p in [pid.lstrip("P"), pid_num]:
                key = f"{v}|{p}"
                if key in participants_lookup:
                    participants[pid] = participants_lookup[key]
                    break
            if pid in participants:
                break
    result["participants_metadata"] = participants

    # 4. Filelist entries (raw rows from filelist.csv for this interaction)
    filelist_by_id = {row["file_id"]: row for row in filelist_rows}
    result["filelist_entries"] = [filelist_by_id[fid] for fid in file_ids if fid in filelist_by_id]

    return result


# =============================================================================
# Step 3: Download files from S3
# =============================================================================

def build_s3_url(label: str, split: str, category: str, file_id: str, ext: str) -> str:
    """
    Construct the full S3 download URL for a specific file.

    The URL pattern is:
        {S3_BASE_URL}/{label}/{split}/{category}/{file_id}.{ext}

    For example:
        https://dl.fbaipublicfiles.com/seamless_interaction/naturalistic/train/
        annotations/3P-IS/V00_S1288_I00000099_P1119.json

    Args:
        label: "naturalistic" or "improvised"
        split: "train", "dev", or "test"
        category: Path component(s) between split and filename,
                  e.g., "annotations/3P-IS" or "audio" or "metadata/vad"
        file_id: The file identifier, e.g., "V00_S1288_I00000099_P1119"
        ext: File extension without dot, e.g., "json", "wav", "jsonl"

    Returns:
        Full URL string.
    """
    return f"{S3_BASE_URL}/{label}/{split}/{category}/{file_id}.{ext}"


def download_file(url: str, output_path: str) -> tuple[bool, str]:
    """
    Download a single file from a URL and save it to disk.

    This function handles the download directly via urllib rather than through
    the seamless_interaction Python package. This avoids a bug in the package's
    _wget_download_from_s3 method (fs.py ~line 997) where annotation files
    (which are JSONL format — one JSON object per line) are incorrectly parsed
    with json.load(), which expects a single JSON object and raises:
        JSONDecodeError: Extra data: line 2 column 1

    By downloading the raw bytes and writing them directly to disk, we avoid
    any parsing issues.

    HTTP 403 responses indicate the file does not exist on S3 (Meta's S3
    bucket returns 403 Forbidden for nonexistent keys rather than 404).
    These are treated as non-fatal skips.

    Args:
        url: Full URL to download from.
        output_path: Local filesystem path to write the downloaded file to.

    Returns:
        Tuple of (success: bool, message: str).
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(data)

        return True, f"OK ({len(data):,} bytes)"

    except urllib.error.HTTPError as e:
        if e.code == 403:
            # 403 = file does not exist on S3 (standard S3 behavior for
            # nonexistent keys when public listing is disabled)
            return False, "NOT FOUND (HTTP 403)"
        return False, f"HTTP ERROR {e.code}"

    except Exception as e:
        return False, f"ERROR: {e}"


def download_participant_files(
    file_id: str,
    label: str,
    split: str,
    output_dir: str,
    include_wav: bool = False,
    include_annotations: bool = True,
    dry_run: bool = False,
) -> list[tuple[str, bool, str]]:
    """
    Download per-participant files for one file_id.

    By default downloads:
        - 3P-IS annotation (perceived internal state)
        - 3P-R annotation (perceived behavior rationale)
        - 3P-V annotation (visual element description)
        - VAD (.jsonl) — Silero Voice Activity Detection segments

    Set include_annotations=False to download only VAD (useful for
    participants who lack 3P annotations).

    The annotation files are JSON with one JSON object per line (JSONL format),
    despite having a .json extension. Each line contains:
        {"annotation": "...", "start_ts": <seconds>, "end_ts": <seconds>}

    The VAD file is also JSONL, with each line containing:
        {"start": <seconds>, "end": <seconds>}

    Args:
        file_id: e.g., "V00_S1288_I00000099_P1119"
        label: "naturalistic"
        split: "train" or "test"
        output_dir: Directory to save files into.
        include_wav: If True, also download the audio .wav file.
        include_annotations: If True (default), download 3P-IS/3P-R/3P-V.
            Set to False for participants without 3P annotations.
        dry_run: If True, print what would be downloaded without downloading.

    Returns:
        List of (filename, success, message) tuples.
    """
    # Define files to download for this participant.
    # Each tuple is (S3_category_path, filename_prefix, extension).
    files_to_download = []

    if include_annotations:
        # Third-party annotations
        files_to_download.extend([
            ("annotations/3P-IS", f"3P-IS_{file_id}", "json"),
            ("annotations/3P-R",  f"3P-R_{file_id}",  "json"),
            ("annotations/3P-V",  f"3P-V_{file_id}",  "json"),
        ])

    # Voice Activity Detection (Silero VAD at 100Hz) — always downloaded
    files_to_download.append(("metadata/vad", f"vad_{file_id}", "jsonl"))

    if include_wav:
        # Audio (mono, 48kHz, 32-bit float, echo-cancelled via Beryl AEC)
        files_to_download.append(("audio", file_id, "wav"))

    results = []
    for category, local_name, ext in files_to_download:
        url = build_s3_url(label, split, category, file_id, ext)
        output_path = os.path.join(output_dir, f"{local_name}.{ext}")

        if dry_run:
            results.append((f"{local_name}.{ext}", True, f"DRY RUN: {url}"))
        else:
            success, msg = download_file(url, output_path)
            results.append((f"{local_name}.{ext}", success, msg))

    return results


# =============================================================================
# Step 4: Orchestrate the full download
# =============================================================================

def download_all_interactions(
    both_annotated: dict[str, list[str]],
    filelist_rows: list[dict],
    repo_path: str,
    output_dir: str,
    include_wav: bool = False,
    num_workers: int = 4,
    dry_run: bool = False,
):
    """
    Download all 100 both-annotated interactions with their metadata.

    For each interaction:
    1. Creates the directory structure:
           {interaction_id}/interaction/
           {interaction_id}/participant_a_{pid}/
           {interaction_id}/participant_b_{pid}/
    2. Writes interaction-level metadata JSON files.
    3. Downloads per-participant files (annotations, VAD, and optionally audio).

    Participant ordering:
        Participants are sorted alphabetically by ID. The lower-sorted ID
        becomes participant_a, the higher becomes participant_b. This is an
        arbitrary but deterministic convention — it does NOT necessarily
        correspond to the participant_a/participant_b prompt assignment in
        interactions.csv.

    Args:
        both_annotated: Dict from find_both_annotated_interactions().
        filelist_rows: Full filelist.csv data.
        repo_path: Path to cloned seamless_interaction repo.
        output_dir: Path to create annotated_interactions/ directory in.
        include_wav: If True, also download audio .wav files.
        num_workers: Number of parallel download threads.
        dry_run: If True, skip actual downloads.
    """
    # Load the CSV lookup tables for metadata assembly
    print("\nLoading metadata CSVs...")
    interactions_lookup = load_interactions_csv(repo_path)
    relationships_lookup = load_relationships_csv(repo_path)
    participants_lookup = load_participants_csv(repo_path)

    # Build a lookup from file_id to its filelist row (for split info)
    filelist_by_id = {row["file_id"]: row for row in filelist_rows}

    # Create root output directory
    root_dir = os.path.join(output_dir, "annotated_interactions")
    os.makedirs(root_dir, exist_ok=True)

    # Process each interaction
    total = len(both_annotated)
    download_tasks = []  # Collect all download tasks for parallel execution

    for idx, (interaction_id, file_ids) in enumerate(sorted(both_annotated.items()), 1):
        print(f"\n[{idx}/{total}] {interaction_id}")

        # Determine participant_a and participant_b (alphabetical by participant ID)
        pid_a = extract_participant_id(file_ids[0])
        pid_b = extract_participant_id(file_ids[1])
        fid_a = file_ids[0]
        fid_b = file_ids[1]

        # Create directory structure
        interaction_dir = os.path.join(root_dir, interaction_id)
        meta_dir = os.path.join(interaction_dir, "interaction")
        dir_a = os.path.join(interaction_dir, f"participant_a_{pid_a}")
        dir_b = os.path.join(interaction_dir, f"participant_b_{pid_b}")

        for d in [meta_dir, dir_a, dir_b]:
            os.makedirs(d, exist_ok=True)

        # --- Write interaction-level metadata ---
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
            if not dry_run:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
            print(f"  interaction/{filename}: written")

        # --- Queue participant downloads ---
        # Look up the split for each file (needed to construct S3 URL)
        split_a = filelist_by_id[fid_a]["split"]
        split_b = filelist_by_id[fid_b]["split"]

        download_tasks.append((fid_a, "naturalistic", split_a, dir_a, include_wav, f"  participant_a_{pid_a}"))
        download_tasks.append((fid_b, "naturalistic", split_b, dir_b, include_wav, f"  participant_b_{pid_b}"))

    # --- Execute all downloads with thread pool ---
    print(f"\n{'=' * 60}")
    print(f"Downloading participant files ({len(download_tasks)} participants, {num_workers} threads)...")
    print(f"{'=' * 60}\n")

    success_count = 0
    fail_count = 0
    skip_count = 0

    def _do_download(task):
        fid, label, split, out_dir, inc_wav, prefix = task
        results = download_participant_files(fid, label, split, out_dir, include_wav=inc_wav, dry_run=dry_run)
        return prefix, results

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Interactions:  {total}")
    print(f"  Files OK:      {success_count}")
    print(f"  Files skipped: {skip_count} (not available on server)")
    print(f"  Files failed:  {fail_count}")
    print(f"  Output dir:    {root_dir}")


# =============================================================================
# Interaction duration
# =============================================================================


def get_interaction_duration(interaction_id, interactions_dir="./annotated_interactions"):
    """
    Return the duration (in seconds) of an interaction's video by probing
    the actual MP4 file on Meta's S3 bucket via ffprobe.

    Reads the first participant's video URL from filelist_entries.json
    (both participants share the same video duration) and uses ffprobe to
    extract the duration without downloading the full file.

    Requires ffprobe to be installed (part of the ffmpeg package).

    Args:
        interaction_id: e.g. "V00_S1132_I00000333"
        interactions_dir: path to the interactions directory

    Returns:
        Duration in seconds (float).

    Raises:
        FileNotFoundError: if filelist_entries.json doesn't exist.
        RuntimeError: if ffprobe fails.
    """
    import subprocess

    entries_path = os.path.join(
        interactions_dir, interaction_id, "interaction", "filelist_entries.json"
    )
    with open(entries_path) as f:
        entries = json.load(f)

    entry = entries[0]
    url = (
        f"{S3_BASE_URL}/{entry.get('label', 'naturalistic')}"
        f"/{entry['split']}/video/{entry['file_id']}.mp4"
    )

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            url,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {interaction_id}: {result.stderr.strip()}"
        )

    return float(result.stdout.strip())


# =============================================================================
# Video viewer generation
# =============================================================================

def generate_video_viewer(interaction_id, interactions_dir="./annotated_interactions"):
    """
    Create an HTML file with side-by-side video players for both participants
    in an interaction, streaming from Meta's S3 bucket.

    Assumes filelist_entries.json and interaction_metadata.json already exist
    in the interaction's interaction/ directory (created by
    download_all_interactions).

    Both prompts from interaction_metadata.json are displayed above the videos.
    The dataset's participant_a/participant_b prompt roles do NOT map to our
    alphabetically-assigned participant_a/participant_b directories, so both
    prompts are shown without attempting to assign them to specific participants.

    Args:
        interaction_id: e.g. "V00_S1132_I00000333"
        interactions_dir: path to annotated_interactions/

    Returns:
        Path to the generated HTML file.
    """
    interaction_dir = os.path.join(interactions_dir, interaction_id, "interaction")

    entries_path = os.path.join(interaction_dir, "filelist_entries.json")
    with open(entries_path) as f:
        entries = json.load(f)

    # Load prompts from interaction_metadata.json
    metadata_path = os.path.join(interaction_dir, "interaction_metadata.json")
    prompt_a = ""
    prompt_b = ""
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        prompt_a = metadata.get("participant_a_prompt_text", "")
        prompt_b = metadata.get("participant_b_prompt_text", "")

    # Sort alphabetically by participant ID (same convention as directory naming)
    entries.sort(key=lambda e: e["file_id"].rsplit("_", 1)[1])

    participants = []
    for i, entry in enumerate(entries):
        pid = entry["file_id"].rsplit("_", 1)[1]
        letter = "a" if i == 0 else "b"
        url = f"{S3_BASE_URL}/naturalistic/{entry['split']}/video/{entry['file_id']}.mp4"
        participants.append((pid, letter, url))

    # Build prompts HTML — show both if they differ, single block if identical
    if prompt_a and prompt_b:
        if prompt_a == prompt_b:
            prompts_html = f"""<div class="prompts">
  <div class="prompt"><span class="prompt-label">Prompt (same for both):</span> {prompt_a}</div>
</div>"""
        else:
            prompts_html = f"""<div class="prompts">
  <div class="prompt"><span class="prompt-label">Prompt 1:</span> {prompt_a}</div>
  <div class="prompt"><span class="prompt-label">Prompt 2:</span> {prompt_b}</div>
  <div class="prompt-note">One prompt was given to each participant; the prompt-to-participant mapping was not found in the released metadata.</div>
</div>"""
    else:
        prompts_html = ""

    # Load 3P-IS, 3P-R, and 3P-V annotations for each participant.
    # All three share the same time windows; we merge them by (start_ts, end_ts).
    mois_by_pid = {}
    for pid, letter, url in participants:
        pdir = os.path.join(interactions_dir, interaction_id, f"participant_{letter}_{pid}")
        # Keyed by (start_ts, end_ts) → {"is": ..., "r": ..., "v": ...}
        merged = {}
        if os.path.isdir(pdir):
            for prefix, field in [("3P-IS_", "is"), ("3P-R_", "r"), ("3P-V_", "v")]:
                for fname in sorted(os.listdir(pdir)):
                    if fname.startswith(prefix) and fname.endswith(".json"):
                        fpath = os.path.join(pdir, fname)
                        with open(fpath) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                ann = json.loads(line)
                                key = (ann.get("start_ts"), ann.get("end_ts"))
                                if key not in merged:
                                    merged[key] = {"start_ts": key[0], "end_ts": key[1]}
                                merged[key][field] = ann.get("annotation", "")
        mois = sorted(merged.values(), key=lambda m: m.get("start_ts", 0))
        mois_by_pid[pid] = mois

    def _fmt_time(seconds):
        """Format seconds as M:SS.s"""
        if seconds is None:
            return "?"
        m = int(seconds) // 60
        s = seconds - m * 60
        return f"{m}:{s:04.1f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{interaction_id}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 24px; }}
  h1 {{ font-size: 18px; font-weight: 500; margin-bottom: 16px; }}
  .prompts {{ margin-bottom: 18px; }}
  .prompt {{ font-size: 13px; color: #ccc; line-height: 1.5; margin-bottom: 8px; }}
  .prompt-label {{ color: #888; font-weight: 600; }}
  .prompt-note {{ font-size: 12px; color: #666; font-style: italic; }}
  .controls {{ text-align: center; margin-bottom: 16px; }}
  .sync-btn {{ background: #2d5f8a; color: #eee; border: none; border-radius: 6px; padding: 10px 28px; font-size: 14px; cursor: pointer; font-family: inherit; }}
  .sync-btn:hover {{ background: #3a7ab5; }}
  .videos {{ display: flex; gap: 16px; }}
  .vid-container {{ flex: 1; }}
  .vid-container label {{ display: block; font-size: 14px; color: #aaa; margin-bottom: 6px; }}
  video {{ width: 100%; border-radius: 6px; background: #000; }}
  .moi-list {{ margin-top: 12px; }}
  .moi-list h3 {{ font-size: 13px; color: #888; margin: 0 0 6px 0; font-weight: 600; }}
  .moi-item {{ font-size: 12px; color: #bbb; padding: 8px; border-bottom: 1px solid #222; }}
  .moi-header {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }}
  .moi-time {{ color: #6a9fcf; font-family: monospace; white-space: nowrap; font-weight: 600; }}
  .moi-detail {{ line-height: 1.5; margin-left: 4px; }}
  .moi-label {{ color: #888; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
  .moi-text {{ color: #ccc; }}
  .jump-btn {{ background: none; border: 1px solid #445; color: #6a9fcf; border-radius: 4px; padding: 1px 7px; font-size: 11px; cursor: pointer; font-family: monospace; white-space: nowrap; }}
  .jump-btn:hover {{ background: #1a3350; border-color: #6a9fcf; }}
  .no-moi {{ font-size: 12px; color: #555; font-style: italic; }}
</style>
</head>
<body>
<h1>{interaction_id}</h1>
{prompts_html}
<div class="controls">
  <button class="sync-btn" onclick="syncPlay()">&#9654; Play Both</button>
</div>
<div class="videos">
"""
    for pid, letter, url in participants:
        mois = mois_by_pid.get(pid, [])
        moi_html = ""
        if mois:
            moi_html = '<div class="moi-list"><h3>Moments of Interest</h3>\n'
            for m in mois:
                start = _fmt_time(m.get("start_ts"))
                end = _fmt_time(m.get("end_ts"))
                start_ts = m.get("start_ts", 0)
                details = ""
                for key, label in [("is", "Internal State"), ("r", "Rationale"), ("v", "Visual Element")]:
                    text = m.get(key, "")
                    if text:
                        details += f'      <div class="moi-detail"><span class="moi-label">{label}:</span> <span class="moi-text">{text}</span></div>\n'
                moi_html += f'    <div class="moi-item"><div class="moi-header"><button class="jump-btn" onclick="jumpTo({start_ts})">&#9654;</button><span class="moi-time">{start}\u2013{end}</span></div>\n{details}    </div>\n'
            moi_html += "</div>"
        else:
            moi_html = '<div class="no-moi">No annotated MOIs</div>'

        html += f"""  <div class="vid-container">
    <label>{pid} (participant_{letter})</label>
    <video controls preload="metadata">
      <source src="{url}" type="video/mp4">
    </video>
    {moi_html}
  </div>
"""
    html += """</div>
<script>
function syncPlay() {
  const videos = document.querySelectorAll('video');
  const btn = document.querySelector('.sync-btn');
  const allPaused = Array.from(videos).every(v => v.paused);
  if (allPaused) {
    const anyEnded = Array.from(videos).some(v => v.ended);
    if (anyEnded) videos.forEach(v => { v.currentTime = 0; });
    videos.forEach(v => { v.currentTime = videos[0].currentTime; v.play(); });
    btn.textContent = '\\u275A\\u275A Pause Both';
  } else {
    videos.forEach(v => v.pause());
    btn.textContent = '\\u25B6 Play Both';
  }
}
function jumpTo(t) {
  const videos = document.querySelectorAll('video');
  const btn = document.querySelector('.sync-btn');
  videos.forEach(v => { v.currentTime = t; v.play(); });
  btn.textContent = '\\u275A\\u275A Pause Both';
}
document.querySelectorAll('video').forEach(v => {
  v.addEventListener('ended', () => {
    const allEnded = Array.from(document.querySelectorAll('video')).every(v => v.paused || v.ended);
    if (allEnded) document.querySelector('.sync-btn').textContent = '\\u25B6 Play Both';
  });
  v.addEventListener('pause', () => {
    const allPaused = Array.from(document.querySelectorAll('video')).every(v => v.paused);
    if (allPaused) document.querySelector('.sync-btn').textContent = '\\u25B6 Play Both';
  });
});
</script>
</body>
</html>"""

    output_path = os.path.join(
        interactions_dir, interaction_id, "interaction", "video_viewer.html"
    )
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the 100 naturalistic interactions from the Seamless Interaction "
            "dataset where both participants have third-party (3P) annotations."
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
        default=".",
        help="Directory to create annotated_interactions/ in. Default: current directory.",
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
    print("Seamless Interaction: Download Both-Annotated Interactions")
    print("=" * 60)

    # Step 1: Identify the 100 target interactions
    print("\nStep 1: Identifying target interactions from filelist.csv...")
    filelist_rows = load_filelist(args.repo_path)
    print(f"  Total file entries in filelist.csv: {len(filelist_rows)}")
    both_annotated = find_both_annotated_interactions(filelist_rows)

    # Step 2 & 3: Download everything
    print(f"\nStep 2: Downloading interaction metadata and participant files...")
    download_all_interactions(
        both_annotated=both_annotated,
        filelist_rows=filelist_rows,
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        include_wav=args.include_wav,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
