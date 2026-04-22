#!/usr/bin/env python3
"""
build_poc_manifest.py

Select a small, participant-disjoint, stranger/familiar-balanced subsample of
Seamless naturalistic IPC-conversation interactions for the turn-taking
proof-of-concept (Idea 1).

What this script guarantees about the resulting sample
------------------------------------------------------
1. Every selected interaction has BOTH participants' audio available on S3
   (naturalistic + ipc_conversation prompts always have per-participant wavs).
2. Every selected interaction is "conversational" in the sense relevant to
   turn-taking research: it is
       label            == "naturalistic"        (free-conversation recording,
                                                   not scripted/improvised)
       interaction_type == "ipc_conversation"   (genuine back-and-forth, not
                                                   charades, gesture games,
                                                   or collaborative storytelling)
3. Participants are DISJOINT across {train, val, test}. Seamless's own splits
   are NOT strictly participant-disjoint within the ipc-conversation subset
   (15 participants leak train→dev, 31 leak train→test at dataset scale), so
   this script enforces disjointness itself by sampling val and test first,
   collecting their participant IDs, then restricting train sampling to
   interactions whose participants are not in that reserved set.
4. Each split contains ~equal numbers of "stranger" and "familiar" dyads.
5. Selection is deterministic given --seed (default 42).

Inputs
------
- Seamless repo assets (filelist.csv, interactions.csv, relationships.csv,
  participants.csv) located at --repo-path.

Outputs
-------
- poc_manifest.csv at --output-path (default:
  <project>/turn_taking_analysis/manifests/poc_manifest.csv)

Manifest columns
----------------
    split               "train" | "val" | "test"   (our POC split, not Seamless's)
    seamless_split      Seamless's own split for this interaction (for reference)
    interaction_id      e.g. "V00_S1132_I00000333"
    vendor_id           e.g. "V00"
    session_id          e.g. "1132"
    prompt_hash         e.g. "00000333"
    interaction_type    always "ipc_conversation" in this manifest
    label               always "naturalistic" in this manifest
    relationship        "stranger" or "familiar"
    relationship_detail "friends" | "coworkers" | "stranger" | ...
    file_id_a           e.g. "V00_S1132_I00000333_P1093"  (alphabetically-first PID)
    file_id_b           e.g. "V00_S1132_I00000333_P0737"  — wait, see note
    participant_a       e.g. "P1093"
    participant_b       e.g. "P0737"
    audio_url_a         full S3 URL for participant_a's .wav
    audio_url_b         full S3 URL for participant_b's .wav
    video_url_a         full S3 URL for participant_a's .mp4
    video_url_b         full S3 URL for participant_b's .mp4
    vad_url_a           full S3 URL for participant_a's VAD .jsonl
    vad_url_b           full S3 URL for participant_b's VAD .jsonl

Note on participant_a/participant_b ordering
--------------------------------------------
Following download_annotated_interactions.py, participant_a = the lower-sorted
participant ID (alphabetical). This is an arbitrary but deterministic
convention and does NOT correspond to the participant_a/participant_b prompt
roles in interactions.csv (Seamless does not publish that mapping).

Future download scripts should consume this manifest row-by-row and save all
fetched assets under the designated subset directory structure:

    turn_taking_analysis/subset/
      audio/       {file_id}.wav
      video/       {file_id}.mp4
      vad/         {file_id}.jsonl
      openface/    {file_id}/<OpenFace output files>     (only for selected subset)
      hubert/      {file_id}.npy                         (derived downstream)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# -----------------------------------------------------------------------------
# Re-use the Seamless loading helpers from csci535-project/scripts. This script
# lives at turn_taking_analysis/scripts/, so we resolve the sibling directory
# and insert it on sys.path before importing.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # .../csci535-project
SHARED_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (  # noqa: E402
    S3_BASE_URL,
    build_s3_url,
    extract_interaction_id,
    extract_participant_id,
    extract_prompt_hash,
    extract_vendor_and_session,
    load_filelist,
    load_interactions_csv,
    load_relationships_csv,
)


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_REPO_PATH = str(PROJECT_ROOT.parent / "seamless_interaction")
DEFAULT_OUTPUT_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "poc_manifest.csv"
)

# POC sample sizes (Tier 2 from the plan: 16/4/4 = 24 total interactions).
DEFAULT_N_TRAIN = 16
DEFAULT_N_VAL = 4
DEFAULT_N_TEST = 4
DEFAULT_SEED = 42


# =============================================================================
# Step 1 — Build the candidate pool
# =============================================================================

def build_candidate_pool(
    filelist_rows: list[dict],
    interactions_lookup: dict[str, dict],
    relationships_lookup: dict[str, dict],
) -> list[dict]:
    """
    Build the list of candidate interactions for the POC.

    An interaction is kept if and only if:
        - it has exactly 2 participants in filelist.csv,
        - both participants' rows have label == "naturalistic",
        - both rows are in the same Seamless split (sanity),
        - its prompt_hash resolves in interactions.csv,
        - interaction_type == "ipc_conversation",
        - session relationship metadata exists in relationships.csv.

    Returns a list of dicts with all the fields needed for sampling and
    manifest emission (see module docstring for the column list).
    """
    # Group filelist rows by interaction_id (strip the _P{pid} suffix).
    by_iid: dict[str, list[dict]] = defaultdict(list)
    for row in filelist_rows:
        by_iid[extract_interaction_id(row["file_id"])].append(row)

    pool: list[dict] = []
    for interaction_id, rows in by_iid.items():
        # Require a full dyad.
        if len(rows) != 2:
            continue
        # Require both participants be from the naturalistic subset.
        if not all(r["label"] == "naturalistic" for r in rows):
            continue
        # Sanity: Seamless places both participants of a dyad in the same split.
        if rows[0]["split"] != rows[1]["split"]:
            continue

        prompt_hash = extract_prompt_hash(interaction_id)
        interaction_meta = interactions_lookup.get(prompt_hash)
        if interaction_meta is None:
            continue
        if interaction_meta.get("interaction_type") != "ipc_conversation":
            continue

        vendor, session = extract_vendor_and_session(interaction_id)
        rel = relationships_lookup.get(f"{vendor}|{session}")
        if rel is None:
            continue

        # Alphabetical participant_a / participant_b ordering.
        sorted_rows = sorted(rows, key=lambda r: extract_participant_id(r["file_id"]))
        fid_a = sorted_rows[0]["file_id"]
        fid_b = sorted_rows[1]["file_id"]
        pid_a = extract_participant_id(fid_a)
        pid_b = extract_participant_id(fid_b)
        seamless_split = sorted_rows[0]["split"]

        pool.append({
            "interaction_id":       interaction_id,
            "vendor_id":            vendor,
            "session_id":           session,
            "prompt_hash":          prompt_hash,
            "interaction_type":     interaction_meta["interaction_type"],
            "label":                "naturalistic",
            "relationship":         rel["relationship"],
            "relationship_detail":  rel["relationship_detail"],
            "seamless_split":       seamless_split,
            "file_id_a":            fid_a,
            "file_id_b":            fid_b,
            "participant_a":        pid_a,
            "participant_b":        pid_b,
        })

    return pool


# =============================================================================
# Step 2 — Sampling helpers
# =============================================================================

def _split_counts(pool: list[dict]) -> dict[tuple[str, str], int]:
    """Count (seamless_split, relationship) pairs in a pool — for diagnostics."""
    return Counter((x["seamless_split"], x["relationship"]) for x in pool)


def stratified_sample(
    candidates: list[dict],
    n_stranger: int,
    n_familiar: int,
    exclude_participants: set[str],
    rng: random.Random,
) -> list[dict]:
    """
    Sample `n_stranger` stranger dyads and `n_familiar` familiar dyads from
    `candidates`, with the constraints that:
        1. No sampled interaction uses any participant in `exclude_participants`.
        2. No participant appears in more than one *sampled* interaction
           (within this call — prevents a single heavily-recurrent participant
           from dominating the split).

    Raises ValueError if the constraints cannot be satisfied.
    """
    def _sample_pool(pool: list[dict], n: int, label: str) -> list[dict]:
        # Shuffle, then greedily pick while respecting the "no repeated
        # participant" constraint within this sub-sample.
        shuffled = list(pool)
        rng.shuffle(shuffled)
        picked: list[dict] = []
        used: set[str] = set(exclude_participants)
        for cand in shuffled:
            if cand["participant_a"] in used or cand["participant_b"] in used:
                continue
            picked.append(cand)
            used.add(cand["participant_a"])
            used.add(cand["participant_b"])
            if len(picked) == n:
                return picked
        raise ValueError(
            f"Could not sample {n} {label} interactions from pool of {len(pool)} "
            f"under the participant-disjointness constraint (got {len(picked)})."
        )

    strangers = [c for c in candidates if c["relationship"] == "stranger"]
    familiars = [c for c in candidates if c["relationship"] == "familiar"]
    return _sample_pool(strangers, n_stranger, "stranger") + \
           _sample_pool(familiars, n_familiar, "familiar")


# =============================================================================
# Step 3 — Select the full POC sample
# =============================================================================

def select_poc_sample(
    pool: list[dict],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> dict[str, list[dict]]:
    """
    Produce {"train": [...], "val": [...], "test": [...]} where each split
    contains ceil(n/2) stranger dyads + floor(n/2) familiar dyads (balanced),
    and no participant appears in more than one split.

    Sampling order is: val → test → train. We sample val and test first
    because they draw from the much smaller Seamless dev/test pools; if a
    conflict arises we'd rather it cost us a train interaction (of which we
    have 35k+) than a dev interaction (of which we have ~500).
    """
    rng = random.Random(seed)

    by_split: dict[str, list[dict]] = defaultdict(list)
    for entry in pool:
        by_split[entry["seamless_split"]].append(entry)

    # We use Seamless's own split membership as the upstream pool for each of
    # our POC splits: val is drawn from Seamless `dev`, test from Seamless
    # `test`, train from Seamless `train`. This preserves Seamless's own
    # data-hygiene boundaries as far as they go, and we enforce the
    # participant-disjointness constraint on top.
    val_pool = by_split.get("dev", [])
    test_pool = by_split.get("test", [])
    train_pool = by_split.get("train", [])

    def _halves(n: int) -> tuple[int, int]:
        """Split n into (stranger, familiar) halves; stranger gets the extra
        on odd counts since strangers are more abundant in Seamless."""
        n_str = (n + 1) // 2
        n_fam = n // 2
        return n_str, n_fam

    # --- Sample val (from Seamless dev) ---
    n_val_str, n_val_fam = _halves(n_val)
    val = stratified_sample(val_pool, n_val_str, n_val_fam, set(), rng)

    # --- Sample test (from Seamless test), excluding val participants ---
    excluded: set[str] = set()
    for v in val:
        excluded.add(v["participant_a"])
        excluded.add(v["participant_b"])
    n_test_str, n_test_fam = _halves(n_test)
    test = stratified_sample(test_pool, n_test_str, n_test_fam, excluded, rng)

    # --- Sample train (from Seamless train), excluding val+test participants ---
    for t in test:
        excluded.add(t["participant_a"])
        excluded.add(t["participant_b"])
    n_train_str, n_train_fam = _halves(n_train)
    train = stratified_sample(train_pool, n_train_str, n_train_fam, excluded, rng)

    return {"train": train, "val": val, "test": test}


# =============================================================================
# Step 4 — Enrich each row with S3 URLs and emit the manifest
# =============================================================================

def _url(file_id: str, label: str, seamless_split: str, category: str, ext: str) -> str:
    """Thin wrapper around the shared build_s3_url helper."""
    return build_s3_url(label, seamless_split, category, file_id, ext)


def enrich_row_with_urls(entry: dict, poc_split: str) -> dict:
    """
    Add the full set of S3 download URLs and the POC split label to a sample
    entry, returning the row that will be written to the manifest.
    """
    label = entry["label"]                     # "naturalistic"
    ss = entry["seamless_split"]               # "train" | "dev" | "test"
    fid_a = entry["file_id_a"]
    fid_b = entry["file_id_b"]

    return {
        "split":                poc_split,
        "seamless_split":       ss,
        "interaction_id":       entry["interaction_id"],
        "vendor_id":            entry["vendor_id"],
        "session_id":           entry["session_id"],
        "prompt_hash":          entry["prompt_hash"],
        "interaction_type":     entry["interaction_type"],
        "label":                label,
        "relationship":         entry["relationship"],
        "relationship_detail":  entry["relationship_detail"],
        "file_id_a":            fid_a,
        "file_id_b":            fid_b,
        "participant_a":        entry["participant_a"],
        "participant_b":        entry["participant_b"],
        "audio_url_a":          _url(fid_a, label, ss, "audio",        "wav"),
        "audio_url_b":          _url(fid_b, label, ss, "audio",        "wav"),
        "video_url_a":          _url(fid_a, label, ss, "video",        "mp4"),
        "video_url_b":          _url(fid_b, label, ss, "video",        "mp4"),
        "vad_url_a":            _url(fid_a, label, ss, "metadata/vad", "jsonl"),
        "vad_url_b":            _url(fid_b, label, ss, "metadata/vad", "jsonl"),
    }


MANIFEST_COLUMNS = [
    "split",
    "seamless_split",
    "interaction_id",
    "vendor_id",
    "session_id",
    "prompt_hash",
    "interaction_type",
    "label",
    "relationship",
    "relationship_detail",
    "file_id_a",
    "file_id_b",
    "participant_a",
    "participant_b",
    "audio_url_a",
    "audio_url_b",
    "video_url_a",
    "video_url_b",
    "vad_url_a",
    "vad_url_b",
]


def write_manifest(selection: dict[str, list[dict]], out_path: str) -> None:
    """Write the enriched manifest CSV, one row per selected interaction."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows: list[dict] = []
    # Deterministic output ordering: train, val, test; then by interaction_id.
    for split_name in ["train", "val", "test"]:
        for entry in sorted(selection[split_name], key=lambda e: e["interaction_id"]):
            rows.append(enrich_row_with_urls(entry, split_name))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Step 5 — Verification + summary printing
# =============================================================================

def verify_disjoint_participants(selection: dict[str, list[dict]]) -> None:
    """Fail loudly if any participant appears in more than one POC split."""
    participants_by_split: dict[str, set[str]] = {}
    for split, entries in selection.items():
        s = set()
        for e in entries:
            s.add(e["participant_a"])
            s.add(e["participant_b"])
        participants_by_split[split] = s

    splits = list(participants_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = participants_by_split[a] & participants_by_split[b]
            if overlap:
                raise AssertionError(
                    f"Participants leaked between '{a}' and '{b}': {sorted(overlap)}"
                )


def print_summary(pool: list[dict], selection: dict[str, list[dict]]) -> None:
    """Print a human-readable summary of the candidate pool and the selection."""
    print("\n" + "=" * 64)
    print("CANDIDATE POOL")
    print("=" * 64)
    print(f"  Total naturalistic ipc_conversation dyads in Seamless: {len(pool):,}")
    counts = _split_counts(pool)
    for (seamless_split, rel), n in sorted(counts.items()):
        print(f"    seamless_split={seamless_split:5s}  {rel:8s}: {n:>6}")

    print("\n" + "=" * 64)
    print("POC SELECTION")
    print("=" * 64)
    for split in ["train", "val", "test"]:
        rows = selection[split]
        rel_counts = Counter(r["relationship"] for r in rows)
        detail_counts = Counter(r["relationship_detail"] for r in rows)
        unique_participants = set()
        for r in rows:
            unique_participants.add(r["participant_a"])
            unique_participants.add(r["participant_b"])
        print(f"  {split:5s}  n_dyads={len(rows):>2}  "
              f"stranger={rel_counts['stranger']}  familiar={rel_counts['familiar']}  "
              f"unique_participants={len(unique_participants)}")
        print(f"         relationship_detail: {dict(detail_counts)}")
        print(f"         seamless_splits: {Counter(r['seamless_split'] for r in rows)}")
    print("=" * 64 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a participant-disjoint, stranger/familiar-balanced "
            "subsample of naturalistic ipc_conversation dyads from Seamless "
            "for the turn-taking proof-of-concept."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-path",
        default=DEFAULT_REPO_PATH,
        help=(
            "Path to the cloned seamless_interaction repository (must contain "
            "assets/). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write poc_manifest.csv. Default: %(default)s",
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN,
                        help="Number of training dyads (default: %(default)s).")
    parser.add_argument("--n-val",   type=int, default=DEFAULT_N_VAL,
                        help="Number of validation dyads (default: %(default)s).")
    parser.add_argument("--n-test",  type=int, default=DEFAULT_N_TEST,
                        help="Number of test dyads (default: %(default)s).")
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED,
                        help="Random seed (default: %(default)s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary but do not write the manifest file.")
    args = parser.parse_args()

    # Sanity check: asset files exist.
    assets_dir = Path(args.repo_path) / "assets"
    for fname in ["filelist.csv", "interactions.csv", "relationships.csv"]:
        p = assets_dir / fname
        if not p.exists():
            sys.exit(f"ERROR: required asset {p} not found. "
                     f"Adjust --repo-path (got: {args.repo_path}).")

    # Load Seamless metadata.
    print(f"Loading Seamless metadata from {assets_dir} ...")
    filelist_rows        = load_filelist(args.repo_path)
    interactions_lookup  = load_interactions_csv(args.repo_path)
    relationships_lookup = load_relationships_csv(args.repo_path)
    print(f"  filelist rows:         {len(filelist_rows):,}")
    print(f"  interactions (prompts): {len(interactions_lookup):,}")
    print(f"  relationships:         {len(relationships_lookup):,}")

    # Build pool and sample.
    pool = build_candidate_pool(filelist_rows, interactions_lookup, relationships_lookup)
    selection = select_poc_sample(
        pool,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )

    # Verify participant-disjointness across {train, val, test} (this should
    # always hold by construction; assert guards against future changes).
    verify_disjoint_participants(selection)

    # Print diagnostic summary.
    print_summary(pool, selection)

    # Emit manifest.
    if args.dry_run:
        print(f"[dry-run] Would write manifest to: {args.output_path}")
    else:
        write_manifest(selection, args.output_path)
        print(f"Wrote manifest: {args.output_path}")


if __name__ == "__main__":
    main()
