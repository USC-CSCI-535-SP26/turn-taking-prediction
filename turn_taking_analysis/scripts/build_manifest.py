#!/usr/bin/env python3
"""
build_manifest.py

Build the full-project turn-taking manifest covering every interaction we have
on disk for the project:

  1. The POC rows from poc_manifest.csv (all columns kept verbatim EXCEPT
     'split' — their split assignments are redone alongside the rest).
  2. Every interaction directory under both_annotated_interactions/.
  3. Every interaction directory under single_annotated_interactions/.

Per-interaction metadata for (2) and (3) is read from the local
<interaction_id>/interaction/ subdirectory (filelist_entries.json,
interaction_metadata.json, session_relationship.json), so this script does
NOT need the full Seamless metadata assets.

Split logic
-----------
- Target ratio is 70 / 15 / 15 (configurable via --train-frac / --val-frac).
- Participants are disjoint across {train, val, test}. This is enforced
  exactly by treating the pool as a participant-graph and assigning
  connected components (not individual interactions) as atomic units.
- Stranger / familiar is NOT stratified (relationship columns are kept for
  reference only).
- Placement uses Longest-Processing-Time bin-packing: components are sorted
  largest-first, then each is placed into the split with the most remaining
  capacity. This respects target proportions without any forced/override
  branches and without any drops.
- Selection is deterministic given --seed (seed controls component
  tie-breaking only; the component structure is a pure function of the
  pool).

Manifest columns match poc_manifest.csv exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# -----------------------------------------------------------------------------
# Re-use the Seamless helpers from csci535-project/scripts. This script lives
# at turn_taking_analysis/scripts/, so parents[2] resolves to csci535-project.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                  # .../csci535-project
SHARED_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (  # noqa: E402
    build_s3_url,
    extract_interaction_id,
    extract_participant_id,
    extract_prompt_hash,
    extract_vendor_and_session,
)


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_POC_MANIFEST = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "poc_manifest.csv"
)
DEFAULT_OUTPUT_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "manifest.csv"
)
DEFAULT_BOTH_ANNOTATED_DIR = str(PROJECT_ROOT / "both_annotated_interactions")
DEFAULT_SINGLE_ANNOTATED_DIR = str(PROJECT_ROOT / "single_annotated_interactions")

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
DEFAULT_SEED = 42

# Interactions excluded from the pool before splits are computed.
# Each has empty (0-byte) VAD and/or transcript files at Meta's S3
# source for one or both participants — verified by HTTP HEAD
# (content-length: 0). The label-generation pipeline can't produce
# valid samples without those upstream artifacts, so dropping them
# here keeps manifest.csv aligned with what the labeler can actually
# emit.
EXCLUDED_INTERACTION_IDS = frozenset({
    "V00_S1204_I00000196",   # P1109 VAD + transcript both 0 bytes
    "V03_S0132_I00000132",   # both pids: VAD 0 bytes
    "V03_S0134_I00000498",   # both pids: VAD + transcript both 0 bytes
    "V03_S0190_I00000486",   # both pids: VAD + transcript both 0 bytes
    "V03_S0199_I00000498",   # both pids: VAD + transcript both 0 bytes
    "V03_S0329_I00000068",   # both pids: VAD + transcript both 0 bytes
    "V03_S0702_I00000209",   # both pids: VAD + transcript both 0 bytes
    "V03_S0712_I00000421",   # both pids: VAD 0 bytes
})


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


# =============================================================================
# Step 1 — Build pool entries.
# =============================================================================

def _urls_for(file_id: str, label: str, seamless_split: str) -> dict:
    """Construct the full set of S3 URLs for one participant's assets."""
    return {
        "audio": build_s3_url(label, seamless_split, "audio",        file_id, "wav"),
        "video": build_s3_url(label, seamless_split, "video",        file_id, "mp4"),
        "vad":   build_s3_url(label, seamless_split, "metadata/vad", file_id, "jsonl"),
    }


def load_local_interaction(interaction_dir: Path) -> dict:
    """
    Read the metadata files under <interaction_dir>/interaction/ and return a
    fully-enriched pool entry (with URLs baked in, no 'split' column yet).
    """
    idir = interaction_dir / "interaction"
    filelist = json.loads((idir / "filelist_entries.json").read_text())
    meta = json.loads((idir / "interaction_metadata.json").read_text())
    rel = json.loads((idir / "session_relationship.json").read_text())

    if len(filelist) != 2:
        raise ValueError(
            f"{interaction_dir.name}: expected 2 filelist entries, got {len(filelist)}"
        )
    labels = {r["label"] for r in filelist}
    splits = {r["split"] for r in filelist}
    if len(labels) != 1 or len(splits) != 1:
        raise ValueError(
            f"{interaction_dir.name}: mixed labels/splits across dyad "
            f"(labels={labels}, splits={splits})"
        )
    label = next(iter(labels))
    seamless_split = next(iter(splits))

    sorted_rows = sorted(filelist, key=lambda r: extract_participant_id(r["file_id"]))
    fid_a = sorted_rows[0]["file_id"]
    fid_b = sorted_rows[1]["file_id"]
    pid_a = extract_participant_id(fid_a)
    pid_b = extract_participant_id(fid_b)

    interaction_id = extract_interaction_id(fid_a)
    vendor, session = extract_vendor_and_session(interaction_id)

    urls_a = _urls_for(fid_a, label, seamless_split)
    urls_b = _urls_for(fid_b, label, seamless_split)

    return {
        "seamless_split":       seamless_split,
        "interaction_id":       interaction_id,
        "vendor_id":            vendor,
        "session_id":           session,
        "prompt_hash":          extract_prompt_hash(interaction_id),
        "interaction_type":     meta["interaction_type"],
        "label":                label,
        "relationship":         rel["relationship"],
        "relationship_detail":  rel["relationship_detail"],
        "file_id_a":            fid_a,
        "file_id_b":            fid_b,
        "participant_a":        pid_a,
        "participant_b":        pid_b,
        "audio_url_a":          urls_a["audio"],
        "audio_url_b":          urls_b["audio"],
        "video_url_a":          urls_a["video"],
        "video_url_b":          urls_b["video"],
        "vad_url_a":            urls_a["vad"],
        "vad_url_b":            urls_b["vad"],
    }


def load_new_entries(both_annotated_dir: Path, single_annotated_dir: Path) -> list[dict]:
    """Collect pool entries from every interaction directory on disk."""
    entries: list[dict] = []
    for base in (both_annotated_dir, single_annotated_dir):
        if not base.exists():
            print(f"WARNING: {base} does not exist — skipping.", file=sys.stderr)
            continue
        for interaction_dir in sorted(base.iterdir()):
            if not interaction_dir.is_dir():
                continue
            entries.append(load_local_interaction(interaction_dir))
    return entries


def load_poc_entries(poc_manifest_path: Path) -> list[dict]:
    """
    Read the POC manifest and drop the 'split' column so POC rows are split
    afresh alongside the new rows. All other columns (URLs, relationship,
    seamless_split, etc.) are kept verbatim.
    """
    with poc_manifest_path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.pop("split", None)
    return rows


# =============================================================================
# Step 2 — Compute targets.
# =============================================================================

def compute_targets(total: int, train_frac: float, val_frac: float) -> dict[str, int]:
    """Integer target counts per split; the remainder is absorbed by test."""
    n_train = round(total * train_frac)
    n_val = round(total * val_frac)
    n_test = total - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test}


# =============================================================================
# Step 3 — Assign splits via participant-component LPT packing.
# =============================================================================

def assign_splits(
    pool: list[dict],
    targets: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[dict]], list[list[dict]]]:
    """
    Partition `pool` into train/val/test such that no participant appears in
    more than one split.

    Approach:
      1. Build a union-find over participants, unioning the two participants
         of each interaction.
      2. Group pool entries by their root → participant-connected components.
         Every interaction in a component must share a split with every other.
      3. Place components using Longest-Processing-Time bin-packing: sort
         components largest-first (ties broken by seed-shuffled order),
         assign each to the split with the most remaining capacity.

    Returns (assignments, components) — components is the ordered list of
    placed components, useful for diagnostics.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in pool:
        for p in (e["participant_a"], e["participant_b"]):
            parent.setdefault(p, p)
        union(e["participant_a"], e["participant_b"])

    by_root: dict[str, list[dict]] = defaultdict(list)
    for e in pool:
        by_root[find(e["participant_a"])].append(e)
    components = list(by_root.values())

    rng = random.Random(seed)
    rng.shuffle(components)                          # deterministic tiebreak
    components.sort(key=lambda c: -len(c))           # LPT: largest first

    assignments: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    remaining = dict(targets)
    for comp in components:
        s = max(("train", "val", "test"), key=lambda s: remaining[s])
        assignments[s].extend(comp)
        remaining[s] -= len(comp)

    return assignments, components


# =============================================================================
# Step 4 — Emit the manifest.
# =============================================================================

def write_manifest(
    assignments: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Write the manifest: every entry gets its freshly-assigned split."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for split in ("train", "val", "test"):
            for entry in sorted(assignments[split], key=lambda e: e["interaction_id"]):
                writer.writerow({**entry, "split": split})


# =============================================================================
# Step 5 — Verification + summary.
# =============================================================================

def verify_disjoint_participants(assignments: dict[str, list[dict]]) -> None:
    """Belt-and-suspenders check. Component assignment guarantees this, but
    crash loudly if a future refactor breaks it."""
    by_split: dict[str, set[str]] = {s: set() for s in ("train", "val", "test")}
    for s, entries in assignments.items():
        for e in entries:
            by_split[s].update((e["participant_a"], e["participant_b"]))

    splits = list(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise AssertionError(
                    f"Participants leaked between '{a}' and '{b}': {sorted(overlap)}"
                )


def print_summary(
    pool: list[dict],
    components: list[list[dict]],
    assignments: dict[str, list[dict]],
    targets: dict[str, int],
) -> None:
    print("\n" + "=" * 64)
    print("POOL")
    print("=" * 64)
    print(f"  Interactions: {len(pool):>4}")
    ss_counts = Counter(e["seamless_split"] for e in pool)
    for ss, n in sorted(ss_counts.items()):
        print(f"    seamless_split={ss:5s}: {n}")

    print("\n" + "=" * 64)
    print("PARTICIPANT-CONNECTED COMPONENTS")
    print("=" * 64)
    size_hist = Counter(len(c) for c in components)
    print(f"  Total components: {len(components)}")
    print(f"  Max component size: {max((len(c) for c in components), default=0)}")
    for sz, n in sorted(size_hist.items()):
        print(f"    size {sz:>3}: {n} component(s) ({sz * n} interaction(s))")

    print("\n" + "=" * 64)
    print("FINAL SPLITS")
    print("=" * 64)
    for split in ("train", "val", "test"):
        rows = assignments[split]
        rel = Counter(r["relationship"] for r in rows)
        ss = Counter(r["seamless_split"] for r in rows)
        print(f"  {split:5s}  n={len(rows):>3}  target={targets[split]:>3}  "
              f"relationship={dict(rel)}")
        print(f"         seamless_splits={dict(ss)}")
    print("=" * 64 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full-project turn-taking manifest. Pools POC rows with "
            "every interaction under both_annotated_interactions/ and "
            "single_annotated_interactions/, then assigns splits via "
            "participant-component bin-packing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--poc-manifest", default=DEFAULT_POC_MANIFEST,
                        help="Path to poc_manifest.csv. Default: %(default)s")
    parser.add_argument("--both-annotated-dir", default=DEFAULT_BOTH_ANNOTATED_DIR,
                        help="Directory of interactions with both-participant annotations. "
                             "Default: %(default)s")
    parser.add_argument("--single-annotated-dir", default=DEFAULT_SINGLE_ANNOTATED_DIR,
                        help="Directory of single-annotated interactions. "
                             "Default: %(default)s")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH,
                        help="Where to write manifest.csv. Default: %(default)s")
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC,
                        help="Train fraction (default: %(default)s).")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                        help="Val fraction (default: %(default)s).")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                        help="Test fraction (default: %(default)s). "
                             "Note: derived as 1 − train_frac − val_frac; used only "
                             "for the sum-to-1 sanity check.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default: %(default)s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary but do not write the manifest file.")
    args = parser.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        sys.exit(
            f"ERROR: --train-frac + --val-frac + --test-frac must sum to 1.0, "
            f"got {args.train_frac + args.val_frac + args.test_frac}"
        )

    poc_entries = load_poc_entries(Path(args.poc_manifest))
    new_entries = load_new_entries(
        Path(args.both_annotated_dir),
        Path(args.single_annotated_dir),
    )

    # Dedupe: if any on-disk interaction_id is already in the POC manifest,
    # prefer the POC row (its URLs are already enriched and known-good).
    poc_iids = {e["interaction_id"] for e in poc_entries}
    new_entries = [e for e in new_entries if e["interaction_id"] not in poc_iids]

    pool = poc_entries + new_entries
    # Drop hand-flagged interactions with empty upstream VAD/transcript
    # files before splits are computed (see EXCLUDED_INTERACTION_IDS).
    pool = [e for e in pool if e["interaction_id"] not in EXCLUDED_INTERACTION_IDS]
    targets = compute_targets(len(pool), args.train_frac, args.val_frac)

    assignments, components = assign_splits(pool, targets, args.seed)

    verify_disjoint_participants(assignments)
    print_summary(pool, components, assignments, targets)

    if args.dry_run:
        print(f"[dry-run] Would write manifest to: {args.output_path}")
    else:
        write_manifest(assignments, Path(args.output_path))
        print(f"Wrote manifest: {args.output_path}")


if __name__ == "__main__":
    main()
