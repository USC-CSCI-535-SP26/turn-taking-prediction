#!/usr/bin/env python3
"""Extract SMPL-H bundles for every participant in poc_manifest.csv.

Builds the minimal on-disk layout `enrich_with_smplh()` expects, then calls
it pointed at the project-root `smplh_features/` directory.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
POC = PROJ / "turn_taking_analysis/manifests/poc_manifest.csv"
OUTPUT = PROJ / "smplh_features_poc"
SCRIPTS = PROJ / "scripts"

sys.path.insert(0, str(SCRIPTS))
from download_annotated_interactions import enrich_with_smplh, extract_participant_id  # noqa: E402

OUTPUT.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    tmp_root = Path(tmp)
    with open(POC) as f:
        for row in csv.DictReader(f):
            iid = row["interaction_id"]
            label = row["label"]
            split = row["seamless_split"]
            fid_a, fid_b = row["file_id_a"], row["file_id_b"]
            pid_a, pid_b = extract_participant_id(fid_a), extract_participant_id(fid_b)

            idir = tmp_root / iid
            (idir / "interaction").mkdir(parents=True)
            (idir / "interaction" / "filelist_entries.json").write_text(json.dumps([
                {"file_id": fid_a, "label": label, "split": split},
                {"file_id": fid_b, "label": label, "split": split},
            ]))
            (idir / f"participant_a_{pid_a}").mkdir()
            (idir / f"participant_b_{pid_b}").mkdir()

    result = enrich_with_smplh(
        interactions_dir=str(tmp_root),
        output_dir=str(OUTPUT),
        num_workers=8,
    )
    print(f"\nwritten: {len(result['written'])}")
    print(f"skipped: {len(result['skipped'])}")
    print(f"failed:  {len(result['failed'])}")