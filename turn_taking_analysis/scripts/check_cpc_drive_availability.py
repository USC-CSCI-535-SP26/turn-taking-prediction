#!/usr/bin/env python3
"""Cross-reference manifest.csv against CPC tarballs already in Drive, then
re-split the available subset using the same participant-component LPT
bin-packing logic as build_manifest.py.

Each tarball is named `<interaction_id>.tar` and contains both participants'
spliced dirs (built by upload_cpc.sh). So tarball-existence ⇒ both-participants
data available for use.

Splits are recomputed (not preserved from manifest.csv): no participant appears
in more than one split, ratios target the configured train/val/test fractions
of the *available* pool. This matters because the upload is in progress — the
available pool grows over time, and naively preserving the original splits
would leave val/test arbitrarily small.

Outputs:
- Stdout summary: count, per-split breakdown, missing/orphan diagnostics.
- `manifests/manifest_cpc_available.csv`: drop-in subset manifest with
  freshly-computed splits, preserving the full 20-column schema. Set this as
  `MANIFEST_PATH` in fusion_experiments.ipynb.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Reuse the split logic from build_manifest.py (same dir).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from build_manifest import (                                          # noqa: E402
    MANIFEST_COLUMNS,
    assign_splits,
    compute_targets,
    print_summary,
    verify_disjoint_participants,
)


def list_drive_tarballs(remote: str, drive_path: str) -> set[str]:
    """Return the set of interaction_ids that have a `<iid>.tar` in Drive."""
    try:
        result = subprocess.run(
            ['rclone', 'lsf', f'{remote}:{drive_path}',
             '--files-only', '--include', '*.tar'],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        sys.exit('ERROR: rclone not found on PATH. Install with `brew install rclone`.')
    except subprocess.CalledProcessError as e:
        sys.exit(f'ERROR: rclone lsf failed: {e.stderr.strip()}')
    return {
        line[:-len('.tar')]
        for line in result.stdout.splitlines()
        if line.endswith('.tar')
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote',     default='seamless_cpc',
                   help='rclone remote name (default: seamless_cpc)')
    p.add_argument('--drive-path', default='535_project_data/data/spliced/cpc_tar',
                   help='path under remote where tarballs live')
    p.add_argument('--manifest',   default=None,
                   help='source manifest (default: ../manifests/manifest.csv)')
    p.add_argument('--out',        default=None,
                   help='filtered output manifest '
                        '(default: ../manifests/manifest_cpc_available.csv)')
    p.add_argument('--train-frac', type=float, default=0.70)
    p.add_argument('--val-frac',   type=float, default=0.15)
    p.add_argument('--test-frac',  type=float, default=0.15,
                   help='used only for sum-to-1 sanity check (test = remainder)')
    p.add_argument('--seed',       type=int,   default=42,
                   help='controls component tie-breaking only; default 42 '
                        '(matches build_manifest.py default)')
    args = p.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        sys.exit(f'ERROR: train+val+test fracs must sum to 1.0; got '
                 f'{args.train_frac + args.val_frac + args.test_frac}')

    proj = HERE.parent
    manifest_path = Path(args.manifest) if args.manifest else proj / 'manifests' / 'manifest.csv'
    out_path      = Path(args.out)      if args.out      else proj / 'manifests' / 'manifest_cpc_available.csv'

    # --- Step 1: list tarballs currently in Drive ----------------------------
    print(f'Listing {args.remote}:{args.drive_path} ...', flush=True)
    available_iids = list_drive_tarballs(args.remote, args.drive_path)
    print(f'Drive has {len(available_iids)} CPC tarballs')

    # --- Step 2: read manifest, filter to available subset -------------------
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    print(f'Manifest has {len(all_rows)} interactions')

    manifest_iids = {r['interaction_id'] for r in all_rows}
    available_rows = [r for r in all_rows if r['interaction_id'] in available_iids]
    missing_rows   = [r for r in all_rows if r['interaction_id'] not in available_iids]
    orphans        = available_iids - manifest_iids   # in Drive but not in manifest

    if not available_rows:
        sys.exit('ERROR: no manifest interactions have CPC tarballs in Drive yet. '
                 'Wait for upload_cpc.sh to make progress, then re-run.')

    # --- Step 3: re-split the available subset (LPT participant-components) --
    # Strip the existing split column so build_manifest's assigner doesn't see
    # stale assignments. The assigner only needs interaction_id, participant_a,
    # participant_b — but it preserves all other columns through to write.
    pool = [{**r, 'split': ''} for r in available_rows]
    targets = compute_targets(len(pool), args.train_frac, args.val_frac)
    assignments, components = assign_splits(pool, targets, args.seed)
    verify_disjoint_participants(assignments)

    # --- Step 4: write filtered manifest with new splits ---------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for split in ('train', 'val', 'test'):
            for entry in sorted(assignments[split], key=lambda e: e['interaction_id']):
                w.writerow({**entry, 'split': split})

    # --- Step 5: report ------------------------------------------------------
    pct = 100 * len(available_rows) / max(len(all_rows), 1)
    print()
    print(f'=== CPC availability summary ===')
    print(f'Available for use: {len(available_rows)} / {len(all_rows)} ({pct:.1f}%)')

    # build_manifest's full diagnostic block (component sizes + per-split rels).
    print_summary(pool, components, assignments, targets)

    print(f'Wrote sub-manifest: {out_path}')
    print(f'  → set MANIFEST_PATH to this file in fusion_experiments.ipynb')

    if orphans:
        print(f'\nNOTE: {len(orphans)} tarballs in Drive don\'t match any '
              f'manifest interaction_id (legacy / wrong files in dest):')
        for iid in sorted(orphans)[:5]:
            print(f'  {iid}')
        if len(orphans) > 5:
            print(f'  ... and {len(orphans)-5} more')

    if missing_rows:
        print(f'\n{len(missing_rows)} interactions still pending upload (sample):')
        for r in missing_rows[:8]:
            print(f'  {r["split"]:5s}  {r["interaction_id"]}')
        if len(missing_rows) > 8:
            print(f'  ... and {len(missing_rows)-8} more')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
