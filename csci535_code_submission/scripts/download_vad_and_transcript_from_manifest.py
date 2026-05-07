from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -----------------------------------------------------------------------------
# Re-use helpers from the shared download script in the same directory.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                 # .../csci535-project
SHARED_SCRIPTS_DIR = THIS_FILE.parent  # download_annotated_interactions.py is a sibling
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (  # noqa: E402
    build_s3_url,
    download_file,
    extract_interaction_id,
)

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_MANIFEST_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "manifest.csv"
)
# Output roots — one flat directory per stream, paralleling the existing
# subset/{audio,video,wavlm,openface}/ conventions. See the module docstring
# for the "flat layout rationale" comment.
DEFAULT_VAD_OUTPUT_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "vad"
)
DEFAULT_TRANSCRIPT_OUTPUT_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "transcript"
)

# S3 category paths for each stream. These MUST match what `build_s3_url`
# consumes; see the shared helper's docstring. They also match what
# download_annotated_interactions.py:download_participant_files uses for VAD.
VAD_CATEGORY = "metadata/vad"
TRANSCRIPT_CATEGORY = "metadata/transcript"
VAD_EXT = "jsonl"
TRANSCRIPT_EXT = "jsonl"

VAD_LOCAL_PREFIX = ""
TRANSCRIPT_LOCAL_PREFIX = ""

# =============================================================================
# Manifest parsing — mirrors download_audio_from_manifest.py
# =============================================================================

REQUIRED_COLUMNS = {
    "split",              # POC split (bookkeeping + --filter-split)
    "seamless_split",     # Seamless's split — used for URL construction
    "label",              # "naturalistic" — used for URL construction
    "interaction_id",     # logging + cross-check
    "file_id_a",
    "file_id_b",
}

# URL-affecting fields validated per-row before we build an S3 URL. A bad
# value here silently turns into a 403/404 for every task from that row;
# validating up-front converts a cryptic S3-error deluge into a single
# clear skip-with-reason log line.
VALID_SEAMLESS_SPLITS = frozenset({"train", "dev", "test"})
VALID_LABELS = frozenset({"naturalistic"})   # POC is all naturalistic

def load_manifest(manifest_path: str) -> list[dict]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run build_manifest.py first to regenerate it."
        )

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: "
                f"{sorted(missing)}. If the manifest was built by an "
                f"older build_manifest.py, rebuild it."
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest {manifest_path} has no rows.")

    return rows

# =============================================================================
# Task descriptor
# =============================================================================

class DownloadTask:
    __slots__ = (
        "stream", "split", "interaction_id",
        "participant_role", "file_id", "url", "output_path",
    )

    def __init__(
        self,
        stream: str,
        split: str,
        interaction_id: str,
        participant_role: str,
        file_id: str,
        url: str,
        output_path: str,
    ):
        self.stream = stream
        self.split = split
        self.interaction_id = interaction_id
        self.participant_role = participant_role
        self.file_id = file_id
        self.url = url
        self.output_path = output_path

    def __repr__(self) -> str:
        return f"DownloadTask({self.stream}:{self.file_id})"

# =============================================================================
# Building the download task list
# =============================================================================

def _build_stream_task(
    stream: str,
    category: str,
    ext: str,
    local_prefix: str,
    label: str,
    seamless_split: str,
    poc_split: str,
    interaction_id: str,
    role: str,
    file_id: str,
    output_dir: str,
) -> DownloadTask:
    url = build_s3_url(
        label=label,
        split=seamless_split,
        category=category,
        file_id=file_id,
        ext=ext,
    )

    # --- Destination path (flat under the stream's output dir) ------------
    # Example: subset/vad/V00_S0691_I00000482_P0500.jsonl
    # `local_prefix` is "" by default; retained as a configurable in case a
    # future reader wants stream-tagged filenames.
    output_path = os.path.join(output_dir, f"{local_prefix}{file_id}.{ext}")

    return DownloadTask(
        stream=stream,
        split=poc_split,
        interaction_id=interaction_id,
        participant_role=role,
        file_id=file_id,
        url=url,
        output_path=output_path,
    )

def build_tasks(
    rows: list[dict],
    vad_output_dir: str,
    transcript_output_dir: str,
    filter_split: str | None,
    include_vad: bool,
    include_transcripts: bool,
) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    for row in rows:
        if filter_split and row["split"] != filter_split:
            continue

        label = row["label"]
        seamless_split = row["seamless_split"]
        poc_split = row["split"]

        # Validate URL-affecting fields up front. A bad value here silently
        # turns every task from this row into a 404 — cheaper to catch it
        # as a single skip-with-reason than to drown in S3 error spam.
        if seamless_split not in VALID_SEAMLESS_SPLITS:
            print(
                f"  ERROR: row {row['interaction_id']} has "
                f"seamless_split={seamless_split!r}; expected one of "
                f"{sorted(VALID_SEAMLESS_SPLITS)}. Skipping row.",
                file=sys.stderr,
            )
            continue
        if label not in VALID_LABELS:
            print(
                f"  ERROR: row {row['interaction_id']} has "
                f"label={label!r}; expected one of "
                f"{sorted(VALID_LABELS)}. Skipping row.",
                file=sys.stderr,
            )
            continue

        derived_iid = extract_interaction_id(row["file_id_a"])
        if derived_iid != row["interaction_id"]:
            print(
                f"  WARN: file_id_a {row['file_id_a']} implies interaction "
                f"{derived_iid} but manifest row says {row['interaction_id']}; "
                f"using derived value.",
                file=sys.stderr,
            )
        # Use the derived value as authoritative — build_poc_manifest.py
        # already does this when it writes the manifest, so re-deriving
        # here should match.
        interaction_id = derived_iid

        # Iterate both participants. Role is forced to alphabetical a/b per
        # the manifest convention (lower-sorted PID → "a"). This is the
        # same convention `download_annotated_interactions.py` enforces, so
        # directory names stay consistent across every download stream.
        for role in ("a", "b"):
            file_id = row[f"file_id_{role}"]
            if not file_id:
                print(
                    f"  WARN: row {row['interaction_id']} has empty "
                    f"file_id_{role}; skipping this participant.",
                    file=sys.stderr,
                )
                continue

            if include_vad:
                tasks.append(_build_stream_task(
                    stream="vad",
                    category=VAD_CATEGORY,
                    ext=VAD_EXT,
                    local_prefix=VAD_LOCAL_PREFIX,
                    label=label,
                    seamless_split=seamless_split,
                    poc_split=poc_split,
                    interaction_id=interaction_id,
                    role=role,
                    file_id=file_id,
                    output_dir=vad_output_dir,
                ))
            if include_transcripts:
                tasks.append(_build_stream_task(
                    stream="transcript",
                    category=TRANSCRIPT_CATEGORY,
                    ext=TRANSCRIPT_EXT,
                    local_prefix=TRANSCRIPT_LOCAL_PREFIX,
                    label=label,
                    seamless_split=seamless_split,
                    poc_split=poc_split,
                    interaction_id=interaction_id,
                    role=role,
                    file_id=file_id,
                    output_dir=transcript_output_dir,
                ))

    # Deterministic ordering so progress output looks stable across runs —
    # group by (split, interaction_id, participant_role, stream) so all
    # files for one participant appear together.
    tasks.sort(key=lambda t: (
        t.split, t.interaction_id, t.participant_role, t.stream,
    ))
    return tasks

# =============================================================================
# Execution
# =============================================================================

def already_have(output_path: str) -> bool:
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def run_task(task: DownloadTask, overwrite: bool, dry_run: bool) -> dict:
    if not overwrite and already_have(task.output_path):
        return {
            "task": task,
            "status": "skipped",
            "message": f"already on disk "
                       f"({os.path.getsize(task.output_path):,} bytes)",
        }

    if dry_run:
        return {"task": task, "status": "dry", "message": f"DRY: {task.url}"}

    os.makedirs(os.path.dirname(task.output_path), exist_ok=True)

    success, msg = download_file(task.url, task.output_path)
    if success:
        return {"task": task, "status": "ok", "message": msg}

    # The helper returns "NOT FOUND" on 403 per its contract (see its
    # docstring at download_annotated_interactions.py:427). We keep both
    # the "NOT FOUND" and bare "403" sniffing here to be robust against
    # message-format drift.
    if "NOT FOUND" in msg or "403" in msg:
        return {"task": task, "status": "missing", "message": msg}
    return {"task": task, "status": "failed", "message": msg}

def run_tasks(
    tasks: list[DownloadTask],
    num_workers: int,
    overwrite: bool,
    dry_run: bool,
) -> list[dict]:
    results: list[dict] = []

    if tasks and not dry_run:
        for parent in {os.path.dirname(t.output_path) for t in tasks}:
            os.makedirs(parent, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(run_task, t, overwrite, dry_run): t for t in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            task = result["task"]
            # Example line:
            #   [ok     ] val   vad          V00_S0691_I00000482_P0500  OK (1234 bytes)
            print(
                f"  [{result['status']:<7}] {task.split:<5} "
                f"{task.stream:<10} {task.file_id}  {result['message']}"
            )
            results.append(result)

    # Sort deterministically so the summary + any downstream log-diff is
    # stable. Matches the sort order in build_tasks.
    results.sort(key=lambda r: (
        r["task"].split, r["task"].interaction_id,
        r["task"].participant_role, r["task"].stream,
    ))
    return results

# =============================================================================
# Summary
# =============================================================================

def print_summary(
    results: list[dict],
    elapsed_s: float,
    dry_run: bool,
    strict_vad: bool,
) -> None:
    # counts[(stream, status)] = n
    counts: dict[tuple[str, str], int] = {}
    bytes_downloaded = 0
    for r in results:
        key = (r["task"].stream, r["status"])
        counts[key] = counts.get(key, 0) + 1
        if r["status"] == "ok":
            # download_file returns "OK (NNN bytes)" — parse it back out
            # so we can total up on-disk bytes. Same best-effort parse as
            # download_audio_from_manifest.py.
            try:
                bytes_downloaded += int(
                    r["message"].split("(")[1].split(" ")[0].replace(",", "")
                )
            except (IndexError, ValueError):
                pass

    print("\n" + "=" * 72)
    print("LABEL-SOURCES DOWNLOAD SUMMARY" + ("  [DRY-RUN]" if dry_run else ""))
    print("=" * 72)
    print(f"  Total tasks:  {len(results)}")

    # Per-stream breakdown so readers see VAD separately from transcripts.
    streams_seen = sorted({r["task"].stream for r in results})
    for stream in streams_seen:
        print(f"\n  Stream: {stream}")
        for status in ("ok", "skipped", "dry", "missing", "failed"):
            n = counts.get((stream, status), 0)
            if n:
                note = ""
                if stream == "transcript" and status == "missing":
                    note = "  (expected — optional per Seamless release; paper A.1.4)"
                if stream == "vad" and status == "missing":
                    note = "  (UNEXPECTED — VAD should always be on S3)"
                print(f"    {status:<10} {n}{note}")

    if bytes_downloaded:
        mib = bytes_downloaded / (1024 * 1024)
        print(f"\n  Total downloaded: {bytes_downloaded:,} bytes "
              f"({mib:,.2f} MiB)")
    print(f"  Elapsed:          {elapsed_s:,.1f} s")
    if strict_vad:
        print(f"  VAD policy:       strict (missing VAD → non-zero exit)")
    else:
        print(f"  VAD policy:       lenient (missing VAD logged only)")
    print("=" * 72)

# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download VAD and WhisperX-transcript metadata for every "
            "interaction in a POC manifest, into a flat layout consumed "
            "by build_labeled_windows_from_manifest.py. Reuses "
            "build_s3_url + download_file from csci535-project/scripts/"
            "download_annotated_interactions.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- I/O paths --------------------------------------------------------
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST_PATH,
        help="Path to the manifest CSV. Default: %(default)s",
    )
    parser.add_argument(
        "--vad-output-dir", default=DEFAULT_VAD_OUTPUT_DIR,
        help="Flat directory to save VAD .jsonl files into. Each file "
             "is named `{file_id}.jsonl`. Default: %(default)s",
    )
    parser.add_argument(
        "--transcript-output-dir", default=DEFAULT_TRANSCRIPT_OUTPUT_DIR,
        help="Flat directory to save WhisperX transcript .jsonl files "
             "into. Each file is named `{file_id}.jsonl`. "
             "Default: %(default)s",
    )

    # --- Stream selection --------------------------------------------------
    parser.add_argument(
        "--skip-vad", action="store_true",
        help="Do not download VAD files. Useful if VAD is already on disk.",
    )
    parser.add_argument(
        "--skip-transcripts", action="store_true",
        help="Do not download WhisperX transcripts. Useful for VAD-only runs.",
    )

    # --- Runtime plumbing --------------------------------------------------
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of concurrent download threads. Default: %(default)s.",
    )
    parser.add_argument(
        "--filter-split", choices=["train", "val", "test"], default=None,
        help="If set, only download rows whose POC `split` column matches. "
             "Handy for staged debugging: pull val first, confirm the "
             "layout works end-to-end, then pull train + test.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download files that already exist on disk. Default is "
             "to skip them (idempotent resume).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print URLs that would be downloaded without fetching anything.",
    )
    # argparse.BooleanOptionalAction (Python 3.9+) registers the flag pair
    # `--strict-vad / --no-strict-vad` atomically. Default is strict because
    # every Seamless participant should have a VAD JSONL; a 403 is a real
    # anomaly. --no-strict-vad downgrades it to a warning + exit code 3.
    parser.add_argument(
        "--strict-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat a missing VAD file on S3 as a hard failure. "
             "Default: on. Use --no-strict-vad to downgrade to a warning.",
    )

    args = parser.parse_args()

    # --- Sanity: at least one stream must be requested. -------------------
    if args.skip_vad and args.skip_transcripts:
        print("ERROR: both --skip-vad and --skip-transcripts given — "
              "nothing to download.", file=sys.stderr)
        return 1

    # --- Load manifest -----------------------------------------------------
    print(f"Loading manifest:  {args.manifest}")
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  {len(rows):,} interactions in manifest "
          f"({2 * len(rows):,} participant entries)")

    # --- Build the task list ----------------------------------------------
    tasks = build_tasks(
        rows=rows,
        vad_output_dir=args.vad_output_dir,
        transcript_output_dir=args.transcript_output_dir,
        filter_split=args.filter_split,
        include_vad=not args.skip_vad,
        include_transcripts=not args.skip_transcripts,
    )
    if args.filter_split:
        print(f"  Filtered to POC split='{args.filter_split}': "
              f"{len(tasks):,} tasks")
    if not tasks:
        print("Nothing to download. Exiting.")
        return 0

    streams_requested = []
    if not args.skip_vad:
        streams_requested.append(f"vad → {args.vad_output_dir}")
    if not args.skip_transcripts:
        streams_requested.append(f"transcript → {args.transcript_output_dir}")

    print(f"Streams:")
    for s in streams_requested:
        print(f"  {s}")
    print(f"Workers:           {args.num_workers}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")
    print(f"Starting {len(tasks):,} download task(s)...\n")

    # --- Execute ----------------------------------------------------------
    start = time.time()
    results = run_tasks(
        tasks,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    elapsed = time.time() - start

    print_summary(results, elapsed, args.dry_run, args.strict_vad)

    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_missing_vad = sum(
        1 for r in results
        if r["status"] == "missing" and r["task"].stream == "vad"
    )
    n_missing_transcript = sum(
        1 for r in results
        if r["status"] == "missing" and r["task"].stream == "transcript"
    )

    if n_failed:
        return 2
    if n_missing_vad and args.strict_vad:
        # Unexpected — VAD should be universal. Surface loudly.
        print(f"\nERROR: {n_missing_vad} VAD file(s) returned 403 on S3. "
              f"Run with --no-strict-vad to downgrade to a warning.",
              file=sys.stderr)
        return 2
    if n_missing_vad or n_missing_transcript:
        # Non-fatal misses (typically transcripts).
        return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
