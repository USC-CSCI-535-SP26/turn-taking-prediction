#!/usr/bin/env python3
"""
download_vad_and_transcript_from_manifest.py

Download the two metadata streams that the turn-taking label generator
consumes — per-participant **Silero VAD** (voice activity detection,
100 Hz speech/non-speech segments) and per-participant **WhisperX
word-aligned transcripts** — for every interaction listed in the POC
manifest (`manifests/manifest.csv`).

Why these two streams, and only these two?
------------------------------------------
Our 3-class per-window label schema {HOLD, YIELD, BACKCHANNEL} is built
from turn boundaries. To find turn boundaries we need:

  1. **VAD**: who is speaking at each moment. `generate_timestamps_by_turn.py`
     (in the parent project's `csci535-project/scripts/`) merges
     consecutive same-participant VAD segments into "turns" and flags
     per-turn overlap. A turn boundary occurs when the floor transfers
     from one participant to the other. The VAD is the canonical source
     of those spans. We also reuse the `overlapping` flag for the
     erstwhile-INTERRUPT exclusion filter (spec §5 task 3).

  2. **WhisperX transcripts**: word-level start/end timestamps and the
     word strings themselves. BACKCHANNEL labels are **lexical** — a
     short utterance like "mm-hmm" / "yeah" / "right" by the non-floor
     participant that does NOT take the floor. Pure VAD cannot
     distinguish a backchannel ("uh-huh") from a very short interjection
     ("but—") from a competitive-interrupt start that happened to be
     clipped by Silero. The transcript's word list is required to run
     the lexical backchannel filter.

Audio (.wav) is already on disk at `subset/audio/` (via
`download_audio_from_manifest.py`). Video is at `subset/video/`. OpenFace
features are at `subset/openface/` and WavLM features at `subset/wavlm/`.
This script completes the data-side of the pipeline by fetching the
per-interaction **label source** files that let us emit per-window
ground truth.

Why NOT reuse the parent project's `annotated_interactions/`?
-------------------------------------------------------------
The parent project's `annotated_interactions/` directory was populated
by `csci535-project/scripts/download_annotated_interactions.py`, which
selects the 100 naturalistic interactions where BOTH participants have
3P annotations (3P-IS / 3P-R / 3P-V). Our project manifest
(`manifests/manifest.csv`, built by `build_manifest.py` from the
historical POC + every interaction under `both_annotated_interactions/`
+ `single_annotated_interactions/`) is a different, larger sample with
its own participant-disjoint splits. There is NO guarantee the parent
project's selection overlaps ours, so we download VAD/transcript
material into a dedicated `subset/` directory rather than trying to
splice two differently-scoped interaction populations together.

Output layout (flat per-stream, mirroring the convention established by
`subset/audio/`, `subset/video/`, `subset/wavlm/`, `subset/openface/`):

    turn_taking_analysis/
      subset/
        vad/
          V00_S0691_I00000482_P0500.jsonl
          V00_S0691_I00000482_P0840.jsonl
          V00_S0743_I00000483_P0885.jsonl
          ...
        transcript/
          V00_S0691_I00000482_P0500.jsonl
          V00_S0691_I00000482_P0840.jsonl
          V00_S0743_I00000483_P0885.jsonl
          ...

Flat layout rationale
---------------------
A Seamless `file_id` of the form `V{vendor}_S{session}_I{hash}_P{pid}`
already encodes every bit of the {vendor, session, interaction,
participant} tuple. Nesting by interaction_id or participant adds no
information and complicates downstream globbing (we already do flat
globs for `subset/audio/*.wav`, `subset/wavlm/*.npy`, etc.). Downstream
code can always recover the interaction_id via
`extract_interaction_id(file_id)` and group flat files by interaction
when it needs a nested view.

Downstream consumer
-------------------
The active consumer of the emitted `subset/vad/*.jsonl` and
`subset/transcript/*.jsonl` files is
`build_labeled_windows_from_manifest.py`, which reads them directly
via its own `load_vad_jsonl` / `load_transcript_jsonl` helpers — no
adapter is required.

Historical note: the parent project's `generate_timestamps_by_turn.py`
expects a nested `{interaction_id}/participant_{a|b}_{pid}/...` layout
and is NOT called from this project's label-generation path. If a
future extension reintroduces it, a ~10-line grouping wrapper on
`subset/vad/*.jsonl` (key: `extract_interaction_id(filename)`) is all
that would be needed; the flat output here is deliberately the primary
contract.

S3 URL scheme (reproduced from the shared helpers, not hand-rolled)
-------------------------------------------------------------------
The Seamless S3 bucket serves files under the URL pattern:

    https://dl.fbaipublicfiles.com/seamless_interaction/{label}/{split}/{category}/{file_id}.{ext}

Per the paper §3.4 (public release layout) and as encoded in
`csci535-project/scripts/download_annotated_interactions.py:build_s3_url`:

  - label:    "naturalistic"        (always, for our manifest)
  - split:    "train" | "dev" | "test" (Seamless's own split — we
                                       read it from the manifest's
                                       `seamless_split` column, NOT
                                       our POC split)
  - category: "metadata/vad"        — VAD JSONL
              "metadata/transcript" — WhisperX JSONL
  - file_id:  "V{vendor}_S{session}_I{hash}_P{pid}"
  - ext:      "jsonl" for both

WhisperX transcripts are flagged as OPTIONAL in the Seamless release
(see `download_interaction_transcripts.py` header comment: "not every
file_id has one, and missing transcripts return HTTP 403"). The paper's
Appendix A.1.4 (p. 54) documents the known short-utterance
timestamp-drift issue in these WhisperX outputs; we accept that at POC
scope per spec §8.1. HTTP 403 is therefore treated as "transcript not
on S3 for this participant" — a logged warning, not a fatal error. VAD
files, by contrast, are mandatory — every Seamless participant has a
VAD JSONL, and a 403 on VAD is a hard failure that should be surfaced.

Reuse of shared helpers (mirrors `download_audio_from_manifest.py`)
-------------------------------------------------------------------

    build_s3_url()          — canonical S3 URL construction; single
                              source of truth for the URL scheme.
    download_file()         — HTTP-to-disk writer with 403-as-missing
                              semantics, parent-dir autocreate, and
                              byte-accurate OK messages.
    extract_interaction_id()— file_id -> interaction_id; used only for
                              per-row cross-check against the manifest's
                              `interaction_id` column. (The flat output
                              layout does not nest by interaction.)

Per `download_audio_from_manifest.py`'s precedent, we do NOT read the
manifest's precomputed `vad_url_a`/`vad_url_b` columns — URLs are
regenerated via `build_s3_url` so the canonical URL scheme remains in
one place. If the S3 path ever changes, only `build_s3_url` needs to
update.

Usage
-----
    # Default: fetch VAD + transcripts for all 24 dyads. VAD lands in
    # turn_taking_analysis/subset/vad/, transcripts in
    # turn_taking_analysis/subset/transcript/ (both flat).
    python download_vad_and_transcript_from_manifest.py

    # Only the val split (staged debugging)
    python download_vad_and_transcript_from_manifest.py --filter-split val

    # Only VAD (skip transcripts — useful if we're in a hurry and
    # BACKCHANNEL labels will be revisited later)
    python download_vad_and_transcript_from_manifest.py --skip-transcripts

    # Only transcripts (skip VAD — useful if VAD is already on disk)
    python download_vad_and_transcript_from_manifest.py --skip-vad

    # Print URLs without downloading
    python download_vad_and_transcript_from_manifest.py --dry-run

    # Force re-download of everything (ignore on-disk cache)
    python download_vad_and_transcript_from_manifest.py --overwrite

Exit codes (match `download_audio_from_manifest.py`)
----------------------------------------------------
    0 — all tasks succeeded (or were skipped as already on disk).
    1 — hard setup error (missing manifest, bad column, etc.).
    2 — at least one download hit a non-HTTP-403 failure, OR at least
        one VAD file returned HTTP 403 under --strict-vad (the default).
        In the latter case the error message names the miss count.
    3 — VAD was missing but --no-strict-vad downgraded it to a warning,
        and/or some transcripts were missing (expected per paper
        Appendix A.1.4). Returned instead of 0 so the caller can
        distinguish "everything clean" from "some missing".
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# -----------------------------------------------------------------------------
# Re-use helpers from csci535-project/scripts. This script lives at
# turn_taking_analysis/scripts/, so we walk two directories up to reach the
# shared-scripts directory. This is exactly the pattern
# `download_audio_from_manifest.py`, `extract_wavlm_from_manifest.py`, and
# `build_poc_manifest.py` already use — keeping it here means the sys.path
# hack is uniform across all project-root scripts.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                 # .../csci535-project
SHARED_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
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
# download_annotated_interactions.py:download_participant_files uses for VAD
# and what download_interaction_transcripts.py uses for transcripts.
VAD_CATEGORY = "metadata/vad"
TRANSCRIPT_CATEGORY = "metadata/transcript"
VAD_EXT = "jsonl"
TRANSCRIPT_EXT = "jsonl"

# Local filename prefixes. Empty: the stream is implied by the containing
# directory (subset/vad/ vs subset/transcript/), which matches the naming
# pattern used by our other flat feature directories — subset/audio/ writes
# `{file_id}.wav` with no prefix, subset/wavlm/ writes `{file_id}.npy`, etc.
# Prefix constants are retained as variables (rather than inlined as "")
# so a future maintainer who wants to re-enable prefixes can set them in
# one place.
VAD_LOCAL_PREFIX = ""
TRANSCRIPT_LOCAL_PREFIX = ""


# =============================================================================
# Manifest parsing — mirrors download_audio_from_manifest.py
# =============================================================================

# We need everything required to regenerate both URLs (label + seamless_split +
# file_id) plus the bookkeeping columns used in per-line progress output.
# `split` is our POC split (train/val/test). `seamless_split` is Seamless's
# own split (train/dev/test) and it's the one that appears in the S3 URL.
# Participant IDs are intentionally NOT required — the flat output layout
# encodes the participant in the file_id itself.
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
    """
    Load manifest.csv into a list of dict rows.

    Fails fast if the manifest is missing or schema has drifted. This
    is the same defensive posture as `load_manifest` in
    `download_audio_from_manifest.py` — if the manifest changes shape,
    we want a clear error here, not rows silently producing None
    downstream.
    """
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
    """
    One file to fetch: URL + destination path + enough bookkeeping tags to
    make per-line progress output readable.

    Fields:
        stream           — "vad" or "transcript". Used by the progress
                           printer and by the summary logic to route
                           missing-on-S3 results into mandatory-vs-
                           optional buckets.
        split            — our POC split (train/val/test). Used for
                           progress output + sort key only; the S3 URL
                           is built from the manifest's `seamless_split`
                           column, not from this field.
        interaction_id   — kept purely as a sort key so progress output
                           groups by dyad. The flat output layout does
                           not nest by interaction.
        participant_role — "a" or "b" from the manifest column suffix.
                           Sort-key only. The output path does NOT
                           encode role: directory layout is flat
                           (`subset/vad/{file_id}.jsonl`,
                           `subset/transcript/{file_id}.jsonl`). See
                           the module docstring "Flat layout rationale"
                           for why.
        file_id          — Seamless participant file_id, e.g.
                           `V00_S0691_I00000482_P0500`. Surfaces in
                           every progress line.
        url / output_path — the actual fetch target + destination.
    """

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
    """
    Build one DownloadTask for a single (stream, participant) pair.

    Pulling this out into a helper avoids duplicating the VAD and
    transcript task-construction logic in `build_tasks` — the only
    differences between the two are the S3 category, the local filename
    prefix, the stream tag, and the output directory, all of which are
    arguments here.
    """
    # --- S3 URL via the shared helper (single source of truth). -----------
    # file_id is passed BARE to build_s3_url — Seamless's S3 bucket stores
    # these files as `{category}/{file_id}.{ext}` with no prefix. The
    # `vad_` / `transcript_` prefix is a LOCAL-filename convention only,
    # kept for self-describing filenames (see VAD_LOCAL_PREFIX comment).
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
    """
    Expand manifest rows into a flat list of DownloadTasks — one task per
    (stream, participant) pair. A full manifest row can therefore emit up
    to 4 tasks (VAD × 2 participants + transcript × 2 participants).

    Each stream lands in its own flat output directory (vad_output_dir
    for VAD, transcript_output_dir for transcripts), preserving the
    subset/{audio,video,wavlm,openface}/ flat-layout convention.

    --filter-split (POC split) prunes rows that don't match. --skip-vad
    and --skip-transcripts prune streams.
    """
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

        # Cross-check that the interaction_id embedded in file_id_a matches
        # the manifest's `interaction_id` column. This guards against an
        # accidental hand-edit of the manifest that decouples the two.
        # It's the same defensive check that `download_audio_from_manifest.py`
        # runs — keeping it here maintains that property across the
        # family of manifest-driven downloaders.
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
    """
    True iff the destination file exists on disk AND is nonempty.

    A 0-byte file from an interrupted prior run is treated as missing so
    the next invocation re-downloads it. This matches
    `download_audio_from_manifest.py` and `extract_wavlm_from_manifest.py`.
    """
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def run_task(task: DownloadTask, overwrite: bool, dry_run: bool) -> dict:
    """
    Execute one DownloadTask. Actual HTTP work is delegated to
    `download_file` (the shared helper), which handles:

      - HTTP 403 = "not on S3" → returned as (False, "... NOT FOUND ...")
      - Parent-directory auto-create
      - Raw-byte write (no JSON parsing — the helper is format-agnostic)
      - Timeout = 120 s (plenty for JSONL files, which are small)

    We re-wrap its (success, message) tuple into a status-coded dict so
    the summary printer can group results cleanly:

        "skipped"  — on-disk already; honoured unless --overwrite.
        "dry"      — --dry-run; URL printed but not fetched.
        "ok"       — downloaded successfully.
        "missing"  — S3 returned 403 (expected for transcripts; fatal
                     for VAD unless --strict-vad is disabled).
        "failed"   — any other error; the message carries detail.
    """
    if not overwrite and already_have(task.output_path):
        return {
            "task": task,
            "status": "skipped",
            "message": f"already on disk "
                       f"({os.path.getsize(task.output_path):,} bytes)",
        }

    if dry_run:
        return {"task": task, "status": "dry", "message": f"DRY: {task.url}"}

    # Make sure the stream's output directory exists before calling the
    # helper. `download_file` DOES call os.makedirs itself, but doing it
    # here lets us print a clearer failure if the filesystem is hostile
    # (read-only, out of space, etc.). Since the layout is flat, this is
    # a single mkdir per stream rather than per-participant.
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
    """
    Run all tasks concurrently and return the per-task results.

    Parallelism is bounded by --num-workers (default 4). Higher is fine
    for these tiny JSONL files (each is well under 1 MB), but Meta's S3
    bucket does not advertise any specific concurrency limit — 4 is a
    polite default that mirrors the parent project's scripts.
    """
    results: list[dict] = []

    # Pre-create the per-stream output directories so a --dry-run still
    # materializes the skeleton. `download_file` will also mkdir, but this
    # makes the first logs cleaner. Under the flat layout, every task in a
    # given stream shares the same parent directory, so `set` dedupes to
    # at most 2 mkdirs.
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
    """
    Print counts by (stream, status). Transcripts missing on S3 is a
    non-fatal, expected condition (paper Appendix A.1.4 acknowledges
    transcripts are optional). VAD missing on S3 is unexpected — we
    surface it at report time and, under --strict-vad (default), return
    an error exit code.
    """
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

    # --- Exit code --------------------------------------------------------
    # Mirrors download_audio_from_manifest.py's conventions:
    #   0 — clean
    #   2 — any hard failure (connection error, unknown error, etc.)
    #   3 — some "missing"; treatment varies by stream + --strict-vad.
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
