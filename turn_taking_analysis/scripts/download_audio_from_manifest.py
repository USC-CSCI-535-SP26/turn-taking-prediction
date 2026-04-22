#!/usr/bin/env python3
"""
download_audio_from_manifest.py

Download per-participant .wav files for every interaction listed in a
turn-taking manifest CSV (as produced by build_manifest.py) and save
them under a flat directory.

Default layout
--------------
    turn_taking_analysis/
      manifests/manifest.csv         <-- input
      subset/
        audio/
          V00_S0743_I00000483_P0885.wav  <-- output
          V00_S0743_I00000483_P0888.wav
          V01_S0325_I00000314_P1666.wav
          ...

Scope
-----
Originally built for the 24-dyad POC (48 wavs, ~2-5 GB). The project has
since scaled up to all 467 naturalistic ipc_conversation interactions
(~934 wavs, ~50-100 GB over a multi-hour run). The defaults here still
target the POC paths — override `--manifest` and `--output-dir` when
running against the full-scale manifest. The implementation below
includes scale-robustness features (retry, streamed writes, temp-file
rename, Content-Length integrity check, tqdm progress, transfer-stall
detection) that were added specifically because a multi-hour download
over hundreds of files WILL experience transient failures.

Reuse of shared helpers
-----------------------
This script orchestrates around helpers in
`csci535-project/scripts/download_annotated_interactions.py`:

    build_s3_url()          — canonical S3 URL construction
    extract_interaction_id()— file_id -> interaction_id
    extract_participant_id()— file_id -> participant_id

Note that we no longer use the shared `download_file`; at full-project
scale it proved too brittle (no retries, no timeout on body reads, no
partial-file detection, full-response buffering in memory). We ship a
local `_download_file_robust` with those properties here. The shared
helper is still used by sibling scripts that don't have the same scale
concerns.

The script intentionally does NOT read the manifest's precomputed
`audio_url_a` / `audio_url_b` columns; it regenerates URLs via
build_s3_url() so the canonical URL scheme lives in one place. If the
S3 path layout ever changes, only build_s3_url needs to update.

Why a flat directory
--------------------
Every Seamless file_id already encodes {vendor, session, interaction,
participant} uniquely, so nesting by interaction_id adds no information
and complicates downstream globbing for WavLM extraction. Downstream
scripts can recover the interaction_id via extract_interaction_id() or
by joining back to the manifest on file_id.

Behavior
--------
- Reads (split, seamless_split, label, file_id_a, file_id_b, interaction_id)
  from the manifest. URLs are regenerated via build_s3_url.
- For each row, queues both participant wavs for download.
- Writes each file to `{output_path}.part` first, then atomically renames
  on success. On restart, any `.part` left over from a killed prior run
  is deleted in `already_have` so the task is re-attempted cleanly.
- Skips files that already exist at the FINAL path (not .part) with
  nonempty size. Idempotent resume survives Ctrl-C, system crashes, and
  connection resets without ever masking a partial download as complete.
- Downloads concurrently via ThreadPoolExecutor.
- Retries transient errors (5xx, connection reset, socket timeout, DNS
  hiccup) with exponential backoff. HTTP 403 / 404 are never retried —
  they indicate the file is genuinely not on S3.
- Verifies downloaded byte count against the server's Content-Length (if
  present). Size mismatch → treated as transient error and retried.
- Streams response body in 64 KiB chunks directly to disk. Memory use is
  O(chunk_size) per worker, not O(file_size).
- Socket timeout of 120 s on any single read — catches stalled-stream
  hangs where bytes stop arriving mid-transfer.
- Progress bar via tqdm; per-task status lines use `tqdm.write()` so the
  bar doesn't get mangled.
- At end of run, any HTTP-403 "missing audio" results get a loud
  warning block — audio is a primary modality and an S3 gap is a data-
  integrity issue worth escalating.

Usage
-----
    # Default: download everything in the manifest to .../subset/audio/
    python download_audio_from_manifest.py

    # Only the val split (staged debugging)
    python download_audio_from_manifest.py --filter-split val

    # Print URLs without downloading
    python download_audio_from_manifest.py --dry-run

    # Force re-download (ignore on-disk cache)
    python download_audio_from_manifest.py --overwrite

    # Crank parallelism + tune retries for a long, flaky network
    python download_audio_from_manifest.py --num-workers 16 --max-retries 5

Exit codes
----------
    0 — all tasks completed successfully (any status in {ok, skipped, dry}).
    1 — setup error (missing manifest, bad columns, unreadable output dir).
    2 — at least one task failed permanently (retries exhausted or hard error).
    3 — at least one task returned "missing" (HTTP 403). Non-fatal, but
        for audio specifically it signals a data-integrity issue — see
        the loud warning block printed before exit.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
import urllib.request


# -----------------------------------------------------------------------------
# Re-use helpers from csci535-project/scripts. This script lives at
# turn_taking_analysis/scripts/, so we resolve the sibling directory and
# insert it on sys.path — matching the pattern in build_poc_manifest.py.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                 # .../csci535-project
SHARED_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

# Note: we DON'T import download_file from the shared helper anymore;
# we ship our own _download_file_robust below. extract_* helpers and
# build_s3_url are still reused.
from download_annotated_interactions import (  # noqa: E402
    build_s3_url,
    extract_interaction_id,
    extract_participant_id,
)


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_MANIFEST_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "manifest.csv"
)
DEFAULT_OUTPUT_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "audio"
)

# Category + extension are fixed for this script — we only fetch wavs.
S3_CATEGORY = "audio"
FILE_EXT = "wav"

# Robust-download tunables. These are the values you're most likely to
# want to change at full-project scale over a bad network.
DEFAULT_CHUNK_SIZE = 64 * 1024              # 64 KiB per streamed chunk
DEFAULT_SOCKET_TIMEOUT_S = 120.0            # no-data-for-N-seconds → socket timeout
DEFAULT_MAX_RETRIES = 3                     # per-file attempt budget
DEFAULT_INITIAL_BACKOFF_S = 2.0             # doubles per retry; caps at MAX_BACKOFF_S
DEFAULT_MAX_BACKOFF_S = 30.0
DEFAULT_USER_AGENT = "csci535-turn-taking-downloader/1.0"


# =============================================================================
# Manifest parsing
# =============================================================================

# We regenerate URLs via build_s3_url, so the manifest's audio_url_* columns
# are not consumed — but we do need the split/label/file_id fields.
REQUIRED_COLUMNS = {
    "split",              # our turn-taking split: train/val/test
    "seamless_split",     # Seamless's own split: train/dev/test (for S3 URL)
    "label",              # "naturalistic" (for S3 URL)
    "interaction_id",     # for logging only
    "file_id_a",
    "file_id_b",
}


def load_manifest(manifest_path: str) -> list[dict]:
    """
    Load the turn-taking manifest CSV into a list of dict rows.

    Validates that every required column is present — surfaces schema
    drift immediately rather than letting rows silently produce None.
    Works for any manifest that matches the REQUIRED_COLUMNS contract
    (POC's 24-dyad manifest, the 467-interaction full-project manifest,
    or any subset/superset thereof).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run build_manifest.py first."
        )

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: "
                f"{sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest {manifest_path} has no rows.")

    return rows


# =============================================================================
# Building the download task list
# =============================================================================

class DownloadTask:
    """One wav to fetch: url + destination path + bookkeeping tags.

    Tags come from the manifest row (split, interaction_id) plus the
    shared helpers (file_id -> interaction_id / participant_id via the
    extract_* helpers). Storing them on the task makes the per-line
    progress output readable without re-deriving anything downstream.
    """

    __slots__ = (
        "split", "interaction_id", "participant_id",
        "file_id", "url", "output_path",
    )

    def __init__(self, split: str, interaction_id: str, participant_id: str,
                 file_id: str, url: str, output_path: str):
        self.split = split
        self.interaction_id = interaction_id
        self.participant_id = participant_id
        self.file_id = file_id
        self.url = url
        self.output_path = output_path

    def __repr__(self) -> str:
        return f"DownloadTask({self.file_id})"


def build_tasks(
    rows: list[dict],
    output_dir: str,
    filter_split: str | None,
) -> list[DownloadTask]:
    """
    Expand manifest rows into one DownloadTask per participant wav
    (two tasks per row).

    URLs are generated via build_s3_url from (label, seamless_split,
    file_id) — NOT read from the manifest's audio_url_* columns — so
    the S3 URL scheme stays canonicalized in the shared helper.

    Tag fields (interaction_id, participant_id) are derived from file_id
    via the shared extract_* helpers, again to keep the parsing logic
    in one place. They could be read from the manifest directly, but
    routing through the helpers ensures consistency with the rest of
    the scripts/ tooling.
    """
    tasks: list[DownloadTask] = []
    for row in rows:
        if filter_split and row["split"] != filter_split:
            continue

        label = row["label"]
        seamless_split = row["seamless_split"]

        for side in ("a", "b"):
            file_id = row[f"file_id_{side}"]
            if not file_id:
                print(f"  WARN: row {row['interaction_id']} has empty "
                      f"file_id_{side}; skipping.", file=sys.stderr)
                continue

            # Use the shared URL builder for canonical S3 construction.
            url = build_s3_url(
                label=label,
                split=seamless_split,
                category=S3_CATEGORY,
                file_id=file_id,
                ext=FILE_EXT,
            )

            # Use shared extractors so interaction_id/participant_id parsing
            # stays in one place. We already know interaction_id from the
            # manifest column, but re-deriving and cross-checking guards
            # against accidental manifest edits.
            derived_iid = extract_interaction_id(file_id)
            if derived_iid != row["interaction_id"]:
                print(f"  WARN: file_id {file_id} implies interaction "
                      f"{derived_iid} but manifest row says "
                      f"{row['interaction_id']}; using derived value.",
                      file=sys.stderr)

            output_path = os.path.join(output_dir, f"{file_id}.{FILE_EXT}")

            tasks.append(DownloadTask(
                split=row["split"],
                interaction_id=derived_iid,
                participant_id=extract_participant_id(file_id),
                file_id=file_id,
                url=url,
                output_path=output_path,
            ))

    tasks.sort(key=lambda t: (t.split, t.interaction_id, t.file_id))
    return tasks


# =============================================================================
# Robust download helper (replaces shared `download_file`)
# =============================================================================

def _is_transient_error(err: BaseException) -> bool:
    """True if this exception class is worth retrying.

    - HTTP 5xx  → retry
    - HTTP 4xx (other than what the caller handles first, e.g. 403/404)
                → NOT retry (permanent client-side issue)
    - socket.timeout, TimeoutError, ConnectionError, URLError → retry
      (all classic network hiccups)
    - OSError  → retry (catches many urllib-wrapped socket errors we
                don't want to enumerate by name)
    """
    if isinstance(err, HTTPError):
        return 500 <= err.code < 600
    if isinstance(err, (socket.timeout, TimeoutError,
                        ConnectionError, URLError)):
        return True
    return isinstance(err, OSError)


def _cleanup_partial(tmp_path: str) -> None:
    """Remove a `.part` file from a failed download so the next attempt
    starts clean. Idempotent: silent no-op if the file isn't there."""
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # rare race with another process; not worth aborting


def _download_file_robust(
    url: str,
    output_path: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    socket_timeout_s: float = DEFAULT_SOCKET_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff_s: float = DEFAULT_INITIAL_BACKOFF_S,
    max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[bool, str]:
    """
    Robust replacement for the shared `download_file`. Designed for
    multi-hour runs over flaky networks at full-project scale.

    Mechanics:

    1. **Temp-file + atomic rename.** Writes to `{output_path}.part`; on
       success, `os.rename` → final path. `os.rename` is atomic on
       POSIX, so external observers (including this script's resume
       path) never see a partial file at the final path. If the final
       path exists before rename, we remove it first (Windows doesn't
       allow rename-over-existing).
    2. **Chunked streaming write.** Reads `chunk_size` bytes at a time
       and writes directly to disk. Memory use is O(chunk_size) per
       worker, not O(file_size). Critical for 50-100 MB wavs × many
       workers.
    3. **Stall detection via socket timeout.** `urlopen(..., timeout=)`
       applies to each socket operation, including individual `.read()`
       calls. If `chunk_size` bytes haven't arrived in `socket_timeout_s`
       seconds, `socket.timeout` fires and we retry (or give up if the
       retry budget is exhausted).
    4. **Content-Length integrity check.** If the server advertises a
       Content-Length header, we compare it to the actual byte count
       after streaming completes. Mismatch → treated as transient and
       retried. This catches truncated transfers that otherwise end
       cleanly (connection half-closed partway through).
    5. **Retry with exponential backoff.** Transient failures (5xx,
       timeouts, connection resets) are retried up to `max_retries`
       times; backoff starts at `initial_backoff_s` and doubles each
       time, capped at `max_backoff_s`. HTTP 403 / 404 are never retried
       — they indicate the file is genuinely absent on S3.
    6. **Cleanup on failure.** Any exit path that doesn't produce a
       complete file removes the `.part` sibling so the next run starts
       clean.
    """
    tmp_path = output_path + ".part"

    # Always remove stale .part from a killed prior run BEFORE any retry
    # loop begins. The retry loop will create its own fresh .part.
    _cleanup_partial(tmp_path)

    attempt = 0
    backoff = initial_backoff_s
    last_err_msg = None

    while attempt < max_retries:
        attempt += 1
        n_written = 0
        expected_len = None
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=socket_timeout_s) as response:
                cl_hdr = response.headers.get("Content-Length")
                if cl_hdr and cl_hdr.strip().isdigit():
                    expected_len = int(cl_hdr)

                with open(tmp_path, "wb") as f:
                    while True:
                        # response.read(chunk_size) inherits the socket's
                        # timeout from urlopen(). A stalled connection
                        # where no bytes arrive for `socket_timeout_s`
                        # fires socket.timeout here.
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        n_written += len(chunk)

            # Integrity: content length must match if advertised.
            if expected_len is not None and n_written != expected_len:
                raise IOError(
                    f"size mismatch: downloaded {n_written:,} bytes, "
                    f"Content-Length advertised {expected_len:,}"
                )

            # A legitimate audio wav should never be 0 bytes.
            if n_written == 0:
                raise IOError("received empty body (0 bytes)")

            # Atomic rename. Windows requires removing the destination first.
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(tmp_path, output_path)
            return True, f"OK ({n_written:,} bytes)"

        except HTTPError as e:
            # 403 / 404 = file genuinely not on S3. Never retry.
            if e.code in (403, 404):
                _cleanup_partial(tmp_path)
                return False, f"NOT FOUND (HTTP {e.code})"
            # 5xx is transient; 4xx (other) is permanent.
            if _is_transient_error(e) and attempt < max_retries:
                last_err_msg = f"HTTP {e.code}"
                _cleanup_partial(tmp_path)
                time.sleep(min(backoff, max_backoff_s))
                backoff *= 2
                continue
            _cleanup_partial(tmp_path)
            return False, f"HTTP ERROR {e.code}"

        except Exception as e:
            # socket.timeout, ConnectionError, URLError, our own IOError
            # from size mismatch / empty body — all fall here.
            if _is_transient_error(e) and attempt < max_retries:
                last_err_msg = f"{type(e).__name__}: {e}"
                _cleanup_partial(tmp_path)
                time.sleep(min(backoff, max_backoff_s))
                backoff *= 2
                continue
            _cleanup_partial(tmp_path)
            return False, f"ERROR: {type(e).__name__}: {e}"

    _cleanup_partial(tmp_path)
    return False, (
        f"ERROR: exhausted {max_retries} retries "
        f"(last={last_err_msg or 'unknown'})"
    )


# =============================================================================
# Execution
# =============================================================================

def already_have(output_path: str) -> bool:
    """True iff the FINAL output file is on disk and nonempty.

    As a side effect, removes any stale `.part` sibling — those are
    unambiguously incomplete (the download helper only renames on
    success), so there's no value in keeping them around. If the final
    path is present too (unusual but possible if something cleaned up
    badly), we trust the final path and silently remove .part.

    A 0-byte file at the final path is treated as not-yet-downloaded so
    the next run regenerates it.
    """
    tmp_path = output_path + ".part"
    _cleanup_partial(tmp_path)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def run_task(
    task: DownloadTask,
    overwrite: bool,
    dry_run: bool,
    chunk_size: int,
    socket_timeout_s: float,
    max_retries: int,
    initial_backoff_s: float,
) -> dict:
    """
    Execute a single DownloadTask. HTTP work is delegated to
    `_download_file_robust` which handles retries, streaming, rename,
    and integrity checking.

    Result status codes:
        "skipped"  — on disk already; not overwriting.
        "dry"      — --dry-run; URL printed but not fetched.
        "ok"       — downloaded successfully.
        "missing"  — S3 returned 403/404 (file not on S3).
        "failed"   — retries exhausted or hard error; message carries detail.
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

    success, msg = _download_file_robust(
        task.url, task.output_path,
        chunk_size=chunk_size,
        socket_timeout_s=socket_timeout_s,
        max_retries=max_retries,
        initial_backoff_s=initial_backoff_s,
    )
    if success:
        return {"task": task, "status": "ok", "message": msg}
    if "NOT FOUND" in msg or "HTTP 403" in msg or "HTTP 404" in msg:
        return {"task": task, "status": "missing", "message": msg}
    return {"task": task, "status": "failed", "message": msg}


def run_tasks(
    tasks: list[DownloadTask],
    num_workers: int,
    overwrite: bool,
    dry_run: bool,
    chunk_size: int,
    socket_timeout_s: float,
    max_retries: int,
    initial_backoff_s: float,
) -> list[dict]:
    """Run all tasks concurrently and return the per-task results.

    Progress is shown via tqdm; per-task status lines are written with
    `tqdm.write()` so they don't get mangled by the bar. The returned
    list is sorted deterministically for stable diffing across runs.
    """
    # Lazy-import tqdm so --dry-run + --help work without the dep.
    try:
        from tqdm.auto import tqdm
    except ImportError:
        # Graceful degradation: minimal context-manager-capable stand-in.
        # Supports `with tqdm(...) as pbar: pbar.update(); pbar.set_postfix(...)`
        # and the staticmethod-style `tqdm.write(...)` used below.
        class _FakeTqdm:
            def __init__(self, total=None, **kwargs):
                self.total = total
                self.n = 0
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def update(self, n=1):
                self.n += n
            def set_postfix(self, **kwargs):
                pass
            @staticmethod
            def write(msg, file=None):
                print(msg, file=file or sys.stdout)
        tqdm = _FakeTqdm  # type: ignore[assignment]

    results: list[dict] = []

    # Ensure the output directory exists up-front rather than on every
    # call. _download_file_robust also calls os.makedirs per retry, but
    # this pre-create makes a --dry-run that still mkdirs feel cleaner
    # and surfaces permission errors before any threads spin up.
    if tasks and not dry_run:
        os.makedirs(os.path.dirname(tasks[0].output_path), exist_ok=True)

    # Counters + byte total maintained live so tqdm's postfix is useful.
    counts = {"ok": 0, "skipped": 0, "dry": 0, "missing": 0, "failed": 0}
    bytes_ok = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                run_task, t, overwrite, dry_run,
                chunk_size, socket_timeout_s, max_retries, initial_backoff_s,
            ): t for t in tasks
        }

        with tqdm(total=len(tasks), unit="file", desc="download",
                  dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                task = result["task"]
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1

                if status == "ok":
                    # Parse "OK (NNN,NNN bytes)" for the live byte total.
                    try:
                        bytes_ok += int(
                            result["message"].split("(")[1].split(" ")[0]
                                             .replace(",", "")
                        )
                    except (IndexError, ValueError):
                        pass

                # tqdm.write goes to stderr (by default) without breaking the bar.
                tqdm.write(
                    f"  [{status:<7}] {task.split:<5} "
                    f"{task.file_id}  {result['message']}"
                )
                pbar.update(1)
                pbar.set_postfix(
                    ok=counts["ok"],
                    missing=counts["missing"],
                    failed=counts["failed"],
                    skipped=counts["skipped"],
                    MB=f"{bytes_ok / (1024 * 1024):.0f}",
                )
                results.append(result)

    results.sort(key=lambda r: (r["task"].split, r["task"].interaction_id,
                                r["task"].file_id))
    return results


# =============================================================================
# Summary
# =============================================================================

def print_summary(results: list[dict], elapsed_s: float, dry_run: bool) -> None:
    """Print counts by status + total downloaded bytes.

    At the end, if any task returned "missing" (HTTP 403/404), prints a
    loud warning block — audio is a primary modality and its absence on
    S3 is a data-integrity issue that should not be lost in the scroll.
    """
    counts: dict[str, int] = {}
    bytes_downloaded = 0
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "ok":
            # download_file returns "OK (NNN bytes)"; parse back out.
            try:
                bytes_downloaded += int(
                    r["message"].split("(")[1].split(" ")[0].replace(",", "")
                )
            except (IndexError, ValueError):
                pass

    print("\n" + "=" * 64)
    print("DOWNLOAD SUMMARY" + ("  [DRY-RUN]" if dry_run else ""))
    print("=" * 64)
    print(f"  Total tasks:  {len(results)}")
    for status in ("ok", "skipped", "dry", "missing", "failed"):
        n = counts.get(status, 0)
        if n:
            print(f"  {status:<10} {n}")
    if bytes_downloaded:
        mib = bytes_downloaded / (1024 * 1024)
        print(f"  Downloaded:   {bytes_downloaded:,} bytes ({mib:,.1f} MiB)")
    print(f"  Elapsed:      {elapsed_s:,.1f} s")
    print("=" * 64)

    # Loud warning for missing audio. Unlike transcripts or annotations,
    # audio is a PRIMARY modality: every participant should have a wav
    # on S3, and a 403 here signals one of:
    #   (a) the manifest is pointing at a dyad Meta never published audio for,
    #   (b) an upload / release gap on Meta's side,
    #   (c) the S3 URL scheme changed (check build_s3_url).
    # In any of those cases we want to fail loudly, not quietly — a user
    # scrolling past a summary shouldn't miss this.
    n_missing = counts.get("missing", 0)
    if n_missing and not dry_run:
        banner = "!" * 72
        print()
        print(banner, file=sys.stderr)
        print(f"!!  WARNING: {n_missing} AUDIO FILE(S) NOT ON S3 (HTTP 403/404)",
              file=sys.stderr)
        print("!!",
              file=sys.stderr)
        print("!!  Audio is a primary modality — missing files here are NOT",
              file=sys.stderr)
        print("!!  analogous to missing transcripts (which are optional per",
              file=sys.stderr)
        print("!!  Seamless release notes). This likely indicates one of:",
              file=sys.stderr)
        print("!!    (a) the manifest references a dyad Meta did not publish",
              file=sys.stderr)
        print("!!        audio for,",
              file=sys.stderr)
        print("!!    (b) an upload / release gap on Meta's side, or",
              file=sys.stderr)
        print("!!    (c) the S3 path scheme changed (check build_s3_url).",
              file=sys.stderr)
        print("!!",
              file=sys.stderr)
        print("!!  Review the per-file log above. To proceed, either drop",
              file=sys.stderr)
        print("!!  the affected interactions from the manifest or reach out",
              file=sys.stderr)
        print("!!  to the Seamless team. Exit code is 3 (non-fatal but non-zero).",
              file=sys.stderr)
        print(banner, file=sys.stderr)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download per-participant .wav files for every interaction "
            "in a turn-taking manifest CSV. Reuses build_s3_url from "
            "csci535-project/scripts/download_annotated_interactions.py; "
            "ships a local robust downloader with retries, streaming, "
            "temp-file rename, and Content-Length integrity checking."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST_PATH,
        help="Path to the manifest CSV. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save .wav files into. Default: %(default)s",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help=(
            "Number of concurrent download threads. Default: %(default)s. "
            "At full-project scale (~934 files), 8-16 is reasonable; "
            "avoid > 32 — Meta's CDN may rate-limit and per-worker memory "
            "use rises linearly."
        ),
    )
    parser.add_argument(
        "--filter-split", choices=["train", "val", "test"], default=None,
        help=(
            "If set, only download rows whose manifest `split` column "
            "matches (useful for staged debugging: download val first, "
            "confirm everything works, then train/test)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Re-download files that already exist on disk. Default is "
            "to skip them (idempotent resume)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print URLs that would be downloaded without fetching anything.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help="Per-file retry budget for transient network errors. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--socket-timeout-s", type=float, default=DEFAULT_SOCKET_TIMEOUT_S,
        help="Per-socket-operation timeout in seconds. Catches stalled "
             "streams where bytes stop arriving mid-transfer. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Streamed-read chunk size in bytes. Default: %(default)s "
             "(= 64 KiB). Bigger = less syscall overhead, more memory.",
    )
    parser.add_argument(
        "--initial-backoff-s", type=float, default=DEFAULT_INITIAL_BACKOFF_S,
        help="Initial backoff between retries; doubles per attempt, "
             f"capped at {DEFAULT_MAX_BACKOFF_S}s. Default: %(default)s.",
    )
    args = parser.parse_args()

    print(f"Loading manifest:  {args.manifest}")
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  {len(rows):,} interactions in manifest "
          f"({2 * len(rows):,} participant wavs)")

    tasks = build_tasks(rows, args.output_dir, args.filter_split)
    if args.filter_split:
        print(f"  Filtered to split='{args.filter_split}': "
              f"{len(tasks):,} tasks")
    if not tasks:
        print("Nothing to download. Exiting.")
        return 0

    print(f"Output directory:  {args.output_dir}")
    print(f"Workers:           {args.num_workers}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")
    print(f"Retries per file:  {args.max_retries}  "
          f"(initial backoff {args.initial_backoff_s}s, "
          f"cap {DEFAULT_MAX_BACKOFF_S}s)")
    print(f"Socket timeout:    {args.socket_timeout_s}s  "
          f"(stalled-stream detection)")
    print(f"Chunk size:        {args.chunk_size:,} bytes")
    print(f"Starting {len(tasks):,} download task(s)...\n")

    # monotonic() rather than time() — survives NTP / DST / sleep jumps.
    start = time.monotonic()
    results = run_tasks(
        tasks,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size,
        socket_timeout_s=args.socket_timeout_s,
        max_retries=args.max_retries,
        initial_backoff_s=args.initial_backoff_s,
    )
    elapsed = time.monotonic() - start

    print_summary(results, elapsed, args.dry_run)

    # Exit code reflects hard failures only. "missing" is non-zero but
    # non-fatal so callers can detect it without catching it as an error.
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_missing = sum(1 for r in results if r["status"] == "missing")
    if n_failed:
        return 2
    if n_missing:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
