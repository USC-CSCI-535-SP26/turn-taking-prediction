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

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                 # .../csci535-project
SHARED_SCRIPTS_DIR = THIS_FILE.parent  # download_annotated_interactions.py is a sibling
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
DEFAULT_CHUNK_SIZE = 64 * 1024 # 64 KiB per streamed chunk
DEFAULT_SOCKET_TIMEOUT_S = 120.0 # no-data-for-N-seconds → socket timeout
DEFAULT_MAX_RETRIES = 3  # per-file attempt budget
DEFAULT_INITIAL_BACKOFF_S = 2.0   # doubles per retry; caps at MAX_BACKOFF_S
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
