#!/usr/bin/env python3
"""
Download the mp4 + boxes_and_keypoints (box, is_valid_box, keypoints) for each
of the 136 canonical file_ids from S3 to local disk.

What this does NOT download:
  * emotion_features.npz — Imitator ground truth. Already on disk under
    {both,single}_annotated_interactions/.../. Verified by common.discover_file_ids().
  * Anything else from the dataset (smplh, movement features, transcripts,
    annotation JSONs, etc.) — out of scope for the bake-off.

Design notes (things we deliberately avoid from the prior notebook):
  * Does NOT call SeamlessInteractionFS.gather_file_id_data_from_s3(). That
    aggregator tries to parse the dataset's annotation JSON files, which are
    actually JSONL-formatted, so json.load() raises and the aggregator never
    writes its output npz. We bypass it entirely and fetch only the 4 URLs
    per file_id that we actually need.
  * Success criterion is on-disk files, not exception absence. A file_id is
    "done" iff all 4 bundle files exist (and can be opened, per
    common.bundle_is_usable).
  * Atomic writes: every download goes to `{path}.part` and is renamed on
    success. A killed process never leaves a "looks complete but is truncated"
    file behind.
  * Errors are loud. No logger silencing.

Usage:
    python download.py --limit 2                  # smoke-test on 2 file_ids
    python download.py                            # run all 136
    python download.py --workers 8                # more parallelism
    python download.py --dry-run                  # HEAD-check URLs only
    python download.py --file-id V00_S1132_I00000333_P0737
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import common
from common import (
    ALL_MODALITIES,
    DATA_DIR,
    FileIdInfo,
    Modality,
    bundle_files_present,
    bundle_is_usable,
    bundle_size_bytes,
    data_path_for,
    discover_file_ids,
    get_logger,
    s3_url_for,
)

log = get_logger("download")

# Network tuning. Conservative defaults; override via CLI.
DEFAULT_WORKERS: int = 4
MAX_RETRIES: int = 3                # attempts = 1 + retries
INITIAL_BACKOFF_SEC: float = 2.0
CHUNK_BYTES: int = 1 << 20          # 1 MiB streaming reads
CONNECT_TIMEOUT_SEC: int = 30


# ---------------------------------------------------------------------------
# Per-file_id result record
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    file_id: str
    # Real run:   "ok" | "skipped" | "failed"
    # Dry run:    "dry_run_ok" | "skipped" | "dry_run_failed"
    # "skipped" in both modes means all 4 modalities were already on disk.
    status: str
    modalities_downloaded: list[Modality] = field(default_factory=list)  # real run only
    modalities_head_ok: list[Modality] = field(default_factory=list)     # dry run only
    modalities_skipped: list[Modality] = field(default_factory=list)     # both
    bytes_downloaded: int = 0         # real run only — bytes actually written
    bytes_would_download: int = 0     # dry run only — sum of Content-Length
    elapsed_sec: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Network I/O — single URL download with streaming + retry
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest: Path, *, log_ctx: str) -> int:
    """
    Download `url` to `dest`, atomically. Streams to a `.part` sibling, verifies
    byte count against Content-Length when the server provides it, then renames.
    Returns bytes written. Raises on any failure. Caller is responsible for
    retry / deletion of partials on final failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # Clean up any stale partial from an earlier killed run.
    if part.exists():
        log.warning("%s: removing stale partial %s", log_ctx, part.name)
        part.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "compare_models/download.py"})
    written = 0
    with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_SEC) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        expected = resp.headers.get("Content-Length")
        expected_int = int(expected) if expected and expected.isdigit() else None

        with open(part, "wb") as f:
            while True:
                chunk = resp.read(CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)

    if expected_int is not None and written != expected_int:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Content-Length mismatch for {url}: "
            f"expected {expected_int}, got {written}"
        )

    # Atomic rename. On POSIX this is atomic within a filesystem.
    os.replace(part, dest)
    return written


def _download_with_retry(url: str, dest: Path, *, log_ctx: str) -> int:
    """
    Wrapper around _stream_download with exponential backoff retry.
    Raises after MAX_RETRIES + 1 failed attempts.
    """
    last_err: Optional[Exception] = None
    backoff = INITIAL_BACKOFF_SEC
    for attempt in range(1, MAX_RETRIES + 2):  # attempts 1..MAX_RETRIES+1
        try:
            return _stream_download(url, dest, log_ctx=log_ctx)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, OSError) as e:
            last_err = e
            if attempt <= MAX_RETRIES:
                log.warning(
                    "%s: attempt %d/%d failed (%s: %s); retrying in %.1fs",
                    log_ctx, attempt, MAX_RETRIES + 1, type(e).__name__, e, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
    assert last_err is not None
    # Re-raise the last error for the caller to package into a FileResult.
    raise last_err


# ---------------------------------------------------------------------------
# Per-file_id orchestration
# ---------------------------------------------------------------------------

def _head_check(url: str, *, log_ctx: str) -> int:
    """Issue a HEAD request; return reported content-length (or 0 if absent)."""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "compare_models/download.py"})
    with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_SEC) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HEAD HTTP {resp.status} for {url}")
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else 0


def _download_file_id(info: FileIdInfo, *, dry_run: bool) -> FileResult:
    """Download all missing modalities for one file_id. Returns a FileResult."""
    result = FileResult(file_id=info.file_id, status="ok")
    t0 = time.monotonic()

    for modality in ALL_MODALITIES:
        dest = data_path_for(info.file_id, modality)
        url = s3_url_for(info.file_id, modality)
        log_ctx = f"{info.file_id}:{modality}"

        if dest.exists():
            result.modalities_skipped.append(modality)
            continue

        try:
            if dry_run:
                size = _head_check(url, log_ctx=log_ctx)
                log.info("%s: dry-run HEAD ok, Content-Length=%d", log_ctx, size)
                # Track the would-be download in a separate field so the report
                # never claims bytes were written that weren't.
                result.modalities_head_ok.append(modality)
                result.bytes_would_download += size
            else:
                bytes_written = _download_with_retry(url, dest, log_ctx=log_ctx)
                result.modalities_downloaded.append(modality)
                result.bytes_downloaded += bytes_written
                log.info("%s: downloaded %s (%d bytes)", log_ctx, dest.name, bytes_written)
        except Exception as e:
            result.status = "dry_run_failed" if dry_run else "failed"
            result.error = f"{modality}: {type(e).__name__}: {e}"
            log.error("%s: FAILED after retries — %s", log_ctx, e)
            break  # don't keep trying other modalities for this file_id

    # Finalize status based on what actually happened.
    if result.status not in ("failed", "dry_run_failed"):
        if dry_run:
            # In dry-run, everything either skipped (already on disk) or
            # HEAD-checked successfully. Distinct status vocabulary so a
            # dry-run report can never be confused with a real manifest.
            if result.modalities_head_ok:
                result.status = "dry_run_ok"
            else:
                # All four modalities already on disk — a real run would have skipped
                result.status = "skipped"
        else:
            # Real run: if every modality was already on disk, mark as skipped
            # (not ok) so the summary distinguishes "new work done" from
            # "nothing to do".
            if not result.modalities_downloaded and len(result.modalities_skipped) == len(ALL_MODALITIES):
                result.status = "skipped"

    result.elapsed_sec = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _select_file_ids(
    all_infos: list[FileIdInfo],
    *,
    limit: Optional[int],
    only_file_id: Optional[str],
) -> list[FileIdInfo]:
    if only_file_id is not None:
        matches = [i for i in all_infos if i.file_id == only_file_id]
        if not matches:
            raise SystemExit(
                f"--file-id {only_file_id!r} is not in the canonical list. "
                f"Run `python common.py` to see the 136 valid file_ids."
            )
        return matches
    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be positive")
        return all_infos[:limit]
    return all_infos


def _write_report(results: list[FileResult], path: Path, *, dry_run: bool) -> None:
    """
    Write the per-run report to disk. Dry-run and real runs use DIFFERENT
    filenames and have a top-level `dry_run` flag so neither can be mistaken
    for the other.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        summary = {
            "dry_run": True,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(results),
            "dry_run_ok": sum(1 for r in results if r.status == "dry_run_ok"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "dry_run_failed": sum(1 for r in results if r.status == "dry_run_failed"),
            "bytes_would_download": sum(r.bytes_would_download for r in results),
            "elapsed_sec_total": sum(r.elapsed_sec for r in results),
            "per_file": [asdict(r) for r in results],
        }
    else:
        summary = {
            "dry_run": False,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(results),
            "ok": sum(1 for r in results if r.status == "ok"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "bytes_downloaded": sum(r.bytes_downloaded for r in results),
            "elapsed_sec_total": sum(r.elapsed_sec for r in results),
            "per_file": [asdict(r) for r in results],
        }
    path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s: %s", "dry-run report" if dry_run else "manifest", path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N file_ids (sorted); convention: always run --limit 2 first")
    p.add_argument("--file-id", type=str, default=None,
                   help="process only this single file_id (ignored if --limit is set)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"thread pool size at the file_id level (default {DEFAULT_WORKERS})")
    p.add_argument("--dry-run", action="store_true",
                   help="HEAD-check each URL but don't download")
    args = p.parse_args(argv)

    if args.workers <= 0:
        p.error("--workers must be positive")

    log.info("Discovering canonical file_ids …")
    all_infos = discover_file_ids()
    log.info("Canonical file_ids: %d", len(all_infos))

    todo = _select_file_ids(all_infos, limit=args.limit, only_file_id=args.file_id)
    log.info("Processing %d file_id(s) with %d worker(s)%s",
             len(todo), args.workers, "  [DRY RUN]" if args.dry_run else "")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results: list[FileResult] = []
    t_start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="dl") as pool:
            futures = {pool.submit(_download_file_id, info, dry_run=args.dry_run): info
                       for info in todo}
            for i, fut in enumerate(as_completed(futures), start=1):
                info = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    # Unexpected: an exception escaped _download_file_id itself.
                    r = FileResult(file_id=info.file_id, status="failed",
                                   error=f"unexpected: {type(e).__name__}: {e}")
                    log.exception("%s: unexpected exception", info.file_id)
                results.append(r)
                log.info(
                    "[%d/%d] %s  %s  (%s in %.1fs)",
                    i, len(todo), r.file_id, r.status,
                    _human_bytes(r.bytes_downloaded), r.elapsed_sec,
                )
    except KeyboardInterrupt:
        log.warning("Interrupted — writing partial manifest")

    elapsed_total = time.monotonic() - t_start

    # Report — separate filename for dry-run so a real manifest is never
    # overwritten by a dry-run, and so scripts consuming the manifest can
    # never accidentally load a dry-run report.
    report_path = DATA_DIR / ("dry_run_report.json" if args.dry_run else "download_manifest.json")
    _write_report(results, report_path, dry_run=args.dry_run)

    # Summary
    if args.dry_run:
        n_head_ok = sum(1 for r in results if r.status == "dry_run_ok")
        n_skip    = sum(1 for r in results if r.status == "skipped")
        n_fail    = sum(1 for r in results if r.status == "dry_run_failed")
        bytes_total = sum(r.bytes_would_download for r in results)
        log.info("=" * 60)
        log.info("DRY-RUN summary (no files written):")
        log.info("  processed:                 %d", len(results))
        log.info("  HEAD ok (would download):  %d", n_head_ok)
        log.info("  skipped (already on disk): %d", n_skip)
        log.info("  HEAD failed:               %d", n_fail)
        log.info("  bytes a real run would fetch: %s", _human_bytes(bytes_total))
        log.info("  wall time:                 %.1fs", elapsed_total)
    else:
        n_ok   = sum(1 for r in results if r.status == "ok")
        n_skip = sum(1 for r in results if r.status == "skipped")
        n_fail = sum(1 for r in results if r.status == "failed")
        bytes_total = sum(r.bytes_downloaded for r in results)
        log.info("=" * 60)
        log.info("Summary:")
        log.info("  processed:          %d", len(results))
        log.info("  downloaded:         %d", n_ok)
        log.info("  skipped (on disk):  %d", n_skip)
        log.info("  failed:             %d", n_fail)
        log.info("  bytes downloaded:   %s", _human_bytes(bytes_total))
        log.info("  wall time:          %.1fs", elapsed_total)

    if n_fail > 0:
        log.info("Failed file_ids:")
        for r in results:
            if r.status in ("failed", "dry_run_failed"):
                log.info("  %s  %s", r.file_id, r.error)

    # Post-hoc usability check (only meaningful in real runs — dry-run has no files to open)
    if not args.dry_run and results:
        log.info("Verifying bundles with bundle_is_usable() …")
        n_usable = sum(1 for r in results if bundle_is_usable(r.file_id))
        log.info("  usable bundles: %d/%d", n_usable, len(results))

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
