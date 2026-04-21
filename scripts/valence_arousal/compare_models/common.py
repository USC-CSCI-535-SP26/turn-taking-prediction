"""
Shared constants, paths, and helpers for the emotion-extractor bake-off pipeline.

Consumed by download.py, extract.py, and score.py. Single source of truth for:
  - the canonical 136 file_ids (cross-check of filelist.csv `has_imitator_movement=1`
    against on-disk `emotion_features.npz`)
  - absolute paths (data/, predictions/, results/, logs/, annotated-interaction trees)
  - `has_usable_bundle(file_id)` — the one agreed-upon definition of "ready to extract"
  - S3 URL construction
  - a get_logger() that writes stdout + a per-run log file
  - a to_device() helper that sets PYTORCH_ENABLE_MPS_FALLBACK=1 and prefers MPS

Running `python common.py` prints a summary and fails loudly if the filesystem
disagrees with the filelist (e.g., a branch switch left things out of sync).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — all absolute, all rooted at the repo.
# ---------------------------------------------------------------------------

# This file lives at .../csci535-project/scripts/valence_arousal/compare_models/common.py
HERE: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = HERE.parent.parent.parent  # csci535-project/
SEAMLESS_ROOT: Final[Path] = REPO_ROOT.parent / "seamless_interaction"

FILELIST_CSV: Final[Path] = SEAMLESS_ROOT / "assets" / "filelist.csv"

BOTH_TREE: Final[Path] = REPO_ROOT / "both_annotated_interactions"
SINGLE_TREE: Final[Path] = REPO_ROOT / "single_annotated_interactions"

# Pipeline output directories (all created on first use)
DATA_DIR: Final[Path] = HERE / "data"
PREDICTIONS_DIR: Final[Path] = HERE / "predictions"
RESULTS_DIR: Final[Path] = HERE / "results"
LOGS_DIR: Final[Path] = HERE / "logs"

# S3 URL construction
S3_BUCKET: Final[str] = "dl.fbaipublicfiles.com"
S3_PREFIX: Final[str] = "seamless_interaction"

# The four modalities we pull per file_id. Keys are local shorthand;
# values describe how to build the S3 path and the extension used on disk.
Modality = Literal["mp4", "box", "is_valid_box", "keypoints"]

MODALITIES: Final[dict[Modality, dict[str, str]]] = {
    "mp4":          {"s3_subpath": "video",                           "ext": "mp4"},
    "box":          {"s3_subpath": "boxes_and_keypoints/box",         "ext": "npy"},
    "is_valid_box": {"s3_subpath": "boxes_and_keypoints/is_valid_box", "ext": "npy"},
    "keypoints":    {"s3_subpath": "boxes_and_keypoints/keypoints",   "ext": "npy"},
}
ALL_MODALITIES: Final[tuple[Modality, ...]] = tuple(MODALITIES.keys())


# ---------------------------------------------------------------------------
# file_id parsing & canonical list
# ---------------------------------------------------------------------------

# The project's local annotation tree uses plain "V#_S#_I#_P#" file_ids only.
# The upstream filelist.csv also has trailing-letter variants (e.g. "...P0844A"),
# which are a different take / annotation cohort and are out of scope here —
# none of them have emotion_features.npz under our local 443 interactions.
_PARTICIPANT_DIR_RE = re.compile(r"^participant_([ab])_(P\d+)$")


@dataclass(frozen=True)
class FileIdInfo:
    """All metadata about a file_id needed by the pipeline. Immutable."""
    file_id: str
    interaction_id: str       # V..._S..._I... (no participant)
    participant_id: str       # P...
    label: str                # "improvised" | "naturalistic"
    split: str                # "train" | "dev" | "test"
    imitator_npz: Path        # the emotion_features.npz on disk
    source_tree: str          # "both_annotated_interactions" | "single_annotated_interactions"


@lru_cache(maxsize=1)
def _filelist() -> pd.DataFrame:
    """Load filelist.csv once per process. Indexed by file_id."""
    if not FILELIST_CSV.exists():
        raise FileNotFoundError(
            f"filelist.csv not found at {FILELIST_CSV}. "
            f"Expected the seamless_interaction repo to be cloned next to csci535-project."
        )
    df = pd.read_csv(FILELIST_CSV)
    required = {"file_id", "label", "split", "has_imitator_movement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filelist.csv missing columns: {missing}")
    return df.set_index("file_id")


@lru_cache(maxsize=1)
def discover_file_ids() -> list[FileIdInfo]:
    """
    Return the canonical, sorted list of file_ids for the bake-off.

    Discovery works *from the filesystem outward*: we walk both annotation trees,
    collect every `emotion_features.npz` we find, build the implied file_id
    ("{interaction_id}_{participant_id}", plain format — no trailing letter),
    and validate each against filelist.csv.

    A file_id qualifies iff:
      (1) An `emotion_features.npz` exists at
          {tree}/{interaction_id}/participant_{a|b}_{participant_id}/emotion_features.npz
      (2) filelist.csv has the matching file_id with has_imitator_movement == 1.

    Any on-disk npz whose implied file_id is missing from the filelist or is
    flagged has_imitator_movement=0 is an orphan — we raise loudly rather than
    silently ignore, because a mismatch almost certainly means a stale branch or
    an out-of-sync filelist.
    """
    df = _filelist()

    infos: list[FileIdInfo] = []
    orphans: list[tuple[Path, str]] = []  # (path, reason)

    for tree in (BOTH_TREE, SINGLE_TREE):
        if not tree.exists():
            continue
        for npz_path in sorted(tree.glob("*/participant_*/emotion_features.npz")):
            interaction_id = npz_path.parents[1].name
            dir_match = _PARTICIPANT_DIR_RE.match(npz_path.parent.name)
            if not dir_match:
                orphans.append((npz_path, "participant dir name doesn't match participant_[ab]_P<digits>"))
                continue
            _letter, pid = dir_match.groups()
            file_id = f"{interaction_id}_{pid}"

            if file_id not in df.index:
                orphans.append((npz_path, f"file_id {file_id!r} not in filelist.csv"))
                continue
            row = df.loc[file_id]
            if int(row["has_imitator_movement"]) != 1:
                orphans.append((npz_path, f"file_id {file_id!r} has has_imitator_movement={row['has_imitator_movement']}"))
                continue

            infos.append(FileIdInfo(
                file_id=file_id,
                interaction_id=interaction_id,
                participant_id=pid,
                label=str(row["label"]),
                split=str(row["split"]),
                imitator_npz=npz_path,
                source_tree=tree.name,
            ))

    if orphans:
        lines = [
            f"Found {len(orphans)} emotion_features.npz file(s) on disk that do not match "
            f"filelist.csv. Refusing to proceed — a mismatch usually means a stale git "
            f"branch or an out-of-sync filelist."
        ]
        for path, reason in orphans[:10]:
            lines.append(f"  {path}  ({reason})")
        if len(orphans) > 10:
            lines.append(f"  ... and {len(orphans) - 10} more")
        raise RuntimeError("\n".join(lines))

    # Duplicate-file_id check: should be impossible if the tree is well-formed, but
    # assert to catch the (interaction_id, participant_id) appearing under both trees.
    seen: dict[str, Path] = {}
    for info in infos:
        if info.file_id in seen:
            raise RuntimeError(
                f"Duplicate file_id {info.file_id!r} found in both\n"
                f"  {seen[info.file_id]}\nand\n  {info.imitator_npz}"
            )
        seen[info.file_id] = info.imitator_npz

    infos.sort(key=lambda x: x.file_id)
    return infos


def file_id_info(file_id: str) -> FileIdInfo:
    """Look up one file_id. Raises if it isn't in the canonical list."""
    for info in discover_file_ids():
        if info.file_id == file_id:
            return info
    raise KeyError(f"{file_id} is not in the canonical list (has_imitator_movement=1 + on-disk)")


# ---------------------------------------------------------------------------
# Data paths + bundle checks
# ---------------------------------------------------------------------------

def data_path_for(file_id: str, modality: Modality) -> Path:
    """
    Where this (file_id, modality) lives on local disk after download.

    Layout mirrors the S3 subpath:
        data/{label}/{split}/video/{file_id}.mp4
        data/{label}/{split}/boxes_and_keypoints/box/{file_id}.npy
        ...

    Matching the S3 layout locally means any scripts or notebooks written
    against the original dataset layout work unchanged.
    """
    info = file_id_info(file_id)
    cfg = MODALITIES[modality]
    return DATA_DIR / info.label / info.split / cfg["s3_subpath"] / f"{file_id}.{cfg['ext']}"


def s3_url_for(file_id: str, modality: Modality) -> str:
    """HTTPS URL where this (file_id, modality) lives on S3."""
    info = file_id_info(file_id)
    cfg = MODALITIES[modality]
    return (
        f"https://{S3_BUCKET}/{S3_PREFIX}/"
        f"{info.label}/{info.split}/{cfg['s3_subpath']}/{file_id}.{cfg['ext']}"
    )


def bundle_files_present(file_id: str) -> bool:
    """Cheap check: all 4 bundle files exist on disk. Used by download.py skip logic."""
    return all(data_path_for(file_id, m).exists() for m in ALL_MODALITIES)


def bundle_is_usable(file_id: str) -> bool:
    """
    Strict check: bundle exists AND each file opens cleanly.

    Used by extract.py before deciding to skip an already-extracted file_id.
    Partial/corrupt files from a killed prior run must re-extract, not silently
    pass. Mp4 is checked by header magic bytes; npy by mmap-based load.
    """
    if not bundle_files_present(file_id):
        return False

    mp4 = data_path_for(file_id, "mp4")
    try:
        with open(mp4, "rb") as f:
            head = f.read(12)
        # ISO-BMFF: bytes 4..8 are 'ftyp'
        if len(head) < 12 or head[4:8] != b"ftyp":
            return False
    except OSError:
        return False

    for m in ("box", "is_valid_box", "keypoints"):
        try:
            # mmap_mode='r' reads only the header — fast validity check
            np.load(data_path_for(file_id, m), allow_pickle=False, mmap_mode="r")
        except (ValueError, OSError, EOFError):
            return False

    return True


def bundle_size_bytes(file_id: str) -> int:
    """Sum of on-disk sizes for the 4 bundle files (for logging/progress)."""
    return sum(
        data_path_for(file_id, m).stat().st_size
        for m in ALL_MODALITIES
        if data_path_for(file_id, m).exists()
    )


# ---------------------------------------------------------------------------
# Imitator ground-truth access
# ---------------------------------------------------------------------------

IMITATOR_KEYS: Final[tuple[str, ...]] = (
    "emotion_arousal",       # shape (N, 1)
    "emotion_valence",       # shape (N, 1)
    "emotion_scores",        # shape (N, 8)  -- raw logits, not probabilities
    "EmotionArousalToken",   # shape (N, 1)
    "EmotionValenceToken",   # shape (N, 1)
)


def load_imitator(file_id: str) -> dict[str, np.ndarray]:
    """
    Load the Imitator ground-truth NPZ for this file_id as a plain dict.

    NOTE: `emotion_scores` is raw logits (not probabilities). Score.py handles
    the calibration; callers should not softmax these naively.
    """
    info = file_id_info(file_id)
    with np.load(info.imitator_npz, allow_pickle=False) as z:
        # Reify to a dict so the caller doesn't have to keep the NpzFile alive
        return {k: z[k] for k in IMITATOR_KEYS if k in z.files}


# ---------------------------------------------------------------------------
# Logging — stdout + per-run file, under logs/{script_name}_{timestamp}.log
# ---------------------------------------------------------------------------

_LOG_CONFIGURED: dict[str, bool] = {}


def get_logger(name: str, *, to_file: bool = True) -> logging.Logger:
    """
    Get a configured logger. The first call creates a rotating log file under
    logs/{name}_{YYYYmmdd-HHMMSS}.log; subsequent calls within the same process
    reuse the handlers.
    """
    logger = logging.getLogger(name)
    if _LOG_CONFIGURED.get(name):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fpath = LOGS_DIR / f"{name}_{stamp}.log"
        fh = logging.FileHandler(fpath, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Log file: %s", fpath)

    _LOG_CONFIGURED[name] = True
    return logger


# ---------------------------------------------------------------------------
# Device selection — prefers MPS, falls back to CPU, CUDA only if explicitly set.
# ---------------------------------------------------------------------------

# Must be set BEFORE `import torch` to take effect. Setting here means any
# module that imports common will have the var ready before torch is imported
# later in the same process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> str:
    """Return a torch device string: 'mps', 'cuda', or 'cpu'. Import torch lazily."""
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_device(obj, device: str | None = None):
    """
    Move a tensor or nn.Module to the chosen device. If `device` is None,
    pick_device() is used. Extractors should call this instead of .cuda().
    """
    dev = device or pick_device()
    return obj.to(dev)


# ---------------------------------------------------------------------------
# Self-check: `python common.py` prints a summary, or crashes loudly if
# the filesystem and filelist disagree.
# ---------------------------------------------------------------------------

def _self_check() -> None:
    print(f"REPO_ROOT       = {REPO_ROOT}")
    print(f"SEAMLESS_ROOT   = {SEAMLESS_ROOT}")
    print(f"FILELIST_CSV    = {FILELIST_CSV}  (exists={FILELIST_CSV.exists()})")
    print(f"BOTH_TREE       = {BOTH_TREE}  (exists={BOTH_TREE.exists()})")
    print(f"SINGLE_TREE     = {SINGLE_TREE}  (exists={SINGLE_TREE.exists()})")
    print(f"DATA_DIR        = {DATA_DIR}")
    print(f"PREDICTIONS_DIR = {PREDICTIONS_DIR}")
    print(f"RESULTS_DIR     = {RESULTS_DIR}")
    print(f"LOGS_DIR        = {LOGS_DIR}")
    print()

    infos = discover_file_ids()
    print(f"Canonical file_ids: {len(infos)}")
    by_tree = pd.Series([i.source_tree for i in infos]).value_counts()
    print(f"  by source tree:")
    for tree, n in by_tree.items():
        print(f"    {tree:40s} {n}")
    by_split = pd.Series([(i.label, i.split) for i in infos]).value_counts()
    print(f"  by (label, split):")
    for (label, split), n in by_split.items():
        print(f"    {label:14s} {split:6s} {n}")
    print()

    # Bundle status on disk (without downloading anything)
    present = sum(bundle_files_present(i.file_id) for i in infos)
    usable = sum(bundle_is_usable(i.file_id) for i in infos)
    print(f"Bundle status:")
    print(f"  all 4 files present:   {present}/{len(infos)}")
    print(f"  all 4 files usable:    {usable}/{len(infos)}  (opens + validates headers)")
    print()

    # Sample one
    sample = infos[0]
    print(f"Sample file_id: {sample.file_id}")
    print(f"  interaction_id = {sample.interaction_id}")
    print(f"  participant_id = {sample.participant_id}")
    print(f"  label/split    = {sample.label}/{sample.split}")
    print(f"  imitator_npz   = {sample.imitator_npz}")
    for m in ALL_MODALITIES:
        p = data_path_for(sample.file_id, m)
        print(f"  local {m:13s} = {p}  (exists={p.exists()})")
        print(f"  s3    {m:13s} = {s3_url_for(sample.file_id, m)}")
    print()

    # Device
    try:
        import torch
        print(f"PyTorch device: {pick_device()}  (torch={torch.__version__})")
    except ImportError:
        print("PyTorch not installed in this environment — skipping device check.")


if __name__ == "__main__":
    _self_check()
