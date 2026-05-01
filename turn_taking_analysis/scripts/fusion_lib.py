#!/usr/bin/env python3
"""
fusion_lib.py

Library for the turn-taking dyadic-fusion notebook. Provides:

  - Modality-registry-driven sample enumeration + DataLoader construction
    over the spliced per-window per-participant layout
    (turn_taking_analysis/subset/<modality>/<iid_pid>/<start>-<end>_<fid>.npy).
  - Three fusion-family models (unimodal GRU, early-fusion GRU, neural-concat
    fusion) ported from model_assets/model_classes.py and generalized to
    arbitrary stream counts / feature dims.
  - Training / evaluation helpers with class-weighted CE loss and
    val-macro-F1-based model selection.
  - A single `run_experiment(config, ...)` entrypoint the notebook calls.
  - A τ-sweep helper that re-evaluates trained weights against each per-τ
    label file in turn_taking_analysis/model_input/labels/.

Design choices (per discussion):

  - Uses `manifest.csv` (full) OR `poc_manifest.csv` (first pass) as the
    source of splits + interaction metadata. The `split` column on the
    manifest row is authoritative — no re-splitting happens here.
  - Speaker/listener resolution: for every sample, the speaker's file_id
    is embedded in the label basename. The co-participant (listener)
    file_id is resolved via the manifest's participant_a / participant_b
    columns.
  - No shape asserts at load time — the splicing pipeline (splice_wavs.py,
    extract_cpc_from_manifest.py, etc.) guarantees uniform sizes by
    construction. Keeping shape-check out of the hot path.
  - No late fusion in this pass (deferred).
  - No pad/clip logic. Spliced windows are already uniform length per
    modality; `np.load` returns the tensor directly.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable
import pandas as pd


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import math

# =============================================================================
# Constants
# =============================================================================

NUM_CLASSES = 3                                # HOLD, YIELD, BACKCHANNEL
CLASS_NAMES = ("HOLD", "YIELD", "BACKCHANNEL")
TRAINING_CLASSES = frozenset({0, 1, 2})
# Classes 3–5 (INTERRUPT, FAILED, LAPSE) are stored in the label files for
# exclusion-stat reporting but filtered out of training/eval by default

WINDOW_S = 2.0                                 # input-context duration
STRIDE_S = 0.5                                 # spacing between adjacent samples
# Both match splice_wavs.py and build_labeled_windows_from_manifest.py.
# A spliced file 'NNNN.NN-NNNN.NN_{fid}.npy' has end - start = WINDOW_S.

DEFAULT_TAU_GRID_MS: tuple[int, ...] = (100, 200, 400, 500, 800, 1600)
DEFAULT_TRAIN_TAU_MS = 400

# =============================================================================
# COORDINATION CONSTANTS
# =============================================================================

COORDINATION_FEATURE_COLUMNS = [
    "mean_max_abs_corr",
    "std_max_abs_corr",
    "median_max_abs_corr",
    "max_abs_corr",
    "mean_signed_peak_corr",
    "std_signed_peak_corr",
    "mean_zero_lag_corr",
    "mean_abs_zero_lag_corr",
    "std_zero_lag_corr",
    "mean_best_lag_frames",
    "std_best_lag_frames",
    "mean_abs_best_lag_frames",
    "mean_best_lag_sec",
    "mean_abs_best_lag_sec",
    "speaker_lead_ratio",
    "listener_lead_ratio",
    "zero_lag_ratio",
    "speaker_listener_lead_score",
    "high_sync_ratio",
]

COORDINATION_FEATURE_DIM = len(COORDINATION_FEATURE_COLUMNS)

# =============================================================================
# Filename <-> basename helpers
# =============================================================================

def splice_filename_for(start_s: float, end_s: float, file_id: str) -> str:
    """Construct the spliced-modality filename for a given window.

    Matches the convention emitted by splice_wavs.py and
    extract_cpc_from_manifest.py: 'NNNN.NN-NNNN.NN_{file_id}.npy'.
    """
    return f"{start_s:07.2f}-{end_s:07.2f}_{file_id}.npy"

def basename_to_window(basename: str) -> tuple[str, float, float]:
    """Parse a label basename '{file_id}_t_{t_ms:07d}' into (file_id, start_s, end_s).

    basename's `t` component is the window's end time in seconds; start is
    end - WINDOW_S (2.0 s by convention).
    """
    file_id, tms = basename.rsplit("_t_", 1)
    end_s = int(tms) / 1000.0
    start_s = end_s - WINDOW_S
    return file_id, start_s, end_s

# =============================================================================
# LOADERS
# =============================================================================

def _load_feature_file(path_no_ext: str) -> np.ndarray:
    """
    Load feature arrays saved as .npy or .json.
    path_no_ext should be the expected path without extension.
    """
    npy_path = path_no_ext + ".npy"
    json_path = path_no_ext + ".json"

    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32, copy=False)

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)

        # adjust this if your JSON has a wrapper key like {"features": ...}
        if isinstance(data, dict):
            if "features" in data:
                data = data["features"]
            elif "embeddings" in data:
                data = data["embeddings"]
            else:
                raise ValueError(f"JSON feature file has unknown keys: {list(data.keys())}")

        return np.asarray(data, dtype=np.float32)

    raise FileNotFoundError(f"Missing feature file: {npy_path} or {json_path}")

# =============================================================================
# Modality registry
# =============================================================================

def make_modality_registry(entries: dict[str, dict]) -> dict[str, dict]:
    """Validate and normalize a modality registry.

    Each entry: {'dir': str, 'feature_dim': int, 'frame_rate_hz': float}.
    `frame_rate_hz` is metadata only (not consumed by the loader); kept for
    documentation alongside each modality's on-disk contract.
    """
    normalized: dict[str, dict] = {}
    for name, cfg in entries.items():
        required = {"dir", "feature_dim"}
        missing = required - set(cfg)
        if missing:
            raise ValueError(
                f"modality '{name}' missing keys: {sorted(missing)}"
            )
        if not os.path.isdir(cfg["dir"]):
            raise FileNotFoundError(
                f"modality '{name}' dir does not exist: {cfg['dir']}"
            )
        normalized[name] = {
            "dir": str(cfg["dir"]),
            "feature_dim": int(cfg["feature_dim"]),
            "frame_rate_hz": float(cfg.get("frame_rate_hz", 0.0)),
        }
    return normalized

def make_modality_registry_coordination(entries: dict[str, dict]) -> dict[str, dict]:
    """
    Validate and normalize a modality registry.

    Standard modalities:
        {
            "kind": "spliced",
            "dir": str,
            "feature_dim": int,
            "frame_rate_hz": float,
        }

    Coordination summary modality:
        {
            "kind": "coordination",
            "coordination_mode": "summary",
            "feature_dim": 19,
        }

    Coordination continuous modality:
        {
            "kind": "coordination",
            "coordination_mode": "continuous",
            "dir": str,
            "feature_dim": F_wcc,
            "frame_rate_hz": optional float,
        }
    """
    normalized: dict[str, dict] = {}

    for name, cfg in entries.items():
        if "feature_dim" not in cfg:
            raise ValueError(f"modality {name!r} missing key: 'feature_dim'")

        kind = cfg.get("kind", "spliced")

        if kind == "coordination":
            mode = cfg.get("coordination_mode", "summary")
            if mode not in {"summary", "continuous"}:
                raise ValueError(
                    f"coordination modality {name!r} has invalid "
                    f"coordination_mode={mode!r}; use 'summary' or 'continuous'"
                )

            if mode == "continuous":
                if "dir" not in cfg:
                    raise ValueError(
                        f"continuous coordination modality {name!r} needs 'dir'"
                    )
                if not os.path.isdir(cfg["dir"]):
                    raise FileNotFoundError(
                        f"continuous coordination dir does not exist: {cfg['dir']}"
                    )

            normalized[name] = {
                "kind": "coordination",
                "coordination_mode": mode,
                "dir": str(cfg.get("dir", "")),
                "feature_dim": int(cfg["feature_dim"]),
                "frame_rate_hz": float(cfg.get("frame_rate_hz", 0.0)),
            }
            continue

        if kind != "spliced":
            raise ValueError(
                f"modality {name!r} has invalid kind={kind!r}; "
                f"use 'spliced' or 'coordination'"
            )

        if "dir" not in cfg:
            raise ValueError(f"modality {name!r} missing key: 'dir'")

        if not os.path.isdir(cfg["dir"]):
            raise FileNotFoundError(
                f"modality {name!r} dir does not exist: {cfg['dir']}"
            )

        normalized[name] = {
            "kind": "spliced",
            "coordination_mode": None,
            "dir": str(cfg["dir"]),
            "feature_dim": int(cfg["feature_dim"]),
            "frame_rate_hz": float(cfg.get("frame_rate_hz", 0.0)),
        }

    return normalized

# =============================================================================
# Manifest + labels loading
# =============================================================================

REQUIRED_MANIFEST_COLS = frozenset({
    "split", "interaction_id",
    "file_id_a", "file_id_b",
    "participant_a", "participant_b",
})

def load_manifest(path: str) -> list[dict]:
    """Load a manifest CSV. Validates required columns; fails fast on drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_MANIFEST_COLS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"manifest {path} missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest {path} has no rows")
    return rows

def build_fid_lookup(manifest_rows: list[dict]) -> dict[str, tuple[str, str, str]]:
    """Map file_id -> (interaction_id, co_participant_file_id, split).

    Used to look up the listener's file_id given the speaker's file_id
    (which is embedded in the label basename). Each interaction contributes
    two entries (one per participant).
    """
    lookup: dict[str, tuple[str, str, str]] = {}
    for row in manifest_rows:
        iid = row["interaction_id"]
        fid_a, fid_b = row["file_id_a"], row["file_id_b"]
        split = row["split"]
        lookup[fid_a] = (iid, fid_b, split)
        lookup[fid_b] = (iid, fid_a, split)
    return lookup

VALID_LABEL_INTS = frozenset({0, 1, 2, 3, 4, 5})
# The full 6-class range emitted by build_labeled_windows_from_manifest.py
# (training classes 0-2 + exclusion classes 3-5). Anything outside this set
# is a producer bug or file corruption and we fail loud.

def load_labels(labels_path: str) -> dict[str, int]:
    """Load a labels_tau_XXXX.json file into a {basename: class_int} dict.

    Performs strict up-front validation so producer drift (another session
    is editing build_labeled_windows_from_manifest.py) fails loud at load
    rather than silently filtering everything downstream:

      - File must exist and parse as JSON.
      - Top-level must be a dict / JSON object.
      - Every key must be a string.
      - Every value must cast to int without error.
      - Every value must fall in VALID_LABEL_INTS (0..5).

    Normalizes values to Python int regardless of whether the JSON stored
    them as `0` or `"0"`.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels JSON not found: {labels_path}")
    with open(labels_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"labels JSON at {labels_path} must be a top-level object, "
            f"got {type(data).__name__}"
        )
    validated: dict[str, int] = {}
    for basename, raw in data.items():
        if not isinstance(basename, str):
            raise ValueError(
                f"labels JSON key must be a string basename; got "
                f"{type(basename).__name__}: {basename!r}"
            )
        try:
            cls = int(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"labels JSON value for {basename!r} is not int-"
                f"convertible: {raw!r}"
            ) from e
        if cls not in VALID_LABEL_INTS:
            raise ValueError(
                f"labels JSON value for {basename!r} = {cls} is outside "
                f"the 6-class range {sorted(VALID_LABEL_INTS)}"
            )
        validated[basename] = cls
    return validated

# =============================================================================
# Sample enumeration
# =============================================================================

def enumerate_samples(
    manifest_rows: list[dict],
    labels_dict: dict[str, int],
    training_classes: Iterable[int] = TRAINING_CLASSES,
) -> list[dict]:
    """Flatten labels into a list of training-usable sample dicts.

    Each sample:
        basename, speaker_file_id, listener_file_id, interaction_id,
        split, start_s, end_s, label.

    Samples whose speaker_file_id isn't in the manifest are dropped (e.g.,
    when labels_dict spans a larger manifest than we're training on).
    Samples with non-training classes (3/4/5) are dropped.
    """
    fid_lookup = build_fid_lookup(manifest_rows)
    training_classes = set(training_classes)
    out: list[dict] = []
    for basename, raw_cls in labels_dict.items():
        # Cast to int up front. JSON spec permits both `0` and `"0"` and
        # the producer of `labels_tau_*.json` is being edited in another
        # session — a type drift would otherwise cause every sample to
        # silently fail the `not in {0,1,2}` membership check (string
        # never equals int) and produce an empty dataset with no error.
        try:
            cls = int(raw_cls)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"label for basename {basename!r} is not an integer-"
                f"convertible value: {raw_cls!r}"
            ) from e
        if cls not in training_classes:
            continue
        speaker_fid, start_s, end_s = basename_to_window(basename)
        role_info = fid_lookup.get(speaker_fid)
        if role_info is None:
            continue
        iid, listener_fid, split = role_info
        out.append({
            "basename": basename,
            "speaker_file_id": speaker_fid,
            "listener_file_id": listener_fid,
            "interaction_id": iid,
            "split": split,
            "start_s": start_s,
            "end_s": end_s,
            "label": cls,
        })
    return out

def split_samples(samples: list[dict]) -> dict[str, list[dict]]:
    """Partition samples by their `split` field."""
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in samples:
        out.setdefault(s["split"], []).append(s)
    return out

def summarize_samples(samples: list[dict]) -> dict:
    """Quick stats: counts per split, per class, per (split, class)."""
    per_split = Counter(s["split"] for s in samples)
    per_class = Counter(s["label"] for s in samples)
    per_split_class: dict[str, Counter] = {}
    for s in samples:
        per_split_class.setdefault(s["split"], Counter())[s["label"]] += 1
    return {
        "total": len(samples),
        "per_split": dict(per_split),
        "per_class": {CLASS_NAMES[c]: n for c, n in per_class.items()},
        "per_split_class": {
            split: {CLASS_NAMES[c]: n for c, n in cnt.items()}
            for split, cnt in per_split_class.items()
        },
    }

# =============================================================================
# Dataset
# =============================================================================

class TurnTakingDataset(Dataset):
    """Per-sample loader.

    For each sample, returns a tuple `(streams, label)` where:
      - streams is a list of torch.FloatTensor, one per configured stream,
        each of shape (T, feature_dim). T varies by modality (CPC=200,
        OpenFace=60, etc.) but is consistent within a modality.
      - label is an int in {0, 1, 2}.

    A 'stream' is a (modality, role) pair. role ∈ {'speaker', 'listener'}.
    The DataLoader's `streams` list order is preserved in the tuple.
    """

    _VALID_ROLES = frozenset({"speaker", "listener"})

    def __init__(
        self,
        samples: list[dict],
        streams: list[dict],
        modality_registry: dict[str, dict],
    ):
        # Validate stream specs up front. Without this, a typo like
        # role="Speaker" or role="listner" would silently fall through
        # to the `else` branch in __getitem__ (loading the listener's
        # data instead of failing) — wrong-stream training with no
        # error. Likewise, an unregistered modality name would fail
        # later in the hot path with a less obvious KeyError.
        for s in streams:
            role = s.get("role")
            if role not in self._VALID_ROLES:
                raise ValueError(
                    f"stream config has invalid role {role!r}; "
                    f"must be one of {sorted(self._VALID_ROLES)}"
                )
            mod = s.get("modality")
            if mod not in modality_registry:
                raise ValueError(
                    f"stream config references unknown modality {mod!r}; "
                    f"registered modalities: {sorted(modality_registry)}"
                )

        self.samples = samples
        self.streams = streams
        self.registry = modality_registry
        # Pre-resolve per-stream (dir, role) so __getitem__ stays tight.
        self._resolved = [
            (self.registry[s["modality"]]["dir"], s["role"])
            for s in self.streams
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        speaker_fid = s["speaker_file_id"]
        listener_fid = s["listener_file_id"]
        start_s, end_s = s["start_s"], s["end_s"]

        tensors: list[torch.Tensor] = []

        for (mod_dir, role), stream_cfg in zip(self._resolved, self.streams):
            fid = speaker_fid if role == "speaker" else listener_fid

            fname = splice_filename_for(start_s, end_s, fid)
            path = os.path.join(mod_dir, fid, fname)

            arr = np.load(path).astype(np.float32, copy=False)

            # --- enforce fixed length ---
            cfg = self.registry[stream_cfg["modality"]]
            target_t = int(round(cfg.get("frame_rate_hz", 0.0) * WINDOW_S))
            feature_dim = int(cfg["feature_dim"])

            if target_t > 0:
                arr = _fix_seq_len(arr, target_t=target_t, feature_dim=feature_dim)

            tensors.append(torch.from_numpy(arr))

        return tensors, s["label"]
    
class TurnTakingDatasetCoordination(Dataset):
    """
    Per-sample loader supporting:
      - standard spliced modalities: CPC, OpenFace, etc.
      - coordination summary pseudo-modality from CSV: (1, F)
      - continuous coordination pseudo-modality from .npy: (T_wcc, F_wcc)

    For coordination streams, use role='speaker' because the coordination
    features are already speaker-oriented.
    """

    _VALID_ROLES = frozenset({"speaker", "listener"})

    def __init__(
        self,
        samples: list[dict],
        streams: list[dict],
        modality_registry: dict[str, dict],
        coordination_lookup: dict | None = None,
        coordination_feature_dim: int | None = None,
        missing_coordination: str = "zeros",
    ):
        if missing_coordination not in {"zeros", "error"}:
            raise ValueError("missing_coordination must be 'zeros' or 'error'")

        for s in streams:
            role = s.get("role")
            if role not in self._VALID_ROLES:
                raise ValueError(
                    f"stream config has invalid role {role!r}; "
                    f"must be one of {sorted(self._VALID_ROLES)}"
                )

            mod = s.get("modality")
            if mod not in modality_registry:
                raise ValueError(
                    f"stream config references unknown modality {mod!r}; "
                    f"registered modalities: {sorted(modality_registry)}"
                )

            cfg = modality_registry[mod]
            if cfg.get("kind", "spliced") == "coordination" and role != "speaker":
                raise ValueError(
                    "coordination features are speaker-oriented. "
                    "Use role='speaker' for coordination streams."
                )

        self.samples = samples
        self.streams = streams
        self.registry = modality_registry
        self.coordination_lookup = coordination_lookup or {}
        self.coordination_feature_dim = coordination_feature_dim
        self.missing_coordination = missing_coordination

        self._resolved = []
        for s in self.streams:
            mod = s["modality"]
            role = s["role"]
            cfg = self.registry[mod]
            kind = cfg.get("kind", "spliced")
            mode = cfg.get("coordination_mode", None)
            self._resolved.append((kind, mode, cfg, role, mod))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_summary_coordination(self, sample: dict, feature_dim: int) -> torch.Tensor:
        speaker_fid = sample["speaker_file_id"]
        speaker_pid = speaker_fid.split("_")[-1]

        key = _coord_key(
            sample["interaction_id"],
            sample["start_s"],
            sample["end_s"],
            speaker_pid,
        )

        arr = self.coordination_lookup.get(key)

        if arr is None:
            if self.missing_coordination == "error":
                raise KeyError(f"Missing summary coordination features for key={key}")

            arr = np.zeros((1, feature_dim), dtype=np.float32)

        arr = arr.astype(np.float32, copy=False)

        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.ndim != 2:
            raise ValueError(
                f"Summary coordination must have shape (1, F), got {arr.shape}"
            )

        if arr.shape[-1] != feature_dim:
            raise ValueError(
                f"Summary coordination feature dim mismatch: "
                f"expected F={feature_dim}, got shape={arr.shape}"
            )

        return torch.from_numpy(arr)

    def _load_continuous_coordination(self, sample: dict, cfg: dict) -> torch.Tensor:
        feature_dim = int(cfg["feature_dim"])
        target_t = int(round(cfg.get("frame_rate_hz", 0.0) * WINDOW_S))

        arr = _load_continuous_coordination_file(
            base_dir=cfg["dir"],
            sample=sample,
            feature_dim=feature_dim,
            target_t=target_t,
            missing_coordination=self.missing_coordination,
        )

        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        speaker_fid = sample["speaker_file_id"]
        listener_fid = sample["listener_file_id"]
        start_s, end_s = sample["start_s"], sample["end_s"]

        tensors: list[torch.Tensor] = []

        for kind, mode, cfg, role, mod in self._resolved:
            if kind == "coordination":
                if mode == "summary":
                    feature_dim = (
                        int(self.coordination_feature_dim)
                        if self.coordination_feature_dim is not None
                        else int(cfg["feature_dim"])
                    )
                    tensors.append(
                        self._load_summary_coordination(sample, feature_dim)
                    )
                    continue

                if mode == "continuous":
                    tensors.append(
                        self._load_continuous_coordination(sample, cfg)
                    )
                    continue

                raise ValueError(
                    f"coordination modality {mod!r} missing valid "
                    f"coordination_mode; got {mode!r}"
                )

            # Standard spliced modality.
            mod_dir = cfg["dir"]
            fid = speaker_fid if role == "speaker" else listener_fid
            fname = splice_filename_for(start_s, end_s, fid)
            stem = os.path.splitext(fname)[0]
            path_no_ext = os.path.join(mod_dir, fid, stem)

            arr = _load_feature_file(path_no_ext)

            if arr.ndim == 1:
                arr = arr[None, :]

            target_t = int(round(cfg.get("frame_rate_hz", 0.0) * WINDOW_S))
            feature_dim = int(cfg["feature_dim"])

            if target_t > 0:
                arr = _fix_seq_len(arr, target_t=target_t, feature_dim=feature_dim)

            if arr.shape[-1] != feature_dim:
                raise ValueError(
                    f"{mod} feature dim mismatch for {path_no_ext}: "
                    f"expected F={feature_dim}, got shape={arr.shape}"
                )

            tensors.append(torch.from_numpy(arr.astype(np.float32, copy=False)))

        return tensors, sample["label"]

def _collate_streams(batch):
    """Stack each stream across the batch. Returns (list_of_stacked, labels)."""
    streams_per_sample, labels = zip(*batch)
    n_streams = len(streams_per_sample[0])
    stacked = [
        torch.stack([sample[i] for sample in streams_per_sample], dim=0)
        for i in range(n_streams)
    ]
    labels_t = torch.tensor(labels, dtype=torch.long)
    return stacked, labels_t

def make_dataloader(
    samples: list[dict],
    streams: list[dict],
    modality_registry: dict[str, dict],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader over a sample list."""
    ds = TurnTakingDataset(samples, streams, modality_registry)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_streams,
    )

def make_dataloader_coordination(
    samples: list[dict],
    streams: list[dict],
    modality_registry: dict[str, dict],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    coordination_lookup: dict | None = None,
    coordination_feature_dim: int | None = None,
    missing_coordination: str = "zeros",
) -> DataLoader:
    ds = TurnTakingDatasetCoordination(
        samples=samples,
        streams=streams,
        modality_registry=modality_registry,
        coordination_lookup=coordination_lookup,
        coordination_feature_dim=coordination_feature_dim,
        missing_coordination=missing_coordination,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_streams,
    )

# =============================================================================
# Models — ported from model_assets/model_classes.py (the practicum's clean
# reference), generalized to arbitrary feature-dim streams.
# =============================================================================

class GRUClassifier(nn.Module):
    """Unimodal single-stream GRU classifier.

    Forward signature accepts either a single tensor (B, T, F) or a
    length-1 list/tuple of tensors (for uniformity with the fusion
    families' list-of-streams input).
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(
                    f"GRUClassifier expects 1 stream, got {len(x)}"
                )
            x = x[0]
        _, h_n = self.gru(x)
        h = h_n[-1]
        return self.fc2(self.drop(self.relu(self.fc1(h))))

class EarlyFusionGRU(nn.Module):
    """Early fusion: concatenate all streams on the feature axis at every
    timestep, feed a single multi-layer GRU over the joint stream.

    Assumes all streams share the same T (sequence length). For modalities
    sampled at different rates, a pre-alignment step would be needed —
    currently none of our modalities mix in a single early-fusion config.

    Practicum default is `num_layers=3` with dropout=0.3 between layers.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_classes: int = NUM_CLASSES, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        # nn.GRU only applies `dropout` between stacked layers when
        # num_layers > 1; for num_layers=1 torch warns and ignores it.
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=gru_dropout,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        # Verify matching T across streams.
        lengths = {t.shape[1] for t in xs}
        if len(lengths) != 1:
            raise ValueError(
                f"EarlyFusionGRU: streams have mismatched T: "
                f"{sorted(lengths)}. Early fusion requires same-rate streams."
            )
        x = torch.cat(xs, dim=-1)
        _, h_n = self.gru(x)
        return self.fc2(self.drop(self.relu(self.fc1(h_n[-1]))))


class NeuralConcatFusion(nn.Module):
    """N-stream multi-branch GRU with hidden-state concatenation.

    Each stream gets its own GRU, so streams may differ in T (sequence
    length) and feature_dim. The final hidden state of each per-stream
    GRU is concatenated along the feature axis and fed through an FC
    head. Requires N >= 2 streams; for N == 1 use GRUClassifier.

    Stream identity is positional. The order of `feat_dims` at __init__
    must match the order of `xs` at forward time, AND must match the
    order of streams in the experiment config that produced the
    DataLoader, so that state-dict reload (e.g. after sweep_tau or
    checkpoint restore) lines up the right GRU with the right modality.
    """

    def __init__(self, feat_dims: list[int], hidden_size: int = 64,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        if len(feat_dims) < 2:
            raise ValueError(
                f"NeuralConcatFusion expects >= 2 streams; got "
                f"{len(feat_dims)}. For 1 stream use GRUClassifier."
            )
        self.feat_dims = list(feat_dims)
        # nn.ModuleList (NOT a plain Python list) is required for these
        # GRUs to be discovered as submodules of `self`. A plain list
        # would still hold the module objects but PyTorch would not
        # register their parameters with the parent — Adam would see no
        # parameters from them and they'd silently never train.
        self.grus = nn.ModuleList([
            nn.GRU(d, hidden_size, batch_first=True) for d in feat_dims
        ])
        self.fc1 = nn.Linear(hidden_size * len(feat_dims), hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != len(self.feat_dims):
            raise ValueError(
                f"NeuralConcatFusion expects {len(self.feat_dims)} streams; "
                f"got {len(xs) if isinstance(xs, (list, tuple)) else 'non-list'}"
            )
        last_hidden: list[torch.Tensor] = []
        for gru, x in zip(self.grus, xs):
            _, h = gru(x)
            last_hidden.append(h[-1])  # (B, hidden_size)
        h = torch.cat(last_hidden, dim=-1)  # (B, hidden_size * N)
        return self.fc2(self.drop(self.relu(self.fc1(h))))

class SelfAttentionFusion(nn.Module):
    """
    Multi-stream self-attention fusion.

    Each stream gets:
      1. Linear projection from raw feature_dim -> attention_dim
      2. N stacked TransformerEncoderLayer blocks
      3. Temporal pooling into one hidden vector

    Then all stream vectors are concatenated and passed to an FC classifier.

    This handles:
      - CPC streams:          (B, T_cpc, F_cpc)
      - OpenFace streams:     (B, T_of, F_of)
      - summary coordination: (B, 1, F_summary)
      - continuous WCC:       (B, T_wcc, F_wcc)

    SELF-ATTN(x) = TRANSFORMER(q=x, k=x, v=x) according to VAP paper
    In PyTorch, TransformerEncoderLayer implements this self-attention pattern.
    """

    def __init__(
        self,
        feat_dims: list[int],
        attention_dim: int = 64,
        num_heads: int | list[int] = 4,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()

        if len(feat_dims) < 1:
            raise ValueError("SelfAttentionFusion needs at least one stream")

        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")

        self.feat_dims = list(feat_dims)
        self.attention_dim = int(attention_dim)
        self.num_layers = int(num_layers)
        self.pooling = pooling

        if isinstance(num_heads, int):
            heads_per_stream = [num_heads] * len(feat_dims)
        else:
            heads_per_stream = list(num_heads)
            if len(heads_per_stream) != len(feat_dims):
                raise ValueError(
                    f"num_heads list must match number of streams: "
                    f"{len(heads_per_stream)} vs {len(feat_dims)}"
                )

        for h in heads_per_stream:
            if self.attention_dim % h != 0:
                raise ValueError(
                    f"attention_dim={self.attention_dim} must be divisible "
                    f"by num_heads={h}"
                )

        self.input_projs = nn.ModuleList([
            nn.Linear(d, self.attention_dim) for d in feat_dims
        ])


        self.transformers = nn.ModuleList() #one branch per stream
        for h in heads_per_stream:
            layer = nn.TransformerEncoderLayer( #one transformer per head per stream
                d_model=self.attention_dim,
                nhead=h,
                dim_feedforward=self.attention_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            encoder = nn.TransformerEncoder( #stack num_layers transformers
                encoder_layer=layer,
                num_layers=self.num_layers,
            )
            self.transformers.append(encoder) #add stream-specific encoder to the list.

        #just initializing the layers here to call when they are being used
        self.fc1 = nn.Linear(self.attention_dim * len(feat_dims), attention_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(attention_dim, num_classes) 

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, T, D)
        returns: (B, D)
        """
        if self.pooling == "last":
            return z[:, -1, :]
        return z.mean(dim=1)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)):
            xs = [xs]

        if len(xs) != len(self.feat_dims):
            raise ValueError(
                f"SelfAttentionFusion expects {len(self.feat_dims)} streams; "
                f"got {len(xs)}"
            )

        pooled = []

        #loop through each stream and grab correct modules
        for x, proj, encoder, expected_dim in zip(
            xs,
            self.input_projs,
            self.transformers,
            self.feat_dims,
        ):
            if x.ndim != 3:
                raise ValueError(
                    f"Each stream must have shape (B, T, F); got {x.shape}"
                )

            if x.shape[-1] != expected_dim:
                raise ValueError(
                    f"Stream feature dim mismatch: expected F={expected_dim}, "
                    f"got shape={x.shape}"
                )

            # Project raw modality features into shared attention dimension.
            z = proj(x)  # (B, T, attention_dim)

            # Self-attention: q = k = v = z internally.
            z = encoder(z)  # (B, T, attention_dim) #encode each stream

            pooled.append(self._pool(z))

        h = torch.cat(pooled, dim=-1)
        return self.fc2(self.drop(self.relu(self.fc1(h))))
    
# =============================================================================
# Training / evaluation
# =============================================================================

def compute_class_weights(
    samples: list[dict], num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    weight[c] = total_samples / (num_classes * count[c]).
    Classes absent from `samples` are given weight 0 (they can't appear).

    Raises:
        ValueError: if `samples` is empty. An all-zero weight vector
            would silently produce a zero-loss training loop with no
            error otherwise, so we fail fast instead.
    """
    if not samples:
        raise ValueError(
            "compute_class_weights called on an empty sample list — "
            "no training data available. This usually indicates an "
            "earlier filter (manifest mismatch, label parsing, "
            "training-class membership) silently dropped everything; "
            "verify enumerate_samples() output before retrying."
        )
    counts = Counter(s["label"] for s in samples)
    total = sum(counts.values())
    weights: list[float] = []
    for c in range(num_classes):
        n = counts.get(c, 0)
        weights.append(total / (num_classes * n) if n > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32)

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    desc: str | None = None,
    show_progress: bool = False,
):
    """Standard train/eval loop. optimizer=None → eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total = 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        iterator = loader
        if show_progress:
            iterator = tqdm(loader, desc=desc or "batches", leave=False)

        for streams, labels in iterator:
            streams = [t.to(device) for t in streams]
            labels = labels.to(device)
            logits = model(streams)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / max(total, 1), all_preds, all_labels

def f1_stats(
    preds: list[int], labels: list[int],
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Compute per-class F1 + macro-averaged F1 in a single pass.

    Returns a dict with keys:
        'macro_f1'     — float, mean of per-class F1
        'per_class_f1' — {class_name: float}, one entry per class index

    Convention for absent classes (match sklearn's
    `f1_score(..., average='macro', zero_division=0)`):

      - If a class never appears as a prediction OR never appears in
        the ground-truth labels, its F1 contribution is 0 and it IS
        included in the macro mean (i.e. the mean divides by
        num_classes, not by the number of present classes). This drags
        the macro-F1 down when a class is entirely missing — a
        deliberate choice so a model that "solves" a binary subset of
        the 3-class problem doesn't score well on the 3-class macro.
    """
    per_class: dict[str, float] = {}
    for c in range(num_classes):
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        if tp + fp == 0 or tp + fn == 0:
            per_class[CLASS_NAMES[c]] = 0.0
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        per_class[CLASS_NAMES[c]] = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
    macro = sum(per_class.values()) / num_classes if num_classes else 0.0
    return {"macro_f1": macro, "per_class_f1": per_class}

def macro_f1(preds: list[int], labels: list[int],
             num_classes: int = NUM_CLASSES) -> float:
    """Thin wrapper around `f1_stats` for callers that only need the scalar."""
    return f1_stats(preds, labels, num_classes)["macro_f1"]

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str | torch.device,
    patience: int = 4,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Train for up to N epochs with patience-based early stopping on
    val-macro-F1. Returns (model_with_best_val_weights, history).

    Matches the practicum's train_model contract (patience=4) but
    monitors val macro-F1 instead of accuracy — macro-F1 is better
    suited to the class-imbalanced 3-class turn-taking setup.

    Early-stop semantics:
      - Track best val-macro-F1 seen so far.
      - Increment a no-improve counter every epoch that doesn't beat
        the best. Reset to 0 on improvement.
      - Stop when no-improve >= patience.
      - Always restore best-val weights before returning, EVEN if the
        training loop raises (Ctrl-C, OOM, etc.) — wrapped in
        try / finally.

    Set `patience >= epochs` to disable early stopping.
    """
    best_f1 = -1.0
    best_state: dict | None = None
    no_improve = 0
    history: list[dict] = []

    epoch_iter = tqdm(range(1, epochs + 1), total=epochs, desc="Epochs", leave=True)

    try:
        for epoch in epoch_iter:
            t0 = time.time()

            tr_loss, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, device,
                desc=f"train epoch {epoch}",
                show_progress=True,
            )

            va_loss, va_p, va_y = run_epoch(
                model, val_loader, criterion, None, device,
                desc=f"val epoch {epoch}",
                show_progress=True,
            )

            va_f1 = macro_f1(va_p, va_y)

            history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_macro_f1": va_f1,
                "elapsed_s": time.time() - t0,
            })

            improved = va_f1 > best_f1

            if improved:
                best_f1 = va_f1
                best_state = {
                    k: v.detach().clone().cpu()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            epoch_iter.set_postfix({
                "train_loss": f"{tr_loss:.4f}",
                "val_loss": f"{va_loss:.4f}",
                "val_f1": f"{va_f1:.4f}",
                "no_improve": f"{no_improve}/{patience}",
            })

            if verbose:
                marker = " *" if improved else "  "
                print(
                    f"  epoch {epoch:>2}/{epochs}  "
                    f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                    f"val_macroF1={va_f1:.4f}{marker} "
                    f"(no_improve={no_improve}/{patience}, "
                    f"{history[-1]['elapsed_s']:.1f}s)"
                )

            if no_improve >= patience:
                if verbose:
                    print(
                        f"  early stopping at epoch {epoch} "
                        f"(best val macroF1={best_f1:.4f})"
                    )
                break
    finally:
        # Always restore best-val weights before returning, even on
        # exception. Without this, a Ctrl-C mid-epoch leaves `model`
        # holding the latest (possibly worse) weights.
        if best_state is not None:
            model.load_state_dict(best_state)

    return model, history

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict:
    """Evaluate on a loader; return loss, preds, labels, macro-F1, per-class F1.

    Uses a single `f1_stats` call — per-class F1 is computed once and
    reused in the macro average, so we don't scan the predictions twice.
    """
    loss, preds, labels = run_epoch(model, loader, criterion, None, device)
    stats = f1_stats(preds, labels)
    return {
        "loss": loss,
        "macro_f1": stats["macro_f1"],
        "per_class_f1": stats["per_class_f1"],
        "preds": preds,
        "labels": labels,
    }

# =============================================================================
# Experiment runner
# =============================================================================

def build_model(
    fusion: str,
    stream_dims: list[int],
    hidden_size: int = 64,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.3,
    num_layers_early: int = 3,
    attention_dim: int | None = None,
    attention_heads: int | list[int] = 4,
    attention_layers: int = 2,
    attention_pooling: str = "mean",
) -> nn.Module:
    """
    Dispatch to the right model class.

    Supported fusion:
      - unimodal
      - early
      - neural_concat
      - self_attention
    """
    if fusion == "unimodal":
        if len(stream_dims) != 1:
            raise ValueError(
                f"unimodal fusion needs exactly 1 stream; got {len(stream_dims)}"
            )
        return GRUClassifier(
            stream_dims[0],
            hidden_size,
            num_classes,
            dropout=dropout,
        )

    if fusion == "early":
        return EarlyFusionGRU(
            sum(stream_dims),
            hidden_size,
            num_classes,
            num_layers=num_layers_early,
            dropout=dropout,
        )

    if fusion == "neural_concat":
        return NeuralConcatFusion(
            stream_dims,
            hidden_size,
            num_classes,
            dropout=dropout,
        )

    if fusion in {"self_attention", "attention"}:
        return SelfAttentionFusion(
            feat_dims=stream_dims,
            attention_dim=attention_dim or hidden_size,
            num_heads=attention_heads,
            num_layers=attention_layers,
            num_classes=num_classes,
            dropout=dropout,
            pooling=attention_pooling,
        )

    raise ValueError(
        f"unknown fusion family {fusion!r}; supported: "
        f"unimodal, early, neural_concat, self_attention"
    )

def resolve_device(prefer: str = "auto") -> torch.device:
    """Pick cuda > mps > cpu unless overridden."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_experiment(
    *,
    name: str,
    streams: list[dict],
    fusion: str,
    modality_registry: dict[str, dict],
    manifest_rows: list[dict],
    labels_path: str,
    hidden_size: int = 64,
    dropout: float = 0.3,
    num_layers_early: int = 3,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 4,
    num_workers: int = 0,
    device: torch.device | str = "auto",
    seed: int = 42,
    max_samples_per_split: int | None = None,
    verbose: bool = True,

    # self-attention options
    attention_dim: int | None = None,
    attention_heads: int | list[int] = 4,
    attention_layers: int = 2,
    attention_pooling: str = "mean",
) -> dict:
    """
    Train one non-coordination experiment end-to-end.

    Supports:
      - standard GRU: unimodal / early / neural_concat
      - SSA: self_attention over non-coordination streams
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_t = resolve_device(device) if isinstance(device, str) else device

    labels = load_labels(labels_path)
    samples = enumerate_samples(manifest_rows, labels)
    by_split = split_samples(samples)

    has_file_stream = any(
        modality_registry[s["modality"]].get("kind", "spliced") != "coordination"
        for s in streams
    )

    if has_file_stream:
        filtered_by_split = {}
        total_dropped = 0

        for split_name, samples_for_split in by_split.items():
            kept, dropped = filter_samples_with_existing_files(
                samples_for_split,
                streams,
                modality_registry,
            )
            filtered_by_split[split_name] = kept
            total_dropped += len(dropped)

            if verbose and dropped:
                print(
                    f"  dropped {len(dropped)} {split_name} samples "
                    f"missing feature files"
                )
                if dropped[0][1]:
                    print("  example missing:", dropped[0][1][0])

        by_split = filtered_by_split
        samples = [s for sample_list in by_split.values() for s in sample_list]

    if max_samples_per_split is not None:
        rng = random.Random(seed)
        for split in list(by_split.keys()):
            rng.shuffle(by_split[split])
            by_split[split] = by_split[split][:max_samples_per_split]
        samples = [
            s
            for split_samples_list in by_split.values()
            for s in split_samples_list
        ]

    stream_dims = [int(modality_registry[s["modality"]]["feature_dim"]) for s in streams]

    if verbose:
        print(f"=== Experiment: {name} ===")
        print(f"  streams:     {streams}")
        print(f"  fusion:      {fusion}")
        print(f"  stream_dims: {stream_dims}")
        print(f"  device:      {device_t}")

        summary = summarize_samples(samples)
        print(
            f"  samples:     total={summary['total']} "
            f"per_split={summary['per_split']}"
        )
        for sp, dist in summary["per_split_class"].items():
            print(f"               {sp:5s} class-dist: {dist}")

    train_loader = make_dataloader(
        by_split.get("train", []),
        streams,
        modality_registry,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = make_dataloader(
        by_split.get("val", []),
        streams,
        modality_registry,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = make_dataloader(
        by_split.get("test", []),
        streams,
        modality_registry,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(
        fusion,
        stream_dims,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers_early=num_layers_early,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        attention_layers=attention_layers,
        attention_pooling=attention_pooling,
    ).to(device_t)

    class_weights = compute_class_weights(by_split.get("train", [])).to(device_t)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model:       {type(model).__name__} ({n_params:,} params)")
        print(f"  class_w:     {class_weights.cpu().tolist()}")
        print(f"  epochs:      up to {epochs} (patience={patience})")
        print(f"  dropout:     {dropout}")
        if fusion in {"self_attention", "attention"}:
            print(
                f"  attention:   dim={attention_dim or hidden_size}, "
                f"heads={attention_heads}, layers={attention_layers}, "
                f"pooling={attention_pooling}"
            )
        print()

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=epochs,
        patience=patience,
        device=device_t,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  final test eval ({name}):")

    test_eval = evaluate(model, test_loader, criterion, device_t)

    if verbose:
        print(f"    macro_f1 = {test_eval['macro_f1']:.4f}")
        print(f"    per-class F1 = {test_eval['per_class_f1']}")

    return {
        "name": name,
        "config": {
            "streams": streams,
            "fusion": fusion,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_layers_early": num_layers_early,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "labels_path": labels_path,
            "attention_dim": attention_dim,
            "attention_heads": attention_heads,
            "attention_layers": attention_layers,
            "attention_pooling": attention_pooling,
        },
        "stream_dims": stream_dims,
        "device": str(device_t),
        "samples_summary": summarize_samples(samples),
        "train_history": history,
        "test_eval": {
            "loss": test_eval["loss"],
            "macro_f1": test_eval["macro_f1"],
            "per_class_f1": test_eval["per_class_f1"],
        },
        "model_state": model.state_dict(),
    }


# =============================================================================
# Experiment runner with coordination
# =============================================================================


def run_experiment_coordination(
    *,
    name: str,
    streams: list[dict],
    fusion: str,
    modality_registry: dict[str, dict],
    manifest_rows: list[dict],
    labels_path: str,
    hidden_size: int = 64,
    dropout: float = 0.3,
    num_layers_early: int = 3,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 4,
    num_workers: int = 0,
    device: torch.device | str = "auto",
    seed: int = 42,
    max_samples_per_split: int | None = None,
    verbose: bool = True,
    coordination_csv_path: str | None = None,
    coordination_feature_cols: list[str] = COORDINATION_FEATURE_COLUMNS,
    missing_coordination: str = "zeros",

    # self-attention options
    attention_dim: int | None = None,
    attention_heads: int | list[int] = 4,
    attention_layers: int = 2,
    attention_pooling: str = "mean",
) -> dict:
    """
    Train one experiment end-to-end with optional coordination streams.

    Coordination streams can be:
      - summary: loaded from CSV lookup, shape (1, F)
      - continuous: loaded from .npy, shape (T_wcc, F_wcc)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_t = resolve_device(device) if isinstance(device, str) else device

    labels = load_labels(labels_path)
    samples = enumerate_samples(manifest_rows, labels)
    by_split = split_samples(samples)

    has_file_stream = any(
        modality_registry[s["modality"]].get("kind", "spliced") != "coordination"
        for s in streams
    )

    has_summary_coordination = any(
        modality_registry[s["modality"]].get("kind", "spliced") == "coordination"
        and modality_registry[s["modality"]].get("coordination_mode", "summary") == "summary"
        for s in streams
    )

    has_continuous_coordination = any(
        modality_registry[s["modality"]].get("kind", "spliced") == "coordination"
        and modality_registry[s["modality"]].get("coordination_mode") == "continuous"
        for s in streams
    )

    # Filter missing file-backed streams: CPC/OpenFace/etc.
    # Continuous coordination files are handled by the dataset because they are
    # coordination pseudo-streams and can use missing_coordination='zeros' or 'error'.
    if has_file_stream:
        filtered_by_split = {}
        total_dropped = 0

        for split_name, samples_for_split in by_split.items():
            kept, dropped = filter_samples_with_existing_files(
                samples_for_split,
                streams,
                modality_registry,
            )

            filtered_by_split[split_name] = kept
            total_dropped += len(dropped)

            if verbose and dropped:
                print(
                    f"  dropped {len(dropped)} {split_name} samples "
                    f"missing feature files"
                )
                if dropped[0][1]:
                    print("  example missing:", dropped[0][1][0])

        by_split = filtered_by_split
        samples = [s for sample_list in by_split.values() for s in sample_list]

    # Summary coordination needs CSV. Continuous coordination does not.
    if has_summary_coordination:
        if coordination_csv_path is None:
            raise ValueError(
                "At least one summary coordination stream is configured, "
                "but coordination_csv_path=None."
            )

        coordination_lookup = load_coordination_features(
            coordination_csv_path,
            feature_cols=coordination_feature_cols,
        )
        summary_coordination_dim = len(coordination_feature_cols)
    else:
        coordination_lookup = {}
        summary_coordination_dim = None

    if max_samples_per_split is not None:
        rng = random.Random(seed)
        for split in list(by_split.keys()):
            rng.shuffle(by_split[split])
            by_split[split] = by_split[split][:max_samples_per_split]
        samples = [
            s
            for split_samples_list in by_split.values()
            for s in split_samples_list
        ]

    stream_dims = []
    for s in streams:
        cfg = modality_registry[s["modality"]]
        if cfg.get("kind", "spliced") == "coordination":
            mode = cfg.get("coordination_mode", "summary")
            if mode == "summary":
                stream_dims.append(summary_coordination_dim or int(cfg["feature_dim"]))
            elif mode == "continuous":
                stream_dims.append(int(cfg["feature_dim"]))
            else:
                raise ValueError(f"Unknown coordination mode: {mode!r}")
        else:
            stream_dims.append(int(cfg["feature_dim"]))

    has_coordination = has_summary_coordination or has_continuous_coordination

    if has_coordination and fusion == "early":
        raise ValueError(
            "Early fusion with coordination is not supported by default. "
            "Summary coordination has T=1 and continuous WCC usually has a "
            "different T than CPC/OpenFace. Use fusion='neural_concat' or "
            "fusion='self_attention'."
        )

    if verbose:
        print(f"=== Experiment: {name} ===")
        print(f"  streams:   {streams}")
        print(f"  fusion:    {fusion}")
        print(f"  stream_dims: {stream_dims}")
        print(f"  device:    {device_t}")
        print(f"  summary_coordination:    {has_summary_coordination}")
        print(f"  continuous_coordination: {has_continuous_coordination}")

        summary = summarize_samples(samples)
        print(
            f"  samples:   total={summary['total']}  "
            f"per_split={summary['per_split']}"
        )
        for sp, dist in summary["per_split_class"].items():
            print(f"             {sp:5s} class-dist: {dist}")

    train_loader = make_dataloader_coordination(
        by_split.get("train", []),
        streams,
        modality_registry,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        coordination_lookup=coordination_lookup,
        coordination_feature_dim=summary_coordination_dim,
        missing_coordination=missing_coordination,
    )

    val_loader = make_dataloader_coordination(
        by_split.get("val", []),
        streams,
        modality_registry,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        coordination_lookup=coordination_lookup,
        coordination_feature_dim=summary_coordination_dim,
        missing_coordination=missing_coordination,
    )

    test_loader = make_dataloader_coordination(
        by_split.get("test", []),
        streams,
        modality_registry,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        coordination_lookup=coordination_lookup,
        coordination_feature_dim=summary_coordination_dim,
        missing_coordination=missing_coordination,
    )

    model = build_model(
        fusion,
        stream_dims,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers_early=num_layers_early,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        attention_layers=attention_layers,
        attention_pooling=attention_pooling,
    ).to(device_t)

    class_weights = compute_class_weights(by_split.get("train", [])).to(device_t)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model:     {type(model).__name__}  ({n_params:,} params)")
        print(f"  class_w:   {class_weights.cpu().tolist()}")
        print(f"  epochs:    up to {epochs} (patience={patience})")
        print(f"  dropout:   {dropout}")
        if fusion in {"self_attention", "attention"}:
            print(
                f"  attention: dim={attention_dim or hidden_size}, "
                f"heads={attention_heads}, layers={attention_layers}, "
                f"pooling={attention_pooling}"
            )
        print()

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=epochs,
        patience=patience,
        device=device_t,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  final test eval ({name}):")

    test_eval = evaluate(model, test_loader, criterion, device_t)

    if verbose:
        print(f"    macro_f1 = {test_eval['macro_f1']:.4f}")
        print(f"    per-class F1 = {test_eval['per_class_f1']}")

    return {
        "name": name,
        "config": {
            "streams": streams,
            "fusion": fusion,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_layers_early": num_layers_early,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "labels_path": labels_path,
            "coordination_csv_path": coordination_csv_path,
            "coordination_feature_cols": coordination_feature_cols,
            "missing_coordination": missing_coordination,
            "attention_dim": attention_dim,
            "attention_heads": attention_heads,
            "attention_layers": attention_layers,
            "attention_pooling": attention_pooling,
        },
        "stream_dims": stream_dims,
        "device": str(device_t),
        "samples_summary": summarize_samples(samples),
        "train_history": history,
        "test_eval": {
            "loss": test_eval["loss"],
            "macro_f1": test_eval["macro_f1"],
            "per_class_f1": test_eval["per_class_f1"],
        },
        "model_state": model.state_dict(),
        "coordination_csv_path": coordination_csv_path,
        "coordination_feature_cols": coordination_feature_cols,
        "missing_coordination": missing_coordination,
    }

# =============================================================================
# τ-sweep: re-evaluate trained weights against each per-τ label file
# =============================================================================
def sweep_tau(
    *,
    experiment_result: dict,
    modality_registry: dict[str, dict],
    manifest_rows: list[dict],
    labels_dir: str,
    tau_grid_ms: tuple[int, ...] = DEFAULT_TAU_GRID_MS,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | str = "auto",
    max_samples_per_split: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[int, dict]:
    """
    Evaluate on trained non-coordination model across tau label files.

    Rebuild test loaders at each τ, re-evaluate the trained model.

    Returns: {tau_ms: {'macro_f1': float, 'per_class_f1': dict,
                       'loss': float, 'n_samples': int}}.

    Input shape (features) is τ-invariant — only labels change per τ. So
    we keep the same trained weights and just swap the label dict.

    Loss criterion: class-weighted CrossEntropyLoss using the SAME weights
    that `run_experiment` computed during training (derived from the
    train-split at the training τ, recorded via `config['labels_path']`).
    Keeping the criterion consistent with the training objective means
    the per-τ loss numbers are directly interpretable as "how well did
    the optimization target transfer to this τ."

    Works for:
      - standard GRU runs
      - SSA self_attention runs
    """
    cfg = experiment_result["config"]
    streams = cfg["streams"]
    fusion = cfg["fusion"]
    hidden_size = cfg["hidden_size"]
    dropout = cfg.get("dropout", 0.3)
    num_layers_early = cfg.get("num_layers_early", 3)
    train_labels_path = cfg["labels_path"]
    stream_dims = experiment_result["stream_dims"]

    device_t = resolve_device(device) if isinstance(device, str) else device

    model = build_model(
        fusion,
        stream_dims,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers_early=num_layers_early,
        attention_dim=cfg.get("attention_dim"),
        attention_heads=cfg.get("attention_heads", 4),
        attention_layers=cfg.get("attention_layers", 2),
        attention_pooling=cfg.get("attention_pooling", "mean"),
    ).to(device_t)

    model.load_state_dict(experiment_result["model_state"])

    train_labels = load_labels(train_labels_path)
    train_samples = enumerate_samples(manifest_rows, train_labels)
    train_split = [s for s in train_samples if s["split"] == "train"]

    train_split, _ = filter_samples_with_existing_files(
        train_split,
        streams,
        modality_registry,
    )

    if not train_split:
        raise ValueError(
            f"sweep_tau: no train-split samples from {train_labels_path}"
        )

    class_weights = compute_class_weights(train_split).to(device_t)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    out: dict[int, dict] = {}

    for tau_ms in tau_grid_ms:
        labels_path = os.path.join(labels_dir, f"labels_tau_{tau_ms:04d}.json")

        if not os.path.exists(labels_path):
            if verbose:
                print(f"  τ={tau_ms}ms: labels file missing at {labels_path}; skipping")
            continue

        labels = load_labels(labels_path)
        samples = enumerate_samples(manifest_rows, labels)
        test_samples = [s for s in samples if s["split"] == "test"]

        test_samples, dropped = filter_samples_with_existing_files(
            test_samples,
            streams,
            modality_registry,
        )

        if verbose and dropped:
            print(f"  τ={tau_ms}ms: dropped {len(dropped)} missing-file samples")
            if dropped[0][1]:
                print("  example missing:", dropped[0][1][0])

        if max_samples_per_split is not None:
            rng_tau = random.Random(seed)
            rng_tau.shuffle(test_samples)
            test_samples = test_samples[:max_samples_per_split]

        if not test_samples:
            if verbose:
                print(f"  τ={tau_ms}ms: no test samples; skipping")
            continue

        loader = make_dataloader(
            test_samples,
            streams,
            modality_registry,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        ev = evaluate(model, loader, criterion, device_t)

        out[tau_ms] = {
            "macro_f1": ev["macro_f1"],
            "per_class_f1": ev["per_class_f1"],
            "loss": ev["loss"],
            "n_samples": len(test_samples),
        }

        if verbose:
            print(
                f"  τ={tau_ms:>4}ms  n={len(test_samples):>5}  "
                f"loss={ev['loss']:.4f}  macroF1={ev['macro_f1']:.4f}  "
                f"per-class={ev['per_class_f1']}"
            )

    return out

def sweep_tau_coordination(
    *,
    experiment_result: dict,
    modality_registry: dict[str, dict],
    manifest_rows: list[dict],
    labels_dir: str,
    tau_grid_ms: tuple[int, ...] = DEFAULT_TAU_GRID_MS,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | str = "auto",
    max_samples_per_split: int | None = None,
    seed: int = 42,
    verbose: bool = True,
    coordination_csv_path: str | None = None,
    coordination_feature_cols: list[str] = COORDINATION_FEATURE_COLUMNS,
    missing_coordination: str = "zeros",
) -> dict[int, dict]:
    """Rebuild test loaders at each τ, re-evaluate the trained model.

    Returns: {tau_ms: {'macro_f1': float, 'per_class_f1': dict,
                       'loss': float, 'n_samples': int}}.

    Input shape (features) is τ-invariant — only labels change per τ. So
    we keep the same trained weights and just swap the label dict.

    Loss criterion: class-weighted CrossEntropyLoss using the SAME weights
    that `run_experiment` computed during training (derived from the
    train-split at the training τ, recorded via `config['labels_path']`).
    Keeping the criterion consistent with the training objective means
    the per-τ loss numbers are directly interpretable as "how well did
    the optimization target transfer to this τ."
    """
    cfg = experiment_result["config"]
    streams = cfg["streams"]
    fusion = cfg["fusion"]
    hidden_size = cfg["hidden_size"]
    dropout = cfg.get("dropout", 0.3)
    num_layers_early = cfg.get("num_layers_early", 3)
    train_labels_path = cfg["labels_path"]
    stream_dims = experiment_result["stream_dims"]
    device_t = resolve_device(device) if isinstance(device, str) else device

    coordination_csv_path = (
    coordination_csv_path
    if coordination_csv_path is not None
    else cfg.get("coordination_csv_path")
)

    coordination_feature_cols = (
        coordination_feature_cols
        if coordination_feature_cols is not None
        else cfg.get("coordination_feature_cols", COORDINATION_FEATURE_COLUMNS)
    )

    missing_coordination = cfg.get("missing_coordination", missing_coordination)

    coordination_lookup = load_coordination_features(
        coordination_csv_path,
        feature_cols=coordination_feature_cols,
    ) if coordination_csv_path is not None else {}

    coordination_feature_dim = len(coordination_feature_cols)

    # Rebuild the model with the SAME architectural hyperparams as
    # training so the state dict loads cleanly (in particular,
    # num_layers_early affects the shape of the GRU's parameter tensors
    # for EarlyFusionGRU).
    model = build_model(
        fusion, stream_dims,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers_early=num_layers_early,
    ).to(device_t)
    model.load_state_dict(experiment_result["model_state"])

    # Reconstruct the training criterion — class-weighted CE from the
    # train split at training-τ. This matches what `run_experiment`
    # optimized against, so per-τ loss numbers are comparable to
    # per-epoch train_loss in the training history.
    train_labels = load_labels(train_labels_path)
    train_samples = enumerate_samples(manifest_rows, train_labels)
    train_split = [s for s in train_samples if s["split"] == "train"]
    if not train_split:
        raise ValueError(
            f"sweep_tau: no train-split samples derivable from "
            f"{train_labels_path}; cannot reconstruct class-weighted "
            f"criterion. Did the manifest change between training "
            f"and the τ-sweep?"
        )
    class_weights = compute_class_weights(train_split).to(device_t)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    out: dict[int, dict] = {}
    rng = random.Random(seed)
    for tau_ms in tau_grid_ms:
        labels_path = os.path.join(labels_dir, f"labels_tau_{tau_ms:04d}.json")
        if not os.path.exists(labels_path):
            if verbose:
                print(f"  τ={tau_ms}ms: labels file missing at {labels_path}; skipping")
            continue
        labels = load_labels(labels_path)
        samples = enumerate_samples(manifest_rows, labels)
        test_samples = [s for s in samples if s["split"] == "test"]

        # Always filter missing feature files, regardless of max_samples_per_split
        has_file_stream = any(
            modality_registry[s["modality"]].get("kind", "spliced") != "coordination"
            for s in streams
        )

        if has_file_stream:
            test_samples, dropped = filter_samples_with_existing_files(
                test_samples,
                streams,
                modality_registry,
            )

            if verbose and dropped:
                print(f"  τ={tau_ms}ms: dropped {len(dropped)} test samples missing feature files")
                if dropped[0][1]:
                    print("  example missing:", dropped[0][1][0])

        if max_samples_per_split is not None:
            # Shuffle before truncating so the dry-run subset is
            # representative (same rationale as run_experiment). Uses
            # a per-τ-stable seed so different τs see the same subset
            # of test samples where possible (basenames that exist at
            # every τ stay aligned across τs).
            rng_tau = random.Random(seed)
            rng_tau.shuffle(test_samples)
            test_samples = test_samples[:max_samples_per_split]
            

        if not test_samples:
            if verbose:
                print(f"  τ={tau_ms}ms: no test samples; skipping")
            continue
        loader = make_dataloader_coordination(
            test_samples, streams, modality_registry,
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            coordination_lookup=coordination_lookup,
            coordination_feature_dim=coordination_feature_dim,
            missing_coordination=missing_coordination,
        )
        ev = evaluate(model, loader, criterion, device_t)
        out[tau_ms] = {
            "macro_f1": ev["macro_f1"],
            "per_class_f1": ev["per_class_f1"],
            "loss": ev["loss"],
            "n_samples": len(test_samples),
        }
        if verbose:
            print(f"  τ={tau_ms:>4}ms  n={len(test_samples):>5}  "
                  f"loss={ev['loss']:.4f}  macroF1={ev['macro_f1']:.4f}  "
                  f"per-class={ev['per_class_f1']}")
    return out


def sweep_tau_coordination_safe(
    *,
    experiment_result: dict,
    modality_registry: dict[str, dict],
    manifest_rows: list[dict],
    labels_dir: str,
    tau_grid_ms: tuple[int, ...] = DEFAULT_TAU_GRID_MS,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | str = "auto",
    max_samples_per_split: int | None = None,
    seed: int = 42,
    verbose: bool = True,
    coordination_csv_path: str | None = None,
    coordination_feature_cols: list[str] | None = None,
    missing_coordination: str = "zeros",
) -> dict[int, dict]:
    """
    Evaluate one trained coordination model across tau label files.

    Works for:
      - coordination summary GRU/neural_concat
      - CSA continuous coordination self_attention
    """
    cfg = experiment_result["config"]
    streams = cfg["streams"]
    fusion = cfg["fusion"]

    device_t = resolve_device(device) if isinstance(device, str) else device

    coordination_csv_path = (
        coordination_csv_path
        if coordination_csv_path is not None
        else cfg.get("coordination_csv_path")
    )

    coordination_feature_cols = (
        coordination_feature_cols
        if coordination_feature_cols is not None
        else cfg.get("coordination_feature_cols", COORDINATION_FEATURE_COLUMNS)
    )

    missing_coordination = cfg.get("missing_coordination", missing_coordination)

    has_summary_coordination = any(
        modality_registry[s["modality"]].get("kind", "spliced") == "coordination"
        and modality_registry[s["modality"]].get("coordination_mode", "summary") == "summary"
        for s in streams
    )

    has_file_stream = any(
        modality_registry[s["modality"]].get("kind", "spliced") != "coordination"
        for s in streams
    )

    if has_summary_coordination:
        if coordination_csv_path is None:
            raise ValueError(
                "Summary coordination stream needs coordination_csv_path."
            )

        coordination_lookup = load_coordination_features(
            coordination_csv_path,
            feature_cols=coordination_feature_cols,
        )
        coordination_feature_dim = len(coordination_feature_cols)
    else:
        coordination_lookup = {}
        coordination_feature_dim = None

    model = build_model(
        fusion,
        experiment_result["stream_dims"],
        hidden_size=cfg["hidden_size"],
        dropout=cfg.get("dropout", 0.3),
        num_layers_early=cfg.get("num_layers_early", 3),
        attention_dim=cfg.get("attention_dim"),
        attention_heads=cfg.get("attention_heads", 4),
        attention_layers=cfg.get("attention_layers", 2),
        attention_pooling=cfg.get("attention_pooling", "mean"),
    ).to(device_t)

    model.load_state_dict(experiment_result["model_state"])

    train_labels = load_labels(cfg["labels_path"])
    train_samples = enumerate_samples(manifest_rows, train_labels)
    train_samples = [s for s in train_samples if s["split"] == "train"]

    if has_file_stream:
        train_samples, _ = filter_samples_with_existing_files(
            train_samples,
            streams,
            modality_registry,
        )

    if not train_samples:
        raise ValueError("No train samples available for class-weight reconstruction.")

    class_weights = compute_class_weights(train_samples).to(device_t)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    out: dict[int, dict] = {}

    for tau_ms in tau_grid_ms:
        labels_path = os.path.join(labels_dir, f"labels_tau_{tau_ms:04d}.json")

        if not os.path.exists(labels_path):
            if verbose:
                print(f"  τ={tau_ms}ms: missing labels, skipping")
            continue

        labels = load_labels(labels_path)
        samples = enumerate_samples(manifest_rows, labels)
        test_samples = [s for s in samples if s["split"] == "test"]

        if has_file_stream:
            before = len(test_samples)
            test_samples, dropped = filter_samples_with_existing_files(
                test_samples,
                streams,
                modality_registry,
            )

            if verbose and dropped:
                print(
                    f"  τ={tau_ms}ms: kept {len(test_samples)}/{before}, "
                    f"dropped {len(dropped)} missing-file samples"
                )
                if dropped[0][1]:
                    print("    example missing:", dropped[0][1][0])

        if max_samples_per_split is not None:
            rng_tau = random.Random(seed)
            rng_tau.shuffle(test_samples)
            test_samples = test_samples[:max_samples_per_split]

        if not test_samples:
            if verbose:
                print(f"  τ={tau_ms}ms: no test samples, skipping")
            continue

        loader = make_dataloader_coordination(
            test_samples,
            streams,
            modality_registry,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            coordination_lookup=coordination_lookup,
            coordination_feature_dim=coordination_feature_dim,
            missing_coordination=missing_coordination,
        )

        ev = evaluate(model, loader, criterion, device_t)

        out[tau_ms] = {
            "macro_f1": ev["macro_f1"],
            "per_class_f1": ev["per_class_f1"],
            "loss": ev["loss"],
            "n_samples": len(test_samples),
        }

        if verbose:
            print(
                f"  τ={tau_ms:>4}ms  n={len(test_samples):>5}  "
                f"loss={ev['loss']:.4f}  macroF1={ev['macro_f1']:.4f}  "
                f"per-class={ev['per_class_f1']}"
            )

    return out

##########################
# COORDINATION ANALYSIS
#########################

def _coord_key(interaction_id: str, start_s: float, end_s: float, speaker: str):
    return (
        str(interaction_id),
        round(float(start_s), 3),
        round(float(end_s), 3),
        str(speaker),
    )


def load_coordination_features(
    coordination_csv_path: str,
    feature_cols: list[str] = COORDINATION_FEATURE_COLUMNS,
) -> dict[tuple, np.ndarray]:
    """
    Load speaker-oriented dyadic coordination features.

    Expected CSV columns:
        interaction_name,start_s,end_s,speaker,listener,<coordination features...>

    Returns:
        lookup[(interaction_id, start_s, end_s, speaker_pid)] = (1, F) float32 array

    Shape is (1, F), so it can be treated as a sequence with T=1 by the GRU.
    Use with neural_concat fusion, not early fusion with CPC/OpenFace unless you
    explicitly repeat it over time.
    """
    if coordination_csv_path is None:
        return {}

    if not os.path.exists(coordination_csv_path):
        raise FileNotFoundError(
            f"coordination CSV not found: {coordination_csv_path}"
        )

    df = pd.read_csv(coordination_csv_path)

    required = {"interaction_name", "start_s", "end_s", "speaker"} | set(feature_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"coordination CSV missing columns: {sorted(missing)}"
        )

    lookup = {}

    for _, row in df.iterrows():
        key = _coord_key(
            row["interaction_name"],
            row["start_s"],
            row["end_s"],
            row["speaker"],
        )

        vals = (
            row[feature_cols]
            .astype(float)
            .to_numpy(dtype=np.float32)
            .reshape(1, -1)
        )

        lookup[key] = vals

    return lookup

##########################
# sequence fixer
#########################

def _fix_seq_len(arr: np.ndarray, target_t: int, feature_dim: int) -> np.ndarray:
    """
    Force arr to shape (target_t, feature_dim) by trimming or zero-padding.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")

    if arr.shape[1] != feature_dim:
        raise ValueError(
            f"Feature dim mismatch: expected {feature_dim}, got {arr.shape[1]}"
        )

    T = arr.shape[0]

    if T == target_t:
        return arr

    if T > target_t:
        return arr[:target_t]

    pad = np.zeros((target_t - T, feature_dim), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)

def filter_samples_with_existing_files(samples, streams, modality_registry):
    kept, dropped = [], []

    for s in samples:
        ok = True
        missing = []

        for stream in streams:
            mod = stream["modality"]
            cfg = modality_registry[mod]

            if cfg.get("kind", "spliced") == "coordination":
                continue

            role = stream["role"]
            fid = s["speaker_file_id"] if role == "speaker" else s["listener_file_id"]

            fname = splice_filename_for(s["start_s"], s["end_s"], fid)
            stem = os.path.splitext(fname)[0]
            path_no_ext = os.path.join(cfg["dir"], fid, stem)

            if not (
                os.path.exists(path_no_ext + ".npy")
                or os.path.exists(path_no_ext + ".json")
            ):
                ok = False
                missing.append(path_no_ext)

        if ok:
            kept.append(s)
        else:
            dropped.append((s, missing))

    return kept, dropped


############
# continuous coodrination helpers
###########

def _format_coord_window_filename(start_s: float, end_s: float, speaker_fid: str) -> str:
    """
    Continuous coordination filename convention:
        0000.00-0002.00_V00_S0691_I00000482_P0500.npy
    """
    return f"{start_s:07.2f}-{end_s:07.2f}_{speaker_fid}.npy"


def _coordination_continuous_candidates(
    base_dir: str,
    sample: dict,
) -> list[str]:
    """
    Try a few reasonable layouts so the loader is robust.

    Preferred:
        base_dir / speaker_fid / 0000.00-0002.00_speaker_fid.npy

    Also tries:
        base_dir / interaction_id / filename
        base_dir / filename
    """
    speaker_fid = sample["speaker_file_id"]
    interaction_id = sample["interaction_id"]
    fname = _format_coord_window_filename(
        sample["start_s"],
        sample["end_s"],
        speaker_fid,
    )

    return [
        os.path.join(base_dir, speaker_fid, fname),
        os.path.join(base_dir, interaction_id, fname),
        os.path.join(base_dir, fname),
    ]


def _load_continuous_coordination_file(
    base_dir: str,
    sample: dict,
    feature_dim: int,
    target_t: int = 0,
    missing_coordination: str = "zeros",
) -> np.ndarray:
    candidates = _coordination_continuous_candidates(base_dir, sample)

    path = next((p for p in candidates if os.path.exists(p)), None)

    if path is None:
        if missing_coordination == "error":
            raise FileNotFoundError(
                "Missing continuous coordination file. Tried:\n"
                + "\n".join(candidates)
            )

        # Conservative fallback: one timestep of zeros.
        return np.zeros((1, feature_dim), dtype=np.float32)

    arr = np.load(path).astype(np.float32, copy=False)

    # Allow saved shape (F,) but normalize to (1, F)
    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2:
        raise ValueError(
            f"Continuous coordination file must have shape (T, F), "
            f"got {arr.shape} at {path}"
        )

    if arr.shape[-1] != feature_dim:
        raise ValueError(
            f"Continuous coordination feature dim mismatch at {path}: "
            f"expected F={feature_dim}, got shape={arr.shape}"
        )

    if target_t > 0:
        arr = _fix_seq_len(arr, target_t=target_t, feature_dim=feature_dim)

    return arr