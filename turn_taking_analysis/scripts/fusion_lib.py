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
        coordination_feature_dim: int = COORDINATION_FEATURE_DIM,
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
    
# Cross-attention fusion

class _SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added at the input projection step.

    At the input rather than inside any attention block so the same
    positional signal covers both self-attention (in SelfCrossAttentionFusion)
    and cross-attention. No learned parameters.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Buffer (not parameter): moves with .to(device), not trained
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, T: int) -> torch.Tensor:
        """Return (1, T, d_model); broadcasts across batch when added."""
        return self.pe[:, :T, :]


class _CrossAttentionBlock(nn.Module):
    """Bidirectional cross-modality attention block.

    Each stream attends to the other. Pre-norm + residual. No FFN inside
    the block (FFN doubles param count and we are extremely data-poor at
    ~492 samples — easy to add later if the model is underfitting).

    Both directions consume the pre-norm versions of both inputs, so the
    cross-attention is parallel/symmetric — avoids an arbitrary "which
    modality updates first" ordering choice.

    nn.MultiheadAttention handles unequal Q vs K/V sequence lengths
    natively, so visual (e.g. 60 timesteps at 30Hz) and acoustic (e.g.
    200 timesteps at CPC's 100Hz) flow through without resampling.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        # a attends to b: query = a, key/value = b
        self.attn_a2b = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        # b attends to a: query = b, key/value = a
        self.attn_b2a = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a_n = self.norm_a(a)
        b_n = self.norm_b(b)
        a_ca, _ = self.attn_a2b(query=a_n, key=b_n, value=b_n,
                                need_weights=False)
        b_ca, _ = self.attn_b2a(query=b_n, key=a_n, value=a_n,
                                need_weights=False)
        a_out = a + self.resid_drop(a_ca)
        b_out = b + self.resid_drop(b_ca)
        return a_out, b_out


class CrossAttentionFusion(nn.Module):
    """2-stream cross-attention fusion model (no self-attention).

    Streams are projected to a shared attention_dim, given sinusoidal
    positional encoding, cross-attended to one another, pooled over time,
    and concatenated for the FC head.

    Stream identity is positional. feat_dims[0] = visual, feat_dims[1] =
    acoustic. Order must match the experiment-config `streams` list and
    the order at forward time.

    This is the "cross-attention only" ablation. For the combined
    "self-attention then cross-attention" model use
    SelfCrossAttentionFusion.
    """

    def __init__(
        self,
        feat_dims: list[int],
        attention_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pooling: str = "mean",
        max_len: int = 512,
    ):
        super().__init__()
        if len(feat_dims) != 2:
            raise ValueError(
                f"CrossAttentionFusion expects exactly 2 streams "
                f"(visual + acoustic); got {len(feat_dims)}."
            )
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")
        self.feat_dims = list(feat_dims)
        self.attention_dim = int(attention_dim)
        self.pooling = pooling
        d = self.attention_dim
        # Per-stream input projection to shared attention_dim
        self.input_projs = nn.ModuleList([
            nn.Linear(feat_dims[0], d),
            nn.Linear(feat_dims[1], d),
        ])
        # Positional encoding shared across both streams
        self.pos_enc = _SinusoidalPositionalEncoding(d, max_len=max_len)
        # Single cross-attention block, bidirectional.
        self.cross_attn = _CrossAttentionBlock(
            d_model=d, n_heads=num_heads, dropout=dropout,
        )
        # FC head: same shape pattern as NeuralConcatFusion / SelfAttentionFusion
        self.fc1 = nn.Linear(d * 2, d)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d, num_classes)

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) -> (B, D)."""
        if self.pooling == "last":
            return z[:, -1, :]
        return z.mean(dim=1)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError(
                f"CrossAttentionFusion expects 2 streams; got "
                f"{len(xs) if isinstance(xs, (list, tuple)) else 'non-list'}"
            )
        a, b = xs  # (B, T_a, feat_a), (B, T_b, feat_b)
        # Same per-stream shape validation as SelfAttentionFusion does
        for x, expected_dim, idx in zip(xs, self.feat_dims, ("a", "b")):
            if x.ndim != 3:
                raise ValueError(
                    f"Stream {idx} must have shape (B, T, F); got {x.shape}"
                )
            if x.shape[-1] != expected_dim:
                raise ValueError(
                    f"Stream {idx} feature dim mismatch: expected "
                    f"F={expected_dim}, got shape={x.shape}"
                )
        # Project + add positional encoding (broadcasts over batch)
        a = self.input_projs[0](a) + self.pos_enc(a.size(1))
        b = self.input_projs[1](b) + self.pos_enc(b.size(1))
        # Bidirectional cross-attention
        a, b = self.cross_attn(a, b)
        # Pool, concat, classify
        h = torch.cat([self._pool(a), self._pool(b)], dim=-1)  # (B, 2 * d)
        return self.fc2(self.drop(self.relu(self.fc1(h))))


class SelfCrossAttentionFusion(nn.Module):
    """2-stream self-attention then cross-attention fusion model.

    Pipeline:
      1. Per-stream Linear projection -> attention_dim
      2. Add sinusoidal positional encoding
      3. Per-stream TransformerEncoder (self-attention) with num_layers blocks
      4. Bidirectional cross-attention between the two streams
      5. Mean/last pool over time, concat, FC head

    Serial composition of within-modality self-attention and across-modality cross-attention. 
    The self-attention portion mirrors SelfAttentionFusion (same TransformerEncoderLayer
    setup) so the two are directly comparable.

    Stream identity is positional: feat_dims[0] = visual, feat_dims[1] =
    acoustic.
    """

    def __init__(
        self,
        feat_dims: list[int],
        attention_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pooling: str = "mean",
        max_len: int = 512,
    ):
        super().__init__()
        if len(feat_dims) != 2:
            raise ValueError(
                f"SelfCrossAttentionFusion expects exactly 2 streams "
                f"(visual + acoustic); got {len(feat_dims)}."
            )
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={attention_dim} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.feat_dims = list(feat_dims)
        self.attention_dim = int(attention_dim)
        self.num_layers = int(num_layers)
        self.pooling = pooling
        d = self.attention_dim

        # Per-stream input projection
        self.input_projs = nn.ModuleList([
            nn.Linear(feat_dims[0], d),
            nn.Linear(feat_dims[1], d),
        ])
        self.pos_enc = _SinusoidalPositionalEncoding(d, max_len=max_len)

        # Per-stream self-attention (TransformerEncoder)
        self.self_attn = nn.ModuleList()
        for _ in range(2):
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=num_heads,
                dim_feedforward=d * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.self_attn.append(
                nn.TransformerEncoder(encoder_layer=layer,
                                      num_layers=self.num_layers)
            )

        # Cross-attention block
        self.cross_attn = _CrossAttentionBlock(
            d_model=d, n_heads=num_heads, dropout=dropout,
        )

        # FC head
        self.fc1 = nn.Linear(d * 2, d)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d, num_classes)

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return z[:, -1, :]
        return z.mean(dim=1)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError(
                f"SelfCrossAttentionFusion expects 2 streams; got "
                f"{len(xs) if isinstance(xs, (list, tuple)) else 'non-list'}"
            )
        for x, expected_dim, idx in zip(xs, self.feat_dims, ("a", "b")):
            if x.ndim != 3:
                raise ValueError(
                    f"Stream {idx} must have shape (B, T, F); got {x.shape}"
                )
            if x.shape[-1] != expected_dim:
                raise ValueError(
                    f"Stream {idx} feature dim mismatch: expected "
                    f"F={expected_dim}, got shape={x.shape}"
                )

        a, b = xs
        # Project + positional encoding.
        a = self.input_projs[0](a) + self.pos_enc(a.size(1))
        b = self.input_projs[1](b) + self.pos_enc(b.size(1))
        # Self-attention per stream.
        a = self.self_attn[0](a)
        b = self.self_attn[1](b)
        # Cross-attention between streams.
        a, b = self.cross_attn(a, b)
        # Pool, concat, classify.
        h = torch.cat([self._pool(a), self._pool(b)], dim=-1)
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

    if fusion in {"cross_attention", "cross_attn"}:
        # 2-stream cross-attention only (no self-attention)
        head_count = (
            attention_heads[0] if isinstance(attention_heads, list)
            else attention_heads
        )
        return CrossAttentionFusion(
            feat_dims=stream_dims,
            attention_dim=attention_dim or hidden_size,
            num_heads=head_count,
            num_classes=num_classes,
            dropout=dropout,
            pooling=attention_pooling,
        )

    if fusion in {"self_cross_attention", "self_cross_attn"}:
        # 2-stream self-attention then cross-attention (combined)
        head_count = (
            attention_heads[0] if isinstance(attention_heads, list)
            else attention_heads
        )
        return SelfCrossAttentionFusion(
            feat_dims=stream_dims,
            attention_dim=attention_dim or hidden_size,
            num_heads=head_count,
            num_layers=attention_layers,
            num_classes=num_classes,
            dropout=dropout,
            pooling=attention_pooling,
        )

    raise ValueError(
        f"unknown fusion family {fusion!r}; supported: "
        f"unimodal, early, neural_concat, self_attention, self_attention, "
        f"cross_attention, self_cross_attention"
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
    Train one non-coordination non-coordination experiment end-to-end.

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
                    f""
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
        print(f"  streams:       {streams}")
        print(f"  fusion:        {fusion}")
        print(f"  stream_dims: {stream_dims}")
        print(f"  device:        {device_t}")
        summary = summarize_samples(samples)
        print(
            f"  samples:     total={summary['total']} "
            f"per_split={summary['per_split']}"
        )
        print(
            f"  samples:     total={summary['total']} "
            f"per_split={summary['per_split']}"
        )
        for sp, dist in summary["per_split_class"].items():
            print(f"                 {sp:5s} class-dist: {dist}")

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
        if fusion in {"self_attention", "attention",
                      "cross_attention", "cross_attn",
                      "self_cross_attention", "self_cross_attn"}:
            print(
                f"attention:   dim={attention_dim or hidden_size}, "
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
        if fusion in {"self_attention", "attention",
                      "cross_attention", "cross_attn",
                      "self_cross_attention", "self_cross_attn"}:
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

    Per-τ test cohort: each τ independently enumerates its own class-{0,1,2}
    test set, then `filter_samples_with_existing_files` drops any sample
    whose required feature files don't exist on disk (the filter now covers
    both spliced and continuous-coord streams).

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
                f"τ={tau_ms:>4}ms  n={len(test_samples):>5}  "
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

    Per-τ behavior: each τ independently enumerates its own class-{0,1,2}
    test set, then `filter_samples_with_existing_files` drops any sample
    whose required feature files don't exist on disk. The filter now
    covers continuous-coord streams, so a sample whose τ=400 class was
    {3,4,5} (no WCC file written by the extractor) is dropped before it
    can reach the loader — preventing the (1, F) zero-fallback vs real
    (T, F) torch.stack mismatch that previously crashed CSA τ-sweeps.

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
    """Drop samples whose required feature files don't exist on disk.

    Per-stream behavior:
      * Spliced (`kind="spliced"`): require the spliced .npy (or .json
        fallback for coord-aware loaders) to exist at the expected path.
        Missing → drop the sample.
      * Coordination summary (`kind="coordination", mode="summary"`): no
        per-window file (the lookup is CSV-keyed). Missing keys under
        `missing_coordination='zeros'` return `(1, F)` zeros which
        shape-matches a real summary row, so `torch.stack` succeeds. Skip
        — no drop needed.
      * Coordination continuous (`kind="coordination", mode="continuous"`):
        per-window WCC file. Missing under `'zeros'` returns `(1, F)`
        zeros which DOES NOT shape-match real `(T_wcc, F_wcc)` →
        `torch.stack` crashes. Missing under `'error'` raises
        `FileNotFoundError`. **Drop the sample so it never reaches the
        loader.**
    """
    kept, dropped = [], []

    for s in samples:
        ok = True
        missing = []

        for stream in streams:
            mod = stream["modality"]
            cfg = modality_registry[mod]
            kind = cfg.get("kind", "spliced")

            if kind == "coordination":
                mode = cfg.get("coordination_mode", "summary")
                if mode == "summary":
                    # No per-window file; CSV-keyed; zero fallback is shape-safe.
                    continue
                if mode == "continuous":
                    candidates = _coordination_continuous_candidates(cfg["dir"], s)
                    if not any(os.path.exists(p) for p in candidates):
                        ok = False
                        # Record the first candidate as the representative
                        # missing path (mirrors the loader's first-existing-
                        # path resolution).
                        missing.append(candidates[0] if candidates else "")
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


# =============================================================================
# PREFLIGHT — Idea 1 + Idea 2 combined
#
# Goal: catch torch.stack shape-divergence errors (e.g. continuous-coord zero-
# fallback (1, F) mixed with real (T, F) WCC arrays in the same batch) AND
# config / model-build errors BEFORE the experiment runner is ever called.
#
# Idea 1 — header-only audit. For every sample × stream pair, resolve the
# file path the dataset would resolve and read just the .npy header (no
# tensor deserialization). Tabulate shapes per stream; any divergence is a
# stacking hazard. Fast: ~microseconds per file, single-minute total for
# tens of thousands of samples.
#
# Idea 2 — static config validation + synthetic CPU forward. Validate every
# experiment's stream config + fusion family + attention dims. Then build the
# model and run a single forward on synthetic CPU tensors of registry-declared
# shape. Catches model-build issues (attention_dim % heads, GRU input_size
# mismatch, early-fusion T mismatch) without touching disk.
#
# Behavior: does NOT raise. Continues across all errors. Returns a structured
# report with provenance (interaction_id, speaker/listener file_id +
# participant, split, window timestamps, tried paths, observed/expected shape)
# for every error.
# =============================================================================

PREFLIGHT_FUSION_FAMILIES = frozenset(
    {"unimodal", "early", "neural_concat", "self_attention", "attention"}
)

# Error kinds with NO runtime safety net — they cause an actual crash
# (RuntimeError, ValueError, KeyError, FileNotFoundError, etc.) at training
# or evaluation time. The preflight FAIL verdict gates on the presence of any
# of these.
HARD_ERROR_KINDS = frozenset({
    "feature_dim_mismatch",      # _fix_seq_len raises (or downstream GRU input_size)
    "header_read_failed",        # np.load raises
    "summary_csv_key_missing",   # _load_summary_coordination raises KeyError ('error')
    "shape_divergence",          # _load_summary_coordination raises ValueError
    "audit_internal_error",      # preflight bug; rest of report can't be trusted
})

# Error kinds with a documented runtime safety net that silently handles the
# divergence. They surface in the per-experiment report under `soft_warnings`
# for visibility, but do NOT gate the FAIL verdict.
SOFT_ERROR_KINDS = frozenset({
    "T_mismatch",                # _fix_seq_len coerces leading dim to target_t
    "missing_file",              # filter_samples_with_existing_files drops it
})


def _is_hard_error(err: dict) -> bool:
    """Classify an error record as hard (gates FAIL) or soft (warning only).

    Runtime safety nets — exhaustive list of what's silently handled:

      * `_fix_seq_len` (`fusion_lib.py:_fix_seq_len`) — for any stream with
        `frame_rate_hz > 0`, the dataset trims/zero-pads the leading dim to
        `target_t = round(frame_rate_hz * WINDOW_S)` before stacking. So any
        `T_mismatch` recorded by the audit is silently coerced at runtime.

      * `filter_samples_with_existing_files` (`fusion_lib.py:filter_samples_
        with_existing_files`) — called by `run_experiment`,
        `run_experiment_coordination`, `sweep_tau`, and
        `sweep_tau_coordination_safe` before the per-batch loader spins up.
        Samples with a missing feature file (spliced OR continuous coord)
        are dropped entirely. Summary coord has no per-window file (the
        lookup is CSV-keyed) and its zero fallback is shape-safe, so the
        filter skips summary coord — no drop needed there.

    Anything else — feature_dim mismatch, header parse failures, summary-coord
    CSV key missing under `'error'` policy, summary-coord shape divergence,
    audit-internal errors — has no safety net and is flagged HARD.
    """
    error_kind = err.get("error_kind", "")
    if error_kind in HARD_ERROR_KINDS:
        return True
    if error_kind in SOFT_ERROR_KINDS:
        return False
    # Unknown kinds: be conservative and treat as hard so a future addition
    # surfaces loudly.
    return True


def _participant_from_file_id(file_id: str) -> str:
    """Extract the trailing Pxxxx token from a file_id like 'V00_S0691_I00000482_P0500'."""
    if not file_id:
        return ""
    return file_id.split("_")[-1]


def _read_npy_header_shape(path: str) -> tuple[int, ...] | None:
    """Read just the .npy header to get the array shape.

    Returns the shape tuple on success, None if the file does not exist OR the
    header cannot be parsed (corrupt / truncated / unsupported version).

    No tensor deserialization — just the magic + header. Microseconds per file.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            elif hasattr(np.lib.format, "read_array_header_3_0") and version == (3, 0):
                shape, _, _ = np.lib.format.read_array_header_3_0(f)
            else:
                return None
        return tuple(shape)
    except Exception:
        return None


def _read_feature_file_shape(path: str) -> tuple[int, ...] | None:
    """Read shape of either a .npy or .json feature file.

    Mirrors `_load_feature_file`'s dual-format support so the audit picks up the
    same files the dataset would. JSON parsing is the full json.load, not a
    cheap header — if the dataset later loads JSON it pays the same cost. .npy
    path uses the cheap header read.

    Returns None if file does not exist or shape cannot be determined.
    """
    if path.endswith(".json"):
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "features" in data:
                    data = data["features"]
                elif "embeddings" in data:
                    data = data["embeddings"]
                else:
                    return None
            arr = np.asarray(data)
            return tuple(arr.shape)
        except Exception:
            return None
    return _read_npy_header_shape(path)


def _resolve_stream_paths(
    sample: dict,
    stream_cfg: dict,
    modality_registry: dict[str, dict],
) -> list[str]:
    """Return the candidate file paths the dataset would resolve for this
    (sample, stream) pair.

    - Spliced: 2 candidate paths (.npy, .json) — mirrors `_load_feature_file`'s
      dual-format support used by `TurnTakingDatasetCoordination`. The
      original `TurnTakingDataset` only tries .npy, but auditing both is safe
      (audit picks the first existing path).
    - Continuous coord: 3 candidate paths (matches the loader's fallback chain).
    - Summary coord: empty list (lookup is CSV-keyed, not path-based).
    """
    cfg = modality_registry[stream_cfg["modality"]]
    kind = cfg.get("kind", "spliced")
    role = stream_cfg["role"]

    if kind == "coordination":
        mode = cfg.get("coordination_mode", "summary")
        if mode == "continuous":
            return _coordination_continuous_candidates(cfg["dir"], sample)
        return []  # summary coord — no path to resolve

    fid = sample["speaker_file_id"] if role == "speaker" else sample["listener_file_id"]
    fname = splice_filename_for(sample["start_s"], sample["end_s"], fid)
    base_no_ext = os.path.join(cfg["dir"], fid, os.path.splitext(fname)[0])
    return [base_no_ext + ".npy", base_no_ext + ".json"]


def _make_error_record(
    *,
    experiment_name: str,
    ablation_kind: str,
    stream_idx: int,
    stream_cfg: dict,
    sample: dict,
    error_kind: str,
    error_message: str,
    tried_paths: list[str] | None = None,
    observed_shape: tuple | None = None,
    expected_shape: tuple | None = None,
) -> dict:
    """Build a provenance-rich error record for a single (sample, stream) failure.

    Every error carries enough context to grep the offending file directly:
    interaction_id, speaker/listener file_id + participant, split, window
    timestamps, the candidate paths the loader would have tried, and the
    observed vs. expected shape (where applicable).
    """
    speaker_fid = sample.get("speaker_file_id", "")
    listener_fid = sample.get("listener_file_id", "")
    return {
        "experiment_name": experiment_name,
        "ablation_kind": ablation_kind,
        "stream_idx": stream_idx,
        "stream_modality": stream_cfg.get("modality", ""),
        "stream_role": stream_cfg.get("role", ""),
        "sample_basename": sample.get("basename", ""),
        "interaction_id": sample.get("interaction_id", ""),
        "speaker_file_id": speaker_fid,
        "listener_file_id": listener_fid,
        "speaker_participant": _participant_from_file_id(speaker_fid),
        "listener_participant": _participant_from_file_id(listener_fid),
        "split": sample.get("split", ""),
        "window_start_s": float(sample.get("start_s", 0.0)),
        "window_end_s": float(sample.get("end_s", 0.0)),
        "tried_paths": list(tried_paths) if tried_paths else [],
        "observed_shape": tuple(observed_shape) if observed_shape is not None else None,
        "expected_shape": tuple(expected_shape) if expected_shape is not None else None,
        "error_kind": error_kind,
        "error_message": error_message,
    }


def _validate_experiment_config(
    exp_cfg: dict,
    modality_registry: dict[str, dict],
    *,
    coordination_csv_path: str | None,
    coordination_lookup_provided: bool = False,
    attention_dim: int | None,
    attention_heads: "int | list[int]" = 4,
    hidden_size: int,
) -> list[str]:
    """Pure-config validation. No data, no model build. Returns list of error
    strings (empty list = pass). Continues past every error; never raises.

    `attention_heads` is the global default used when an experiment has no
    `attention_heads_by_modality` and no `attention_heads` of its own — must
    match the value `_synthetic_forward` would receive at the same call site,
    so the divisibility check is consistent with what would actually run.

    `coordination_lookup_provided` should be True iff the caller (typically
    `preflight_experiments`) was given a non-None `coordination_lookup` arg
    — i.e. a pre-loaded summary-coord lookup is available regardless of
    whether `coordination_csv_path` is set. Used to suppress the false-
    positive "csv path not set" error when the lookup arrives by the
    alternative route added in Fix 6.
    """
    errors: list[str] = []
    streams = exp_cfg.get("streams", [])
    fusion = exp_cfg.get("fusion", "")

    if fusion not in PREFLIGHT_FUSION_FAMILIES:
        errors.append(
            f"unknown fusion family {fusion!r}; "
            f"supported: {sorted(PREFLIGHT_FUSION_FAMILIES)}"
        )

    if not streams:
        errors.append("experiment has no streams")
        return errors

    has_summary_coord = False
    has_coord_any = False
    spliced_target_Ts: list[int] = []

    for i, s in enumerate(streams):
        mod = s.get("modality")
        role = s.get("role")
        if mod not in modality_registry:
            errors.append(f"stream[{i}] references unknown modality {mod!r}")
            continue
        if role not in {"speaker", "listener"}:
            errors.append(f"stream[{i}] has invalid role {role!r}")
            continue
        cfg = modality_registry[mod]
        kind = cfg.get("kind", "spliced")
        if kind == "coordination":
            has_coord_any = True
            mode = cfg.get("coordination_mode", "summary")
            if role != "speaker":
                errors.append(
                    f"stream[{i}] coordination modality {mod!r} requires "
                    f"role='speaker'; got {role!r}"
                )
            if mode == "summary":
                has_summary_coord = True
                expected_dim = len(COORDINATION_FEATURE_COLUMNS)
                if int(cfg.get("feature_dim", -1)) != expected_dim:
                    errors.append(
                        f"stream[{i}] summary coord {mod!r} feature_dim="
                        f"{cfg.get('feature_dim')} != "
                        f"len(COORDINATION_FEATURE_COLUMNS)={expected_dim}"
                    )
            elif mode == "continuous":
                d = cfg.get("dir")
                if not d or not os.path.isdir(d):
                    errors.append(
                        f"stream[{i}] continuous coord {mod!r} dir does not exist: {d!r}"
                    )
            else:
                errors.append(
                    f"stream[{i}] coord modality {mod!r} has invalid "
                    f"coordination_mode={mode!r}"
                )
        else:
            target_T = int(round(float(cfg.get("frame_rate_hz", 0.0)) * WINDOW_S))
            if target_T > 0:
                spliced_target_Ts.append(target_T)

    # Fusion-family-specific checks
    if fusion == "unimodal" and len(streams) != 1:
        errors.append(f"unimodal requires exactly 1 stream; got {len(streams)}")

    if fusion == "neural_concat" and len(streams) < 2:
        errors.append(f"neural_concat requires >=2 streams; got {len(streams)}")

    if fusion == "early":
        if has_coord_any:
            errors.append(
                "early fusion is not supported with coordination streams "
                "(summary T=1; continuous T differs from CPC/OpenFace). "
                "Use neural_concat or self_attention."
            )
        if len(set(spliced_target_Ts)) > 1:
            errors.append(
                f"early fusion requires identical T across spliced streams; "
                f"got {sorted(set(spliced_target_Ts))}"
            )

    if fusion in {"self_attention", "attention"}:
        eff_dim = attention_dim if attention_dim is not None else hidden_size
        # Resolution order mirrors the synthetic forward / runner:
        #   per-experiment attention_heads_by_modality -> per-experiment
        #   attention_heads -> global attention_heads parameter -> 4.
        heads_cfg = exp_cfg.get(
            "attention_heads_by_modality",
            exp_cfg.get("attention_heads", attention_heads),
        )
        if isinstance(heads_cfg, int):
            heads_list = [heads_cfg] * len(streams)
        else:
            heads_list = list(heads_cfg)
            if len(heads_list) != len(streams):
                errors.append(
                    f"attention_heads_by_modality length {len(heads_list)} "
                    f"!= number of streams {len(streams)}"
                )
        for i, h in enumerate(heads_list[: len(streams)]):
            try:
                h_int = int(h)
            except Exception:
                errors.append(f"attention_heads[{i}] is not an int: {h!r}")
                continue
            if h_int <= 0:
                errors.append(f"attention_heads[{i}]={h_int} must be positive")
                continue
            if eff_dim % h_int != 0:
                errors.append(
                    f"attention_dim={eff_dim} not divisible by num_heads={h_int} "
                    f"for stream[{i}]"
                )

    # Coord-runner requirement: at least one source for the summary lookup
    # (CSV path OR pre-loaded lookup passed in by the caller) must exist.
    if has_summary_coord and not coordination_csv_path and not coordination_lookup_provided:
        errors.append(
            "experiment has a summary coordination stream but neither "
            "coordination_csv_path nor coordination_lookup is set"
        )

    return errors


def _audit_stream_shapes(
    *,
    experiment_name: str,
    ablation_kind: str,
    samples: list[dict],
    streams: list[dict],
    modality_registry: dict[str, dict],
    coordination_lookup: dict | None,
    missing_coordination: str,
    header_cache: dict[str, tuple[str, tuple[int, ...] | None]] | None = None,
) -> dict:
    """Header-only shape audit over every (sample, stream) pair.

    For every stream:
      - resolve the path(s) the dataset would resolve
      - find the first existing path; read only the .npy header
      - tabulate observed shapes; record a provenance-rich error record per
        sample whose shape diverges, whose file is missing, or whose header
        cannot be read

    Returns:
        {
          "stream_summaries": [  # one entry per stream, in stream-config order
            {
              "stream_idx", "modality", "role", "kind",
              "expected_T", "expected_F",
              "observed_shapes":   {shape_tuple: count, ...},
              "missing_count":     int,
              "shape_mismatch_count": int,
              "stack_hazard":      bool,
            }, ...
          ],
          "errors": [error_record, ...]
        }
    """
    if header_cache is None:
        header_cache = {}
    coordination_lookup = coordination_lookup or {}

    stream_summaries: list[dict] = []
    errors: list[dict] = []

    for stream_idx, s in enumerate(streams):
        cfg = modality_registry.get(s["modality"])
        if cfg is None:
            continue  # caught by config validation; skip silently here

        kind = cfg.get("kind", "spliced")
        mode = cfg.get("coordination_mode", None)
        feature_dim = int(cfg["feature_dim"])
        frame_rate_hz = float(cfg.get("frame_rate_hz", 0.0))
        # `expected_T_from_rate` is the leading dim the loader will coerce to
        # via `_fix_seq_len` whenever frame_rate_hz > 0 (applies to both spliced
        # and continuous-coord streams). Used for both T_mismatch reporting
        # and the post-coerce effective-shape calculation in stack_hazard.
        expected_T_from_rate = (
            int(round(frame_rate_hz * WINDOW_S)) if frame_rate_hz > 0 else None
        )

        observed_shapes: Counter = Counter()
        missing_count = 0
        shape_mismatch_count = 0

        for sample in samples:
            # ---- summary coord: CSV-keyed lookup, no path ----
            if kind == "coordination" and mode == "summary":
                speaker_fid = sample["speaker_file_id"]
                speaker_pid = _participant_from_file_id(speaker_fid)
                key = _coord_key(
                    sample["interaction_id"],
                    sample["start_s"],
                    sample["end_s"],
                    speaker_pid,
                )
                arr = coordination_lookup.get(key)
                if arr is None:
                    missing_count += 1
                    if missing_coordination == "error":
                        errors.append(_make_error_record(
                            experiment_name=experiment_name,
                            ablation_kind=ablation_kind,
                            stream_idx=stream_idx,
                            stream_cfg=s,
                                sample=sample,
                            error_kind="summary_csv_key_missing",
                            error_message=(
                                f"coordination CSV has no row for key={key}; "
                                f"missing_coordination='error' will raise at training."
                            ),
                            expected_shape=(1, feature_dim),
                        ))
                    # Under 'zeros' policy summary returns (1, F) — same as real,
                    # so no stack hazard. Record (1, F) so the per-stream summary
                    # reflects what would actually go into torch.stack.
                    observed_shapes[(1, feature_dim)] += 1
                    continue

                shape = tuple(arr.shape)
                observed_shapes[shape] += 1
                if shape != (1, feature_dim):
                    shape_mismatch_count += 1
                    errors.append(_make_error_record(
                        experiment_name=experiment_name,
                        ablation_kind=ablation_kind,
                        stream_idx=stream_idx,
                        stream_cfg=s,
                        sample=sample,
                        error_kind="shape_divergence",
                        error_message=(
                            f"summary coord shape {shape} != expected (1, {feature_dim})"
                        ),
                        observed_shape=shape,
                        expected_shape=(1, feature_dim),
                    ))
                continue

            # ---- file-backed streams: spliced OR continuous coord ----
            tried_paths = _resolve_stream_paths(sample, s, modality_registry)

            # Populate cache for any unseen path. Cache entry is
            #   ("present", shape) | ("present", None=corrupt) | ("missing", None)
            # Uses _read_feature_file_shape so .json fallbacks are handled the
            # same way the dataset's _load_feature_file would.
            for p in tried_paths:
                if p in header_cache:
                    continue
                if os.path.exists(p):
                    header_cache[p] = ("present", _read_feature_file_shape(p))
                else:
                    header_cache[p] = ("missing", None)

            # Pick the first existing path (mirror loader behavior).
            chosen: tuple[str, str, tuple[int, ...] | None] | None = None
            for p in tried_paths:
                status, sh = header_cache[p]
                if status == "present":
                    chosen = (p, status, sh)
                    break

            if chosen is None:
                # No candidate path exists.
                missing_count += 1
                if kind == "coordination" and missing_coordination == "zeros":
                    errors.append(_make_error_record(
                        experiment_name=experiment_name,
                        ablation_kind=ablation_kind,
                        stream_idx=stream_idx,
                        stream_cfg=s,
                        sample=sample,
                        error_kind="missing_file",
                        error_message=(
                            f"continuous coord file missing; missing_coordination="
                            f"'zeros' will substitute (1, {feature_dim}) and trigger "
                            f"a torch.stack mismatch against real-shaped windows in "
                            f"the same batch."
                        ),
                        tried_paths=tried_paths,
                        expected_shape=(expected_T_from_rate, feature_dim)
                            if expected_T_from_rate else None,
                    ))
                else:
                    errors.append(_make_error_record(
                        experiment_name=experiment_name,
                        ablation_kind=ablation_kind,
                        stream_idx=stream_idx,
                        stream_cfg=s,
                        sample=sample,
                        error_kind="missing_file",
                        error_message="feature file does not exist at any candidate path",
                        tried_paths=tried_paths,
                        expected_shape=(expected_T_from_rate, feature_dim)
                            if expected_T_from_rate else None,
                    ))
                continue

            chosen_path, _status, shape = chosen

            if shape is None:
                errors.append(_make_error_record(
                    experiment_name=experiment_name,
                    ablation_kind=ablation_kind,
                    stream_idx=stream_idx,
                    stream_cfg=s,
                    sample=sample,
                    error_kind="header_read_failed",
                    error_message=f"could not read .npy header at {chosen_path}",
                    tried_paths=tried_paths,
                ))
                continue

            observed_shapes[shape] += 1

            # Trailing-dim check: must match registry feature_dim.
            if shape and shape[-1] != feature_dim:
                shape_mismatch_count += 1
                errors.append(_make_error_record(
                    experiment_name=experiment_name,
                    ablation_kind=ablation_kind,
                    stream_idx=stream_idx,
                    stream_cfg=s,
                    sample=sample,
                    error_kind="feature_dim_mismatch",
                    error_message=(
                        f"trailing dim {shape[-1]} != registry feature_dim {feature_dim} "
                        f"(file: {chosen_path})"
                    ),
                    observed_shape=shape,
                    expected_shape=(None, feature_dim),
                ))
                continue

            # Leading-dim (T) check for spliced streams. The loader will
            # _fix_seq_len trim/pad, so this is non-fatal for stacking but is
            # recorded as divergence so the user sees it.
            if kind != "coordination" and expected_T_from_rate is not None:
                if len(shape) >= 2 and shape[0] != expected_T_from_rate:
                    shape_mismatch_count += 1
                    errors.append(_make_error_record(
                        experiment_name=experiment_name,
                        ablation_kind=ablation_kind,
                        stream_idx=stream_idx,
                        stream_cfg=s,
                        sample=sample,
                        error_kind="T_mismatch",
                        error_message=(
                            f"leading dim {shape[0]} != expected T={expected_T_from_rate} "
                            f"(loader will _fix_seq_len trim/pad to expected — recorded "
                            f"as non-fatal divergence)."
                        ),
                        observed_shape=shape,
                        expected_shape=(expected_T_from_rate, feature_dim),
                    ))

        # Stack-hazard verdict: compute the set of shapes that would actually
        # reach torch.stack at runtime, accounting for the loader's coercion
        # behavior. False positives must be avoided here because stack_hazard
        # is the headline verdict callers act on.
        #
        # Coercion rules mirroring fusion_lib's loaders:
        #   * Spliced + frame_rate_hz > 0: _fix_seq_len trims/pads leading dim
        #     to expected_T_from_rate. Effective shape is (expected_T_from_rate, F).
        #   * Continuous coord + frame_rate_hz > 0: same _fix_seq_len applies
        #     to REAL files. Missing files under 'zeros' policy short-circuit
        #     before _fix_seq_len and return raw (1, F).
        #   * Continuous coord + frame_rate_hz = 0: no coercion; raw shape used.
        #   * Summary coord: always (1, F) at runtime (real or zeros fallback).
        #   * F-mismatched samples crash _fix_seq_len before stacking, so they
        #     never contribute a shape to torch.stack — exclude them here so
        #     they don't double-count as both feature_dim_mismatch and
        #     stack_hazard.
        effective_shapes: set[tuple] = set()
        for raw_shape in observed_shapes:
            if not raw_shape or raw_shape[-1] != feature_dim:
                continue  # would crash before stacking; reported elsewhere
            if expected_T_from_rate is not None and len(raw_shape) >= 2:
                effective_shapes.add((expected_T_from_rate, raw_shape[-1]))
            else:
                effective_shapes.add(raw_shape)
        # Note: continuous-coord missing files under 'zeros' policy used to
        # be injected here as `(1, F)` to flag a stack hazard. That hazard
        # is now closed at the filter layer — `filter_samples_with_existing_
        # files` drops missing-coord samples before they reach the loader,
        # so the (1, F) zero fallback never hits torch.stack at runtime.
        # The per-stream `stack_hazard` flag now fires only on real shape
        # divergence within the surviving samples.
        stack_hazard = len(effective_shapes) > 1

        stream_summaries.append({
            "stream_idx": stream_idx,
            "modality": s.get("modality", ""),
            "role": s.get("role", ""),
            "kind": kind,
            "expected_T": expected_T_from_rate,
            "expected_F": feature_dim,
            "observed_shapes": {repr(k): v for k, v in observed_shapes.items()},
            "missing_count": missing_count,
            "shape_mismatch_count": shape_mismatch_count,
            "stack_hazard": stack_hazard,
        })

    return {"stream_summaries": stream_summaries, "errors": errors}


def _synthetic_forward(
    *,
    streams: list[dict],
    fusion: str,
    modality_registry: dict[str, dict],
    hidden_size: int,
    dropout: float,
    num_layers_early: int,
    attention_dim: int | None,
    attention_heads,
    attention_layers: int,
    attention_pooling: str,
    continuous_coord_T: int = 21,
) -> str | None:
    """Build the fusion model on CPU and run a single forward pass on synthetic
    tensors of registry-declared shape. Returns None on success, a short error
    message on failure.

    No DataLoader, no disk I/O. B=2 is enough to exercise per-batch broadcast.
    """
    try:
        stream_dims: list[int] = []
        synthetic_inputs: list[torch.Tensor] = []
        B = 2

        for s in streams:
            cfg = modality_registry[s["modality"]]
            kind = cfg.get("kind", "spliced")
            mode = cfg.get("coordination_mode", None)
            F = int(cfg["feature_dim"])
            stream_dims.append(F)

            if kind == "coordination" and mode == "summary":
                T = 1
            elif kind == "coordination" and mode == "continuous":
                T = continuous_coord_T
            else:
                fr = float(cfg.get("frame_rate_hz", 0.0))
                T = int(round(fr * WINDOW_S)) if fr > 0 else 1
                if T <= 0:
                    T = 1

            synthetic_inputs.append(torch.zeros(B, T, F, dtype=torch.float32))

        model = build_model(
            fusion,
            stream_dims,
            hidden_size=hidden_size,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            num_layers_early=num_layers_early,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            attention_pooling=attention_pooling,
        )
        model.eval()
        with torch.inference_mode():
            logits = model(synthetic_inputs)
        if tuple(logits.shape) != (B, NUM_CLASSES):
            return (
                f"model returned shape {tuple(logits.shape)}; "
                f"expected ({B}, {NUM_CLASSES})"
            )
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def preflight_experiments(
    *,
    ablation_kind: str,
    experiments: dict[str, dict],
    modality_registry: dict[str, dict],
    manifest_rows: list[dict],
    labels_path: str,
    coordination_csv_path: str | None = None,
    coordination_feature_cols: list[str] = COORDINATION_FEATURE_COLUMNS,
    coordination_lookup: dict | None = None,
    missing_coordination: str = "zeros",
    max_samples_per_split: int | None = None,
    seed: int = 42,
    hidden_size: int = 64,
    dropout: float = 0.3,
    num_layers_early: int = 3,
    attention_dim: int | None = None,
    attention_heads: "int | list[int]" = 4,
    attention_layers: int = 2,
    attention_pooling: str = "mean",
    fail_fast: bool = False,
    verbose: bool = True,
    header_cache: dict | None = None,
) -> dict:
    """
    Idea 1 + Idea 2 combined preflight for one ablation block.

    For each experiment in `experiments`:
      1) Static config validation (no data, no model build).
      2) Header-only shape audit over every sample × stream pair.
         Walks the runtime training universe — class-{0,1,2} samples at the
         training-τ label file. The runtime's `filter_samples_with_existing_
         files` (now covering both spliced and continuous-coord streams)
         drops missing-file samples before the loader is invoked at both
         training and τ-sweep time, so missing-file errors recorded by the
         audit are downgraded to "soft warnings" rather than FAILs.
      3) Synthetic CPU forward through the built model.

    FAIL criterion (after the per-experiment audit):
        passed = not config_errors
                 AND not hard_errors
                 AND not any_stream_stack_hazard
                 AND synthetic_forward_error is None

    Hard vs soft error classification (see `_is_hard_error` for the
    exhaustive table) — runtime safety nets that downgrade an error to
    "soft warning" rather than FAIL:
      * `T_mismatch` — `_fix_seq_len` trims/zero-pads the leading dim to
        `target_t` whenever `frame_rate_hz > 0`. Recorded as soft.
      * `missing_file` (any stream) — `filter_samples_with_existing_files`
        drops the offending sample before it reaches any loader call, in
        both training and τ-sweep paths. The filter covers spliced AND
        continuous-coord streams (it correctly skips summary coord, whose
        zero fallback shape-matches real). Recorded as soft.
      * Everything else (feature_dim mismatch, header read failure,
        summary-coord CSV key missing under `'error'`, summary-coord shape
        divergence, audit-internal error, any per-stream
        `stack_hazard=True` from real shape divergence) — no runtime
        safety net. Recorded as hard.

    Continues across ALL errors and reports them all. Each error record carries
    full provenance (interaction_id, speaker/listener file_id + participant,
    split, window timestamps, tried paths, observed/expected shape, error kind).

    The same `header_cache` (path -> ("present"|"missing", shape|None)) can be
    threaded through multiple `preflight_experiments` calls (one per ablation)
    so shared paths (e.g. CPC speaker streams referenced by all ablations) are
    only header-read once across the entire preflight.

    `coordination_lookup` accepts a pre-loaded lookup dict (output of
    `load_coordination_features`) so the CSV is parsed once across all
    ablations rather than once per call. If None and `coordination_csv_path`
    is set, the CSV is loaded internally.

    Returns:
        {
          "ablation_kind": str,
          "experiments": {
              exp_name: {
                  "status": "PASS" | "FAIL",
                  "config_errors":             [str, ...],
                  # Backward-compatible breakdown by error_kind (raw):
                  "shape_errors":              [error_record, ...],
                  "missing_file_errors":       [error_record, ...],
                  # Severity breakdown (new):
                  "hard_errors":               [error_record, ...],
                  "soft_warnings":             [error_record, ...],
                  "any_stream_stack_hazard":   bool,
                  "synthetic_forward_error":   str | None,
                  "stream_summaries":          [{...}, ...],
              },
              ...
          },
          "summary": {
              "total_experiments", "passed", "failed",
              "total_config_errors",
              "total_shape_errors", "total_missing_files",       # raw
              "total_hard_errors", "total_soft_warnings",         # severity
              "total_stack_hazards",
              "total_synthetic_forward_failures",
          },
        }
    """
    # Resolve sample list once; all experiments at the same ablation share it
    # (they all read the same training-τ labels file).
    #
    # Use the default class filter `{0,1,2}` — the audit's universe matches
    # the runtime's training universe. An earlier version of this preflight
    # passed `training_classes=VALID_LABEL_INTS` to expand the audit to the
    # full `{0..5}` label range; that expansion is no longer needed because
    # the runtime's `filter_samples_with_existing_files` now drops samples
    # with missing files (including continuous-coord) before the loader is
    # invoked, so any τ-sweep sample whose τ=400 class was {3,4,5} (and
    # therefore has no extracted coord file) is silently dropped at runtime
    # rather than crashing torch.stack.
    labels = load_labels(labels_path)
    samples = enumerate_samples(manifest_rows, labels)

    if max_samples_per_split is not None:
        rng = random.Random(seed)
        by_split = split_samples(samples)
        for split in list(by_split.keys()):
            rng.shuffle(by_split[split])
            by_split[split] = by_split[split][:max_samples_per_split]
        samples = [s for lst in by_split.values() for s in lst]

    # Capture whether the caller passed a lookup explicitly BEFORE we mutate
    # the local variable. This signal feeds the config validator so it doesn't
    # falsely flag "csv path not set" when the alternative route is in use.
    caller_provided_lookup = coordination_lookup is not None

    # Use the caller-provided coordination_lookup if given; otherwise load once
    # from CSV. Sharing across ablation calls saves the CSV-parse cost per
    # ablation (the lookup itself is read-only; safe to share).
    if coordination_lookup is None:
        coordination_lookup = {}
        if coordination_csv_path is not None and os.path.exists(coordination_csv_path):
            try:
                coordination_lookup = load_coordination_features(
                    coordination_csv_path,
                    feature_cols=coordination_feature_cols,
                )
            except Exception as e:
                if verbose:
                    print(f"[preflight] WARNING: failed to load coordination CSV: {e}")

    if header_cache is None:
        header_cache = {}

    per_exp_reports: dict[str, dict] = {}
    totals = {
        "total_experiments": 0,
        "passed": 0,
        "failed": 0,
        "total_config_errors": 0,
        # Raw breakdown by error_kind (backward-compatible, includes both
        # hard and soft):
        "total_shape_errors": 0,
        "total_missing_files": 0,
        # Severity breakdown:
        "total_hard_errors": 0,
        "total_soft_warnings": 0,
        "total_stack_hazards": 0,
        "total_synthetic_forward_failures": 0,
    }

    for exp_name, exp_cfg in experiments.items():
        totals["total_experiments"] += 1
        if verbose:
            print(f"[preflight] {ablation_kind}/{exp_name} ...")

        # 1) Config validation — never raises.
        config_errors = _validate_experiment_config(
            exp_cfg,
            modality_registry,
            coordination_csv_path=coordination_csv_path,
            coordination_lookup_provided=caller_provided_lookup,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            hidden_size=hidden_size,
        )

        streams = exp_cfg.get("streams", [])
        all_mods_known = all(s.get("modality") in modality_registry for s in streams)

        # 2) Shape audit — skipped only if streams reference unknown modalities
        # (would crash the audit). All other config errors still let the audit
        # run so the user sees data issues simultaneously.
        audit = {"stream_summaries": [], "errors": []}
        if streams and all_mods_known:
            try:
                audit = _audit_stream_shapes(
                    experiment_name=exp_name,
                    ablation_kind=ablation_kind,
                    samples=samples,
                    streams=streams,
                    modality_registry=modality_registry,
                    coordination_lookup=coordination_lookup,
                    missing_coordination=missing_coordination,
                    header_cache=header_cache,
                )
            except Exception as e:
                audit["errors"].append({
                    "experiment_name": exp_name,
                    "ablation_kind": ablation_kind,
                    "error_kind": "audit_internal_error",
                    "error_message": f"{type(e).__name__}: {e}",
                    "stream_idx": -1,
                    "stream_modality": "",
                    "stream_role": "",
                    "sample_basename": "",
                    "interaction_id": "",
                    "speaker_file_id": "",
                    "listener_file_id": "",
                    "speaker_participant": "",
                    "listener_participant": "",
                    "split": "",
                    "window_start_s": 0.0,
                    "window_end_s": 0.0,
                    "tried_paths": [],
                    "observed_shape": None,
                    "expected_shape": None,
                })

        # 3) Synthetic forward — only meaningful if the config is at least
        # minimally sane (known modalities, valid roles, fusion family known).
        synthetic_forward_error: str | None = None
        if streams and all_mods_known and exp_cfg.get("fusion") in PREFLIGHT_FUSION_FAMILIES:
            heads_for_exp = exp_cfg.get(
                "attention_heads_by_modality",
                exp_cfg.get("attention_heads", attention_heads),
            )
            synthetic_forward_error = _synthetic_forward(
                streams=streams,
                fusion=exp_cfg.get("fusion", ""),
                modality_registry=modality_registry,
                hidden_size=hidden_size,
                dropout=dropout,
                num_layers_early=num_layers_early,
                attention_dim=attention_dim,
                attention_heads=heads_for_exp,
                attention_layers=attention_layers,
                attention_pooling=attention_pooling,
            )

        # Backward-compatible breakdown by error_kind (raw) — preserves the
        # report shape callers may already be reading.
        shape_errors = [e for e in audit["errors"]
                        if e.get("error_kind") != "missing_file"]
        missing_file_errors = [e for e in audit["errors"]
                               if e.get("error_kind") == "missing_file"]

        # New breakdown by runtime severity. Hard errors gate FAIL; soft
        # warnings surface for visibility but indicate a runtime safety net
        # silently handles the divergence (see `_is_hard_error` for the
        # exhaustive classification table).
        hard_errors = [e for e in audit["errors"] if _is_hard_error(e)]
        soft_warnings = [e for e in audit["errors"] if not _is_hard_error(e)]

        # Stack hazard at any audited stream is itself a FAIL trigger even if
        # no individual error record fires (e.g. divergent observed shapes
        # across samples that the loader can't reconcile).
        any_stream_stack_hazard = any(
            ss.get("stack_hazard", False)
            for ss in audit["stream_summaries"]
        )

        passed = (
            not config_errors
            and not hard_errors
            and not any_stream_stack_hazard
            and synthetic_forward_error is None
        )

        per_exp_reports[exp_name] = {
            "status": "PASS" if passed else "FAIL",
            "config_errors": config_errors,
            # Backward-compat raw breakdown:
            "shape_errors": shape_errors,
            "missing_file_errors": missing_file_errors,
            # Severity breakdown:
            "hard_errors": hard_errors,
            "soft_warnings": soft_warnings,
            "any_stream_stack_hazard": any_stream_stack_hazard,
            "synthetic_forward_error": synthetic_forward_error,
            "stream_summaries": audit["stream_summaries"],
        }

        totals["passed"] += int(passed)
        totals["failed"] += int(not passed)
        totals["total_config_errors"] += len(config_errors)
        totals["total_shape_errors"] += len(shape_errors)
        totals["total_missing_files"] += len(missing_file_errors)
        totals["total_hard_errors"] += len(hard_errors)
        totals["total_soft_warnings"] += len(soft_warnings)
        totals["total_stack_hazards"] += int(any_stream_stack_hazard)
        totals["total_synthetic_forward_failures"] += int(
            synthetic_forward_error is not None
        )

        if verbose:
            verdict = "PASS" if passed else "FAIL"
            print(
                f"[preflight] {ablation_kind}/{exp_name}: {verdict} "
                f"(cfg_err={len(config_errors)}, "
                f"hard={len(hard_errors)}, soft={len(soft_warnings)}, "
                f"stack_hazard={'YES' if any_stream_stack_hazard else 'no'}, "
                f"synth_fwd={'ok' if synthetic_forward_error is None else 'FAIL'})"
            )

        if fail_fast and not passed:
            break

    return {
        "ablation_kind": ablation_kind,
        "experiments": per_exp_reports,
        "summary": totals,
    }



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