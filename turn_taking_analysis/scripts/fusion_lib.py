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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Constants
# =============================================================================

NUM_CLASSES = 3                                # HOLD, YIELD, BACKCHANNEL
CLASS_NAMES = ("HOLD", "YIELD", "BACKCHANNEL")
TRAINING_CLASSES = frozenset({0, 1, 2})
# Classes 3–5 (INTERRUPT, FAILED, LAPSE) are stored in the label files for
# exclusion-stat reporting but filtered out of training/eval by default.

WINDOW_S = 2.0                                 # input-context duration
STRIDE_S = 0.5                                 # spacing between adjacent samples
# Both match splice_wavs.py and build_labeled_windows_from_manifest.py.
# A spliced file 'NNNN.NN-NNNN.NN_{fid}.npy' has end - start = WINDOW_S.

DEFAULT_TAU_GRID_MS: tuple[int, ...] = (100, 200, 400, 500, 800, 1600)
DEFAULT_TRAIN_TAU_MS = 400

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
        for mod_dir, role in self._resolved:
            fid = speaker_fid if role == "speaker" else listener_fid
            fname = splice_filename_for(start_s, end_s, fid)
            path = os.path.join(mod_dir, fid, fname)
            arr = np.load(path).astype(np.float32, copy=False)
            tensors.append(torch.from_numpy(arr))
        return tensors, s["label"]

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
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device,
):
    """Standard train/eval loop. optimizer=None → eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total = 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for streams, labels in loader:
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

    try:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            va_loss, va_p, va_y = run_epoch(
                model, val_loader, criterion, None, device,
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
                # Copy to CPU so we don't pin GPU memory across epochs
                # for a model that might get reloaded on a different
                # device later (e.g. sweep_tau rebuilding on MPS from
                # a CPU-saved state_dict).
                best_state = {
                    k: v.detach().clone().cpu()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                marker = " *" if improved else "  "
                print(f"  epoch {epoch:>2}/{epochs}  "
                      f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                      f"val_macroF1={va_f1:.4f}{marker} "
                      f"(no_improve={no_improve}/{patience}, "
                      f"{history[-1]['elapsed_s']:.1f}s)")

            if no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} "
                          f"(best val macroF1={best_f1:.4f})")
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
) -> nn.Module:
    """Dispatch to the right model class based on fusion family + stream count.

    Hyperparams:
      hidden_size, dropout  — applied to all fusion families.
      num_layers_early      — GRU stack depth for EarlyFusionGRU only;
                              unimodal and neural-concat use single-layer
                              GRUs to match the practicum's configuration.
    """
    if fusion == "unimodal":
        if len(stream_dims) != 1:
            raise ValueError(
                f"unimodal fusion needs exactly 1 stream; got {len(stream_dims)}"
            )
        return GRUClassifier(
            stream_dims[0], hidden_size, num_classes, dropout=dropout,
        )
    if fusion == "early":
        return EarlyFusionGRU(
            sum(stream_dims), hidden_size, num_classes,
            num_layers=num_layers_early, dropout=dropout,
        )
    if fusion == "neural_concat":
        return NeuralConcatFusion(
            stream_dims, hidden_size, num_classes, dropout=dropout,
        )
    raise ValueError(
        f"unknown fusion family {fusion!r}; supported: "
        f"unimodal, early, neural_concat"
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
) -> dict:
    """Train one experiment end-to-end.

    Args:
      streams: list of {'modality': <name>, 'role': 'speaker'|'listener'}.
      fusion: 'unimodal' | 'early' | 'neural_concat'.
      labels_path: path to labels_tau_XXXX.json (picks training τ).
      hidden_size, dropout: applied to every fusion family.
      num_layers_early: GRU stack depth for EarlyFusionGRU only
        (unimodal and neural-concat use single-layer GRUs per practicum).
      epochs, batch_size, learning_rate, patience: standard training
        hyperparameters. Defaults match the practicum (30 / 32 / 1e-3 / 4).
      max_samples_per_split: if set, truncate each split to at most this
        many samples after a seed-deterministic shuffle (dry-run mode).

    Returns:
      dict with keys: name, config, samples_summary, train_history,
        test_eval, model_state (best-val weights), stream_dims, device.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_t = resolve_device(device) if isinstance(device, str) else device

    labels = load_labels(labels_path)
    samples = enumerate_samples(manifest_rows, labels)
    by_split = split_samples(samples)

    if max_samples_per_split is not None:
        # Shuffle each split BEFORE truncating so the cap pulls a random
        # subset rather than the first-N samples in label-iteration
        # order. Without the shuffle, the first N samples all come from
        # the first 1–2 interactions in the labels JSON — wildly non-
        # representative class distribution and useless for dry-run
        # validation. Deterministic given `seed`.
        rng = random.Random(seed)
        for split in list(by_split.keys()):
            rng.shuffle(by_split[split])
            by_split[split] = by_split[split][:max_samples_per_split]
        samples = [s for split_samples_list in by_split.values()
                   for s in split_samples_list]

    stream_dims = [modality_registry[s["modality"]]["feature_dim"]
                   for s in streams]

    if verbose:
        print(f"=== Experiment: {name} ===")
        print(f"  streams:   {streams}")
        print(f"  fusion:    {fusion}")
        print(f"  device:    {device_t}")
        summary = summarize_samples(samples)
        print(f"  samples:   total={summary['total']}  "
              f"per_split={summary['per_split']}")
        for sp, dist in summary["per_split_class"].items():
            print(f"             {sp:5s} class-dist: {dist}")

    train_loader = make_dataloader(
        by_split.get("train", []), streams, modality_registry,
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = make_dataloader(
        by_split.get("val", []), streams, modality_registry,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = make_dataloader(
        by_split.get("test", []), streams, modality_registry,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    model = build_model(
        fusion, stream_dims,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers_early=num_layers_early,
    ).to(device_t)
    class_weights = compute_class_weights(by_split.get("train", [])).to(device_t)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model:     {type(model).__name__}  ({n_params:,} params)")
        print(f"  class_w:   {class_weights.cpu().tolist()}")
        print(f"  epochs:    up to {epochs} (patience={patience})")
        print(f"  dropout:   {dropout}  num_layers_early: {num_layers_early}")
        print()

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=epochs, patience=patience, device=device_t, verbose=verbose,
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
        loader = make_dataloader(
            test_samples, streams, modality_registry,
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
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


