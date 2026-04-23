#!/usr/bin/env python3
"""
extract_cpc_from_manifest.py

Run Facebook Research CPC (Rivière et al. 2020, 256-dim @ 100 Hz) over
every spliced .wav produced by splice_wavs.py, saving features in a
directory tree that exactly mirrors the spliced-wav tree.

NOTE on filename: this script retains its historical name for branch
compatibility but no longer reads a manifest CSV. It walks the output
tree of splice_wavs.py instead.

!!! TROUBLESHOOTING: network / torch.hub errors — read this first !!!
---------------------------------------------------------------------
Symptoms you might see when the default (torch.hub) path fails:
  - "Cannot find callable CPC_default in hubconf"
  - "It looks like there is no internet connection..."
  - "SSL: CERTIFICATE_VERIFY_FAILED" on github.com or dl.fbaipublicfiles.com
  - "torch.hub.load(...) failed: ..."

Cause: your network is blocking GitHub's or Meta's CDN. Typical on SSL-
inspecting corporate / university WiFi (e.g. USC secure-wireless).

Fix — do the two downloads manually on any network that works, then pass
both paths to the script as CLI args:

  # 1. Clone the CPC_audio repo (the Python package source):
  git clone https://github.com/facebookresearch/CPC_audio /path/to/CPC_audio

  # 2. Download the pretrained Rivière 2020 checkpoint (~7 MB):
  curl -L -o /path/to/60k_epoch4-d0f474de.pt \\
      https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt

  # 3. Re-run this script with both paths passed via CLI:
  python extract_cpc_from_manifest.py \\
      --spliced-dir ... --output-dir ... \\
      --cpc-repo /path/to/CPC_audio \\
      --checkpoint /path/to/60k_epoch4-d0f474de.pt

Full rationale in the "CPC model loading" section below.

Input / output layout
---------------------
    --spliced-dir/
        V03_S1930_I00000105_P5055/                         <-- original wav stem
            0000.00-0002.00_V03_S1930_I00000105_P5055.wav  <-- spliced window
            0000.50-0002.50_V03_S1930_I00000105_P5055.wav
            ...
        V00_S0743_I00000483_P0885/
            ...

    --output-dir/                                          <-- exact mirror
        V03_S1930_I00000105_P5055/
            0000.00-0002.00_V03_S1930_I00000105_P5055.npy  <-- float32 features
            0000.00-0002.00_V03_S1930_I00000105_P5055.json <-- sidecar (provenance)
            ...

Each spliced wav produces one .npy (features, float32 at 100 Hz native
or pooled rate) plus one sibling .json sidecar. The .npy filenames
mirror the spliced .wav names 1:1; the .json is an orthogonal metadata
artifact that preserves mean_pool_file()'s post-hoc pooling workflow
(it reads feature_rate_hz from the sidecar).

Why per-window CPC?
-------------------
CPC is a strictly causal encoder: the feature at frame t is a function
of audio only up to t. If we ran CPC over an entire multi-minute
interaction, the aggregator's hidden state at t=4:55 would have absorbed
~5 minutes of context, while at t=0:05 it would have absorbed a few
seconds. That asymmetry confounds turn-taking modeling, where we want
every prediction conditioned on the same bounded window. Splicing the
audio first and running CPC independently per window gives every output
the same context budget.

Applying the same windowing across every modality (CPC, WavLM, SMPL-H,
OpenFace…) also lets us stack features without any resampling or
realignment — window boundaries are shared by construction.

Runtime (batched inference)
---------------------------
A torch DataLoader with --num-workers worker processes overlaps audio
load + resample on CPU with the CPC forward pass on GPU. Each batch is
(B, 1, 32000) with uniform T — splice_wavs.py's partial-trailing-window
skip guarantees every window is exactly --window-len seconds, so no
padding is needed. Writes go to a background ThreadPoolExecutor so disk
I/O doesn't block the next forward pass.

Defaults tuned for Apple-Silicon (MPS backend, unified memory):
  --batch-size     64     # kernel-launch amortization across samples
  --num-workers    8      # parallel wav load + 48→16 kHz resample
  --progress-every 500    # throttle per-task log lines for ~900 prints

Start/stop resilience
---------------------
Idempotent resume by default. For each spliced wav the script checks
whether a valid .npy + .json pair already exists (both non-zero bytes,
json parses); if so, the task is skipped. Writes go in order (.npy
first, then .json), so a mid-write interruption leaves the pair
invalid and gets re-run on the next invocation:

  - killed before .npy finishes    -> .json missing -> already_have
                                      returns False -> re-run
  - killed between .npy and .json  -> .json missing -> re-run
  - killed mid-.json-write         -> json.load fails -> re-run
  - both finished cleanly          -> skipped on rerun

The background writer's shutdown(wait=True) is in a `finally` block so a
Ctrl-C drains pending writes before the process exits (as many .npy +
.json pairs finish atomically as possible), preserving the invariant
above for those in-flight tasks.

Pass --overwrite to force re-extraction even when both files are present.

Why CPC?
--------
CPC is the canonical causal / non-bidirectional audio encoder used by the
VAP family (Ekstedt & Skantze 2022; Inoue et al. 2024; MM-VAP 2025). Its
aggregator is a strictly autoregressive RNN — unlike WavLM / HuBERT /
wav2vec 2.0, the feature at frame t is a function of audio only up to t.
For a prediction-horizon task the per-window slicing + CPC combination
gives features whose causality is guaranteed by construction, and a
clean ablation target against WavLM-base+ features.

Full architecture / citations: docs/audio_encoder_research.md entry 1.

What the script does
--------------------
For each spliced wav under --spliced-dir/<orig_stem>/:

  1. Parse the filename `NNNN.NN-NNNN.NN_<orig_stem>.wav` to recover
     the window's [start, end] seconds relative to the original wav.
  2. Load the wav via torchaudio and resample to 16 kHz (CPC's training
     sample rate — splice_wavs.py preserves the source sample rate, so
     if the source was 48 kHz the spliced wav is also 48 kHz). Load +
     resample happen in a DataLoader worker.
  3. Stack B samples into (B, 1, 32000). Shape uniformity is guaranteed
     by splice_wavs' partial-trailing-window skip.
  4. Run CPC forward on the batch. CPC returns (c_feature, z_feature,
     label) where
     - c_feature is the aggregator (AR RNN) output — the "context" stream
     - z_feature is the CNN-encoder-only output — the "encoded" stream
     Both are 256-dim @ 100 Hz. We keep whichever --feature-type selects.
  5. Per-sample: NaN/Inf guard, mean-pool every --pool-factor frames
     along time (default 1 = no pooling), hand to a background writer.
  6. Background writer saves --output-dir/<orig_stem>/<spliced_stem>.npy
     (float32, shape (T, 256) at the chosen frame rate) and
     <spliced_stem>.json sidecar with window timing and full CPC
     provenance. Writes .npy FIRST for the resume invariant.

Two-stage pooling workflow
--------------------------
Default: extract once at native 100 Hz. Later invoke
`mean_pool_file(file, save_to, pool_to=10.0)` on each .npy to produce
a separate pooled corpus — no CPC re-run required. The sidecar next to
each .npy records feature_rate_hz so mean_pool_file can validate that
the requested pool_to evenly divides the source rate.

CPC model loading
-----------------
Two equivalent loading paths.

(A) torch.hub (default, requires internet):

    torch.hub.load('facebookresearch/CPC_audio', 'CPC_audio',
                   pretrained=True, trust_repo=True)

    Returns a CPCModel instance. Under the hood, hubconf.CPC_audio calls
    torch.hub.load_state_dict_from_url to fetch the pretrained checkpoint
    from:
        https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt

    So this path makes TWO outbound connections: one to github.com (for
    the repo's hubconf.py) and one to dl.fbaipublicfiles.com (for the
    .pt). If either host is blocked, the load fails.

(B) Local repo + checkpoint (always works, no internet needed at run-time):

    python extract_cpc_from_manifest.py \\
        --cpc-repo /path/to/CPC_audio \\
        --checkpoint /path/to/60k_epoch4-d0f474de.pt ...

    The script sys.path-inserts the repo, torch.load's the checkpoint,
    and replicates hubconf.py's model-construction flow exactly. Use
    this when outbound GitHub access is unavailable.

Exit codes
----------
    0 — all tasks succeeded (or skipped-as-already-extracted).
    1 — hard setup error (missing --spliced-dir, missing --output-dir,
        missing --cpc-repo/--checkpoint pairing, model load failure).
    2 — one or more files failed extraction (see logs).
    3 — one or more input .wav files were missing on disk (non-fatal).

Usage
-----
    # Both dirs must already exist.
    mkdir -p ../subset/cpc_100hz

    python extract_cpc_from_manifest.py \\
        --spliced-dir ../subset/audio_sliced \\
        --output-dir  ../subset/cpc_100hz

    # Tune for your machine
    python extract_cpc_from_manifest.py \\
        --spliced-dir ../subset/audio_sliced \\
        --output-dir  ../subset/cpc_100hz \\
        --batch-size 64 --num-workers 8

    # Dry-run to enumerate what would be extracted
    python extract_cpc_from_manifest.py \\
        --spliced-dir ../subset/audio_sliced \\
        --output-dir  ../subset/cpc_100hz --dry-run

    # Force re-extraction
    python extract_cpc_from_manifest.py \\
        --spliced-dir ../subset/audio_sliced \\
        --output-dir  ../subset/cpc_100hz --overwrite

    # Pool to 10 Hz during extraction (10x smaller on disk)
    mkdir -p ../subset/cpc_10hz
    python extract_cpc_from_manifest.py \\
        --spliced-dir ../subset/audio_sliced \\
        --output-dir  ../subset/cpc_10hz --mean-pool

    # Use CNN-encoder output (z) instead of aggregator output (c)
    python extract_cpc_from_manifest.py \\
        --spliced-dir ../subset/audio_sliced \\
        --output-dir  ../subset/cpc_100hz --feature-type encoded
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


# -----------------------------------------------------------------------------
# Re-use helpers from csci535-project/scripts. Same bootstrap as
# extract_wavlm_from_manifest.py — resolve the sibling directory and insert
# it on sys.path so the extract_*_id helpers from the download script are
# importable.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]                 # .../csci535-project
SHARED_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (  # noqa: E402
    extract_interaction_id,
    extract_participant_id,
)


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
# Intentional: --spliced-dir and --output-dir have NO defaults. Writing CPC
# features is high-volume and the "right" destination depends on whether
# you're producing the 100 Hz native corpus, a 10 Hz pooled corpus, or a
# one-off debug extraction. Forcing the caller to name both directories
# prevents typo-driven reads from / writes to an unexpected location.

DEFAULT_HUB_REPO = "facebookresearch/CPC_audio"
DEFAULT_HUB_ENTRYPOINT = "CPC_audio"    # the real entrypoint name in hubconf.py;
                                        # it needs pretrained=True to actually load
                                        # the 60k_epoch4 weights (else it returns a
                                        # randomly-initialized model).

DEFAULT_FEATURE_TYPE = "context"   # 'context' = aggregator output c_t (causal RNN output)
                                   # 'encoded' = CNN encoder output z_t (local-in-time)
DEFAULT_POOL_FACTOR = 1            # 1 = no pooling; save native 100 Hz features.
                                   # Use --mean-pool (→10) to pool during extraction,
                                   # or leave default and invoke mean_pool_file() later.
DEFAULT_TARGET_RATE_HZ = 10.0      # Default target rate for mean_pool_file() and --mean-pool.

# Batched-inference defaults, tuned for Apple-Silicon M5 Max 32-core
# GPU / 36 GB unified memory. CUDA users can usually push batch-size
# much higher; CPU-only users should drop it to 1.
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 8
DEFAULT_WRITER_THREADS = 4
DEFAULT_PROGRESS_EVERY = 500

CPC_SAMPLE_RATE = 16_000           # CPC was trained on 16 kHz mono
CPC_SAMPLES_PER_FRAME = 160        # CPC's 5-layer CNN stride over 16 kHz → 100 Hz
CPC_FEATURE_DIM = 256              # Rivière 2020 baseline channel width


# =============================================================================
# Extraction-task descriptor
# =============================================================================

class ExtractTask:
    """One spliced wav to featurize: input path + destination paths + timing."""

    __slots__ = (
        "orig_stem", "interaction_id", "participant_id",
        "window_start_s", "window_end_s",
        "audio_path", "output_path", "sidecar_path",
    )

    def __init__(
        self,
        orig_stem: str,
        interaction_id: str,
        participant_id: str,
        window_start_s: float,
        window_end_s: float,
        audio_path: str,
        output_path: str,
        sidecar_path: str,
    ):
        self.orig_stem = orig_stem
        self.interaction_id = interaction_id
        self.participant_id = participant_id
        self.window_start_s = window_start_s
        self.window_end_s = window_end_s
        self.audio_path = audio_path
        self.output_path = output_path
        self.sidecar_path = sidecar_path

    def __repr__(self) -> str:
        return f"ExtractTask({Path(self.audio_path).name})"


def _parse_window_filename(stem: str):
    """Parse `NNNN.NN-NNNN.NN_<orig_stem>` from a spliced wav's stem.

    Returns (window_start_s, window_end_s, orig_stem) or None if the name
    does not match the splice_wavs.py format. Uses `partition` to split
    only on the FIRST underscore, since orig_stem itself contains '_'
    (e.g. `V03_S1930_I00000105_P5055`).
    """
    head, sep, orig = stem.partition("_")
    if not sep or not orig:
        return None
    start_str, dash, end_str = head.partition("-")
    if not dash:
        return None
    try:
        start_s = float(start_str)
        end_s = float(end_str)
    except ValueError:
        return None
    return start_s, end_s, orig


def build_tasks(spliced_dir: str, output_dir: str) -> list[ExtractTask]:
    """Walk spliced_dir/<orig_stem>/*.wav and build one ExtractTask per wav.

    The subdir name is the original wav stem (e.g.
    `V03_S1930_I00000105_P5055`); `extract_interaction_id` +
    `extract_participant_id` derive the IDs from it. Each spliced wav
    filename encodes its [start, end] seconds relative to the original
    (see splice_wavs.py's `_fmt_time`).

    Outputs mirror the input tree exactly: `output_dir/<orig_stem>/
    <spliced_stem>.npy` plus a sibling `<spliced_stem>.json` sidecar.
    """
    spliced_root = Path(spliced_dir)
    tasks: list[ExtractTask] = []

    for sub in sorted(spliced_root.iterdir()):
        if not sub.is_dir():
            continue
        orig_stem = sub.name

        try:
            interaction_id = extract_interaction_id(orig_stem)
            participant_id = extract_participant_id(orig_stem)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: subdir {orig_stem!r} does not match the expected "
                  f"file-id pattern ({e}); skipping.", file=sys.stderr)
            continue

        out_sub = Path(output_dir) / orig_stem

        for wav in sorted(sub.glob("*.wav")):
            parsed = _parse_window_filename(wav.stem)
            if parsed is None:
                print(f"  WARN: could not parse window times from "
                      f"{wav.name!r}; skipping.", file=sys.stderr)
                continue
            window_start_s, window_end_s, _orig = parsed

            tasks.append(ExtractTask(
                orig_stem=orig_stem,
                interaction_id=interaction_id,
                participant_id=participant_id,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                audio_path=str(wav),
                output_path=str(out_sub / f"{wav.stem}.npy"),
                sidecar_path=str(out_sub / f"{wav.stem}.json"),
            ))

    return tasks


# =============================================================================
# Model loading
# =============================================================================

def pick_device(requested: str) -> str:
    """Resolve --device. 'auto' picks cuda > mps > cpu."""
    if requested != "auto":
        return requested

    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cpc(
    hub_repo: str,
    hub_entrypoint: str,
    cpc_repo: str | None,
    checkpoint: str | None,
    device: str,
):
    """Load a pretrained CPC model.

    Two supported paths:

    (A) torch.hub (default). Calls
        torch.hub.load(hub_repo, hub_entrypoint)
        and takes the first element of the returned tuple. The facebook
        research repo returns (model, hiddenGar, hiddenEncoder) — where
        hiddenGar is the aggregator output dim (= dim of c_t = the
        "context" stream) and hiddenEncoder is the CNN-encoder output
        dim (= dim of z_t = the "encoded" stream). We log both and
        return the aggregator dim as the canonical feature_dim.

    (B) Local repo + checkpoint. If --cpc-repo and --checkpoint are both set,
        we sys.path-insert the repo, mimic hubconf.py's CPC_audio() flow:
        torch.load the checkpoint (expecting the release format
        {"config": ..., "weights": ...}), build the model via
        get_default_cpc_config + getEncoder + getAR + CPCModel, and
        load_state_dict from ckpt["weights"]. Use this when outbound
        GitHub access is unavailable (CI, airgapped hosts, or USC
        secure-wireless doing SSL inspection on the GitHub API).
        Note: this path does NOT use cpc.feature_loader.loadModel —
        that helper expects a training-output directory with
        checkpoint_logs.json + checkpoint_args.json sidecars, which the
        release .pt bundle does not ship.

    Returns the model in eval mode on `device`.
    """
    import torch  # noqa: PLC0415

    if cpc_repo and checkpoint:
        # Local path: sys.path insert the repo, import the model class +
        # its helpers, load the release-format checkpoint manually.
        #
        # Why not cpc.feature_loader.loadModel?  loadModel expects a
        # training-output *directory* containing:
        #   - <11-char-prefix><digits>.pt         (e.g. checkpoint_42.pt)
        #   - checkpoint_logs.json
        #   - checkpoint_args.json
        # The published release checkpoints (e.g. 60k_epoch4-d0f474de.pt)
        # don't ship with those sidecars — they're standalone dicts with
        # {"config": ..., "weights": ...} keys, designed for the hubconf
        # path. We replicate that path here so the same .pt file works
        # whether the user goes via torch.hub or --cpc-repo/--checkpoint.
        repo_path = Path(cpc_repo).resolve()
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"--cpc-repo {cpc_repo} is not a directory. Clone "
                f"{DEFAULT_HUB_REPO} first."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        print(f"  Loading CPC:       local repo {repo_path}")
        print(f"  Checkpoint:        {checkpoint}")

        import argparse as _argparse  # noqa: PLC0415
        try:
            # Same imports hubconf.py does, in the same order.
            from cpc.model import CPCModel as _CPCModel  # noqa: PLC0415
            from cpc.cpc_default_config import get_default_cpc_config  # noqa: PLC0415
            from cpc.feature_loader import getEncoder, getAR, loadArgs  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                f"Could not import cpc.* helpers from {repo_path}. Is "
                f"{repo_path} actually a clone of {DEFAULT_HUB_REPO}? ({e})"
            ) from e

        # Load the checkpoint dict. weights_only=False because the dict
        # contains a plain Python 'config' subdict (not a pure tensor
        # state-dict); newer torch emits a warning without the explicit flag.
        try:
            ckpt_data = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to torch.load {checkpoint}: {e}"
            ) from e

        if not isinstance(ckpt_data, dict) or "config" not in ckpt_data \
                or "weights" not in ckpt_data:
            raise RuntimeError(
                f"Checkpoint {checkpoint} is not the release-format CPC "
                f"checkpoint (expected a dict with 'config' and 'weights' "
                f"keys). Got: {type(ckpt_data).__name__}"
                + (f", keys={list(ckpt_data.keys())}"
                   if isinstance(ckpt_data, dict) else "")
                + ". Download the canonical Rivière 2020 checkpoint from "
                  "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/"
                  "60k_epoch4-d0f474de.pt"
            )

        # Rebuild the model the hubconf.py way: start from
        # get_default_cpc_config(), overlay the checkpoint's training
        # args, then construct encoder + AR + CPCModel and load weights.
        loc_args = get_default_cpc_config()
        loadArgs(loc_args, _argparse.Namespace(**ckpt_data["config"]))
        encoder_net = getEncoder(loc_args)
        ar_net = getAR(loc_args)
        model = _CPCModel(encoder_net, ar_net)
        # strict=False because the release checkpoint omits criterion /
        # optimizer tensors that aren't part of CPCModel itself.
        model.load_state_dict(ckpt_data["weights"], strict=False)

        # Post-getAR() is when the transformer arMode path reassigns
        # hiddenGar = hiddenEncoder, so read both AFTER those calls.
        hidden_context_dim = int(getattr(loc_args, "hiddenGar", CPC_FEATURE_DIM))
        hidden_encoder_dim = int(getattr(loc_args, "hiddenEncoder", CPC_FEATURE_DIM))

    else:
        print(f"  Loading CPC:       torch.hub {hub_repo} :: {hub_entrypoint}")
        try:
            # pretrained=True is required by hubconf.CPC_audio to actually
            # download + load the 60k_epoch4 weights. Without it, you get
            # a randomly-initialized model and meaningless features.
            loaded = torch.hub.load(
                hub_repo, hub_entrypoint,
                pretrained=True, trust_repo=True,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"torch.hub.load({hub_repo!r}, {hub_entrypoint!r}) failed: "
                f"{e}\n"
                f"\n"
                f"This is almost always caused by a network that blocks "
                f"github.com or dl.fbaipublicfiles.com — common on SSL-\n"
                f"inspecting corporate / university WiFi (e.g. USC secure-"
                f"wireless).\n"
                f"\n"
                f"Workaround: download the repo + checkpoint manually on a "
                f"network that works, then re-run with both paths as CLI args:\n"
                f"\n"
                f"  1. Clone the CPC_audio repo:\n"
                f"     git clone https://github.com/facebookresearch/CPC_audio "
                f"/path/to/CPC_audio\n"
                f"\n"
                f"  2. Download the Rivière 2020 pretrained checkpoint (~7 MB):\n"
                f"     curl -L -o /path/to/60k_epoch4-d0f474de.pt \\\n"
                f"         https://dl.fbaipublicfiles.com/librilight/"
                f"CPC_checkpoints/60k_epoch4-d0f474de.pt\n"
                f"\n"
                f"  3. Re-run this script with both paths:\n"
                f"     python extract_cpc_from_manifest.py ... \\\n"
                f"         --cpc-repo /path/to/CPC_audio \\\n"
                f"         --checkpoint /path/to/60k_epoch4-d0f474de.pt"
            ) from e

        # hubconf.CPC_audio returns just the model (not a tuple). We
        # tolerate the tuple form too for forward-compat with older /
        # custom hub entrypoints like loadModel-returning-tuples.
        if isinstance(loaded, tuple):
            model = loaded[0]
            hidden_context_dim = loaded[1] if len(loaded) > 1 else CPC_FEATURE_DIM
            hidden_encoder_dim = loaded[2] if len(loaded) > 2 else CPC_FEATURE_DIM
        else:
            model = loaded
            # For the hubconf path we don't get dim hints back; default to
            # the Rivière 2020 baseline width. If a downstream sidecar
            # feature_dim disagrees, the feature-save path reads the
            # actual tensor shape so truth wins.
            hidden_context_dim = CPC_FEATURE_DIM
            hidden_encoder_dim = CPC_FEATURE_DIM

    model.eval()
    model.to(device)

    if hidden_context_dim != CPC_FEATURE_DIM or hidden_encoder_dim != CPC_FEATURE_DIM:
        # Defensive warning: the Rivière baseline is 256-dim everywhere.
        # If we see something else (e.g., a larger custom CPC), we honor
        # the loaded size but log it so the sidecar's feature_dim stays truthful.
        print(f"  NOTE: loaded CPC has context_dim={hidden_context_dim}, "
              f"encoder_dim={hidden_encoder_dim} (expected {CPC_FEATURE_DIM}).")

    print(f"  Device:            {device}")
    print(f"  CPC feature dim:   {hidden_context_dim}")
    return model, int(hidden_context_dim)


# =============================================================================
# Per-sample helpers
# =============================================================================

def _mean_pool(features, pool_factor: int):
    """Mean-pool along time by pool_factor. Input (T, D) → (T // pool_factor, D).

    In-memory helper. Used both during extraction (when --pool-factor > 1 or
    --mean-pool is set) and by mean_pool_file() below for post-hoc pooling of
    already-saved files.

    Tail frames that don't complete a pool window are truncated (≤ pool_factor
    − 1 frames dropped; at 100 Hz native, < 10 ms loss at pool_factor=10).
    """
    import torch  # noqa: PLC0415

    t, d = features.shape
    keep = (t // pool_factor) * pool_factor
    if keep == 0:
        return torch.zeros((0, d), dtype=features.dtype, device=features.device)
    trimmed = features[:keep]
    return trimmed.reshape(keep // pool_factor, pool_factor, d).mean(dim=1)


def mean_pool_file(
    file: str,
    save_to: str,
    pool_to: float = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    """Mean-pool a saved CPC feature file to a target frame rate.

    Post-hoc pooling entrypoint. Reads an already-extracted .npy (or .npz)
    plus its sibling .json sidecar, determines the current frame rate from
    the sidecar, validates that `current_rate / pool_to` is an integer,
    applies in-memory mean-pooling, and writes the pooled .npy + an updated
    sidecar to `save_to`. The input files are NOT modified.

    Intended workflow: run extract_cpc_from_manifest.py at the default
    (pool_factor=1, saving native 100 Hz) into e.g. `subset/cpc_100hz/`,
    then later call mean_pool_file on each .npy to produce a parallel 10 Hz
    corpus at e.g. `subset/cpc_10hz/` without re-running CPC. The
    extraction script's `--output-dir` is mandatory and must already
    exist; mean_pool_file's `save_to` is also mandatory but will create
    the directory if missing (asymmetric on purpose — the extraction run
    is the expensive one, so it gets the stricter footgun guard).

    Parameters
    ----------
    file : str
        Path to a .npy (canonical) or .npz (first array used) feature file.
        Its sibling sidecar at `{stem}.json` in the same directory MUST
        exist — this is where the source feature rate is read from.
    save_to : str
        Directory to write the pooled .npy + updated .json into. Created
        if missing. The output filename mirrors the input basename.
    pool_to : float, default 10.0
        Target frame rate in Hz. Must exactly divide the source rate (e.g.
        100 → 10 is pool_factor=10 ✓; 100 → 15 raises because 100/15 is
        not an integer). Use DEFAULT_TARGET_RATE_HZ (10.0) for the standard
        downstream-aligned rate.

    Returns
    -------
    dict
        Keys: status ("ok" | "failed"), input_file, output_file,
        pool_factor, input_frames, output_frames, source_rate_hz,
        target_rate_hz, message.

    Raises
    ------
    FileNotFoundError
        Input .npy or its sibling sidecar is missing.
    ValueError
        Non-integer pool factor, sidecar missing feature_rate_hz, empty
        .npz archive, non-2-D feature array, or target rate exceeds
        source rate (no upsampling supported).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # --- 1. Resolve + validate paths ---------------------------------------
    input_path = Path(file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input feature file not found: {file}")

    sidecar_path = input_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar not found at {sidecar_path}. mean_pool_file needs the "
            f"sidecar to read feature_rate_hz. Did you move the .npy without "
            f"its .json?"
        )

    # --- 2. Read sidecar, determine current rate, compute pool factor ------
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    current_rate = sidecar.get("feature_rate_hz")
    if current_rate is None:
        raise ValueError(
            f"Sidecar {sidecar_path} is missing `feature_rate_hz`. Cannot "
            f"determine pool factor without knowing the source rate."
        )
    current_rate = float(current_rate)

    if pool_to <= 0:
        raise ValueError(f"pool_to must be positive; got {pool_to}")
    if pool_to > current_rate:
        raise ValueError(
            f"Target rate {pool_to} Hz exceeds source rate {current_rate} "
            f"Hz — mean_pool_file does not upsample."
        )

    pool_factor_float = current_rate / pool_to
    pool_factor = int(round(pool_factor_float))
    if abs(pool_factor_float - pool_factor) > 1e-6:
        raise ValueError(
            f"Non-integer pool factor: {current_rate} Hz / {pool_to} Hz = "
            f"{pool_factor_float}. Pick a pool_to that exactly divides the "
            f"source rate (e.g. 100→10, 100→20, 100→25, 100→50)."
        )
    if pool_factor < 1:
        # Guarded by the pool_to > current_rate check above; defensive.
        raise ValueError(f"Computed pool_factor < 1 (={pool_factor}).")

    # --- 3. Load feature array (both .npy and .npz handled) ----------------
    if input_path.suffix == ".npz":
        with np.load(input_path) as npz:
            keys = list(npz.keys())
            if not keys:
                raise ValueError(f"Empty .npz archive: {file}")
            # Canonical choice: first stored array. Most of our files are .npy,
            # so this branch is a courtesy for interop.
            arr = np.asarray(npz[keys[0]])
    else:
        arr = np.load(input_path)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D feature array (T, D) from {file}; got shape "
            f"{arr.shape}. mean_pool_file pools along the time axis only."
        )

    # --- 4. Pool (reuse the same helper the extraction path uses) ----------
    features_tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    pooled = _mean_pool(features_tensor, pool_factor).numpy().astype(np.float32)

    # --- 5. Write pooled .npy + updated sidecar ----------------------------
    save_dir = Path(save_to)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Always write .npy regardless of input suffix — downstream consumers
    # expect the canonical format.
    output_npy = save_dir / f"{input_path.stem}.npy"
    output_json = save_dir / f"{input_path.stem}.json"

    # Guard against accidental in-place overwrite of the source file. If
    # the caller passes save_to == input_path.parent, the output filename
    # collides with the input filename and we'd silently clobber the
    # native 100 Hz source. Refuse rather than destroy data.
    if output_npy.resolve() == input_path.resolve():
        raise ValueError(
            f"Refusing to overwrite the input feature file at {input_path}. "
            f"save_to ({save_to!r}) resolves to the same directory as the "
            f"input; pick a different save_to (e.g. 'subset/cpc_10hz')."
        )

    np.save(output_npy, pooled)

    # Preserve the full original sidecar as provenance, then overlay the
    # fields that change on pooling. `pool_factor` is multiplicative so
    # chained pool steps (rare, but supported) reflect cumulative downsampling.
    out_sidecar = dict(sidecar)
    out_sidecar["pool_factor"] = int(sidecar.get("pool_factor", 1)) * pool_factor
    out_sidecar["feature_rate_hz"] = float(pool_to)
    out_sidecar["feature_frames"] = int(pooled.shape[0])
    out_sidecar["feature_dim"] = (int(pooled.shape[1]) if pooled.size
                                  else int(sidecar.get("feature_dim", 0)))
    # Post-pool audit trail: who pooled this file and from where.
    out_sidecar["post_pool_source_file"] = str(input_path)
    out_sidecar["post_pool_source_rate_hz"] = current_rate
    out_sidecar["post_pool_factor"] = pool_factor

    with open(output_json, "w") as f:
        json.dump(out_sidecar, f, indent=2)

    return {
        "status": "ok",
        "input_file": str(input_path),
        "output_file": str(output_npy),
        "pool_factor": pool_factor,
        "input_frames": int(arr.shape[0]),
        "output_frames": int(pooled.shape[0]),
        "source_rate_hz": current_rate,
        "target_rate_hz": float(pool_to),
        "message": (
            f"pooled {arr.shape[0]:,}×{arr.shape[1]} @ {current_rate:g} Hz "
            f"-> {pooled.shape[0]:,}×{pooled.shape[1]} @ {pool_to:g} Hz "
            f"(factor={pool_factor})"
        ),
    }


def _build_sidecar(
    task: ExtractTask,
    pooled,
    orig_sr: int,
    duration_s: float,
    feature_type: str,
    feature_dim: int,
    pool_factor: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
) -> dict:
    """Assemble the per-window sidecar dict shared by the writer path.

    Pulled out of the extraction loop so the schema lives in exactly one
    place. Consumers (downstream feature-stacking code, `mean_pool_file`)
    rely on `feature_rate_hz` and `feature_dim` being present and
    accurate.
    """
    return {
        "orig_stem":                task.orig_stem,
        "spliced_stem":             Path(task.audio_path).stem,
        "interaction_id":           task.interaction_id,
        "participant_id":           task.participant_id,
        "window_start_s":           task.window_start_s,
        "window_end_s":             task.window_end_s,
        "window_duration_s":        task.window_end_s - task.window_start_s,
        "audio_source":             task.audio_path,
        "source_sample_rate_hz":    int(orig_sr),
        "resampled_sample_rate_hz": CPC_SAMPLE_RATE,
        "duration_seconds":         duration_s,
        # Model provenance. Populated differently depending on the
        # load path (torch.hub vs. local checkpoint).
        "model_family":             "cpc",
        "model_id":                 (f"local:{checkpoint}" if checkpoint
                                     else f"{hub_repo}::{hub_entrypoint}"),
        "cpc_hub_repo":             hub_repo,
        "cpc_hub_entrypoint":       hub_entrypoint,
        "cpc_checkpoint":           checkpoint,
        "cpc_feature_type":         feature_type,   # 'context' (c_t) or 'encoded' (z_t)
        "cpc_samples_per_frame":    CPC_SAMPLES_PER_FRAME,
        "raw_feature_rate_hz":      CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME,  # 100
        "pool_factor":              pool_factor,
        "feature_rate_hz":          (CPC_SAMPLE_RATE
                                     / CPC_SAMPLES_PER_FRAME
                                     / pool_factor),
        "feature_dim":              int(pooled.shape[1]) if pooled.size else feature_dim,
        "feature_frames":           int(pooled.shape[0]),
        "dtype":                    str(pooled.dtype),
        # Chunking provenance. Kept for schema parity with
        # extract_wavlm_from_manifest.py and the pre-batched version of
        # this script; batched inference over spliced windows never
        # chunks (each wav is already short), so these are always 0.
        "chunk_samples":            0,
        "chunk_length_s":           0.0,
    }


def _write_outputs(npy_path: str, json_path: str, pooled, sidecar: dict) -> None:
    """Write .npy then .json. Runs in a writer-pool thread.

    Order is load-bearing for the resume invariant (see already_have and
    the Start/stop resilience section of the module docstring): .npy
    must land before .json, so a mid-write crash leaves the pair in a
    state that `already_have` flags as incomplete.
    """
    import numpy as np  # noqa: PLC0415

    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, pooled)
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def _checkpoints_equivalent(a: str | None, b: str | None) -> bool:
    """True iff two checkpoint paths refer to the same file (or are
    both None, i.e. both runs are using torch.hub).

    Handles the common non-literal-match cases that would otherwise
    make the compat check spuriously fail:
      - symlinks and FS-level inode aliasing (via os.path.samefile)
      - relative vs. absolute paths
      - `~` / `$HOME` expansion
      - trailing slashes, redundant `.` / `..` segments
      - case differences on Windows (normcase)

    Deliberately NOT handled: content-level equivalence (a byte-for-
    byte identical .pt at a different path). That would require
    hashing, which we skip; users who genuinely moved a checkpoint
    while preserving contents should just pass --overwrite.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Inode-level identity — fastest and catches the symlink / FS-
    # aliasing cases. Only works when both files exist on disk.
    try:
        if os.path.samefile(a, b):
            return True
    except OSError:
        pass

    # Fall back to normalized-path string compare for the
    # doesn't-exist-yet or cross-FS cases.
    def _norm(p: str) -> str:
        return os.path.normcase(os.path.realpath(os.path.expanduser(p)))

    return _norm(a) == _norm(b)


def check_output_dir_params_compat(
    output_dir: str,
    feature_type: str,
    pool_factor: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
) -> str | None:
    """Verify existing sidecars in output_dir match the current run's
    model identity and extraction parameters.

    Mixing incompatible parameters in one output directory produces a
    silently-heterogeneous corpus: `already_have` would skip existing
    files (seeing them as complete) even though their sidecars disagree
    with the new run's intent. To prevent that, we sample a real CPC
    sidecar and compare its full extraction identity against the
    current args:

      - cpc_feature_type   (context vs. encoded)
      - pool_factor        (100 Hz native vs. pooled rates)
      - cpc_hub_repo       (upstream repo; detects silent fork swap)
      - cpc_hub_entrypoint (hubconf entrypoint; detects rename)
      - cpc_checkpoint     (local .pt path; detects checkpoint swap)

    The checkpoint comparison uses `_checkpoints_equivalent`, which
    treats symlinked / relative / `~`-expanded paths pointing at the
    same file as equal, so runs that reference the same .pt through
    different paths don't trip the check spuriously.

    To avoid sampling stray / foreign .json files that happen to live
    in an `<output_dir>/<stem>/` subdir, we require each candidate
    sidecar to satisfy two structural sanity checks:

      1. A sibling .npy file exists at the same stem (indicating this
         json is paired with a real feature file, as every CPC sidecar
         should be).
      2. `sidecar["model_family"] == "cpc"` (filters out sidecars from
         other feature extractors — e.g. WavLM's).

    We iterate candidates until one passes both checks, then use it
    for comparison. Sampling just the first qualifying sidecar is
    intentional: if the first one matches, we assume the whole
    directory was written by a single consistent run (which is the
    invariant we're trying to maintain). Directories hand-edited to
    contain mixed params can still slip through — the per-file
    sidecar records the truth and downstream consumers can re-check.

    Returns None if no qualifying sidecars exist, if the sampled
    sidecar matches current params, or if every candidate failed to
    parse (a corrupt sidecar isn't evidence of param mismatch; the
    resume path will re-extract it on the next run). Returns a
    human-readable error message listing the mismatched fields on
    real disagreement; the caller decides whether to refuse the run
    or proceed.
    """
    root = Path(output_dir)
    sidecar: dict | None = None
    sampled_path: Path | None = None
    read_errors: list[tuple[Path, Exception]] = []

    # Iterate candidates rather than picking the first match. A subdir
    # may contain a stray file that isn't our sidecar (e.g. a manifest,
    # a README, a partial scratch file); skipping those and continuing
    # avoids a spurious mismatch error against a non-CPC json.
    for candidate in root.glob("*/*.json"):
        # Structural check 1: a sibling .npy at the same stem.
        if not candidate.with_suffix(".npy").exists():
            continue
        # Parse. On error, remember but keep looking — another subdir
        # may have a clean sidecar.
        try:
            with open(candidate) as f:
                parsed = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            read_errors.append((candidate, e))
            continue
        # Guard: json.load can legitimately return any JSON value
        # (list, scalar, null). A stray .json in this subdir tree
        # might contain any of those, and calling .get on a non-dict
        # would AttributeError and crash the whole compat check.
        # Treat anything that isn't a JSON object as "not a CPC
        # sidecar" and keep searching.
        if not isinstance(parsed, dict):
            continue
        # Structural check 2: model_family tag. Filters out foreign
        # sidecars (e.g. from extract_wavlm_from_manifest.py) that
        # happen to share the subdir layout.
        if parsed.get("model_family") != "cpc":
            continue
        sidecar = parsed
        sampled_path = candidate
        break

    if sidecar is None:
        # No qualifying CPC sidecar was found. If we hit parse errors
        # along the way, surface a one-time warning so the user isn't
        # surprised by the verification being skipped; otherwise stay
        # silent (a fresh / empty output_dir is the normal case).
        if read_errors:
            first_path, first_err = read_errors[0]
            n = len(read_errors)
            print(
                f"WARNING: found {n} existing sidecar-shaped .json "
                f"file(s) under {output_dir} but could not parse any "
                f"(first error: {first_path}: {first_err}); proceeding "
                f"without parameter-compatibility verification. Pass "
                f"--overwrite if you want to discard prior output and "
                f"start clean.",
                file=sys.stderr,
                flush=True,
            )
        return None

    # --- Compare the sampled sidecar's params against current args ---
    # String / integer fields: literal equality. A missing key
    # (sidecar.get → None) counts as a mismatch against a non-None
    # current value so an ancient / foreign sidecar missing required
    # fields can't slip through.
    string_checks = (
        ("cpc_feature_type",   sidecar.get("cpc_feature_type"),   feature_type),
        ("pool_factor",        sidecar.get("pool_factor"),        pool_factor),
        ("cpc_hub_repo",       sidecar.get("cpc_hub_repo"),       hub_repo),
        ("cpc_hub_entrypoint", sidecar.get("cpc_hub_entrypoint"), hub_entrypoint),
    )
    mismatches = [
        f"{name}: sidecar has {existing!r}, current run requests {current!r}"
        for name, existing, current in string_checks
        if existing != current
    ]

    # Checkpoint path: use path-equivalence rather than literal string
    # compare so symlink / relative / home-dir variants of the same .pt
    # don't trip the check.
    existing_ckpt = sidecar.get("cpc_checkpoint")
    if not _checkpoints_equivalent(existing_ckpt, checkpoint):
        mismatches.append(
            f"cpc_checkpoint: sidecar has {existing_ckpt!r}, "
            f"current run requests {checkpoint!r}"
        )

    if not mismatches:
        return None

    bullet_list = "\n".join(f"           - {m}" for m in mismatches)
    return (
        f"existing sidecar {sampled_path} disagrees with the current "
        f"run's parameters:\n"
        f"{bullet_list}\n"
        f"       Mixing these in one output directory would produce an "
        f"inconsistent corpus (already_have would silently skip the "
        f"old-params files). Either:\n"
        f"         - use a different --output-dir for the new params, or\n"
        f"         - pass --overwrite to discard the prior extraction."
    )


# =============================================================================
# Dataset / collate — async audio load + resample in DataLoader workers
# =============================================================================

class _SplicedWavDataset:
    """Dataset that loads + resamples one spliced wav per __getitem__.

    Returns a 5-tuple:
        (waveform_1d_fp32_16k, source_sr, task_index, err_msg, status)

    On success:  (waveform, sr,   idx, None,    None)
    On missing:  (None,     None, idx, err_msg, "missing")  # file not on disk
    On failure:  (None,     None, idx, err_msg, "failed")   # load/decode error

    The "missing" vs "failed" split is preserved end-to-end through the
    collate so main() can return exit code 3 for runs where inputs
    weren't on disk (non-fatal) vs 2 for runs with real extraction
    errors.

    Each DataLoader worker gets its own copy of this instance after
    pickle, so the torchaudio.transforms.Resample cache lives per-worker.
    In practice every spliced wav in the corpus shares a single source
    sample rate (48 kHz from Seamless), so the cache resolves to one
    Resample object per worker after the first file.
    """

    def __init__(self, tasks: list[ExtractTask]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int):
        import torchaudio  # noqa: PLC0415

        task = self.tasks[idx]

        # Differentiate "file not on disk" ("missing", non-fatal at the
        # run level — causes main() to exit 3) from other load errors
        # ("failed", exit 2). This restores the semantic the old
        # extract_one path used to emit before the rewrite.
        if not os.path.exists(task.audio_path):
            return (None, None, idx,
                    f"input wav not on disk: {task.audio_path}",
                    "missing")

        try:
            waveform, sr = torchaudio.load(task.audio_path)  # (1, T)
            if sr != CPC_SAMPLE_RATE:
                cache = getattr(self, "_resamplers", None)
                if cache is None:
                    cache = {}
                    self._resamplers = cache
                resampler = cache.get(sr)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=CPC_SAMPLE_RATE
                    )
                    cache[sr] = resampler
                waveform = resampler(waveform)
            return waveform.squeeze(0), sr, idx, None, None
        except Exception as e:  # noqa: BLE001
            return None, None, idx, f"{type(e).__name__}: {e}", "failed"


def _collate_batch(items):
    """Stack successful loads into (B, T); keep failures separate.

    Every spliced wav has identical length by construction (splice_wavs
    skips partial trailing windows), so `torch.stack` doesn't need
    padding.

    Defensive length-mismatch handling: if lengths disagree (should not
    happen given splice_wavs' contract, but might if a spliced wav is
    corrupt or truncated on disk), we keep whichever length is most
    common in the batch and DEMOTE the outliers to `failures` with a
    length-mismatch error. This preserves the majority's features
    intact, rather than truncating every sample in the batch down to
    the shortest outlier (which would silently gut the majority's
    audio).

    Each failure carries its status ("missing", "failed", or now
    "failed" with a length-mismatch message) so the caller can preserve
    per-status exit-code semantics.

    Returns (batch_tensor, ok_srs, ok_local_idxs, orig_lengths,
             failures) where:
      - ok_srs/ok_local_idxs/orig_lengths cover ONLY the samples that
        made it into batch_tensor (post-outlier-drop, if any),
      - failures is list[(local_idx, err_msg, status)].
    """
    import torch  # noqa: PLC0415

    ok_waves, ok_srs, ok_idxs, orig_lengths = [], [], [], []
    failures = []
    for wav, sr, idx, err, status in items:
        if wav is None:
            failures.append((idx, err, status))
        else:
            ok_waves.append(wav)
            ok_srs.append(sr)
            ok_idxs.append(idx)
            orig_lengths.append(int(wav.shape[0]))

    if not ok_waves:
        return None, [], [], [], failures

    length_counts = Counter(orig_lengths)
    if len(length_counts) == 1:
        # Happy path — every wav has the same length, as splice_wavs
        # guarantees. No outliers to drop.
        batch = torch.stack(ok_waves, dim=0)
        return batch, ok_srs, ok_idxs, orig_lengths, failures

    # Lengths disagree. Use the modal (most common) length as the
    # authoritative shape for this batch, and drop everything else
    # into `failures` rather than truncating batchmates to match a
    # short outlier.
    #
    # Tie-break via explicit sort rather than Counter.most_common:
    # the language reference doesn't guarantee most_common's ordering
    # for equal counts, so relying on CPython's implementation
    # (insertion order) would not be portable. `max` with a key of
    # (count, length) is deterministic everywhere — primary order
    # count DESC (most common wins), secondary order length DESC
    # (prefer the longer value on ties; corruption typically produces
    # shorter outliers, so trusting the longer length is the safer
    # default).
    dominant_length, _count = max(
        length_counts.items(),
        key=lambda item: (item[1], item[0]),
    )
    kept_waves, kept_srs, kept_idxs, kept_lengths = [], [], [], []
    for wav, sr, idx, length in zip(ok_waves, ok_srs, ok_idxs, orig_lengths):
        if length == dominant_length:
            kept_waves.append(wav)
            kept_srs.append(sr)
            kept_idxs.append(idx)
            kept_lengths.append(length)
        else:
            failures.append((
                idx,
                (f"length mismatch: {length} samples, batch dominant "
                 f"length is {dominant_length}; dropped to avoid "
                 f"truncating batchmates"),
                "failed",
            ))

    # `dominant_length` is drawn from `orig_lengths`, so at least one
    # sample always matches it — kept_waves is guaranteed non-empty.
    batch = torch.stack(kept_waves, dim=0)
    return batch, kept_srs, kept_idxs, kept_lengths, failures


# =============================================================================
# Orchestration
# =============================================================================

def already_have(output_path: str, sidecar_path: str) -> bool:
    """True iff both .npy and .json exist and parse — idempotent resume.

    A 0-byte .npy from a crashed prior run is treated as missing so the
    next run re-extracts it. Truncated sidecars (killed mid-write) fail
    json.load and are also re-run. See the Start/stop resilience block
    in the module docstring for the full invariant.
    """
    for p in (output_path, sidecar_path):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
    try:
        with open(sidecar_path) as f:
            json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return True


def _batched_forward_on_device(batch_tensor, model, device: str, feature_type: str):
    """Run CPC forward over a stacked (B, T) batch; return (B, T_frames, D)
    on CPU. Raises RuntimeError on forward failure — caller decides
    whether to fall back to per-sample retries.
    """
    import torch  # noqa: PLC0415

    x = batch_tensor.to(device=device, dtype=torch.float32).unsqueeze(1)
    # x: (B, 1, T)
    with torch.inference_mode():
        out = model(x, None)
    if isinstance(out, tuple):
        c_feature = out[0]
        z_feature = out[1] if len(out) > 1 else out[0]
    else:
        c_feature = z_feature = out
    chosen = c_feature if feature_type == "context" else z_feature
    return chosen.to(torch.float32).cpu()


def _per_sample_fallback_forward(
    batch_tensor,
    model,
    device: str,
    feature_type: str,
    batched_error: Exception,
):
    """When a batched forward raises, try each row individually so only
    the genuinely-bad sample(s) get labeled failed.

    Motivation: an OOM or shape error on a B=64 batch otherwise marks
    every innocent bystander as failed, wasting compute when the cause
    is isolated to one or two inputs. Running each sample as its own
    B=1 forward also resolves most OOM-caused batch failures without
    the user having to rerun with a smaller --batch-size.

    Catches any `Exception` (not just `RuntimeError`) from a single
    sample's forward and converts it to an error message for that row.
    This is the whole point of the per-sample path: isolate ANY
    individual-sample failure so the surviving batchmates still get
    processed. `RuntimeError` alone would miss the less common
    `AttributeError` / `TypeError` / `AssertionError` / `ValueError`
    paths that torch or a third-party model can raise. Note: by
    catching `Exception` rather than `BaseException`, `KeyboardInterrupt`
    and `SystemExit` still propagate normally — Ctrl-C still aborts.

    Returns a list of (features_cpu_tensor | None, error_msg | None) —
    one entry per row of batch_tensor, aligned with its rows.
    """
    import torch  # noqa: PLC0415

    batched_is_oom = "out of memory" in str(batched_error).lower()
    outputs = []
    for i in range(batch_tensor.shape[0]):
        single = batch_tensor[i:i + 1].to(
            device=device, dtype=torch.float32
        ).unsqueeze(1)
        # single: (1, 1, T)
        try:
            with torch.inference_mode():
                out = model(single, None)
            if isinstance(out, tuple):
                c_feature = out[0]
                z_feature = out[1] if len(out) > 1 else out[0]
            else:
                c_feature = z_feature = out
            chosen = c_feature if feature_type == "context" else z_feature
            features = chosen.squeeze(0).to(torch.float32).cpu()
            outputs.append((features, None))
        except Exception as e:  # noqa: BLE001 — intentionally broad (see docstring)
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            # `is_oom` is only meaningful for RuntimeError-family
            # messages; other exception types don't carry the OOM
            # signature, so the check correctly falls through to the
            # "deterministic failure" branch for them.
            is_oom = "out of memory" in str(e).lower()
            # Distinguish cause in the message: batch-level OOM that
            # resolves per-sample means the sample is fine; per-sample
            # OOM means the sample itself is too big or the device is
            # pathologically low on memory.
            batch_note = (
                f" (batched forward failed with: "
                f"{type(batched_error).__name__}: {batched_error})"
            )
            hint = ""
            if is_oom and batched_is_oom:
                hint = (" — OOM even at B=1; try a different --device "
                        "or shrink the spliced window length")
            elif not is_oom:
                hint = (" — deterministic failure on this sample; "
                        "check input and model compatibility")
            outputs.append((
                None,
                f"CPC forward failed on isolated sample "
                f"({type(e).__name__}: {e}){hint}.{batch_note}",
            ))
    return outputs


def run_tasks(
    tasks: list[ExtractTask],
    model,
    device: str,
    feature_type: str,
    feature_dim: int,
    pool_factor: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
    overwrite: bool,
    dry_run: bool,
    batch_size: int,
    num_workers: int,
    writer_threads: int,
    progress_every: int,
) -> list[dict]:
    """Batched CPC extraction with async I/O.

    Pipeline:
      [DataLoader workers]    load + 48->16 kHz resample on CPU cores
              ↓
      [main]                  stack to (B, 1, T), run CPC forward on MPS/CUDA
                              — on RuntimeError, fall back to B=1 per-sample
                                so one bad input doesn't fail its batchmates
              ↓
      [writer ThreadPool]     NaN guard → pool → write .npy + .json

    The writer pool decouples disk I/O from the GPU loop: once a batch's
    forward is done, per-sample saves are submitted and the main thread
    moves on to the next forward. `writer.shutdown(wait=True)` in the
    `finally` block drains pending writes on Ctrl-C so in-flight tasks
    either complete atomically or are re-run on resume. The same
    `finally` also backfills any result slots that never got populated
    (e.g. after a mid-loop KeyboardInterrupt, a DataLoader worker crash,
    or an unhandled exception from the forward path) with a terminal
    failure entry, so `print_summary` and main()'s exit-code computation
    never encounter `None`.

    Resilience is preserved end-to-end: _write_outputs writes .npy before
    .json per task, and already_have() is checked up front to skip any
    task whose pair already exists. See module docstring.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    n_total = len(tasks)
    results: list[dict | None] = [None] * n_total

    # Partition up-front into {skipped, dry, todo}. already_have() is
    # cheap (stat + json.load of a tiny file) and done here rather than
    # inside the DataLoader workers so the inner loop is all compute.
    todo_tasks: list[ExtractTask] = []
    todo_to_global: list[int] = []

    for g_idx, task in enumerate(tasks):
        if not overwrite and already_have(task.output_path, task.sidecar_path):
            results[g_idx] = {
                "task": task, "status": "skipped",
                "message": f"already extracted "
                           f"({os.path.getsize(task.output_path):,} bytes)",
            }
        elif dry_run:
            results[g_idx] = {
                "task": task, "status": "dry",
                "message": f"DRY: {task.audio_path} -> {task.output_path}",
            }
        else:
            todo_to_global.append(g_idx)
            todo_tasks.append(task)

    n_to_do = len(todo_tasks)
    if n_to_do == 0:
        # Nothing to run through the loader — just print and return.
        _print_results(results, verbose=True)
        return results  # type: ignore[return-value]

    print(f"  dispatching {n_to_do:,} new tasks "
          f"(already-done: {n_total - n_to_do:,})")

    ds = _SplicedWavDataset(todo_tasks)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_batch,
        pin_memory=False,   # irrelevant on MPS (unified memory) and CPU
        shuffle=False,
        persistent_workers=(num_workers > 0),
    )

    writer = concurrent.futures.ThreadPoolExecutor(
        max_workers=writer_threads,
        thread_name_prefix="cpc-write",
    )
    # pending holds (future, g_idx, task, pooled, duration_s) — we need
    # the pooled array reference alive until the future completes
    # because np.save reads from it asynchronously.
    pending: list[tuple] = []

    done = 0
    last_progress = 0
    start_time = time.time()

    # Track fallbacks from batched → per-sample forward. Printed per-
    # batch up to a cap (to avoid spamming logs if fallback is
    # persistent), plus a total in the end-of-run summary so silent
    # throughput drops don't go unnoticed.
    n_fallback_batches = 0
    FALLBACK_LOG_CAP = 5

    def _reap_completed(force: bool) -> None:
        """Move completed futures into results[]. Free pooled arrays."""
        nonlocal pending
        still = []
        for fut, g_idx, task, pooled, duration_s in pending:
            if fut.done() or force:
                try:
                    fut.result()
                    results[g_idx] = {
                        "task": task, "status": "ok",
                        "message": f"OK ({pooled.shape[0]:,}×{pooled.shape[1]}, "
                                   f"{duration_s:.1f}s -> {pooled.nbytes:,} bytes)",
                    }
                except Exception as e:  # noqa: BLE001
                    results[g_idx] = {
                        "task": task, "status": "failed",
                        "message": f"save failed: {e}",
                    }
            else:
                still.append((fut, g_idx, task, pooled, duration_s))
        pending = still

    def _queue_sample_write(
        features,
        task: ExtractTask,
        g_idx: int,
        orig_length: int,
        orig_sr: int,
    ) -> None:
        """NaN guard → pool → build sidecar → submit to writer pool.

        If features contain NaN/Inf, fills results[g_idx] with a failed
        entry and returns without submitting. Otherwise submits the
        write and appends to `pending` for later reaping.

        `orig_length` is the waveform's TRUE sample count before any
        collate-side length truncation, so the sidecar's
        `duration_seconds` reflects the actual input duration even when
        the defensive truncation path fired.
        """
        nonlocal done
        if not torch.isfinite(features).all():
            nan_count = int(torch.isnan(features).sum())
            inf_count = int(torch.isinf(features).sum())
            results[g_idx] = {
                "task": task, "status": "failed",
                "message": (f"non-finite features: nan={nan_count:,} "
                            f"inf={inf_count:,} of {features.numel():,}"),
            }
            done += 1
            return

        pooled = _mean_pool(features, pool_factor).numpy().astype(np.float32)
        duration_s = orig_length / CPC_SAMPLE_RATE
        sidecar = _build_sidecar(
            task=task,
            pooled=pooled,
            orig_sr=orig_sr,
            duration_s=duration_s,
            feature_type=feature_type,
            feature_dim=feature_dim,
            pool_factor=pool_factor,
            hub_repo=hub_repo,
            hub_entrypoint=hub_entrypoint,
            checkpoint=checkpoint,
        )
        fut = writer.submit(
            _write_outputs,
            task.output_path, task.sidecar_path, pooled, sidecar,
        )
        pending.append((fut, g_idx, task, pooled, duration_s))
        done += 1

    try:
        for batch in loader:
            (batch_tensor, ok_srs, ok_local_idxs,
             orig_lengths, failures) = batch

            # 1) Record load-time failures (DataLoader-worker errors).
            #    Preserve the "missing" vs "failed" distinction so
            #    main() can return exit code 3 for files-not-on-disk.
            for local_idx, err_msg, status in failures:
                g_idx = todo_to_global[local_idx]
                results[g_idx] = {
                    "task": todo_tasks[local_idx],
                    "status": status,
                    "message": err_msg,
                }
                done += 1

            if batch_tensor is None:
                continue

            # 2) Try the batched forward. If it raises, fall back to
            #    per-sample forwards so an OOM or bad single input
            #    doesn't mark all 64 batchmates failed.
            batched_chosen_cpu = None
            per_sample_outputs = None
            try:
                batched_chosen_cpu = _batched_forward_on_device(
                    batch_tensor, model, device, feature_type,
                )
            except RuntimeError as batched_error:
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                n_fallback_batches += 1
                # Surface the first few fallbacks so the user isn't
                # blindsided by a throughput drop. Cap the per-batch
                # noise; the total count is printed in the summary.
                if n_fallback_batches <= FALLBACK_LOG_CAP:
                    is_oom = "out of memory" in str(batched_error).lower()
                    oom_hint = (" — reduce --batch-size if this is OOM"
                                if is_oom else "")
                    print(
                        f"  [WARN] batched forward failed on batch "
                        f"#{n_fallback_batches} "
                        f"({type(batched_error).__name__}: {batched_error}); "
                        f"falling back to per-sample retry{oom_hint}",
                        file=sys.stderr, flush=True,
                    )
                    if n_fallback_batches == FALLBACK_LOG_CAP:
                        print(
                            f"  [WARN] further batched-forward fallbacks "
                            f"will be suppressed; total reported in summary.",
                            file=sys.stderr, flush=True,
                        )
                per_sample_outputs = _per_sample_fallback_forward(
                    batch_tensor, model, device, feature_type, batched_error,
                )

            # 3) Per-sample: pick features from whichever forward
            #    succeeded, then NaN guard + pool + submit write.
            for i, local_idx in enumerate(ok_local_idxs):
                task = todo_tasks[local_idx]
                g_idx = todo_to_global[local_idx]

                if batched_chosen_cpu is not None:
                    features = batched_chosen_cpu[i]
                else:
                    # per_sample_outputs is populated whenever
                    # batched_chosen_cpu is None.
                    features, err_msg = per_sample_outputs[i]  # type: ignore[index]
                    if err_msg is not None:
                        results[g_idx] = {
                            "task": task, "status": "failed",
                            "message": err_msg,
                        }
                        done += 1
                        continue

                _queue_sample_write(
                    features,
                    task,
                    g_idx,
                    orig_lengths[i],
                    ok_srs[i],
                )

            # 4) Opportunistically reap completed writes so pooled-array
            #    references can be freed.
            _reap_completed(force=False)

            # 5) Progress ping.
            if progress_every > 0 and done - last_progress >= progress_every:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_s = (n_to_do - done) / rate if rate > 0 else 0.0
                print(f"  progress: {done:,} / {n_to_do:,} "
                      f"({100.0 * done / n_to_do:5.1f}%)  "
                      f"rate: {rate:6.1f}/s  "
                      f"ETA: {eta_s / 60:5.1f} min")
                last_progress = done
    finally:
        # Drain whatever remains and shut down the writer pool. In
        # `finally` so Ctrl-C during the loop still flushes in-flight
        # writes — each completed (npy, json) pair is atomic per the
        # resume invariant, so we preserve as much progress as possible
        # before exiting.
        _reap_completed(force=True)
        writer.shutdown(wait=True)

        # Backfill any result slots that never got populated. This can
        # happen if a KeyboardInterrupt propagates out of the for-loop
        # body, a DataLoader worker dies, or any other exception
        # escapes. Without this, `print_summary` and main()'s
        # n_failed/n_missing computation would TypeError on a None
        # subscript. "failed" is the right label: these tasks have
        # no .npy/.json pair on disk, so the resume path on the next
        # run will see them as not-yet-done and retry.
        for g_idx in range(n_total):
            if results[g_idx] is None:
                results[g_idx] = {
                    "task": tasks[g_idx], "status": "failed",
                    "message": ("task not processed before run terminated "
                                "(Ctrl-C or loader error); rerun to retry"),
                }

    # Surface the total number of batches that fell back to per-sample
    # forward. A non-zero count means throughput was (at best) ~B× slower
    # for those batches; the user should consider shrinking --batch-size
    # on the next run.
    if n_fallback_batches > 0:
        print(
            f"\n  [NOTE] {n_fallback_batches:,} batch(es) fell back to "
            f"per-sample forward after a batched-forward failure. "
            f"Consider --batch-size smaller next run (the current "
            f"batched-mode throughput claim does not apply to the "
            f"fallback path).",
            file=sys.stderr,
        )

    # Per-task log. Verbose for small runs (debug), failures-only for
    # large runs (because 500k lines of OK is useless and slow).
    _print_results(results, verbose=(n_total < 10_000))
    return results  # type: ignore[return-value]


def _print_results(results: list, verbose: bool) -> None:
    for r in results:
        if r is None:
            continue
        if verbose or r["status"] != "ok":
            print(f"  [{r['status']:<7}] {r['task'].orig_stem}/"
                  f"{Path(r['task'].audio_path).name}  {r['message']}")


def print_summary(results: list, elapsed_s: float, dry_run: bool) -> None:
    """Counts by status + total bytes written.

    Defensive against `None` entries in `results`: run_tasks's finally
    block backfills any un-populated slots with a "failed" entry, but
    we still tolerate None here so an unexpected code path that bypasses
    that backfill can't crash the summary (and hide whatever the actual
    error was).
    """
    counts: dict[str, int] = {}
    bytes_written = 0
    for r in results:
        if r is None:
            # Shouldn't happen — run_tasks fills Nones in `finally` — but
            # if it somehow does, treat as failed and keep going.
            counts["failed"] = counts.get("failed", 0) + 1
            continue
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "ok":
            try:
                bytes_written += int(
                    r["message"].split("->")[1]
                    .split(" bytes")[0]
                    .strip()
                    .replace(",", "")
                )
            except (IndexError, ValueError):
                pass

    print("\n" + "=" * 64)
    print("CPC EXTRACTION SUMMARY" + ("  [DRY-RUN]" if dry_run else ""))
    print("=" * 64)
    print(f"  Total tasks:  {len(results)}")
    for status in ("ok", "skipped", "dry", "missing", "failed"):
        n = counts.get(status, 0)
        if n:
            print(f"  {status:<10} {n}")
    if bytes_written:
        mib = bytes_written / (1024 * 1024)
        print(f"  Written:      {bytes_written:,} bytes ({mib:,.1f} MiB)")
    print(f"  Elapsed:      {elapsed_s:,.1f} s")
    print("=" * 64)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Rivière-2020 CPC features (256-dim @ 100 Hz native) "
            "from every spliced wav produced by splice_wavs.py using "
            "batched inference. Saves native 100 Hz by default; pass "
            "--mean-pool (or --pool-factor N) to pool during extraction, "
            "or leave default and invoke mean_pool_file() post-hoc. "
            "Mirrors the input tree under --output-dir: <orig_stem>/"
            "<spliced_stem>.npy + <spliced_stem>.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O paths. No defaults on purpose — see the note in the Defaults
    # block above.
    parser.add_argument(
        "--spliced-dir", required=True,
        help="REQUIRED. Parent directory containing splice_wavs.py output. "
             "Expected layout: <spliced-dir>/<orig_stem>/*.wav where each "
             "subdir is named after the original wav stem and holds one .wav "
             "per window. Must already exist.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="REQUIRED. Directory to save .npy / .json feature files. "
             "Mirrors the --spliced-dir tree exactly: <output-dir>/"
             "<orig_stem>/<spliced_stem>.npy + .json. Must already exist on "
             "disk (the script will not auto-create it — this is a footgun "
             "guard against typos that would otherwise silently land features "
             "in an unexpected location). Create it explicitly first: "
             "`mkdir -p subset/cpc_100hz` etc.",
    )

    # Model loading
    parser.add_argument(
        "--hub-repo", default=DEFAULT_HUB_REPO,
        help="torch.hub repo spec. Default: %(default)s",
    )
    parser.add_argument(
        "--hub-entrypoint", default=DEFAULT_HUB_ENTRYPOINT,
        help="torch.hub entrypoint name. Override if upstream renames it. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--cpc-repo", default=None,
        help="Path to a local clone of facebookresearch/CPC_audio. "
             "Overrides torch.hub when combined with --checkpoint. "
             "Use this on airgapped / no-GitHub environments.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a pretrained CPC checkpoint (.pt). Required when "
             "--cpc-repo is set. Use the canonical Rivière 2020 checkpoint "
             "from the facebookresearch/CPC_audio README.",
    )

    # Extraction config
    parser.add_argument(
        "--feature-type", choices=["context", "encoded"],
        default=DEFAULT_FEATURE_TYPE,
        help=("Which CPC stream to save. 'context' (default) = aggregator "
              "output c_t (causal RNN output — the standard VAP-compatible "
              "feature). 'encoded' = CNN-encoder output z_t (local-in-time, "
              "no RNN; used in some ablations and in ZeroSpeech-style "
              "phonetic probing)."),
    )
    # Pooling: default is to NOT pool during extraction (save native 100 Hz).
    # `--mean-pool` is a boolean shortcut for the common pool-to-10-Hz case.
    # `--pool-factor N` remains as an advanced escape hatch for custom rates
    # (e.g. --pool-factor 2 for 50 Hz VAP-parity). They are mutually
    # exclusive: pick one intent, not both.
    _native_rate = CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME                  # 100
    _mean_pool_factor = int(round(_native_rate / DEFAULT_TARGET_RATE_HZ))    # 10
    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument(
        "--mean-pool", action="store_true",
        help=f"Mean-pool during extraction to {DEFAULT_TARGET_RATE_HZ:g} Hz "
             f"(shortcut for --pool-factor {_mean_pool_factor}). Off by "
             f"default — the recommended flow is to leave this off, save "
             f"native {_native_rate} Hz features, and call mean_pool_file() "
             f"post-hoc to produce a pooled corpus in a separate directory.",
    )
    pool_group.add_argument(
        "--pool-factor", type=int, default=DEFAULT_POOL_FACTOR,
        help="Advanced: explicit integer mean-pool factor along time. "
             "Default %(default)s = no pooling (save native 100 Hz). "
             "Use 10 to produce 10 Hz (same as --mean-pool); use 2 for the "
             "50 Hz rate VAP uses internally.",
    )

    # Runtime — batching + I/O
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Number of spliced wavs to run through CPC in one forward "
             "pass. Default %(default)s (tuned for M5 Max 32-core GPU). "
             "Drop to 1 for sequential/debug behavior; raise on larger "
             "CUDA GPUs. Memory scales linearly with B.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes for audio load + 48->16 kHz "
             "resample. Default %(default)s. Set to 0 to run everything "
             "in the main process (useful for debugging, slower in "
             "practice).",
    )
    parser.add_argument(
        "--writer-threads", type=int, default=DEFAULT_WRITER_THREADS,
        help="Threads in the .npy + .json background writer pool. "
             "Default %(default)s. Disk I/O on internal SSD rarely "
             "bottlenecks so this doesn't need tuning unless you're "
             "writing to a slow/networked drive.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress/rate/ETA line every N tasks. Default "
             "%(default)s. Set to 0 to silence progress lines.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device. 'auto' picks cuda > mps > cpu. Default: auto.",
    )

    # Selection / dry-run plumbing.
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract files that already have .npy + .json on disk. "
             "Default: skip-if-present so a killed run resumes cleanly.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate inputs/outputs without loading the model.",
    )

    args = parser.parse_args()

    # --- Validate --spliced-dir and --output-dir FIRST, before any other work.
    # These have to fail loudly and early: if either path is a typo or
    # points somewhere unintended, we don't want to have already resolved
    # a CUDA device or — worst — downloaded the CPC checkpoint from
    # torch.hub. Do the cheap filesystem checks first.
    if not os.path.isdir(args.spliced_dir):
        print(f"ERROR: --spliced-dir is not an existing directory: "
              f"{args.spliced_dir}", file=sys.stderr)
        return 1

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: --output-dir is not an existing directory: "
              f"{args.output_dir}\n"
              f"       This script intentionally does NOT auto-create the "
              f"output directory, to avoid writing features to an unintended "
              f"location on typos. Create it explicitly first:\n"
              f"           mkdir -p {args.output_dir!r}\n"
              f"       and re-run.", file=sys.stderr)
        return 1

    if args.batch_size < 1:
        print(f"ERROR: --batch-size must be >= 1; got {args.batch_size}",
              file=sys.stderr)
        return 1
    if args.num_workers < 0:
        print(f"ERROR: --num-workers must be >= 0; got {args.num_workers}",
              file=sys.stderr)
        return 1
    # Pool factor must be a positive integer. 0 or negative would
    # crash _mean_pool (ZeroDivisionError or reshape-with-negative-dim)
    # mid-run after the CPC model is already loaded and a real batch
    # is in flight; catch it up front so the user sees a clean error
    # and loses no work. Note: --mean-pool overrides --pool-factor via
    # the mutually-exclusive group, so we don't need to re-validate
    # the --mean-pool branch's derived value (which is always 10).
    if args.pool_factor < 1:
        print(f"ERROR: --pool-factor must be >= 1; got {args.pool_factor}",
              file=sys.stderr)
        return 1

    # Enforce that --cpc-repo and --checkpoint are paired when used.
    if bool(args.cpc_repo) != bool(args.checkpoint):
        print("ERROR: --cpc-repo and --checkpoint must be set together "
              "(local-load path) or neither (torch.hub path).",
              file=sys.stderr)
        return 1

    # --- Enumerate tasks -----------------------------------------------
    print(f"Scanning spliced dir: {args.spliced_dir}")
    tasks = build_tasks(args.spliced_dir, args.output_dir)
    n_subdirs = len({t.orig_stem for t in tasks})
    print(f"  {len(tasks):,} spliced wavs across {n_subdirs:,} original wavs")

    if not tasks:
        print("Nothing to extract. Exiting.")
        return 0

    # Resolve effective pool factor from the mutually-exclusive pool group.
    # --mean-pool is a convenience shortcut that overrides pool_factor to
    # the factor implied by DEFAULT_TARGET_RATE_HZ against the native 100 Hz.
    if args.mean_pool:
        effective_pool_factor = int(round(
            (CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME) / DEFAULT_TARGET_RATE_HZ
        ))
    else:
        effective_pool_factor = args.pool_factor

    # Derived output rate for logging / sanity.
    native_rate = CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME          # 100
    out_rate = native_rate / effective_pool_factor

    print(f"Output directory:  {args.output_dir}")
    print(f"Extraction config: feature_type={args.feature_type}  "
          f"pool_factor={effective_pool_factor} "
          f"({native_rate} Hz -> {out_rate:g} Hz)  "
          f"batch={args.batch_size}  workers={args.num_workers}  "
          f"writers={args.writer_threads}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")

    # --- Verify any existing sidecars in --output-dir were written with
    # the same model identity + extraction parameters as this run.
    # Mixing them would silently produce a heterogeneous corpus
    # (already_have would skip prior-params files despite the
    # mismatch). --overwrite bypasses this check since the user has
    # explicitly opted into discarding the prior extraction.
    if not args.overwrite:
        compat_err = check_output_dir_params_compat(
            output_dir=args.output_dir,
            feature_type=args.feature_type,
            pool_factor=effective_pool_factor,
            hub_repo=args.hub_repo,
            hub_entrypoint=args.hub_entrypoint,
            checkpoint=args.checkpoint,
        )
        if compat_err:
            print(f"ERROR: {compat_err}", file=sys.stderr)
            return 1

    # --- Load model (skipped on dry-run) --------------------------------
    model = None
    device = "cpu"
    feature_dim = CPC_FEATURE_DIM
    if not args.dry_run:
        device = pick_device(args.device)
        try:
            model, feature_dim = load_cpc(
                hub_repo=args.hub_repo,
                hub_entrypoint=args.hub_entrypoint,
                cpc_repo=args.cpc_repo,
                checkpoint=args.checkpoint,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: model load failed: {e}", file=sys.stderr)
            return 1

    print(f"Starting {len(tasks):,} extraction task(s)...\n")

    start = time.time()
    results = run_tasks(
        tasks=tasks,
        model=model,
        device=device,
        feature_type=args.feature_type,
        feature_dim=feature_dim,
        pool_factor=effective_pool_factor,
        hub_repo=args.hub_repo,
        hub_entrypoint=args.hub_entrypoint,
        checkpoint=args.checkpoint,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        writer_threads=args.writer_threads,
        progress_every=args.progress_every,
    )
    elapsed = time.time() - start

    print_summary(results, elapsed, args.dry_run)

    # Defensive against a None slipping through (shouldn't, since
    # run_tasks backfills in `finally`, but belt-and-suspenders). Treat
    # an unpopulated slot as failed so exit code reflects the problem.
    n_failed = sum(1 for r in results
                   if r is None or r["status"] == "failed")
    n_missing = sum(1 for r in results
                    if r is not None and r["status"] == "missing")
    if n_failed:
        return 2
    if n_missing:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
