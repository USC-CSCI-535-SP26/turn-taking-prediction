#!/usr/bin/env python3
"""
extract_cpc_from_manifest.py

Run Facebook Research CPC (Rivière et al. 2020, 256-dim @ 100 Hz) over every
per-participant .wav in a turn-taking manifest and save pooled features to a
flat output directory.

Repo link to clone if you are asked to do so: https://github.com/facebookresearch/CPC_audio.git
if local, path to this repo must be passed as --cpc-repo

CPC checkpoint download file to save if you are using the locally cloned repo: https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt
if using --cpc-repo flag, must also pass path to this CPC checkpoint local file using --checkpoint


Default layout
--------------
    turn_taking_analysis/
      manifests/manifest.csv              <-- input (or poc_manifest.csv)
      subset/
        audio/
          V00_S0743_I00000483_P0885.wav   <-- input (from download_audio_from_manifest.py)
          ...
        cpc_100hz/                         <-- output dir (REQUIRED, no default —
          V00_S0743_I00000483_P0885.npy      pass --output-dir explicitly). Shape:
          V00_S0743_I00000483_P0885.json     (T, 256) float32 at 100 Hz by default,
          ...                                or (T/10, 256) at 10 Hz with --mean-pool.

Both the 48-wav POC and the ~934-wav full project are supported by swapping
--manifest. --output-dir is REQUIRED (no default) and must point to an
existing directory — create it explicitly first with `mkdir -p`. The POC
invocation looks like

    mkdir -p ../subset/cpc_100hz
    python extract_cpc_from_manifest.py \\
        --manifest ../manifests/poc_manifest.csv \\
        --output-dir ../subset/cpc_100hz

The full-project invocation (using DEFAULT_MANIFEST_PATH) is

    mkdir -p ../subset/cpc_100hz
    python extract_cpc_from_manifest.py --output-dir ../subset/cpc_100hz

Why CPC?
--------
CPC is the canonical causal / non-bidirectional audio encoder used by the VAP
family (Ekstedt & Skantze 2022; Inoue et al. 2024; MM-VAP 2025). Its aggregator
is a strictly autoregressive RNN — unlike WavLM / HuBERT / wav2vec 2.0, the
feature at frame t is a function of audio only up to t. For a prediction-
horizon task (this POC's τ-curve), that property is load-bearing: with a
bidirectional encoder, features inside the [t − 2 s, t] input window can have
self-attended across the [t, t + τ] horizon and leaked the label. Running
CPC on the same wavs gives a feature stream whose causality is guaranteed
by construction, and a clean ablation target against WavLM-base+ features.

Full architecture / citations: docs/audio_encoder_research.md entry 1.

What the script does
--------------------
For each file_id in the manifest (both file_id_a and file_id_b):

  1. Load {audio_dir}/{file_id}.wav via torchaudio and resample to 16 kHz
     (CPC's training sample rate — Seamless ships 48 kHz mono wavs per
     paper §2.4.2, so no downmix is needed).
  2. Shape to (1, 1, T) as CPC expects (batch, channel=1, audio_samples).
  3. Run CPC forward once per file (or per chunk if --chunk-length-s is set).
     CPC returns (c_feature, z_feature, label) where:
        - c_feature is the aggregator (AR RNN) output — the "context" stream,
        - z_feature is the CNN-encoder-only output — the "encoded" stream.
     Both are 256-dim @ 100 Hz. We keep whichever --feature-type selects.
  4. Mean-pool every `--pool-factor` frames along time. Default pool_factor
     = 1 — i.e. **no pooling by default**; the saved features are the
     native 100 Hz CPC stream. Pass `--mean-pool` to pool to 10 Hz during
     extraction (matches the WavLM-script 50/5=10 Hz raster), or pass
     `--pool-factor N` for a custom integer factor.
  5. NaN/Inf guard: refuse to save poisoned features.
  6. Save {output_dir}/{file_id}.npy (float32, shape (T, 256)) and
     {output_dir}/{file_id}.json sidecar with full provenance.

Two-stage pooling workflow (recommended)
----------------------------------------
Default: extract once at native 100 Hz into a single folder. Later invoke
the `mean_pool_file(file, save_to, pool_to=10.0)` function on each .npy
to produce a separate pooled corpus — no CPC re-run required.

    # 1. Extract native 100 Hz. Output dir is REQUIRED and must exist.
    mkdir -p subset/cpc_100hz
    python extract_cpc_from_manifest.py --output-dir subset/cpc_100hz

    # 2. Later, pool to 10 Hz into a parallel directory (also must exist).
    mkdir -p subset/cpc_10hz
    from extract_cpc_from_manifest import mean_pool_file
    from pathlib import Path
    for npy in Path("subset/cpc_100hz").glob("*.npy"):
        mean_pool_file(str(npy), "subset/cpc_10hz", pool_to=10.0)

`mean_pool_file` reads each .npy plus its sibling sidecar (to learn the
current frame rate), validates that `current_rate / pool_to` is an integer,
applies the same reshape-mean used during extraction, and writes the pooled
.npy + an updated sidecar that records the pool step for audit.

Chunking
--------
CPC is ~5 M params (vs. WavLM-base+'s ~94 M) and runs comfortably on a T4.
For most Seamless interactions (< 10 min each) we process the whole file in
a single forward pass. If a file OOMs, pass --chunk-length-s to split the
waveform into fixed-length chunks and concatenate the outputs.

Because CPC is strictly causal, naive non-overlapping chunks introduce only
a brief RNN warm-up transient at each chunk boundary (< 100 frames at
100 Hz, so < 10 frames at the pooled 10 Hz rate). This is orders of
magnitude smaller than the 30-s-seam artefact WavLM's bidirectional
transformer would have produced — which is why this script does NOT do the
overlap-stitching dance extract_wavlm_from_manifest.py has. For the POC
τ-curve framing (±50 ms label tolerance at worst) the transient is
negligible. If you need to verify this empirically, run once with
--chunk-length-s 0 (whole-file, default) and once with --chunk-length-s N
and diff the feature arrays.

CPC model loading
-----------------
Loading is routed through torch.hub:

    torch.hub.load('facebookresearch/CPC_audio', 'CPC_default')

The hub entrypoint returns a tuple (model, hiddenGar, hiddenEncoder) —
hiddenGar is the aggregator (c_t) dim, hiddenEncoder is the CNN-encoder
(z_t) dim; for the Rivière 2020 baseline both are 256. We only need the
model itself. Override the entrypoint name with --hub-entrypoint if the
upstream repo renames it.

If torch.hub is blocked in your environment (no outbound GitHub access, etc.),
pass --cpc-repo /path/to/cloned/CPC_audio and --checkpoint /path/to/60k.pt
and the script will sys.path-insert the repo and load the checkpoint
directly. This mirrors how the Seamless repo is consumed elsewhere in
csci535-project.

Exit codes (mirror extract_wavlm_from_manifest.py)
--------------------------------------------------
    0 — all tasks succeeded (or skipped-as-already-extracted).
    1 — hard setup error (missing --output-dir, missing manifest,
        missing audio dir, missing --cpc-repo/--checkpoint pairing,
        model load failure).
    2 — one or more files failed extraction (see logs).
    3 — one or more input .wav files were missing on disk (non-fatal).

Usage
-----
    # --output-dir is REQUIRED and must already exist. Create it first.
    mkdir -p ../subset/cpc_100hz

    # Extract everything in the full manifest at native 100 Hz
    python extract_cpc_from_manifest.py --output-dir ../subset/cpc_100hz

    # POC manifest
    python extract_cpc_from_manifest.py \\
        --manifest ../manifests/poc_manifest.csv \\
        --output-dir ../subset/cpc_100hz

    # (Every invocation below also needs --output-dir — omitted here to keep
    #  the examples short. --output-dir must name an already-existing dir.)

    # Dry-run to enumerate what would be extracted
    python extract_cpc_from_manifest.py \\
        --output-dir ../subset/cpc_100hz --dry-run

    # Force re-extraction
    python extract_cpc_from_manifest.py \\
        --output-dir ../subset/cpc_100hz --overwrite

    # Chunk long files to cap VRAM
    python extract_cpc_from_manifest.py \\
        --output-dir ../subset/cpc_100hz --chunk-length-s 60

    # Use CNN-encoder output (z) instead of aggregator output (c)
    python extract_cpc_from_manifest.py \\
        --output-dir ../subset/cpc_100hz --feature-type encoded

    # Pool to 10 Hz during extraction (old default behavior).
    # Mutually exclusive with --pool-factor. Note the different output dir
    # so the 10 Hz corpus doesn't overwrite the 100 Hz one.
    python extract_cpc_from_manifest.py \\
        --output-dir ../subset/cpc_10hz --mean-pool

    # Custom integer pool factor (e.g., 2 → 50 Hz for VAP parity)
    python extract_cpc_from_manifest.py \\
        --output-dir ../subset/cpc_50hz --pool-factor 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
DEFAULT_MANIFEST_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "manifest.csv"
)
DEFAULT_AUDIO_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "audio"
)
# Intentional: --output-dir has NO default. Writing CPC features is
# high-volume and the "right" destination depends on whether you're
# producing the 100 Hz native corpus, a 10 Hz pooled corpus, or a
# one-off debug extraction. Forcing the caller to name the directory
# prevents typo-driven writes to an unexpected location.

# torch.hub entrypoint for the pretrained CPC checkpoint shipped with
# facebookresearch/CPC_audio. If the upstream repo renames the hubconf
# entrypoint, override with --hub-entrypoint.
DEFAULT_HUB_REPO = "facebookresearch/CPC_audio"
DEFAULT_HUB_ENTRYPOINT = "CPC_default"

DEFAULT_FEATURE_TYPE = "context"   # 'context' = aggregator output c_t (causal RNN output)
                                   # 'encoded' = CNN encoder output z_t (local-in-time)
DEFAULT_POOL_FACTOR = 1            # 1 = no pooling; save native 100 Hz features.
                                   # Use --mean-pool (→10) to pool during extraction,
                                   # or leave default and invoke mean_pool_file() later.
DEFAULT_TARGET_RATE_HZ = 10.0      # Default target rate for mean_pool_file() and --mean-pool.
DEFAULT_CHUNK_LENGTH_S = 0.0       # 0 = process full file in one forward pass

CPC_SAMPLE_RATE = 16_000           # CPC was trained on 16 kHz mono
CPC_SAMPLES_PER_FRAME = 160        # CPC's 5-layer CNN stride over 16 kHz → 100 Hz
CPC_FEATURE_DIM = 256              # Rivière 2020 baseline channel width


# =============================================================================
# Manifest parsing — identical contract to extract_wavlm_from_manifest.py
# =============================================================================

REQUIRED_COLUMNS = {
    "split",
    "interaction_id",
    "file_id_a",
    "file_id_b",
}


def load_manifest(manifest_path: str) -> list[dict]:
    """Load the manifest CSV into a list of dict rows and validate columns.

    Fails fast on schema drift rather than producing silently-wrong tasks.
    Works for any manifest that matches REQUIRED_COLUMNS (48-wav POC,
    467-interaction full project, or any subset).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run build_manifest.py first (or pass --manifest)."
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
# Extraction-task descriptor
# =============================================================================

class ExtractTask:
    """One wav to featurize: input path + destination path + bookkeeping tags."""

    __slots__ = (
        "split", "interaction_id", "participant_id",
        "file_id", "audio_path", "output_path", "sidecar_path",
    )

    def __init__(
        self,
        split: str,
        interaction_id: str,
        participant_id: str,
        file_id: str,
        audio_path: str,
        output_path: str,
        sidecar_path: str,
    ):
        self.split = split
        self.interaction_id = interaction_id
        self.participant_id = participant_id
        self.file_id = file_id
        self.audio_path = audio_path
        self.output_path = output_path
        self.sidecar_path = sidecar_path

    def __repr__(self) -> str:
        return f"ExtractTask({self.file_id})"


def build_tasks(
    rows: list[dict],
    audio_dir: str,
    output_dir: str,
    filter_split: str | None,
) -> list[ExtractTask]:
    """Expand manifest rows into one ExtractTask per participant wav.

    Input wavs follow download_audio_from_manifest.py's flat naming:
    {audio_dir}/{file_id}.wav. Outputs mirror that: {output_dir}/{file_id}.npy
    plus {file_id}.json.
    """
    tasks: list[ExtractTask] = []
    for row in rows:
        if filter_split and row["split"] != filter_split:
            continue

        for side in ("a", "b"):
            file_id = row[f"file_id_{side}"]
            if not file_id:
                print(f"  WARN: row {row['interaction_id']} has empty "
                      f"file_id_{side}; skipping.", file=sys.stderr)
                continue

            # Cross-check file_id's implied interaction against the row.
            # Same guard as extract_wavlm_from_manifest.py.
            derived_iid = extract_interaction_id(file_id)
            if derived_iid != row["interaction_id"]:
                print(f"  WARN: file_id {file_id} implies interaction "
                      f"{derived_iid} but manifest row says "
                      f"{row['interaction_id']}; using derived value.",
                      file=sys.stderr)

            audio_path = os.path.join(audio_dir, f"{file_id}.wav")
            output_path = os.path.join(output_dir, f"{file_id}.npy")
            sidecar_path = os.path.join(output_dir, f"{file_id}.json")

            tasks.append(ExtractTask(
                split=row["split"],
                interaction_id=derived_iid,
                participant_id=extract_participant_id(file_id),
                file_id=file_id,
                audio_path=audio_path,
                output_path=output_path,
                sidecar_path=sidecar_path,
            ))

    tasks.sort(key=lambda t: (t.split, t.interaction_id, t.file_id))
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
        we sys.path-insert the repo, import cpc.model.CPCModel (or the
        repo's equivalent), and load_state_dict from the checkpoint. Use
        this when outbound GitHub access is unavailable (CI, airgapped hosts).

    Returns the model in eval mode on `device`.
    """
    import torch  # noqa: PLC0415

    if cpc_repo and checkpoint:
        # Local path: sys.path insert the repo, import the model class,
        # build it with a default config, load the checkpoint.
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

        # The facebookresearch/CPC_audio repo ships a load helper at
        # cpc.feature_loader.loadModel. Import defensively — its exact
        # name can drift across commits.
        try:
            from cpc.feature_loader import loadModel  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                f"Could not import cpc.feature_loader.loadModel from "
                f"{repo_path}. Is {repo_path} actually a clone of "
                f"{DEFAULT_HUB_REPO}? ({e})"
            ) from e

        # loadModel returns (model, hiddenGar, hiddenEncoder).
        # hiddenGar  = aggregator output dim (= dim of c_t, the context stream).
        # hiddenEncoder = CNN encoder output dim (= dim of z_t, the encoded stream).
        # For the Rivière 2020 baseline, both are 256.
        model, hidden_context_dim, hidden_encoder_dim = loadModel([checkpoint])

    else:
        print(f"  Loading CPC:       torch.hub {hub_repo} :: {hub_entrypoint}")
        try:
            loaded = torch.hub.load(hub_repo, hub_entrypoint, trust_repo=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"torch.hub.load({hub_repo!r}, {hub_entrypoint!r}) failed: "
                f"{e}\n"
                f"Either fix your network / GitHub access, or clone the repo "
                f"locally and pass --cpc-repo /path/to/CPC_audio "
                f"--checkpoint /path/to/cpc_pretrained.pt."
            ) from e

        # Hub returns (model, hiddenGar, hiddenEncoder) per the upstream
        # hubconf. hiddenGar is the aggregator (c_t) dim; hiddenEncoder is
        # the CNN-encoder (z_t) dim. For the Rivière 2020 baseline, both
        # are 256. We also tolerate the degenerate 'just model' return for
        # forward-compat with older / custom hub entrypoints.
        if isinstance(loaded, tuple):
            model = loaded[0]
            hidden_context_dim = loaded[1] if len(loaded) > 1 else CPC_FEATURE_DIM
            hidden_encoder_dim = loaded[2] if len(loaded) > 2 else CPC_FEATURE_DIM
        else:
            model = loaded
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
# Per-file extraction
# =============================================================================

def _load_and_resample(audio_path: str):
    """Load a wav and resample to 16 kHz.

    Returns (waveform_1d_float32_cpu, original_sample_rate).

    Seamless ships per-participant mono wavs (one lapel mic per person,
    single-channel .wav; see Seamless paper §2.4.2), so no downmix is
    needed — torchaudio.load returns shape (1, T) directly. If you point
    this script at a non-Seamless dataset that ships stereo wavs, add a
    `waveform = waveform.mean(dim=0, keepdim=True)` here.

    CPC was trained on raw 16 kHz audio with no further normalization; we
    do NOT apply zero-mean/unit-var (WavLM's script does, but CPC's training
    recipe in facebookresearch/CPC_audio uses the raw-ish torchaudio.load
    output directly).
    """
    import torchaudio  # noqa: PLC0415

    waveform, sr = torchaudio.load(audio_path)  # (1, T) float32 in ~[-1, 1]

    if sr != CPC_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=CPC_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0), sr  # (T,), original_sr


def _cpc_forward(model, waveform_1d, device: str, feature_type: str):
    """Run CPC once over a 1-D waveform tensor. Return (T_frames, feature_dim) on CPU.

    The facebookresearch/CPC_audio model expects input of shape (B, 1, T_audio)
    and returns (c_feature, z_feature, label) — c is the aggregator (AR RNN)
    output, z is the CNN-encoder-only output, label is used only during
    training. Both features are 256-dim @ 100 Hz.
    """
    import torch  # noqa: PLC0415

    x = waveform_1d.to(device=device, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0)   # (1, 1, T_audio)

    with torch.inference_mode():
        out = model(x, None)

    # Handle both 2- and 3-tuple return shapes (old vs. new CPC_audio commits).
    if isinstance(out, tuple):
        if len(out) >= 2:
            c_feature, z_feature = out[0], out[1]
        else:
            # Single-tensor fallback — treat it as whichever feature was asked for.
            c_feature = z_feature = out[0]
    else:
        c_feature = z_feature = out

    chosen = c_feature if feature_type == "context" else z_feature
    # Shape from CPC: (1, T_frames, feature_dim). Squeeze batch.
    chosen = chosen.squeeze(0).to(torch.float32).cpu()
    return chosen


def _chunked_forward(
    model,
    waveform_1d,
    device: str,
    feature_type: str,
    chunk_samples: int,
):
    """Chunked forward for long waveforms. Concatenates per-chunk outputs.

    Because CPC is causal, chunking without state carry only introduces a
    brief RNN warm-up transient at each chunk boundary (< 1 s of frames)
    rather than the full-utterance contamination a bidirectional encoder
    would exhibit. For a 10 Hz downstream task this is negligible. If you
    need to verify empirically, compare against the no-chunk path (leave
    --chunk-length-s at 0).
    """
    import torch  # noqa: PLC0415

    n_samples = waveform_1d.shape[0]
    pieces = []
    start = 0
    while start < n_samples:
        end = min(n_samples, start + chunk_samples)
        piece = _cpc_forward(
            model, waveform_1d[start:end], device, feature_type
        )
        pieces.append(piece)
        start = end

    if not pieces:
        return torch.zeros((0, CPC_FEATURE_DIM), dtype=torch.float32)
    return torch.cat(pieces, dim=0)


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


def extract_one(
    task: ExtractTask,
    model,
    device: str,
    feature_type: str,
    feature_dim: int,
    pool_factor: int,
    chunk_samples: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
) -> dict:
    """Run CPC over one wav end-to-end and write {file_id}.npy + .json.

    Returns {task, status, message} with status in
    {"ok", "missing", "skipped", "failed"} — same vocabulary as
    extract_wavlm_from_manifest.py for a readable summary line.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not os.path.exists(task.audio_path):
        return {
            "task": task, "status": "missing",
            "message": f"input wav not on disk: {task.audio_path}",
        }

    try:
        waveform, orig_sr = _load_and_resample(task.audio_path)
    except Exception as e:  # noqa: BLE001
        return {"task": task, "status": "failed",
                "message": f"audio load failed: {e}"}

    n_samples = waveform.shape[0]
    duration_s = n_samples / CPC_SAMPLE_RATE

    try:
        if chunk_samples > 0 and n_samples > chunk_samples:
            features = _chunked_forward(
                model, waveform, device, feature_type, chunk_samples)
        else:
            features = _cpc_forward(model, waveform, device, feature_type)
    except RuntimeError as e:
        # Most common cause: CUDA OOM on a long file. Tell the user to retry
        # with --chunk-length-s rather than silently degrading.
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        is_oom = "out of memory" in str(e).lower()
        hint = (f" — retry with --chunk-length-s <seconds> "
                f"(e.g. 60)" if is_oom else "")
        return {
            "task": task, "status": "failed",
            "message": f"CPC forward failed ({type(e).__name__}: {e}){hint}",
        }

    # NaN / Inf guard. CPC in fp32 rarely produces non-finite values, but
    # adversarial inputs (zero-energy clips, NaN wavs) can.
    if not torch.isfinite(features).all():
        nan_count = int(torch.isnan(features).sum())
        inf_count = int(torch.isinf(features).sum())
        return {
            "task": task, "status": "failed",
            "message": (f"non-finite features: nan={nan_count:,} "
                        f"inf={inf_count:,} of {features.numel():,}"),
        }

    pooled = _mean_pool(features, pool_factor).numpy().astype(np.float32)

    try:
        os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
        np.save(task.output_path, pooled)

        # Sidecar schema mirrors extract_wavlm_from_manifest.py field-for-field
        # where meaningful, so downstream consumers can treat WavLM and CPC
        # features interchangeably (modulo feature_dim). CPC-specific fields
        # are prefixed with `cpc_` for clarity.
        sidecar = {
            "file_id":                  task.file_id,
            "interaction_id":           task.interaction_id,
            "participant_id":           task.participant_id,
            "split":                    task.split,
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
            # Chunking provenance (0 = whole-file forward).
            "chunk_samples":            chunk_samples,
            "chunk_length_s":           (chunk_samples / CPC_SAMPLE_RATE
                                         if chunk_samples else 0.0),
        }
        with open(task.sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

    except Exception as e:  # noqa: BLE001
        return {"task": task, "status": "failed",
                "message": f"save failed: {e}"}

    return {
        "task": task, "status": "ok",
        "message": f"OK ({pooled.shape[0]:,}×{pooled.shape[1]}, "
                   f"{duration_s:.1f}s -> {pooled.nbytes:,} bytes)",
    }


# =============================================================================
# Orchestration
# =============================================================================

def already_have(output_path: str, sidecar_path: str) -> bool:
    """True iff both .npy and .json exist and parse — idempotent resume.

    A 0-byte .npy from a crashed prior run is treated as missing so the
    next run re-extracts it. Truncated sidecars (killed mid-write) fail
    json.load and are also re-run.
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


def run_tasks(
    tasks: list[ExtractTask],
    model,
    device: str,
    feature_type: str,
    feature_dim: int,
    pool_factor: int,
    chunk_samples: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
    overwrite: bool,
    dry_run: bool,
) -> list[dict]:
    """Sequential single-GPU task runner — same structure as the WavLM script."""
    results: list[dict] = []

    if tasks and not dry_run:
        os.makedirs(os.path.dirname(tasks[0].output_path), exist_ok=True)

    for task in tasks:
        if not overwrite and already_have(task.output_path, task.sidecar_path):
            result = {
                "task": task, "status": "skipped",
                "message": f"already extracted "
                           f"({os.path.getsize(task.output_path):,} bytes)",
            }
        elif dry_run:
            result = {"task": task, "status": "dry",
                      "message": f"DRY: {task.audio_path} -> {task.output_path}"}
        else:
            result = extract_one(
                task=task,
                model=model,
                device=device,
                feature_type=feature_type,
                feature_dim=feature_dim,
                pool_factor=pool_factor,
                chunk_samples=chunk_samples,
                hub_repo=hub_repo,
                hub_entrypoint=hub_entrypoint,
                checkpoint=checkpoint,
            )

        print(f"  [{result['status']:<7}] {task.split:<5} "
              f"{task.file_id}  {result['message']}")
        results.append(result)

    return results


def print_summary(results: list[dict], elapsed_s: float, dry_run: bool) -> None:
    """Counts by status + total bytes written."""
    counts: dict[str, int] = {}
    bytes_written = 0
    for r in results:
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
            "from every participant wav in a turn-taking manifest. "
            "Saves native 100 Hz by default; pass --mean-pool (or "
            "--pool-factor N) to pool during extraction, or leave default "
            "and invoke mean_pool_file() post-hoc to produce a pooled "
            "corpus. Mirrors extract_wavlm_from_manifest.py's CLI and "
            "sidecar schema so the two feature streams are drop-in-"
            "swappable downstream."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O paths — identical surface to the WavLM script so `--manifest
    # ../manifests/poc_manifest.csv` works the same way on both.
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST_PATH,
        help="Path to the manifest CSV. Default: %(default)s",
    )
    parser.add_argument(
        "--audio-dir", default=DEFAULT_AUDIO_DIR,
        help="Directory containing input .wav files. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="REQUIRED. Directory to save .npy / .json feature files. "
             "Must already exist on disk (the script will not auto-create it "
             "— this is a footgun guard against typos that would otherwise "
             "silently land features in an unexpected location). Create it "
             "explicitly first: `mkdir -p subset/cpc_100hz` etc.",
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
    # Derive the --mean-pool shortcut factor from the rate constants so the
    # help text stays in sync with any future change to CPC_SAMPLES_PER_FRAME
    # or DEFAULT_TARGET_RATE_HZ (rather than hard-coding 100 / 10 = 10 here).
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
    parser.add_argument(
        "--chunk-length-s", type=float, default=DEFAULT_CHUNK_LENGTH_S,
        help="If > 0, split long waveforms into chunks of this many seconds "
             "before the forward pass (concatenating outputs). Default 0 = "
             "process whole file in one pass. Raise to 30-60 if a file OOMs.",
    )

    # Runtime
    parser.add_argument(
        "--device", default="auto",
        help="Torch device. 'auto' picks cuda > mps > cpu. Default: auto.",
    )

    # Selection / dry-run plumbing — identical surface to the WavLM script.
    parser.add_argument(
        "--filter-split", choices=["train", "val", "test"], default=None,
        help="If set, only process rows whose manifest `split` column matches.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract files that already have .npy + .json on disk.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate inputs/outputs without loading the model.",
    )

    args = parser.parse_args()

    # --- Validate --output-dir FIRST, before any other work. -----------
    # This has to fail loudly and early: if the path is a typo or points
    # somewhere unintended, we don't want to have already loaded the
    # manifest, resolved a CUDA device, or — worst — downloaded the CPC
    # checkpoint from torch.hub. Do the cheap filesystem check first.
    if not os.path.isdir(args.output_dir):
        print(f"ERROR: --output-dir is not an existing directory: "
              f"{args.output_dir}\n"
              f"       This script intentionally does NOT auto-create the "
              f"output directory, to avoid writing features to an unintended "
              f"location on typos. Create it explicitly first:\n"
              f"           mkdir -p {args.output_dir!r}\n"
              f"       and re-run.", file=sys.stderr)
        return 1

    # Enforce that --cpc-repo and --checkpoint are paired when used.
    if bool(args.cpc_repo) != bool(args.checkpoint):
        print("ERROR: --cpc-repo and --checkpoint must be set together "
              "(local-load path) or neither (torch.hub path).",
              file=sys.stderr)
        return 1

    # --- Load manifest --------------------------------------------------
    print(f"Loading manifest:  {args.manifest}")
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  {len(rows):,} interactions in manifest "
          f"({2 * len(rows):,} participant wavs)")

    if not os.path.isdir(args.audio_dir):
        print(f"ERROR: audio directory not found: {args.audio_dir}\n"
              f"       Run download_audio_from_manifest.py first, or "
              f"pass --audio-dir.", file=sys.stderr)
        return 1

    tasks = build_tasks(
        rows=rows,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        filter_split=args.filter_split,
    )
    if args.filter_split:
        print(f"  Filtered to split='{args.filter_split}': "
              f"{len(tasks):,} tasks")
    if not tasks:
        print("Nothing to extract. Exiting.")
        return 0

    chunk_samples = int(args.chunk_length_s * CPC_SAMPLE_RATE)

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

    print(f"Audio directory:   {args.audio_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Extraction config: feature_type={args.feature_type}  "
          f"pool_factor={effective_pool_factor} "
          f"({native_rate} Hz -> {out_rate:g} Hz)  "
          f"chunk={'whole-file' if chunk_samples == 0 else f'{args.chunk_length_s}s'}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")

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
        chunk_samples=chunk_samples,
        hub_repo=args.hub_repo,
        hub_entrypoint=args.hub_entrypoint,
        checkpoint=args.checkpoint,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    elapsed = time.time() - start

    print_summary(results, elapsed, args.dry_run)

    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_missing = sum(1 for r in results if r["status"] == "missing")
    if n_failed:
        return 2
    if n_missing:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
