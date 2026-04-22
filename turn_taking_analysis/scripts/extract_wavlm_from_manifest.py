#!/usr/bin/env python3
"""
extract_wavlm_from_manifest.py

Run WavLM-base+ over every per-participant .wav in a turn-taking
manifest and save pooled hidden-state features under a flat directory.

Default layout
--------------
    turn_taking_analysis/
      manifests/manifest.csv         <-- input
      subset/
        audio/
          V00_S0743_I00000483_P0885.wav  <-- input (from download_audio_from_manifest.py)
          ...
        wavlm/
          V00_S0743_I00000483_P0885.npy  <-- output: (T_10hz, 768) float32
          V00_S0743_I00000483_P0885.json <-- sidecar: extraction metadata
          ...

Scope
-----
Originally written for the 24-dyad POC (48 wavs, runs in ~2 min on T4).
The project has since scaled up to all 467 naturalistic
ipc_conversation interactions (~934 wavs, ~5-8 hours on T4). The
defaults here still target the POC paths; override `--manifest` and
`--audio-dir` / `--output-dir` for a full-scale run. Everything below
is designed to survive a multi-hour run (NaN detection, OOM-aware
retry with chunk halving, autocast rather than in-place fp16).

What the script does
--------------------
For each file_id in the manifest (both `file_id_a` and `file_id_b`):

  1. Load {audio_dir}/{file_id}.wav via torchaudio, downmix to mono,
     resample to 16 kHz (WavLM's training sample rate — Seamless ships
     48 kHz wavs per paper §3.4).
  2. Chunk the waveform into OVERLAPPING windows of `--chunk-length-s`
     seconds with `--context-s` of audio margin on each side. Each
     forward pass sees `chunk_length_s` seconds of audio, but only the
     middle `chunk_length_s - 2 * context_s` seconds of output frames is
     retained — the margin frames are thrown away after feeding
     self-attention. This eliminates the 30-s-seam artifact that a
     naive non-overlapping chunker produces (WavLM's transformer has no
     cross-chunk attention, so edge-of-chunk frames are unreliable).
  3. For each chunk, run the model with `output_hidden_states=True` and
     pick `--layer` (default -2 = penultimate transformer layer).
     Rationale for penultimate: Chen et al. (2022) and the SUPERB
     benchmark show that for WavLM the penultimate layer carries
     prosody / speaker info most relevant to paralinguistic tasks — the
     final layer is more biased toward phonetic masking pretraining.
  4. Concatenate the kept (core) per-chunk outputs along time -> (T_50hz, 768).
  5. Mean-pool every `--pool-factor` frames (default 5) to downsample
     50 Hz -> 10 Hz, matching the downstream alignment rate used by
     OpenFace AU resampling and the turn-taking label rasterization.
     The tail is truncated if T_50hz is not divisible by pool_factor
     (<= pool_factor - 1 frames dropped; at 10 Hz this is <= 80 ms).
  6. NaN/Inf guard: if the extracted features contain any non-finite
     values (fp16 can overflow on pathological inputs), the whole file
     is marked "failed" rather than silently saving poisoned features
     downstream.
  7. Save the (T_10hz, 768) float32 array as {output_dir}/{file_id}.npy
     and a sidecar {file_id}.json with extraction metadata including
     provenance of the overlap-stitching configuration.

Reuse of shared helpers
-----------------------
This script is a feature-extraction layer, not a downloader, so the S3
helpers from csci535-project/scripts/download_annotated_interactions.py
aren't directly applicable. What IS reused:

    extract_interaction_id()  — for cross-checking manifest consistency
    extract_participant_id()  — for logging / derived bookkeeping

The overall scaffold (manifest loader with REQUIRED_COLUMNS validation,
task-per-file structure, status-coded result dict, per-line progress
output, tallied summary, exit-code conventions) mirrors the sibling
turn_taking_analysis/scripts/download_audio_from_manifest.py so this
file reads as part of the same family.

Why WavLM-base+ (and not HuBERT-base)
-------------------------------------
WavLM-base+ is pretrained on ~94k hours of mixed speech data including
the GigaSpeech / VoxPopuli conversational corpora and uses the gated
relative-position bias ("masked speech prediction with denoising") that
makes it more robust on conversational / noisy speech than HuBERT-base
(which is pretrained on LibriSpeech read speech, Librilight). For
turn-taking — a paralinguistic, prosody-heavy task on conversational
dyads — WavLM-base+ is the better-matched encoder. Same parameter count
(~94M) and same 50 Hz frame rate as HuBERT-base, so downstream pipeline
dims and rates are unchanged.

fp16 / mixed-precision handling
-------------------------------
On CUDA we use `torch.amp.autocast('cuda', dtype=torch.float16)` around
each forward pass rather than `model.half()`. autocast keeps numerically
sensitive ops (layer norms, softmax reductions, variance computations)
in fp32 while matmuls run in fp16 — same throughput as full-fp16 but
less prone to NaN/Inf explosions. The `--no-fp16` flag disables
autocast and runs everything in fp32.

After extraction we explicitly check `torch.isfinite(features).all()`.
If non-finite values leaked through (very rare with autocast; can happen
on adversarially silent or clipped inputs), the file is marked "failed"
rather than written to disk. Downstream consumers never see NaN features.

OOM-aware retry
---------------
If a forward pass raises CUDA OutOfMemoryError (or an equivalent
RuntimeError), the script:
  - Calls `torch.cuda.empty_cache()` to free fragmented allocator state
  - Halves `chunk_samples` for THIS file only and re-runs the entire
    streaming extraction from scratch
  - Caps OOM halving at `--max-oom-fallbacks` (default 3) — below that,
    the file is marked "failed" rather than halving further, since too-
    small chunks lose so much attention context that feature quality
    degrades anyway.
A small fixed retry budget (`--max-retries`) also handles transient
non-OOM RuntimeErrors (e.g., transient CUDA launch failures).

Usage
-----
    # Default: extract everything in the manifest to .../subset/wavlm/
    python extract_wavlm_from_manifest.py

    # Only val split first (staged debugging)
    python extract_wavlm_from_manifest.py --filter-split val

    # Dry-run: list inputs+outputs without loading the model
    python extract_wavlm_from_manifest.py --dry-run

    # Force re-extraction
    python extract_wavlm_from_manifest.py --overwrite

    # Use a different WavLM checkpoint / layer / pool
    python extract_wavlm_from_manifest.py \\
        --model microsoft/wavlm-base-plus \\
        --layer -2 \\
        --pool-factor 5

    # Tune context margin (audio seconds fed to attention on each side
    # of the kept core). Higher = better edge quality, slightly more compute.
    python extract_wavlm_from_manifest.py --context-s 3

Compute notes
-------------
- Defaults target Colab Pro T4 (16 GB VRAM).
- autocast fp16 on CUDA; fp32 everywhere else.
- At full-project scale (~934 files): ~5-8 hours on T4.
- The overlap-stitching adds ~15% compute vs. non-overlapping chunks
  (chunk_len / (chunk_len - 2*context_s) = 30 / 26 ≈ 1.15) in exchange
  for eliminating the 30-s-seam feature artifact.

Exit codes
----------
    0 — all tasks succeeded (or skipped-as-already-extracted).
    1 — hard setup error (missing manifest, missing audio dir, etc.).
    2 — one or more files failed extraction (see logs for which).
    3 — one or more input .wav files were missing on disk (non-fatal).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# -----------------------------------------------------------------------------
# Re-use helpers from csci535-project/scripts. This script lives at
# turn_taking_analysis/scripts/, so we resolve the sibling directory and
# insert it on sys.path — matching build_poc_manifest.py and the sibling
# download_audio_from_manifest.py / download_video_from_manifest.py.
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
DEFAULT_OUTPUT_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "wavlm"
)

DEFAULT_MODEL_ID = "microsoft/wavlm-base-plus"
DEFAULT_LAYER = -2                # penultimate transformer layer
DEFAULT_POOL_FACTOR = 5           # 50 Hz -> 10 Hz
DEFAULT_CHUNK_LENGTH_S = 30.0     # seconds of audio per forward pass (= attention window)
DEFAULT_CONTEXT_S = 2.0           # seconds of audio margin on EACH side of the kept core
DEFAULT_MAX_RETRIES = 3           # non-OOM transient retries per file
DEFAULT_MAX_OOM_FALLBACKS = 3     # how many times to halve chunk_samples on OOM

WAVLM_SAMPLE_RATE = 16_000        # WavLM expects 16 kHz mono

# WavLM's CNN feature extractor has stride 320 over the raw 16 kHz audio,
# which means one hidden-state frame per 320 audio samples (= 50 Hz). Used
# to convert between audio-sample positions and feature-frame positions
# when deciding how many margin frames to discard from each chunk.
SAMPLES_PER_FRAME = 320


# =============================================================================
# Manifest parsing
# =============================================================================

# We only need the file_ids + split tags. label / seamless_split /
# interaction_id are kept so per-line logs stay readable and so a future
# split-stratified extraction order remains trivial to add.
REQUIRED_COLUMNS = {
    "split",              # our turn-taking split: train/val/test
    "interaction_id",     # for logging / cross-check
    "file_id_a",
    "file_id_b",
}


def load_manifest(manifest_path: str) -> list[dict]:
    """Load the turn-taking manifest CSV into a list of dict rows and
    validate columns.

    Fails fast on schema drift rather than producing silently-wrong
    tasks. Works for any manifest that matches the REQUIRED_COLUMNS
    contract (24-dyad POC, 467-interaction full project, or any subset).
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
# Extraction-task descriptor
# =============================================================================

class ExtractTask:
    """One wav to featurize: input path + destination path + bookkeeping tags.

    Storing split / interaction_id / participant_id on the task keeps the
    per-line progress output readable without re-deriving anything from
    the file path mid-loop.
    """

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

    The input .wav path follows the flat naming convention established
    by download_audio_from_manifest.py: {audio_dir}/{file_id}.wav.
    The output path mirrors that: {output_dir}/{file_id}.npy (+ .json).
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

            # Cross-check that the file_id's implied interaction matches the
            # manifest row. Guards against accidental manifest edits —
            # identical to the check in download_audio_from_manifest.py.
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
# Model loading (lazy — we import torch / transformers only when needed so
# --dry-run doesn't require the full ML stack to be installed).
# =============================================================================

def pick_device(requested: str) -> str:
    """Resolve a --device argument to a concrete torch device string.

    'auto' prefers CUDA -> MPS -> CPU. Any explicit value is returned
    as-is (letting torch itself raise if the value is invalid).
    """
    if requested != "auto":
        return requested

    # Lazy import — we don't want torch just for the CLI help text.
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_wavlm(model_id: str, device: str, use_fp16: bool):
    """Load WavLM-base+ once, return (model, feature_extractor, use_autocast_fp16).

    The feature extractor is used only to know the expected sample rate
    and normalization convention; we bypass its Python-side chunking
    and feed raw tensors directly to the model.

    Mixed-precision handling: on CUDA with use_fp16=True we keep the
    MODEL in fp32 and enable autocast in the forward loop. This is
    safer than calling `model.half()` (which casts layer norms,
    softmaxes, and variance reductions to fp16 and can produce NaN/Inf
    on adversarial inputs). MPS and CPU stay fp32 — fp16 on CPU is
    slower than fp32 on most CPUs, and MPS fp16 has had correctness
    issues historically.
    """
    # Lazy import so --dry-run / --help don't need transformers.
    import torch  # noqa: PLC0415
    from transformers import AutoFeatureExtractor, WavLMModel  # noqa: PLC0415

    print(f"  Loading model:     {model_id}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMModel.from_pretrained(model_id)
    model.eval()
    model.to(device)

    # We no longer call model.half() even on CUDA — autocast around the
    # forward pass is strictly better (same throughput, better numerics).
    # `use_autocast_fp16` is a simple Boolean flag consumed in extract_one.
    use_autocast_fp16 = bool(use_fp16 and device == "cuda")
    if use_autocast_fp16:
        print(f"  Precision:         autocast fp16 (cuda); model weights remain fp32")
    else:
        print(f"  Precision:         fp32 ({device})")

    if feature_extractor.sampling_rate != WAVLM_SAMPLE_RATE:
        # Defensive: if HF ever ships a 24 kHz WavLM variant, we want to
        # hear about it rather than silently resampling to the wrong rate.
        raise RuntimeError(
            f"Feature extractor expects {feature_extractor.sampling_rate} Hz "
            f"but this script assumes {WAVLM_SAMPLE_RATE} Hz. Update "
            f"WAVLM_SAMPLE_RATE or swap models."
        )

    return model, feature_extractor, use_autocast_fp16


# =============================================================================
# Per-file extraction
# =============================================================================

def _load_and_resample(audio_path: str):
    """Load a wav, downmix to mono, resample to 16 kHz. Returns a 1-D
    float32 torch.Tensor on CPU.

    Seamless wavs are 48 kHz mono per paper §3.4, but we don't assume —
    we check channels and sample rate and handle both branches.
    """
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    waveform, sr = torchaudio.load(audio_path)  # (C, T), float32
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != WAVLM_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=WAVLM_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # WavLM expects zero-mean / unit-variance input. HF's feature
    # extractor applies this per-input tensor; we apply it once to the
    # whole waveform here for efficiency. For stationary speech the two
    # are numerically equivalent; for a file with a long silent prefix
    # or a mid-recording loudness jump they can disagree slightly.
    waveform = waveform.squeeze(0)  # (T,)
    waveform = waveform - waveform.mean()
    # Avoid div-by-zero on pathological silent clips.
    std = waveform.std().clamp(min=1e-7)
    waveform = waveform / std

    return waveform, sr


def _chunk_with_context(waveform, chunk_idx, chunk_samples, context_samples):
    """Build one overlapping chunk for feeding the transformer.

    Yields a tuple of (chunk_tensor, left_discard_frames, right_discard_frames)
    where:
      - chunk_tensor is the audio slice the model will see (up to
        chunk_samples long; possibly shorter at file edges).
      - left_discard_frames is how many output frames at the start of
        the hidden-state sequence correspond to the left-context region
        and therefore should be DROPPED from the retained stream.
      - right_discard_frames is the same for the right side.

    The "core" region (what's kept) is the middle
        core_samples = chunk_samples - 2 * context_samples
    seconds of audio. core positions are laid out at
        core_start_audio = chunk_idx * core_samples
    so consecutive chunks' cores abut perfectly with no overlap.

    Edge chunks are handled naturally: if core_start_audio - context_samples
    would be negative (chunk 0), we clamp audio_start to 0 and record
    left_discard_frames = 0 (no left context to throw away). Symmetric
    at the end of the file.

    Returns None if this chunk_idx is past the end of the audio.
    """
    n_samples = waveform.shape[0]
    core_samples = chunk_samples - 2 * context_samples
    if core_samples <= 0:
        # Pathological config: context_samples too large. Fall back to
        # no-context chunking (equivalent to the pre-overlap version).
        core_samples = chunk_samples
        context_samples = 0

    core_start = chunk_idx * core_samples
    if core_start >= n_samples:
        return None  # past end of file

    audio_start = max(0, core_start - context_samples)
    audio_end = min(n_samples, core_start + core_samples + context_samples)
    chunk = waveform[audio_start:audio_end]

    # Margin-frame counts. Each feature frame corresponds to
    # SAMPLES_PER_FRAME (= 320) audio samples at 50 Hz.
    left_margin_samples = core_start - audio_start
    right_margin_samples = audio_end - (core_start + core_samples)
    left_discard_frames = left_margin_samples // SAMPLES_PER_FRAME
    right_discard_frames = max(0, right_margin_samples // SAMPLES_PER_FRAME)

    return chunk, left_discard_frames, right_discard_frames


def _streaming_forward(
    waveform,
    model,
    device: str,
    use_autocast_fp16: bool,
    layer: int,
    chunk_samples: int,
    context_samples: int,
):
    """Streamed overlap-stitched forward over a full waveform.

    For each chunk:
      1. Slice audio as (left_context + core + right_context).
      2. Run WavLM; grab `layer` hidden states.
      3. Discard the first `left_discard_frames` and last
         `right_discard_frames` rows — those correspond to the margin
         regions that existed only to feed attention.
    Concatenate the kept (core-only) outputs across chunks. Returns a
    (T_50hz, feat_dim) float32 tensor on CPU.
    """
    import torch  # noqa: PLC0415

    n_samples = waveform.shape[0]
    core_samples = chunk_samples - 2 * context_samples
    if core_samples <= 0:
        core_samples = chunk_samples

    # Number of chunks needed to cover every core region.
    num_chunks = max(1, (n_samples + core_samples - 1) // core_samples)

    hidden_stream = []
    with torch.inference_mode():
        for i in range(num_chunks):
            built = _chunk_with_context(
                waveform, i, chunk_samples, context_samples)
            if built is None:
                break
            chunk, left_drop, right_drop = built
            chunk_in = chunk.to(device=device, dtype=torch.float32).unsqueeze(0)

            if use_autocast_fp16:
                # autocast keeps layer norms / softmax reductions in fp32
                # while matmuls run in fp16. Strictly safer than model.half().
                with torch.amp.autocast(device_type="cuda",
                                        dtype=torch.float16):
                    out = model(chunk_in, output_hidden_states=True)
            else:
                out = model(chunk_in, output_hidden_states=True)

            # hidden_states: tuple of (num_transformer_layers + 1);
            # index 0 = CNN output embeddings, 1..N = transformer layer
            # outputs. layer=-2 therefore selects the penultimate
            # transformer layer's output.
            hs = out.hidden_states[layer]            # (1, T_frame, 768)
            hs = hs.squeeze(0).to(torch.float32).cpu()

            # Discard margin frames; keep only the core.
            n_frames = hs.shape[0]
            keep_end = n_frames - right_drop
            if keep_end <= left_drop:
                # Degenerate chunk (shorter than advertised margins);
                # keep everything we have rather than emit nothing.
                kept = hs
            else:
                kept = hs[left_drop:keep_end]
            hidden_stream.append(kept)

    if not hidden_stream:
        # Empty audio — return an empty (0, feat_dim) tensor so the
        # pooling step produces (0, D) cleanly.
        return torch.zeros((0, 768), dtype=torch.float32)
    return torch.cat(hidden_stream, dim=0)


def _mean_pool(features, pool_factor: int):
    """Mean-pool along the time dimension by `pool_factor`.

    Input: (T, D) tensor. Output: (T // pool_factor, D).
    Tail frames that don't complete a pooling window are truncated.
    """
    import torch  # noqa: PLC0415

    t, d = features.shape
    keep = (t // pool_factor) * pool_factor
    if keep == 0:
        # Edge case: clip is shorter than a single pooling window.
        return torch.zeros((0, d), dtype=features.dtype, device=features.device)
    trimmed = features[:keep]
    pooled = trimmed.reshape(keep // pool_factor, pool_factor, d).mean(dim=1)
    return pooled


def _is_cuda_oom(err: BaseException) -> bool:
    """Classify an exception as CUDA OOM.

    PyTorch 2.1+ raises `torch.cuda.OutOfMemoryError`; earlier versions
    raise a plain RuntimeError with 'out of memory' in the message.
    We cover both to stay robust across PyTorch versions.
    """
    try:
        import torch  # noqa: PLC0415
        if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
                err, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def extract_one(
    task: ExtractTask,
    model,
    device: str,
    use_autocast_fp16: bool,
    layer: int,
    pool_factor: int,
    chunk_samples: int,
    context_samples: int,
    max_retries: int,
    max_oom_fallbacks: int,
) -> dict:
    """Run WavLM over one wav end-to-end and write {file_id}.npy + .json.

    Returns a result dict with keys: task, status, message. Status codes
    match the sibling download script for readability of the summary line:

        "ok"       — features extracted and saved.
        "missing"  — input .wav not on disk.
        "skipped"  — handled at the caller via already_have().
        "failed"   — model / IO error (after retries); message carries detail.

    Robustness:
      - On CUDA OOM, calls `torch.cuda.empty_cache()`, halves
        chunk_samples for THIS file only, and retries the streaming
        extraction from scratch. Up to `max_oom_fallbacks` times.
      - Non-OOM transient errors (rare CUDA-launch failures etc.) are
        retried up to `max_retries` times with a 1-second sleep.
      - After extraction, checks `isfinite(features).all()` and fails
        the file rather than saving NaN/Inf features to disk.
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
    duration_s = n_samples / WAVLM_SAMPLE_RATE

    # --- Streaming extraction with OOM-aware retry ----------------------
    #
    # Design: the happy path runs once through _streaming_forward with
    # the configured chunk / context samples. Any CUDA OOM halves the
    # chunk_samples and retries the whole file; we cap the halving at
    # max_oom_fallbacks to avoid unbounded degradation. Other transient
    # RuntimeErrors get a small fixed retry budget with a 1s sleep.
    #
    current_chunk_samples = chunk_samples
    current_context_samples = context_samples
    oom_fallbacks = 0
    retries = 0
    features = None
    last_err_msg = None
    while True:
        try:
            features = _streaming_forward(
                waveform=waveform,
                model=model,
                device=device,
                use_autocast_fp16=use_autocast_fp16,
                layer=layer,
                chunk_samples=current_chunk_samples,
                context_samples=current_context_samples,
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err_msg = f"{type(e).__name__}: {e}"
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

            if _is_cuda_oom(e):
                # Halve the chunk and retry the whole file. Preserve the
                # (context ≤ chunk/4) ratio so chunks stay well-shaped.
                halved = current_chunk_samples // 2
                if (oom_fallbacks < max_oom_fallbacks
                        and halved > 2 * current_context_samples):
                    oom_fallbacks += 1
                    current_chunk_samples = halved
                    current_context_samples = min(
                        current_context_samples, halved // 4)
                    continue
                # Can't halve further — give up.
                return {
                    "task": task, "status": "failed",
                    "message": (f"CUDA OOM after {oom_fallbacks} chunk halving(s); "
                                f"current chunk_samples={current_chunk_samples}; "
                                f"{last_err_msg}"),
                }

            # Non-OOM transient: retry with a brief sleep.
            retries += 1
            if retries > max_retries:
                return {
                    "task": task, "status": "failed",
                    "message": (f"forward failed after {retries-1} retries "
                                f"(oom_fallbacks={oom_fallbacks}): {last_err_msg}"),
                }
            time.sleep(1.0)

    # --- NaN / Inf guard ------------------------------------------------
    #
    # fp16 paths can produce non-finite values on adversarial inputs
    # (extended silence with near-zero std, clipped audio, etc.). We
    # refuse to write those features and flag the file "failed" so the
    # user can re-run with --no-fp16 or drop that file. Cheap check —
    # scans the whole tensor once.
    if not torch.isfinite(features).all():
        nan_count = int(torch.isnan(features).sum())
        inf_count = int(torch.isinf(features).sum())
        total = features.numel()
        return {
            "task": task, "status": "failed",
            "message": (f"non-finite features in output: "
                        f"nan={nan_count:,} inf={inf_count:,} of {total:,}; "
                        f"try --no-fp16 for this file"),
        }

    # --- Pool 50 Hz -> 10 Hz and save ----------------------------------
    pooled = _mean_pool(features, pool_factor).numpy().astype(np.float32)

    try:
        os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
        np.save(task.output_path, pooled)

        sidecar = {
            "file_id":                  task.file_id,
            "interaction_id":           task.interaction_id,
            "participant_id":           task.participant_id,
            "split":                    task.split,
            "audio_source":             task.audio_path,
            "source_sample_rate_hz":    int(orig_sr),
            "resampled_sample_rate_hz": WAVLM_SAMPLE_RATE,
            "duration_seconds":         duration_s,
            "model_id":                 getattr(model.config, "name_or_path",
                                                "unknown"),
            "hidden_layer_index":       layer,
            "raw_feature_rate_hz":      50,   # WavLM-base+ stride 320 over 16 kHz
            "pool_factor":              pool_factor,
            "feature_rate_hz":          50.0 / pool_factor,
            "feature_dim":              int(pooled.shape[1]) if pooled.size else 0,
            "feature_frames":           int(pooled.shape[0]),
            "dtype":                    str(pooled.dtype),
            # Overlap-stitching provenance
            "chunk_samples":            current_chunk_samples,
            "chunk_length_s":           current_chunk_samples / WAVLM_SAMPLE_RATE,
            "context_samples":          current_context_samples,
            "context_s":                current_context_samples / WAVLM_SAMPLE_RATE,
            "oom_fallbacks":            oom_fallbacks,
            "use_autocast_fp16":        bool(use_autocast_fp16),
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
    """True iff both the .npy and .json exist on disk and are nonempty.

    A 0-byte .npy from a crashed prior run is treated as missing so the
    next run re-extracts it. We also try to json.load the sidecar so a
    truncated sidecar from a killed write gets retried rather than
    trusted. (An idempotent run with a corrupt sidecar on disk would
    otherwise skip silently and break downstream JSON parsers.)
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
    use_autocast_fp16: bool,
    layer: int,
    pool_factor: int,
    chunk_samples: int,
    context_samples: int,
    max_retries: int,
    max_oom_fallbacks: int,
    overwrite: bool,
    dry_run: bool,
    io_threads: int,
) -> list[dict]:
    """Run every task sequentially through the model.

    Extraction is CPU-bound on the model forward pass, which is already
    parallelised internally by torch on GPU / MKL on CPU. Running tasks
    concurrently at the Python level only creates GPU contention, so the
    model loop runs on the main thread. `io_threads` is retained for
    potential async prefetch of the NEXT wav while the current one is
    being featurized — but that optimisation is out-of-scope here and
    is left as a no-op hook.
    """
    del io_threads  # reserved; see docstring.
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
                use_autocast_fp16=use_autocast_fp16,
                layer=layer,
                pool_factor=pool_factor,
                chunk_samples=chunk_samples,
                context_samples=context_samples,
                max_retries=max_retries,
                max_oom_fallbacks=max_oom_fallbacks,
            )

        print(f"  [{result['status']:<7}] {task.split:<5} "
              f"{task.file_id}  {result['message']}")
        results.append(result)

    return results


# =============================================================================
# Summary
# =============================================================================

def print_summary(results: list[dict], elapsed_s: float, dry_run: bool) -> None:
    """Print counts by status + total bytes written (feature files only)."""
    counts: dict[str, int] = {}
    bytes_written = 0
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "ok":
            # extract_one returns "OK (...) -> NNN bytes)"; best-effort parse.
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
    print("WAVLM EXTRACTION SUMMARY" + ("  [DRY-RUN]" if dry_run else ""))
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
            "Extract WavLM-base+ penultimate-layer features from all "
            "participant wavs listed in a turn-taking manifest. Uses "
            "overlap-stitched chunking to eliminate 30-s seam artifacts, "
            "autocast fp16 on CUDA (not model.half()), NaN/Inf guards on "
            "the output, and OOM-aware chunk halving for mid-run VRAM "
            "pressure. Reuses extract_interaction_id / extract_participant_id "
            "from csci535-project/scripts/download_annotated_interactions.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O paths
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST_PATH,
        help="Path to the manifest CSV. Default: %(default)s",
    )
    parser.add_argument(
        "--audio-dir", default=DEFAULT_AUDIO_DIR,
        help=(
            "Directory containing the input .wav files "
            "(output of download_audio_from_manifest.py). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save .npy / .json feature files into. "
             "Default: %(default)s",
    )

    # Model / extraction
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help="HuggingFace model id (default: %(default)s). "
             "Pass a different WavLM / HuBERT / WavLM-large checkpoint "
             "to run an ablation with the rest of the pipeline unchanged.",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER,
        help="Index into the hidden-states tuple. -1 = final layer, "
             "-2 = penultimate (default; see docstring for rationale). "
             "0 = CNN output embeddings (pre-transformer).",
    )
    parser.add_argument(
        "--pool-factor", type=int, default=DEFAULT_POOL_FACTOR,
        help="Mean-pool factor along time. Default %(default)s -> "
             "10 Hz features from a 50 Hz model. Use 1 to keep 50 Hz.",
    )
    parser.add_argument(
        "--chunk-length-s", type=float, default=DEFAULT_CHUNK_LENGTH_S,
        help=(
            "Seconds of audio per forward pass (= transformer attention "
            "window). Lowering this reduces peak VRAM; raising it gives "
            "attention more context. Default: %(default)s s. Note: of "
            "this, the middle (chunk_length_s - 2 * context_s) seconds "
            "is the 'core' whose output frames are kept; the margins on "
            "each side feed attention and are discarded."
        ),
    )
    parser.add_argument(
        "--context-s", type=float, default=DEFAULT_CONTEXT_S,
        help=(
            "Seconds of audio margin on EACH side of the kept core. "
            "The full chunk fed to WavLM is "
            "(left_context + core + right_context). Only the core's "
            "output frames are retained; margins feed self-attention and "
            "are discarded. Eliminates 30-s-seam feature artifacts. "
            "Default: %(default)s s. Set to 0 to recover the old "
            "non-overlapping behavior."
        ),
    )

    # Robustness knobs
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help="Per-file retry budget for transient non-OOM forward-pass "
             "errors (e.g., rare CUDA launch failures). Each retry waits "
             "1 s before re-attempting. Default: %(default)s.",
    )
    parser.add_argument(
        "--max-oom-fallbacks", type=int, default=DEFAULT_MAX_OOM_FALLBACKS,
        help="On CUDA OutOfMemoryError, halve chunk_samples and retry "
             "this file from scratch. This flag caps how many times we "
             "halve before giving up. Default: %(default)s.",
    )

    # Runtime
    parser.add_argument(
        "--device", default="auto",
        help="Torch device. 'auto' picks cuda > mps > cpu. Or pass "
             "'cuda', 'cuda:0', 'mps', 'cpu' explicitly. Default: auto.",
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable autocast fp16 on CUDA; run everything in fp32. "
             "Use if the NaN/Inf guard trips on fp16 outputs or you want "
             "exact reproducibility against a reference fp32 run.",
    )

    # Selection / dry-run plumbing (same surface as download scripts)
    parser.add_argument(
        "--filter-split", choices=["train", "val", "test"], default=None,
        help="If set, only process rows whose manifest `split` column matches.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract files that already have .npy + .json on disk. "
             "Default is to skip them (idempotent resume).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate inputs/outputs without loading the model or "
             "reading any wavs.",
    )

    args = parser.parse_args()

    # --- Load manifest --------------------------------------------------
    print(f"Loading manifest:  {args.manifest}")
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  {len(rows):,} interactions in manifest "
          f"({2 * len(rows):,} participant wavs)")

    # --- Validate audio dir existence (early fail on dry-run) -----------
    if not os.path.isdir(args.audio_dir):
        print(f"ERROR: audio directory not found: {args.audio_dir}\n"
              f"       Run download_audio_from_manifest.py first, or "
              f"pass --audio-dir.", file=sys.stderr)
        return 1

    # --- Build task list ------------------------------------------------
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

    # --- Validate context-s ≤ chunk_length_s / 2 ------------------------
    if 2 * args.context_s >= args.chunk_length_s:
        print(f"ERROR: --context-s ({args.context_s}s) must be less than "
              f"--chunk-length-s/2 ({args.chunk_length_s/2}s) — otherwise "
              f"the core region (chunk - 2*context) is non-positive.",
              file=sys.stderr)
        return 1

    print(f"Audio directory:   {args.audio_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Extraction config: model={args.model}  layer={args.layer}  "
          f"pool_factor={args.pool_factor}  "
          f"chunk={args.chunk_length_s}s  context={args.context_s}s "
          f"(core={args.chunk_length_s - 2*args.context_s}s)"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")
    print(f"Retries:           {args.max_retries} transient, "
          f"{args.max_oom_fallbacks} OOM fallbacks per file")

    # --- Load model (skipped on dry-run) --------------------------------
    model = None
    device = "cpu"
    use_autocast_fp16 = False
    if not args.dry_run:
        device = pick_device(args.device)
        print(f"  Device:            {device}")
        try:
            model, _fe, use_autocast_fp16 = load_wavlm(
                model_id=args.model,
                device=device,
                use_fp16=not args.no_fp16,
            )
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: model load failed: {e}", file=sys.stderr)
            return 1

    chunk_samples = int(args.chunk_length_s * WAVLM_SAMPLE_RATE)
    context_samples = int(args.context_s * WAVLM_SAMPLE_RATE)
    print(f"Chunk samples:     {chunk_samples:,} "
          f"({args.chunk_length_s:.1f}s @ {WAVLM_SAMPLE_RATE} Hz)")
    print(f"Context samples:   {context_samples:,} "
          f"({args.context_s:.1f}s per side)")
    print(f"Starting {len(tasks):,} extraction task(s)...\n")

    start = time.time()
    results = run_tasks(
        tasks=tasks,
        model=model,
        device=device,
        use_autocast_fp16=use_autocast_fp16,
        layer=args.layer,
        pool_factor=args.pool_factor,
        chunk_samples=chunk_samples,
        context_samples=context_samples,
        max_retries=args.max_retries,
        max_oom_fallbacks=args.max_oom_fallbacks,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        io_threads=0,
    )
    elapsed = time.time() - start

    print_summary(results, elapsed, args.dry_run)

    # Exit codes match the download script's conventions.
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_missing = sum(1 for r in results if r["status"] == "missing")
    if n_failed:
        return 2
    if n_missing:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
