#!/usr/bin/env python3
"""
Run one face-emotion model across the 136 canonical participant videos and
save per-frame valence, arousal, and 8-class logits to
`predictions/{model}/{file_id}.npz`.

V1 implements EmoNet end-to-end. HSEmotion is stubbed with a NotImplementedError
and will be filled in once the EmoNet path has been validated on a `--limit 2`
dry-run + real run. This split keeps the alignment/IO scaffolding correct for
one model before we add a second.

Usage:
    python extract.py --model emonet --dry-run --limit 2
    python extract.py --model emonet --limit 2
    python extract.py --model emonet
    python extract.py --model emonet --file-id V00_S1132_I00000333_P0737
    python extract.py --model hsemotion        # raises NotImplementedError (V1)

Design notes:

* Per-file output NPZ contains:
      valence       (N_frames,) float32
      arousal       (N_frames,) float32
      class_logits  (N_frames, 8) float32
      is_valid      (N_frames,) bool  — True if alignment+forward succeeded
      class_order   (8,) <U12       — model's native class label order
      model_name    (,)  <U64
      n_frames      () int32
      n_valid       () int32
      alignment     (,) <U8         — "bbox" | "5pt"
  Rows where is_valid=False have NaN v/a and all-zero class_logits.

* Resume-safety: if `predictions/{model}/{fid}.npz` already exists and opens
  cleanly, the file_id is skipped. Corrupt output from a killed prior run
  re-extracts (it won't open cleanly).

* Per-run summary JSON: `logs/extract_{model}_run_{timestamp}.json` or
  `logs/extract_{model}_dryrun_{timestamp}.json`. Diagnostic only — score.py
  does not read this. See METHODOLOGY.md "Per-run summary JSON".

* Dry-run: no NPZ writes. Instead, the first few aligned crops of each
  file_id are saved to `logs/align_sanity/{model}/{fid}/frame{idx}.png` for
  visual inspection.

* Invalid-frame reason tracking:
      is_valid_box_0 — the per-frame is_valid_box flag is 0
      low_conf       — mean face-landmark confidence < MIN_FACE_CONF
      align_fail     — alignment or forward raised (degenerate bbox, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Import common first so PYTORCH_ENABLE_MPS_FALLBACK is set before torch loads.
import common
from common import (
    FileIdInfo,
    LOGS_DIR,
    PREDICTIONS_DIR,
    bundle_is_usable,
    data_path_for,
    discover_file_ids,
    get_logger,
    pick_device,
)

import torch

from alignment import AlignmentMode, align_face

# Logger is initialized inside main() — not at module import time — so that
# `import extract` from a REPL, test, or debugger does not create a stray
# `logs/extract_{timestamp}.log` file as a side effect.

HERE: Path = Path(__file__).resolve().parent
VENDORS_DIR: Path = HERE / "vendors"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Minimum mean face-landmark confidence to consider a frame valid. Sapiens
# face confidences are typically 0.93–0.99 (see METHODOLOGY.md); 0.5 catches
# occluded / partially-out-of-frame cases without being overly aggressive.
MIN_FACE_CONF: float = 0.5

# Default batch size per model. Re-tune from the --limit 2 dry run.
DEFAULT_BATCH_SIZE: dict[str, int] = {
    "emonet": 32,
    "hsemotion": 64,
}

# Canonical class orderings. EmoNet's ordering is from vendors/emonet/demo.py:24.
# HSEmotion order is from vendors/hsemotion/hsemotion/facial_emotions.py:36.
CLASS_ORDER: dict[str, list[str]] = {
    "emonet": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
    "hsemotion": ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"],
}

# Model input sizes (square).
INPUT_SIZE: dict[str, int] = {
    "emonet": 256,
    "hsemotion": 224,  # enet_b0_8_va_mtl → 224; b2 variants would be 260
}

# Alignment mode per model. EmoNet uses its own bbox-from-landmarks transform;
# HSEmotion uses ArcFace-style 5pt similarity.
ALIGN_MODE: dict[str, AlignmentMode] = {
    "emonet": "bbox",
    "hsemotion": "5pt",
}

# How many sanity crops to save per file_id during --dry-run.
SANITY_CROPS_PER_FILE: int = 3


# ---------------------------------------------------------------------------
# Per-file result record (mirrors download.py's pattern for consistency)
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    file_id: str
    status: str                        # "ok"|"skipped"|"failed" or "dry_run_ok"|"dry_run_failed"
    n_frames: int = 0                  # from keypoints length (the output grid size)
    n_video_frames: int = 0            # frames actually decoded from the mp4
    n_valid: int = 0
    invalid_reasons: dict = field(default_factory=lambda: {
        "is_valid_box_0": 0, "low_conf": 0, "align_fail": 0,
    })
    elapsed_sec: float = 0.0
    batches: int = 0
    output_npz: Optional[str] = None     # real runs only
    sanity_crops_dir: Optional[str] = None  # dry runs only
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Extractor base class + per-model implementations
# ---------------------------------------------------------------------------

class EmotionExtractor:
    """
    Base class for a single face-emotion model. Concrete subclasses implement
    weight loading, preprocessing, and the forward pass. Everything else
    (video iteration, alignment, batching, output writing) is model-agnostic
    and lives in `process_one`.

    Attributes set by subclasses:
        model_name: str  — used to key output dirs and NPZ `model_name`
        input_size: int
        align_mode: AlignmentMode
        class_order: list[str]
        device: torch.device
        weights_path: Path   — for logging / summary JSON
    """
    model_name: str = ""
    input_size: int = 0
    align_mode: AlignmentMode = "bbox"
    class_order: list[str] = []
    weights_path: Path = Path("")

    def __init__(self, device: str, logger):
        self.device = device
        self.log = logger

    def preprocess(self, aligned_rgb: np.ndarray) -> torch.Tensor:
        """
        Convert an (H, W, 3) uint8 RGB aligned crop to the model's expected
        (3, H, W) float tensor on `self.device`. Subclasses override.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def forward_batch(self, batch: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Run a (B, 3, H, W) batch through the model and return numpy arrays:
            {"valence": (B,) f32, "arousal": (B,) f32, "class_logits": (B, 8) f32}
        Subclasses override.
        """
        raise NotImplementedError


# --- EmoNet --------------------------------------------------------------

class EmoNetExtractor(EmotionExtractor):
    """
    github.com/face-analysis/emonet, trained on AffectNet-8.

    Forward output per `vendors/emonet/emonet/models/emonet.py:222` is a dict
    with keys 'heatmap', 'expression' (B, 8), 'valence' (B,), 'arousal' (B,).
    We clamp valence/arousal to [-1, 1] matching `demo.py:50`.

    Preprocessing matches `demo.py:46`: RGB → (3, H, W) float / 255.0. No
    mean subtraction.

    Weights are loaded from `vendors/emonet/pretrained/emonet_8.pth`. The
    shipped checkpoint has `module.` prefixes from DataParallel training; we
    strip those and load with `strict=False`.
    """
    model_name = "emonet"
    input_size = INPUT_SIZE["emonet"]
    align_mode: AlignmentMode = ALIGN_MODE["emonet"]
    class_order = CLASS_ORDER["emonet"]

    def __init__(self, device: str, logger):
        super().__init__(device, logger)

        # Make `from emonet.models import EmoNet` resolvable without polluting
        # the install. Prepending is fine — the vendored dir is self-contained
        # and has no module name collisions with the rest of the pipeline.
        emonet_root = VENDORS_DIR / "emonet"
        if str(emonet_root) not in sys.path:
            sys.path.insert(0, str(emonet_root))

        from emonet.models import EmoNet  # noqa: E402  — deferred import after path hack

        self.weights_path = emonet_root / "pretrained" / "emonet_8.pth"
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"EmoNet weights not found at {self.weights_path}. "
                f"Clone https://github.com/face-analysis/emonet and ensure "
                f"pretrained/emonet_8.pth is present."
            )

        self.log.info("EmoNet: loading weights from %s", self.weights_path)
        state_dict = torch.load(str(self.weights_path), map_location="cpu")
        # Strip the `module.` prefix left over from DataParallel-trained checkpoints.
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.net = EmoNet(n_expression=8)
        # EmoNet's checkpoint has extra keys that don't align 1:1 with the
        # current module tree (cf. demo.py:34 also uses strict=False).
        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        if missing:
            self.log.info("EmoNet: %d missing keys (strict=False OK): first=%s",
                          len(missing), missing[0] if missing else None)
        if unexpected:
            self.log.info("EmoNet: %d unexpected keys (strict=False OK): first=%s",
                          len(unexpected), unexpected[0] if unexpected else None)
        # Belt-and-suspenders: a catastrophically wrong checkpoint (wrong file,
        # truncated download, unrelated model) leaves most of the net randomly
        # initialized and would silently produce plausible-looking but
        # meaningless predictions. A correct emonet_8.pth load has at most a
        # handful of legitimately missing keys (running stats on a renamed
        # layer etc.) — anything above the threshold means the load is bogus.
        MAX_ALLOWED_MISSING = 10
        if len(missing) > MAX_ALLOWED_MISSING:
            raise RuntimeError(
                f"EmoNet checkpoint load looks broken: {len(missing)} missing keys "
                f"(threshold {MAX_ALLOWED_MISSING}). The net is largely uninitialized "
                f"and would produce meaningless predictions. Checkpoint path: "
                f"{self.weights_path}. First few missing: {missing[:5]}"
            )

        # Do NOT chain `.to(dev).eval()`: EmoNet overrides `eval()` and does
        # not `return self` (see vendors/emonet/emonet/models/emonet.py:225).
        # Chaining would leave self.net = None and the first forward pass
        # would fail with "'NoneType' object is not callable".
        self.net = self.net.to(self.device)
        self.net.eval()

        # Normalization constants captured as tensors for preprocess speed.
        # EmoNet uses raw [0,1] range (no mean subtraction); kept here so the
        # subclass contract is explicit even when both are trivial.
        self._scale = 1.0 / 255.0

    def preprocess(self, aligned_rgb: np.ndarray) -> torch.Tensor:
        # aligned_rgb: (H, W, 3) uint8 RGB
        # Target: (3, H, W) float tensor on self.device in [0, 1].
        t = torch.from_numpy(aligned_rgb).to(self.device).float() * self._scale
        return t.permute(2, 0, 1).contiguous()

    @torch.inference_mode()
    def forward_batch(self, batch: torch.Tensor) -> dict[str, np.ndarray]:
        out = self.net(batch)
        # Clamp v/a to [-1, 1] per the demo.
        valence = out["valence"].clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32).reshape(-1)
        arousal = out["arousal"].clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32).reshape(-1)
        # `expression` is the 8-class logits head per emonet.py:222.
        class_logits = out["expression"].detach().cpu().numpy().astype(np.float32)
        if class_logits.shape[1] != 8:
            raise RuntimeError(
                f"EmoNet class_logits unexpected width {class_logits.shape} — expected (B, 8). "
                "Check n_expression kwarg."
            )
        return {"valence": valence, "arousal": arousal, "class_logits": class_logits}


# --- HSEmotion (stubbed for V1) ------------------------------------------

class HSEmotionExtractor(EmotionExtractor):
    """
    Stubbed for V1. Slotted in after EmoNet has been validated end-to-end.

    When filled in, the implementation will:
      * Use `vendors/hsemotion/hsemotion/facial_emotions.py:get_model_path`
        to fetch `enet_b0_8_va_mtl.pt` into `~/.hsemotion/`.
      * Load via `torch.load(path, map_location=...)` — the file is a pickled
        full model, not a state_dict (facial_emotions.py:50).
      * Replace `model.classifier` with Identity and stash the linear
        weights/bias for an out-of-model matmul (same trick as the library,
        but batched across frames on MPS).
      * ImageNet normalize inputs (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
      * Split the 10-dim output: first 8 dims = class logits, last 2 = (valence, arousal).
    """
    model_name = "hsemotion"
    input_size = INPUT_SIZE["hsemotion"]
    align_mode: AlignmentMode = ALIGN_MODE["hsemotion"]
    class_order = CLASS_ORDER["hsemotion"]

    def __init__(self, device: str, logger):
        super().__init__(device, logger)
        raise NotImplementedError(
            "HSEmotionExtractor is stubbed in V1. Validate the EmoNet path "
            "first, then fill this in following the recipe in the class docstring."
        )


def load_extractor(model: str, device: str, logger) -> EmotionExtractor:
    if model == "emonet":
        return EmoNetExtractor(device, logger)
    if model == "hsemotion":
        return HSEmotionExtractor(device, logger)
    raise ValueError(f"unknown model {model!r}; choose from emonet | hsemotion")


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def iter_video_frames(mp4_path: Path):
    """
    Yield (frame_idx, rgb_uint8) pairs for every frame in `mp4_path`. Handles
    BGR → RGB conversion. Raises on open failure (loudness > silence).
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open {mp4_path}")
    try:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield idx, rgb
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def _save_npz_atomic(
    dest: Path,
    *,
    valence: np.ndarray,
    arousal: np.ndarray,
    class_logits: np.ndarray,
    is_valid: np.ndarray,
    class_order: list[str],
    model_name: str,
    alignment_mode: AlignmentMode,
) -> None:
    """
    Write the per-file output NPZ atomically (tmp + os.replace).

    Any prior `.part` is clobbered; the final path replaces atomically so a
    killed process can never leave a "looks complete but is truncated" file
    behind.

    Note: `np.savez` auto-appends `.npz` to any string/Path argument whose name
    doesn't already end in `.npz`, which would corrupt the `.part` suffix
    scheme. Pass an open file handle instead — numpy writes exactly there.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    with open(part, "wb") as f:
        np.savez(
            f,
            valence=valence,
            arousal=arousal,
            class_logits=class_logits,
            is_valid=is_valid,
            class_order=np.asarray(class_order, dtype="<U12"),
            model_name=np.asarray(model_name, dtype="<U64"),
            n_frames=np.int32(valence.shape[0]),
            n_valid=np.int32(int(is_valid.sum())),
            alignment=np.asarray(alignment_mode, dtype="<U8"),
        )
    os.replace(part, dest)


def _existing_output_usable(npz_path: Path) -> bool:
    """True iff the NPZ opens cleanly and has the expected keys."""
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            required = {"valence", "arousal", "class_logits", "is_valid",
                        "class_order", "model_name", "n_frames", "n_valid", "alignment"}
            return required.issubset(set(z.files))
    except (OSError, ValueError, EOFError):
        return False


def _flush_batch(
    extractor: EmotionExtractor,
    batch_tensors: list[torch.Tensor],
    batch_frame_idxs: list[int],
    *,
    valence_out: np.ndarray,
    arousal_out: np.ndarray,
    logits_out: np.ndarray,
) -> None:
    """Stack the pending tensors into a batch, forward, scatter back into the per-frame outputs."""
    if not batch_tensors:
        return
    batch = torch.stack(batch_tensors, dim=0)
    out = extractor.forward_batch(batch)
    idx = np.asarray(batch_frame_idxs, dtype=np.int64)
    valence_out[idx] = out["valence"]
    arousal_out[idx] = out["arousal"]
    logits_out[idx] = out["class_logits"]


def _save_sanity_crop(crop_rgb: np.ndarray, out_path: Path, logger) -> None:
    """
    Save a sanity-check aligned crop. cv2.imwrite returns False on silent
    failure (disk full, bad permissions, unsupported extension) — surface that
    via `log.warning` instead of dropping the crop without a trace.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects BGR; we have RGB.
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), bgr)
    if not ok:
        logger.warning("cv2.imwrite returned False for %s — sanity crop not saved", out_path)


def process_one(
    info: FileIdInfo,
    extractor: EmotionExtractor,
    args: argparse.Namespace,
    logger,
) -> FileResult:
    """
    Run `extractor` across every frame of one file_id's mp4.

    Returns a FileResult. Exceptions from inside this function are caught by
    main() and packaged into a 'failed' result rather than crashing the batch.
    """
    result = FileResult(file_id=info.file_id, status="ok")
    t0 = time.monotonic()

    # 1. Inputs on disk.
    if not bundle_is_usable(info.file_id):
        result.status = "dry_run_failed" if args.dry_run else "failed"
        result.error = "bundle_is_usable() returned False — download.py not run for this fid?"
        result.elapsed_sec = time.monotonic() - t0
        return result

    # Resume-safety: skip if we already have a clean output.
    out_path = PREDICTIONS_DIR / extractor.model_name / f"{info.file_id}.npz"
    if not args.dry_run and _existing_output_usable(out_path):
        result.status = "skipped"
        result.output_npz = str(out_path.relative_to(HERE))
        result.elapsed_sec = time.monotonic() - t0
        return result

    # 2. Load keypoints + is_valid_box, slice to face sub-array.
    kp_path = data_path_for(info.file_id, "keypoints")
    ivb_path = data_path_for(info.file_id, "is_valid_box")
    mp4_path = data_path_for(info.file_id, "mp4")

    # mmap keeps memory low for big files; we only need per-frame slices.
    kp = np.load(kp_path, mmap_mode="r")               # (N, 133, 3)
    is_valid_box = np.load(ivb_path, mmap_mode="r")    # (N,) or (N, 1)
    if is_valid_box.ndim > 1:
        is_valid_box = is_valid_box.reshape(-1)

    if kp.shape[1] != 133 or kp.shape[2] != 3:
        result.status = "dry_run_failed" if args.dry_run else "failed"
        result.error = f"unexpected keypoints shape {kp.shape} — expected (N, 133, 3)"
        result.elapsed_sec = time.monotonic() - t0
        return result

    if is_valid_box.shape[0] != kp.shape[0]:
        result.status = "dry_run_failed" if args.dry_run else "failed"
        result.error = (
            f"is_valid_box length {is_valid_box.shape[0]} != keypoints length "
            f"{kp.shape[0]} — bundle shape mismatch"
        )
        result.elapsed_sec = time.monotonic() - t0
        return result

    n_frames = int(kp.shape[0])
    result.n_frames = n_frames

    # 3. Allocate per-frame outputs. NaN for v/a on invalid frames so score.py
    #    never silently interprets a fallback 0 as a real prediction.
    valence_out = np.full((n_frames,), np.nan, dtype=np.float32)
    arousal_out = np.full((n_frames,), np.nan, dtype=np.float32)
    logits_out = np.zeros((n_frames, 8), dtype=np.float32)
    is_valid_out = np.zeros((n_frames,), dtype=bool)

    sanity_dir: Optional[Path] = None
    if args.dry_run:
        sanity_dir = LOGS_DIR / "align_sanity" / extractor.model_name / info.file_id
        sanity_dir.mkdir(parents=True, exist_ok=True)
        result.sanity_crops_dir = str(sanity_dir.relative_to(HERE))

    # 4. Iterate video frames, align, batch, forward.
    batch_tensors: list[torch.Tensor] = []
    batch_frame_idxs: list[int] = []
    n_sanity_saved = 0
    n_batches = 0
    n_video_frames = 0     # how many frames iter_video_frames actually yielded

    try:
        for frame_idx, rgb in iter_video_frames(mp4_path):
            n_video_frames = frame_idx + 1
            if frame_idx >= n_frames:
                # Video has more frames than keypoints — ignore the tail. Shouldn't
                # happen given Sapiens processed every frame, but handle defensively.
                break

            if int(is_valid_box[frame_idx]) != 1:
                result.invalid_reasons["is_valid_box_0"] += 1
                continue

            face68 = kp[frame_idx, 23:91, :]        # (68, 3) — x, y, conf
            face_conf = float(face68[:, 2].mean())
            if face_conf < MIN_FACE_CONF:
                result.invalid_reasons["low_conf"] += 1
                continue

            try:
                aligned = align_face(
                    rgb, face68[:, :2], mode=extractor.align_mode, out_size=extractor.input_size
                )
            except Exception as e:  # noqa: BLE001 — any alignment failure counts the same
                result.invalid_reasons["align_fail"] += 1
                logger.debug("%s: align_fail @ frame %d: %s", info.file_id, frame_idx, e)
                continue

            if args.dry_run and n_sanity_saved < SANITY_CROPS_PER_FILE and sanity_dir is not None:
                _save_sanity_crop(aligned, sanity_dir / f"frame{frame_idx}.png", logger)
                n_sanity_saved += 1

            # Build the tensor and stage it in the current batch.
            batch_tensors.append(extractor.preprocess(aligned))
            batch_frame_idxs.append(frame_idx)
            is_valid_out[frame_idx] = True

            if len(batch_tensors) >= args.batch_size:
                _flush_batch(
                    extractor, batch_tensors, batch_frame_idxs,
                    valence_out=valence_out, arousal_out=arousal_out, logits_out=logits_out,
                )
                n_batches += 1
                batch_tensors.clear()
                batch_frame_idxs.clear()

        # Final flush
        if batch_tensors:
            _flush_batch(
                extractor, batch_tensors, batch_frame_idxs,
                valence_out=valence_out, arousal_out=arousal_out, logits_out=logits_out,
            )
            n_batches += 1
            batch_tensors.clear()
            batch_frame_idxs.clear()

    except Exception as e:  # noqa: BLE001
        result.status = "dry_run_failed" if args.dry_run else "failed"
        result.error = f"{type(e).__name__}: {e}"
        logger.error("%s: FAILED during extraction — %s\n%s",
                     info.file_id, e, traceback.format_exc())
        result.elapsed_sec = time.monotonic() - t0
        return result

    result.n_valid = int(is_valid_out.sum())
    result.batches = n_batches
    result.n_video_frames = n_video_frames

    # Video-vs-keypoints length sanity. A truncated mp4 would otherwise leave
    # the tail of every output array as NaN / False silently; score.py would
    # filter those out via is_valid and never flag the mismatch to the user.
    if n_video_frames < n_frames:
        logger.warning(
            "%s: video yielded %d frame(s) but keypoints have %d — tail %d "
            "output row(s) remain invalid/NaN (is_valid=False). Possible "
            "truncated mp4.",
            info.file_id, n_video_frames, n_frames, n_frames - n_video_frames,
        )

    # 5. Write output NPZ (real runs only).
    if not args.dry_run:
        try:
            _save_npz_atomic(
                out_path,
                valence=valence_out,
                arousal=arousal_out,
                class_logits=logits_out,
                is_valid=is_valid_out,
                class_order=extractor.class_order,
                model_name=extractor.model_name,
                alignment_mode=extractor.align_mode,
            )
            result.output_npz = str(out_path.relative_to(HERE))
        except Exception as e:  # noqa: BLE001
            result.status = "failed"
            result.error = f"npz write: {type(e).__name__}: {e}"
            logger.exception("%s: NPZ write failed", info.file_id)
            result.elapsed_sec = time.monotonic() - t0
            return result
    else:
        # Dry run: status conveys success of the align+forward dry pass.
        result.status = "dry_run_ok"

    result.elapsed_sec = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def resolve_file_ids(
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


def _write_summary(
    path: Path,
    *,
    dry_run: bool,
    model: str,
    args: argparse.Namespace,
    device: str,
    weights_path: Path,
    run_started: str,
    run_ended: str,
    elapsed_sec_total: float,
    results: list[FileResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        totals = {
            "file_ids": len(results),
            "dry_run_ok": sum(1 for r in results if r.status == "dry_run_ok"),
            "dry_run_failed": sum(1 for r in results if r.status == "dry_run_failed"),
        }
    else:
        totals = {
            "file_ids": len(results),
            "ok": sum(1 for r in results if r.status == "ok"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "failed": sum(1 for r in results if r.status == "failed"),
        }
    summary = {
        "dry_run": dry_run,
        "model": model,
        "run_started": run_started,
        "run_ended": run_ended,
        "elapsed_sec_total": elapsed_sec_total,
        "args": {
            "limit": args.limit,
            "file_id": args.file_id,
            "batch_size": args.batch_size,
        },
        "device": device,
        "weights_path": str(weights_path) if weights_path else None,
        "totals": totals,
        "per_file": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["emonet", "hsemotion"],
                   help="which face-emotion model to run (one process per model)")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N file_ids (sorted); always use --limit 2 first")
    p.add_argument("--file-id", type=str, default=None,
                   help="process only this single file_id (ignored if --limit is set)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="forward batch size (defaults: emonet=32, hsemotion=64)")
    p.add_argument("--dry-run", action="store_true",
                   help="run alignment+forward but don't write NPZs; save a few sanity crops per file_id")
    args = p.parse_args(argv)

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.model]
    if args.batch_size <= 0:
        p.error("--batch-size must be positive")

    # Logger is initialized here rather than at module import time so that
    # importing this file as a library is side-effect-free.
    log = get_logger("extract")

    run_started_ts = datetime.now()
    run_started_iso = run_started_ts.strftime("%Y-%m-%d %H:%M:%S")
    run_stamp = run_started_ts.strftime("%Y%m%d-%H%M%S")

    log.info("Discovering canonical file_ids …")
    all_infos = discover_file_ids()
    log.info("Canonical file_ids: %d", len(all_infos))

    todo = resolve_file_ids(all_infos, limit=args.limit, only_file_id=args.file_id)
    log.info("Processing %d file_id(s) with model=%s batch_size=%d%s",
             len(todo), args.model, args.batch_size, "  [DRY RUN]" if args.dry_run else "")

    # Device + extractor
    device = pick_device()
    log.info("torch device: %s  (torch=%s)", device, torch.__version__)
    extractor = load_extractor(args.model, device, log)
    log.info("Loaded %s (input=%d, align=%s, weights=%s)",
             extractor.model_name, extractor.input_size, extractor.align_mode, extractor.weights_path)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (PREDICTIONS_DIR / extractor.model_name).mkdir(parents=True, exist_ok=True)

    # Run serially at the file_id level. Models don't cleanly share MPS
    # between threads, and per-file throughput is already dominated by the
    # forward pass, not Python overhead.
    results: list[FileResult] = []
    t_start = time.monotonic()
    try:
        for i, info in enumerate(todo, start=1):
            try:
                r = process_one(info, extractor, args, log)
            except Exception as e:  # noqa: BLE001 — unexpected escape from process_one
                r = FileResult(
                    file_id=info.file_id,
                    status="dry_run_failed" if args.dry_run else "failed",
                    error=f"unexpected: {type(e).__name__}: {e}",
                )
                log.exception("%s: unexpected exception", info.file_id)
            results.append(r)
            log.info(
                "[%d/%d] %s  %s  (n_valid=%d/%d  batches=%d  %.1fs)",
                i, len(todo), r.file_id, r.status, r.n_valid, r.n_frames, r.batches, r.elapsed_sec,
            )
    except KeyboardInterrupt:
        log.warning("Interrupted — writing partial summary")

    run_ended_ts = datetime.now()
    run_ended_iso = run_ended_ts.strftime("%Y-%m-%d %H:%M:%S")
    elapsed_total = time.monotonic() - t_start

    # Per-run summary JSON — distinct filenames for dry-run vs real.
    suffix = "dryrun" if args.dry_run else "run"
    summary_path = LOGS_DIR / f"extract_{args.model}_{suffix}_{run_stamp}.json"
    _write_summary(
        summary_path,
        dry_run=args.dry_run,
        model=args.model,
        args=args,
        device=device,
        weights_path=extractor.weights_path,
        run_started=run_started_iso,
        run_ended=run_ended_iso,
        elapsed_sec_total=elapsed_total,
        results=results,
    )
    log.info("Wrote %s", summary_path)

    # Textual summary
    if args.dry_run:
        n_ok   = sum(1 for r in results if r.status == "dry_run_ok")
        n_fail = sum(1 for r in results if r.status == "dry_run_failed")
        log.info("=" * 60)
        log.info("DRY-RUN summary (no NPZs written):")
        log.info("  processed:            %d", len(results))
        log.info("  dry_run_ok:           %d", n_ok)
        log.info("  dry_run_failed:       %d", n_fail)
        log.info("  wall time:            %.1fs", elapsed_total)
    else:
        n_ok   = sum(1 for r in results if r.status == "ok")
        n_skip = sum(1 for r in results if r.status == "skipped")
        n_fail = sum(1 for r in results if r.status == "failed")
        log.info("=" * 60)
        log.info("Summary:")
        log.info("  processed:            %d", len(results))
        log.info("  ok:                   %d", n_ok)
        log.info("  skipped (already on disk): %d", n_skip)
        log.info("  failed:               %d", n_fail)
        log.info("  wall time:            %.1fs", elapsed_total)

    if any(r.status in ("failed", "dry_run_failed") for r in results):
        log.info("Failed file_ids:")
        for r in results:
            if r.status in ("failed", "dry_run_failed"):
                log.info("  %s  %s", r.file_id, r.error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
