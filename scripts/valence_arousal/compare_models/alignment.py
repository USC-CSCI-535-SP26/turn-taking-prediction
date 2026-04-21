"""
Face alignment utilities for the emotion-extractor bake-off.

Two alignment modes are supported, selected per model:

  * "bbox"  — EmoNet.  Derive a face bounding box from the min/max of the 68
              iBUG landmarks, then apply EmoNet's exact training-pipeline
              transform: `get_scale_center` + `get_transform` (see
              `vendors/emonet/emonet/data_augmentation.py`). Translation +
              uniform scale only; no rotation, no shape-driven similarity fit.

  * "5pt"   — HSEmotion (stubbed for V1; wired up when the HSEmotion extractor
              lands). Derive 5 points from 68 iBUG landmarks (eye centers from
              averaging iBUG 36–41 / 42–47, nose tip at iBUG 30, mouth corners
              at iBUG 48 / 54), fit a 2D similarity transform via Umeyama to
              the ArcFace canonical 5pt template (112×112, scaled to the model
              input size).

The 14.6 px Sapiens quantization on face landmarks (see METHODOLOGY.md
"Keypoint format") is why the 5pt path averages each eye's 6 iBUG points
instead of picking one. The 68pt path used by EmoNet dilutes the quantization
across all 68 samples naturally, and EmoNet's own training transform uses
bbox-from-landmarks rather than a 68-pt similarity fit anyway.

Public surface:
    canonical_5pt(out_size)                  -> (5, 2)
    derive_5pt_from_ibug68(face68)           -> (5, 2)
    similarity_transform(src, dst)           -> (2, 3) affine via Umeyama
    emonet_bbox_transform(bb_xyxy, out_size) -> (2, 3) translation + scale
    align_face(frame, face68, mode, out_size) -> (out_size, out_size, 3) uint8

`align_face` returns an RGB uint8 crop ready to hand to each model's own
tensor-prep (EmoNet: /255; HSEmotion: ImageNet normalize). Callers own all
tensor work; alignment.py is image-domain only.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

AlignmentMode = Literal["5pt", "bbox"]

# ---------------------------------------------------------------------------
# iBUG 68 index groups we reference. Indices are LOCAL to the 68-point face
# sub-array (not the global 133-point COCO-WholeBody array). Callers are
# responsible for slicing `kp[:, 23:91, :]` before calling into this module.
# ---------------------------------------------------------------------------

_IBUG_RIGHT_EYE = slice(36, 42)     # 6 points — subject's right eye (viewer's left, low x)
_IBUG_LEFT_EYE = slice(42, 48)      # 6 points — subject's left eye (viewer's right, high x)
_IBUG_NOSE_TIP = 30
# iBUG-68 mouth corner convention (Sagonas 2013): index 48 is the subject's
# right corner (viewer's left, low x); index 54 is the subject's left corner
# (viewer's right, high x). Naming matches iBUG's subject-perspective convention.
_IBUG_MOUTH_RIGHT_CORNER = 48       # subject's right = viewer's left = low x
_IBUG_MOUTH_LEFT_CORNER = 54        # subject's left = viewer's right = high x

# ArcFace canonical 5-point destination template at 112x112, in (x, y) order,
# matching InsightFace/MTCNN/ArcFace training. Points are annotated by
# image-side (low x = viewer's left) to avoid the subject-vs-viewer
# left/right ambiguity that bites 5pt alignment code. Subject-side equivalence:
#   low-x eye  = subject's right eye
#   high-x eye = subject's left eye
_ARCFACE_5PT_112 = np.array(
    [
        [38.2946, 51.6963],  # eye on low-x side of image
        [73.5318, 51.5014],  # eye on high-x side of image
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # mouth corner on low-x side of image
        [70.7299, 92.2041],  # mouth corner on high-x side of image
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# 5-point alignment (HSEmotion path — stubbed V1, wired for when HSEmotion
# extractor is added).
# ---------------------------------------------------------------------------

def canonical_5pt(out_size: int) -> np.ndarray:
    """
    Return the ArcFace 5-point canonical template scaled to `out_size`×`out_size`.

    The published coords are at 112×112; scale linearly to the requested output
    resolution so HSEmotion's 224×224 input matches the training distribution.
    """
    scale = out_size / 112.0
    return _ARCFACE_5PT_112 * scale


def derive_5pt_from_ibug68(face68: np.ndarray) -> np.ndarray:
    """
    Reduce 68 iBUG landmarks to the 5 canonical points (right eye, left eye,
    nose tip, mouth right, mouth left).

    Eye centers are AVERAGED across each eye's 6 iBUG points rather than taking
    a single indexed landmark, to dampen the ~14.6 px Sapiens quantization
    (see METHODOLOGY.md "Quantization caveat"). Nose tip and mouth corners are
    taken directly — single-point quantization is less damaging for those.

    Input:  face68 shape (68, 2) or (68, 3); only the first two columns are used.
    Output: (5, 2) float32.
    """
    if face68.shape[0] != 68:
        raise ValueError(f"expected 68 face landmarks, got shape {face68.shape}")
    pts = face68[:, :2].astype(np.float32)
    right_eye = pts[_IBUG_RIGHT_EYE].mean(axis=0)           # low-x side of image
    left_eye = pts[_IBUG_LEFT_EYE].mean(axis=0)             # high-x side of image
    nose = pts[_IBUG_NOSE_TIP]
    mouth_right = pts[_IBUG_MOUTH_RIGHT_CORNER]             # low-x side of image
    mouth_left = pts[_IBUG_MOUTH_LEFT_CORNER]               # high-x side of image
    # Order matches _ARCFACE_5PT_112: low-x eye, high-x eye, nose, low-x mouth,
    # high-x mouth. The warp therefore preserves image orientation (no flip).
    return np.stack([right_eye, left_eye, nose, mouth_right, mouth_left], axis=0).astype(np.float32)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Umeyama similarity transform (translation + uniform scale + rotation) that
    maps `src` points to `dst` points in a least-squares sense. Returns a
    2×3 affine matrix suitable for `cv2.warpAffine`.

    Reference: Umeyama 1991, "Least-squares estimation of transformation
    parameters between two point patterns". We implement the 2D case directly
    rather than calling skimage to avoid the extra dep.

    Both `src` and `dst` should be (N, 2) with N >= 2.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"src/dst must both be (N, 2); got {src.shape} and {dst.shape}")
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Covariance
    A = dst_demean.T @ src_demean / n
    U, S, Vt = np.linalg.svd(A)
    # Handle reflection (ensure a proper rotation, det = +1)
    d = np.ones(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt

    # Uniform scale
    src_var = (src_demean ** 2).sum() / n
    if src_var <= 1e-12:
        # Degenerate source (all points coincide); fall back to identity scale.
        scale = 1.0
    else:
        scale = (S * d).sum() / src_var

    t = dst_mean - scale * (R @ src_mean)
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M


# ---------------------------------------------------------------------------
# Bbox alignment (EmoNet path — matches their training transform exactly).
# ---------------------------------------------------------------------------

def _get_scale_center(bb: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Port of `emonet.data_augmentation.get_scale_center`. `bb` is (x0, y0, x1, y1).
    Returns (scale, center) where `scale` is the divisor EmoNet uses inside
    `get_transform`, computed from the bbox diagonal over 220.
    """
    x0, y0, x1, y1 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    center = np.array([x1 - (x1 - x0) / 2.0, y1 - (y1 - y0) / 2.0], dtype=np.float64)
    scale = (x1 - x0 + y1 - y0) / 220.0
    return scale, center


def _get_transform(center: np.ndarray, scale: float, res: tuple[int, int]) -> np.ndarray:
    """
    Port of `emonet.data_augmentation.get_transform` with rot=0 (we do not
    apply random rotation at inference). Returns a 3×3 homogeneous transform
    whose first two rows form the affine used by `cv2.warpAffine`.

    The operation is translate-and-uniform-scale: 200*scale is the input-space
    square side that maps to the full output res.
    """
    h = 200.0 * scale
    t = np.zeros((3, 3), dtype=np.float64)
    t[0, 0] = res[1] / h
    t[1, 1] = res[0] / h
    t[0, 2] = res[1] * (-center[0] / h + 0.5)
    t[1, 2] = res[0] * (-center[1] / h + 0.5)
    t[2, 2] = 1.0
    return t


def emonet_bbox_transform(bb_xyxy: np.ndarray, out_size: int = 256) -> np.ndarray:
    """
    Build the 2×3 affine that EmoNet expects given a face bbox. This replicates
    `DataAugmentor(target_width=256, target_height=256)(..., bb=bb)` at
    inference settings (no random scaling, rotation, or translation).

    Input bbox is (x0, y0, x1, y1) in pixel coordinates on the source frame.
    Output is a 2×3 float32 matrix ready for `cv2.warpAffine(..., dsize=(out_size, out_size))`.
    """
    scale, center = _get_scale_center(np.asarray(bb_xyxy, dtype=np.float64))
    if scale <= 1e-9:
        raise ValueError(f"emonet_bbox_transform: degenerate bbox {bb_xyxy}")
    mat = _get_transform(center, scale, (out_size, out_size))
    return mat[:2, :].astype(np.float32)


def face_bbox_from_landmarks(face68: np.ndarray) -> np.ndarray:
    """
    EmoNet's training pipeline takes bb = [xmin, ymin, xmax, ymax] over the
    landmark point set (`vendors/emonet/emonet/data/affecnet.py:122`). Mirror
    that exactly.
    """
    pts = face68[:, :2]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def align_face(
    frame_rgb: np.ndarray,
    face68: np.ndarray,
    mode: AlignmentMode,
    out_size: int,
) -> np.ndarray:
    """
    Align the face in `frame_rgb` using the 68 iBUG landmarks `face68` and
    return an RGB uint8 crop of shape (out_size, out_size, 3).

    Args:
        frame_rgb: (H, W, 3) uint8 RGB image.
        face68:    (68, 2) or (68, 3) float array of iBUG landmarks in pixel
                   coordinates on the source frame.
        mode:      "bbox" for EmoNet, "5pt" for HSEmotion.
        out_size:  output square size (EmoNet: 256; HSEmotion: 224 for b0).

    Raises ValueError on degenerate input (e.g. a collapsed bbox).
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"frame_rgb must be (H, W, 3); got {frame_rgb.shape}")
    if frame_rgb.dtype != np.uint8:
        raise ValueError(f"frame_rgb must be uint8; got {frame_rgb.dtype}")

    if mode == "bbox":
        bb = face_bbox_from_landmarks(face68)
        mat = emonet_bbox_transform(bb, out_size=out_size)
    elif mode == "5pt":
        src5 = derive_5pt_from_ibug68(face68)
        dst5 = canonical_5pt(out_size)
        mat = similarity_transform(src5, dst5)
    else:
        raise ValueError(f"unknown alignment mode {mode!r}")

    aligned = cv2.warpAffine(
        frame_rgb,
        mat,
        dsize=(out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned
