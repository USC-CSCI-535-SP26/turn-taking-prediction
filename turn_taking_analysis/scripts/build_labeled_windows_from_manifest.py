#!/usr/bin/env python3
"""
build_labeled_windows_from_manifest.py

Build per-sample feature windows and per-τ labels for the 3-class turn-taking
classifier. Consumes VAD + transcript + OpenFace + WavLM artifacts on disk
and emits .npy feature pairs + per-τ labels JSONs ready for the notebook.

This script is the single source of truth for how sample points are selected,
how features are sliced, and how classes are assigned. Every threshold and
definition it uses is documented inline (see the caps-constants block) and
cross-referenced to spec.md §11 (operational label definitions).

Framing
-------
Stride-based sampling, horizon-centric (VAP-style — Ekstedt & Skantze 2022,
Skantze 2017 lineage). For each perspective participant A at each sample
time t:

    input features:  [t - WINDOW_S, t]          # τ-invariant, 2 s of past
    label horizon:   [t, t + τ]                 # τ-dependent, future

The same t produces different labels at different τ. Feature files are
written ONCE per sample; labels are written per-τ into separate JSONs.

Classification (see spec.md §11.4 for authoritative tree)
---------------------------------------------------------
At each (t, τ):
    1. Prerequisite: A is voiced at t.
    2. Build B's VAD segments + transcript words clipped to horizon.
    3. If a substantive-B utterance (dur ≥ 700 ms OR ≥ 3 words OR
       non-BC vocab) fires inside horizon:
         - A ceases within 300 ms of B's start              → YIELD
         - A doesn't cease AND B voiced at t+τ              → INTERRUPT
         - A doesn't cease AND B ends before t+τ            → FAILED
    4. Else if a BC-qualifying B utterance fires AND A voiced at t+τ:
                                                             → BACKCHANNEL
    5. Else if A voiced at t+τ                               → HOLD
    6. Else                                                  → LAPSE

Classes 0/1/2 are used in training; 3/4/5 are excluded from training but
stored in the same labels JSON for exclusion-stat reporting.

Output layout (ablation-friendly, both roles materialized; concat at load)
-------------------------------------------------------------------------
Each emitted sample is a (speaker, listener) pair for one time point in
one dyad. The "speaker" role is the perspective participant — the
current floor holder at time t, the one whose turn behavior is being
predicted. The "listener" is the partner. Features for each role are
stored separately; the notebook can mix-and-match at load time.

For MM-VAP-/Hisada-2024-style dyadic-concat experiments (lit review
items 1.1, 1.4), the notebook's `SequenceDataset` / `BimodalDataset`
classes (Cell 10) accept a length-2 `path_list` per modality slot:
they read from TWO role dirs with identical basenames and concatenate
along the feature axis at __getitem__ time — so no `both/` dir is
materialized here. This saves ~600 MB of disk per run and keeps the
script's on-disk contract minimal. Convention: `path_list[0]` =
speaker, `path_list[1]` = listener. See spec §11.8 for the full
contract. (An earlier spec draft proposed a standalone
`ConcatFEATDataset` class for this; the capability was inlined into
the two existing Dataset classes during implementation — behavior is
identical.)

    model_input/
      openface/
        speaker/{train,val,test}/{basename}.npy     # (60, 20)  — speaker's face
        listener/{train,val,test}/{basename}.npy    # (60, 20)  — listener's face
      wavlm/
        speaker/{train,val,test}/{basename}.npy     # (20, 768) — speaker's audio
        listener/{train,val,test}/{basename}.npy    # (20, 768) — listener's audio
      labels/
        labels_tau_0100.json                         # {basename: 0..5}
        labels_tau_0200.json
        labels_tau_0400.json                         # ← training label file
        labels_tau_0500.json
        labels_tau_0800.json
        labels_tau_1600.json
      manifest.json                                  # config + counts

Basename: {speaker_file_id}_t_{time_ms:07d}. The SPEAKER's file_id is
the perspective tag; a sample at the same t where B is the floor holder
produces a DIFFERENT basename with B's file_id. Both orientations
coexist in every split.

Notebook-side ablation recipes. All rows use the same two Dataset
classes — SequenceDataset and BimodalDataset — parameterized by a
`path_list` of length 1 or 2 per modality slot. Length-1 slots load
a single .npy per sample (rows 1-6). Length-2 slots load from both
paths (convention: [speaker, listener]) and concatenate along the
feature axis at __getitem__ time (rows 7-10 and row 9' on the concat
side). FEAT_V / FEAT_A constants are the effective post-concat dim
a downstream GRU sees and toggle only when a slot is length-2:

    1. Dyadic cross-pair (primary, our framing):
         V = openface/listener                             FEAT_V = 20
         A = wavlm/speaker                                 FEAT_A = 768

    2. Monadic baseline (same-speaker):
         V = openface/speaker                              FEAT_V = 20
         A = wavlm/speaker                                 FEAT_A = 768

    3. Unimodal audio-only:
         A = wavlm/speaker                                 FEAT_A = 768

    4. Unimodal listener-face:
         V = openface/listener                             FEAT_V = 20

    5. Unimodal speaker-face:
         V = openface/speaker                              FEAT_V = 20

    6. Permuted-dyad control (spec §8.2):
         V = openface/listener [shuffled across test]
         A = wavlm/speaker                                 FEAT_A = 768

    7. Both-face + speaker-audio (MM-VAP visual ablation):
         V = [openface/speaker, openface/listener]         FEAT_V = 40
         A = [wavlm/speaker]                               FEAT_A = 768

    8. Listener-face + both-audio (VAP-style stereo audio):
         V = [openface/listener]                           FEAT_V = 20
         A = [wavlm/speaker, wavlm/listener]               FEAT_A = 1536

    9. MM-VAP-style full dyadic (direct replication, Russell & Harte 2025):
         V = [openface/speaker, openface/listener]         FEAT_V = 40
         A = [wavlm/speaker, wavlm/listener]               FEAT_A = 1536

    10. Speaker-face + both-audio:
          V = [openface/speaker]                           FEAT_V = 20
          A = [wavlm/speaker, wavlm/listener]              FEAT_A = 1536

    9'. Permuted-dyad control on row 9 (spec §11.7):
          V = [openface/speaker, openface/listener (listener half shuffled at test)]
          A = [wavlm/speaker,    wavlm/listener    (listener half shuffled at test)]
          Shared permutation σ across V and A.                FEAT_V=40 FEAT_A=1536

A length-1 path_list loads a single .npy; a length-2 path_list loads
from both paths and concatenates on the feature axis. Rows 1-6 are
length-1 on both slots; rows 7-10 and row 9' use length-2 on at least
one slot.

MAX_LEN_V=60, MAX_LEN_A=20, NUM_CLASSES=3 stay unchanged across every
recipe. GRU constructors in the notebook take feat_v / feat_a as args
and adapt automatically when FEAT_V / FEAT_A change.

Reuse
-----
Shares helpers with csci535-project/scripts/:
    extract_interaction_id, extract_participant_id
    (sys.path trick matching sibling manifest-driven scripts)

The merge_into_turns function from generate_timestamps_by_turn.py is NOT
used in the core loop — the stride-based framing only needs per-participant
VAD, not merged per-dyad turns. The merge is retained for an optional
sanity-check pass that validates our boundary detection against that
function's output (see --sanity-check).

Usage
-----
    # Default — build all splits, all τ values
    python build_labeled_windows_from_manifest.py

    # Only val split, useful for debugging
    python build_labeled_windows_from_manifest.py --filter-split val

    # Print sample counts without writing anything
    python build_labeled_windows_from_manifest.py --dry-run

    # Override any of the threshold constants
    python build_labeled_windows_from_manifest.py \\
        --stride-ms 250 \\
        --substantive-duration-ms 600 \\
        --bc-max-duration-ms 400

Exit codes
----------
    0 — all dyads processed successfully.
    1 — hard setup error (missing manifest, inputs, etc.).
    2 — one or more dyads failed during processing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
# pandas is imported lazily inside load_openface_csv — it's the only consumer
# and making it lazy keeps `--help` working even in environments where pandas
# isn't installed.


# -----------------------------------------------------------------------------
# Shared helpers via sys.path (same pattern as every other manifest-driven
# script in this project).
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


# =============================================================================
# CAPS CONSTANTS — authoritative values (mirrors spec.md §11.5).
# Every constant here has an inline comment justifying its value so a future
# reader can edit with full context.
# =============================================================================

# ----- Sampling ------------------------------------------------------------
STRIDE_MS = 500
# Spacing between consecutive sample points WITHIN a perspective participant's
# voiced regions. 500 ms is the Gravano & Hirschberg 2011 canonical "clean"
# inter-turn interval; below this we'd sample inside sub-event dynamics,
# above this we'd start missing short BC/YIELD events. 500 ms also matches
# the BC duration ceiling — adjacent samples can't both fall inside one BC.

WINDOW_S = 2.0
# Input feature-context length in seconds. Matches VAP's native 2 s
# projection window (Ekstedt & Skantze 2022) and MM-VAP (Russell & Harte
# 2025). Long enough for prosodic turn-preparation contour (200–800 ms)
# with surrounding context; short enough to fit comfortably in GRU memory
# on T4. Sliced into MAX_LEN_V frames (OpenFace) + MAX_LEN_A frames (WavLM).

# ----- τ grid --------------------------------------------------------------
TRAIN_TAU_MS = 400
# The single τ used to build training and validation labels. Inoue et al.
# 2024 (IWSDS) report 400 ms as the operationally-useful anticipation
# horizon in streaming-deployed VAP. Sits near the geometric center of the
# τ grid — training at grid-middle lets us measure degradation symmetrically
# toward both ends of the sweep. Changing this biases the degradation
# curve's shape, not the methodology.

TAU_GRID_MS = (100, 200, 400, 500, 800, 1600)
# Evaluation horizon grid. 100/200 probe sub-psycholinguistic-gap range
# (Levinson & Torreira 2015: 200 ms modal inter-turn gap). 400 is the
# operational Inoue 2024 horizon (also the training τ). 500 is the
# MM-VAP commensurability point (Russell & Harte 2025 single-τ hold/shift).
# 800 is double the training τ. 1600 approaches the outer boundary past
# which the 2 s context has low predictive content. Log-spaced with two
# anchored points is standard since Skantze 2017.

# ----- Substantive-B OR-gate (triggers YIELD / INTERRUPT / FAILED) ---------
SUBSTANTIVE_DURATION_MS = 700
# Leg 1 of the substantive-B OR-gate. B-utterance clipped to horizon with
# duration ≥ 700 ms is substantive regardless of word content. 700 ms is
# ≥ 2 content-word durations (Levelt 1989 mean content-word ≈ 250–350 ms);
# sub-second threshold keeps the check responsive at short τ.

SUBSTANTIVE_WORD_COUNT = 3
# Leg 2 of the substantive-B OR-gate. ≥ 3 words is substantive regardless
# of duration or vocabulary. 3+ words exceeds the BC grammatical envelope
# (Gravano 2011, Truong & Heylen 2010). 1–2 word utterances remain in the
# BC-qualifying branch; the 2↔3 gap is the BC/substantive discriminator.

# Leg 3 of the OR-gate is implicit: any clipped B-utterance containing a
# non-BC-lexicon word is substantive. Implemented as
# `any(normalize(w) not in BC_LEXICON for w in words)`.

# ----- BC-qualifying test (triggers BACKCHANNEL) --------------------------
BC_MAX_DURATION_MS = 500
# Upper bound for BC-qualifying clipped-utterance duration. 500 ms is the
# canonical single-filler BC ceiling across English corpora (Ward 2000,
# Gravano 2011). Prosodically BCs are 1–3 syllables; 500 ms ≈ 2 syllables
# at normal rate. Longer utterances are structurally responses, not BCs.

BC_MAX_WORD_COUNT = 2
# Maximum word count for the "auto-qualify" leg of BC-qualifying. ≤ 2
# words auto-qualify (catches "oh yeah", "mm okay", "I see", "no kidding").
# ≥ 3 words must additionally pass the all-BC-lexicon check. Closes the
# gap with SUBSTANTIVE_WORD_COUNT = 3 so the two definitions are disjoint.

# ----- YIELD tolerance -----------------------------------------------------
A_YIELD_TOLERANCE_MS = 300
# Max A-continues-past-B-start overlap permitted for a clean YIELD.
# Heldner & Edlund 2010: modal overlap at clean transitions < 50 ms;
# distribution tails extend to ~300 ms before overlap looks competitive.
# Within 300 ms of B's substantive start, A still counts as yielding;
# beyond this, A is contesting the floor → INTERRUPT or FAILED.

# ----- Feature alignment ---------------------------------------------------
OPENFACE_FPS = 30
# OpenFace extraction rate. Seamless video is 30 fps (paper §3.4 —
# 1920×1080 H.264 at 30 fps). Every row of the OpenFace CSV represents
# 1/30 s ≈ 33.33 ms of video. An earlier draft of this constant used 25
# fps by mistake, which produced a systematic 20% time-shift in visual
# feature slicing (frame_i at time i/25 but actually at i/30). Corrected
# after the dry-run warnings showed every dyad's OF duration exceeding
# WavLM's by exactly 1.20× — the 25:30 ratio.

WAVLM_HZ = 10
# WavLM post-pool feature rate. WavLM-base+ emits 50 Hz features; we
# mean-pool ×5 (see extract_wavlm_from_manifest.py) → 10 Hz. This matches
# the 10 Hz alignment granularity used across the pipeline.

MAX_LEN_V = int(WINDOW_S * OPENFACE_FPS)      # 60 frames (2 s × 30 fps)
MAX_LEN_A = int(WINDOW_S * WAVLM_HZ)          # 20 frames (2 s × 10 Hz)
# Derived sequence lengths. The notebook's MAX_LEN_V constant needs to
# match (bump from 50 → 60 when the notebook points at our output —
# see spec §6.2.c).

FEAT_V = 20
# OpenFace column count after subsetting: 17 AU intensities (AU01_r..AU45_r)
# + 3 head-pose rotation angles (pose_Rx/Ry/Rz). Matches the notebook's
# FEAT_V default. Lit review Item 3 ranks features AUs > head-pose > gaze;
# we include the top two families.

FEAT_A = 768
# WavLM-base+ hidden dim (penultimate layer). Unchanged from the notebook.

# ----- Class integers ------------------------------------------------------
CLASS_HOLD = 0
CLASS_YIELD = 1
CLASS_BACKCHANNEL = 2
CLASS_INTERRUPT = 3        # excluded from training; retained for stats
CLASS_FAILED = 4           # excluded from training; retained for stats
CLASS_LAPSE = 5            # excluded from training; retained for stats

TRAINING_CLASSES = {CLASS_HOLD, CLASS_YIELD, CLASS_BACKCHANNEL}
EXCLUSION_CLASSES = {CLASS_INTERRUPT, CLASS_FAILED, CLASS_LAPSE}

CLASS_NAME = {
    CLASS_HOLD:        "HOLD",
    CLASS_YIELD:       "YIELD",
    CLASS_BACKCHANNEL: "BACKCHANNEL",
    CLASS_INTERRUPT:   "INTERRUPT",
    CLASS_FAILED:      "FAILED",
    CLASS_LAPSE:       "LAPSE",
}

# ----- Class imbalance strategy --------------------------------------------
# NO data-level rebalancing is performed here. All splits (train, val, test)
# are emitted at natural rate. Per-class imbalance is handled at training
# time via inverse-frequency class weights in nn.CrossEntropyLoss — see the
# notebook's weight-computation cell. Rationale: with 467 dyads we have
# enough minority-class (BC) examples that loss-level weighting preserves
# more signal than majority-class subsampling, and avoids a train/test
# prior mismatch. History: an earlier POC-era version of this script
# HOLD-subsampled train/val to target 45/40/15 ratios; dropped at full-
# project scale. See docs/things_to_fix.md #1 and docs/spec.md §5 task 3.

# ----- Backchannel vocabulary ---------------------------------------------
BC_LEXICON = frozenset({
    # non-lexical acknowledgement tokens
    #   Clancy, Thompson, Suzuki & Tao 1996 (cross-linguistic reactive tokens)
    #   Ward & Tsukahara 2000 (Interspeech, English/Japanese BC cues)
    #   Lala, Inoue & Kawahara 2017 (SIGDIAL, responsive tokens)
    #   Tolins & Fox Tree 2014 (J. Pragmatics, generic vs. specific BC)
    "mm", "hm", "hmm", "mhm", "mmhmm",
    "uhhuh", "huh",
    "ah", "aha", "oh",
    # short affirmatives
    #   Gravano & Hirschberg 2011 (Comp. Speech & Lang., Games Corpus)
    #   Ward & Tsukahara 2000
    #   Jurafsky et al. 1997 (SWBD-DAMSL "bf" backchannel tag)
    "yeah", "yep", "yup", "ok", "okay", "sure",
    "right", "true", "exactly", "totally",
    # short assessments / reactives
    #   Ruede, Müller, Stüker & Waibel 2017 (Interspeech, deep BC predictor)
    #   Lala, Inoue & Kawahara 2017
    "wow", "really", "gotcha",
})
# After normalization (lowercase, strip punct, collapse ws/hyphens), lookup
# is O(1). Same list lives in spec.md §11.6 — keep in sync by manual edit.


# =============================================================================
# IO helpers
# =============================================================================

DEFAULT_MANIFEST_PATH = str(
    PROJECT_ROOT / "turn_taking_analysis" / "manifests" / "manifest.csv"
)
DEFAULT_OUTPUT_ROOT = str(
    PROJECT_ROOT / "turn_taking_analysis" / "model_input"
)
DEFAULT_VAD_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "vad"
)
DEFAULT_TRANSCRIPT_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "transcript"
)
DEFAULT_OPENFACE_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "openface"
)
DEFAULT_WAVLM_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "wavlm"
)

# OpenFace columns we keep (17 AU_r + 3 pose_R). Paper §A.1 + spec Appendix A.
OPENFACE_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
    "pose_Rx", "pose_Ry", "pose_Rz",
]
assert len(OPENFACE_COLUMNS) == FEAT_V, \
    f"OPENFACE_COLUMNS has {len(OPENFACE_COLUMNS)} cols but FEAT_V={FEAT_V}"


def load_manifest(path: str) -> list[dict]:
    """Load manifest.csv into a list of dict rows. Fails fast on missing
    file or missing required columns — schema drift must surface immediately."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found at {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "split", "interaction_id",
            "file_id_a", "file_id_b",
            "participant_a", "participant_b",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest {path} has no rows")
    return rows


def load_vad_jsonl(path: str) -> list[dict]:
    """Load a Silero VAD JSONL → list of {'start': float, 'end': float}.
    Missing/empty file is fatal — VAD is mandatory per our download-script
    strict-vad policy."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"VAD file missing or empty: {path}")
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments.append(json.loads(line))
    # Sort by start defensively — Silero output is usually ordered, but
    # enforce it here so downstream binary-search assumptions hold.
    segments.sort(key=lambda s: s["start"])
    return segments


def load_transcript_jsonl(path: str) -> list[dict]:
    """Load a WhisperX transcript JSONL → list of segment dicts. Each has
    `words` (list of {'word','start','end','score'}), `start`, `end`,
    `transcript`. Missing file returns []: transcripts are OPTIONAL per
    Seamless release (paper Appendix A.1.4 p.54). Empty list flows through
    the classifier as "no words in horizon" and biases toward HOLD/LAPSE."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError:
                # Defensive — skip malformed lines rather than failing.
                continue
    return segments


def extract_words_from_transcript(segments: list[dict]) -> list[dict]:
    """Flatten a list of WhisperX segment dicts into a flat word list.
    Each word: {'word': str, 'start': float, 'end': float, 'score': float}.
    Words without valid start/end are dropped (WhisperX occasionally emits
    unaligned words at the end of a segment)."""
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            if "start" in w and "end" in w and w["start"] is not None:
                words.append(w)
    words.sort(key=lambda w: w["start"])
    return words


def load_openface_csv(path: str) -> np.ndarray:
    """Load an OpenFace CSV and slice to the 20-column subset.
    Returns (T_frames, FEAT_V) float32. Confidence / success columns are
    not consumed here — tracker-success filtering is a per-dyad flag in
    manifest.json, not per-frame dropping."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"OpenFace CSV missing: {path}")
    import pandas as pd  # lazy — see note at top of file
    df = pd.read_csv(path)
    # OpenFace prepends a space to column names; our upstream extractor
    # already stripped them (see extract_openface_from_manifest.py), but
    # re-strip defensively.
    df.columns = [c.strip() for c in df.columns]
    missing_cols = [c for c in OPENFACE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"OpenFace CSV {path} missing columns: {missing_cols}")
    arr = df[OPENFACE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    return arr


def load_wavlm_npy(path: str) -> np.ndarray:
    """Load a WavLM .npy. Expected shape (T_10hz, 768) float32."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"WavLM .npy missing: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != FEAT_A:
        raise ValueError(
            f"WavLM .npy {path} has unexpected shape {arr.shape}; "
            f"expected (T, {FEAT_A})"
        )
    return arr.astype(np.float32, copy=False)


# =============================================================================
# Primitive time helpers
# =============================================================================

def is_voiced_at(t_query: float, vad_segments: list[dict]) -> bool:
    """Return True iff any VAD segment contains t_query (closed interval).
    Linear scan is fine at Silero segment counts (~10^2 per recording)."""
    for seg in vad_segments:
        if seg["start"] <= t_query <= seg["end"]:
            return True
    return False


def current_turn_end(t: float, vad_segments: list[dict]) -> float | None:
    """If t is inside a VAD segment, return that segment's end (seconds).
    Else return None. Used for the A-yield-tolerance check."""
    for seg in vad_segments:
        if seg["start"] <= t <= seg["end"]:
            return seg["end"]
    return None


def clip_segment_to_horizon(
    seg: dict, t: float, tau: float
) -> tuple[float, float] | None:
    """Clip a VAD segment to the horizon [t, t+tau]. Returns (clipped_start,
    clipped_end) in seconds, or None if the segment doesn't intersect
    the horizon. See spec §11.1."""
    clipped_start = max(seg["start"], t)
    clipped_end = min(seg["end"], t + tau)
    if clipped_start >= clipped_end:
        return None
    return (clipped_start, clipped_end)


# =============================================================================
# Word normalization and BC lexicon lookup
# =============================================================================

_PUNCT_RE = re.compile(r"[^\w\s-]")
_WS_HYPHEN_RE = re.compile(r"[\s\-]+")


def normalize_word(raw: str) -> str:
    """Apply the spec §11.6 normalization pipeline:
        lowercase → strip leading/trailing punct → collapse internal
        whitespace and hyphens.
    Result: 'Mm-hmm!' / 'mm hmm' / 'mmhmm' all → 'mmhmm'.
    """
    s = raw.lower().strip()
    s = _PUNCT_RE.sub("", s)       # drop punctuation
    s = _WS_HYPHEN_RE.sub("", s)   # collapse whitespace and hyphens
    return s


def is_bc_token(word: str) -> bool:
    """True iff the normalized word is in BC_LEXICON."""
    return normalize_word(word) in BC_LEXICON


# =============================================================================
# Per-horizon B-utterance construction
# =============================================================================

class BUtterance:
    """One B-VAD segment clipped to a given horizon, with the subset of
    B-transcript words whose start falls inside the clipped interval.

    Attributes:
        clipped_start, clipped_end  — seconds; clipped to horizon
        duration_ms                 — 1000 * (end - start)
        words                       — list of word-dicts from transcript
        raw_segment                 — the original VAD dict (for later
                                      INTERRUPT/FAILED distinction — the
                                      raw end is compared to t+τ to decide
                                      if B "continues past horizon")
    """

    __slots__ = ("clipped_start", "clipped_end", "duration_ms",
                 "words", "raw_segment")

    def __init__(self, clipped_start, clipped_end, words, raw_segment):
        self.clipped_start = clipped_start
        self.clipped_end = clipped_end
        self.duration_ms = (clipped_end - clipped_start) * 1000.0
        self.words = words
        self.raw_segment = raw_segment


def build_b_utterances_in_horizon(
    t: float, tau: float,
    b_vad: list[dict],
    b_words: list[dict],
) -> list[BUtterance]:
    """For a given sample point t and horizon length tau, build the list
    of B's clipped-to-horizon utterances. Each utterance:
      - has a clipped VAD interval ⊆ [t, t+tau]
      - carries the B-transcript words whose word['start'] ∈ clipped interval

    Returns utterances sorted by clipped_start."""
    utterances = []
    for seg in b_vad:
        clipped = clip_segment_to_horizon(seg, t, tau)
        if clipped is None:
            continue
        c_start, c_end = clipped
        # Attach words whose start falls in the clipped interval. Using
        # word-start as the anchor (word-end can extend past τ without
        # disqualification — transcription boundaries are noisier than VAD).
        u_words = [
            w for w in b_words
            if c_start <= w["start"] <= c_end
        ]
        utterances.append(BUtterance(c_start, c_end, u_words, seg))
    utterances.sort(key=lambda u: u.clipped_start)
    return utterances


# =============================================================================
# Substantive / BC-qualifying tests
# =============================================================================

def is_substantive(u: BUtterance,
                   substantive_duration_ms: float,
                   substantive_word_count: int) -> bool:
    """The 3-legged OR-gate from spec §11.2 YIELD definition. Any one leg
    qualifies B's clipped utterance as substantive.

        Leg 1:  duration ≥ SUBSTANTIVE_DURATION_MS
        Leg 2:  word count ≥ SUBSTANTIVE_WORD_COUNT
        Leg 3:  any word not in BC_LEXICON
    """
    if u.duration_ms >= substantive_duration_ms:
        return True
    if len(u.words) >= substantive_word_count:
        return True
    if u.words and any(not is_bc_token(w["word"]) for w in u.words):
        return True
    return False


def is_bc_qualifying(u: BUtterance,
                     bc_max_duration_ms: float,
                     bc_max_word_count: int) -> bool:
    """BACKCHANNEL qualification per spec §11.2.
        Duration ≤ BC_MAX_DURATION_MS
        AND (word-count ≤ BC_MAX_WORD_COUNT OR all words in BC_LEXICON)
    """
    if u.duration_ms > bc_max_duration_ms:
        return False
    if len(u.words) <= bc_max_word_count:
        return True
    # ≥ 3 words case: require every word to be BC-lexicon.
    return all(is_bc_token(w["word"]) for w in u.words)


# =============================================================================
# Per-(t, τ) classification
# =============================================================================

def classify_sample(
    t: float,
    tau: float,
    a_vad: list[dict],
    b_vad: list[dict],
    b_words: list[dict],
    *,
    substantive_duration_ms: float = SUBSTANTIVE_DURATION_MS,
    substantive_word_count: int = SUBSTANTIVE_WORD_COUNT,
    bc_max_duration_ms: float = BC_MAX_DURATION_MS,
    bc_max_word_count: int = BC_MAX_WORD_COUNT,
    a_yield_tolerance_s: float = A_YIELD_TOLERANCE_MS / 1000.0,
) -> int:
    """Implement the precedence-ordered decision tree from spec §11.4.
    Returns one of the 6 class integers. Prerequisite (a voiced at t) is
    caller's responsibility — this function assumes it."""
    # Step 1: build B's clipped utterances in horizon.
    b_utts = build_b_utterances_in_horizon(t, tau, b_vad, b_words)

    # Step 2: find first substantive-B utterance (precedence: earliest start).
    first_sub = next(
        (u for u in b_utts
         if is_substantive(u, substantive_duration_ms, substantive_word_count)),
        None,
    )

    # Step 3: substantive-B branch — YIELD / INTERRUPT / FAILED.
    if first_sub is not None:
        # A's currently-active turn endpoint. If A isn't voiced at t the
        # caller shouldn't have called us; assert defensively.
        a_turn_end = current_turn_end(t, a_vad)
        assert a_turn_end is not None, \
            "classify_sample called when A not voiced at t (prerequisite)"

        if a_turn_end <= first_sub.clipped_start + a_yield_tolerance_s:
            return CLASS_YIELD

        # A keeps going past tolerance → floor contest.
        # Distinguish INTERRUPT vs FAILED using B's state at horizon end.
        #   INTERRUPT: B still voiced at t+τ (B winning / unresolved)
        #   FAILED:    B ends in horizon AND A voiced at t+τ (A keeps floor)
        b_voiced_at_tau = is_voiced_at(t + tau, b_vad)
        a_voiced_at_tau = is_voiced_at(t + tau, a_vad)
        if b_voiced_at_tau:
            return CLASS_INTERRUPT
        if a_voiced_at_tau:
            return CLASS_FAILED
        # Both silent at horizon end despite prior contest — degenerate.
        # Treat as INTERRUPT (unresolved state, not a clean FAILED).
        return CLASS_INTERRUPT

    # Step 4: no substantive-B. BACKCHANNEL if qualifying AND A still voiced.
    a_voiced_at_tau = is_voiced_at(t + tau, a_vad)
    if a_voiced_at_tau and any(
        is_bc_qualifying(u, bc_max_duration_ms, bc_max_word_count)
        for u in b_utts
    ):
        return CLASS_BACKCHANNEL

    # Step 5: HOLD if A still voiced.
    if a_voiced_at_tau:
        return CLASS_HOLD

    # Step 6: fall-through. A stopped, no substantive-B. LAPSE.
    return CLASS_LAPSE


# =============================================================================
# Feature slicing
# =============================================================================

def _slice_fixed_length(arr: np.ndarray, frame_start: int,
                        frame_end_exclusive: int, feat_dim: int) -> np.ndarray:
    """Slice arr[frame_start:frame_end_exclusive] and zero-pad to a fixed
    length on the right if the slice is short. Returns (target_len, feat_dim)
    float32 where target_len = frame_end_exclusive - frame_start.

    Used by both OpenFace and WavLM slicers — the logic is identical, only
    the rate + shape differ."""
    target_len = frame_end_exclusive - frame_start
    T = arr.shape[0]
    lo = max(frame_start, 0)
    hi = min(frame_end_exclusive, T)
    if lo >= hi:
        # Entire slice is past-end or pre-start — return all-zeros. Caller
        # should have skipped via prerequisites; log defensively.
        return np.zeros((target_len, feat_dim), dtype=np.float32)
    core = arr[lo:hi].astype(np.float32, copy=False)
    pad_left = lo - frame_start
    pad_right = frame_end_exclusive - hi
    if pad_left == 0 and pad_right == 0:
        return core
    return np.pad(core, ((pad_left, pad_right), (0, 0)), mode="constant")


def slice_openface(arr: np.ndarray, t: float) -> np.ndarray:
    """Return (MAX_LEN_V, FEAT_V) float32 — 2 s of OpenFace ending at t."""
    frame_start = int(round((t - WINDOW_S) * OPENFACE_FPS))
    frame_end = frame_start + MAX_LEN_V
    return _slice_fixed_length(arr, frame_start, frame_end, FEAT_V)


def slice_wavlm(arr: np.ndarray, t: float) -> np.ndarray:
    """Return (MAX_LEN_A, FEAT_A) float32 — 2 s of WavLM ending at t."""
    frame_start = int(round((t - WINDOW_S) * WAVLM_HZ))
    frame_end = frame_start + MAX_LEN_A
    return _slice_fixed_length(arr, frame_start, frame_end, FEAT_A)


# =============================================================================
# Per-interaction, per-participant pipeline
# =============================================================================

class SampleOutput:
    """One emitted sample. Feature files are written to disk; labels are
    accumulated in memory and written out per-τ at the end of the run."""

    __slots__ = ("basename", "split", "file_id", "t_ms",
                 "labels_by_tau", "low_tracking")

    def __init__(self, basename, split, file_id, t_ms):
        self.basename = basename
        self.split = split
        self.file_id = file_id
        self.t_ms = t_ms
        # {tau_ms: class_int} for each tau in TAU_GRID_MS
        self.labels_by_tau: dict[int, int] = {}
        # Set post-hoc from OpenFace sidecar json; reportable in manifest.
        self.low_tracking: bool = False


def enumerate_sample_times(
    a_vad: list[dict],
    file_duration: float,
    stride_ms: int,
    tau_grid_ms: tuple,
) -> list[float]:
    """Enumerate candidate sample times t for a perspective participant A.

    A time t is a candidate iff:
        t - WINDOW_S >= 0                          (2 s of past context fits)
        t + max(tau) <= file_duration              (longest horizon fits)
        is_voiced_at(t, a_vad)                     (A is the floor holder)

    Samples are evenly spaced at `stride_ms` intervals starting from the
    first candidate time >= WINDOW_S. The stride is applied globally
    (across the whole recording) rather than per-turn — which gives us
    deterministic sample placement independent of turn boundaries.
    A's VAD acts as a filter: non-voiced strides are silently skipped."""
    t_min = WINDOW_S
    t_max = file_duration - max(tau_grid_ms) / 1000.0
    if t_max <= t_min:
        return []
    step = stride_ms / 1000.0
    # Build a list of stride points, then filter by A's voicing. (Doing
    # the filter separately keeps enumerate_sample_times readable; the
    # alternative is to inline is_voiced_at here for a micro-speedup that
    # doesn't matter at our scale.)
    candidates = []
    t = t_min
    while t <= t_max:
        if is_voiced_at(t, a_vad):
            candidates.append(t)
        t += step
    return candidates


def process_one_participant(
    *,
    split: str,
    dyad_id: str,
    speaker_file_id: str,
    listener_file_id: str,
    speaker_vad: list[dict],
    listener_vad: list[dict],
    listener_words: list[dict],
    speaker_openface: np.ndarray,
    speaker_wavlm: np.ndarray,
    listener_openface: np.ndarray,
    listener_wavlm: np.ndarray,
    speaker_low_tracking: bool,
    listener_low_tracking: bool,
    output_root: str,
    stride_ms: int,
    tau_grid_ms: tuple,
    substantive_duration_ms: float,
    substantive_word_count: int,
    bc_max_duration_ms: float,
    bc_max_word_count: int,
    a_yield_tolerance_s: float,
    dry_run: bool,
) -> list[SampleOutput]:
    """Emit samples for one perspective participant (the 'speaker' role)
    within a dyad. Both the speaker's and the listener's features are
    saved to disk — so the notebook can mix-and-match at load time for
    ablation recipes without regenerating windows.

    For each sample point t passing prerequisites:
      - Slice OpenFace + WavLM features ending at t for BOTH roles.
      - Save four .npy files (speaker/openface, speaker/wavlm,
        listener/openface, listener/wavlm) under identical basenames so
        the notebook's by-filename pairing still works.
      - Classify per τ using the speaker's VAD + the listener's VAD and
        transcript (classification is dyadic; only the features vary by
        role).

    Feature/label consistency: usable file_duration = min over the four
    arrays' durations (OpenFace frame count / 25 fps, WavLM frame count
    / 10 Hz, for both participants). Guarantees all four slices have
    coverage without trailing zeros polluting features."""
    speaker_of_duration = speaker_openface.shape[0] / OPENFACE_FPS
    speaker_wl_duration = speaker_wavlm.shape[0] / WAVLM_HZ
    listener_of_duration = listener_openface.shape[0] / OPENFACE_FPS
    listener_wl_duration = listener_wavlm.shape[0] / WAVLM_HZ
    file_duration = min(
        speaker_of_duration, speaker_wl_duration,
        listener_of_duration, listener_wl_duration,
    )

    sample_times = enumerate_sample_times(
        a_vad=speaker_vad,
        file_duration=file_duration,
        stride_ms=stride_ms,
        tau_grid_ms=tau_grid_ms,
    )

    outputs: list[SampleOutput] = []

    # Four output directories — one per (modality, role) pair. Identical
    # basenames across all four; the notebook picks the (modality, role)
    # combination via FEAT_PATH_V / FEAT_PATH_A at load time. Dyadic
    # concatenation (speaker|listener along feature axis) is NOT
    # materialized here — it's synthesized on the fly in the notebook via
    # a ConcatFEATDataset wrapper that reads from two of these dirs and
    # concats at __getitem__ time. Saves ~600 MB on disk vs. pre-writing
    # a `both/` subdir; see spec §11.7 for the wrapper contract.
    of_spk_dir = os.path.join(output_root, "openface", "speaker", split)
    of_lst_dir = os.path.join(output_root, "openface", "listener", split)
    wl_spk_dir = os.path.join(output_root, "wavlm", "speaker", split)
    wl_lst_dir = os.path.join(output_root, "wavlm", "listener", split)
    if not dry_run:
        for d in (of_spk_dir, of_lst_dir, wl_spk_dir, wl_lst_dir):
            os.makedirs(d, exist_ok=True)

    for t in sample_times:
        t_ms = int(round(t * 1000))
        # Basename tags the perspective participant (speaker role).
        # Labels.json keys on this basename. Listener features live at
        # the same basename under the listener/ subdir — role is encoded
        # in the directory path, not the filename.
        basename = f"{speaker_file_id}_t_{t_ms:07d}"

        # Slice + save features for both roles (same file reused across all τ).
        if not dry_run:
            v_spk = slice_openface(speaker_openface, t)
            a_spk = slice_wavlm(speaker_wavlm, t)
            v_lst = slice_openface(listener_openface, t)
            a_lst = slice_wavlm(listener_wavlm, t)
            assert v_spk.shape == (MAX_LEN_V, FEAT_V), v_spk.shape
            assert a_spk.shape == (MAX_LEN_A, FEAT_A), a_spk.shape
            assert v_lst.shape == (MAX_LEN_V, FEAT_V), v_lst.shape
            assert a_lst.shape == (MAX_LEN_A, FEAT_A), a_lst.shape
            np.save(os.path.join(of_spk_dir, f"{basename}.npy"), v_spk)
            np.save(os.path.join(wl_spk_dir, f"{basename}.npy"), a_spk)
            np.save(os.path.join(of_lst_dir, f"{basename}.npy"), v_lst)
            np.save(os.path.join(wl_lst_dir, f"{basename}.npy"), a_lst)

        # Classify per τ. Classification depends on the dyad's VAD +
        # listener's transcript — feature-role variation doesn't change
        # labels; the same basename gets the same label regardless of
        # which role's features we pair.
        so = SampleOutput(
            basename=basename, split=split,
            file_id=speaker_file_id, t_ms=t_ms,
        )
        # Track whichever role has low-tracking so the manifest can
        # flag per-sample issues (speaker face is only relevant if the
        # notebook configures the visual path to speaker/; listener face
        # only if listener/).
        so.low_tracking = speaker_low_tracking or listener_low_tracking
        for tau_ms in tau_grid_ms:
            tau = tau_ms / 1000.0
            cls = classify_sample(
                t=t, tau=tau,
                a_vad=speaker_vad, b_vad=listener_vad,
                b_words=listener_words,
                substantive_duration_ms=substantive_duration_ms,
                substantive_word_count=substantive_word_count,
                bc_max_duration_ms=bc_max_duration_ms,
                bc_max_word_count=bc_max_word_count,
                a_yield_tolerance_s=a_yield_tolerance_s,
            )
            so.labels_by_tau[tau_ms] = cls

        outputs.append(so)

    return outputs


def load_low_tracking_flag(openface_json_path: str, threshold: float = 0.85
                           ) -> bool:
    """Return True if this participant's OpenFace sidecar reports
    tracker_success_fraction < threshold (spec §4.2). Used as a per-dyad
    flag in manifest.json; windows are still emitted."""
    if not os.path.exists(openface_json_path):
        return False
    try:
        with open(openface_json_path) as f:
            sidecar = json.load(f)
        return sidecar.get("tracker_success_fraction", 1.0) < threshold
    except (json.JSONDecodeError, OSError):
        return False


def process_interaction(
    row: dict,
    *,
    vad_dir: str, transcript_dir: str, openface_dir: str, wavlm_dir: str,
    output_root: str, stride_ms: int, tau_grid_ms: tuple,
    substantive_duration_ms: float, substantive_word_count: int,
    bc_max_duration_ms: float, bc_max_word_count: int,
    a_yield_tolerance_s: float, dry_run: bool,
) -> dict:
    """Process one manifest row — two perspective participants × many
    samples. Returns a per-dyad result dict for the global manifest."""
    dyad_id = row["interaction_id"]
    split = row["split"]
    file_id_a = row["file_id_a"]
    file_id_b = row["file_id_b"]

    # Load all four streams for each participant. Participant A's
    # features come from A's streams when A is the perspective participant;
    # likewise for B.
    vad_a = load_vad_jsonl(os.path.join(vad_dir, f"{file_id_a}.jsonl"))
    vad_b = load_vad_jsonl(os.path.join(vad_dir, f"{file_id_b}.jsonl"))

    words_a = extract_words_from_transcript(
        load_transcript_jsonl(os.path.join(transcript_dir, f"{file_id_a}.jsonl"))
    )
    words_b = extract_words_from_transcript(
        load_transcript_jsonl(os.path.join(transcript_dir, f"{file_id_b}.jsonl"))
    )

    openface_a = load_openface_csv(os.path.join(openface_dir, f"{file_id_a}.csv"))
    openface_b = load_openface_csv(os.path.join(openface_dir, f"{file_id_b}.csv"))

    wavlm_a = load_wavlm_npy(os.path.join(wavlm_dir, f"{file_id_a}.npy"))
    wavlm_b = load_wavlm_npy(os.path.join(wavlm_dir, f"{file_id_b}.npy"))

    low_track_a = load_low_tracking_flag(
        os.path.join(openface_dir, f"{file_id_a}.json")
    )
    low_track_b = load_low_tracking_flag(
        os.path.join(openface_dir, f"{file_id_b}.json")
    )

    # Sanity-check stream durations. Mismatches > 300 ms are logged; the
    # pipeline uses the min duration so neither stream runs past its data.
    # The threshold is set at 300 ms to silence the benign WavLM-tail
    # residual: WavLM-base+'s CNN stack loses a few frames at chunk-tail
    # receptive-field boundaries, and the ×5 mean-pool truncates up to
    # ~80 ms more. Combined residual is usually 150-250 ms, well inside
    # the tolerance. Anything larger indicates a real problem (e.g. the
    # 20% systematic offset from the earlier 25-vs-30 fps bug).
    def _duration_check(name, of, wl):
        of_s = of.shape[0] / OPENFACE_FPS
        wl_s = wl.shape[0] / WAVLM_HZ
        diff = abs(of_s - wl_s)
        if diff > 0.300:
            print(f"  WARN: {dyad_id} {name}: OF={of_s:.2f}s WavLM={wl_s:.2f}s "
                  f"(|Δ|={diff*1000:.0f}ms). Using min as usable duration.",
                  file=sys.stderr)

    _duration_check(file_id_a, openface_a, wavlm_a)
    _duration_check(file_id_b, openface_b, wavlm_b)

    # Transcript presence flags — used by the manifest for exclusion stats
    # (dyads without transcripts produce biased labels).
    transcript_present_a = len(words_a) > 0
    transcript_present_b = len(words_b) > 0

    all_outputs: list[SampleOutput] = []

    # Emit from A-as-speaker perspective (B is the listener).
    # Both roles' features are saved; classification uses speaker VAD +
    # listener VAD/transcript.
    all_outputs += process_one_participant(
        split=split, dyad_id=dyad_id,
        speaker_file_id=file_id_a, listener_file_id=file_id_b,
        speaker_vad=vad_a, listener_vad=vad_b, listener_words=words_b,
        speaker_openface=openface_a, speaker_wavlm=wavlm_a,
        listener_openface=openface_b, listener_wavlm=wavlm_b,
        speaker_low_tracking=low_track_a,
        listener_low_tracking=low_track_b,
        output_root=output_root,
        stride_ms=stride_ms, tau_grid_ms=tau_grid_ms,
        substantive_duration_ms=substantive_duration_ms,
        substantive_word_count=substantive_word_count,
        bc_max_duration_ms=bc_max_duration_ms,
        bc_max_word_count=bc_max_word_count,
        a_yield_tolerance_s=a_yield_tolerance_s,
        dry_run=dry_run,
    )

    # Emit from B-as-speaker perspective (A is the listener).
    all_outputs += process_one_participant(
        split=split, dyad_id=dyad_id,
        speaker_file_id=file_id_b, listener_file_id=file_id_a,
        speaker_vad=vad_b, listener_vad=vad_a, listener_words=words_a,
        speaker_openface=openface_b, speaker_wavlm=wavlm_b,
        listener_openface=openface_a, listener_wavlm=wavlm_a,
        speaker_low_tracking=low_track_b,
        listener_low_tracking=low_track_a,
        output_root=output_root,
        stride_ms=stride_ms, tau_grid_ms=tau_grid_ms,
        substantive_duration_ms=substantive_duration_ms,
        substantive_word_count=substantive_word_count,
        bc_max_duration_ms=bc_max_duration_ms,
        bc_max_word_count=bc_max_word_count,
        a_yield_tolerance_s=a_yield_tolerance_s,
        dry_run=dry_run,
    )

    return {
        "dyad_id": dyad_id,
        "split": split,
        "n_samples": len(all_outputs),
        "transcript_present_a": transcript_present_a,
        "transcript_present_b": transcript_present_b,
        "low_tracking_a": low_track_a,
        "low_tracking_b": low_track_b,
        "outputs": all_outputs,
    }


# =============================================================================
# Orchestration
# =============================================================================

def build_per_tau_label_files(
    results: list[dict],
    output_root: str,
    tau_grid_ms: tuple,
    dry_run: bool,
) -> dict:
    """Write labels/labels_tau_XXXX.json files, one per τ. All splits are
    emitted at natural rate — class imbalance is handled at training time
    via inverse-frequency weights in the loss. Returns per-(tau, split)
    class-count dicts for the manifest."""
    labels_dir = os.path.join(output_root, "labels")
    if not dry_run:
        os.makedirs(labels_dir, exist_ok=True)

    # Collect all outputs grouped by split.
    all_outputs_by_split: dict[str, list[SampleOutput]] = defaultdict(list)
    for r in results:
        for o in r["outputs"]:
            all_outputs_by_split[o.split].append(o)

    counts_summary: dict[int, dict[str, dict[str, int]]] = {}
    for tau_ms in tau_grid_ms:
        counts_summary[tau_ms] = {}
        labels_for_tau: dict[str, int] = {}

        for split, outputs in all_outputs_by_split.items():
            counts = Counter(o.labels_by_tau[tau_ms] for o in outputs)
            counts_dict = {CLASS_NAME[k]: v for k, v in counts.items()}

            # Every sample flows through to labels.json; no subsampling.
            # EXCLUSION classes (INTERRUPT / FAILED / LAPSE) are included
            # here and filtered out by the loader at training time.
            for o in outputs:
                labels_for_tau[o.basename] = o.labels_by_tau[tau_ms]

            counts_summary[tau_ms][split] = {
                "counts": counts_dict,
                "n_samples": len(outputs),
            }

        # Write the labels JSON for this tau.
        if not dry_run:
            path = os.path.join(labels_dir, f"labels_tau_{tau_ms:04d}.json")
            with open(path, "w") as f:
                json.dump(labels_for_tau, f, indent=2, sort_keys=True)

    return counts_summary


def write_manifest(
    results: list[dict],
    counts_summary: dict,
    output_root: str,
    config: dict,
    dry_run: bool,
) -> None:
    """Write manifest.json capturing full run provenance. Serves as the
    methods-section provenance for the writeup — records every constant,
    per-(tau, split) natural-rate class counts, and per-dyad flags."""
    if dry_run:
        return
    manifest = {
        "config": config,
        "classes": {
            str(k): CLASS_NAME[k] for k in sorted(CLASS_NAME.keys())
        },
        "training_classes": sorted(TRAINING_CLASSES),
        "exclusion_classes": sorted(EXCLUSION_CLASSES),
        "bc_lexicon": sorted(BC_LEXICON),
        "per_tau_per_split_counts": {
            str(tau): v for tau, v in counts_summary.items()
        },
        "per_dyad": [
            {
                "dyad_id": r["dyad_id"],
                "split": r["split"],
                "n_samples": r["n_samples"],
                "transcript_present_a": r["transcript_present_a"],
                "transcript_present_b": r["transcript_present_b"],
                "low_tracking_a": r["low_tracking_a"],
                "low_tracking_b": r["low_tracking_b"],
            }
            for r in results
        ],
    }
    path = os.path.join(output_root, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-sample feature windows and per-τ labels for the "
            "3-class turn-taking classifier. See spec.md §11 for the "
            "authoritative label definitions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O paths
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--vad-dir", default=DEFAULT_VAD_DIR)
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--openface-dir", default=DEFAULT_OPENFACE_DIR)
    parser.add_argument("--wavlm-dir", default=DEFAULT_WAVLM_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    # Threshold overrides (all default to the CAPS constants above)
    parser.add_argument("--stride-ms", type=int, default=STRIDE_MS)
    parser.add_argument("--substantive-duration-ms", type=int,
                        default=SUBSTANTIVE_DURATION_MS)
    parser.add_argument("--substantive-word-count", type=int,
                        default=SUBSTANTIVE_WORD_COUNT)
    parser.add_argument("--bc-max-duration-ms", type=int,
                        default=BC_MAX_DURATION_MS)
    parser.add_argument("--bc-max-word-count", type=int,
                        default=BC_MAX_WORD_COUNT)
    parser.add_argument("--a-yield-tolerance-ms", type=int,
                        default=A_YIELD_TOLERANCE_MS)
    parser.add_argument("--tau-grid-ms", type=str, default=None,
                        help="Comma-separated override, e.g. "
                             "'100,200,400,800,1600'. Default: %(default)s")

    # Runtime plumbing
    parser.add_argument("--filter-split", choices=["train", "val", "test"],
                        default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute labels + counts but don't write files.")

    args = parser.parse_args()

    # Parse tau grid override.
    if args.tau_grid_ms is not None:
        try:
            tau_grid = tuple(int(x) for x in args.tau_grid_ms.split(","))
        except ValueError:
            print(f"ERROR: bad --tau-grid-ms: {args.tau_grid_ms}",
                  file=sys.stderr)
            return 1
    else:
        tau_grid = TAU_GRID_MS

    # Load manifest.
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.filter_split:
        rows = [r for r in rows if r["split"] == args.filter_split]
    if not rows:
        print("No rows to process.", file=sys.stderr)
        return 0

    print(f"Building labeled windows for {len(rows)} dyad(s)")
    print(f"  output_root:              {args.output_root}")
    print(f"  tau_grid_ms:              {tau_grid}")
    print(f"  stride_ms:                {args.stride_ms}")
    print(f"  substantive_duration_ms:  {args.substantive_duration_ms}")
    print(f"  substantive_word_count:   {args.substantive_word_count}")
    print(f"  bc_max_duration_ms:       {args.bc_max_duration_ms}")
    print(f"  bc_max_word_count:        {args.bc_max_word_count}")
    print(f"  a_yield_tolerance_ms:     {args.a_yield_tolerance_ms}")
    print(f"  class balance:            natural-rate (no subsampling); "
          f"use class weights in loss")
    print(f"  dry_run:                  {args.dry_run}")

    a_yield_tolerance_s = args.a_yield_tolerance_ms / 1000.0

    # Ensure the output root exists up front. Subdirectory makedirs later
    # in process_one_participant / build_per_tau_label_files / write_manifest
    # would create it implicitly via their recursive makedirs, but only on
    # the happy path where at least one dyad processes successfully — a
    # first-run, all-dyads-fail case otherwise leaves nothing on disk and
    # write_manifest would raise FileNotFoundError on the parent dir.
    # Creating it here also gives --dry-run a visible skeleton to inspect.
    if not args.dry_run:
        os.makedirs(args.output_root, exist_ok=True)

    start = time.time()
    results = []
    failed = 0
    for i, row in enumerate(rows, 1):
        dyad = row["interaction_id"]
        try:
            result = process_interaction(
                row=row,
                vad_dir=args.vad_dir,
                transcript_dir=args.transcript_dir,
                openface_dir=args.openface_dir,
                wavlm_dir=args.wavlm_dir,
                output_root=args.output_root,
                stride_ms=args.stride_ms,
                tau_grid_ms=tau_grid,
                substantive_duration_ms=args.substantive_duration_ms,
                substantive_word_count=args.substantive_word_count,
                bc_max_duration_ms=args.bc_max_duration_ms,
                bc_max_word_count=args.bc_max_word_count,
                a_yield_tolerance_s=a_yield_tolerance_s,
                dry_run=args.dry_run,
            )
            results.append(result)
            print(f"  [{i:3d}/{len(rows)}] {row['split']:<5} {dyad:<30} "
                  f"{result['n_samples']:>5} samples")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [{i:3d}/{len(rows)}] {row['split']:<5} {dyad:<30} "
                  f"FAILED: {e}", file=sys.stderr)

    # Build per-τ labels and manifest.
    if results:
        counts_summary = build_per_tau_label_files(
            results=results,
            output_root=args.output_root,
            tau_grid_ms=tau_grid,
            dry_run=args.dry_run,
        )

        config = {
            "stride_ms": args.stride_ms,
            "window_s": WINDOW_S,
            "train_tau_ms": TRAIN_TAU_MS,
            "tau_grid_ms": list(tau_grid),
            "substantive_duration_ms": args.substantive_duration_ms,
            "substantive_word_count": args.substantive_word_count,
            "bc_max_duration_ms": args.bc_max_duration_ms,
            "bc_max_word_count": args.bc_max_word_count,
            "a_yield_tolerance_ms": args.a_yield_tolerance_ms,
            "openface_fps": OPENFACE_FPS,
            "wavlm_hz": WAVLM_HZ,
            "max_len_v": MAX_LEN_V,
            "max_len_a": MAX_LEN_A,
            "feat_v": FEAT_V,
            "feat_a": FEAT_A,
            "openface_columns": OPENFACE_COLUMNS,
            "class_balance_strategy": "natural-rate; class weights at train time",
        }
        write_manifest(
            results=results,
            counts_summary=counts_summary,
            output_root=args.output_root,
            config=config,
            dry_run=args.dry_run,
        )

        # Print a summary table per τ for quick sanity.
        print("\n" + "=" * 78)
        print("PER-τ PER-SPLIT CLASS COUNTS (natural rate)")
        print("=" * 78)
        for tau_ms in tau_grid:
            print(f"\nτ = {tau_ms} ms:")
            for split in sorted(counts_summary[tau_ms].keys()):
                c = counts_summary[tau_ms][split]
                print(f"  {split:<6} n_samples={c['n_samples']:>5}")
                print(f"         counts: {c['counts']}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. {failed}/{len(rows)} dyad(s) failed.")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
