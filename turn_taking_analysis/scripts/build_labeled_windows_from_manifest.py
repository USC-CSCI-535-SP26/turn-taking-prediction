#!/usr/bin/env python3
"""
build_labeled_windows_from_manifest.py

Generate per-τ ground-truth labels for the 3-class turn-taking classifier.

Consumes per-participant VAD + transcript artifacts (plus wav-header metadata
for file-duration bounding) and emits one labels JSON per τ in the grid.
Feature slicing and saving are NOT part of this script — each modality is
extracted independently by its own manifest-driven script (see
extract_cpc_from_manifest.py, extract_openface_from_manifest.py, etc.).
The notebook's DataLoader joins labels to per-window feature files by
filename → basename translation (see "Basename ↔ spliced-filename mapping"
below).

This script is the single source of truth for:
  - sample-time enumeration (stride-based, voicing-filtered)
  - per-(t, τ) class assignment via the precedence-ordered decision tree

Every threshold it uses is documented inline (see CAPS constants) and
anchored in turn-taking literature.

Framing
-------
Stride-based sampling, horizon-centric (VAP-style — Ekstedt & Skantze 2022,
Skantze 2017 lineage). For each perspective participant A at each sample
time t:

    input window (external):  [t - WINDOW_S, t)    # τ-invariant, 2 s of past
    label horizon (internal):  [t, t + τ]          # τ-dependent, future

The input window is τ-invariant; the same t produces different labels at
different τ. This script emits labels only; the spliced per-modality
feature .npy files (one per (file_id, window)) are produced elsewhere and
named on the complementary `{start:07.2f}-{end:07.2f}_{file_id}.npy`
convention so the DataLoader can resolve labels by filename.

Basename ↔ spliced-filename mapping
-----------------------------------
Labels key on `{file_id}_t_{t_ms:07d}` where `t_ms = round(end_seconds * 1000)`.
The spliced feature file has `end` in its filename, so:

    splice: "{start:07.2f}-{end:07.2f}_{file_id}.npy"  in  "<modality>/<file_id>/"
    label basename: "{file_id}_t_{int(round(end * 1000)):07d}"

A spliced window exists on disk at every stride step from 0 up to the
file's available window count (splice_wavs.py skips partial trailing
windows, so every on-disk window has exactly WINDOW_S of audio). This
script emits labels only at the SUBSET of those times where the
perspective participant is voiced; the DataLoader ignores spliced
windows whose basename doesn't appear in the label JSON.

Classification (precedence-ordered decision tree, spec §11.4)
-------------------------------------------------------------
At each (t, τ) with A voiced at t:
    1. Build B's VAD segments + transcript words clipped to [t, t+τ].
    2. If a substantive-B utterance fires inside horizon:
         - A ceases within A_YIELD_TOLERANCE of B's start   → YIELD
         - A doesn't cease AND B voiced at t+τ              → INTERRUPT
         - A doesn't cease AND B ends before t+τ            → FAILED
    3. Else if a BC-qualifying B utterance fires AND A voiced at t+τ:
                                                             → BACKCHANNEL
    4. Else if A voiced at t+τ                               → HOLD
    5. Else                                                  → LAPSE

Classes 0/1/2 (HOLD, YIELD, BACKCHANNEL) are used in training; 3/4/5
(INTERRUPT, FAILED, LAPSE) are excluded from training but stored in the
same labels JSONs for exclusion-stat reporting. The DataLoader filters to
the training set at load time.

Output layout
-------------
    <output-root>/
      labels/
        labels_tau_0100.json      # {basename: 0..5}
        labels_tau_0200.json
        labels_tau_0400.json      # ← recommended training τ
        labels_tau_0500.json
        labels_tau_0800.json
        labels_tau_1600.json
      manifest.json               # config + per-(tau, split) counts + per-dyad flags

Usage
-----
    # Default — build all splits, all τ values
    python build_labeled_windows_from_manifest.py

    # Only val split (debugging)
    python build_labeled_windows_from_manifest.py --filter-split val

    # Print counts without writing anything
    python build_labeled_windows_from_manifest.py --dry-run

    # Override thresholds
    python build_labeled_windows_from_manifest.py \
        --stride-ms 250 \
        --substantive-duration-ms 600

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
# CAPS CONSTANTS — authoritative values. Every constant has an inline
# comment justifying its value so a future reader can edit with full
# context.
# =============================================================================

# ----- Sampling ------------------------------------------------------------
STRIDE_MS = 500
# Spacing between consecutive sample points WITHIN a perspective participant's
# voiced regions. Matches splice_wavs.py's default --stride so every emitted
# label lines up with a pre-spliced feature window. 500 ms is the Gravano &
# Hirschberg 2011 canonical "clean" inter-turn interval; below this we'd
# sample inside sub-event dynamics, above this we'd start missing short
# BC/YIELD events. 500 ms also matches the BC duration ceiling — adjacent
# samples can't both fall inside one BC.

WINDOW_S = 2.0
# Input feature-context length in seconds. Matches VAP's native 2 s
# projection window (Ekstedt & Skantze 2022) and MM-VAP (Russell & Harte
# 2025). Same value used by splice_wavs.py's --window-len; together with
# STRIDE_MS this pins the on-disk spliced-window grid one-to-one with the
# label-enumeration grid.

# ----- τ grid --------------------------------------------------------------
TRAIN_TAU_MS = 400
# Recommended single τ for training. Inoue et al. 2024 (IWSDS) report 400 ms
# as the operationally-useful anticipation horizon in streaming-deployed VAP.
# Sits near the geometric center of the τ grid — training at grid-middle
# lets us measure degradation symmetrically toward both ends of the sweep.

TAU_GRID_MS = (100, 200, 400, 500, 800, 1600)
# Evaluation horizon grid. 100/200 probe sub-psycholinguistic-gap range
# (Levinson & Torreira 2015: 200 ms modal inter-turn gap). 400 is the
# operational Inoue 2024 horizon (also the training τ). 500 is the MM-VAP
# commensurability point (Russell & Harte 2025). 800 doubles training τ.
# 1600 approaches the outer boundary past which the 2 s context has low
# predictive content. Log-spaced with two anchored points is standard
# since Skantze 2017.

# ----- BC-qualifying test (triggers BACKCHANNEL) --------------------------
# These are the experiment-facing knobs for "what counts as a BC". The
# substantive thresholds below are DERIVED from these to keep the two
# predicates disjoint by construction — editing only the BC values
# (plus BC_LEXICON) is enough to ablate the BC envelope, and the
# substantive thresholds follow automatically.
BC_MAX_DURATION_MS = 500
# Upper bound for BC-qualifying clipped-utterance duration. 500 ms is the
# canonical single-filler BC ceiling across English corpora (Ward 2000,
# Gravano 2011). Prosodically BCs are 1–3 syllables; 500 ms ≈ 2 syllables
# at normal rate.

BC_MAX_WORD_COUNT = 2
# Maximum word count for the "auto-qualify" leg of BC-qualifying. ≤ 2
# words auto-qualify (catches "oh yeah", "mm okay", "I see", "no kidding").
# Anything above this must additionally pass the all-BC-lexicon check.

# ----- Substantive-B OR-gate (triggers YIELD / INTERRUPT / FAILED) ---------
# DERIVED from the BC thresholds above. The +200 ms / +1 word gaps keep
# the substantive and BC predicates DISJOINT by construction — without
# the gaps, the decision tree's substantive-first precedence would
# silently swallow BC-classifiable utterances (precedence becomes
# load-bearing instead of informational, and the BC ceiling change
# would have no effect for any utterance the substantive predicate also
# matches).
#
# Anchors: the +200 ms gap reflects ≈ 1 content-word duration (Levelt
# 1989 mean content-word ≈ 250–350 ms) — a substantive utterance must
# carry at least one extra content word's worth of speech beyond the
# BC ceiling. The +1 word gap exceeds the BC grammatical envelope
# (Gravano 2011, Truong & Heylen 2010): if BC accepts up to N words,
# substantive starts at N+1.
#
# Default values (BC = 500 / 2) yield SUBSTANTIVE_DURATION_MS = 700 and
# SUBSTANTIVE_WORD_COUNT = 3 — the values that have been used throughout
# the project's history. Setting BC_MAX_DURATION_MS = 1000 and
# BC_MAX_WORD_COUNT = 5 (e.g. for a "longer / more complex BCs"
# experiment) auto-shifts substantive to 1200 / 6.
#
# CLI caveat: the --substantive-duration-ms / --substantive-word-count
# flags continue to override the derived values independently. If you
# raise --bc-max-duration-ms / --bc-max-word-count via CLI, ALSO pass
# the substantive flags (or use the source-edit path via these
# constants) to keep the disjointness invariant.
SUBSTANTIVE_DURATION_MS = BC_MAX_DURATION_MS + 200
# Leg 1 of the substantive-B OR-gate. B-utterance clipped to horizon
# with duration ≥ this is substantive regardless of word content.

SUBSTANTIVE_WORD_COUNT = BC_MAX_WORD_COUNT + 1
# Leg 2 of the OR-gate. Word count ≥ this is substantive regardless of
# duration or vocabulary.

# Leg 3 of the OR-gate is implicit: any clipped B-utterance containing a
# non-BC-lexicon word is substantive. Implemented as
# `any(normalize(w) not in BC_LEXICON for w in words)`.

# ----- YIELD tolerance -----------------------------------------------------
A_YIELD_TOLERANCE_MS = 300
# Max A-continues-past-B-start overlap permitted for a clean YIELD.
# Heldner & Edlund 2010: modal overlap at clean transitions < 50 ms;
# distribution tails extend to ~300 ms before overlap looks competitive.

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
# time via inverse-frequency class weights in nn.CrossEntropyLoss.

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


# =============================================================================
# IO paths + loaders
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
DEFAULT_AUDIO_DIR = str(
    PROJECT_ROOT / "turn_taking_analysis" / "subset" / "audio"
)


def load_manifest(path: str) -> list[dict]:
    """Load manifest.csv into a list of dict rows and validate columns.
    Fails fast on missing file or missing required columns — schema drift
    must surface immediately."""
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


def load_audio_duration(wav_path: str) -> float:
    """Return a wav file's duration in seconds. Reads only the WAV header
    via soundfile.info — no audio decoded. soundfile is already a project
    dependency (used by splice_wavs.py).

    File duration is needed to bound the sample-time enumeration: we only
    emit samples for t such that t + max(τ) ≤ file_duration, so every
    horizon fits inside the file and listener-VAD silence past t+τ
    accurately reflects "B not voiced" rather than "VAD didn't see this
    region".
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio wav missing: {wav_path}")
    import soundfile as sf  # lazy import
    info = sf.info(wav_path)
    return info.frames / info.samplerate


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
    the horizon."""
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
    """Lowercase → strip leading/trailing punct → collapse internal
    whitespace and hyphens. Result: 'Mm-hmm!' / 'mm hmm' / 'mmhmm' all
    → 'mmhmm'."""
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
    B-transcript words whose start falls inside the clipped interval."""

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
    of B's clipped-to-horizon utterances. Returns utterances sorted by
    clipped_start."""
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
    """3-legged OR-gate. Any one leg qualifies B's clipped utterance as
    substantive:
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
    """BACKCHANNEL qualification.
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
    """Implement the precedence-ordered decision tree. Returns one of the
    6 class integers. Prerequisite (A voiced at t) is the caller's
    responsibility — this function assumes it."""
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
        a_turn_end = current_turn_end(t, a_vad)
        assert a_turn_end is not None, \
            "classify_sample called when A not voiced at t (prerequisite)"

        if a_turn_end <= first_sub.clipped_start + a_yield_tolerance_s:
            return CLASS_YIELD

        # A keeps going past tolerance → floor contest.
        # Distinguish INTERRUPT vs FAILED using B's state at horizon end.
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
# Per-interaction, per-participant pipeline
# =============================================================================

class SampleOutput:
    """One emitted sample. Labels are accumulated in memory and written out
    per-τ at the end of the run."""

    __slots__ = ("basename", "split", "file_id", "t_ms", "labels_by_tau")

    def __init__(self, basename, split, file_id, t_ms):
        self.basename = basename
        self.split = split
        self.file_id = file_id
        self.t_ms = t_ms
        # {tau_ms: class_int} for each tau in TAU_GRID_MS
        self.labels_by_tau: dict[int, int] = {}


def enumerate_sample_times(
    a_vad: list[dict],
    file_duration: float,
    stride_ms: int,
    tau_grid_ms: tuple,
) -> list[float]:
    """Enumerate candidate sample times t for a perspective participant A.

    A time t is a candidate iff:
        t >= WINDOW_S                              (2 s of past context fits)
        t + max(tau) <= file_duration              (longest horizon fits)
        is_voiced_at(t, a_vad)                     (A is the floor holder)

    Samples are evenly spaced at `stride_ms` intervals starting from the
    first candidate time >= WINDOW_S. The stride is applied globally
    (across the whole recording) rather than per-turn, giving deterministic
    sample placement independent of turn boundaries. A's VAD acts as a
    filter: non-voiced strides are silently skipped.
    """
    t_min = WINDOW_S
    t_max = file_duration - max(tau_grid_ms) / 1000.0
    if t_max <= t_min:
        return []
    step = stride_ms / 1000.0
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
    speaker_file_id: str,
    speaker_vad: list[dict],
    listener_vad: list[dict],
    listener_words: list[dict],
    file_duration: float,
    stride_ms: int,
    tau_grid_ms: tuple,
    substantive_duration_ms: float,
    substantive_word_count: int,
    bc_max_duration_ms: float,
    bc_max_word_count: int,
    a_yield_tolerance_s: float,
) -> list[SampleOutput]:
    """Emit sample labels for one perspective participant (speaker role) in
    a dyad. Classification uses speaker VAD + listener VAD + listener
    transcript; features are the downstream pipeline's concern, not this
    script's.
    """
    sample_times = enumerate_sample_times(
        a_vad=speaker_vad,
        file_duration=file_duration,
        stride_ms=stride_ms,
        tau_grid_ms=tau_grid_ms,
    )

    outputs: list[SampleOutput] = []
    for t in sample_times:
        t_ms = int(round(t * 1000))
        basename = f"{speaker_file_id}_t_{t_ms:07d}"

        so = SampleOutput(
            basename=basename, split=split,
            file_id=speaker_file_id, t_ms=t_ms,
        )
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


def process_interaction(
    row: dict,
    *,
    vad_dir: str,
    transcript_dir: str,
    audio_dir: str,
    stride_ms: int,
    tau_grid_ms: tuple,
    substantive_duration_ms: float,
    substantive_word_count: int,
    bc_max_duration_ms: float,
    bc_max_word_count: int,
    a_yield_tolerance_s: float,
) -> dict:
    """Process one manifest row — both perspective participants. Returns a
    per-dyad result dict for the global manifest."""
    dyad_id = row["interaction_id"]
    split = row["split"]
    file_id_a = row["file_id_a"]
    file_id_b = row["file_id_b"]

    # VAD + transcript for both participants.
    vad_a = load_vad_jsonl(os.path.join(vad_dir, f"{file_id_a}.jsonl"))
    vad_b = load_vad_jsonl(os.path.join(vad_dir, f"{file_id_b}.jsonl"))

    words_a = extract_words_from_transcript(
        load_transcript_jsonl(os.path.join(transcript_dir, f"{file_id_a}.jsonl"))
    )
    words_b = extract_words_from_transcript(
        load_transcript_jsonl(os.path.join(transcript_dir, f"{file_id_b}.jsonl"))
    )

    # Audio durations (wav header only, fast). Use the shorter of the two
    # as the effective file duration so sample horizons never extend past
    # either participant's VAD coverage.
    duration_a = load_audio_duration(os.path.join(audio_dir, f"{file_id_a}.wav"))
    duration_b = load_audio_duration(os.path.join(audio_dir, f"{file_id_b}.wav"))
    file_duration = min(duration_a, duration_b)

    transcript_present_a = len(words_a) > 0
    transcript_present_b = len(words_b) > 0

    all_outputs: list[SampleOutput] = []

    # A-as-speaker (B is listener).
    all_outputs += process_one_participant(
        split=split,
        speaker_file_id=file_id_a,
        speaker_vad=vad_a, listener_vad=vad_b, listener_words=words_b,
        file_duration=file_duration,
        stride_ms=stride_ms, tau_grid_ms=tau_grid_ms,
        substantive_duration_ms=substantive_duration_ms,
        substantive_word_count=substantive_word_count,
        bc_max_duration_ms=bc_max_duration_ms,
        bc_max_word_count=bc_max_word_count,
        a_yield_tolerance_s=a_yield_tolerance_s,
    )

    # B-as-speaker (A is listener).
    all_outputs += process_one_participant(
        split=split,
        speaker_file_id=file_id_b,
        speaker_vad=vad_b, listener_vad=vad_a, listener_words=words_a,
        file_duration=file_duration,
        stride_ms=stride_ms, tau_grid_ms=tau_grid_ms,
        substantive_duration_ms=substantive_duration_ms,
        substantive_word_count=substantive_word_count,
        bc_max_duration_ms=bc_max_duration_ms,
        bc_max_word_count=bc_max_word_count,
        a_yield_tolerance_s=a_yield_tolerance_s,
    )

    return {
        "dyad_id": dyad_id,
        "split": split,
        "n_samples": len(all_outputs),
        "transcript_present_a": transcript_present_a,
        "transcript_present_b": transcript_present_b,
        "file_duration_s": file_duration,
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
    """Write manifest.json capturing full run provenance — config + per-
    (tau, split) natural-rate class counts + per-dyad flags."""
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
                "file_duration_s": r["file_duration_s"],
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
            "Generate per-τ ground-truth labels for the 3-class turn-taking "
            "classifier. Emits labels JSONs only — feature slicing is the "
            "job of per-modality extraction scripts. See module docstring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O paths.
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH,
                        help="Path to manifest.csv. Default: %(default)s")
    parser.add_argument("--vad-dir", default=DEFAULT_VAD_DIR,
                        help="Per-participant Silero VAD JSONLs. "
                             "Default: %(default)s")
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR,
                        help="Per-participant WhisperX transcript JSONLs. "
                             "Default: %(default)s")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR,
                        help="Per-participant wavs. Only the WAV header is "
                             "read (via soundfile.info) — needed to bound "
                             "sample-time enumeration by file duration. "
                             "Default: %(default)s")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Parent of the emitted labels/ subdir and "
                             "manifest.json. Default: %(default)s")

    # Threshold overrides (all default to the CAPS constants above).
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

    # Runtime plumbing.
    parser.add_argument("--filter-split", choices=["train", "val", "test"],
                        default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute labels + counts but don't write files.")

    args = parser.parse_args()

    # Parse τ-grid override.
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

    print(f"Building labels for {len(rows)} dyad(s)")
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

    # Ensure the output root exists up front — creating it here gives
    # --dry-run a visible skeleton and prevents an all-dyads-fail first
    # run from leaving write_manifest with no parent directory.
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
                audio_dir=args.audio_dir,
                stride_ms=args.stride_ms,
                tau_grid_ms=tau_grid,
                substantive_duration_ms=args.substantive_duration_ms,
                substantive_word_count=args.substantive_word_count,
                bc_max_duration_ms=args.bc_max_duration_ms,
                bc_max_word_count=args.bc_max_word_count,
                a_yield_tolerance_s=a_yield_tolerance_s,
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
