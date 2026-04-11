#!/usr/bin/env python3
"""
gaze_coordination_analysis.py

Computes dyadic gaze coordination across pre-MOI, MOI, and post-MOI windows
using four complementary methods, adapted from body pose coordination analysis
(Section 4.2.2) for gaze_encodings arrays of shape (T, 2) at 30 fps.

Methods
-------
1a. Frame-by-Frame Cosine Similarity
    For each frame t, compute cosine similarity between participant A's gaze
    vector a_t ∈ R² and participant B's gaze vector b_t ∈ R².
    Window alignment = mean over all valid frame pairs.

1b. Summary Cosine Similarity (Mean Vector)
    Average each participant's gaze frames across the window into a single
    2D summary vector, then compute cosine similarity between the two
    summaries. More robust to high-frequency noise than method 1a.

2.  Cross-Correlation (Global Lag)
    Reduce each frame to a scalar gaze magnitude (L2 norm of the 2D gaze
    vector), z-normalise the 1D signals, compute full cross-correlation
    over the combined pre+MOI window, search within ±2 s (±60 frames)
    for the peak correlation and record the corresponding lag.
    Also computed per-window (pre / moi / post) independently.

3.  Multivariate Lagged Correlation (Per Window)
    Apply a multivariate lagged correlation on the full (T, 2) matrix for
    each window separately. For each lag l ∈ [−max_lag, +max_lag], compute
    the mean Pearson r across both dimensions between A[t] and B[t+l].
    Record the peak mean-r and the lag at which it occurs.
    Max lag = 60 frames (≈ 2.0 s) — consistent with prior analysis.

Validity filtering
------------------
Frames where either participant's gaze vector has L2 norm < 1e-8 (zero /
near-zero vectors, analogous to smplh:is_valid = False) are excluded from
all computations. Windows with fewer than 5 valid frames are skipped (NaN).

Output
------
One CSV per window mode:
  gaze_coordination_time_windows.csv
  gaze_coordination_turns.csv

Usage
-----
    python gaze_coordination_analysis.py \\
        --annotated-dir /path/to/annotated_interactions \\
        [--mode both|time|turns] \\
        [--window 15.0] \\
        [--max-lag 2.0] \\
        [--min-valid-frames 5] \\
        [--output-dir .]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS              = 30
MIN_NORM         = 1e-8   # frames below this L2 norm are treated as invalid


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gaze(path: Path):
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        warnings.warn(f"Unexpected gaze shape {arr.shape} in {path}")
        return None
    return arr.astype(np.float64)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_npy_files(interaction_dir: Path) -> dict:
    result = {}
    for d in sorted(interaction_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("participant_")):
            continue
        pid = d.name.split("_", 2)[2]
        npy_files = list(d.glob("*.npy"))
        if npy_files:
            result[pid] = npy_files[0]
    return result


# ---------------------------------------------------------------------------
# Slicing & validity
# ---------------------------------------------------------------------------

def time_to_frame(t: float) -> int:
    return max(0, int(round(t * FPS)))


def slice_and_validate(gaze: np.ndarray, start_s, end_s):
    """
    Slice gaze array to [start_s, end_s) and return (frames, valid_mask).
    valid_mask[t] = True if L2 norm of gaze[t] >= MIN_NORM.
    Returns (None, None) if start/end is None or window is empty.
    """
    if start_s is None or end_s is None:
        return None, None
    T = gaze.shape[0]
    f0 = min(time_to_frame(start_s), T)
    f1 = min(time_to_frame(end_s),   T)
    if f1 <= f0:
        return None, None
    seg = gaze[f0:f1]
    valid = np.linalg.norm(seg, axis=1) >= MIN_NORM
    return seg, valid


# ---------------------------------------------------------------------------
# Method 1a — Frame-by-Frame Cosine Similarity
# ---------------------------------------------------------------------------

def method_1a_framewise_cosine(seg_a, seg_b, valid_a, valid_b, min_frames: int):
    """
    Mean cosine similarity over valid frame pairs.
    Returns (mean_sim, n_valid_frames).
    """
    if seg_a is None or seg_b is None:
        return np.nan, 0

    n = min(len(seg_a), len(seg_b))
    valid = valid_a[:n] & valid_b[:n]
    n_valid = int(valid.sum())

    if n_valid < min_frames:
        return np.nan, n_valid

    a = seg_a[:n][valid]
    b = seg_b[:n][valid]

    # Cosine similarity = dot / (|a| * |b|)
    dots   = np.sum(a * b, axis=1)
    norms  = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    sims   = dots / norms                     # ∈ [−1, 1]

    return float(np.mean(sims)), n_valid


# ---------------------------------------------------------------------------
# Method 1b — Summary Cosine Similarity (Mean Vector)
# ---------------------------------------------------------------------------

def method_1b_summary_cosine(seg_a, seg_b, valid_a, valid_b, min_frames: int):
    """
    Cosine similarity between the mean gaze vectors over valid frames.
    Returns scalar similarity ∈ [−1, 1] or NaN.
    """
    if seg_a is None or seg_b is None:
        return np.nan

    n = min(len(seg_a), len(seg_b))
    valid = valid_a[:n] & valid_b[:n]
    n_valid = int(valid.sum())

    if n_valid < min_frames:
        return np.nan

    mean_a = seg_a[:n][valid].mean(axis=0)   # shape (2,)
    mean_b = seg_b[:n][valid].mean(axis=0)

    norm_a = np.linalg.norm(mean_a)
    norm_b = np.linalg.norm(mean_b)

    if norm_a < MIN_NORM or norm_b < MIN_NORM:
        return np.nan

    return float(np.dot(mean_a, mean_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Method 2 — Cross-Correlation on L2-Norm Magnitude Signal
# ---------------------------------------------------------------------------

def _magnitude_signal(seg, valid) -> np.ndarray | None:
    """Return z-normalised L2-norm magnitude time series (valid frames only → NaN elsewhere)."""
    if seg is None:
        return None
    mag = np.full(len(seg), np.nan)
    mag[valid] = np.linalg.norm(seg[valid], axis=1)
    return mag


def method_2_cross_correlation(seg_a, seg_b, valid_a, valid_b,
                                max_lag_frames: int, min_frames: int):
    """
    Cross-correlation on scalar magnitude signals.
    Searches ±max_lag_frames for the peak correlation.
    Returns (peak_xcorr, lag_s, n_valid).
    Positive lag = B leads A.
    """
    if seg_a is None or seg_b is None:
        return np.nan, np.nan, 0

    n = min(len(seg_a), len(seg_b))
    valid = valid_a[:n] & valid_b[:n]
    n_valid = int(valid.sum())

    if n_valid < min_frames:
        return np.nan, np.nan, n_valid

    # Extract valid-only magnitude (drop NaN positions for xcorr)
    idx = np.where(valid[:n])[0]
    mag_a = np.linalg.norm(seg_a[:n][valid], axis=1)
    mag_b = np.linalg.norm(seg_b[:n][valid], axis=1)

    # Z-normalise
    if np.std(mag_a) < 1e-10 or np.std(mag_b) < 1e-10:
        return np.nan, np.nan, n_valid

    mag_a = (mag_a - np.mean(mag_a)) / np.std(mag_a)
    mag_b = (mag_b - np.mean(mag_b)) / np.std(mag_b)

    # Full cross-correlation
    xcorr  = np.correlate(mag_a, mag_b, mode="full") / n_valid
    center = n_valid - 1

    lo = max(0, center - max_lag_frames)
    hi = min(len(xcorr), center + max_lag_frames + 1)
    xcorr_w = xcorr[lo:hi]
    lags    = np.arange(-(center - lo), len(xcorr_w) - (center - lo))

    peak_idx  = np.argmax(np.abs(xcorr_w))
    peak_corr = float(xcorr_w[peak_idx])
    peak_lag  = float(lags[peak_idx]) / FPS      # seconds

    return peak_corr, peak_lag, n_valid


# ---------------------------------------------------------------------------
# Method 3 — Multivariate Lagged Correlation
# ---------------------------------------------------------------------------

def method_3_multivariate_lagged(seg_a, seg_b, valid_a, valid_b,
                                  max_lag_frames: int, min_frames: int):
    """
    For each lag l ∈ [−max_lag_frames, +max_lag_frames]:
      - Align valid frames of A and B with the given lag
      - Compute Pearson r for each of the 2 gaze dimensions
      - Average r across dimensions → mean_r(l)
    Record peak mean_r and the corresponding lag.

    Z-normalisation is applied per-participant per-dimension within the window,
    consistent with the body-pose method described in the paper.

    Returns (peak_mean_r, lag_at_peak_s, n_valid_at_lag_0).
    Positive lag = B leads A.
    """
    if seg_a is None or seg_b is None:
        return np.nan, np.nan, 0

    n = min(len(seg_a), len(seg_b))
    valid = valid_a[:n] & valid_b[:n]
    n_valid = int(valid.sum())

    if n_valid < min_frames:
        return np.nan, np.nan, n_valid

    # Work on valid frames only; re-index to a compact array
    a = seg_a[:n][valid]   # shape (n_valid, 2)
    b = seg_b[:n][valid]

    # Z-normalise per dimension independently
    for dim in range(2):
        if np.std(a[:, dim]) > 1e-10:
            a[:, dim] = (a[:, dim] - a[:, dim].mean()) / a[:, dim].std()
        if np.std(b[:, dim]) > 1e-10:
            b[:, dim] = (b[:, dim] - b[:, dim].mean()) / b[:, dim].std()

    lags     = np.arange(-max_lag_frames, max_lag_frames + 1)
    mean_rs  = []

    for lag in lags:
        if lag == 0:
            a_aligned, b_aligned = a, b
        elif lag > 0:
            # B leads A: A[t] vs B[t + lag]  → drop last `lag` of A, first `lag` of B
            if lag >= len(a):
                mean_rs.append(np.nan)
                continue
            a_aligned = a[:-lag]
            b_aligned = b[lag:]
        else:
            # A leads B: A[t + |lag|] vs B[t]
            abs_lag = abs(lag)
            if abs_lag >= len(b):
                mean_rs.append(np.nan)
                continue
            a_aligned = a[abs_lag:]
            b_aligned = b[:-abs_lag]

        if len(a_aligned) < min_frames:
            mean_rs.append(np.nan)
            continue

        rs = []
        for dim in range(2):
            if np.std(a_aligned[:, dim]) < 1e-10 or np.std(b_aligned[:, dim]) < 1e-10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = pearsonr(a_aligned[:, dim], b_aligned[:, dim])
            rs.append(r)

        mean_rs.append(np.nanmean(rs) if rs else np.nan)

    mean_rs = np.array(mean_rs, dtype=float)
    valid_mask = ~np.isnan(mean_rs)

    if not valid_mask.any():
        return np.nan, np.nan, n_valid

    peak_idx  = np.nanargmax(np.abs(mean_rs))
    peak_r    = float(mean_rs[peak_idx])
    peak_lag  = float(lags[peak_idx]) / FPS

    return peak_r, peak_lag, n_valid


# ---------------------------------------------------------------------------
# Combined per-window analysis
# ---------------------------------------------------------------------------

def analyze_window(gaze_a, gaze_b, start_s, end_s,
                   max_lag_frames: int, min_frames: int) -> dict:
    """Run all four methods for one time window."""
    nan_row = {
        # Method 1a
        'm1a_cosine_sim_mean': np.nan, 'm1a_n_valid': 0,
        # Method 1b
        'm1b_summary_cosine': np.nan,
        # Method 2
        'm2_xcorr_peak': np.nan, 'm2_xcorr_lag_s': np.nan, 'm2_n_valid': 0,
        # Method 3
        'm3_mv_lag_peak_r': np.nan, 'm3_mv_lag_s': np.nan, 'm3_n_valid': 0,
    }

    if start_s is None or end_s is None:
        return nan_row

    seg_a, valid_a = slice_and_validate(gaze_a, start_s, end_s)
    seg_b, valid_b = slice_and_validate(gaze_b, start_s, end_s)

    if seg_a is None or seg_b is None:
        return nan_row

    sim_1a, n_1a       = method_1a_framewise_cosine(seg_a, seg_b, valid_a, valid_b, min_frames)
    sim_1b             = method_1b_summary_cosine(seg_a, seg_b, valid_a, valid_b, min_frames)
    xc, xl, n_2        = method_2_cross_correlation(seg_a, seg_b, valid_a, valid_b, max_lag_frames, min_frames)
    mv_r, mv_l, n_3    = method_3_multivariate_lagged(seg_a, seg_b, valid_a, valid_b, max_lag_frames, min_frames)

    return {
        'm1a_cosine_sim_mean': sim_1a, 'm1a_n_valid': n_1a,
        'm1b_summary_cosine':  sim_1b,
        'm2_xcorr_peak':       xc,    'm2_xcorr_lag_s':   xl,  'm2_n_valid': n_2,
        'm3_mv_lag_peak_r':    mv_r,  'm3_mv_lag_s':      mv_l, 'm3_n_valid': n_3,
    }


def build_row(base: dict, pre_m: dict, moi_m: dict, post_m: dict) -> dict:
    row = dict(base)
    for k, v in pre_m.items():
        row[f'pre_{k}'] = v
    for k, v in moi_m.items():
        row[f'moi_{k}'] = v
    for k, v in post_m.items():
        row[f'post_{k}'] = v
    return row


# ---------------------------------------------------------------------------
# Per-interaction processing
# ---------------------------------------------------------------------------

def _load_interaction(interaction_dir: Path, mode_label: str):
    """Load gaze arrays and return (pid_a, pid_b, gaze_a, gaze_b) or None."""
    npy_map = find_npy_files(interaction_dir)
    if len(npy_map) < 2:
        print(f"  [{interaction_dir.name}] SKIP ({mode_label}) — fewer than 2 gaze files")
        return None
    pid_a, pid_b = sorted(npy_map)[:2]
    gaze_a = load_gaze(npy_map[pid_a])
    gaze_b = load_gaze(npy_map[pid_b])
    if gaze_a is None or gaze_b is None:
        print(f"  [{interaction_dir.name}] SKIP ({mode_label}) — could not load gaze arrays")
        return None
    return pid_a, pid_b, gaze_a, gaze_b


def process_time_windows(interaction_dir: Path, max_lag_frames: int, min_frames: int) -> list:
    iid = interaction_dir.name
    entries = load_json(
        interaction_dir / "interaction" / "pre_and_post_moi" / "time_windows_pre_post.json"
    )
    if entries is None:
        print(f"  [{iid}] SKIP (time) — missing time_windows_pre_post.json")
        return []

    result = _load_interaction(interaction_dir, "time")
    if result is None:
        return []
    pid_a, pid_b, gaze_a, gaze_b = result

    rows = []
    for e in entries:
        base = {
            'interaction_id':       iid,
            'annotated_participant': e.get('annotated_participant'),
            'event_speaker':         e.get('event_speaker'),
            'participant_a':         pid_a,
            'participant_b':         pid_b,
            'start_moi':             e['start_moi'],
            'end_moi':               e['end_moi'],
            'moi_duration':          e['end_moi'] - e['start_moi'],
        }
        rows.append(build_row(
            base,
            analyze_window(gaze_a, gaze_b, e['start_pre_moi'],  e['end_pre_moi'],  max_lag_frames, min_frames),
            analyze_window(gaze_a, gaze_b, e['start_moi'],      e['end_moi'],      max_lag_frames, min_frames),
            analyze_window(gaze_a, gaze_b, e['start_post_moi'], e['end_post_moi'], max_lag_frames, min_frames),
        ))

    print(f"  [{iid}] (time)  {len(entries)} MOIs → {len(rows)} rows")
    return rows


def process_turns(interaction_dir: Path, max_lag_frames: int, min_frames: int) -> list:
    iid = interaction_dir.name
    entries = load_json(
        interaction_dir / "interaction" / "pre_and_post_moi" / "turns_pre_post.json"
    )
    if entries is None:
        print(f"  [{iid}] SKIP (turns) — missing turns_pre_post.json")
        return []

    result = _load_interaction(interaction_dir, "turns")
    if result is None:
        return []
    pid_a, pid_b, gaze_a, gaze_b = result

    rows = []
    for e in entries:
        base = {
            'interaction_id':           iid,
            'annotated_participant':     e.get('annotated_participant'),
            'non_annotated_participant': e.get('non_annotated_participant'),
            'event_speaker':             e.get('event_speaker'),
            'participant_a':             pid_a,
            'participant_b':             pid_b,
            'start_moi':                 e['start_moi'],
            'end_moi':                   e['end_moi'],
            'moi_duration':              e['end_moi'] - e['start_moi'],
            'pre_overlap':               e.get('pre_overlap'),
            'post_overlap':              e.get('post_overlap'),
        }
        rows.append(build_row(
            base,
            analyze_window(gaze_a, gaze_b, e.get('start_pre_moi'),  e.get('end_pre_moi'),  max_lag_frames, min_frames),
            analyze_window(gaze_a, gaze_b, e['start_moi'],           e['end_moi'],           max_lag_frames, min_frames),
            analyze_window(gaze_a, gaze_b, e.get('start_post_moi'), e.get('end_post_moi'), max_lag_frames, min_frames),
        ))

    print(f"  [{iid}] (turns) {len(entries)} MOIs → {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_csv(rows: list, path: str):
    if not rows:
        print(f"  No rows to save for {path}")
        return
    df = pd.DataFrame(rows)
    df.sort_values(['interaction_id', 'start_moi'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gaze coordination analysis using four complementary methods."
    )
    parser.add_argument('--annotated-dir', required=True)
    parser.add_argument('--mode', choices=['time', 'turns', 'both'], default='both')
    parser.add_argument('--window', type=float, default=15.0,
                        help='Seconds before/after MOI for time windows (default: 15).')
    parser.add_argument('--max-lag', type=float, default=2.0,
                        help='Max lag in seconds for cross-correlation and multivariate lagged '
                             'correlation (default: 2.0).')
    parser.add_argument('--min-valid-frames', type=int, default=5,
                        help='Minimum valid frames required per window (default: 5).')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    annotated_dir  = Path(args.annotated_dir)
    max_lag_frames = int(args.max_lag * FPS)
    os.makedirs(args.output_dir, exist_ok=True)

    if not annotated_dir.exists():
        print(f"ERROR: {annotated_dir} not found")
        sys.exit(1)

    interactions = sorted([
        d for d in annotated_dir.iterdir()
        if d.is_dir() and d.name.startswith("V")
    ])

    print(f"Found {len(interactions)} interaction directories")
    print(f"Mode: {args.mode} | Window: ±{args.window}s | "
          f"Max lag: {args.max_lag}s ({max_lag_frames} frames) | "
          f"Min valid frames: {args.min_valid_frames}\n")

    time_rows, turn_rows = [], []

    for idir in interactions:
        if args.mode in ('time', 'both'):
            time_rows.extend(process_time_windows(idir, max_lag_frames, args.min_valid_frames))
        if args.mode in ('turns', 'both'):
            turn_rows.extend(process_turns(idir, max_lag_frames, args.min_valid_frames))

    print()
    if args.mode in ('time', 'both'):
        save_csv(time_rows,
                 os.path.join(args.output_dir, 'gaze_coordination_time_windows.csv'))
    if args.mode in ('turns', 'both'):
        save_csv(turn_rows,
                 os.path.join(args.output_dir, 'gaze_coordination_turns.csv'))

    print("\nDone.")


if __name__ == "__main__":
    main()
