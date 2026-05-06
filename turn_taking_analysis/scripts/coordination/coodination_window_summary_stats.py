import os
import re
import json
from glob import glob

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/kaitlinzareno/Desktop/csci535/csci535-project/data"

COORD_DIR = (
    "/Users/kaitlinzareno/Desktop/csci535/csci535-project/data/"
    "coordination/openface_continuous_arrays/"
)

SOURCE_ID = "all_videos"

RESULTS_DIR = (
    "/Users/kaitlinzareno/Desktop/csci535/csci535-project/data/results/coordination_stats"
)

TAU = "0400"
LABELS_PATH = os.path.join(BASE_DIR, "labels_tau", f"labels_tau_{TAU}.json")

MAX_LAG = 10
FPS = 30.0  # OpenFace/video frame rate; adjust if yours differs
LABELS_TO_USE = {0, 1, 2}

HIGH_SYNC_THRESHOLD = 0.30


# =========================
# HELPERS
# =========================
def load_labels(labels_path):
    with open(labels_path, "r") as f:
        data = json.load(f)

    return {str(k).replace(".npy", "").replace(".mp4", "").replace(".wav", ""): int(v)
            for k, v in data.items()}


def parse_coord_filename(fp):
    """
    Parses:
        0000.00-0002.00_V00_S0691_I00000482_P0500.npy

    Returns:
        start_s, end_s, interaction_id, participant, file_id
    """
    fname = os.path.basename(fp).replace(".npy", "")

    pattern = (
        r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)_"
        r"(?P<interaction>V\d+_S\d+_I\d+)_"
        r"(?P<participant>P\d+)"
    )

    m = re.search(pattern, fname)
    if m is None:
        raise ValueError(f"Could not parse coordination filename: {fp}")

    start_s = float(m.group("start"))
    end_s = float(m.group("end"))
    interaction_id = m.group("interaction")
    participant = m.group("participant")
    file_id = f"{interaction_id}_{participant}"

    return {
        "filepath": fp,
        "start_s": start_s,
        "end_s": end_s,
        "mid_s": (start_s + end_s) / 2.0,
        "interaction_id": interaction_id,
        "participant": participant,
        "file_id": file_id,
    }


def end_s_to_label_key(file_id, end_s):
    """
    Your label keys look like:
        V00_S0691_I00000482_P0500_t_0002000
    """
    end_ms = int(round(float(end_s) * 1000))
    return f"{file_id}_t_{end_ms:07d}"


def load_coord_array(fp, max_lag=MAX_LAG):
    """
    Supports either:
      1. Your current saved array: shape (21, D), i.e., lags x features
      2. A dict payload with keys "wcc" and "lags", if you later save payloads

    Returns:
        arr_lag_feature: shape (num_lags, num_features)
        lags: shape (num_lags,)
    """
    obj = np.load(fp, allow_pickle=True)

    # Case 1: saved dict payload
    if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
        payload = obj.item()
        wcc = payload["wcc"]          # likely features x lags
        lags = payload["lags"]
        arr = wcc.T                  # convert to lags x features
        return arr.astype(np.float32), lags.astype(int)

    # Case 2: saved array
    arr = np.asarray(obj).astype(np.float32)

    expected_num_lags = 2 * max_lag + 1
    lags = np.arange(-max_lag, max_lag + 1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D coordination array. Got shape {arr.shape}: {fp}")

    # Your old code saved wcc.T, so expected shape is (21, D)
    if arr.shape[0] == expected_num_lags:
        return arr, lags

    # If it accidentally got saved as features x lags, transpose it
    if arr.shape[1] == expected_num_lags:
        return arr.T, lags

    raise ValueError(
        f"Could not infer lag dimension for {fp}. "
        f"Expected one dimension to be {expected_num_lags}, got {arr.shape}."
    )


# =========================
# WINDOW-LEVEL STATISTICS
# =========================
def compute_window_coordination_stats(arr_lag_feature, lags, fps=FPS, high_sync_threshold=HIGH_SYNC_THRESHOLD):
    """
    arr_lag_feature:
        shape (num_lags, num_features)

    lags:
        e.g., [-10, ..., 0, ..., 10]

    negative lag = speaker leads listener
    positive lag = listener/other participant leads.
    """
    arr = np.asarray(arr_lag_feature)
    lags = np.asarray(lags)

    abs_arr = np.abs(arr)

    # Global peak across all lags and all features
    global_peak_idx = np.unravel_index(np.argmax(abs_arr), abs_arr.shape)
    global_peak_lag = int(lags[global_peak_idx[0]])
    global_peak_corr_signed = float(arr[global_peak_idx])
    global_peak_corr_abs = float(abs_arr[global_peak_idx])

    # Per-feature best lag/corr
    # For each feature, find the lag with max absolute correlation
    best_lag_idx_per_feature = np.argmax(abs_arr, axis=0)
    best_lags_per_feature = lags[best_lag_idx_per_feature]
    best_abs_corr_per_feature = abs_arr[best_lag_idx_per_feature, np.arange(arr.shape[1])]
    best_signed_corr_per_feature = arr[best_lag_idx_per_feature, np.arange(arr.shape[1])]

    # Zero lag
    zero_lag_idx = np.where(lags == 0)[0][0]
    zero_lag_corrs = arr[zero_lag_idx, :]
    zero_lag_abs_corrs = abs_arr[zero_lag_idx, :]

    # Average profile over features by lag
    mean_abs_by_lag = abs_arr.mean(axis=1)
    mean_signed_by_lag = arr.mean(axis=1)

    best_mean_lag_idx = int(np.argmax(mean_abs_by_lag))
    best_mean_lag = int(lags[best_mean_lag_idx])
    best_mean_abs_corr = float(mean_abs_by_lag[best_mean_lag_idx])
    best_mean_signed_corr = float(mean_signed_by_lag[best_mean_lag_idx])

    # Directionality
    listener_lead_ratio = float(np.mean(best_lags_per_feature > 0)) #positive lag = speaker at future time point = listener leads
    speaker_lead_ratio = float(np.mean(best_lags_per_feature < 0))
    zero_lag_ratio = float(np.mean(best_lags_per_feature == 0))

    # Convert lag frames to seconds
    global_peak_lag_sec = global_peak_lag / fps
    mean_best_lag_frames = float(np.mean(best_lags_per_feature))
    mean_abs_best_lag_frames = float(np.mean(np.abs(best_lags_per_feature)))
    mean_best_lag_sec = mean_best_lag_frames / fps
    mean_abs_best_lag_sec = mean_abs_best_lag_frames / fps

    return {
        # Strength
        "global_peak_abs_corr": global_peak_corr_abs,
        "global_peak_signed_corr": global_peak_corr_signed,
        "mean_max_abs_corr": float(np.mean(best_abs_corr_per_feature)),
        "median_max_abs_corr": float(np.median(best_abs_corr_per_feature)),
        "std_max_abs_corr": float(np.std(best_abs_corr_per_feature)),

        # Zero-lag synchrony
        "mean_zero_lag_corr": float(np.mean(zero_lag_corrs)),
        "mean_abs_zero_lag_corr": float(np.mean(zero_lag_abs_corrs)),
        "std_zero_lag_corr": float(np.std(zero_lag_corrs)),

        # Best lag based on average across features
        "best_mean_lag_frames": best_mean_lag,
        "best_mean_lag_sec": best_mean_lag / fps,
        "best_mean_abs_corr": best_mean_abs_corr,
        "best_mean_signed_corr": best_mean_signed_corr,

        # Global best lag
        "global_peak_lag_frames": global_peak_lag,
        "global_peak_lag_sec": global_peak_lag_sec,

        # Per-feature lag summary
        "mean_best_lag_frames": mean_best_lag_frames,
        "mean_abs_best_lag_frames": mean_abs_best_lag_frames,
        "mean_best_lag_sec": mean_best_lag_sec,
        "mean_abs_best_lag_sec": mean_abs_best_lag_sec,
        "std_best_lag_frames": float(np.std(best_lags_per_feature)),

        # Directionality
        "speaker_lead_ratio": speaker_lead_ratio,
        "listener_lead_ratio": listener_lead_ratio,
        "zero_lag_ratio": zero_lag_ratio,

        # Thresholded synchrony
        "high_sync_ratio": float(np.mean(best_abs_corr_per_feature >= high_sync_threshold)),
    }


# =========================
# BUILD WINDOW-LEVEL DF
# =========================
def build_coordination_window_stats_df(
    coord_dir=COORD_DIR,
    labels_path=LABELS_PATH,
    max_lag=MAX_LAG,
    fps=FPS,
    labels_to_use=LABELS_TO_USE,
    high_sync_threshold=HIGH_SYNC_THRESHOLD,
):
    labels_dict = load_labels(labels_path)

    files = sorted(glob(os.path.join(coord_dir, "**", "*.npy"), recursive=True))

    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found in: {coord_dir}")

    rows = []
    skipped = 0

    for fp in files:
        try:
            meta = parse_coord_filename(fp)
            label_key = end_s_to_label_key(meta["file_id"], meta["end_s"])

            if label_key not in labels_dict:
                skipped += 1
                continue

            label = int(labels_dict[label_key])

            if label not in labels_to_use:
                skipped += 1
                continue

            arr, lags = load_coord_array(fp, max_lag=max_lag)

            stats = compute_window_coordination_stats(
                arr,
                lags,
                fps=fps,
                high_sync_threshold=high_sync_threshold,
            )

            rows.append({
                **meta,
                "label_key": label_key,
                "label": label,
                **stats,
            })

        except Exception as e:
            skipped += 1
            print(f"Skipping {fp}: {e}")

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError(
            "No valid labeled coordination windows found. "
            "Check coord_dir and labels_path."
        )

    df = df.sort_values(["start_s", "end_s"]).reset_index(drop=True)

    print("\nWindow-level stats")
    print(f"Coord dir: {coord_dir}")
    print(f"Files found: {len(files)}")
    print(f"Usable windows: {len(df)}")
    print(f"Skipped: {skipped}")
    print(f"Labels present: {sorted(df['label'].unique())}")

    return df


# =========================
# AGGREGATE BY LABEL
# =========================
def summarize_coordination_by_label(window_df):
    """
    Produces label-level summary statistics.
    """
    summary = (
        window_df
        .groupby("label")
        .agg(
            n_windows=("label", "size"),
            total_duration_s=("end_s", lambda x: 2.0 * len(x)),

            # Coordination strength
            mean_peak_corr=("mean_max_abs_corr", "mean"),
            std_peak_corr=("mean_max_abs_corr", "std"),
            median_peak_corr=("mean_max_abs_corr", "median"),

            mean_global_peak_corr=("global_peak_abs_corr", "mean"),
            median_global_peak_corr=("global_peak_abs_corr", "median"),

            # Zero-lag synchrony
            mean_zero_lag_abs_corr=("mean_abs_zero_lag_corr", "mean"),
            std_zero_lag_abs_corr=("mean_abs_zero_lag_corr", "std"),
            mean_zero_lag_signed_corr=("mean_zero_lag_corr", "mean"),

            # Lag timing
            mean_best_lag_frames=("mean_best_lag_frames", "mean"),
            std_best_lag_frames=("mean_best_lag_frames", "std"),
            mean_abs_best_lag_frames=("mean_abs_best_lag_frames", "mean"),

            mean_best_lag_sec=("mean_best_lag_sec", "mean"),
            std_best_lag_sec=("mean_best_lag_sec", "std"),
            mean_abs_best_lag_sec=("mean_abs_best_lag_sec", "mean"),

            # Directionality
            mean_speaker_lead_ratio=("speaker_lead_ratio", "mean"),
            mean_listener_lead_ratio=("listener_lead_ratio", "mean"),
            mean_zero_lag_ratio=("zero_lag_ratio", "mean"),

            # Thresholded synchrony
            mean_high_sync_ratio=("high_sync_ratio", "mean"),
        )
        .reset_index()
    )

    return summary

def summarize_coordination_by_video_and_label(window_df):
    video_label_summary = (
        window_df
        .groupby(["file_id", "label"])
        .agg(
            n_windows=("label", "size"),
            mean_peak_corr=("mean_max_abs_corr", "mean"),
            median_peak_corr=("mean_max_abs_corr", "median"),
            mean_zero_lag_abs_corr=("mean_abs_zero_lag_corr", "mean"),
            mean_best_lag_sec=("mean_best_lag_sec", "mean"),
            mean_abs_best_lag_sec=("mean_abs_best_lag_sec", "mean"),
            mean_speaker_lead_ratio=("speaker_lead_ratio", "mean"),
            mean_listener_lead_ratio=("listener_lead_ratio", "mean"),
            mean_zero_lag_ratio=("zero_lag_ratio", "mean"),
            mean_high_sync_ratio=("high_sync_ratio", "mean"),
        )
        .reset_index()
    )

    return video_label_summary


# =========================
# OPTIONAL: LABEL TRANSITION SUMMARY
# =========================
def summarize_label_segments(window_df):
    """
    Useful if you want not just label averages, but contiguous label periods.
    """
    df = window_df.sort_values("start_s").reset_index(drop=True).copy()

    # New segment when label changes or windows are not contiguous
    label_change = df["label"].ne(df["label"].shift())
    time_gap = df["start_s"].ne(df["end_s"].shift())
    df["segment_id"] = (label_change | time_gap).cumsum()

    segment_summary = (
        df.groupby(["segment_id", "label"])
        .agg(
            start_s=("start_s", "min"),
            end_s=("end_s", "max"),
            n_windows=("label", "size"),
            mean_peak_corr=("mean_max_abs_corr", "mean"),
            mean_zero_lag_abs_corr=("mean_abs_zero_lag_corr", "mean"),
            mean_best_lag_sec=("mean_best_lag_sec", "mean"),
            mean_speaker_lead_ratio=("speaker_lead_ratio", "mean"),
            mean_listener_lead_ratio=("listener_lead_ratio", "mean"),
        )
        .reset_index()
    )

    segment_summary["duration_s"] = segment_summary["end_s"] - segment_summary["start_s"]

    return segment_summary


# =========================
# RUN
# =========================
if __name__ == "__main__":
    window_df = build_coordination_window_stats_df(
        coord_dir=COORD_DIR,
        labels_path=LABELS_PATH,
        max_lag=MAX_LAG,
        fps=FPS,
        labels_to_use=LABELS_TO_USE,
        high_sync_threshold=HIGH_SYNC_THRESHOLD,
    )

    label_summary = summarize_coordination_by_label(window_df)
    segment_summary = summarize_label_segments(window_df)
    video_label_summary = summarize_coordination_by_video_and_label(window_df)


    print("\n=== Label-level coordination summary ===")
    print(label_summary.to_string(index=False))

    print("\n=== First few window-level rows ===")
    print(window_df.head().to_string(index=False))

    print("\n=== First few contiguous label segments ===")
    print(segment_summary.head(20).to_string(index=False))

    out_dir = RESULTS_DIR.rstrip("/")
    source_id = "all_videos"

    window_csv = os.path.join(out_dir, f"{source_id}_window_coordination_stats.csv")
    label_csv = os.path.join(out_dir, f"{source_id}_label_coordination_summary.csv")
    segment_csv = os.path.join(out_dir, f"{source_id}_label_segment_summary.csv")
    video_label_csv = os.path.join(out_dir, f"{source_id}_video_label_coordination_summary.csv")

    window_df.to_csv(window_csv, index=False)
    label_summary.to_csv(label_csv, index=False)
    segment_summary.to_csv(segment_csv, index=False)
    video_label_summary.to_csv(video_label_csv, index=False)

    print("\nSaved:")
    print(window_csv)
    print(label_csv)
    print(segment_csv)
    print(video_label_csv)