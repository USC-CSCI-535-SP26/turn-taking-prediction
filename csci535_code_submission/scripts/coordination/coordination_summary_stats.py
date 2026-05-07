import os
import json
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/kaitlinzareno/Desktop/csci535/csci535-project/data"

TAU = "0400"
LABELS_PATH = os.path.join(BASE_DIR, "labels_tau", f"labels_tau_{TAU}.json")
COORDINATION_PATH = os.path.join(
    BASE_DIR, "coordination", "openface", f"all_coordination_features_tau_{TAU}.csv"
)

OUTPUT_BASE = os.path.join(BASE_DIR, "coordination", "openface")
OUTPUT_LABELED_PATH = os.path.join(OUTPUT_BASE, f"all_coordination_features_labeled_tau_{TAU}.csv")
OUTPUT_SUMMARY_PATH = os.path.join(OUTPUT_BASE, f"coordination_summary_by_label_tau_{TAU}.csv")

TRAINING_CLASSES = {0, 1, 2}

def load_labels(labels_path):
    VALID_LABEL_INTS = {0, 1, 2, 3, 4, 5}

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels JSON not found: {labels_path}")

    with open(labels_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"labels JSON must be a dict/object. Got {type(data).__name__}")

    validated = {}
    for basename, raw in data.items():
        if not isinstance(basename, str):
            raise ValueError(f"Label key must be string. Got {basename!r}")

        cls = int(raw)

        if cls not in VALID_LABEL_INTS:
            raise ValueError(f"Label value for {basename!r}={cls} outside valid range 0..5")

        validated[basename] = cls

    return validated

def load_correlation_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Coordination CSV not found: {path}")
    return pd.read_csv(path)

def format_time_to_str(seconds):
    ms = int(round(float(seconds) * 1000))
    return f"{ms:07d}"

def match_coordination_labels(labels_dict, coordination_df):
    df = coordination_df.copy()

    required_cols = {"interaction_name", "speaker", "end_s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"coordination_df missing required columns: {missing}")

    df["end_time_str"] = df["end_s"].apply(format_time_to_str)

    df["label_key"] = (
        df["interaction_name"].astype(str)
        + "_"
        + df["speaker"].astype(str)
        + "_t_"
        + df["end_time_str"]
    )

    df["label"] = df["label_key"].map(labels_dict)

    n_missing = df["label"].isna().sum()
    print(f"Matched labels for {len(df) - n_missing}/{len(df)} rows")
    print(f"Missing labels: {n_missing}")

    return df

def get_summary_stats(df, training_classes=None):
    """
    Summarize coordination features by class label.
    """
    df = df.copy()

    if "label" not in df.columns:
        raise ValueError("Expected column 'label' in dataframe.")

    # Drop rows where no label was found
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Optionally restrict to training classes
    if training_classes is not None:
        df = df[df["label"].isin(training_classes)]

    agg_cols = {
        # magnitude
        "mean_max_abs_corr": "mean",
        "std_max_abs_corr": "mean",
        "median_max_abs_corr": "mean",
        "max_abs_corr": "mean",

        # signed peak
        "mean_signed_peak_corr": "mean",
        "std_signed_peak_corr": "mean",

        # zero-lag
        "mean_zero_lag_corr": "mean",
        "mean_abs_zero_lag_corr": "mean",
        "std_zero_lag_corr": "mean",

        # speaker/listener lag
        "mean_best_lag_frames": "mean",
        "std_best_lag_frames": "mean",
        "mean_abs_best_lag_frames": "mean",
        "mean_best_lag_sec": "mean",
        "mean_abs_best_lag_sec": "mean",

        # directionality
        "speaker_lead_ratio": "mean",
        "listener_lead_ratio": "mean",
        "zero_lag_ratio": "mean",
        "listener_speaker_lead_score": "mean",

        # threshold
        "high_sync_ratio": "mean",
    }

    # Only keep columns that actually exist
    existing_agg_cols = {
        col: func for col, func in agg_cols.items()
        if col in df.columns
    }

    missing_agg_cols = sorted(set(agg_cols) - set(existing_agg_cols))
    if missing_agg_cols:
        print("Warning: missing coordination columns:")
        for col in missing_agg_cols:
            print(f"  - {col}")

    summary = (
        df.groupby("label")
        .agg(
            n_samples=("label", "size"),
            **{
                f"{col}_{func}": (col, func)
                for col, func in existing_agg_cols.items()
            }
        )
        .reset_index()
        .sort_values("label")
    )

    return summary

if __name__ == "__main__":
    labels_dict = load_labels(LABELS_PATH)
    coordination_df = load_correlation_df(COORDINATION_PATH)

    labeled_df = match_coordination_labels(labels_dict, coordination_df)

    # Save full labeled coordination dataframe
    labeled_df.to_csv(OUTPUT_LABELED_PATH, index=False)
    print(f"Saved labeled coordination features to:\n{OUTPUT_LABELED_PATH}")

    # Summary for all matched labels
    summary_all = get_summary_stats(labeled_df, training_classes=None)
    print("\n=== Summary by label: all labels ===")
    print(summary_all)

    # Summary only for training classes 0, 1, 2
    summary_train = get_summary_stats(labeled_df, training_classes=TRAINING_CLASSES)
    print("\n=== Summary by label: training classes only ===")
    print(summary_train)

    summary_train.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    print(f"\nSaved summary to:\n{OUTPUT_SUMMARY_PATH}")

    # Optional debugging example
    example_key = "V03_S2132_I00000216_P5793_t_0120500"
    example_rows = labeled_df[labeled_df["label_key"] == example_key]
    print(f"\nRows matching example key {example_key}: {len(example_rows)}")
    print(example_rows.head())