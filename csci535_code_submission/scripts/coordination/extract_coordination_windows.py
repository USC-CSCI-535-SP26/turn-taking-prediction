# %%
import os
import json
import re
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/kaitlinzareno/Desktop/csci535/csci535-project/data"

INPUT_DIR = os.path.join(BASE_DIR, "openface_new")
OUTPUT_DIR = os.path.join(BASE_DIR, "coordination", "openface_continuous_arrays")

TAUS = ["0100", "0200", "0400", "0500", "0800", "1600"]
TAU = TAUS[2]

LABELS_PATH = os.path.join(BASE_DIR, "labels_tau", f"labels_tau_{TAU}.json")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifests","manifest.csv")

MAX_LAG = 10
TRAINING_CLASSES = {0, 1, 2}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def load_labels(labels_path):
    with open(labels_path, "r") as f:
        data = json.load(f)

    validated = {}
    for basename, raw in data.items():
        cls = int(raw)
        if cls not in {0, 1, 2, 3, 4, 5}:
            raise ValueError(f"Invalid label {cls} for {basename}")
        validated[basename] = cls

    return validated

def basename_to_window(basename):
    b = os.path.basename(str(basename))
    b = b.replace(".npy", "").replace(".mp4", "").replace(".wav", "")

    pattern = (
        r"(?P<interaction>V\d+_S\d+_I\d+)_"
        r"(?P<participant>P\d+)_t_(?P<end>\d+)"
    )

    m = re.search(pattern, b)
    if m is None:
        raise ValueError(f"Could not parse basename: {basename}")

    interaction = m.group("interaction")
    participant = m.group("participant")
    end_s = round(int(m.group("end")) / 1000.0, 3)
    start_s = round(end_s - 2.0, 3)

    return participant, interaction, start_s, end_s

def clean_file_id(x):
    return (
        str(x)
        .replace(".mp4", "")
        .replace(".npy", "")
        .replace(".wav", "")
    )

def load_manifest(manifest_path):
    return pd.read_csv(manifest_path)

def build_fid_lookup(manifest_df):
    required = [
        "interaction_id",
        "split",
        "file_id_a",
        "file_id_b",
        "participant_a",
        "participant_b",
    ]

    missing = [c for c in required if c not in manifest_df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    lookup = {}

    for _, row in manifest_df.iterrows():
        file_id_a = clean_file_id(row["file_id_a"])
        file_id_b = clean_file_id(row["file_id_b"])

        participant_a = str(row["participant_a"])
        participant_b = str(row["participant_b"])

        lookup[file_id_a] = {
            "listener_file_id": file_id_b,
            "speaker_participant": participant_a,
            "listener_participant": participant_b,
            "interaction_id": str(row["interaction_id"]),
            "split": str(row["split"]),
        }

        lookup[file_id_b] = {
            "listener_file_id": file_id_a,
            "speaker_participant": participant_b,
            "listener_participant": participant_a,
            "interaction_id": str(row["interaction_id"]),
            "split": str(row["split"]),
        }

    return lookup

def enumerate_samples(manifest_df, labels_dict):
    fid_lookup = build_fid_lookup(manifest_df)

    rows = []
    skipped_nontraining = 0
    missing_manifest = 0
    parse_failed = 0

    for basename, cls in labels_dict.items():
        cls = int(cls)

        if cls not in TRAINING_CLASSES:
            skipped_nontraining += 1
            continue

        try:
            speaker_pid, interaction_id, start_s, end_s = basename_to_window(basename)
        except Exception:
            parse_failed += 1
            continue

        speaker_file_id = f"{interaction_id}_{speaker_pid}"
        role_info = fid_lookup.get(speaker_file_id)

        if role_info is None:
            missing_manifest += 1
            continue

        rows.append({
            "basename": basename,
            "interaction_id": interaction_id,
            "speaker_file_id": speaker_file_id,
            "listener_file_id": role_info["listener_file_id"],
            "speaker_participant": role_info["speaker_participant"],
            "listener_participant": role_info["listener_participant"],
            "split": role_info["split"],
            "start_s": start_s,
            "end_s": end_s,
            "label": cls,
        })

    df = pd.DataFrame(rows)

    print("\nEnumeration summary")
    print(f"usable samples:       {len(df)}")
    print(f"skipped non-training: {skipped_nontraining}")
    print(f"missing manifest:     {missing_manifest}")
    print(f"parse failed:         {parse_failed}")

    return df

def parse_openface_filename(fp):
    fname = os.path.basename(fp).replace(".npy", "")

    pattern = (
        r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)_"
        r"(?P<interaction>V\d+_S\d+_I\d+)_"
        r"(?P<participant>P\d+)"
    )

    m = re.search(pattern, fname)
    if m is None:
        raise ValueError(f"Could not parse OpenFace filename: {fp}")

    start_s = round(float(m.group("start")), 3)
    end_s = round(float(m.group("end")), 3)
    interaction_id = m.group("interaction")
    participant = m.group("participant")

    return {
        "filepath": fp,
        "interaction_id": interaction_id,
        "participant": participant,
        "file_id": f"{interaction_id}_{participant}",
        "start_s": start_s,
        "end_s": end_s,
    }

def build_file_index(input_dir):
    files = glob(os.path.join(input_dir, "*", "*.npy"))

    rows = []
    for fp in files:
        try:
            rows.append(parse_openface_filename(fp))
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    df = pd.DataFrame(rows)

    print("\nOpenFace index")
    print(f"found files: {len(files)}")
    print(f"parsed rows: {len(df)}")

    return df

def match_samples_to_openface(samples_df, file_df):
    speaker_files = file_df.rename(columns={
        "filepath": "speaker_filepath",
        "participant": "speaker_participant",
    })[
        [
            "interaction_id",
            "speaker_participant",
            "start_s",
            "end_s",
            "speaker_filepath",
        ]
    ]

    listener_files = file_df.rename(columns={
        "filepath": "listener_filepath",
        "participant": "listener_participant",
    })[
        [
            "interaction_id",
            "listener_participant",
            "start_s",
            "end_s",
            "listener_filepath",
        ]
    ]

    matched = samples_df.merge(
        speaker_files,
        on=["interaction_id", "speaker_participant", "start_s", "end_s"],
        how="left",
    )

    matched = matched.merge(
        listener_files,
        on=["interaction_id", "listener_participant", "start_s", "end_s"],
        how="left",
    )

    print("\nMatched samples")
    print(f"before drop:            {len(matched)}")
    print(f"missing speaker files:  {matched['speaker_filepath'].isna().sum()}")
    print(f"missing listener files: {matched['listener_filepath'].isna().sum()}")

    matched = matched.dropna(
        subset=["speaker_filepath", "listener_filepath"]
    ).copy()

    print(f"usable matched samples: {len(matched)}")

    return matched

def compute_wcc_fast(x, y, max_lag=10):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got x={x.shape}, y={y.shape}")

    T = min(len(x), len(y))
    x = x[:T].astype(np.float32)
    y = y[:T].astype(np.float32)

    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature mismatch: x={x.shape}, y={y.shape}")

    if T <= max_lag + 2:
        raise ValueError(f"Window too short: T={T}, max_lag={max_lag}")

    x = (x - x.mean(axis=0, keepdims=True)) / (
        x.std(axis=0, keepdims=True) + 1e-8
    )
    y = (y - y.mean(axis=0, keepdims=True)) / (
        y.std(axis=0, keepdims=True) + 1e-8
    )

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    wcc = np.zeros((x.shape[1], len(lags)), dtype=np.float32)

    for i, lag in enumerate(lags):
        if lag < 0:
            # speaker leads listener:
            # compare Speaker's past (x[:-lag]) to Listener's present (y[lag:])
            wcc[:, i] = np.mean(x[:lag] * y[-lag:], axis=0)
        elif lag > 0:
            # listener leads speaker: 
            # compare Speaker's present (x[-lag:]) to Listener's past (y[:lag])
            wcc[:, i] = np.mean(x[lag:] * y[:-lag], axis=0)
        else:
            # simultaneous
            wcc[:, i] = np.mean(x * y, axis=0)

    return wcc, lags

def make_output_path(row):
    interaction_id = row["interaction_id"]
    speaker = row["speaker_participant"]

    speaker_id = f"{interaction_id}_{speaker}"

    start_s = float(row["start_s"])
    end_s = float(row["end_s"])

    time_window = f"{start_s:07.2f}-{end_s:07.2f}"
    filename = f"{time_window}_{speaker_id}.npy"

    out_dir = os.path.join(OUTPUT_DIR, speaker_id)
    os.makedirs(out_dir, exist_ok=True)

    return os.path.join(out_dir, filename)

if __name__ == "__main__":
    labels_dict = load_labels(LABELS_PATH)
    manifest_df = load_manifest(MANIFEST_PATH)

    samples_df = enumerate_samples(manifest_df, labels_dict)

    file_df = build_file_index(INPUT_DIR)
    matched = match_samples_to_openface(samples_df, file_df)

    written = 0
    skipped = 0

    for _, row in tqdm(matched.iterrows(), total=len(matched)):
        try:
            speaker_x = np.load(row["speaker_filepath"], mmap_mode="r")
            listener_y = np.load(row["listener_filepath"], mmap_mode="r")

            wcc, lags = compute_wcc_fast(
                speaker_x,
                listener_y,
                max_lag=MAX_LAG,
            )

            out_path = make_output_path(row)

            payload = {
                "wcc": wcc.astype(np.float32),
                "lags": lags.astype(np.int32),
                "max_lag": int(MAX_LAG),
                "speaker": row["speaker_participant"],
                "listener": row["listener_participant"],
                "interaction_id": row["interaction_id"],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "label": int(row["label"]),
                "split": row["split"],
            }

            #np.save(out_path, payload, allow_pickle=True)

            coord_arr = wcc.T.astype(np.float32)  # (21, 23)
            np.save(out_path, coord_arr)
            written += 1

        except Exception as e:
            skipped += 1
            continue

    print("\nDone.")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {OUTPUT_DIR}")
