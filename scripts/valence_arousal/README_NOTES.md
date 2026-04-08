# Notes for README — Valence/Arousal Pipeline

These notes capture design decisions and technical details to be folded into the main README when the pipeline is more complete.

## Index Alignment (emotion features ↔ MOI timestamps)

Emotion features in `emotion_features.npz` are sampled at **30 Hz** (30 frames per second). 3P-IS timestamps (`start_ts`, `end_ts`) are **integers in whole seconds** measured from the start of the interaction.

To convert a MOI time window to frame indices:

    start_idx = start_ts × 30   (inclusive)
    end_idx   = end_ts   × 30   (exclusive, Python slice convention)

Because timestamps are whole seconds, each MOI maps to a clean multiple of 30 frames with no rounding ambiguity.

Both participants in the same interaction have emotion feature arrays of **identical length N** (verified empirically: e.g., V00_S1132_I00000333 has 7,860 frames for both P0737 and P1093). This means the same `start_idx:end_idx` slice can be applied to either participant's arrays to extract features for the same moment in time.

The script clamps `end_idx` to `min(end_ts × 30, N)` in case an MOI's `end_ts` extends past the end of the feature array.

## Why store indices alongside summary statistics

`moi_emotion_summaries.json` stores `start_idx`, `end_idx`, and `n_frames` for each MOI, in addition to the precomputed means. This allows downstream scripts to go back to the raw `emotion_features.npz`, grab the exact frame range, and compute any alternative statistic (median, standard deviation, slope, peak, per-frame Ekman distributions, etc.) without redoing the timestamp-to-frame alignment.

## `moi_emotion_summaries.json` schema

Each entry in the JSON array represents one MOI:

| Field | Type | Description |
|---|---|---|
| `annotated_participant` | string | Participant ID whose 3P-IS file this MOI came from |
| `non_annotated_participant` | string | The other participant in the interaction |
| `interaction_id` | string | Interaction directory name (e.g., `V00_S1132_I00000333`) |
| `imitator_emotion_features_present` | bool | True if both participants have `emotion_features.npz` |
| `annotation` | string | Free-text 3P-IS annotation |
| `start_ts` | int | MOI start time in seconds |
| `end_ts` | int | MOI end time in seconds |
| `start_idx` | int | First frame index at 30 Hz (inclusive) |
| `end_idx` | int | Last frame index at 30 Hz (exclusive) |
| `n_frames` | int | Number of frames in the window (`end_idx - start_idx`) |
| `annotated_mean_valence` | float\|null | Mean `emotion_valence` for the annotated participant |
| `annotated_mean_arousal` | float\|null | Mean `emotion_arousal` for the annotated participant |
| `annotated_mean_emotion_scores` | dict\|null | Mean logit for each of the 8 Ekman categories (annotated) |
| `non_annotated_mean_valence` | float\|null | Mean `emotion_valence` for the non-annotated participant |
| `non_annotated_mean_arousal` | float\|null | Mean `emotion_arousal` for the non-annotated participant |
| `non_annotated_mean_emotion_scores` | dict\|null | Mean logit for each of the 8 Ekman categories (non-annotated) |

Fields are `null` when the corresponding participant's `emotion_features.npz` is missing.

## Ekman emotion scores

The `emotion_scores` array in `emotion_features.npz` has shape `(N, 8)`. The 8 columns, in order, are: neutral, happy, sad, surprise, fear, disgust, anger, contempt. These are **raw logits** from the Imitator model's expression encoder, **not** softmax probabilities. Values can be negative and can exceed 1.0 — they should not be interpreted as probabilities and do not sum to 1. The mean across frames is still a valid summary statistic for comparing MOIs (higher logit = stronger signal for that category), but the scale is arbitrary.

## `imitator_emotion_features_present` flag

Of the 100 interactions, **54 have `emotion_features.npz` for both participants**. The remaining 46 have at least one participant where Meta's Imitator face-tracking model was not successfully run on their video (`has_imitator_movement = 0` in `filelist.csv`). The `--both-required` flag on `extract_moi_summaries.py` restricts processing to only those 54 interactions.

## `extract_moi_summaries.py` flags

| Flag | Default | Description |
|---|---|---|
| `--interactions-dir` | `./annotated_interactions` | Path to the `annotated_interactions/` directory |
| `--both-required` | false | Only process interactions where both participants have emotion features |
| `--overwrite` | false | Overwrite existing `moi_emotion_summaries.json` files |
| `--dry-run` | false | Print what would be processed without writing files |
