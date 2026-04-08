# CSCI 535 Project — Seamless Interaction Dataset Analysis

## Directory Structure of `both_annotated_interactions`

```
both_annotated_interactions/                   ◁── 100 interaction directories
├── V00_S1288_I00000099/                       ◁── one interaction (V=vendor, S=session, I=prompt hash)
│   ├── interaction/                           ◁── interaction-level metadata + derived files
│   │   ├── interaction_metadata.json          ← from dataset CSVs (download_annotated_interactions.py)
│   │   ├── session_relationship.json          ← from dataset CSVs (download_annotated_interactions.py)
│   │   ├── participants_metadata.json         ← from dataset CSVs (download_annotated_interactions.py)
│   │   ├── filelist_entries.json              ← from dataset CSVs (download_annotated_interactions.py)
│   │   ├── timestamps_by_turn.json            ← derived by us (generate_timestamps_by_turn.py)
│   │   ├── moi_emotion_summaries.json         ← derived by us (extract_moi_summaries.py)
│   │   ├── video_viewer.html                  ← derived by us (download_annotated_interactions.py)
│   │   └── pre_and_post_moi/                 ◁── pre/post MOI time windows
│   │       ├── time_windows_pre_post.json     ← derived by us (generate_moi_time_windows.py)
│   │       └── turns_pre_post.json            ← derived by us (generate_moi_turns.py)
│   ├── participant_a_P0071/                   ◁── a/b assigned alphabetically by participant ID
│   │   ├── 3P-IS_..._P0071.json              ← from dataset (download_annotated_interactions.py)
│   │   ├── 3P-R_..._P0071.json               ← from dataset (download_annotated_interactions.py)
│   │   ├── 3P-V_..._P0071.json               ← from dataset (download_annotated_interactions.py)
│   │   ├── vad_..._P0071.jsonl                ← from dataset (download_annotated_interactions.py)
│   │   ├── V00_S1288_I00000099_P0071.wav      ← from dataset (excluded from GitHub)
│   │   └── emotion_features.npz               ← from dataset (download_emotion_features.py)
│   └── participant_b_P1119/                   ◁── same files as participant_a
│       └── ...
│
├── V00_S1288_I00000102/
│   └── ...
└── ... (100 interactions total)
```

Each interaction directory is named **`V{vendor}_S{session}_I{prompt_hash}`**: `V{vendor}` is the data collection site (only V00 and V03 appear), `S{session}` is the recording session (a 1-hour continuous recording of a specific dyad), and `I{prompt_hash}` is a hash identifying the prompt pair (not a sequence number). A unique interaction is identified by the full V+S+I combination. Participant subdirectories are assigned alphabetically by participant ID (lower ID = participant_a); this does **not** correspond to the prompt-role assignment in the dataset's `interactions.csv`.

### File Reference

#### `interaction/` — Metadata (from dataset CSVs)

**`interaction_metadata.json`** — Prompt text for both participants, IPC octant codes (Agency/Communion), interaction type (always "naturalistic" for our 100-interaction subset), and video duration in seconds (obtained by probing the MP4 on S3 via `get_interaction_duration()` in `download_annotated_interactions.py`). Assembled from the dataset's `interactions.csv`; the `duration` field is added separately.

**`session_relationship.json`** — Whether the dyad members are strangers or familiar, plus detail (e.g., friends, coworkers). Assembled from `relationships.csv`.

**`participants_metadata.json`** — Big Five personality raw scores for both participants. Many values are "Undisclosed". Assembled from `participants.csv`.

**`filelist_entries.json`** — Raw metadata rows for both participants from `filelist.csv`, including annotation flags, movement feature availability (`has_imitator_movement`), dataset split, and batch info.

#### `interaction/` — Derived files (created by us)

**`timestamps_by_turn.json`** — Conversational turn structure derived from both participants' VAD files. Each entry is one turn (a maximal stretch of one participant's speech uninterrupted by the other). Fields:

- `participant_id` — who is speaking
- `start` — turn start time in seconds
- `end` — turn end time in seconds
- `overlapping` — `true` if the other participant has any VAD activity within this turn's window

Turns are ordered chronologically and alternate speakers unless overlap is present. A turn boundary is created only when the other participant speaks — pauses within a single participant's speech do not create new turns.

**`pre_and_post_moi/time_windows_pre_post.json`** — Fixed-duration time windows around each MOI from 3P-IS annotations. Each entry corresponds to one 3P-IS annotation. Fields:

- `event_speaker` — participant ID of whoever is speaking most during the MOI (determined by overlap with `timestamps_by_turn.json`); `null` if tied or no one is speaking
- `annotated_participant` — participant ID whose 3P-IS file this annotation comes from (i.e., the participant being observed)
- `start_pre_moi` — start of the pre-MOI window (`start_moi - window` seconds)
- `end_pre_moi` — end of the pre-MOI window (equal to `start_moi`)
- `start_moi` — start of the MOI in seconds (from the 3P-IS annotation's `start_ts`)
- `end_moi` — end of the MOI in seconds (from `end_ts`)
- `start_post_moi` — start of the post-MOI window (equal to `end_moi`)
- `end_post_moi` — end of the post-MOI window (`end_moi + window` seconds)

Entries are sorted chronologically by `start_moi`. Default window size is ±15 seconds.

**`pre_and_post_moi/turns_pre_post.json`** — Turn-based windows around each MOI, defined by the nearest turn of the **non-annotated** participant (the person whose 3P-IS file this annotation does NOT come from). Fields:

- `event_speaker` — same as above
- `annotated_participant` — same as above
- `non_annotated_participant` — participant ID of the other participant
- `start_pre_moi` — start of the non-annotated participant's last turn that begins before `start_moi`; `null` if none exists
- `end_pre_moi` — end of that same pre-MOI turn; `null` if none exists
- `pre_overlap` — `true` if the pre-MOI turn extends past `start_moi` into the MOI window
- `start_moi` — start of the MOI in seconds
- `end_moi` — end of the MOI in seconds
- `start_post_moi` — start of the non-annotated participant's first turn that ends after `end_moi`; `null` if none exists
- `end_post_moi` — end of that same post-MOI turn; `null` if none exists
- `post_overlap` — `true` if the post-MOI turn begins before `end_moi`, overlapping into the MOI window

Entries are sorted chronologically by `start_moi`.

**`moi_emotion_summaries.json`** — Per-MOI emotion summary statistics for both participants. Each entry corresponds to one MOI (one 3P-IS annotation) and contains frame indices for slicing the raw `emotion_features.npz` arrays, plus precomputed mean valence, mean arousal, and mean Ekman emotion logits for both the annotated and non-annotated participant. Fields:

- `interaction_id` — interaction directory name (e.g., `V00_S1132_I00000333`)
- `annotated_participant` — participant ID whose 3P-IS file this MOI came from
- `non_annotated_participant` — the other participant in the interaction
- `imitator_emotion_features_present` — `true` if both participants have `emotion_features.npz`
- `annotation` — free-text 3P-IS annotation
- `start_ts` — MOI start time in seconds (integer)
- `end_ts` — MOI end time in seconds (integer)
- `start_idx` — first frame index at 30 Hz (inclusive); computed as `start_ts × 30`
- `end_idx` — last frame index at 30 Hz (exclusive); computed as `end_ts × 30`, clamped to array length
- `n_frames` — number of frames in the window (`end_idx - start_idx`)
- `annotated_mean_valence` — mean `emotion_valence` over the window (annotated participant)
- `annotated_mean_arousal` — mean `emotion_arousal` over the window (annotated participant)
- `annotated_mean_emotion_scores` — dict of mean logits for each of the 8 Ekman categories (annotated participant)
- `non_annotated_mean_valence` — mean `emotion_valence` over the window (non-annotated participant)
- `non_annotated_mean_arousal` — mean `emotion_arousal` over the window (non-annotated participant)
- `non_annotated_mean_emotion_scores` — dict of mean logits for each of the 8 Ekman categories (non-annotated participant)

All emotion fields are `null` when the corresponding participant's `emotion_features.npz` is missing. Entries are sorted chronologically by `start_ts`. The stored indices allow downstream scripts to slice the raw arrays and compute alternative statistics without redoing the timestamp-to-frame alignment.

**`video_viewer.html`** — Self-contained HTML page for viewing both participants' videos side by side, streamed from Meta's S3 bucket. Includes a synchronized Play Both / Pause Both button, both conversation prompts from `interaction_metadata.json` (without prompt-to-participant mapping, which is not available in the released metadata), and a per-participant listing of all Moments of Interest with their three annotation layers (Internal State from 3P-IS, Rationale from 3P-R, Visual Element from 3P-V). Each MOI has a jump button that seeks both videos to that MOI's start time and plays them simultaneously. Generated by `generate_video_viewer()` in `download_annotated_interactions.py`.

#### `participant_a_<id>/` and `participant_b_<id>/` — Per-participant files

**`3P-IS_<file_id>.json`** — Third-party perceived **internal state** annotations. JSONL format (one JSON object per line despite the `.json` extension). Each line: `{"annotation": "<free text>", "start_ts": <seconds>, "end_ts": <seconds>}`. These are the MOI annotations — each line is one Moment of Interest identified by a 3P annotator.

**`3P-R_<file_id>.json`** — Third-party perceived behavior **rationale** annotations. Same JSONL format and timestamps as 3P-IS. Each line explains *why* the annotator thought the participant exhibited the perceived internal state.

**`3P-V_<file_id>.json`** — Third-party **visual element** description annotations. Same JSONL format and timestamps. Each line describes the specific visual cue (gesture, expression, posture change) that prompted the annotation.

**`vad_<file_id>.jsonl`** — Silero Voice Activity Detection segments. JSONL format. Each line: `{"start": <seconds>, "end": <seconds>}`, representing a contiguous speech segment for this participant.

**`<file_id>.wav`** — Per-participant audio (mono, 48 kHz, 32-bit float), echo-cancelled via Beryl AEC. Excluded from the GitHub copy due to size; only present if downloaded with `--include-wav`.

**`emotion_features.npz`** — NumPy compressed archive containing 5 emotion-related features extracted from Meta's Imitator face-tracking model at 30 Hz. Load with `np.load("emotion_features.npz")`. Keys:

- `emotion_arousal` — continuous arousal score per frame, range [-1, 1]
- `EmotionArousalToken` — quantized arousal, 12 discrete bins
- `emotion_valence` — continuous valence score per frame, range [-1, 1]
- `EmotionValenceToken` — quantized valence, 12 discrete bins
- `emotion_scores` — 8-category Ekman emotion logits per frame (shape: n_frames × 8; raw logits, not softmax probabilities)

Not present for participants whose `has_imitator_movement` flag in `filelist.csv` is `0` (Meta's Imitator model was not successfully run on their video).

---

## Project-Level Files

### `3p_is_adjectives.json`

A vocabulary of 190 unique adjectives extracted from the 492 3P-IS annotations across all 100 interactions, organized by engagement and sentiment. Top-level fields:

- `total_annotations` — total number of 3P-IS annotations (492)
- `unique_adjectives` — number of distinct adjectives found (190)
- `total_occurrences` — total adjective occurrences across all annotations (637; some annotations contain multiple adjectives)
- `engaged` — adjectives describing engaged states (174 unique, 604 occurrences)
- `disengaged` — adjectives describing disengaged states (16 unique, 33 occurrences)

Each engagement category (`engaged`, `disengaged`) contains three sentiment sub-groups — `positive`, `neutral`, and `negative` — each with a `count` (number of unique adjectives), `occurrences` (total usage count), and an `adjectives` dict mapping each adjective to its occurrence count. For example, "amused" appears 28 times under engaged/positive, while "surprised" appears 19 times under engaged/negative.

### `moi_valence_arousal_table.csv`

A flat CSV with one row per MOI (492 rows), aggregated from all per-interaction `moi_emotion_summaries.json` files by `build_moi_valence_arousal_table.py`. Columns:

- `interaction_id` — interaction directory name (e.g., `V00_S1132_I00000333`)
- `annotated_participant` — participant ID whose 3P-IS file this MOI came from
- `non_annotated_participant` — the other participant in the interaction
- `emotion_features_present` — `True` if both participants have `emotion_features.npz`, `False` otherwise
- `annotation` — free-text 3P-IS annotation (e.g., "The participant was delighted")
- `start_ts` — MOI start time in seconds
- `end_ts` — MOI end time in seconds
- `start_idx` — first frame index at 30 Hz (inclusive)
- `end_idx` — last frame index at 30 Hz (exclusive)
- `n_frames` — number of frames in the window
- `annotated_mean_valence` — mean valence for the annotated participant over the MOI window
- `annotated_mean_arousal` — mean arousal for the annotated participant over the MOI window
- `non_annotated_mean_valence` — mean valence for the non-annotated participant over the MOI window
- `non_annotated_mean_arousal` — mean arousal for the non-annotated participant over the MOI window

Valence and arousal values are empty for MOIs in the 46 interactions that lack emotion features. Ekman emotion scores are intentionally excluded to keep the CSV spreadsheet-friendly; use the per-interaction `moi_emotion_summaries.json` files for those.

---

## Scripts

Everything below is reference for running and maintaining the download/generation scripts.

### Prerequisites

1. **Python 3.8+** with **numpy** (numpy is required by `download_emotion_features.py`; the other scripts use only the standard library)
2. **Clone the seamless_interaction repository**:
   ```bash
   git clone https://github.com/facebookresearch/seamless_interaction.git
   ```
   The scripts read CSV files from the `assets/` directory in this repo:
   - `filelist.csv` — master list of all files in the dataset with annotation flags
   - `interactions.csv` — prompt text, IPC codes, and interaction types
   - `participants.csv` — Big Five personality scores
   - `relationships.csv` — stranger/familiar status per session
3. **Internet access** to download from Meta's public S3 bucket (`dl.fbaipublicfiles.com`)

### `download_annotated_interactions.py`

Downloads the 100 both-annotated naturalistic interactions from Meta's S3 bucket. For each interaction, it creates the directory structure (`interaction/`, `participant_a_{id}/`, `participant_b_{id}/`), assembles interaction-level metadata JSON files from the dataset's CSVs, and downloads per-participant annotation files (3P-IS, 3P-R, 3P-V) and VAD files. Audio (.wav) files are excluded by default. Also provides `generate_video_viewer()` (creates `video_viewer.html` for any interaction) and `get_interaction_duration()` (probes the MP4 on S3 via ffprobe to get the exact video duration; requires ffprobe), both imported by other scripts.

```bash
# Download annotations, VAD, and metadata only (no audio)
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction

# Also download per-participant audio (.wav) files
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --include-wav

# Preview what would be downloaded without actually downloading
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --dry-run
```

- `--repo-path` (required): path to the cloned [seamless_interaction](https://github.com/facebookresearch/seamless_interaction) GitHub repository. Must contain the `assets/` directory with `filelist.csv`, `interactions.csv`, `participants.csv`, and `relationships.csv`.
- `--output-dir` (optional, default `.`): directory in which to create the `both_annotated_interactions/` folder.
- `--include-wav` (optional): also download per-participant audio files. Excluded by default to keep the download lightweight.
- `--num-workers` (optional, default 4): number of parallel download threads.
- `--dry-run` (optional): print what would be downloaded without actually downloading.

The script identifies the 100 interactions by filtering `filelist.csv`: keep only `label == "naturalistic"` (93,620 of 129,370 entries) → keep only `has_annotation_3p == "1"` (543 entries) → group by interaction ID → keep only interactions with exactly 2 entries (both participants annotated) → **100 interactions** (200 file entries). Without `--include-wav`, the total download is under 100 MB. With `--include-wav`, roughly **5–15 GB**. Common issues: `"filelist.csv not found"` means `--repo-path` should point to the repo root (not `assets/`); HTTP 403 errors are normal for files that don't exist on S3; re-running is safe (existing files are overwritten).

### `download_single_annotated_interactions.py`

Downloads the 343 naturalistic interactions where exactly one of the two participants has 3P annotations. Creates the same directory structure as `both_annotated_interactions/` — interaction-level metadata, per-participant annotation files (3P-IS, 3P-R, 3P-V for the annotated participant only), and VAD for both participants. Also generates all derived files in a single pass: `timestamps_by_turn.json`, `time_windows_pre_post.json`, `turns_pre_post.json`, and `video_viewer.html`. Does **not** download `emotion_features.npz` or generate `moi_emotion_summaries.json`.

```bash
# Download and generate all derived files
python scripts/download_single_annotated_interactions.py --repo-path /path/to/seamless_interaction --output-dir .

# Preview what would be downloaded without actually downloading
python scripts/download_single_annotated_interactions.py --repo-path /path/to/seamless_interaction --output-dir . --dry-run
```

- `--repo-path` (required): path to the cloned [seamless_interaction](https://github.com/facebookresearch/seamless_interaction) GitHub repository.
- `--output-dir` (optional, default `..`): directory in which to create the `single_annotated_interactions/` folder.
- `--include-wav` (optional): also download per-participant audio files. Excluded by default.
- `--num-workers` (optional, default 4): number of parallel download threads.
- `--window` (optional, default 15): seconds before/after each MOI for pre/post time windows.
- `--dry-run` (optional): print what would be downloaded without actually downloading.

The script identifies the 343 interactions by filtering `filelist.csv`: keep only naturalistic entries → group by interaction ID → keep interactions where exactly 1 of 2 participants has `has_annotation_3p == "1"`. Imports and reuses functions from `download_annotated_interactions.py`, `generate_timestamps_by_turn.py`, `generate_moi_time_windows.py`, and `generate_moi_turns.py`.

### `generate_timestamps_by_turn.py`

Generates `timestamps_by_turn.json` for each interaction. Must be run before the two MOI scripts below.

```bash
python generate_timestamps_by_turn.py both_annotated_interactions
```

- Takes a single positional argument: path to the `both_annotated_interactions/` directory.

### `generate_moi_time_windows.py`

Generates `time_windows_pre_post.json` for each interaction.

```bash
python generate_moi_time_windows.py --input-dir both_annotated_interactions --window 15
```

- `--input-dir` (required): path to the `both_annotated_interactions/` directory.
- `--window` (optional, default 15): seconds before and after each MOI.

### `generate_moi_turns.py`

Generates `turns_pre_post.json` for each interaction.

```bash
python generate_moi_turns.py --input-dir both_annotated_interactions
```

- `--input-dir` (required): path to the `both_annotated_interactions/` directory.

### `valence_arousal/download_emotion_features.py`

Downloads 5 emotion features from Meta's S3 bucket for each participant and saves them as `emotion_features.npz` in each participant's directory. Requires `both_annotated_interactions/` to already exist (created by `download_annotated_interactions.py`). Skips participants whose `has_imitator_movement` flag is `0`.

```bash
# Download emotion features for all participants
python download_emotion_features.py --repo-path /path/to/seamless_interaction

# Preview what would be downloaded
python download_emotion_features.py --repo-path /path/to/seamless_interaction --dry-run

# Re-download even if emotion_features.npz already exists
python download_emotion_features.py --repo-path /path/to/seamless_interaction --overwrite
```

- `--repo-path` (required): path to the cloned [seamless_interaction](https://github.com/facebookresearch/seamless_interaction) GitHub repository.
- `--interactions-dir` (optional, default `./both_annotated_interactions`): path to the `both_annotated_interactions/` directory.
- `--num-workers` (optional, default 4): number of parallel download threads.
- `--dry-run` (optional): print what would be downloaded without actually downloading.
- `--overwrite` (optional): re-download even if `emotion_features.npz` already exists.


### `valence_arousal/extract_moi_summaries.py`

For each interaction, reads the 3P-IS annotation files (which define Moments of Interest) and `emotion_features.npz` for both participants, then writes `interaction/moi_emotion_summaries.json` containing one record per MOI with frame indices, mean valence/arousal, and mean Ekman emotion logits for both the annotated and non-annotated participant. Requires `emotion_features.npz` to already exist (created by `download_emotion_features.py`).

```bash
# Extract summaries for all interactions
python scripts/valence_arousal/extract_moi_summaries.py --interactions-dir ./both_annotated_interactions

# Only process interactions where both participants have emotion features (54 of 100)
python scripts/valence_arousal/extract_moi_summaries.py --interactions-dir ./both_annotated_interactions --both-required

# Preview without writing
python scripts/valence_arousal/extract_moi_summaries.py --interactions-dir ./both_annotated_interactions --dry-run
```

- `--interactions-dir` (optional, default `./both_annotated_interactions`): path to the `both_annotated_interactions/` directory.
- `--both-required` (optional): only process interactions where both participants have `emotion_features.npz` (54 of 100). Default: process all.
- `--overwrite` (optional): overwrite existing `moi_emotion_summaries.json` files.
- `--dry-run` (optional): print what would be processed without writing files.

### `valence_arousal/build_moi_valence_arousal_table.py`

Aggregates all per-interaction `moi_emotion_summaries.json` files into a single flat CSV (`moi_valence_arousal_table.csv`) with one row per MOI. Requires `moi_emotion_summaries.json` to already exist in each interaction (created by `extract_moi_summaries.py`). Ekman emotion scores are excluded from the CSV to keep it spreadsheet-friendly; use the per-interaction JSON files for those.

```bash
# Build the table (default output: ./moi_valence_arousal_table.csv)
python scripts/valence_arousal/build_moi_valence_arousal_table.py --interactions-dir ./both_annotated_interactions

# Specify a custom output path
python scripts/valence_arousal/build_moi_valence_arousal_table.py --interactions-dir ./both_annotated_interactions --output ./custom_path.csv
```

- `--interactions-dir` (optional, default `./both_annotated_interactions`): path to the `both_annotated_interactions/` directory.
- `--output` (optional, default `./moi_valence_arousal_table.csv`): output CSV file path.

