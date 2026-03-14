# Download Both-Annotated Naturalistic Interactions

This script downloads the **100 naturalistic interactions** from Meta's [Seamless Interaction dataset](https://github.com/facebookresearch/seamless_interaction) where **both participants** in the dyad have third-party (3P) annotations.

## Background

The Seamless Interaction dataset contains ~65,000 interactions totaling 4,065 hours of in-person dyadic audiovisual data. A subset of interactions were annotated by trained third-party (3P) annotators who identified Moments of Interest (conspicuous visual behaviors) and provided:

- **3P-IS**: Perceived internal state of the participant at that moment
- **3P-R**: Rationale for the perceived behavior
- **3P-V**: Description of the visual element that prompted the annotation

Among the naturalistic interactions, 543 participant-level file entries have 3P annotations, spanning 443 unique interactions. Of those, **100 interactions have both participants annotated** (the remaining 343 have only one participant annotated). This script downloads all data for those 100 both-annotated interactions.

## Prerequisites

1. **Python 3.8+** (no external packages required -- uses only the standard library)
2. **Clone the seamless_interaction repository**:
   ```bash
   git clone https://github.com/facebookresearch/seamless_interaction.git
   ```
   The script reads CSV files from the `assets/` directory in this repo:
   - `filelist.csv` -- master list of all files in the dataset with annotation flags
   - `interactions.csv` -- prompt text, IPC codes, and interaction types
   - `participants.csv` -- Big Five personality scores
   - `relationships.csv` -- stranger/familiar status per session
3. **Internet access** to download from Meta's public S3 bucket (`dl.fbaipublicfiles.com`)

## Usage

```bash
# Basic usage (annotations + VAD only, no audio)
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction

# Include audio (.wav) files
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --include-wav

# Specify output directory
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --output-dir /path/to/output

# Dry run (prints what would be downloaded without downloading)
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --dry-run

# Use more download threads (default: 4)
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --num-workers 8
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--repo-path` | Yes | -- | Path to the cloned `seamless_interaction` GitHub repository |
| `--output-dir` | No | `.` (current directory) | Directory in which to create the `annotated_interactions/` folder |
| `--include-wav` | No | off | Also download per-participant audio (.wav) files |
| `--num-workers` | No | `4` | Number of parallel download threads |
| `--dry-run` | No | off | Print what would be downloaded without actually downloading |

## What Gets Downloaded

### Per interaction (in the `interaction/` subdirectory)

These are assembled from the repo's CSV metadata files (not downloaded from S3):

| File | Source | Contents |
|---|---|---|
| `interaction_metadata.json` | `interactions.csv` | Prompt text for both participants, IPC octant codes (Agency/Communion), interaction type |
| `session_relationship.json` | `relationships.csv` | Whether the dyad members are strangers or familiar, plus detail (e.g., friends, coworkers) |
| `participants_metadata.json` | `participants.csv` | Big Five personality raw scores for both participants (many are "Undisclosed") |
| `filelist_entries.json` | `filelist.csv` | Raw metadata rows for both participants, including annotation flags, movement availability, split, and batch info |

### Per participant (in `participant_a_<id>/` and `participant_b_<id>/` subdirectories)

These are downloaded from Meta's public S3 bucket:

| File | Format | Description |
|---|---|---|
| `3P-IS_<file_id>.json` | JSONL (one JSON object per line) | Third-party perceived internal state annotations |
| `3P-R_<file_id>.json` | JSONL | Third-party perceived behavior rationale annotations |
| `3P-V_<file_id>.json` | JSONL | Third-party visual element description annotations |
| `vad_<file_id>.jsonl` | JSONL | Silero Voice Activity Detection segments (`{"start": <sec>, "end": <sec>}`) |
| `<file_id>.wav` | WAV (mono, 48kHz, 32-bit float) | Per-participant audio, echo-cancelled via Beryl AEC. **Only with `--include-wav`.** |

**Without `--include-wav`**: 4 metadata + 4 per-participant files × 2 participants = **12 files** per interaction (**1,200 total**).

**With `--include-wav`**: 4 metadata + 5 per-participant files × 2 participants = **14 files** per interaction (**1,400 total**).

## Interaction Naming Convention

Each interaction directory is named `V{vendor}_S{session}_I{prompt_hash}`:

- **V{vendor}** — The data collection vendor/site (only V00 and V03 appear in the annotated subset).
- **S{session}** — The recording session: a 1-hour continuous recording of a specific dyad guided by a moderator.
- **I{prompt_hash}** — A hash identifying the prompt pair given to the dyad for this interaction. This is NOT a sequence number — the same prompt hash can appear across many sessions, and the numbering does not reflect chronological order within a session.

A unique interaction instance (one specific dyad performing one specific prompt) is identified by the full V+S+I combination.

## Output Directory Structure

The download script produces `annotated_interactions/` with WAV files included. A separate copy (`annotated_interactions/` without WAVs) is maintained for the GitHub repository.

```
annotated_interactions/
├── V00_S1288_I00000099/
│   ├── interaction/
│   │   ├── interaction_metadata.json      ← created by download script (from dataset CSVs)
│   │   ├── session_relationship.json      ← created by download script (from dataset CSVs)
│   │   ├── participants_metadata.json     ← created by download script (from dataset CSVs)
│   │   ├── filelist_entries.json          ← created by download script (from dataset CSVs)
│   │   ├── timestamps_by_turn.json        ← created by us (from VAD files)
│   │   └── pre_and_post_moi/
│   │       ├── time_windows_pre_post.json ← created by us (from 3P-IS + turns)
│   │       └── turns_pre_post.json        ← created by us (from 3P-IS + turns)
│   ├── participant_a_P0071/
│   │   ├── 3P-IS_..._P0071.json          ← from dataset
│   │   ├── 3P-R_..._P0071.json           ← from dataset
│   │   ├── 3P-V_..._P0071.json           ← from dataset
│   │   ├── V00_S1288_I00000099_P0071.wav  ← from dataset (excluded from GitHub copy)
│   │   └── vad_..._P0071.jsonl            ← from dataset
│   └── participant_b_P1119/
│       ├── 3P-IS_..._P1119.json           ← from dataset
│       ├── 3P-R_..._P1119.json            ← from dataset
│       ├── 3P-V_..._P1119.json            ← from dataset
│       ├── V00_S1288_I00000099_P1119.wav   ← from dataset (excluded from GitHub copy)
│       └── vad_..._P1119.jsonl             ← from dataset
│
├── V00_S1288_I00000102/
│   └── ...
└── ... (100 interactions total)
```

## Files Created by Us

In addition to the files downloaded from the dataset, we generate the following derived files.

**Per interaction** (in `interaction/`):

- **`timestamps_by_turn.json`** — Conversational turn structure derived from both participants' VAD files. Each entry represents one turn (a maximal stretch of one participant's speech uninterrupted by the other) with four fields:
  - `participant_id`: who is speaking
  - `start`: turn start time in seconds
  - `end`: turn end time in seconds
  - `overlapping`: `true` if the other participant has any VAD activity within this turn's time window

  Turns are ordered chronologically. Consecutive turns alternate speakers unless overlap is present. A turn boundary is created only when the other participant speaks — pauses within a single participant's speech do not create new turns.

**Per interaction** (in `interaction/pre_and_post_moi/`):

- **`time_windows_pre_post.json`** — Fixed-duration time windows around each Moment of Interest (MOI) from 3P-IS annotations. Each entry corresponds to one 3P-IS annotation and contains:
  - `event_speaker`: participant ID of whoever is speaking most during the MOI, determined by overlap with `timestamps_by_turn.json`. `null` if both participants have equal overlap or no one is speaking.
  - `annotated_participant`: participant ID whose 3P-IS file this annotation comes from (i.e., the participant being observed).
  - `start_pre_moi`: start of the pre-MOI window (`start_moi - window` seconds).
  - `end_pre_moi`: end of the pre-MOI window (equal to `start_moi`).
  - `start_moi`: start of the MOI in seconds (from the 3P-IS annotation's `start_ts`).
  - `end_moi`: end of the MOI in seconds (from the 3P-IS annotation's `end_ts`).
  - `start_post_moi`: start of the post-MOI window (equal to `end_moi`).
  - `end_post_moi`: end of the post-MOI window (`end_moi + window` seconds).

  Entries are sorted chronologically by `start_moi`. The default window size is ±15 seconds.

- **`turns_pre_post.json`** — Turn-based windows around each MOI, defined by the nearest turn of the **non-annotated** participant (the person whose 3P-IS file this annotation does NOT come from). Each entry contains:
  - `event_speaker`: same as in `time_windows_pre_post.json`.
  - `annotated_participant`: same as in `time_windows_pre_post.json`.
  - `non_annotated_participant`: participant ID of the other participant in the interaction.
  - `start_pre_moi`: start of the non-annotated participant's last turn that begins before `start_moi`. `null` if no such turn exists.
  - `end_pre_moi`: end of that same pre-MOI turn. `null` if no such turn exists.
  - `pre_overlap`: `true` if the pre-MOI turn extends past `start_moi` into the MOI window itself.
  - `start_moi`: start of the MOI in seconds.
  - `end_moi`: end of the MOI in seconds.
  - `start_post_moi`: start of the non-annotated participant's first turn that ends after `end_moi`. `null` if no such turn exists.
  - `end_post_moi`: end of that same post-MOI turn. `null` if no such turn exists.
  - `post_overlap`: `true` if the post-MOI turn begins before `end_moi`, overlapping into the MOI window itself.

  Entries are sorted chronologically by `start_moi`.

## Scripts

**`download_annotated_interactions.py`** — Downloads the 100 both-annotated naturalistic interactions from Meta's S3 bucket. For each interaction, it creates the directory structure (`interaction/`, `participant_a_{id}/`, `participant_b_{id}/`), assembles interaction-level metadata JSON files from the dataset's CSVs (`interactions.csv`, `relationships.csv`, `participants.csv`, `filelist.csv`), and downloads per-participant annotation files (3P-IS, 3P-R, 3P-V) and VAD files. Audio (.wav) files are excluded by default.

```bash
# Download annotations, VAD, and metadata only (no audio)
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction

# Also download per-participant audio (.wav) files
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --include-wav

# Preview what would be downloaded without actually downloading
python download_annotated_interactions.py --repo-path /path/to/seamless_interaction --dry-run
```

- `--repo-path` (required): path to the cloned [seamless_interaction](https://github.com/facebookresearch/seamless_interaction) GitHub repository. Must contain the `assets/` directory with `filelist.csv`, `interactions.csv`, `participants.csv`, and `relationships.csv`.
- `--output-dir` (optional, default `.`): directory in which to create the `annotated_interactions/` folder.
- `--include-wav` (optional): also download per-participant audio files (mono, 48kHz, 32-bit float WAV, echo-cancelled via Beryl AEC). Excluded by default to keep the download lightweight.
- `--num-workers` (optional, default 4): number of parallel download threads.
- `--dry-run` (optional): print what would be downloaded without actually downloading.

**`generate_timestamps_by_turn.py`** — Generates `timestamps_by_turn.json` for each interaction. Must be run before the two scripts below.

```bash
python generate_timestamps_by_turn.py annotated_interactions
```

- Takes a single positional argument: path to the `annotated_interactions/` directory.

**`generate_moi_time_windows.py`** — Generates `time_windows_pre_post.json` for each interaction.

```bash
python generate_moi_time_windows.py --input-dir annotated_interactions --window 15
```

- `--input-dir` (required): path to the `annotated_interactions/` directory.
- `--window` (optional, default 15): seconds before and after each MOI for the pre/post windows.

**`generate_moi_turns.py`** — Generates `turns_pre_post.json` for each interaction.

```bash
python generate_moi_turns.py --input-dir annotated_interactions
```

- `--input-dir` (required): path to the `annotated_interactions/` directory.

## How the 100 Interactions Are Identified

The script applies the following filtering logic to `filelist.csv`:

1. **Filter by label**: Keep only rows where `label == "naturalistic"` (93,620 of 129,370 total entries)
2. **Filter by 3P annotation**: Keep only rows where `has_annotation_3p == "1"` (543 entries)
3. **Group by interaction**: Extract the interaction ID from each file ID by removing the participant suffix (e.g., `V00_S1288_I00000099_P1119` becomes `V00_S1288_I00000099`). This yields 443 unique interactions.
4. **Filter for both annotated**: Keep only interactions where the group contains exactly 2 entries -- meaning both participants were annotated. This yields **100 interactions** (200 file entries). The remaining 343 interactions have only one participant annotated.

The script prints these counts as it runs so you can verify the filtering logic.

## Note on participant_a vs. participant_b

The dataset's `interactions.csv` defines "participant_a" and "participant_b" prompt roles (each participant receives different prompt text), but there is no mapping from those roles to actual participant IDs in the file names. This script assigns participant_a and participant_b **alphabetically by participant ID** (lower ID = participant_a). This is a deterministic convention but does **not** correspond to the prompt assignment in `interactions.csv`.

## Note on the Annotation File Format

The annotation files (3P-IS, 3P-R, 3P-V) have a `.json` extension but are actually **JSONL format** (one JSON object per line). Each line typically looks like:

```json
{"annotation": "The participant appears attentive...", "start_ts": 45.2, "end_ts": 48.7}
```

The official `seamless_interaction` Python package has a known bug (`fs.py`, method `_wget_download_from_s3`, ~line 997) where it calls `json.load()` on these files, which fails with `JSONDecodeError: Extra data: line 2 column 1` because `json.load()` expects a single JSON object. **This script avoids the bug** by downloading annotation files directly via HTTP and saving the raw bytes without parsing.

To read these files in your own code, use:

```python
import json

annotations = []
with open("3P-IS_V00_S1288_I00000099_P1119.json") as f:
    for line in f:
        annotations.append(json.loads(line.strip()))
```

## Download Size Estimate

Without `--include-wav`, each interaction downloads 8 small annotation/metadata files — the total download is lightweight (under 100 MB). With `--include-wav`, each interaction also downloads 2 WAV files (the largest component), and the total size is roughly **5-15 GB** depending on interaction durations. The `--dry-run` flag lets you preview without downloading.

## Troubleshooting

- **"filelist.csv not found"**: Make sure `--repo-path` points to the root of the cloned `seamless_interaction` repository (not the `assets/` subdirectory itself).
- **Many "NOT FOUND (HTTP 403)" messages**: This is normal for some files. Meta's S3 bucket returns 403 for nonexistent keys. Not all features are available for all interactions.
- **Slow downloads**: WAV files can be large. Try reducing `--num-workers` if you're hitting rate limits, or increasing it if your connection can handle more parallelism.
- **Resuming an interrupted download**: Re-running the script is safe; existing files will be overwritten. It does not currently skip already-downloaded files.
