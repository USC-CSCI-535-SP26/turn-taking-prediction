# Annotated Interactions

This directory contains the **100 naturalistic interactions** from Meta's [Seamless Interaction dataset](https://github.com/facebookresearch/seamless_interaction) where **both participants** in the dyad have third-party (3P) annotations. These 100 were identified by filtering the dataset's `filelist.csv` for naturalistic entries with `has_annotation_3p == 1`, grouping by interaction, and keeping only interactions where both participants were annotated.

Audio (`.wav`) files are excluded from this directory to keep the repository lightweight. They can be downloaded separately using `download_annotated_interactions.py` in the parent directory.

## Interaction Naming Convention

Each interaction directory is named `V{vendor}_S{session}_I{prompt_hash}`:

- **V{vendor}** — The data collection vendor/site (only V00 and V03 appear in the annotated subset).
- **S{session}** — The recording session: a 1-hour continuous recording of a specific dyad guided by a moderator.
- **I{prompt_hash}** — A hash identifying the prompt pair given to the dyad for this interaction. This is NOT a sequence number — the same prompt hash can appear across many sessions, and the numbering does not reflect chronological order within a session.

A unique interaction instance (one specific dyad performing one specific prompt) is identified by the full V+S+I combination.

Within each interaction directory, participant subdirectories are named `participant_a_{id}` and `participant_b_{id}`, assigned alphabetically by participant ID (lower ID = participant_a). This is an arbitrary convention and does not correspond to the participant_a/participant_b prompt roles in the dataset's `interactions.csv`.

## Directory Structure

```
annotated_interactions/
├── README.md                              ← this file (created by us)
│
├── V00_S1288_I00000099/
│   ├── interaction/
│   │   ├── interaction_metadata.json      ← created by us (from dataset CSVs)
│   │   ├── session_relationship.json      ← created by us (from dataset CSVs)
│   │   ├── participants_metadata.json     ← created by us (from dataset CSVs)
│   │   ├── filelist_entries.json          ← created by us (from dataset CSVs)
│   │   └── timestamps_by_turn.json        ← created by us (from VAD files)
│   ├── participant_a_P0071/
│   │   ├── 3P-IS_..._P0071.json          ← from dataset
│   │   ├── 3P-R_..._P0071.json           ← from dataset
│   │   ├── 3P-V_..._P0071.json           ← from dataset
│   │   └── vad_..._P0071.jsonl            ← from dataset
│   └── participant_b_P1119/
│       ├── 3P-IS_..._P1119.json           ← from dataset
│       ├── 3P-R_..._P1119.json            ← from dataset
│       ├── 3P-V_..._P1119.json            ← from dataset
│       └── vad_..._P1119.jsonl             ← from dataset
│
├── V00_S1288_I00000102/
│   └── ...
└── ... (100 interactions total)
```

## File Descriptions

### Files from the dataset (downloaded from Meta's S3 bucket)

**Per participant** (in `participant_a_{id}/` and `participant_b_{id}/`):

- **`3P-IS_{file_id}.json`** — Third-party perceived **Internal State** annotations. Each line is a JSON object with `annotation` (free-text description of the participant's perceived emotional/psychological state), `start_ts`, and `end_ts` (seconds). Despite the `.json` extension, these are JSONL format (one JSON object per line).

- **`3P-R_{file_id}.json`** — Third-party perceived **Rationale** annotations. Each line describes the annotator's reasoning for why the participant behaved a certain way at that moment. Same JSONL format and fields as 3P-IS.

- **`3P-V_{file_id}.json`** — Third-party **Visual element** annotations. Each line describes the specific visual behaviors (gestures, facial expressions, body movements) that the annotator observed. Same JSONL format and fields as 3P-IS.

- **`vad_{file_id}.jsonl`** — **Voice Activity Detection** segments produced by Silero VAD. Each line is `{"start": <seconds>, "end": <seconds>}` marking when the participant was speaking. Note that these represent continuous speech segments, not conversational turns — a single turn may consist of multiple VAD segments separated by pauses.

### Files created by us

**Per interaction** (in `interaction/`):

- **`interaction_metadata.json`** — Prompt text for both participants, IPC (Interpersonal Circumplex) octant codes, and interaction type. Assembled from the dataset's `interactions.csv` using the prompt hash.

- **`session_relationship.json`** — Whether the dyad members are strangers or familiar, plus relationship detail (e.g., friends, coworkers, siblings). Assembled from `relationships.csv` using the vendor and session ID.

- **`participants_metadata.json`** — Big Five personality raw scores (extraversion, agreeableness, conscientiousness, neuroticism, openness) for both participants. Many entries are "Undisclosed". Assembled from `participants.csv`.

- **`filelist_entries.json`** — The raw `filelist.csv` rows for both participants, including annotation flags (`has_annotation_1p`, `has_annotation_3p`), imitator movement availability, data split, and batch info.

- **`timestamps_by_turn.json`** — Conversational turn structure derived from both participants' VAD files. Each entry represents one turn (a maximal stretch of one participant's speech uninterrupted by the other) with four fields:
  - `participant_id`: who is speaking
  - `start`: turn start time in seconds
  - `end`: turn end time in seconds
  - `overlapping`: `true` if the other participant has any VAD activity within this turn's time window

  Turns are ordered chronologically. Consecutive turns alternate speakers unless overlap is present. A turn boundary is created only when the other participant speaks — pauses within a single participant's speech do not create new turns.

## Dataset Citation

The underlying data comes from the Seamless Interaction dataset by Meta FAIR:

> Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset. arXiv:2506.22554, 2025.

Licensed under CC-BY-NC 4.0.
