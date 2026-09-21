# Turn-Taking Prediction

Predicting **when a speaker is about to give up the floor** in natural face-to-face
conversation — from audio, facial behavior, and cross-participant *coordination* —
on the [Seamless Interaction](https://ai.meta.com/datasets/seamless-interaction/)
dyadic corpus.

## The task

Given a 2-second window of conversation ending at time *t*, classify what happens at
the window boundary, τ milliseconds ahead of *t*:

| Class | Meaning |
| --- | --- |
| `HOLD` | the current speaker keeps the floor |
| `YIELD` | the floor cleanly transfers to the other participant |
| `BACKCHANNEL` | the listener vocalizes ("mm-hm", "right") without taking the floor |

Labels are derived from per-participant Silero VAD plus WhisperX word-aligned
transcripts. Three further outcomes — `INTERRUPT`, `FAILED`, `LAPSE` — are detected
and retained for corpus statistics but excluded from training, because each is a
genuinely different phenomenon rather than a noisy version of the three above.

## Data

- **456 naturalistic dyadic interactions** from Seamless Interaction, indexed by
  [`manifests/manifest.csv`](manifests/manifest.csv). Prompt-driven "improvised"
  interactions are excluded — the prompt itself confounds turn-taking behavior.
- **320 / 69 / 67** train / val / test interactions, split by participant-component
  bin-packing so no participant appears in two splits.
- [`both_annotated_interactions/`](both_annotated_interactions/) ships the 100
  interactions where *both* participants carry third-party annotations — VAD,
  per-turn timestamps, emotion features, and interaction metadata.

## Approach

**Features**

| Stream | Source |
| --- | --- |
| Audio | CPC representations over 2 s windows (Facebook Research CPC) |
| Face | 23-dim OpenFace per-frame features, windowed |
| Coordination | Windowed cross-correlation (WCC) between participants across a lag grid — both a 19-scalar summary and the full (21, 23) continuous array |

**Fusion architectures** (`scripts/fusion_lib.py`): early fusion, neural concatenation,
self-attention, cross-attention, and self+cross-attention over the stream set.

**Ablation matrix** — 24 experiments across five blocks:

| Block | Experiments | What it isolates |
| --- | --- | --- |
| `standard` | 5 | stream set, no attention |
| `ssa` | 5 | + self-attention |
| `sca` | 2 | + cross-attention between streams |
| `coordination` | 6 | + WCC coordination features |
| `csa` | 6 | coordination + self-attention |

Each is swept over a τ grid of 100 / 200 / 400 / 500 / 800 / 1600 ms; τ = 400 ms is
the training horizon.

## Results

Configurations ranked at τ = 400 ms (test set, macro F1):

| # | Block | Streams | Architecture | Macro F1 | Macro recall |
| --- | --- | --- | --- | --- | --- |
| 1 | Coordination + Self-Attn | Full Dyad + WCC-Continuous | Self-Attention | 0.707 | 0.722 |
| 2 | Standard + Self-Attn | Audio Dyad | Self-Attention | 0.705 | 0.728 |
| 3 | Coordination + Self-Attn | Audio Dyad + WCC-Continuous | Self-Attention | 0.703 | 0.718 |
| 4 | Standard | Audio Dyad | Neural-Concat | 0.697 | 0.739 |
| 5 | Standard + Self-Attn | Full Dyad | Self-Attention | 0.688 | 0.719 |

Results by class:

| Class | Best F1 | Mean F1 across all 24 |
| --- | --- | --- |
| `HOLD` | 0.967 | 0.924 |
| `YIELD` | 0.606 | 0.442 |
| `BACKCHANNEL` | 0.572 | 0.373 |

## Repository layout

```
manifests/           manifest.csv (456 interactions, with splits) + poc_manifest.csv
model_input/         per-τ ground-truth labels (labels_tau_*.json)
both_annotated_interactions/
                     100 interactions with both participants annotated
results/             per-experiment metrics — see results/README.md for the data dictionary
paper/               generated figures and LaTeX tables
scripts/             the pipeline (see below)
tests/               pytest suite for fusion_runner / fusion_lib
```

### `scripts/`

| Stage | Script |
| --- | --- |
| Download | `download_annotated_interactions.py`, `download_audio_from_manifest.py`, `download_video_from_manifest.py`, `download_vad_and_transcript_from_manifest.py` |
| Manifest | `build_manifest.py` — assembles the pool, assigns 70/15/15 participant-disjoint splits |
| Labels | `build_labeled_windows_from_manifest.py` — per-τ HOLD/YIELD/BACKCHANNEL from VAD + transcripts |
| Features | `splice_wavs.py` → `extract_cpc_from_manifest.py`; `coordination/extract_coordination_windows.py` for WCC |
| Experiments | `fusion_lib.py` (models, data loading, train/eval), `fusion_runner.py` (one ablation block), `fusion_experiments.ipynb` (orchestrates all 24) |
| Reporting | `gen_tables_figs.py` — regenerates everything under `paper/` |

## Reproducing

```bash
pip install -r requirements.txt

# Figures and tables from the shipped results — no dataset access needed
python scripts/gen_tables_figs.py

# Tests
pytest
```

The full pipeline requires Seamless Interaction dataset access, then runs in order:
`download_annotated_interactions.py` → `build_manifest.py` →
`download_{audio,vad_and_transcript}_from_manifest.py` →
`build_labeled_windows_from_manifest.py` → `splice_wavs.py` →
`extract_cpc_from_manifest.py` → `coordination/extract_coordination_windows.py` →
`fusion_experiments.ipynb`.

Intermediate artifacts land in `subset/` and `fusion_runs/`, both gitignored — they run
to hundreds of gigabytes. `manifests/`, `model_input/`, and `results/` are committed so
the reporting stage is reproducible without re-running training.

## Requirements

Python 3.12, PyTorch. See [`requirements.txt`](requirements.txt).
