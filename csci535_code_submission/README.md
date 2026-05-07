# CSCI-535 Project — Code Submission

End-of-turn detection on the Seamless Interaction dataset using dyadic multimodal fusion (audio + visual + coordination features). This bundle contains the source code that produced the experiments in our paper, the manifest CSV indexing which interactions were used, and the two results CSVs that the figure/table generator reads.

## File index

- **`requirements.txt`** — third-party Python packages used across pipeline.

### `manifests/`

- **`manifest.csv`** —  456-interaction manifest produced by `build_manifest.py`. One row per dyadic interaction.

### `results/`

- **`data_for_figs_tau_400.csv`** — one row per experiment (24 total), with all metrics pinned to **τ = 400 ms** (the training horizon)
- **`data_for_figs_tau_sweep.csv`** — one row per **(experiment × τ)** pair (114 total) capturing metric trajectories across the τ ∈ {100, 200, 400, 800, 1600 ms} prediction-horizon grid. Superset of other results csv.

### `scripts/` 
- **`splice_wavs.py`** — slices each `.wav` into 2 sec windows.
- **`extract_cpc_from_manifest.py`** — runs Facebook Research CPC over every spliced wav
- **`fusion_lib.py`** — core library: sample enumeration, `DataLoader` construction, the fusion model families, training/eval loops, and τ-sweep helper
- **`fusion_runner.py`** — runs one ablation block end-to-end; invoked by fusion_experiments notebook.
- **`fusion_experiments.ipynb`** — orchestrator notebook that defines the 5-ablation experiment matrix (`standard` / `ssa` / `sca` / `coordination` / `csa`), runs the experiments, and writes json results under `fusion_runs/`
- **`gen_tables_figs.py`** — generates every paper figure/table
- **`download_audio_from_manifest.py`** — downloads per-participant `.wav` files for every interaction in the manifest
- **`download_video_from_manifest.py`** — downloads per-participant `.mp4` files for the manifest.
- **`download_vad_and_transcript_from_manifest.py`** — downloads per-participant Silero VAD and WhisperX word-aligned transcripts.
- **`build_manifest.py`** — assembles the full-project manifest; assigns 70/15/15 train/val/test splits via participant-component LPT bin-packing
- **`build_labeled_windows_from_manifest.py`** — creates per-τ ground-truth labels (HOLD / YIELD / BACKCHANNEL)  from per-participant VAD + WhisperX 
- **`download_annotated_interactions.py`** — downloads the 100 naturalistic interactions with both-participant 3P annotations


### `scripts/coordination/` 
- **`extract_coordination_windows.py`** — given per-participant OpenFace per-frame features, produces per-window per-dyad continuous coordination arrays using WCC across a lag grid. Outputs `openface_continuous_arrays/` tree consumed downstream
- **`extract_coordination.ipynb`** — reads continuous WCC arrays, computes per-(τ, lag) coordination features, writes `all_coordination_features_tau_{TAU}.csv` (one row per window).
- **`coordination_summary_stats.py`** — joins `all_coordination_features_tau_{TAU}.csv` against per-τ labels json and emits two CSVs per τ: a labeled-rows file and a per-class summary-statistics file 
- **`coodination_window_summary_stats.py`** — reads raw `openface_continuous_arrays/` tree (not the precomputed per-window features), computes per class window sync stats