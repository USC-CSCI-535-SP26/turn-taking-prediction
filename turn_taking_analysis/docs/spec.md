# CSCI-535 Turn-Taking — Project Spec

**Updated:** 2026-04-24 (rev. 3 — encoder switched to CPC, 3-class schema, splits-from-manifest)
**Student:** Sika (USC, CSCI-535 Spring 2026)
**Project root:** `/Users/rasikaramanan/Documents/usc/by_semester/sp26/csci535/project/seamless/csci535-project/turn_taking_analysis/`
**Parent directory:** `../csci535-project/` — holds artifacts from an older project direction. The turn-taking pipeline has a small legacy-coupling dependency on a handful of parent-level scripts (consumed by `build_manifest.py`); see §10. Most other parent-level files are out of scope.
**Scope:** Course-paper-scale dyadic multimodal fusion for end-of-turn detection. Prefer minimum-viable changes to existing code over clean rebuilds.

---

## 1. Research question

**Does adding OpenFace-derived facial behavior to a CPC audio encoder measurably extend the prediction horizon of end-of-turn detection in dyadic conversation, relative to an audio-only baseline?**

> **Audio encoder note.** The project uses **CPC** (Contrastive Predictive Coding; Rivière et al. 2020, causal-RNN aggregator output, **256-dim @ 100 Hz native**; pretrained `60k_epoch4-d0f474de.pt` via `facebookresearch/CPC_audio`). This is the same encoder MM-VAP (Russell & Harte, ACL Findings 2025) uses, so it positions the audio side of this work directly in the MM-VAP lineage rather than against it. CPC's strict causality is load-bearing for prediction-horizon tasks: the feature at frame t is a function of audio only up to t, so per-window features cannot leak signal from inside the forward horizon `[t, t + τ]` (a property bidirectional encoders like WavLM, HuBERT, and wav2vec 2.0 do not have). To enforce that bound exactly, audio is sliced into uniform 2-s windows by `scripts/splice_wavs.py` (writing `subset/audio_sliced/<orig_stem>/<spliced_stem>.wav`), then CPC is run independently per window by `scripts/extract_cpc_from_manifest.py` (writing `subset/cpc/<orig_stem>/<spliced_stem>.npy`, shape `(200, 256)` float32 per 2-s window, plus a sibling `.json` sidecar with full provenance). For early-fusion alignment with OpenFace's 30 fps stream, CPC features are pooled post-hoc to 10 Hz via the `mean_pool_file` helper (factor 10 → 20 frames per 2-s window), giving `FEAT_A = 256` and `MAX_LEN_A = 20` after pooling. An earlier rev of this spec specified WavLM-base+; that decision was reverted in favor of CPC's strict causality + direct MM-VAP lineage. The legacy `extract_wavlm_from_manifest.py` is retained on disk for reference but is no longer the active encoder pipeline.

> **Methodological framing.** The 4-way fusion taxonomy from the CSCI-535 practicum (unimodal, late, early, neural-concat) is reinterpreted dyadically: **Participant A and Participant B are the two channels to fuse, not two modalities within one person.** Each participant's feature stack — currently CPC (audio) + OpenFace (visual face), with SMPL-H body pose as a deferred candidate third modality (fetcher already staged at `scripts/extract_poc_smplh.py`) — is itself a pre-fused multimodal bundle; the architectural comparison happens at the cross-person integration step. The taxonomy spans increasing cross-person entanglement (unimodal → late → early → neural-concat), and supports independent per-modality and per-participant ablations: drop a modality entirely, or include a modality for only one of the two participants (e.g., audio-from-A + face-from-B). Concrete experiment grid in §11.7.

- **Primary metric:** per-class F1 vs. prediction-horizon τ (a *curve*, not a single point), over τ ∈ {100, 200, 400, 500, 800, 1600} ms. 500 ms is an anchored commensurability point with MM-VAP's single-τ hold/shift accuracy.
- **Label schema (3 trained classes + 3 exclusion-label classes):**
  - Trained on: **HOLD** (0), **YIELD** (1), **BACKCHANNEL** (2).
  - Excluded from training but *labeled* for analysis: **INTERRUPT** (3), **FAILED** (4), **LAPSE** (5). All six classes live in the same per-τ `labels_tau_XXXX.json`; the DataLoader filters to `{0, 1, 2}`. Full operational definitions in §11.
  - **Framing:** stride-based sampling, VAP-style. At each sample time `t`, perspective participant A is the current floor holder; input features are `[t − 2.0 s, t]`; labels are functions of the forward horizon `[t, t + τ]`. Input is τ-invariant; only labels vary with τ.
- **Required control:** permuted-dyad baseline (participant A's audio paired with a mismatched participant's video). Without this the experiment cannot distinguish a monadic "face predicts own turn-end" effect from a genuinely dyadic one. See §8.

The full lit review (7 items, ~40 citations) is at `docs/lit_review.md`. One-paragraph positioning: this replicates the *direction* of MM-VAP (Russell & Harte, ACL Findings 2025) on a **face-to-face** corpus (Seamless, not videoconferencing) with a **permuted-dyad control** MM-VAP does not run, and reports horizon as a **τ-swept curve** instead of silence-duration-stratified point accuracy.

---

## 2. Dataset

- **Corpus:** Meta Seamless Interaction (Meta FAIR 2025). Referred to as **"the paper"** throughout the spec — the 72-pp release version, not the arXiv abridgement. The PDF is gitignored / sits outside the repo; obtain from Meta's release page when needed.
- **Active scope:** the dyads listed in `manifests/poc_manifest.csv` (early-iteration subset). The full target corpus is `manifests/manifest.csv` (465 interactions).
- **Already staged locally:**
  - `turn_taking_analysis/subset/video/*.mp4` — 48 per-participant videos.
  - `turn_taking_analysis/subset/audio/*.wav` — 48 per-participant audio files.
  - `turn_taking_analysis/manifests/manifest.csv` — the full-project manifest (465 interactions), built by `scripts/build_manifest.py` from `poc_manifest.csv` (produced by `build_poc_manifest.py`) plus `both_annotated_interactions/` and `single_annotated_interactions/`. Consumers (downloaders, label builder, notebook) read `manifest.csv` only.
- **Seamless asset conventions (important):**
  - **Bboxes universal** — Seamless provides per-participant bboxes; skip face detection. Videos in `subset/video/` are already per-participant (one face each), so OpenFace `-f` single-face tracking is the right call.
  - **`has_imitator_movement`** is a per-participant **data-availability flag**, NOT a mimicry label.
  - **WhisperX transcripts** are bundled by Seamless. The paper's Appendix A.1.4 documents that 87% of interactions have ≥1 word with timestamp length >3σ from mean — this is the known BC-class label-noise source.

---

## 3. Pipeline overview

```
per-participant video (.mp4)  ──→ OpenFace FeatureExtraction  ──→ 48-col CSV / video   (30 fps)
per-participant audio (.wav)  ──→ splice_wavs.py (2-s windows) ──→ CPC encoder ──→ 256-dim @ 100 Hz native
                                                                                   (post-hoc ×10 pool → 10 Hz for OpenFace alignment)
                                                                       │
Seamless VAD + WhisperX ──→ generate_timestamps_by_turn.py ──→ per-turn spans
                                                                       │
                              per-window 3-class labels (HOLD/YIELD/BC); overlap-heavy windows excluded
                                                                       │
OpenFace windows + CPC windows + labels  ──→  multimodal_fusion_for_turns.ipynb
                                                 (GRU unimodal × 2, EarlyFusion, Late × 2, NeuralConcat)
                                                                       │
                                       per-class F1 @ τ ∈ {100, 200, 400, 800, 1600} ms
                                       + permuted-dyad control            (load-bearing)
                                       + (deferred) VAP-checkpoint zero-shot baseline  (revisit after main results)
```

---

## 4. What's done (prior session + lit-review session)

### 4.1 OpenFace build on Google Colab — WORKING

Built OpenFace 2.x `FeatureExtraction` on Colab (T4 runtime, 8 vCPUs on the lucky instance, Ubuntu 22.04 Jammy). Pain points resolved — relevant if a rebuild is needed:

1. **dlib version mismatch.** Ubuntu 22.04 apt ships `libdlib-dev` 19.10; OpenFace requires ≥19.13. Fix: remove `libdlib-dev` from apt install, build **dlib 19.24** from source (`http://dlib.net/files/dlib-19.24.tar.bz2`), `make install` to `/usr/local`.
2. **OpenCV 4 vs dlib `cv_image.h` `IplImage` error.** dlib 19.13's `cv_image.h` uses a pre-OpenCV-4 `IplImage` implicit conversion that OpenCV 4 dropped. Fixed by using dlib ≥19.22 (we use **19.24**); OpenFace's `find_package(dlib 19.13)` is a *minimum*, so 19.24 satisfies it.
3. **Colab `%%bash` magic does NOT inherit Python variables.** Any `${PATH}` must be re-declared inside the bash cell.
4. **`wget` without `-N`** creates `.1`-suffixed copies on rerun (harmless).
5. **Partial build tree after a failed cmake.** Must `rm -rf build && mkdir build` before re-running cmake; `mkdir -p` alone leaves a stale CMakeCache that re-triggers the original failure.

The exact working Colab cell is reproduced in Appendix B.

### 4.2 OpenFace feature extraction — RUN COMPLETED (or finishing) this session

- Videos copied Drive → local Colab disk for speed (6.6 GB).
- Parallelism: 7 workers on 8 vCPUs. Expected wall time ~75-90 min (empirical range 60-120 min).
- **Exact flag set** (from upstream handoff §2; do NOT alter without reason):
  ```
  -aus -gaze -pose
  -2Dfp false -3Dfp false -pdmparams false -simalign false -hogalign false -tracked false
  -nomask -multi_view 0 -q
  ```
- **Per-video output:**
  - `{file_id}.csv` — sliced to the **48 target columns** (Appendix A), column names stripped (OpenFace prepends a leading space).
  - `{file_id}.json` — sidecar with `n_frames`, `n_frames_tracker_success`, `tracker_success_fraction`, `openface_cmd`, `runtime_seconds`.
- **Threshold:** `tracker_success_fraction < 0.85` → flag as `LOW_TRACKING` per handoff §5 (per-file flag in the final writeup).
- **Resume/skip** added: if CSV + JSON already exist for a `file_id`, the worker skips and returns a cached sidecar result.
- **Output location:** on Google Drive at `${OUTPUT_DIR}` (configured in the Colab notebook's Cell 1). 15/48 currently present locally at `subset/openface/` — need to pull the remaining 33 from Drive. **TODO: complete download to `subset/openface/` (note: flat directory at `subset/openface/`, NOT `data/openface_features/` — the spec §10 layout was drafted before the local path was chosen; the existing flat sibling dirs (`subset/audio/`, `subset/vad/`, etc.) set the convention).**

### 4.3 CPC audio feature extraction — DONE

Two-stage pipeline:

1. **`scripts/splice_wavs.py`** slices each per-participant `.wav` in `subset/audio/` into uniform 2-s windows (`--window-len 2 --stride 0.5`), writing them under `subset/audio_sliced/<orig_stem>/<spliced_stem>.wav`. `<spliced_stem>` has the form `NNNN.NN-NNNN.NN_<orig_stem>` (zero-padded start/end seconds — sorts lex = temporal). Partial trailing windows are skipped so every spliced wav is exactly 2 s.

2. **`scripts/extract_cpc_from_manifest.py`** runs the pretrained CPC model (`facebookresearch/CPC_audio` torch.hub repo + `60k_epoch4-d0f474de.pt` checkpoint) over every spliced wav, writing features in a directory tree that mirrors `audio_sliced/` exactly: `subset/cpc/<orig_stem>/<spliced_stem>.npy` (shape `(200, 256)` float32 per 2-s window, 100 Hz native) plus a sibling `<spliced_stem>.json` sidecar with full provenance (source sample rate, resampled rate, model checkpoint, feature rate, frame count). Inference uses a torch DataLoader (`--batch-size 128 --num-workers 12 --writer-threads 8`) on Apple Silicon MPS; ~12-20 minutes wall-clock for the full corpus.

- All 453,252 `.npy` + 453,252 `.json` files present at `subset/cpc/` (930 subdirs, one per per-participant audio file). Verified shape uniformity, no NaN/Inf, no orphan/missing pairs.
- Rationale for CPC over WavLM/HuBERT/wav2vec 2.0: strict causality (load-bearing for prediction-horizon work — see §1 audio-encoder note) and direct lineage with MM-VAP (Russell & Harte 2025), which uses the same encoder.
- For OpenFace-rate alignment in early fusion, run `mean_pool_file(npy, save_to, pool_to=10.0)` post-hoc to produce a parallel 10 Hz corpus at e.g. `subset/cpc_10hz/`.

### 4.4 Literature review — DONE (parallel Claude Code session)

7-item review saved at `docs/lit_review.md` (~11k words). One-line summaries per item:

1. **Novelty.** MM-VAP (Russell & Harte, ACL Findings 2025) is the closest published prior art. Our differentiators: face-to-face corpus, permuted-dyad control, τ-curve.
2. **Horizon metric.** Skantze 2017 / Roddy 2018 anchor τ-curve framing; VAP's projection-window probability is the 2022+ native alternative. Our τ-curve needs explicit justification.
3. **Visual features.** MM-VAP's ablation: **AUs > head pose > gaze**. Kendon-1967 gaze-return survives only in qualified form.
4. **WhisperX short-utterance error.** Real (paper Appendix A.1.4) but acceptable at the project's current scale. CrisperWhisper (Wagner 2024) is the named mitigation path if BC-class F1 is structurally low.
5. **INTERRUPT class.** Rare (~5-15% of POC frames). Report per-class F1; do NOT collapse into macro.
6. **Audio-only baseline.** VAP is the 2024-26 default. HuBERT→GRU alone looks weak. Recommend adding a **VAP zero-shot checkpoint** run (Ekstedt & Skantze 2022 release) on the POC test set for credibility.
7. **Participant-disjoint splits.** Load-bearing; confirmed as validity risk in §8.

### 4.5 Helper scripts already available (in parent `../scripts/`)

Use these, do NOT reimplement. Brief signatures:

- **`generate_timestamps_by_turn.py <input_dir>`** — reads per-participant Seamless VAD JSONL files from each interaction under `<input_dir>/V*`, merges consecutive same-participant VAD segments into **turns** (a gap becomes a turn boundary only if the other participant spoke in it), flags overlap. Writes `{interaction}/interaction/timestamps_by_turn.json`. **Use this first** to get turn spans.
- **`download_annotated_interactions.py`, `download_single_annotated_interactions.py`** — fetch Seamless interaction assets from S3; consumed by `build_manifest.py` as part of the manifest-build pipeline.

### 4.6 Project-root scripts already present (in `turn_taking_analysis/scripts/`)

- **`build_manifest.py`** — builds the full-project `manifests/manifest.csv` (~467 interactions) by merging POC rows (from `poc_manifest.csv`) with every interaction under `both_annotated_interactions/` and `single_annotated_interactions/`. Enforces participant-disjoint splits via a connected-component / LPT bin-packing scheme. Splits in `manifest.csv` are authoritative — no re-splitting is needed downstream.
- **`build_poc_manifest.py`** — historical. Built the original 48-row `manifests/poc_manifest.csv` for the POC, with its own `verify_disjoint_participants()` split enforcer. Still lives on disk because `build_manifest.py` uses it as one of its inputs, but consumers no longer read `poc_manifest.csv` directly.
- **`download_audio_from_manifest.py`**, **`download_video_from_manifest.py`** — fetch per-participant audio (`.wav`) and video (`.mp4`) for the manifest's interactions into `subset/audio/` and `subset/video/`.
- **`download_vad_and_transcript_from_manifest.py`** — fetches Seamless VAD JSONLs and WhisperX transcript JSONLs into `subset/vad/` and `subset/transcript/` (transcript fetches may 403 on a small fraction of interactions).
- *(OpenFace extraction has no in-repo driver script — it ran on Google Colab against the OpenFace 2.x `FeatureExtraction` binary; see §4.1–4.2 and Appendix B for the build / run recipe. Outputs land in `subset/openface/`.)*
- **`splice_wavs.py`** — Companion to the CPC extractor: slices each `.wav` in `subset/audio/` into uniform 2-s windows under `subset/audio_sliced/`. Already run.
- **`extract_cpc_from_manifest.py`** — CPC extraction driver (the active audio encoder, see §4.3). Walks `subset/audio_sliced/` and writes per-window features to `subset/cpc/`. Already run.
- **`extract_wavlm_from_manifest.py`** — Legacy WavLM-base+ extraction driver. Superseded by `extract_cpc_from_manifest.py`; the WavLM output directory is no longer materialized. Kept on disk for reference.
- **`extract_poc_smplh.py`** — Fetches per-participant SMPL-H bundles for the POC manifest's dyads from Seamless's S3 (six per-feature `.npy` arrays packed into one compressed `.npz` per participant). Outputs land in `smplh_features_poc/`. Used to make body-pose available as a candidate third per-participant modality (deferred from the active CPC + OpenFace stack — see §1 methodological framing and §6).
- **`build_labeled_windows_from_manifest.py`** — Builds the per-window training/eval samples in `model_input/` from `manifests/manifest.csv` + per-modality features (CPC, OpenFace) + per-turn label spans. Implements the labeling logic in §11.
- **`multimodal_fusion_for_turns.ipynb`** — the practicum-derived notebook to minimally modify. See §6.

---

## 5. What's next (priority order for the new session)

1. **Complete OpenFace download from Google Drive → `subset/openface/`.** 15/48 participants currently local; pull the remaining 33 `.csv` + `.json` pairs. Cheap. Do first so downstream windowing has all 48 participants available.

2. **Read splits from `manifests/manifest.csv`.** The manifest's `split` column is authoritative (participant-disjoint, LPT-packed at a 70 / 15 / 15 target by `build_manifest.py`). Emit `data/splits.json` as `{split: [file_id, ...]}` by joining the manifest against the `file_id_a` / `file_id_b` columns — do NOT re-split.

3. **Per-window labels (stride-based, horizon-centric).** See §11 for full operational definitions. Workflow:
   - Reuse `../scripts/generate_timestamps_by_turn.py:merge_into_turns` (pure function — takes two VAD lists, returns sorted turns with `overlapping` flag). Do NOT depend on its on-disk directory layout.
   - For each dyad: load per-participant VAD (`subset/vad/{fid}.jsonl`), transcript (`subset/transcript/{fid}.jsonl`, may 403), OpenFace (`subset/openface/{fid}.csv`), and per-window CPC features from `subset/cpc/{fid}/<spliced_stem>.npy` (one `.npy` per 2-s window covered by `splice_wavs.py`'s stride).
   - For each participant A (treated as perspective / floor-holder), walk A's turns and emit sample points at `STRIDE_MS = 500` ms intervals inside each turn. Each sample point `t` resolves to one pre-extracted CPC window (`subset/cpc/{fid_A}/<start>-<end>_{fid_A}.npy` whose `[start, end]` covers `[t − 2, t]`) plus 2 s of OpenFace frames ending at `t`. Labels: one per τ (six labels per sample, stored across six per-τ JSON files).
   - Per-sample classification tree (precedence-ordered; full logic in §11.3):
     - Compute B's VAD segments + transcript words clipped to horizon `[t, t+τ]`.
     - Check **substantive-B** (OR-gate: clipped duration ≥ 700 ms, OR ≥ 3 words, OR contains non-BC vocabulary).
     - If substantive-B present and A ceases within `b_substantive.start + 300 ms` → **YIELD**.
     - If substantive-B present and A doesn't cease within 300 ms → **INTERRUPT** (B still voiced at horizon end) or **FAILED** (B ends in horizon, A voiced at horizon end).
     - Else if BC-qualifying-B present AND A voiced at `t+τ` → **BACKCHANNEL**.
     - Else if A voiced at `t+τ` → **HOLD**.
     - Else → **LAPSE**.
   - All six classes are written to `subset/windows/labels/labels_tau_XXXX.json`. The notebook's loader filters to `{HOLD, YIELD, BACKCHANNEL}`; INTERRUPT/FAILED/LAPSE are retained for per-τ exclusion-stat reporting.
   - **Class imbalance: no data-level rebalancing.** All splits are emitted at natural rate. Per-class imbalance is handled at training time via inverse-frequency class weights in `nn.CrossEntropyLoss`. Rationale: at 467 dyads, BC has enough examples (hundreds) that loss-level weighting preserves more signal than HOLD-subsampling, and avoids a train/test prior mismatch. An earlier POC-scale version (24 dyads, ≤50 BCs) HOLD-subsampled train/val to target 45/40/15 — dropped at full-project scale. Manifest records natural-rate counts per (τ, split).

4. **Modify `scripts/multimodal_fusion_for_turns.ipynb`.** Minimum edits in §6.2 (`NUM_CLASSES=3`, label dict, feature paths, sequence lengths, τ-sweep evaluator, permuted-dyad evaluator).

5. **Evaluate across τ grid and build the horizon curve.** Same trained model, different label/window alignments per τ. Plot per-class F1 vs. τ. Optionally also report hold/shift accuracy at τ = 500 ms for MM-VAP commensurability.

6. **Permuted-dyad control (load-bearing).** Same trained model; at test time shuffle audio-video pairings within the test set so each example has a participant-mismatched audio stream. If performance is unchanged, the dyadic-coordination claim fails (scientifically interesting either way, but report framing must change).

7. **(Deferred) VAP zero-shot baseline.** Run Ekstedt & Skantze 2022 VAP checkpoint on the POC test set at τ = 500 ms. ~2 hours of setup (clone repo, interleave per-participant wavs to stereo, inference, extract hold/shift). Revisit after tasks 1–6 land; include if compute budget allows.

---

## 6. The notebook — `scripts/multimodal_fusion_for_turns.ipynb`

### 6.1 What it does today (inherited from CSCI-535 practicum, CREMA-D)

- **Task:** 6-class emotion recognition (ANG / DIS / FEA / HAP / NEU / SAD) on CREMA-D utterances.
- **Features (as the notebook ships today, before our edits):**
  - OpenFace: **20-dim** (17 AU intensity + 3 head pose). Pre-extracted, one `.npy` per utterance.
  - HuBERT: **768-dim** (HuBERT-base penultimate layer, ×5 temporal pool → ~10 Hz). Pre-extracted, one `.npy` per utterance.
  - **For this project: swap the HuBERT features for CPC features (256-dim @ 100 Hz native; pooled ×10 → 10 Hz to match OpenFace's post-pool rate).** Architectural change is minimal: `FEAT_A` shrinks 768 → 256 (or 1536 → 512 for two-participant concat slots, see §11.7 ablation table). Features are at `subset/cpc/<orig_stem>/<spliced_stem>.npy`, one `.npy` per 2-s spliced window, mirroring the `audio_sliced/` layout.
- **Sequence lengths:** `MAX_LEN_V = 16`, `MAX_LEN_A = 40`, `MAX_LEN_EARLY = 40` (merged). Zero-pad / clip.
- **Splits:** `subsampled/openface/{train,val,test}/*.npy` and `subsampled/hubert/{train,val,test}/*.npy`. Labels extracted from filename via `fname.split('_')[2]`.
- **Architectures (PyTorch):**
  - `GRUClassifier` — unimodal. `nn.GRU → last h → FC → ReLU → Dropout(0.3) → FC`.
  - `EarlyFusionGRU` — concat features per-timestep, 3-layer stacked GRU, hidden=128.
  - `NeuralConcatFusion` — dual-branch GRU (one per modality), concat hidden states, shared FC head.
  - `late_fusion_predict` — probability-level weighted average (grid-searched w ∈ [0,1] on val).
- **Training:** Adam, LR=1e-3, BS=32, EPOCHS=30, early stopping patience=4 on val accuracy.
- **Reported results (on the notebook's CREMA-D run):** Early Fusion ~74%, Neural Concat ~73%, Late Fusion (learned w) ~70-72%. Unimodal HuBERT > Unimodal OpenFace.

### 6.2 Minimum edits to repurpose for turn-taking

**Keep untouched:** training loop (`run_epoch`, `train_model`, `evaluate`), all four fusion architectures, `nn.CrossEntropyLoss`, optimizer/LR/dropout, plotting helpers. The practicum's hyperparameters are reasonable defaults.

**Deviation from the practicum — early-stopping metric.** The practicum early-stopped on val accuracy (§6.1). This POC early-stops on **val macro-F1** instead (patience=4). Rationale: with natural-rate class imbalance (HOLD dominates, BC ~15%) and inverse-frequency-weighted cross-entropy, val accuracy is a misleading signal — it rewards correctly predicting the majority class at the expense of the minority classes that the weighted loss is actively trying to lift. Macro-F1 is both the reported metric and the quantity the class weights are implicitly optimizing for. Implemented in `train_model` (notebook Cell 14) via `sklearn.metrics.f1_score(..., average='macro')` on the val split's predictions each epoch.

**Change in priority order:**

(a) **Labels.** Replace `labels_dict = {'ANG':0, ...}` with `{'HOLD':0, 'YIELD':1, 'BACKCHANNEL':2}`. Set `NUM_CLASSES = 3`. Replace `extract_label(fname)` / the `f.split('_')[2]` patterns in `SequenceDataset` / `BimodalDataset` / `MergedDataset.__init__` with a lookup against a per-window label file (e.g., `data/windows/labels.json` — dict mapping window filename → class int).

(b) **Visual feature dim `FEAT_V`.** Two options:
  - **(i) Keep `FEAT_V = 20`** by subsetting OpenFace columns to `AU01_r..AU45_r (17) + pose_Rx/Ry/Rz (3)`, matching the notebook's existing layout. **Minimum change. Recommended for the POC.**
  - **(ii) Use `FEAT_V = 48`** (all columns). Richer features, but requires updating every tensor-shape-dependent assertion and doubling memory. Defer.
  - The lit review (item 3) flags feature-importance ranking **AUs > head pose > gaze** — option (i) already includes the top two.

(c) **Sequence lengths — using 2 s context.** At 30 fps video (Seamless paper §3.4) → `MAX_LEN_V = 60`; at 10 Hz pooled CPC → `MAX_LEN_A = 20`; `MAX_LEN_EARLY = 20` for early fusion (downsample OpenFace by mean-pooling every 3 frames into a 10 Hz track, matching the pooled CPC rate). Note: CPC is extracted at native 100 Hz (`MAX_LEN_A = 200` if used unpooled); the 10 Hz pool is applied post-hoc via `mean_pool_file` to align with OpenFace and to keep `MAX_LEN_A` from blowing up the early-fusion concat. Document the chosen rate (native 100 Hz vs. pooled 10 Hz) in the per-window sidecar. *(An earlier rev used WavLM at 10 Hz natively; with the encoder switch to CPC the source rate is 100 Hz and an explicit pool step is required for OpenFace alignment. An even-earlier draft said 25 fps / `MAX_LEN_V = 50`; that was caught when the dry-run showed OF duration exceeding the audio rate by 1.20× — the 25:30 ratio.)*

(d) **Feature paths.** Point `FEAT_PATH_V` at per-window OpenFace `.npy` files and `FEAT_PATH_A` at the pooled-to-10-Hz per-window CPC `.npy` files (e.g. `model_input/cpc_10hz/{speaker,listener}/{train,val,test}/`). File names must match across modalities so `BimodalDataset` pairs them correctly.

(e) **τ-sweep evaluation.** After training *once*, call `evaluate()` five times with different test-set variants (one per τ). Each variant has the same audio/video windows but labels derived from the boundary shifted by τ. Compile into a per-class F1 × τ table and plot. Do NOT retrain per τ — that defeats the horizon-curve framing.

(f) **Permuted-dyad control.** New wrapper around `evaluate()`: shuffle audio-to-video filename mapping within the test set so `(x_v, x_a)` tuples come from different participants. Re-run the trained bimodal model. Compare to the non-permuted baseline.

**Do NOT** change the train/val/test split *shape* in the notebook — keep the `splits[split]` dict pattern. Just point it at the new `.npy` file lists.

### 6.3 Feature export format (per-window `.npy` contract)

Per labeled example:
- Visual: `(MAX_LEN_V, FEAT_V)` float32, zero-padded if shorter.
- Audio: `(MAX_LEN_A, FEAT_A)` float32, zero-padded if shorter.
- Both modalities share a filename (paired by `BimodalDataset`).
- Per-window label lives in a sidecar file (`labels.json`) — not in the filename, so we can iterate on label schema without re-exporting.

Suggested naming: `{dyad_id}_{participant_id}_{window_start_ms}_{tau_ms}.npy`.

---

## 7. Canonical references / conventions

- **"the paper"** → the 72-pp Meta Seamless Interaction release PDF (Meta FAIR 2025), not the arXiv abridgement. Not in the repo; obtain separately.
- **Compute environment:** mixed — OpenFace ran on Colab Pro (T4); CPC extraction ran on Apple Silicon M5 Max (32-core GPU, 36 GB unified memory, MPS); other steps as noted per script.
- **Bboxes universal:** skip face detection. Videos already per-participant.
- **`has_imitator_movement`:** data-availability flag, not a mimicry label.

---

## 8. Validity risks

**8.1 Whisper timestamp drift vs. horizon SNR.** Backchannel labels depend on WhisperX word timestamps; paper Appendix A.1.4 reports 87% of interactions have ≥1 word >3σ in timestamp length. Mitigation: (i) hand-annotate a 30-60 s slice of one dyad in Praat as a sanity check (cheap, ~30-60 min in Praat), and report the empirical short-utterance error; (ii) report BC-class F1 separately so label-noise impact is visible, not hidden.

**8.2 Permuted-dyad control is load-bearing for the dyadic claim.** Without it, a positive face-helps-TT result could be monadic (either participant's own face predicts their own turn ends) rather than dyadic (partner's face informs transitions). The control is not optional — it is the *defining differentiator* from MM-VAP.

**8.3 Seamless splits not strictly participant-disjoint by default.** If a participant appears in both train and test sets, the model can learn participant identity → class-label shortcuts. Enforce disjoint at the participant level, not just the dyad level. Double-check before reporting numbers.

---

## 9. Open questions — RESOLVED at 2026-04-21 session start

1. **Audio encoder — CLOSED.** CPC (Rivière 2020, causal-RNN aggregator output, 256-dim @ 100 Hz native; `60k_epoch4-d0f474de.pt`). Pooled ×10 → 10 Hz post-hoc for OpenFace alignment via `mean_pool_file`. Already extracted: 453,252 per-window `.npy` files at `subset/cpc/`. Rationale: strict causality (load-bearing for prediction-horizon work — see §1 audio-encoder note) and direct lineage with MM-VAP. An earlier rev of this spec selected WavLM-base+; that decision was reverted in favor of CPC.

2. **Window length — CLOSED.** 2 s context. `MAX_LEN_V = 60` @ 30 fps OpenFace, `MAX_LEN_A = 20` @ 10 Hz pooled CPC (or `MAX_LEN_A = 200` if used at native 100 Hz; `mean_pool_file` produces the 10 Hz parallel corpus).

3. **INTERRUPT class — CLOSED.** Dropped from POC label schema. Overlap-heavy transitions (`overlapping=True` span ≥ 500 ms and not a lexical backchannel) are *excluded* from the labeled window set rather than classified. Label schema is 3-class {HOLD, YIELD, BACKCHANNEL}. `NUM_CLASSES = 3`. Rationale: ≤50 INTERRUPT examples in the 4-dyad test set puts per-class F1 in the noise band; lit review Item 5 Flag #1.

4. **Feature set width — CLOSED.** `FEAT_V = 20` (17 AU_r + 3 pose_R). Minimum-change path; includes the top-ranked features per lit review Item 3 (AUs > head pose > gaze).

5. **VAP zero-shot baseline — DEFERRED.** Revisit after tasks 1–6 of §5 land. ~2-hour setup (clone repo, stereo-interleave wavs, inference). Strong "nice-to-have" for report credibility (lit review M2) but not blocking the main result.

---

## 10. Project file layout

```
turn_taking_analysis/                          ← this project root
├── docs/
│   ├── spec.md                               ← this document
│   └── lit_review.md                         ← 7-item lit review, ~11k words
├── manifests/
│   ├── manifest.csv                          ← full-project manifest (~467 interactions); consumer source of truth
│   └── poc_manifest.csv                      ← historical 48-row POC; now only an input to build_manifest.py
├── scripts/
│   ├── build_manifest.py                     ← builds manifest.csv; participant-disjoint LPT-packed splits
│   ├── build_poc_manifest.py                 ← historical POC builder; still feeds build_manifest.py
│   ├── download_audio_from_manifest.py
│   ├── download_video_from_manifest.py
│   ├── download_vad_and_transcript_from_manifest.py    ← VAD + WhisperX transcript fetcher
│   ├── splice_wavs.py                        ← slices subset/audio/ into 2-s windows for CPC (DONE)
│   ├── extract_cpc_from_manifest.py          ← CPC driver (DONE; active audio encoder)
│   ├── extract_wavlm_from_manifest.py        ← legacy WavLM driver (superseded by CPC)
│   ├── extract_poc_smplh.py                  ← POC SMPL-H fetcher (deferred candidate body-pose modality)
│   ├── build_labeled_windows_from_manifest.py ← per-window sample builder (implements §11)
│   └── multimodal_fusion_for_turns.ipynb     ← the notebook to minimally modify
├── subset/
│   ├── audio/      V*_S*_I*_P*.wav           ← 48 per-participant audio files
│   ├── video/      V*_S*_I*_P*.mp4           ← 48 per-participant videos
│   ├── audio_sliced/ <orig_stem>/<spliced_stem>.wav    ← 453,252 per-window 2-s audio slices (DONE; CPC input)
│   ├── cpc/        <orig_stem>/<spliced_stem>.{npy,json} ← 453,252/453,252 per-window CPC features (DONE)
│   ├── openface/   V*_S*_I*_P*.{csv,json}    ← 15/48 locally; 33 more to pull from Drive
│   ├── vad/        V*_S*_I*_P*.jsonl         ← 48 Silero VAD files (downloader ready)
│   └── transcript/ V*_S*_I*_P*.jsonl         ← up to 48 WhisperX transcripts (optional, some may 403)
├── model_input/                              ← TO CREATE via build_labeled_windows_from_manifest.py
│   ├── openface/{speaker,listener}/{train,val,test}/{basename}.npy
│   ├── cpc/     {speaker,listener}/{train,val,test}/{basename}.npy   ← (10 Hz pooled, FEAT_A=256)
│   ├── labels/labels_tau_XXXX.json           ← one file per τ
│   └── manifest.json                         ← full provenance
│   # Dyadic-concat ("both") configs are synthesized in the notebook via
│   # ConcatFEATDataset (spec §11.8) — no pre-materialized dir needed.
└── results/                                  ← TO CREATE: τ-curves, confusion matrices, permuted-dyad control

# Parent (legacy coupling — turn-taking depends on a small set of parent-level scripts; not aspirational):
../scripts/
├── generate_timestamps_by_turn.py            ← per-turn span builder used in §5 step 3
├── download_annotated_interactions.py        ← consumed by build_manifest.py
└── download_single_annotated_interactions.py ← consumed by build_manifest.py
```

---

## 11. Operational label definitions (authoritative)

These are the verbatim definitions driving `scripts/build_labeled_windows_from_manifest.py`. Every downstream artifact (labels JSONs, notebook edits, analysis stats) derives from this section. Changes here must be propagated to the script's top-of-file constants + inline comments.

### 11.1 Framing

- **Sample point `t`**: a timestamp in seconds. Sampled every `STRIDE_MS` within a perspective participant A's turn.
- **Perspective participant A**: the floor holder at `t`. Must be voiced at `t` (prerequisite). All feature slicing is from A's streams (A's OpenFace, A's CPC).
- **Input context** (τ-invariant): features from `[t − WINDOW_S, t]`. 2 s of the past.
- **Horizon** (τ-dependent): `[t, t + τ]`. The forward interval whose content determines the label.
- **Clipping to horizon**: for a B-segment `[b_start, b_end]`, clipped form is `[max(b_start, t), min(b_end, t + τ)]`. The substantive / BC-qualifying tests operate on the clipped portion, not the underlying full-segment.
- **Per-τ labels**: the same sample `t` yields potentially different labels at different τ. Six label JSON files, one per τ.

### 11.2 Class definitions (verbatim from 2026-04-21 decisions)

**HOLD** (class `0`). A speaks at frame `t` and A is still speaking at frame `t+τ` with no intervening onset of substantive speech by B. Operationally: A's VAD = voiced at `t` and at `t+τ`, and B emits no word (or only a backchannel token — see BACKCHANNEL below) during `[t, t+τ]`.

**YIELD** (class `1`). A is speaking at `t`, but by `t+τ` the floor has transferred to B. Operationally: the first word boundary in `[t, t+τ]` where B begins a **substantive** utterance (substantive = duration ≥ 700 ms, OR ≥ 3 words, OR not in the backchannel vocabulary — any one leg), and A ceases speaking by the start of that B-utterance (allowing ≤ 300 ms overlap for natural turn-competitive onset, i.e. `a_vad_end ≤ b_substantive_utterance.start + 300 ms`).

**BACKCHANNEL** (class `2`). A keeps the floor; B produces a short vocalization during `[t, t+τ]` that does not constitute a turn. Operationally: B emits speech within `[t, t+τ]`, the utterance is ≤ 500 ms AND (word-count ≤ 2 OR lexical match to a curated filler vocabulary), AND A continues speaking afterward (A voiced at `t+τ`). Nonverbal backchannels (nods, smiles) are deliberately excluded.

### 11.3 Exclusion sub-types (classes 3–5)

Not used in training. Recorded in the same per-τ labels JSON for post-hoc stats.

**INTERRUPT** (class `3`). Substantive-B fires inside horizon AND A does not cease within 300 ms of B's substantive start AND B is still voiced at `t+τ`. Floor contest unresolved inside horizon, or B winning the contest at horizon end.

**FAILED** (class `4`). Substantive-B fires AND A does not cease within 300 ms AND B's substantive utterance ends within `[t, t+τ]` AND A is voiced at `t+τ`. B tried, A didn't yield, B cedes. A keeps the floor.

**LAPSE** (class `5`). Reached when A is not voiced at `t+τ` and no substantive-B utterance exists in the horizon. Covers four fall-through sub-cases: (F1) A trails into silence, B never speaks; (F2) A trails off and B emits only backchannel tokens; (F3) rare "medium-BC" 500–700 ms all-BC utterance with A trailing off; (F4) ambiguous B event not matching substantive or BC-qualifying with A trailing off. All four share: `not a_is_voiced_at(t+τ) AND no substantive-B`.

### 11.4 Decision tree (precedence-ordered)

```
Prerequisite: a_is_voiced_at(t). If False → sample not emitted at all.

1. Compute b_utterances_in_horizon(t, τ):
   clipped B-VAD segments, each attached with the subset of B transcript
   words whose start ∈ [t, t+τ].

2. Find first_substantive_b = first u in b_utterances where is_substantive(u).

3. If first_substantive_b is not None:
     If a_vad_end_in_current_turn ≤ first_substantive_b.start + 0.300:
         → YIELD
     Else:
         If b_is_voiced_at(t + τ):
             → INTERRUPT
         Else:
             → FAILED

4. Else if any u in b_utterances satisfies is_bc_qualifying(u)
        and a_is_voiced_at(t + τ):
     → BACKCHANNEL

5. Else if a_is_voiced_at(t + τ):
     → HOLD

6. Else:
     → LAPSE
```

### 11.5 Module-level constants (authoritative values)

| Constant | Value | Unit | Role |
|---|---|---|---|
| `STRIDE_MS` | 500 | ms | Sample point spacing within each A-turn |
| `WINDOW_S` | 2.0 | s | Input context length (past) |
| `TRAIN_TAU_MS` | 400 | ms | Training / val label τ |
| `TAU_GRID_MS` | `[100, 200, 400, 500, 800, 1600]` | ms | Evaluation τ grid |
| `SUBSTANTIVE_DURATION_MS` | 700 | ms | Substantive-B OR-gate leg 1 |
| `SUBSTANTIVE_WORD_COUNT` | 3 | words | Substantive-B OR-gate leg 2 |
| `BC_MAX_DURATION_MS` | 500 | ms | BC-qualifying upper duration |
| `BC_MAX_WORD_COUNT` | 2 | words | BC-qualifying word-count leg |
| `A_YIELD_TOLERANCE_MS` | 300 | ms | Max A-overlap-past-B-start for clean YIELD |

All overridable via CLI flags. Rationale for each value is documented inline in the script's constant declarations.

### 11.6 Backchannel vocabulary (lit-review-derived)

```python
BC_LEXICON = {
    # non-lexical acknowledgement (Clancy 1996, Ward 2000, Lala 2017, Tolins 2014)
    "mm", "hm", "hmm", "mhm", "mmhmm",
    "uhhuh", "huh",
    "ah", "aha", "oh",
    # short affirmatives (Gravano 2011, Ward 2000, Jurafsky SWBD-DAMSL 1997)
    "yeah", "yep", "yup", "ok", "okay", "sure",
    "right", "true", "exactly", "totally",
    # short assessments (Ruede 2017, Lala 2017)
    "wow", "really", "gotcha",
}
```

**Normalization pipeline** (applied before lookup): lowercase → strip leading/trailing punctuation → collapse internal whitespace and hyphens. Result: `"Mm-hmm!"`, `"mm hmm"`, `"mmhmm"` all → `"mmhmm"`; `"Uh-huh"`, `"uh huh"`, `"uhhuh"` all → `"uhhuh"`.

### 11.7 Output layout (ablation-friendly)

Each emitted sample is a `(speaker, listener)` pair at one time point. The **speaker** role is the perspective participant (current floor holder, subject of the turn prediction). The **listener** is the partner. Only the two role variants are materialized on disk. Dyadic-concat configurations (MM-VAP-/Hisada-2024-style, lit review items 1.1, 1.4) are synthesized on-the-fly in the notebook via a `ConcatFEATDataset` wrapper — see §11.8.

```
model_input/
  openface/
    speaker/{train,val,test}/{basename}.npy          # (60, 20)   — speaker's face
    listener/{train,val,test}/{basename}.npy         # (60, 20)   — listener's face
  cpc/
    speaker/{train,val,test}/{basename}.npy          # (20, 256) at 10 Hz pooled — speaker's audio
    listener/{train,val,test}/{basename}.npy         # (20, 256) at 10 Hz pooled — listener's audio
  labels/
    labels_tau_0100.json                              # {basename: class_int 0..5}
    labels_tau_0200.json
    labels_tau_0400.json                              # ← training/val label file
    labels_tau_0500.json
    labels_tau_0800.json
    labels_tau_1600.json
  manifest.json                                       # config + per-(τ, split) class counts
```

**Basename** = `{speaker_file_id}_t_{time_ms:07d}`. The speaker file_id is the perspective tag; a sample at the same `t` where the other participant is the floor holder produces a different basename with that participant's file_id. Both orientations coexist in every split.

**Pairing by filename** holds across all four `(modality, role)` directories — identical basenames reference the same sample point. All rows use the same two Dataset classes — `SequenceDataset` and `BimodalDataset` (notebook Cell 10) — parameterized by a `path_list` of length 1 or 2 per modality slot. Length-1 slots load a single `.npy` per sample; length-2 slots load from both paths (convention: `[speaker, listener]`) and concatenate along the feature axis at `__getitem__` time. Rows 1–6 use length-1 slots; rows 7–10 and row 9′ use length-2 slots for the concat modality. See §11.8 for the contract.

**Notebook-side ablation recipes.** `MAX_LEN_V=60`, `MAX_LEN_A=20`, `NUM_CLASSES=3` are fixed across rows. `FEAT_V` / `FEAT_A` toggle only when a slot uses a length-2 path_list (the sum of per-path feature dims).

| # | Experiment | `FEAT_PATH_V` | `FEAT_V` | `FEAT_PATH_A` | `FEAT_A` | Model(s) | Anchor |
|---|---|---|---|---|---|---|---|
| 1 | Dyadic cross-pair *(primary)* | `openface/listener` | 20 | `cpc/speaker` | 256 | `EarlyFusionGRU` + `NeuralConcatFusion` + `late_fusion_predict` | our framing |
| 2 | Monadic baseline | `openface/speaker` | 20 | `cpc/speaker` | 256 | `EarlyFusionGRU` + `NeuralConcatFusion` + `late_fusion_predict` | control |
| 3 | Unimodal audio | — | — | `cpc/speaker` | 256 | `GRUClassifier` *(audio)* | VAP baseline (Item 6) |
| 4 | Unimodal listener-face | `openface/listener` | 20 | — | — | `GRUClassifier` *(visual)* | Kendrick 2023 (Item 3.3) |
| 5 | Unimodal speaker-face | `openface/speaker` | 20 | — | — | `GRUClassifier` *(visual)* | Nota 2021 (Item 3.4) |
| 6 | Permuted-dyad control (§8.2) | `openface/listener` *(shuffled)* | 20 | `cpc/speaker` | 256 | *(test-time re-eval of row 1's trained weights)* | dyadic-coordination test |
| 7 | Both-face + speaker-audio | concat(`openface/speaker`, `openface/listener`) | 40 | `cpc/speaker` | 256 | `EarlyFusionGRU` + `NeuralConcatFusion` + `late_fusion_predict` | MM-VAP visual ablation |
| 8 | Listener-face + both-audio | `openface/listener` | 20 | concat(`cpc/speaker`, `cpc/listener`) | 512 | `EarlyFusionGRU` + `NeuralConcatFusion` + `late_fusion_predict` | VAP-style stereo audio |
| 9 | Full dyadic *(MM-VAP replication)* | concat(`openface/speaker`, `openface/listener`) | 40 | concat(`cpc/speaker`, `cpc/listener`) | 512 | `EarlyFusionGRU` + `NeuralConcatFusion` + `late_fusion_predict` | Russell & Harte 2025 (Item 1.1) |
| 10 | Speaker-face + both-audio | `openface/speaker` | 20 | concat(`cpc/speaker`, `cpc/listener`) | 512 | `EarlyFusionGRU` + `NeuralConcatFusion` + `late_fusion_predict` | own-face + VAP-style audio |
| 9′ | Permuted-dyad control on row 9 | concat(`openface/speaker`, `openface/listener` *(listener half shuffled)*) | 40 | concat(`cpc/speaker`, `cpc/listener` *(listener half shuffled)*) | 512 | *(test-time re-eval of row 9's trained weights)* | dyadic-coordination test for full-dyadic replication |

**Note on `late_fusion_predict`**: it consumes two pre-trained unimodal `GRUClassifier` models and averages their softmax outputs with a grid-searched weight on val. For bimodal rows that include it, the matching unimodal `GRUClassifier`s for the same `FEAT_PATH_V` + `FEAT_PATH_A` must be trained first. Rows 3–5 cover the standard unimodal cases; rows 7–10 additionally require unimodal `GRUClassifier`s trained on their concat slot as input (trivially — same class, larger `input_size`).

**Permutation (row 6).** Done at eval time by shuffling the listener-face filename assignment within the test split, keeping the speaker-audio and labels attached to each sample. A performance drop vs. row 1 → dyadic-coordination effect is real.

**Permutation (row 9′) — promoted from parenthetical §11.8 mention to a formal ablation row.** Done at eval time by shuffling the listener half of BOTH the visual concat slot AND the audio concat slot using a **shared permutation `σ`** — so the "fake listener" is coherent across face and voice (i.e., the face and audio of the permuted-in listener come from the same randomly-mismatched dyad). Primary (speaker) halves are left in their true pairing; only the secondary (listener) halves are shuffled. A performance drop vs. row 9 → the MM-VAP-style full-dyadic gain is genuinely cross-participant coupling rather than listener-stream side-information leakage. Controls for a distinct failure mode than row 6 (row 6 tests whether the listener face helps at all; row 9′ tests whether the full-dyadic coupling is dyadic).

Paired by basename across modalities and roles. Class integers `{0:HOLD, 1:YIELD, 2:BACKCHANNEL, 3:INTERRUPT, 4:FAILED, 5:LAPSE}`. Manifest.json records all constants, per-(τ, split) class counts (natural + balanced), excluded-sample counts, and per-dyad flags (missing transcript, OpenFace low-tracking, duration mismatches).

### 11.8 Dyadic-concat via length-2 `path_list` (notebook Cell 10)

Dyadic-concat features (rows 7–10, row 9′) are synthesized on the fly by the same `SequenceDataset` / `BimodalDataset` classes that handle rows 1–6 — there is **no separate `ConcatFEATDataset` class**. An earlier spec draft proposed a standalone wrapper; during notebook implementation the concat logic was inlined into the two existing Dataset classes to eliminate a code path and keep the loader factory uniform. Behavior is identical to what the proposed wrapper would have produced; only the class surface differs.

**Contract.** Each modality slot accepts a `path_list` of 1 or 2 directory paths:

- **Length-1 slot** (rows 1–6, and the non-concat slot of rows 7/8/10): `np.load(path_list[0] / basename)` → pad/clip to `target_size` → return.
- **Length-2 slot** (rows 7–10 concat side, row 9′ both sides): `np.load` from `path_list[0]` (primary) and `path_list[1]` (secondary), pad each to `target_size` with per-path feature dim `feat_total // 2`, concatenate along the feature axis → return a tensor of shape `(target_size, feat_total)`.

**Convention.** `path_list[0]` = **speaker** stream; `path_list[1]` = **listener** stream. Feature axis order is `[speaker-dims ; listener-dims]`, matching MM-VAP's speaker-first concat convention (Russell & Harte 2025).

**Effective feature dim.** For a length-2 slot, `feat_total` (e.g. `feat_v = 40` for row 9's visual concat, `feat_a = 1536` for row 9's audio concat) equals per-path-dim + per-path-dim. The GRU constructors (`GRUClassifier`, `EarlyFusionGRU`, `NeuralConcatFusion`) take `feat_v` / `feat_a` as arguments and adapt automatically — no architectural branching on slot-length is needed.

**Permuted-dyad control against a concat slot (row 9′, formalized in §11.7).** The `BimodalDataset` constructor accepts `permute_listener_v` and `permute_listener_a` flags. When set, a single shared permutation `σ` is drawn once per Dataset instance and applied to the **secondary (listener) file list of each length-2 slot** — primary (speaker) filenames stay aligned with the sample identity / label. For length-1 slots whose path IS a listener path (i.e., row 6's visual slot), the same flag shuffles the primary file list instead, since there is no secondary. Permutation flags are applied **only to the test loader** by `make_loaders_for_experiment(..., test_only_permute=True)` so the model is trained on real pairings.

Implementation lives in notebook Cell 10. A separate standalone wrapper class is not materialized on disk or in code; references to "ConcatFEATDataset" in older drafts of this spec or in the docstring of `scripts/build_labeled_windows_from_manifest.py` refer to the same capability that the length-2 `path_list` now provides.

---

## Appendix A: OpenFace 48-column schema (exact, after `col.strip()`)

```
QA (5):      frame, face_id, timestamp, confidence, success
Pose (6):    pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz
Gaze (2):    gaze_angle_x, gaze_angle_y
AU_r (17):   AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r,
             AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r,
             AU25_r, AU26_r, AU45_r
AU_c (18):   AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c,
             AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c,
             AU25_c, AU26_c, AU28_c, AU45_c
```

Note AU_c has 18 entries (includes `AU28_c`, which has no intensity counterpart in OpenFace). AU_r has 17. Total: 5 + 6 + 2 + 17 + 18 = 48.

---

## Appendix B: OpenFace Colab build recipe (known-working as of 2026-04-21)

Against Ubuntu 22.04 Jammy + OpenCV 4.5.4 + Python 3.10-3.12. Entire build idempotent; rerun is safe.

```bash
%%bash
set -e

OPENFACE_INSTALL="/content/drive/<...>/openface_build"
FEATURE_BIN="${OPENFACE_INSTALL}/build/bin/FeatureExtraction"

# Idempotent guard — skip if binary already built
if [ -f "${FEATURE_BIN}" ] && [ -s "${FEATURE_BIN}" ]; then
    echo "Binary exists, skipping build."
    exit 0
fi

# apt deps MINUS libdlib-dev (apt version 19.10 is too old for OpenFace)
apt-get update -qq
apt-get install -y libopenblas-dev liblapack-dev libboost-all-dev libopencv-dev cmake g++-9

# Clone OpenFace to Drive (idempotent)
mkdir -p "$(dirname "${OPENFACE_INSTALL}")"
[ -d "${OPENFACE_INSTALL}" ] || \
    git clone https://github.com/TadasBaltrusaitis/OpenFace.git "${OPENFACE_INSTALL}"

# Build dlib 19.24 from source (≥19.22 is required for OpenCV-4-compat cv_image.h)
cd /content
wget -q http://dlib.net/files/dlib-19.24.tar.bz2
tar xf dlib-19.24.tar.bz2
cd dlib-19.24 && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- -j$(nproc)
make install
ldconfig

# Download OpenFace models (~430 MB, to Drive — persisted across runtimes)
cd "${OPENFACE_INSTALL}"
bash download_models.sh

# Build OpenFace. MUST `rm -rf build` — stale CMakeCache from a failed prior run
# will re-trigger the original failure even if deps have been fixed.
rm -rf build && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)

# Verify
./bin/FeatureExtraction -help 2>&1 | head -3 || true
ls -lh bin/FeatureExtraction
```

---

## Appendix C: OpenFace FeatureExtraction run config (from this session)

Exact `process_one` logic used, condensed:

```python
OPENFACE_ARGS = [
    "-aus", "-gaze", "-pose",
    "-2Dfp", "false", "-3Dfp", "false",
    "-pdmparams", "false", "-simalign", "false",
    "-hogalign", "false", "-tracked", "false",
    "-nomask", "-multi_view", "0", "-q",
]

cmd = [FEATURE_EXTRACTION, "-f", video_path, "-out_dir", raw_out] + OPENFACE_ARGS
# cwd=WORKING_DIR (= OpenFace build/bin dir, so the binary finds its model subdirs)

# After OpenFace writes {file_id}.csv:
df = pd.read_csv(raw_csv)
df.columns = [c.strip() for c in df.columns]    # OpenFace prepends a space
sliced = df[TARGET_COLS]                        # the 48 columns in Appendix A

# Sidecar
sidecar = {
    "file_id": file_id,
    "n_frames": len(sliced),
    "n_frames_tracker_success": int(sliced["success"].sum()),
    "tracker_success_fraction": round(n_success / n_frames, 4),
    "openface_cmd": " ".join(cmd),
    "runtime_seconds": round(elapsed, 2),
}

# Resume/skip: before running, check {file_id}.csv + .json exist and skip if so.
```

Parallel execution: `ProcessPoolExecutor(max_workers=min(os.cpu_count()-1, 2))` is the prudent cap on T4 (even though Cell 4 used 7). OpenFace's internal OpenCV threading competes for the same cores, so effective speedup caps around 3-4× regardless of worker count.
