# Data dictionary

This directory holds intermediate CSV files used to build the tables and
figures in the CSCI-535 turn-taking paper. Each section below documents one
CSV. 
---

## `data_for_figs_tau_400.csv`

One row per experiment, with all metrics pinned to **τ=400 ms** (the
training horizon). Sourced from the canonical full-corpus run at
`turn_taking_analysis/fusion_runs/manifest/20260504_021136/` on the
457-interaction `manifest.csv`. 24 rows spanning 5 ablation blocks.
Used to populate τ=400 ms tables and figures.

### Columns

- **rank_macro_f1** — 1-indexed rank by `macro_f1` descending across all 24 experiments. Rows are stored in this order.
- **ablation** — which ablation block the experiment belongs to: `Standard`, `Standard + Self-Attention (SSA)`, `Standard + Cross-Attention (SCA)`, `Coordination`, or `Coordination + Self-Attention (CSA)`.
- **experiment** — original machine-readable experiment key from `fusion_experiments.ipynb` (e.g. `cpc_both_neural_concat`).
- **experiment_name** — paper-friendly description of the streams (e.g. `Audio Dyad (Intermediate Fusion)`). Use this in table row labels.
- **config** — compact `MODALITY(roles)` notation for the stream set. `A` = speaker, `B` = listener. Modality tokens: `CPC` (audio), `OF` (OpenFace face), `Coord_sum` (19-dim scalar summary coordination features), `Coord_cont` (continuous windowed cross-correlation arrays). Example: `CPC(A+B) + OF(B) + Coord_sum(A)`.
- **fusion** — the fusion-model class from `fusion_lib.py` used (e.g. `NeuralConcatFusion`, `SelfAttentionFusion`).
- **arch** — paper-friendly architecture label derived from `fusion` (e.g. `Neural-Concat`, `Self-Attention`, `Cross-Attention`, `Self+Cross-Attention`, `Early Fusion`).
- **n_streams** — number of input streams (modality × role pairs) the model consumes.
- **streams_summary** — semicolon-delimited `<modality>/<role>` listing of every stream (e.g. `cpc/speaker; cpc/listener`). Raw form; `config` is the paper-friendly version.
- **has_cpc** — 1 if any stream uses CPC audio, else 0.
- **has_openface** — 1 if any stream uses OpenFace face features, else 0.
- **has_coord_summary** — 1 if a summary-mode coordination stream is present, else 0.
- **has_coord_continuous** — 1 if a continuous-mode (WCC array) coordination stream is present, else 0.
- **n_params** — total number of trainable parameters in the model.
- **best_epoch** — epoch at which the best validation macro-F1 was achieved (the saved checkpoint).
- **epochs_run** — total epochs trained before early stopping fired (= `best_epoch` + patience, capped at `EPOCHS=30`).
- **macro_f1** — unweighted mean of per-class F1 across HOLD/YIELD/BACKCHANNEL on the held-out test set at τ=400 ms. Headline metric for ranking.
- **weighted_f1** — support-weighted mean of per-class F1 (each class's F1 weighted by its test-set support). Less discriminating than `macro_f1` because HOLD dominates support.
- **macro_recall** — unweighted mean of per-class recall.
- **accuracy** — overall test-set classification accuracy. Mathematically equivalent to support-weighted macro recall.
- **h_f1** — F1 for the HOLD class.
- **h_recall** — recall for the HOLD class.
- **y_f1** — F1 for the YIELD class.
- **y_recall** — recall for the YIELD class.
- **b_f1** — F1 for the BACKCHANNEL class.
- **b_recall** — recall for the BACKCHANNEL class.

---

## `data_for_figs_tau_sweep.csv`

One row per **(experiment × τ)** pair, capturing how each experiment's
metrics evolve across the prediction-horizon grid. Sourced from the same
canonical full-corpus run at
`turn_taking_analysis/fusion_runs/manifest/20260504_021136/`. **114 rows**:
- Standard / SSA / SCA / CSA: full τ-grid `(100, 200, 400, 500, 800, 1600)` ms (5 + 5 + 2 + 6 = 18 experiments × 6 τs = 108 rows).
- Coordination: τ=400 ms only by design (silent zero-substitution hazard at other τs). 6 experiments × 1 τ = 6 rows.

Used to populate cross-τ tables and figures (per-class F1 vs τ curves,
heatmaps, τ-robustness analysis, etc.).

Row order: ablation (notebook order: standard → ssa → sca → coordination → csa)
→ experiment (notebook order within ablation) → τ ascending.

### Columns

- **tau_ms** — the prediction horizon (ms) for this row.
- **ablation** — same as in `data_for_figs_tau_400.csv`.
- **experiment** — same as in `data_for_figs_tau_400.csv`.
- **experiment_name** — same as in `data_for_figs_tau_400.csv`.
- **config** — same as in `data_for_figs_tau_400.csv`.
- **fusion** — same as in `data_for_figs_tau_400.csv`.
- **arch** — same as in `data_for_figs_tau_400.csv`.
- **n_streams** — same as in `data_for_figs_tau_400.csv`.
- **streams_summary** — same as in `data_for_figs_tau_400.csv`.
- **has_cpc** — same as in `data_for_figs_tau_400.csv`.
- **has_openface** — same as in `data_for_figs_tau_400.csv`.
- **has_coord_summary** — same as in `data_for_figs_tau_400.csv`.
- **has_coord_continuous** — same as in `data_for_figs_tau_400.csv`.
- **n_params** — same as in `data_for_figs_tau_400.csv`. Constant across τs for one experiment (training is τ-invariant).
- **best_epoch** — same as in `data_for_figs_tau_400.csv`. Constant across τs for one experiment.
- **epochs_run** — same as in `data_for_figs_tau_400.csv`. Constant across τs for one experiment.
- **n_test_samples** — number of test samples evaluated at this τ. Varies slightly across τs because labels and the sample-existence filter both depend on τ.
- **test_loss** — class-weighted CrossEntropyLoss on the test set at this τ (using weights derived from the training-τ class distribution).
- **macro_precision** — unweighted mean of per-class precision.
- **macro_recall** — unweighted mean of per-class recall.
- **macro_f1** — unweighted mean of per-class F1.
- **weighted_f1** — support-weighted mean of per-class F1.
- **h_precision** — precision for the HOLD class.
- **h_recall** — recall for the HOLD class.
- **h_f1** — F1 for the HOLD class.
- **h_support** — count of test samples whose true label is HOLD at this τ. The class distribution shifts across τs (more boundary events fall inside longer horizons).
- **h_tp** — true positives for HOLD.
- **h_fp** — false positives for HOLD.
- **h_fn** — false negatives for HOLD.
- **y_precision** — precision for the YIELD class.
- **y_recall** — recall for the YIELD class.
- **y_f1** — F1 for the YIELD class.
- **y_support** — count of test samples whose true label is YIELD at this τ.
- **y_tp** — true positives for YIELD.
- **y_fp** — false positives for YIELD.
- **y_fn** — false negatives for YIELD.
- **b_precision** — precision for the BACKCHANNEL class.
- **b_recall** — recall for the BACKCHANNEL class.
- **b_f1** — F1 for the BACKCHANNEL class.
- **b_support** — count of test samples whose true label is BACKCHANNEL at this τ.
- **b_tp** — true positives for BACKCHANNEL.
- **b_fp** — false positives for BACKCHANNEL.
- **b_fn** — false negatives for BACKCHANNEL.
- **cm_h_h** — confusion matrix cell: true=HOLD, pred=HOLD. Equal to `h_tp`.
- **cm_h_y** — confusion matrix cell: true=HOLD, pred=YIELD.
- **cm_h_b** — confusion matrix cell: true=HOLD, pred=BACKCHANNEL.
- **cm_y_h** — confusion matrix cell: true=YIELD, pred=HOLD.
- **cm_y_y** — confusion matrix cell: true=YIELD, pred=YIELD. Equal to `y_tp`.
- **cm_y_b** — confusion matrix cell: true=YIELD, pred=BACKCHANNEL.
- **cm_b_h** — confusion matrix cell: true=BACKCHANNEL, pred=HOLD.
- **cm_b_y** — confusion matrix cell: true=BACKCHANNEL, pred=YIELD.
- **cm_b_b** — confusion matrix cell: true=BACKCHANNEL, pred=BACKCHANNEL. Equal to `b_tp`.
- **per_interaction_macro_f1_mean** — mean of per-interaction macro-F1 over the 67 test-set interactions. Measures average per-conversation performance.
- **per_interaction_macro_f1_std** — std of per-interaction macro-F1 across interactions. High std = uneven performance across conversations.
- **per_interaction_macro_f1_min** — minimum per-interaction macro-F1.
- **per_interaction_macro_f1_max** — maximum per-interaction macro-F1.
- **per_interaction_macro_f1_median** — median per-interaction macro-F1.
- **per_speaker_macro_f1_mean** — mean of per-speaker macro-F1 over the 134 test-set speakers (each speaker_file_id; both participants of each interaction count as separate speakers).
- **per_speaker_macro_f1_std** — std of per-speaker macro-F1 across speakers.
- **per_speaker_macro_f1_min** — minimum per-speaker macro-F1.
- **per_speaker_macro_f1_max** — maximum per-speaker macro-F1.
- **per_speaker_macro_f1_median** — median per-speaker macro-F1.
- **accuracy** — overall test-set classification accuracy at this τ. Computed as `(cm_h_h + cm_y_y + cm_b_b) / n_test_samples`.
- **balanced_accuracy** — mean of per-class recall, equivalent to `macro_recall`. Provided for convenience.
