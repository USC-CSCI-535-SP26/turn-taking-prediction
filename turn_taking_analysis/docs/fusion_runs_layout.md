# fusion_experiments.ipynb output layout

Target on-disk layout for `fusion_experiments.ipynb` after the per-experiment
incremental-save refactor (collapses today's Sections 4 + 5 + 7 writes into a
single per-experiment write inside each ablation block).

```
<BASE>/turn_taking_analysis/fusion_runs/
    <manifest_type>/                          ← e.g. manifest (level for the data used)
        <RUN_ID>/                             ← time.strftime() at notebook start
            index.json                        ← cross-ablation run-level summary
            <ablation>/                       ← standard | ssa | sca | coordination | csa
                run_index.json                ← per-ablation summary (crash-resilient)
                tau_curve.png                 ← per-ablation τ-curve
                <exp_name>/
                    results.json              ← merged S4+S5+S7 detailed (incrementally written)
                    model_state.pt            ← best-val-F1 checkpoint
                    partial.json              ← failure-only
```

## Path-segment definitions

- `<BASE>` — repo root (locally `/Users/rasikaramanan/.../csci535-project`).
- `<manifest_type>` — `MANIFEST_PATH.stem` (e.g. `manifest`, `poc_manifest`).
- `<RUN_ID>` — `time.strftime("%Y%m%d_%H%M%S")` resolved **once** at notebook
  start; shared across all five ablation blocks within one run.
- `<ablation>` — one of `standard`, `ssa`, `sca`, `coordination`, `csa`.
- `<exp_name>` — experiment key (e.g. `cpc_speaker_coord_self_attention`).

## File contents

- `results.json` — merged S4+S7 shape: `experiment_name`, `config`,
  `stream_dims`, `device`, `n_params`, `class_weights_used`, `samples_summary`,
  `train_history`, `best_epoch`, `best_val_macro_f1`, `epochs_completed`,
  `tau_results[<tau_ms>]` (n_test_samples, test_loss, macro p/r/f1,
  weighted_f1, per_class p/r/f1/support/tp/fp/fn, confusion_matrix,
  class_distribution_test, per_interaction / per_speaker / per_relationship /
  per_relationship_detail breakdowns, sample_records). Written incrementally
  after every successful τ inside the experiment loop, then a final overwrite
  once that experiment's τ-grid completes.
- `model_state.pt` — `torch.save` of best-val-macro-F1 checkpoint state dict.
- `partial.json` — same shape as `results.json`; written **only** from the
  `except` handler when a τ raises mid-loop. Absent on a clean run.
- `run_index.json` — per-ablation: `run_id`, ISO timestamp, `manifest_path`,
  `train_labels_path`, `train_tau_ms`, `tau_grid_ms`, full `modality_registry`,
  `global_hyperparams`, `experiments` (this ablation's keys), per-experiment
  summary (`fusion`, `streams`, `n_params`, `best_epoch`, `epochs_run`,
  `test_macro_f1_at_train_tau`), `environment` block. Written at end of block
  so each ablation's run_index lands the moment its block finishes.
- `tau_curve.png` — per-ablation τ-curve (one line per experiment, log-x);
  saved via `fig.savefig(..., dpi=150, bbox_inches="tight")` after the
  experiment loop in that ablation block.
- `index.json` — cross-ablation rollup written once at notebook end.
  Aggregates the five `run_index.json` summaries into a single comparison view
  plus shared global hyperparams (which are identical across ablations within
  one run). Written by `fusion_runner.write_run_index(...)`.

## Behavioral contracts

- **All writes land under `turn_taking_analysis/fusion_runs/`.** Nothing
  writes to `<BASE>/fusion_runs/` (the legacy `OUT_ROOT` and `TAU_OUT_ROOT`
  hardcoded paths in the current notebook).
- **Re-running a single ablation cell overwrites that ablation's dir
  in-place** under the current `RUN_ID`. To start fresh, restart the notebook
  so `RUN_ID` rotates.
- **`fusion_lib.py` does no file writes** under this layout. All I/O lives
  in `fusion_runner.py`'s `run_ablation_block` and `write_run_index`,
  imported and orchestrated by the notebook.
- **No `tau_curves_all.png` is persisted.** Cross-ablation τ-curve plots can
  be reconstructed offline from any subset of `<ablation>/<exp_name>/results.json`
  files since each carries the full per-τ macro-F1 grid.
