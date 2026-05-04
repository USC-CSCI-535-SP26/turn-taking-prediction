"""
fusion_runner.py — drives one ablation block end-to-end.

Each call to run_ablation_block trains every experiment in the ablation,
rebuilds each model from its best-val-F1 checkpoint, runs the full per-τ
detailed evaluation, and saves all artifacts before returning. The cross-
ablation index.json is written by write_run_index() once after every
ablation block has finished.

Output layout (per turn_taking_analysis/docs/fusion_runs_layout.md):

    <out_root>/<ablation_name>/
        run_index.json
        tau_curve.png
        <exp_name>/
            results.json     — incrementally written after every τ
            model_state.pt   — best-val-F1 checkpoint
            partial.json     — failure-only

This module owns all the file-I/O logic that previously lived in the
notebook's Sections 4 + 5 + 7. Keep that contract: the notebook passes
configuration, this module performs all writes.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import fusion_lib as fl


# ---------------------------------------------------------------------------
# Detailed-eval helpers (lifted from the old Section 7 cell, no logic change).
# ---------------------------------------------------------------------------


def _inference_with_probs(model, loader, device):
    model.eval()
    probs_chunks, preds, labels = [], [], []
    with torch.inference_mode():
        for streams, ys in loader:
            streams = [t.to(device) for t in streams]
            logits = model(streams)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_chunks.append(p)
            preds.extend(p.argmax(-1).tolist())
            labels.extend(ys.cpu().tolist())
    probs = (
        np.concatenate(probs_chunks, axis=0)
        if probs_chunks
        else np.zeros((0, fl.NUM_CLASSES), dtype=np.float32)
    )
    return probs, preds, labels


def _confusion_matrix(preds, labels, num_classes=None):
    if num_classes is None:
        num_classes = fl.NUM_CLASSES
    cm = [[0] * num_classes for _ in range(num_classes)]
    for p, y in zip(preds, labels):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            cm[y][p] += 1
    return cm


def _per_class_metrics(preds, labels, num_classes=None):
    if num_classes is None:
        num_classes = fl.NUM_CLASSES
    out = {}
    for c in range(num_classes):
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        support = sum(1 for y in labels if y == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[fl.CLASS_NAMES[c]] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": support, "tp": tp, "fp": fp, "fn": fn,
        }
    return out


def _aggregate_macro(per_class):
    classes = list(per_class.values())
    n = len(classes) or 1
    macro_p = sum(c["precision"] for c in classes) / n
    macro_r = sum(c["recall"] for c in classes) / n
    macro_f1 = sum(c["f1"] for c in classes) / n
    total_support = sum(c["support"] for c in classes) or 1
    weighted_f1 = sum(c["f1"] * c["support"] for c in classes) / total_support
    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def _grouped_metrics(samples, preds, labels, key_fn):
    groups = defaultdict(lambda: {"preds": [], "labels": []})
    for s, p, y in zip(samples, preds, labels):
        k = key_fn(s)
        groups[k]["preds"].append(p)
        groups[k]["labels"].append(y)
    out = {}
    for k, d in groups.items():
        per_class = _per_class_metrics(d["preds"], d["labels"])
        agg = _aggregate_macro(per_class)
        out[k] = {
            "n_samples": len(d["preds"]),
            **agg,
            "per_class": per_class,
            "confusion_matrix": _confusion_matrix(d["preds"], d["labels"]),
        }
    return out


def _build_iid_lookup(manifest_rows, field, default="unknown"):
    return {r["interaction_id"]: r.get(field, default) for r in manifest_rows}


# ---------------------------------------------------------------------------
# Internal: per-ablation outputs (run_index.json, tau_curve.png).
# ---------------------------------------------------------------------------


def _build_run_index(
    *,
    ablation_name,
    detailed_results,
    manifest_path,
    train_labels_path,
    train_tau_ms,
    tau_grid_ms,
    modality_registry,
    global_hyperparams,
    device,
):
    return {
        "ablation": ablation_name,
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "manifest_path": str(manifest_path),
        "train_labels_path": str(train_labels_path),
        "train_tau_ms": train_tau_ms,
        "tau_grid_ms": list(tau_grid_ms),
        "modality_registry": _modality_registry_for_json(modality_registry),
        "global_hyperparams": dict(global_hyperparams),
        "experiments": list(detailed_results.keys()),
        "experiment_summary": {
            name: {
                "fusion": d["config"]["fusion"],
                "streams": d["config"]["streams"],
                "n_params": d["n_params"],
                "best_epoch": d["best_epoch"],
                "epochs_run": d["epochs_completed"],
                "test_macro_f1_at_train_tau": (
                    d["tau_results"].get(str(int(train_tau_ms)), {}).get("macro_f1")
                ),
            }
            for name, d in detailed_results.items()
        },
        "environment": {
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "device": str(device),
            "platform": sys.platform,
        },
    }


def _modality_registry_for_json(registry):
    """Coerce Path values inside the registry to strings so json.dump works."""
    return {
        name: {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
        for name, cfg in registry.items()
    }


def _save_tau_curve_png(out_dir, ablation_name, detailed_results, train_tau_ms):
    """One line per experiment, log-x. Saved to <out_dir>/tau_curve.png."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plotted_any = False
    for exp_name, d in detailed_results.items():
        tau_results = d.get("tau_results", {})
        if not tau_results:
            continue
        taus = sorted(int(t) for t in tau_results.keys())
        f1s = [tau_results[str(t)]["macro_f1"] for t in taus]
        ax.plot(taus, f1s, marker="o", label=exp_name)
        plotted_any = True

    ax.set_xlabel("τ (ms) — prediction horizon")
    ax.set_ylabel("test macro-F1")
    ax.set_title(
        f"τ-curve [{ablation_name}] (trained @ τ={train_tau_ms} ms)"
    )
    if plotted_any:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = Path(out_dir) / "tau_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def run_ablation_block(
    *,
    ablation_name,
    experiments,
    use_coordination,
    use_attention,
    out_root,
    modality_registry,
    manifest_rows,
    manifest_path,
    train_labels_path,
    labels_dir,
    tau_grid_ms,
    train_tau_ms,
    coordination_csv_path=None,
    hidden_size=64,
    dropout=0.3,
    num_layers_early=3,
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    patience=4,
    num_workers=0,
    seed=42,
    max_samples_per_split=None,
    device="auto",
):
    """
    Train every experiment, rebuild each best checkpoint, run per-τ detailed
    eval, and persist artifacts incrementally. After the per-experiment loop:
    write the per-ablation run_index.json and tau_curve.png.

    Returns (detailed_results, out_dir).
    """
    out_dir = Path(out_root) / ablation_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device_t = fl.resolve_device(device) if isinstance(device, str) else device

    coordination_lookup = None
    if coordination_csv_path is not None and Path(str(coordination_csv_path)).exists():
        coordination_lookup = fl.load_coordination_features(
            str(coordination_csv_path),
            feature_cols=fl.COORDINATION_FEATURE_COLUMNS,
        )

    iid_to_relationship = _build_iid_lookup(manifest_rows, "relationship")
    iid_to_relationship_detail = _build_iid_lookup(manifest_rows, "relationship_detail")

    detailed_results = {}

    for exp_name, cfg in tqdm(experiments.items(), desc=f"{ablation_name} train+eval"):
        print(f"\n[{ablation_name}] {exp_name}")

        runner = fl.run_experiment_coordination if use_coordination else fl.run_experiment
        kwargs = dict(
            name=exp_name,
            streams=cfg["streams"],
            fusion=cfg["fusion"],
            modality_registry=modality_registry,
            manifest_rows=manifest_rows,
            labels_path=str(train_labels_path),
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers_early=num_layers_early,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
            max_samples_per_split=max_samples_per_split,
        )
        if use_coordination:
            kwargs.update(
                coordination_csv_path=str(coordination_csv_path) if coordination_csv_path else None,
                missing_coordination="zeros",
            )
        if use_attention:
            kwargs.update(
                attention_dim=cfg.get("attention_dim", hidden_size),
                attention_heads=cfg.get(
                    "attention_heads_by_modality",
                    cfg.get("attention_heads", 4),
                ),
                attention_layers=cfg.get("attention_layers", 2),
                attention_pooling=cfg.get("attention_pooling", "mean"),
            )

        train_res = runner(**kwargs)

        train_cfg = train_res["config"]
        stream_cfg = train_cfg["streams"]
        fusion = train_cfg["fusion"]
        stream_dims = train_res["stream_dims"]

        # SSA/CSA/SCA models must rebuild with the same attention shape they
        # were trained with. CSA in particular: cfg["attention_heads"] is a
        # list like [4,4,4,4,2]; without forwarding it the rebuild silently
        # uses scalar 4. State-dict load still succeeds (MultiheadAttention
        # parameter shapes don't depend on num_heads), but the runtime head
        # partitioning differs from training and predictions diverge. GRU
        # families ignore these kwargs.
        model = fl.build_model(
            fusion,
            stream_dims,
            hidden_size=train_cfg["hidden_size"],
            dropout=train_cfg.get("dropout", 0.3),
            num_layers_early=train_cfg.get("num_layers_early", 3),
            attention_dim=train_cfg.get("attention_dim"),
            attention_heads=train_cfg.get("attention_heads", 4),
            attention_layers=train_cfg.get("attention_layers", 2),
            attention_pooling=train_cfg.get("attention_pooling", "mean"),
        ).to(device_t)
        model.load_state_dict(train_res["model_state"])

        train_labels_for_weights = fl.load_labels(train_cfg["labels_path"])
        train_samples_for_weights = [
            s for s in fl.enumerate_samples(manifest_rows, train_labels_for_weights)
            if s["split"] == "train"
        ]
        class_weights_used = (
            fl.compute_class_weights(train_samples_for_weights).cpu().tolist()
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights_used, dtype=torch.float32, device=device_t)
        )

        history = train_res["train_history"]
        best_epoch_entry = (
            max(history, key=lambda h: h["val_macro_f1"]) if history else None
        )
        n_params = sum(int(p.numel()) for p in model.parameters())

        detailed = {
            "experiment_name": exp_name,
            "config": train_cfg,
            "stream_dims": stream_dims,
            "device": train_res["device"],
            "n_params": n_params,
            "class_weights_used": class_weights_used,
            "samples_summary": train_res["samples_summary"],
            "train_history": history,
            "best_epoch": best_epoch_entry["epoch"] if best_epoch_entry else None,
            "best_val_macro_f1": (
                best_epoch_entry["val_macro_f1"] if best_epoch_entry else None
            ),
            "epochs_completed": len(history),
            "tau_results": {},
        }
        detailed_results[exp_name] = detailed

        exp_dir = out_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        results_path = exp_dir / "results.json"
        partial_path = exp_dir / "partial.json"
        model_path = exp_dir / "model_state.pt"

        for tau_ms in tau_grid_ms:
            try:
                labels_path = Path(labels_dir) / f"labels_tau_{int(tau_ms):04d}.json"
                if not labels_path.exists():
                    print(f"  τ={tau_ms}: labels missing, skipping ({labels_path})")
                    continue

                labels_dict = fl.load_labels(str(labels_path))
                samples = fl.enumerate_samples(manifest_rows, labels_dict)
                test_samples = [s for s in samples if s["split"] == "test"]
                test_samples, _ = fl.filter_samples_with_existing_files(
                    test_samples, stream_cfg, modality_registry,
                )
                if not test_samples:
                    print(f"  τ={tau_ms}: no test samples after file filter, skipping")
                    continue

                loader = fl.make_dataloader_coordination(
                    test_samples,
                    stream_cfg,
                    modality_registry,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    coordination_lookup=coordination_lookup,
                    coordination_feature_dim=len(fl.COORDINATION_FEATURE_COLUMNS),
                    missing_coordination="zeros",
                )

                # Single pass: inference probs + loss accumulation. The old
                # S7 cell iterated the loader twice — folded here for ~2×
                # speedup at no behavioral cost.
                model.eval()
                probs_chunks, preds, labels_list = [], [], []
                test_loss_sum, n_seen = 0.0, 0
                with torch.inference_mode():
                    for streams_batch, ys in loader:
                        streams_batch = [t.to(device_t) for t in streams_batch]
                        ys = ys.to(device_t)
                        logits = model(streams_batch)
                        p = torch.softmax(logits, dim=-1).cpu().numpy()
                        probs_chunks.append(p)
                        preds.extend(p.argmax(-1).tolist())
                        labels_list.extend(ys.cpu().tolist())
                        test_loss_sum += criterion(logits, ys).item() * ys.size(0)
                        n_seen += ys.size(0)
                probs = (
                    np.concatenate(probs_chunks, axis=0)
                    if probs_chunks
                    else np.zeros((0, fl.NUM_CLASSES), dtype=np.float32)
                )
                test_loss = test_loss_sum / max(n_seen, 1)

                per_class = _per_class_metrics(preds, labels_list)
                agg = _aggregate_macro(per_class)

                sample_records = [
                    {
                        "basename": s["basename"],
                        "interaction_id": s["interaction_id"],
                        "speaker_file_id": s["speaker_file_id"],
                        "listener_file_id": s["listener_file_id"],
                        "window_start_s": s["start_s"],
                        "window_end_s": s["end_s"],
                        "split": s["split"],
                        "label": int(y),
                        "pred": int(p),
                        "probs": [float(x) for x in row],
                        "correct": int(p) == int(y),
                    }
                    for s, p, y, row in zip(test_samples, preds, labels_list, probs)
                ]

                detailed["tau_results"][str(int(tau_ms))] = {
                    "n_test_samples": len(test_samples),
                    "test_loss": test_loss,
                    **agg,
                    "per_class": per_class,
                    "confusion_matrix": _confusion_matrix(preds, labels_list),
                    "class_distribution_test": dict(Counter(labels_list)),
                    "per_interaction": _grouped_metrics(
                        test_samples, preds, labels_list,
                        key_fn=lambda s: s["interaction_id"],
                    ),
                    "per_speaker": _grouped_metrics(
                        test_samples, preds, labels_list,
                        key_fn=lambda s: s["speaker_file_id"],
                    ),
                    "per_relationship": _grouped_metrics(
                        test_samples, preds, labels_list,
                        key_fn=lambda s: iid_to_relationship.get(s["interaction_id"], "unknown"),
                    ),
                    "per_relationship_detail": _grouped_metrics(
                        test_samples, preds, labels_list,
                        key_fn=lambda s: iid_to_relationship_detail.get(s["interaction_id"], "unknown"),
                    ),
                    "sample_records": sample_records,
                }

                with open(results_path, "w") as f:
                    json.dump(detailed, f, indent=2, default=str)
                print(
                    f"  τ={tau_ms}: macro_f1={agg['macro_f1']:.4f}  "
                    f"saved → {results_path}"
                )

            except Exception as e:
                print(f"  τ={tau_ms}: FAILED — {e!r}")
                with open(partial_path, "w") as f:
                    json.dump(detailed, f, indent=2, default=str)
                print(f"  partial save → {partial_path}")
                raise

        # Final overwrite (safety net) + checkpoint.
        with open(results_path, "w") as f:
            json.dump(detailed, f, indent=2, default=str)
        torch.save(train_res["model_state"], model_path)
        print(f"  wrote {results_path}  +  {model_path}")

    run_index = _build_run_index(
        ablation_name=ablation_name,
        detailed_results=detailed_results,
        manifest_path=manifest_path,
        train_labels_path=train_labels_path,
        train_tau_ms=train_tau_ms,
        tau_grid_ms=tau_grid_ms,
        modality_registry=modality_registry,
        global_hyperparams={
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_layers_early": num_layers_early,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "num_workers": num_workers,
            "seed": seed,
            "max_samples_per_split": max_samples_per_split,
        },
        device=device_t,
    )
    with open(out_dir / "run_index.json", "w") as f:
        json.dump(run_index, f, indent=2, default=str)

    _save_tau_curve_png(out_dir, ablation_name, detailed_results, train_tau_ms)

    print(f"[{ablation_name}] block complete → {out_dir}")
    return detailed_results, out_dir


def write_run_index(
    *,
    run_root,
    ablation_results,
    manifest_path,
    train_labels_path,
    train_tau_ms,
    tau_grid_ms_by_ablation,
    modality_registry,
    global_hyperparams,
    device,
):
    """
    Cross-ablation rollup written to <run_root>/index.json once after every
    ablation block finishes. ablation_results: {ablation_name: detailed_results}.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    ablations_block = {}
    for ablation_name, detailed_results in ablation_results.items():
        ablations_block[ablation_name] = {
            "experiments": list(detailed_results.keys()),
            "tau_grid_ms": list(tau_grid_ms_by_ablation.get(ablation_name, ())),
            "experiment_summary": {
                name: {
                    "fusion": d["config"]["fusion"],
                    "streams": d["config"]["streams"],
                    "n_params": d["n_params"],
                    "best_epoch": d["best_epoch"],
                    "epochs_run": d["epochs_completed"],
                    "test_macro_f1_at_train_tau": (
                        d["tau_results"].get(str(int(train_tau_ms)), {}).get("macro_f1")
                    ),
                }
                for name, d in detailed_results.items()
            },
        }

    index = {
        "run_id": run_root.name,
        "timestamp": datetime.now().isoformat(),
        "manifest_path": str(manifest_path),
        "train_labels_path": str(train_labels_path),
        "train_tau_ms": train_tau_ms,
        "modality_registry": _modality_registry_for_json(modality_registry),
        "global_hyperparams": dict(global_hyperparams),
        "ablations": ablations_block,
        "environment": {
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "device": str(device),
            "platform": sys.platform,
        },
    }

    out_path = run_root / "index.json"
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    return out_path
