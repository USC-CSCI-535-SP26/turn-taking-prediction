"""
Tier 1 — fully mocked unit tests for fusion_runner.run_ablation_block and
fusion_runner.write_run_index.

Goal: verify the harness logic (path layout, file shapes, failure paths,
attention forwarding, legacy-path isolation) without running real training.
Should complete in a few seconds end-to-end.

Strategy: monkeypatch every fusion_lib function that run_ablation_block
calls so we control inputs/outputs deterministically. We do NOT mock torch
itself — we use a tiny real nn.Module so model.eval(), state_dict round-
trip, and DataLoader iteration all use the real machinery.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import fusion_lib as fl  # noqa: E402  imported through conftest sys.path
import fusion_runner as fr  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles.
# ---------------------------------------------------------------------------


class _FakeModel(torch.nn.Module):
    """Tiny model with at least one parameter so n_params > 0."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(1, num_classes)

    def forward(self, streams):
        # streams is a list of (B, T, F) tensors. Return zero logits per class.
        B = streams[0].shape[0]
        return self.linear(
            torch.zeros(B, 1, device=streams[0].device)
        )


def _fake_loader(n_samples: int = 4, batch_size: int = 2):
    """Yields (streams_list, labels_tensor) tuples like fl.make_dataloader_*."""
    batches = []
    classes = [0, 1, 2]
    i = 0
    while i < n_samples:
        b = min(batch_size, n_samples - i)
        streams = [torch.zeros(b, 5, 8)]  # (B, T=5, F=8)
        labels = torch.tensor(
            [classes[(i + k) % len(classes)] for k in range(b)],
            dtype=torch.long,
        )
        batches.append((streams, labels))
        i += b
    return iter(batches)


def _make_train_result(*, name, streams_cfg, fusion="neural_concat", n_epochs=2):
    """Mimics fl.run_experiment's return dict, with model_state included."""
    fake_model = _FakeModel()
    return {
        "name": name,
        "config": {
            "streams": streams_cfg,
            "fusion": fusion,
            "hidden_size": 16,
            "dropout": 0.3,
            "num_layers_early": 2,
            "epochs": n_epochs,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "patience": 2,
            "labels_path": "/dev/null/labels_tau_0400.json",
            "attention_dim": None,
            "attention_heads": 4,
            "attention_layers": 2,
            "attention_pooling": "mean",
        },
        "stream_dims": [8],
        "device": "cpu",
        "samples_summary": {
            "total": 12,
            "per_split": {"train": 6, "val": 3, "test": 3},
            "per_split_class": {
                "train": {0: 2, 1: 2, 2: 2},
                "val": {0: 1, 1: 1, 2: 1},
                "test": {0: 1, 1: 1, 2: 1},
            },
        },
        "train_history": [
            {"epoch": e, "train_loss": 1.0 - 0.1 * e, "val_loss": 0.9 - 0.1 * e,
             "val_macro_f1": 0.1 + 0.1 * e}
            for e in range(1, n_epochs + 1)
        ],
        "test_eval": {"loss": 0.5, "macro_f1": 0.42, "per_class_f1": [0.4, 0.4, 0.4]},
        "model_state": fake_model.state_dict(),
    }


def _fake_samples(n: int = 3, split: str = "test"):
    return [
        {
            "basename": f"basename_{i}",
            "interaction_id": f"iid_{i % 2}",
            "speaker_file_id": f"iid_{i % 2}_p0",
            "listener_file_id": f"iid_{i % 2}_p1",
            "start_s": float(i),
            "end_s": float(i) + 2.0,
            "split": split,
            "label": i % 3,
        }
        for i in range(n)
    ]


def _fake_manifest_rows():
    return [
        {"interaction_id": "iid_0", "split": "test",
         "relationship": "friends", "relationship_detail": "close-friends"},
        {"interaction_id": "iid_1", "split": "test",
         "relationship": "strangers", "relationship_detail": "first-meeting"},
    ]


# ---------------------------------------------------------------------------
# Shared mock-installer fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fl(monkeypatch):
    """
    Patch every fusion_lib symbol that fusion_runner.run_ablation_block
    touches. Tests can override individual mocks via the returned dict.
    """
    train_results: dict[str, dict] = {}

    def fake_run_experiment(**kwargs):
        name = kwargs["name"]
        res = _make_train_result(name=name, streams_cfg=kwargs["streams"],
                                  fusion=kwargs["fusion"])
        train_results[name] = res
        return res

    fake_build_model = MagicMock(side_effect=lambda *a, **kw: _FakeModel())
    fake_load_coordination_features = MagicMock(return_value={})
    fake_resolve_device = MagicMock(return_value=torch.device("cpu"))
    fake_load_labels = MagicMock(return_value={"basename_0": 0, "basename_1": 1, "basename_2": 2})
    fake_enumerate_samples = MagicMock(side_effect=lambda mr, lbl: _fake_samples(n=3, split="test"))
    fake_filter = MagicMock(side_effect=lambda samples, streams, reg: (samples, []))
    fake_make_loader = MagicMock(side_effect=lambda *a, **kw: _fake_loader(n_samples=3, batch_size=2))
    fake_compute_class_weights = MagicMock(
        return_value=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    )

    monkeypatch.setattr(fr.fl, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(fr.fl, "run_experiment_coordination", fake_run_experiment)
    monkeypatch.setattr(fr.fl, "build_model", fake_build_model)
    monkeypatch.setattr(fr.fl, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(fr.fl, "load_coordination_features", fake_load_coordination_features)
    monkeypatch.setattr(fr.fl, "load_labels", fake_load_labels)
    monkeypatch.setattr(fr.fl, "enumerate_samples", fake_enumerate_samples)
    monkeypatch.setattr(fr.fl, "filter_samples_with_existing_files", fake_filter)
    monkeypatch.setattr(fr.fl, "make_dataloader_coordination", fake_make_loader)
    monkeypatch.setattr(fr.fl, "compute_class_weights", fake_compute_class_weights)

    return {
        "train_results": train_results,
        "build_model": fake_build_model,
        "load_coordination_features": fake_load_coordination_features,
        "load_labels": fake_load_labels,
        "make_dataloader_coordination": fake_make_loader,
    }


@pytest.fixture
def label_files(tmp_path):
    """Create sentinel label files so the τ-loop's exists()-check passes."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for tau in (400, 800):
        (labels_dir / f"labels_tau_{tau:04d}.json").write_text("{}")
    return labels_dir


@pytest.fixture
def standard_kwargs(tmp_path, label_files):
    """Common kwargs for run_ablation_block."""
    return dict(
        ablation_name="standard",
        experiments={
            "exp_a": {"streams": [{"modality": "cpc", "role": "speaker"}],
                       "fusion": "unimodal"},
            "exp_b": {"streams": [{"modality": "cpc", "role": "speaker"}],
                       "fusion": "unimodal"},
        },
        use_coordination=False,
        use_attention=False,
        out_root=tmp_path / "fusion_runs" / "manifest" / "20260101_000000",
        modality_registry={
            "cpc": {"dir": "/fake/cpc", "feature_dim": 8, "frame_rate_hz": 100.0},
        },
        manifest_rows=_fake_manifest_rows(),
        manifest_path="/fake/manifest.csv",
        train_labels_path=str(label_files / "labels_tau_0400.json"),
        labels_dir=label_files,
        tau_grid_ms=(400, 800),
        train_tau_ms=400,
        hidden_size=16,
        dropout=0.3,
        num_layers_early=2,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        patience=2,
        num_workers=0,
        seed=42,
        max_samples_per_split=4,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_clean_run_writes_expected_files(mock_fl, standard_kwargs):
    detailed_results, out_dir = fr.run_ablation_block(**standard_kwargs)

    # out_dir is exactly <out_root>/<ablation_name> — no nested timestamp.
    assert out_dir == Path(standard_kwargs["out_root"]) / "standard"
    assert out_dir.exists()

    # Per-experiment artifacts exist; partial.json absent on clean run.
    for exp_name in ("exp_a", "exp_b"):
        exp_dir = out_dir / exp_name
        assert exp_dir.is_dir(), exp_dir
        assert (exp_dir / "results.json").is_file()
        assert (exp_dir / "model_state.pt").is_file()
        assert not (exp_dir / "partial.json").exists()

    # Per-ablation artifacts exist.
    assert (out_dir / "run_index.json").is_file()
    assert (out_dir / "tau_curve.png").is_file()
    assert (out_dir / "tau_curve.png").stat().st_size > 0


def test_results_json_top_level_shape(mock_fl, standard_kwargs):
    detailed_results, out_dir = fr.run_ablation_block(**standard_kwargs)

    res = json.loads((out_dir / "exp_a" / "results.json").read_text())

    expected_keys = {
        "experiment_name", "config", "stream_dims", "device", "n_params",
        "class_weights_used", "samples_summary", "train_history",
        "best_epoch", "best_val_macro_f1", "epochs_completed", "tau_results",
    }
    assert expected_keys <= set(res.keys()), (
        f"missing keys: {expected_keys - set(res.keys())}"
    )
    assert res["experiment_name"] == "exp_a"
    assert isinstance(res["tau_results"], dict)


def test_tau_results_per_tau_shape(mock_fl, standard_kwargs):
    detailed_results, out_dir = fr.run_ablation_block(**standard_kwargs)

    res = json.loads((out_dir / "exp_a" / "results.json").read_text())

    # Every τ in the grid is a key.
    assert set(res["tau_results"].keys()) == {"400", "800"}

    expected_per_tau_keys = {
        "n_test_samples", "test_loss",
        "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "per_class", "confusion_matrix", "class_distribution_test",
        "per_interaction", "per_speaker", "per_relationship",
        "per_relationship_detail", "sample_records",
    }
    for tau_key, tau_block in res["tau_results"].items():
        assert expected_per_tau_keys <= set(tau_block.keys()), (
            f"τ={tau_key} missing keys: {expected_per_tau_keys - set(tau_block.keys())}"
        )
        # sample_records cardinality matches n_test_samples.
        assert len(tau_block["sample_records"]) == tau_block["n_test_samples"]
        # confusion_matrix is square and matches NUM_CLASSES.
        cm = tau_block["confusion_matrix"]
        assert len(cm) == fl.NUM_CLASSES
        for row in cm:
            assert len(row) == fl.NUM_CLASSES


def test_partial_json_on_failure(mock_fl, standard_kwargs, monkeypatch):
    """Force the second τ to raise; assert partial.json captures the in-progress
    detailed dict and the exception re-raises."""
    call_count = {"n": 0}
    real_make_loader = mock_fl["make_dataloader_coordination"].side_effect

    def maybe_raising_loader(*a, **kw):
        call_count["n"] += 1
        # First experiment, first τ → fine. Second call (still first exp,
        # second τ) raises.
        if call_count["n"] == 2:
            raise RuntimeError("boom mid-tau")
        return real_make_loader(*a, **kw)

    monkeypatch.setattr(
        fr.fl, "make_dataloader_coordination",
        MagicMock(side_effect=maybe_raising_loader),
    )

    with pytest.raises(RuntimeError, match="boom mid-tau"):
        fr.run_ablation_block(**standard_kwargs)

    out_dir = Path(standard_kwargs["out_root"]) / "standard"
    # First experiment failed at τ=800 — partial.json present, results.json
    # missing the final overwrite (still has τ=400 data from the in-loop save).
    assert (out_dir / "exp_a" / "partial.json").is_file()
    partial = json.loads((out_dir / "exp_a" / "partial.json").read_text())
    assert "400" in partial["tau_results"]
    assert "800" not in partial["tau_results"]

    # Second experiment never started — its dir should not exist.
    assert not (out_dir / "exp_b").exists()


def test_coord_grid_override(mock_fl, standard_kwargs):
    """Single-τ grid produces single-τ tau_results."""
    standard_kwargs["tau_grid_ms"] = (400,)
    detailed_results, out_dir = fr.run_ablation_block(**standard_kwargs)

    res = json.loads((out_dir / "exp_a" / "results.json").read_text())
    assert set(res["tau_results"].keys()) == {"400"}


def test_attention_args_forwarded_to_build_model(mock_fl, standard_kwargs):
    """SSA/CSA/SCA need attention_heads/dim/layers/pooling forwarded from cfg
    so the rebuilt model matches training-time architecture. This is the
    bug Sika fixed in S7 manually; the new harness must preserve it."""
    standard_kwargs["use_attention"] = True
    standard_kwargs["experiments"] = {
        "exp_a": {
            "streams": [{"modality": "cpc", "role": "speaker"}],
            "fusion": "self_attention",
            "attention_heads": [4, 4, 4, 4, 2],  # CSA-shape list
            "attention_layers": 3,
            "attention_pooling": "max",
        },
    }
    fr.run_ablation_block(**standard_kwargs)

    # build_model is called once per experiment (after training).
    assert mock_fl["build_model"].call_count == 1
    _, kwargs = mock_fl["build_model"].call_args
    # The fake training stub copies cfg verbatim, so attention args land in
    # train_cfg["attention_*"]. fusion_runner forwards them.
    assert kwargs.get("attention_heads") == 4  # default scalar — see note below
    # Note: our fake _make_train_result hardcodes attention_heads=4 in the
    # returned config (it doesn't echo the cfg the runner was called with).
    # That's deliberate — the test is verifying that whatever lands in
    # train_cfg gets forwarded, not what the user passed in. To check the
    # forwarding contract more strictly, override _make_train_result.


def test_attention_args_forwarded_strict(mock_fl, standard_kwargs, monkeypatch):
    """Same as above but stub the runner to echo the user's attention args
    into the returned config so we can assert end-to-end forwarding."""
    def echoing_runner(**kwargs):
        res = _make_train_result(name=kwargs["name"], streams_cfg=kwargs["streams"],
                                  fusion=kwargs["fusion"])
        # Echo the attention args the runner was called with into config.
        for k in ("attention_dim", "attention_heads", "attention_layers", "attention_pooling"):
            if k in kwargs:
                res["config"][k] = kwargs[k]
        return res

    monkeypatch.setattr(fr.fl, "run_experiment", echoing_runner)

    standard_kwargs["use_attention"] = True
    standard_kwargs["experiments"] = {
        "csa_demo": {
            "streams": [{"modality": "cpc", "role": "speaker"}],
            "fusion": "self_attention",
            "attention_heads_by_modality": [4, 4, 4, 4, 2],
            "attention_layers": 3,
            "attention_pooling": "max",
        },
    }
    fr.run_ablation_block(**standard_kwargs)

    _, build_kwargs = mock_fl["build_model"].call_args
    assert build_kwargs["attention_heads"] == [4, 4, 4, 4, 2]
    assert build_kwargs["attention_layers"] == 3
    assert build_kwargs["attention_pooling"] == "max"


def test_no_legacy_path_leak(mock_fl, standard_kwargs, tmp_path):
    """Nothing should land outside out_root — verifies no hardcoded
    <BASE>/fusion_runs/summary or <BASE>/fusion_runs/tau_sweeps writes."""
    fr.run_ablation_block(**standard_kwargs)

    # Anything under tmp_path that is NOT under out_root is forbidden.
    out_root = Path(standard_kwargs["out_root"])
    for p in tmp_path.rglob("*"):
        if p.is_file():
            try:
                p.relative_to(out_root)
            except ValueError:
                # File is outside out_root — only the label-files fixture is
                # allowed to live elsewhere.
                if p.parent.name == "labels" and p.name.startswith("labels_tau_"):
                    continue
                raise AssertionError(f"unexpected file outside out_root: {p}")


def test_run_index_json_per_ablation(mock_fl, standard_kwargs):
    detailed_results, out_dir = fr.run_ablation_block(**standard_kwargs)

    idx = json.loads((out_dir / "run_index.json").read_text())

    assert idx["ablation"] == "standard"
    assert set(idx["experiments"]) == {"exp_a", "exp_b"}
    assert idx["tau_grid_ms"] == [400, 800]
    assert idx["train_tau_ms"] == 400
    assert idx["manifest_path"] == "/fake/manifest.csv"
    # Per-experiment summary exists with expected fields.
    for name in ("exp_a", "exp_b"):
        es = idx["experiment_summary"][name]
        assert es["fusion"] == "unimodal"
        assert es["streams"]
        assert es["n_params"] > 0
        assert es["epochs_run"] == 2
        # test_macro_f1_at_train_tau is plumbed from tau_results["400"]["macro_f1"].
        assert "test_macro_f1_at_train_tau" in es


def test_write_run_index_aggregates_ablations(mock_fl, standard_kwargs, tmp_path):
    """write_run_index produces a top-level rollup keyed by ablation name."""
    # Run two ablations into the same RUN_ROOT to exercise aggregation.
    run_root = tmp_path / "fusion_runs" / "manifest" / "20260101_000000"

    standard_kwargs["out_root"] = run_root
    standard_kwargs["ablation_name"] = "standard"
    standard_results, _ = fr.run_ablation_block(**standard_kwargs)

    standard_kwargs["ablation_name"] = "ssa"
    standard_kwargs["use_attention"] = True
    ssa_results, _ = fr.run_ablation_block(**standard_kwargs)

    out_path = fr.write_run_index(
        run_root=run_root,
        ablation_results={
            "standard": standard_results,
            "ssa": ssa_results,
        },
        manifest_path="/fake/manifest.csv",
        train_labels_path=str(standard_kwargs["train_labels_path"]),
        train_tau_ms=400,
        tau_grid_ms_by_ablation={"standard": (400, 800), "ssa": (400, 800)},
        modality_registry=standard_kwargs["modality_registry"],
        global_hyperparams={"hidden_size": 16, "seed": 42},
        device=torch.device("cpu"),
    )

    assert out_path == run_root / "index.json"
    idx = json.loads(out_path.read_text())
    assert set(idx["ablations"].keys()) == {"standard", "ssa"}
    for ablation_name in ("standard", "ssa"):
        block = idx["ablations"][ablation_name]
        assert set(block["experiments"]) == {"exp_a", "exp_b"}
        assert block["tau_grid_ms"] == [400, 800]
        assert "experiment_summary" in block


def test_results_json_written_incrementally(mock_fl, standard_kwargs, monkeypatch):
    """results.json must be written after EACH τ, not just at the end —
    that's the whole point of the refactor. Verify by counting json.dump
    calls inside fusion_runner."""
    n_writes = {"results": 0}

    real_dump = json.dump

    def counting_dump(obj, fp, *a, **kw):
        # Only count writes that target results.json (not partial/run_index/index).
        name = getattr(fp, "name", "")
        if name.endswith("results.json"):
            n_writes["results"] += 1
        return real_dump(obj, fp, *a, **kw)

    monkeypatch.setattr(fr.json, "dump", counting_dump)

    fr.run_ablation_block(**standard_kwargs)

    # 2 experiments × (2 τs in-loop + 1 final overwrite) = 6 writes minimum.
    # Allow ≥6 to absorb any future additional safety-net writes.
    assert n_writes["results"] >= 6, (
        f"expected at least 6 incremental writes, got {n_writes['results']}"
    )
