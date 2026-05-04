"""
Tier 2 — real-data smoke test for fusion_runner.run_ablation_block.

Runs all 5 ablations end-to-end with one experiment each, MAX_SAMPLES_PER_SPLIT=4,
1 epoch, 2-element τ-grid (single τ for coordination per the spec). Exercises the
full data → train → rebuild → per-τ eval → save flow that the mocked Tier 1
tests can't see.

Marked `smoke` so default `pytest` skips them. Run with `pytest -m smoke`.
~5 min on M5 Max.

Skipped if any required data path is missing — keeps a clean checkout (no
features) from failing the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import fusion_lib as fl  # noqa: E402  via conftest sys.path
import fusion_runner as fr  # noqa: E402


# Resolve repo paths — anchor on the repo root so tests are cwd-independent.
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent.parent

CPC_DIR = _REPO_ROOT / "turn_taking_analysis/subset/cpc"
OPENFACE_DIR = _REPO_ROOT / "turn_taking_analysis/subset/openface_windows"
LABELS_DIR = _REPO_ROOT / "turn_taking_analysis/model_input/labels"
MANIFEST_PATH = _REPO_ROOT / "turn_taking_analysis/manifests/manifest.csv"
COORDINATION_CSV_PATH = (
    _REPO_ROOT / "turn_taking_analysis/subset/coordination/openface/all_coordination_features.csv"
)
COORDINATION_CONTINUOUS_DIR = (
    _REPO_ROOT / "turn_taking_analysis/subset/coordination/openface_continuous_arrays"
)
TRAIN_LABELS_PATH = LABELS_DIR / "labels_tau_0400.json"

REQUIRED_PATHS = [
    CPC_DIR, OPENFACE_DIR, LABELS_DIR, MANIFEST_PATH,
    COORDINATION_CSV_PATH, COORDINATION_CONTINUOUS_DIR, TRAIN_LABELS_PATH,
]

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        any(not p.exists() for p in REQUIRED_PATHS),
        reason="smoke test requires real feature data; "
               f"missing: {[str(p) for p in REQUIRED_PATHS if not p.exists()]}",
    ),
]


# Minimal experiment per ablation — pick the simplest config in each block.
_SMOKE_SPECS = {
    "standard": {
        "exp": "cpc_both_neural_concat",
        "cfg": {
            "streams": [
                {"modality": "cpc", "role": "speaker"},
                {"modality": "cpc", "role": "listener"},
            ],
            "fusion": "neural_concat",
        },
        "use_coordination": False,
        "use_attention": False,
        "tau_grid": (400, 800),
        "coord_csv": None,
    },
    "ssa": {
        "exp": "cpc_both_attention",
        "cfg": {
            "streams": [
                {"modality": "cpc", "role": "speaker"},
                {"modality": "cpc", "role": "listener"},
            ],
            "fusion": "self_attention",
        },
        "use_coordination": False,
        "use_attention": True,
        "tau_grid": (400, 800),
        "coord_csv": None,
    },
    "sca": {
        "exp": "cpc_speaker_of_listener_cross_attention",
        "cfg": {
            "streams": [
                {"modality": "cpc", "role": "speaker"},
                {"modality": "openface", "role": "listener"},
            ],
            "fusion": "cross_attention",
        },
        "use_coordination": False,
        "use_attention": True,
        "tau_grid": (400, 800),
        "coord_csv": None,
    },
    "coordination": {
        "exp": "cpc_speaker_plus_coordination",
        "cfg": {
            "streams": [
                {"modality": "cpc", "role": "speaker"},
                {"modality": "coordination_openface", "role": "speaker"},
            ],
            "fusion": "neural_concat",
        },
        "use_coordination": True,
        "use_attention": False,
        "tau_grid": (400,),  # coord ablation pinned per the layout spec
        "coord_csv": COORDINATION_CSV_PATH,
    },
    "csa": {
        "exp": "cpc_speaker_coord_self_attention",
        "cfg": {
            "streams": [
                {"modality": "cpc", "role": "speaker"},
                {"modality": "coordination_openface_continuous", "role": "speaker"},
            ],
            "fusion": "self_attention",
            "attention_heads_by_modality": [4, 2],
        },
        "use_coordination": True,
        "use_attention": True,
        "tau_grid": (400, 800),
        "coord_csv": COORDINATION_CSV_PATH,
    },
}


@pytest.fixture(scope="module")
def modality_registry():
    return fl.make_modality_registry_coordination({
        "cpc": {
            "kind": "spliced",
            "dir": str(CPC_DIR),
            "feature_dim": 256,
            "frame_rate_hz": 100.0,
        },
        "openface": {
            "kind": "spliced",
            "dir": str(OPENFACE_DIR),
            "feature_dim": 23,
            "frame_rate_hz": 30.0,
        },
        "coordination_openface": {
            "kind": "coordination",
            "coordination_mode": "summary",
            "feature_dim": len(fl.COORDINATION_FEATURE_COLUMNS),
            "frame_rate_hz": 0.0,
        },
        "coordination_openface_continuous": {
            "kind": "coordination",
            "coordination_mode": "continuous",
            "dir": str(COORDINATION_CONTINUOUS_DIR),
            "feature_dim": 23,
            "frame_rate_hz": 0.0,
        },
    })


@pytest.fixture(scope="module")
def manifest_rows():
    return fl.load_manifest(str(MANIFEST_PATH))


def _assert_block_outputs(out_dir, exp_name, tau_grid):
    """Assert per-ablation block produced expected files + per-τ contents."""
    assert out_dir.is_dir(), out_dir
    assert (out_dir / "run_index.json").is_file()
    assert (out_dir / "tau_curve.png").is_file()
    assert (out_dir / "tau_curve.png").stat().st_size > 0

    exp_dir = out_dir / exp_name
    assert exp_dir.is_dir()
    assert (exp_dir / "results.json").is_file()
    assert (exp_dir / "model_state.pt").is_file()
    assert not (exp_dir / "partial.json").exists()

    res = json.loads((exp_dir / "results.json").read_text())
    assert res["experiment_name"] == exp_name
    assert set(res["tau_results"].keys()) == {str(t) for t in tau_grid}
    for tau in tau_grid:
        block = res["tau_results"][str(tau)]
        assert block["n_test_samples"] >= 0
        if block["n_test_samples"] > 0:
            assert "macro_f1" in block
            assert len(block["sample_records"]) == block["n_test_samples"]

    # Checkpoint round-trips.
    state = torch.load(exp_dir / "model_state.pt", map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0


def test_full_smoke(tmp_path, modality_registry, manifest_rows):
    """One pass through all 5 ablations + cross-ablation index. ~5 min."""
    run_root = tmp_path / "fusion_runs" / "manifest" / "smoke"
    common = dict(
        out_root=run_root,
        modality_registry=modality_registry,
        manifest_rows=manifest_rows,
        manifest_path=MANIFEST_PATH,
        train_labels_path=TRAIN_LABELS_PATH,
        labels_dir=LABELS_DIR,
        train_tau_ms=400,
        hidden_size=16,
        dropout=0.3,
        num_layers_early=2,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        patience=2,
        num_workers=0,
        seed=42,
        max_samples_per_split=4,
        device="cpu",
    )

    detailed_by_ablation = {}

    for ablation_name, spec in _SMOKE_SPECS.items():
        kwargs = dict(common)
        if spec["coord_csv"] is not None:
            kwargs["coordination_csv_path"] = spec["coord_csv"]

        detailed, out_dir = fr.run_ablation_block(
            ablation_name=ablation_name,
            experiments={spec["exp"]: spec["cfg"]},
            use_coordination=spec["use_coordination"],
            use_attention=spec["use_attention"],
            tau_grid_ms=spec["tau_grid"],
            **kwargs,
        )

        _assert_block_outputs(out_dir, spec["exp"], spec["tau_grid"])
        detailed_by_ablation[ablation_name] = detailed

    # Cross-ablation index.json.
    out_path = fr.write_run_index(
        run_root=run_root,
        ablation_results=detailed_by_ablation,
        manifest_path=MANIFEST_PATH,
        train_labels_path=TRAIN_LABELS_PATH,
        train_tau_ms=400,
        tau_grid_ms_by_ablation={n: s["tau_grid"] for n, s in _SMOKE_SPECS.items()},
        modality_registry=modality_registry,
        global_hyperparams={"hidden_size": 16, "seed": 42, "epochs": 1},
        device=fl.resolve_device("cpu"),
    )

    assert out_path == run_root / "index.json"
    idx = json.loads(out_path.read_text())
    assert set(idx["ablations"].keys()) == set(_SMOKE_SPECS.keys())
    for name, spec in _SMOKE_SPECS.items():
        block = idx["ablations"][name]
        assert spec["exp"] in block["experiments"]
        assert block["tau_grid_ms"] == list(spec["tau_grid"])

    # Negative assertion: nothing landed outside run_root.
    for p in tmp_path.rglob("*"):
        if p.is_file():
            try:
                p.relative_to(run_root)
            except ValueError:
                raise AssertionError(f"unexpected file outside run_root: {p}")
