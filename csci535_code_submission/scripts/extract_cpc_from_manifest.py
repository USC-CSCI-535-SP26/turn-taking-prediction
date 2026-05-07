from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
SHARED_SCRIPTS_DIR = THIS_FILE.parent
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (
    extract_interaction_id,
    extract_participant_id,
)

DEFAULT_HUB_REPO = "facebookresearch/CPC_audio"
DEFAULT_HUB_ENTRYPOINT = "CPC_audio"    

DEFAULT_FEATURE_TYPE = "context"   

DEFAULT_POOL_FACTOR = 1

DEFAULT_TARGET_RATE_HZ = 10.0

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 8
DEFAULT_WRITER_THREADS = 4
DEFAULT_PROGRESS_EVERY = 500

CPC_SAMPLE_RATE = 16_000           
CPC_SAMPLES_PER_FRAME = 160        
CPC_FEATURE_DIM = 256             

class ExtractTask:
    """One spliced wav to featurize: input path + destination paths + timing."""

    __slots__ = (
        "orig_stem", "interaction_id", "participant_id",
        "window_start_s", "window_end_s",
        "audio_path", "output_path", "sidecar_path",
    )

    def __init__(
        self,
        orig_stem: str,
        interaction_id: str,
        participant_id: str,
        window_start_s: float,
        window_end_s: float,
        audio_path: str,
        output_path: str,
        sidecar_path: str,
    ):
        self.orig_stem = orig_stem
        self.interaction_id = interaction_id
        self.participant_id = participant_id
        self.window_start_s = window_start_s
        self.window_end_s = window_end_s
        self.audio_path = audio_path
        self.output_path = output_path
        self.sidecar_path = sidecar_path

    def __repr__(self) -> str:
        return f"ExtractTask({Path(self.audio_path).name})"

def _parse_window_filename(stem: str):
    head, sep, orig = stem.partition("_")
    if not sep or not orig:
        return None
    start_str, dash, end_str = head.partition("-")
    if not dash:
        return None
    try:
        start_s = float(start_str)
        end_s = float(end_str)
    except ValueError:
        return None
    return start_s, end_s, orig

def build_tasks(spliced_dir: str, output_dir: str) -> list[ExtractTask]:
    spliced_root = Path(spliced_dir)
    tasks: list[ExtractTask] = []

    for sub in sorted(spliced_root.iterdir()):
        if not sub.is_dir():
            continue
        orig_stem = sub.name

        try:
            interaction_id = extract_interaction_id(orig_stem)
            participant_id = extract_participant_id(orig_stem)
        except Exception as e:
            print(f"  WARN: subdir {orig_stem!r} does not match the expected "
                  f"file-id pattern ({e}); skipping.", file=sys.stderr)
            continue

        out_sub = Path(output_dir) / orig_stem

        for wav in sorted(sub.glob("*.wav")):
            parsed = _parse_window_filename(wav.stem)
            if parsed is None:
                print(f"  WARN: could not parse window times from "
                      f"{wav.name!r}; skipping.", file=sys.stderr)
                continue
            window_start_s, window_end_s, _orig = parsed

            tasks.append(ExtractTask(
                orig_stem=orig_stem,
                interaction_id=interaction_id,
                participant_id=participant_id,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                audio_path=str(wav),
                output_path=str(out_sub / f"{wav.stem}.npy"),
                sidecar_path=str(out_sub / f"{wav.stem}.json"),
            ))

    return tasks

def pick_device(requested: str) -> str:
    """Resolve --device. 'auto' picks cuda > mps > cpu."""
    if requested != "auto":
        return requested

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_cpc(
    hub_repo: str,
    hub_entrypoint: str,
    cpc_repo: str | None,
    checkpoint: str | None,
    device: str,
):
    """Load a pretrained CPC model.

    Returns the model in eval mode on `device`.
    """
    import torch

    if cpc_repo and checkpoint:
        repo_path = Path(cpc_repo).resolve()
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"--cpc-repo {cpc_repo} is not a directory. Clone "
                f"{DEFAULT_HUB_REPO} first."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        print(f"  Loading CPC:       local repo {repo_path}")
        print(f"  Checkpoint:        {checkpoint}")

        import argparse as _argparse
        try:
            from cpc.model import CPCModel as _CPCModel
            from cpc.cpc_default_config import get_default_cpc_config
            from cpc.feature_loader import getEncoder, getAR, loadArgs
        except ImportError as e:
            raise RuntimeError(
                f"Could not import cpc.* helpers from {repo_path}. Is "
                f"{repo_path} actually a clone of {DEFAULT_HUB_REPO}? ({e})"
            ) from e

        try:
            ckpt_data = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to torch.load {checkpoint}: {e}"
            ) from e

        if not isinstance(ckpt_data, dict) or "config" not in ckpt_data \
                or "weights" not in ckpt_data:
            raise RuntimeError(
                f"Checkpoint {checkpoint} is not the release-format CPC "
                f"checkpoint (expected a dict with 'config' and 'weights' "
                f"keys). Got: {type(ckpt_data).__name__}"
                + (f", keys={list(ckpt_data.keys())}"
                   if isinstance(ckpt_data, dict) else "")
                + ". Download the canonical Rivière 2020 checkpoint from "
                  "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/"
                  "60k_epoch4-d0f474de.pt"
            )

        loc_args = get_default_cpc_config()
        loadArgs(loc_args, _argparse.Namespace(**ckpt_data["config"]))
        encoder_net = getEncoder(loc_args)
        ar_net = getAR(loc_args)
        model = _CPCModel(encoder_net, ar_net)

        model.load_state_dict(ckpt_data["weights"], strict=False)

        hidden_context_dim = int(getattr(loc_args, "hiddenGar", CPC_FEATURE_DIM))
        hidden_encoder_dim = int(getattr(loc_args, "hiddenEncoder", CPC_FEATURE_DIM))

    else:
        print(f"  Loading CPC:       torch.hub {hub_repo} :: {hub_entrypoint}")
        try:
            loaded = torch.hub.load(
                hub_repo, hub_entrypoint,
                pretrained=True, trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"torch.hub.load({hub_repo!r}, {hub_entrypoint!r}) failed: "
            ) from e

        if isinstance(loaded, tuple):
            model = loaded[0]
            hidden_context_dim = loaded[1] if len(loaded) > 1 else CPC_FEATURE_DIM
            hidden_encoder_dim = loaded[2] if len(loaded) > 2 else CPC_FEATURE_DIM
        else:
            model = loaded

            hidden_context_dim = CPC_FEATURE_DIM
            hidden_encoder_dim = CPC_FEATURE_DIM

    model.eval()
    model.to(device)

    if hidden_context_dim != CPC_FEATURE_DIM or hidden_encoder_dim != CPC_FEATURE_DIM:
        print(f"  NOTE: loaded CPC has context_dim={hidden_context_dim}, "
              f"encoder_dim={hidden_encoder_dim} (expected {CPC_FEATURE_DIM}).")

    print(f"  Device:            {device}")
    print(f"  CPC feature dim:   {hidden_context_dim}")
    return model, int(hidden_context_dim)

def _mean_pool(features, pool_factor: int):
    import torch

    t, d = features.shape
    keep = (t // pool_factor) * pool_factor
    if keep == 0:
        return torch.zeros((0, d), dtype=features.dtype, device=features.device)
    trimmed = features[:keep]
    return trimmed.reshape(keep // pool_factor, pool_factor, d).mean(dim=1)

def mean_pool_file(
    file: str,
    save_to: str,
    pool_to: float = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    import numpy as np
    import torch

    input_path = Path(file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input feature file not found: {file}")

    sidecar_path = input_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar not found at {sidecar_path}. mean_pool_file needs the "
            f"sidecar to read feature_rate_hz. Did you move the .npy without "
            f"its .json?"
        )

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    current_rate = sidecar.get("feature_rate_hz")
    if current_rate is None:
        raise ValueError(
            f"Sidecar {sidecar_path} is missing `feature_rate_hz`. Cannot "
            f"determine pool factor without knowing the source rate."
        )
    current_rate = float(current_rate)

    if pool_to <= 0:
        raise ValueError(f"pool_to must be positive; got {pool_to}")
    if pool_to > current_rate:
        raise ValueError(
            f"Target rate {pool_to} Hz exceeds source rate {current_rate} "
            f"Hz — mean_pool_file does not upsample."
        )

    pool_factor_float = current_rate / pool_to
    pool_factor = int(round(pool_factor_float))
    if abs(pool_factor_float - pool_factor) > 1e-6:
        raise ValueError(
            f"Non-integer pool factor: {current_rate} Hz / {pool_to} Hz = "
            f"{pool_factor_float}. Pick a pool_to that exactly divides the "
            f"source rate (e.g. 100→10, 100→20, 100→25, 100→50)."
        )
    if pool_factor < 1:
        raise ValueError(f"Computed pool_factor < 1 (={pool_factor}).")

    if input_path.suffix == ".npz":
        with np.load(input_path) as npz:
            keys = list(npz.keys())
            if not keys:
                raise ValueError(f"Empty .npz archive: {file}")

            arr = np.asarray(npz[keys[0]])
    else:
        arr = np.load(input_path)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D feature array (T, D) from {file}; got shape "
            f"{arr.shape}. mean_pool_file pools along the time axis only."
        )

    features_tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    pooled = _mean_pool(features_tensor, pool_factor).numpy().astype(np.float32)

    save_dir = Path(save_to)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_npy = save_dir / f"{input_path.stem}.npy"
    output_json = save_dir / f"{input_path.stem}.json"

    if output_npy.resolve() == input_path.resolve():
        raise ValueError(
            f"Refusing to overwrite the input feature file at {input_path}. "
            f"save_to ({save_to!r}) resolves to the same directory as the "
            f"input; pick a different save_to (e.g. 'subset/cpc_10hz')."
        )

    np.save(output_npy, pooled)

    out_sidecar = dict(sidecar)
    out_sidecar["pool_factor"] = int(sidecar.get("pool_factor", 1)) * pool_factor
    out_sidecar["feature_rate_hz"] = float(pool_to)
    out_sidecar["feature_frames"] = int(pooled.shape[0])
    out_sidecar["feature_dim"] = (int(pooled.shape[1]) if pooled.size
                                  else int(sidecar.get("feature_dim", 0)))

    out_sidecar["post_pool_source_file"] = str(input_path)
    out_sidecar["post_pool_source_rate_hz"] = current_rate
    out_sidecar["post_pool_factor"] = pool_factor

    with open(output_json, "w") as f:
        json.dump(out_sidecar, f, indent=2)

    return {
        "status": "ok",
        "input_file": str(input_path),
        "output_file": str(output_npy),
        "pool_factor": pool_factor,
        "input_frames": int(arr.shape[0]),
        "output_frames": int(pooled.shape[0]),
        "source_rate_hz": current_rate,
        "target_rate_hz": float(pool_to),
        "message": (
            f"pooled {arr.shape[0]:,}×{arr.shape[1]} @ {current_rate:g} Hz "
            f"-> {pooled.shape[0]:,}×{pooled.shape[1]} @ {pool_to:g} Hz "
            f"(factor={pool_factor})"
        ),
    }

def _build_sidecar(
    task: ExtractTask,
    pooled,
    orig_sr: int,
    duration_s: float,
    feature_type: str,
    feature_dim: int,
    pool_factor: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
) -> dict:
    return {
        "orig_stem":                task.orig_stem,
        "spliced_stem":             Path(task.audio_path).stem,
        "interaction_id":           task.interaction_id,
        "participant_id":           task.participant_id,
        "window_start_s":           task.window_start_s,
        "window_end_s":             task.window_end_s,
        "window_duration_s":        task.window_end_s - task.window_start_s,
        "audio_source":             task.audio_path,
        "source_sample_rate_hz":    int(orig_sr),
        "resampled_sample_rate_hz": CPC_SAMPLE_RATE,
        "duration_seconds":         duration_s,

        "model_family":             "cpc",
        "model_id":                 (f"local:{checkpoint}" if checkpoint
                                     else f"{hub_repo}::{hub_entrypoint}"),
        "cpc_hub_repo":             hub_repo,
        "cpc_hub_entrypoint":       hub_entrypoint,
        "cpc_checkpoint":           checkpoint,
        "cpc_feature_type":         feature_type,
        "cpc_samples_per_frame":    CPC_SAMPLES_PER_FRAME,
        "raw_feature_rate_hz":      CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME,
        "pool_factor":              pool_factor,
        "feature_rate_hz":          (CPC_SAMPLE_RATE
                                     / CPC_SAMPLES_PER_FRAME
                                     / pool_factor),
        "feature_dim":              int(pooled.shape[1]) if pooled.size else feature_dim,
        "feature_frames":           int(pooled.shape[0]),
        "dtype":                    str(pooled.dtype),

        "chunk_samples":            0,
        "chunk_length_s":           0.0,
    }

def _write_outputs(npy_path: str, json_path: str, pooled, sidecar: dict) -> None:
    import numpy as np

    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, pooled)
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)

def _checkpoints_equivalent(a: str | None, b: str | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    try:
        if os.path.samefile(a, b):
            return True
    except OSError:
        pass

    def _norm(p: str) -> str:
        return os.path.normcase(os.path.realpath(os.path.expanduser(p)))

    return _norm(a) == _norm(b)

def check_output_dir_params_compat(
    output_dir: str,
    feature_type: str,
    pool_factor: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
) -> str | None:
    root = Path(output_dir)
    sidecar: dict | None = None
    sampled_path: Path | None = None
    read_errors: list[tuple[Path, Exception]] = []

    for candidate in root.glob("*/*.json"):
        if not candidate.with_suffix(".npy").exists():
            continue

        try:
            with open(candidate) as f:
                parsed = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            read_errors.append((candidate, e))
            continue

        if not isinstance(parsed, dict):
            continue

        if parsed.get("model_family") != "cpc":
            continue
        sidecar = parsed
        sampled_path = candidate
        break

    if sidecar is None:
        if read_errors:
            first_path, first_err = read_errors[0]
            n = len(read_errors)
            print(
                f"WARNING: found {n} existing sidecar-shaped .json "
                f"file(s) under {output_dir} but could not parse any "
                f"(first error: {first_path}: {first_err}); proceeding "
                f"without parameter-compatibility verification. Pass "
                f"--overwrite if you want to discard prior output and "
                f"start clean.",
                file=sys.stderr,
                flush=True,
            )
        return None

    string_checks = (
        ("cpc_feature_type",   sidecar.get("cpc_feature_type"),   feature_type),
        ("pool_factor",        sidecar.get("pool_factor"),        pool_factor),
        ("cpc_hub_repo",       sidecar.get("cpc_hub_repo"),       hub_repo),
        ("cpc_hub_entrypoint", sidecar.get("cpc_hub_entrypoint"), hub_entrypoint),
    )
    mismatches = [
        f"{name}: sidecar has {existing!r}, current run requests {current!r}"
        for name, existing, current in string_checks
        if existing != current
    ]

    existing_ckpt = sidecar.get("cpc_checkpoint")
    if not _checkpoints_equivalent(existing_ckpt, checkpoint):
        mismatches.append(
            f"cpc_checkpoint: sidecar has {existing_ckpt!r}, "
            f"current run requests {checkpoint!r}"
        )

    if not mismatches:
        return None

    bullet_list = "\n".join(f"           - {m}" for m in mismatches)
    return (
        f"existing sidecar {sampled_path} disagrees with the current "
        f"run's parameters:\n"
        f"{bullet_list}\n"
        f"       Mixing these in one output directory would produce an "
        f"inconsistent corpus (already_have would silently skip the "
        f"old-params files). Either:\n"
        f"         - use a different --output-dir for the new params, or\n"
        f"         - pass --overwrite to discard the prior extraction."
    )

class _SplicedWavDataset:
    def __init__(self, tasks: list[ExtractTask]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int):
        import torchaudio

        task = self.tasks[idx]

        if not os.path.exists(task.audio_path):
            return (None, None, idx,
                    f"input wav not on disk: {task.audio_path}",
                    "missing")

        try:
            waveform, sr = torchaudio.load(task.audio_path)
            if sr != CPC_SAMPLE_RATE:
                cache = getattr(self, "_resamplers", None)
                if cache is None:
                    cache = {}
                    self._resamplers = cache
                resampler = cache.get(sr)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=CPC_SAMPLE_RATE
                    )
                    cache[sr] = resampler
                waveform = resampler(waveform)
            return waveform.squeeze(0), sr, idx, None, None
        except Exception as e:
            return None, None, idx, f"{type(e).__name__}: {e}", "failed"

def _collate_batch(items):
    import torch

    ok_waves, ok_srs, ok_idxs, orig_lengths = [], [], [], []
    failures = []
    for wav, sr, idx, err, status in items:
        if wav is None:
            failures.append((idx, err, status))
        else:
            ok_waves.append(wav)
            ok_srs.append(sr)
            ok_idxs.append(idx)
            orig_lengths.append(int(wav.shape[0]))

    if not ok_waves:
        return None, [], [], [], failures

    length_counts = Counter(orig_lengths)
    if len(length_counts) == 1:
        batch = torch.stack(ok_waves, dim=0)
        return batch, ok_srs, ok_idxs, orig_lengths, failures

    dominant_length, _count = max(
        length_counts.items(),
        key=lambda item: (item[1], item[0]),
    )
    kept_waves, kept_srs, kept_idxs, kept_lengths = [], [], [], []
    for wav, sr, idx, length in zip(ok_waves, ok_srs, ok_idxs, orig_lengths):
        if length == dominant_length:
            kept_waves.append(wav)
            kept_srs.append(sr)
            kept_idxs.append(idx)
            kept_lengths.append(length)
        else:
            failures.append((
                idx,
                (f"length mismatch: {length} samples, batch dominant "
                 f"length is {dominant_length}; dropped to avoid "
                 f"truncating batchmates"),
                "failed",
            ))

    batch = torch.stack(kept_waves, dim=0)
    return batch, kept_srs, kept_idxs, kept_lengths, failures

def already_have(output_path: str, sidecar_path: str) -> bool:
    for p in (output_path, sidecar_path):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
    try:
        with open(sidecar_path) as f:
            json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return True

def _batched_forward_on_device(batch_tensor, model, device: str, feature_type: str):
    """Run CPC forward over a stacked (B, T) batch; return (B, T_frames, D)
    on CPU. Raises RuntimeError on forward failure — caller decides
    whether to fall back to per-sample retries.
    """
    import torch

    x = batch_tensor.to(device=device, dtype=torch.float32).unsqueeze(1)

    with torch.inference_mode():
        out = model(x, None)
    if isinstance(out, tuple):
        c_feature = out[0]
        z_feature = out[1] if len(out) > 1 else out[0]
    else:
        c_feature = z_feature = out
    chosen = c_feature if feature_type == "context" else z_feature
    return chosen.to(torch.float32).cpu()

def _per_sample_fallback_forward(
    batch_tensor,
    model,
    device: str,
    feature_type: str,
    batched_error: Exception,
):
    import torch

    batched_is_oom = "out of memory" in str(batched_error).lower()
    outputs = []
    for i in range(batch_tensor.shape[0]):
        single = batch_tensor[i:i + 1].to(
            device=device, dtype=torch.float32
        ).unsqueeze(1)

        try:
            with torch.inference_mode():
                out = model(single, None)
            if isinstance(out, tuple):
                c_feature = out[0]
                z_feature = out[1] if len(out) > 1 else out[0]
            else:
                c_feature = z_feature = out
            chosen = c_feature if feature_type == "context" else z_feature
            features = chosen.squeeze(0).to(torch.float32).cpu()
            outputs.append((features, None))
        except Exception as e:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            is_oom = "out of memory" in str(e).lower()
            batch_note = (
                f" (batched forward failed with: "
                f"{type(batched_error).__name__}: {batched_error})"
            )
            hint = ""
            if is_oom and batched_is_oom:
                hint = (" — OOM even at B=1; try a different --device "
                        "or shrink the spliced window length")
            elif not is_oom:
                hint = (" — deterministic failure on this sample; "
                        "check input and model compatibility")
            outputs.append((
                None,
                f"CPC forward failed on isolated sample "
                f"({type(e).__name__}: {e}){hint}.{batch_note}",
            ))
    return outputs

def run_tasks(
    tasks: list[ExtractTask],
    model,
    device: str,
    feature_type: str,
    feature_dim: int,
    pool_factor: int,
    hub_repo: str,
    hub_entrypoint: str,
    checkpoint: str | None,
    overwrite: bool,
    dry_run: bool,
    batch_size: int,
    num_workers: int,
    writer_threads: int,
    progress_every: int,
) -> list[dict]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    n_total = len(tasks)
    results: list[dict | None] = [None] * n_total

    todo_tasks: list[ExtractTask] = []
    todo_to_global: list[int] = []

    for g_idx, task in enumerate(tasks):
        if not overwrite and already_have(task.output_path, task.sidecar_path):
            results[g_idx] = {
                "task": task, "status": "skipped",
                "message": f"already extracted "
                           f"({os.path.getsize(task.output_path):,} bytes)",
            }
        elif dry_run:
            results[g_idx] = {
                "task": task, "status": "dry",
                "message": f"DRY: {task.audio_path} -> {task.output_path}",
            }
        else:
            todo_to_global.append(g_idx)
            todo_tasks.append(task)

    n_to_do = len(todo_tasks)
    if n_to_do == 0:
        _print_results(results, verbose=True)
        return results

    print(f"  dispatching {n_to_do:,} new tasks "
          f"(already-done: {n_total - n_to_do:,})")

    ds = _SplicedWavDataset(todo_tasks)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_batch,
        pin_memory=False,
        shuffle=False,
        persistent_workers=(num_workers > 0),
    )

    writer = concurrent.futures.ThreadPoolExecutor(
        max_workers=writer_threads,
        thread_name_prefix="cpc-write",
    )

    pending: list[tuple] = []

    done = 0
    last_progress = 0
    start_time = time.time()

    n_fallback_batches = 0
    FALLBACK_LOG_CAP = 5

    def _reap_completed(force: bool) -> None:
        """Move completed futures into results[]. Free pooled arrays."""
        nonlocal pending
        still = []
        for fut, g_idx, task, pooled, duration_s in pending:
            if fut.done() or force:
                try:
                    fut.result()
                    results[g_idx] = {
                        "task": task, "status": "ok",
                        "message": f"OK ({pooled.shape[0]:,}×{pooled.shape[1]}, "
                                   f"{duration_s:.1f}s -> {pooled.nbytes:,} bytes)",
                    }
                except Exception as e:
                    results[g_idx] = {
                        "task": task, "status": "failed",
                        "message": f"save failed: {e}",
                    }
            else:
                still.append((fut, g_idx, task, pooled, duration_s))
        pending = still

    def _queue_sample_write(
        features,
        task: ExtractTask,
        g_idx: int,
        orig_length: int,
        orig_sr: int,
    ) -> None:
        nonlocal done
        if not torch.isfinite(features).all():
            nan_count = int(torch.isnan(features).sum())
            inf_count = int(torch.isinf(features).sum())
            results[g_idx] = {
                "task": task, "status": "failed",
                "message": (f"non-finite features: nan={nan_count:,} "
                            f"inf={inf_count:,} of {features.numel():,}"),
            }
            done += 1
            return

        pooled = _mean_pool(features, pool_factor).numpy().astype(np.float32)
        duration_s = orig_length / CPC_SAMPLE_RATE
        sidecar = _build_sidecar(
            task=task,
            pooled=pooled,
            orig_sr=orig_sr,
            duration_s=duration_s,
            feature_type=feature_type,
            feature_dim=feature_dim,
            pool_factor=pool_factor,
            hub_repo=hub_repo,
            hub_entrypoint=hub_entrypoint,
            checkpoint=checkpoint,
        )
        fut = writer.submit(
            _write_outputs,
            task.output_path, task.sidecar_path, pooled, sidecar,
        )
        pending.append((fut, g_idx, task, pooled, duration_s))
        done += 1

    try:
        for batch in loader:
            (batch_tensor, ok_srs, ok_local_idxs,
             orig_lengths, failures) = batch

            for local_idx, err_msg, status in failures:
                g_idx = todo_to_global[local_idx]
                results[g_idx] = {
                    "task": todo_tasks[local_idx],
                    "status": status,
                    "message": err_msg,
                }
                done += 1

            if batch_tensor is None:
                continue

            batched_chosen_cpu = None
            per_sample_outputs = None
            try:
                batched_chosen_cpu = _batched_forward_on_device(
                    batch_tensor, model, device, feature_type,
                )
            except RuntimeError as batched_error:
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                n_fallback_batches += 1

                if n_fallback_batches <= FALLBACK_LOG_CAP:
                    is_oom = "out of memory" in str(batched_error).lower()
                    oom_hint = (" — reduce --batch-size if this is OOM"
                                if is_oom else "")
                    print(
                        f"  [WARN] batched forward failed on batch "
                        f"#{n_fallback_batches} "
                        f"({type(batched_error).__name__}: {batched_error}); "
                        f"falling back to per-sample retry{oom_hint}",
                        file=sys.stderr, flush=True,
                    )
                    if n_fallback_batches == FALLBACK_LOG_CAP:
                        print(
                            f"  [WARN] further batched-forward fallbacks "
                            f"will be suppressed; total reported in summary.",
                            file=sys.stderr, flush=True,
                        )
                per_sample_outputs = _per_sample_fallback_forward(
                    batch_tensor, model, device, feature_type, batched_error,
                )

            for i, local_idx in enumerate(ok_local_idxs):
                task = todo_tasks[local_idx]
                g_idx = todo_to_global[local_idx]

                if batched_chosen_cpu is not None:
                    features = batched_chosen_cpu[i]
                else:
                    features, err_msg = per_sample_outputs[i]
                    if err_msg is not None:
                        results[g_idx] = {
                            "task": task, "status": "failed",
                            "message": err_msg,
                        }
                        done += 1
                        continue

                _queue_sample_write(
                    features,
                    task,
                    g_idx,
                    orig_lengths[i],
                    ok_srs[i],
                )

            _reap_completed(force=False)

            if progress_every > 0 and done - last_progress >= progress_every:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_s = (n_to_do - done) / rate if rate > 0 else 0.0
                print(f"  progress: {done:,} / {n_to_do:,} "
                      f"({100.0 * done / n_to_do:5.1f}%)  "
                      f"rate: {rate:6.1f}/s  "
                      f"ETA: {eta_s / 60:5.1f} min")
                last_progress = done
    finally:
        _reap_completed(force=True)
        writer.shutdown(wait=True)

        for g_idx in range(n_total):
            if results[g_idx] is None:
                results[g_idx] = {
                    "task": tasks[g_idx], "status": "failed",
                    "message": ("task not processed before run terminated "
                                "(Ctrl-C or loader error); rerun to retry"),
                }

    if n_fallback_batches > 0:
        print(
            f"\n  [NOTE] {n_fallback_batches:,} batch(es) fell back to "
            f"per-sample forward after a batched-forward failure. "
            f"Consider --batch-size smaller next run (the current "
            f"batched-mode throughput claim does not apply to the "
            f"fallback path).",
            file=sys.stderr,
        )

    _print_results(results, verbose=(n_total < 10_000))
    return results

def _print_results(results: list, verbose: bool) -> None:
    for r in results:
        if r is None:
            continue
        if verbose or r["status"] != "ok":
            print(f"  [{r['status']:<7}] {r['task'].orig_stem}/"
                  f"{Path(r['task'].audio_path).name}  {r['message']}")

def print_summary(results: list, elapsed_s: float, dry_run: bool) -> None:
    counts: dict[str, int] = {}
    bytes_written = 0
    for r in results:
        if r is None:
            counts["failed"] = counts.get("failed", 0) + 1
            continue
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "ok":
            try:
                bytes_written += int(
                    r["message"].split("->")[1]
                    .split(" bytes")[0]
                    .strip()
                    .replace(",", "")
                )
            except (IndexError, ValueError):
                pass

    print("\n" + "=" * 64)
    print("CPC EXTRACTION SUMMARY" + ("  [DRY-RUN]" if dry_run else ""))
    print("=" * 64)
    print(f"  Total tasks:  {len(results)}")
    for status in ("ok", "skipped", "dry", "missing", "failed"):
        n = counts.get(status, 0)
        if n:
            print(f"  {status:<10} {n}")
    if bytes_written:
        mib = bytes_written / (1024 * 1024)
        print(f"  Written:      {bytes_written:,} bytes ({mib:,.1f} MiB)")
    print(f"  Elapsed:      {elapsed_s:,.1f} s")
    print("=" * 64)

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Rivière-2020 CPC features (256-dim @ 100 Hz native) "
            "from every spliced wav produced by splice_wavs.py using "
            "batched inference. Saves native 100 Hz by default; pass "
            "--mean-pool (or --pool-factor N) to pool during extraction, "
            "or leave default and invoke mean_pool_file() post-hoc. "
            "Mirrors the input tree under --output-dir: <orig_stem>/"
            "<spliced_stem>.npy + <spliced_stem>.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--spliced-dir", required=True,
        help="REQUIRED. Parent directory containing splice_wavs.py output. "
             "Expected layout: <spliced-dir>/<orig_stem>/*.wav where each "
             "subdir is named after the original wav stem and holds one .wav "
             "per window. Must already exist.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="REQUIRED. Directory to save .npy / .json feature files. "
             "Mirrors the --spliced-dir tree exactly: <output-dir>/"
             "<orig_stem>/<spliced_stem>.npy + .json. Must already exist on "
             "disk (the script will not auto-create it — this is a footgun "
             "guard against typos that would otherwise silently land features "
             "in an unexpected location). Create it explicitly first: "
             "`mkdir -p subset/cpc_100hz` etc.",
    )

    parser.add_argument(
        "--hub-repo", default=DEFAULT_HUB_REPO,
        help="torch.hub repo spec. Default: %(default)s",
    )
    parser.add_argument(
        "--hub-entrypoint", default=DEFAULT_HUB_ENTRYPOINT,
        help="torch.hub entrypoint name. Override if upstream renames it. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--cpc-repo", default=None,
        help="Path to a local clone of facebookresearch/CPC_audio. "
             "Overrides torch.hub when combined with --checkpoint. "
             "Use this on airgapped / no-GitHub environments.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a pretrained CPC checkpoint (.pt). Required when "
             "--cpc-repo is set. Use the canonical Rivière 2020 checkpoint "
             "from the facebookresearch/CPC_audio README.",
    )

    parser.add_argument(
        "--feature-type", choices=["context", "encoded"],
        default=DEFAULT_FEATURE_TYPE,
        help=("Which CPC stream to save. 'context' (default) = aggregator "
              "output c_t (causal RNN output — the standard VAP-compatible "
              "feature). 'encoded' = CNN-encoder output z_t (local-in-time, "
              "no RNN; used in some ablations and in ZeroSpeech-style "
              "phonetic probing)."),
    )

    _native_rate = CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME
    _mean_pool_factor = int(round(_native_rate / DEFAULT_TARGET_RATE_HZ))
    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument(
        "--mean-pool", action="store_true",
        help=f"Mean-pool during extraction to {DEFAULT_TARGET_RATE_HZ:g} Hz "
             f"(shortcut for --pool-factor {_mean_pool_factor}). Off by "
             f"default — the recommended flow is to leave this off, save "
             f"native {_native_rate} Hz features, and call mean_pool_file() "
             f"post-hoc to produce a pooled corpus in a separate directory.",
    )
    pool_group.add_argument(
        "--pool-factor", type=int, default=DEFAULT_POOL_FACTOR,
        help="Advanced: explicit integer mean-pool factor along time. "
             "Default %(default)s = no pooling (save native 100 Hz). "
             "Use 10 to produce 10 Hz (same as --mean-pool); use 2 for the "
             "50 Hz rate VAP uses internally.",
    )

    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Number of spliced wavs to run through CPC in one forward "
             "pass. Default %(default)s (tuned for M5 Max 32-core GPU). "
             "Drop to 1 for sequential/debug behavior; raise on larger "
             "CUDA GPUs. Memory scales linearly with B.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes for audio load + 48->16 kHz "
             "resample. Default %(default)s. Set to 0 to run everything "
             "in the main process (useful for debugging, slower in "
             "practice).",
    )
    parser.add_argument(
        "--writer-threads", type=int, default=DEFAULT_WRITER_THREADS,
        help="Threads in the .npy + .json background writer pool. "
             "Default %(default)s. Disk I/O on internal SSD rarely "
             "bottlenecks so this doesn't need tuning unless you're "
             "writing to a slow/networked drive.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress/rate/ETA line every N tasks. Default "
             "%(default)s. Set to 0 to silence progress lines.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device. 'auto' picks cuda > mps > cpu. Default: auto.",
    )

    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract files that already have .npy + .json on disk. "
             "Default: skip-if-present so a killed run resumes cleanly.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate inputs/outputs without loading the model.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.spliced_dir):
        print(f"ERROR: --spliced-dir is not an existing directory: "
              f"{args.spliced_dir}", file=sys.stderr)
        return 1

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: --output-dir is not an existing directory: "
              f"{args.output_dir}\n"
              f"       This script intentionally does NOT auto-create the "
              f"output directory, to avoid writing features to an unintended "
              f"location on typos. Create it explicitly first:\n"
              f"           mkdir -p {args.output_dir!r}\n"
              f"       and re-run.", file=sys.stderr)
        return 1

    if args.batch_size < 1:
        print(f"ERROR: --batch-size must be >= 1; got {args.batch_size}",
              file=sys.stderr)
        return 1
    if args.num_workers < 0:
        print(f"ERROR: --num-workers must be >= 0; got {args.num_workers}",
              file=sys.stderr)
        return 1

    if args.pool_factor < 1:
        print(f"ERROR: --pool-factor must be >= 1; got {args.pool_factor}",
              file=sys.stderr)
        return 1

    if bool(args.cpc_repo) != bool(args.checkpoint):
        print("ERROR: --cpc-repo and --checkpoint must be set together "
              "(local-load path) or neither (torch.hub path).",
              file=sys.stderr)
        return 1

    print(f"Scanning spliced dir: {args.spliced_dir}")
    tasks = build_tasks(args.spliced_dir, args.output_dir)
    n_subdirs = len({t.orig_stem for t in tasks})
    print(f"  {len(tasks):,} spliced wavs across {n_subdirs:,} original wavs")

    if not tasks:
        print("Nothing to extract. Exiting.")
        return 0

    if args.mean_pool:
        effective_pool_factor = int(round(
            (CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME) / DEFAULT_TARGET_RATE_HZ
        ))
    else:
        effective_pool_factor = args.pool_factor

    native_rate = CPC_SAMPLE_RATE // CPC_SAMPLES_PER_FRAME
    out_rate = native_rate / effective_pool_factor

    print(f"Output directory:  {args.output_dir}")
    print(f"Extraction config: feature_type={args.feature_type}  "
          f"pool_factor={effective_pool_factor} "
          f"({native_rate} Hz -> {out_rate:g} Hz)  "
          f"batch={args.batch_size}  workers={args.num_workers}  "
          f"writers={args.writer_threads}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}"
          f"{'  [OVERWRITE]' if args.overwrite else ''}")

    if not args.overwrite:
        compat_err = check_output_dir_params_compat(
            output_dir=args.output_dir,
            feature_type=args.feature_type,
            pool_factor=effective_pool_factor,
            hub_repo=args.hub_repo,
            hub_entrypoint=args.hub_entrypoint,
            checkpoint=args.checkpoint,
        )
        if compat_err:
            print(f"ERROR: {compat_err}", file=sys.stderr)
            return 1

    model = None
    device = "cpu"
    feature_dim = CPC_FEATURE_DIM
    if not args.dry_run:
        device = pick_device(args.device)
        try:
            model, feature_dim = load_cpc(
                hub_repo=args.hub_repo,
                hub_entrypoint=args.hub_entrypoint,
                cpc_repo=args.cpc_repo,
                checkpoint=args.checkpoint,
                device=device,
            )
        except Exception as e:
            print(f"ERROR: model load failed: {e}", file=sys.stderr)
            return 1

    print(f"Starting {len(tasks):,} extraction task(s)...\n")

    start = time.time()
    results = run_tasks(
        tasks=tasks,
        model=model,
        device=device,
        feature_type=args.feature_type,
        feature_dim=feature_dim,
        pool_factor=effective_pool_factor,
        hub_repo=args.hub_repo,
        hub_entrypoint=args.hub_entrypoint,
        checkpoint=args.checkpoint,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        writer_threads=args.writer_threads,
        progress_every=args.progress_every,
    )
    elapsed = time.time() - start

    print_summary(results, elapsed, args.dry_run)

    n_failed = sum(1 for r in results
                   if r is None or r["status"] == "failed")
    n_missing = sum(1 for r in results
                    if r is not None and r["status"] == "missing")
    if n_failed:
        return 2
    if n_missing:
        return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
