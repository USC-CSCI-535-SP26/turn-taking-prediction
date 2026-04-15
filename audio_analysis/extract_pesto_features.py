import os
import re
import pickle
import logging
import argparse
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torchaudio
import librosa
import pesto
from pesto import load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PestoFeatureExtractor:
    """
    Extract full-segment PESTO features from WAV files.

    Supports:
      - model loaded once and reused across files
      - frame-level outputs
      - optional interpolation to fixed output hop
      - optional per-second summaries
    """

    def __init__(
        self,
        target_sr: Optional[int] = None,
        step_size_ms: float = 10.0,
        align_hop_seconds: Optional[float] = None,
        use_gpu: bool = False,
        model_name: str = "mir-1k",
    ):
        """
        Parameters
        ----------
        target_sr : Optional[int]
            If provided, resample audio before inference.
        step_size_ms : float
            Desired PESTO step size in milliseconds.
        align_hop_seconds : Optional[float]
            If provided, interpolate frame outputs onto a fixed grid
            (e.g. 1/30 for 30 FPS video alignment).
        use_gpu : bool
            Use CUDA if available.
        model_name : str
            Pretrained model name for pesto.load_model().
        """
        self.target_sr = target_sr
        self.step_size_ms = step_size_ms
        self.align_hop_seconds = align_hop_seconds
        self.device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")

        logger.info(f"Loading PESTO model once on device={self.device}...")
        self.model = load_model(model_name, step_size=step_size_ms).to(self.device)
        self.model.eval()

    def load_audio(self, wav_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load WAV, convert to mono, optionally resample.
        Returns:
            x: 1D torch.Tensor
            sr: int
        """
        x, sr = torchaudio.load(wav_path)

        # Convert stereo/multi-channel to mono.
        # PESTO docs warn stereo otherwise gets treated as separate channels/batch.
        if x.ndim == 2:
            x = x.mean(dim=0)
        else:
            x = x.squeeze()

        x = x.float()

        if self.target_sr is not None and sr != self.target_sr:
            x_np = x.cpu().numpy()
            x_np = librosa.resample(x_np, orig_sr=sr, target_sr=self.target_sr)
            x = torch.from_numpy(x_np).float()
            sr = self.target_sr

        return x, sr

    def run_pesto(self, x: torch.Tensor, sr: int) -> Dict[str, Any]:
        """
        Run the preloaded PESTO model on a waveform.

        Based on the documented advanced API, the loaded model's forward path
        returns predictions, confidence, and activations.
        """
        x = x.to(self.device)

        with torch.no_grad():
            # Loaded model handles waveform -> pitch inference.
            # The docs show:
            #   predictions, confidence, activations = pesto_model(x, sr)
            predictions, confidence, activations = self.model(x, sr)

        predictions = self._to_numpy(predictions).astype(np.float32)
        confidence = self._to_numpy(confidence).astype(np.float32)
        activations = self._to_numpy(activations) if activations is not None else None

        # Build times from actual number of prediction steps
        n_frames = len(predictions)
        hop_s = self.step_size_ms / 1000.0
        times = np.arange(n_frames, dtype=np.float32) * hop_s

        frame_outputs = {
            "times": times,
            "f0_hz": predictions,
            "confidence": confidence,
            "activations": activations,
        }

        if self.align_hop_seconds is not None:
            duration_s = float(len(x) / sr)
            frame_outputs = self.interpolate_frame_outputs(
                frame_outputs=frame_outputs,
                duration_s=duration_s,
                hop_s=self.align_hop_seconds,
            )

        return frame_outputs

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _interp_1d(old_times: np.ndarray, values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        old_times = np.asarray(old_times, dtype=np.float32)
        new_times = np.asarray(new_times, dtype=np.float32)

        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.full_like(new_times, np.nan, dtype=np.float32)

        return np.interp(new_times, old_times[valid], values[valid]).astype(np.float32)

    def interpolate_frame_outputs(
        self,
        frame_outputs: Dict[str, Any],
        duration_s: float,
        hop_s: float,
    ) -> Dict[str, Any]:
        """
        Interpolate frame-level outputs to a fixed time grid.
        Useful for aligning with video frames.
        """
        new_times = np.arange(0, duration_s, hop_s, dtype=np.float32)
        old_times = frame_outputs["times"]

        out = {
            "times": new_times,
            "f0_hz": self._interp_1d(old_times, frame_outputs["f0_hz"], new_times),
            "confidence": self._interp_1d(old_times, frame_outputs["confidence"], new_times),
            # Keep activations at native resolution
            "activations": frame_outputs["activations"],
        }
        return out

    @staticmethod
    def _safe_stats(x: np.ndarray) -> List[float]:
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return [np.nan, np.nan, np.nan, np.nan]
        return [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
        ]

    def summarize_per_second(self, frame_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize frame-level outputs into one feature vector per second.
        """
        times = frame_outputs["times"]
        f0 = frame_outputs["f0_hz"]
        conf = frame_outputs["confidence"]

        if len(times) == 0:
            return {
                "times": np.zeros((0,), dtype=np.float32),
                "features": np.zeros((0, 10), dtype=np.float32),
                "feature_names": [
                    "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
                    "conf_mean", "conf_std", "conf_min", "conf_max",
                    "voiced_ratio", "n_valid_f0_frames",
                ],
            }

        duration_s = int(np.ceil(times[-1])) + 1
        second_times = []
        feature_rows = []

        for sec in range(duration_s):
            start_s = float(sec)
            end_s = float(sec + 1)
            mask = (times >= start_s) & (times < end_s)

            f0_sec = f0[mask]
            conf_sec = conf[mask]

            f0_stats = self._safe_stats(f0_sec)
            conf_stats = self._safe_stats(conf_sec)

            voiced_ratio = float(np.mean(np.isfinite(f0_sec))) if f0_sec.size > 0 else np.nan
            n_valid_f0 = float(np.sum(np.isfinite(f0_sec))) if f0_sec.size > 0 else 0.0

            row = np.array(f0_stats + conf_stats + [voiced_ratio, n_valid_f0], dtype=np.float32)
            second_times.append(start_s)
            feature_rows.append(row)

        return {
            "times": np.asarray(second_times, dtype=np.float32),
            "features": np.vstack(feature_rows).astype(np.float32),
            "feature_names": [
                "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
                "conf_mean", "conf_std", "conf_min", "conf_max",
                "voiced_ratio", "n_valid_f0_frames",
            ],
        }

    def extract_from_wav(
        self,
        wav_path: str,
        return_frame_level: bool = True,
        return_per_second: bool = True,
    ) -> Dict[str, Any]:
        x, sr = self.load_audio(wav_path)
        frame_outputs = self.run_pesto(x, sr)

        result = {
            "wav_path": wav_path,
            "sample_rate": sr,
            "duration_seconds": float(len(x) / sr),
            "n_samples": int(len(x)),
            "step_size_ms": self.step_size_ms,
            "align_hop_seconds": self.align_hop_seconds,
        }

        if return_frame_level:
            result["frame_level"] = frame_outputs

        if return_per_second:
            result["per_second"] = self.summarize_per_second(frame_outputs)

        return result


def parse_participant_from_dirname(dirname: str) -> Optional[str]:
    """
    participant_a_P0737 -> P0737
    participant_b_P1093 -> P1093
    """
    m = re.match(r"participant_[ab]_(P\d+)$", dirname)
    if m:
        return m.group(1)
    return None


def build_wav_path(session_dir: str, session: str, participant_dirname: str, participant_id: str) -> str:
    """
    Expected WAV path:
    {session_dir}/{participant_dirname}/{session}_{participant}.wav
    """
    return os.path.join(session_dir, participant_dirname, f"{session}_{participant_id}.wav")


def save_features(
    result: Dict[str, Any],
    output_root: str,
    session: str,
    participant_id: str,
) -> str:
    """
    Save to:
      ./pesto_features/{session}/{session}_{participant}.pkl
    """
    session_out_dir = os.path.join(output_root, session)
    os.makedirs(session_out_dir, exist_ok=True)

    save_path = os.path.join(session_out_dir, f"{session}_{participant_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"Saved features to: {save_path}")
    return save_path


def process_single_file(
    wav_path: str,
    output_path: str,
    extractor: PestoFeatureExtractor,
    return_frame_level: bool = True,
    return_per_second: bool = True,
) -> str:
    """
    Extract from one arbitrary WAV file and save to one arbitrary PKL path.
    """
    result = extractor.extract_from_wav(
        wav_path=wav_path,
        return_frame_level=return_frame_level,
        return_per_second=return_per_second,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"Saved single-file features to: {output_path}")
    return output_path


def process_session(
    session_dir: str,
    output_root: str,
    extractor: PestoFeatureExtractor,
    return_frame_level: bool = True,
    return_per_second: bool = True,
) -> List[str]:
    """
    Process one session directory:
      annotated_interactions_wav/{session}/participant_a_{id}/{session}_{id}.wav
      annotated_interactions_wav/{session}/participant_b_{id}/{session}_{id}.wav
    """
    session = os.path.basename(session_dir.rstrip("/"))
    saved_paths = []

    participant_dirs = [
        d for d in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, d)) and re.match(r"participant_[ab]_P\d+$", d)
    ]

    for participant_dirname in sorted(participant_dirs):
        participant_id = parse_participant_from_dirname(participant_dirname)
        if participant_id is None:
            logger.warning(f"Could not parse participant id from {participant_dirname}, skipping.")
            continue

        wav_path = build_wav_path(session_dir, session, participant_dirname, participant_id)

        if not os.path.exists(wav_path):
            logger.warning(f"Missing WAV file: {wav_path}")
            continue

        logger.info(f"Processing session={session}, participant={participant_id}")
        result = extractor.extract_from_wav(
            wav_path=wav_path,
            return_frame_level=return_frame_level,
            return_per_second=return_per_second,
        )

        save_path = save_features(
            result=result,
            output_root=output_root,
            session=session,
            participant_id=participant_id,
        )
        saved_paths.append(save_path)

    return saved_paths


def process_all_sessions(
    input_root: str,
    output_root: str,
    extractor: PestoFeatureExtractor,
    return_frame_level: bool = True,
    return_per_second: bool = True,
) -> List[str]:
    """
    Process all sessions under ./annotated_interactions_wav
    """
    saved_paths = []

    sessions = [
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ]

    for session in sorted(sessions):
        session_dir = os.path.join(input_root, session)
        logger.info(f"=== Processing session: {session} ===")
        session_saved = process_session(
            session_dir=session_dir,
            output_root=output_root,
            extractor=extractor,
            return_frame_level=return_frame_level,
            return_per_second=return_per_second,
        )
        saved_paths.extend(session_saved)

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all_sessions", "single_session", "single_file"],
        default="all_sessions",
        help="Run over all sessions, one session, or one file.",
    )

    parser.add_argument(
        "--input-root",
        type=str,
        default="./annotated_interactions_wav",
        help="Root directory containing all session folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./pesto_features",
        help="Root directory for saved feature PKLs.",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Path to one session directory when mode=single_session.",
    )

    parser.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Path to one WAV file when mode=single_file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Exact output PKL path when mode=single_file.",
    )

    parser.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Optional resampling rate before inference.",
    )
    parser.add_argument(
        "--step-size-ms",
        type=float,
        default=10.0,
        help="PESTO prediction step size in ms.",
    )
    parser.add_argument(
        "--align-hop-seconds",
        type=float,
        default=None,
        help="Optional fixed output hop in seconds for video alignment, e.g. 0.0333333.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--no-frame-level",
        action="store_true",
        help="Skip saving frame-level outputs.",
    )
    parser.add_argument(
        "--no-per-second",
        action="store_true",
        help="Skip saving per-second outputs.",
    )

    args = parser.parse_args()

    extractor = PestoFeatureExtractor(
        target_sr=args.target_sr,
        step_size_ms=args.step_size_ms,
        align_hop_seconds=args.align_hop_seconds,
        use_gpu=args.gpu,
        model_name="mir-1k",
    )

    return_frame_level = not args.no_frame_level
    return_per_second = not args.no_per_second

    if args.mode == "all_sessions":
        saved = process_all_sessions(
            input_root=args.input_root,
            output_root=args.output_root,
            extractor=extractor,
            return_frame_level=return_frame_level,
            return_per_second=return_per_second,
        )
        logger.info(f"Done. Saved {len(saved)} files.")

    elif args.mode == "single_session":
        if args.session_dir is None:
            raise ValueError("--session-dir is required when --mode single_session")
        saved = process_session(
            session_dir=args.session_dir,
            output_root=args.output_root,
            extractor=extractor,
            return_frame_level=return_frame_level,
            return_per_second=return_per_second,
        )
        logger.info(f"Done. Saved {len(saved)} files.")

    elif args.mode == "single_file":
        if args.wav is None or args.output_path is None:
            raise ValueError("--wav and --output-path are required when --mode single_file")
        process_single_file(
            wav_path=args.wav,
            output_path=args.output_path,
            extractor=extractor,
            return_frame_level=return_frame_level,
            return_per_second=return_per_second,
        )
        logger.info("Done.")


#Running Code
##ALL Sessions
# python extract_pesto.py \
#   --mode all_sessions \
#   --input-root ./annotated_interactions_wav \
#   --output-root ./pesto_features

##Single Sessions
# python extract_pesto.py \
#   --mode single_session \
#   --session-dir ./annotated_interactions_wav/V00_S1132_I00000333 \
#   --output-root ./pesto_features

##Single File
# python extract_pesto.py \
#   --mode single_file \
#   --wav ./annotated_interactions_wav/V00_S1132_I00000333/participant_a_P0737/V00_S1132_I00000333_P0737.wav \
#   --output-path ./pesto_features/test_p0737.pkl

##Alignment to video sampling
# python extract_pesto.py \
#   --mode all_sessions \
#   --input-root ./annotated_interactions_wav \
#   --output-root ./pesto_features \
#   --align-hop-seconds 0.0333333333 #default is 30fps