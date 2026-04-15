import os
import json
import pickle
import torchaudio
import torch
import tqdm
import json, pickle
import torch, librosa
import logging, argparse
import whisper, opensmile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from moshi.models.loaders import CheckpointInfo
from transformers import AutoFeatureExtractor
import re

from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##############################
# Embedding Extractor Classes
##############################

class EmbeddingExtractor:
    """Base class for embedding extraction"""

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError


class WhisperEmbeddingExtractor(EmbeddingExtractor):
    """Extract Whisper encoder embeddings for semantic content"""

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        self.model_name = model_name

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract Whisper encoder embeddings"""
        # Ensure audio is right length and sample rate
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Pad or truncate to 30 seconds (Whisper's max)
        target_length = 30 * 16000
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # Get mel spectrogram and extract features
        # Use the model's mel_filters to ensure correct dimensions
        mel = whisper.log_mel_spectrogram(
            torch.from_numpy(audio).float(), n_mels=self.model.dims.n_mels
        )

        with torch.no_grad():
            # Ensure mel tensor is on the same device as the model
            mel = mel.to(self.device)
            features = self.model.encoder(mel.unsqueeze(0))
            # Average over time dimension to get a single embedding #[C, T, D] 1,D
            embedding = features.mean(dim=1).squeeze().cpu().numpy()

        return embedding

class MimiEmbeddingExtractor(EmbeddingExtractor):
    """Extract Mimi encoder embeddings for semantic content"""

    def __init__(self, device: str = "cpu"):
        self.device = device

        # One-time: load model
        checkpoint = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        self.model = checkpoint.get_mimi(device=self.device)
        self.model.eval()

        # One-time: load feature extractor (only if you actually use it)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")


    def extract(self, audio: np.ndarray, sr: int = 24000) -> np.ndarray:
        """
        Extract Mimi embeddings from a specific waveform. {user}.wav
        """
        
        # Ensure float32 + mono
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)


        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000) # target_sr = 24000
            sr = 24000

        audio_tensor = torch.from_numpy(audio)[None, None, :].to(self.device)

        with torch.no_grad():
            features = self.model.encode_to_latent(audio_tensor, quantize=False)

        if features.ndim != 3:
            raise ValueError(f"Unexpected feats shape: {tuple(features.shape)}")

        # feats is either (B, T, D) or (B, D, T)
        if features.shape[1] == 512:
            # (B, D, T): pool over T, this seems to be the case for the mimi embeddings [B, D, T]
            embedding = features.mean(dim=2).squeeze(0)
        elif features.shape[2] == 512:
            # (B, T, D): pool over T
            embedding = features.mean(dim=1).squeeze(0)
        else:
            raise ValueError(f"Neither axis looks like hidden_size=512. feats shape: {tuple(features.shape)}")

        return embedding.detach().cpu().numpy()
        


if __name__ == "__main__":

    INTERACTION_DIR_BASE = '/Users/kaitlinzareno/Downloads/csci535_project/csci535-project/annotated_interactions_wav'
    SAVE_DIR = "/Users/kaitlinzareno/Downloads/csci535_project/deep_embeddings/"
    wav_turn_folder = 'wavs_by_turn'
    pattern = re.compile(r"utterance_\d+_(\d+\.\d+)s-(\d+\.\d+)s\.wav")
    os.makedirs(SAVE_DIR, exist_ok=True)

    #Note: sampling rate of wav files for this dataset is 4800

    DEVICE = 'cpu'
    EXPECTED_SAMPLE_RATE = 16000


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="sampling rate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/data2/zareno/nobu-shared/deep_embeds",
        help="Output directory for results",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["whisper", "mimi",],
        default="whisper",
        help="Embedding extractor: whisper or mimi",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size for audio embeddings",
    )
    parser.add_argument(
        "--user_type",
        type=str,
        default="child",
        help="Whisper model size for audio embeddings",
    )

    args = parser.parse_args()

    if args.extractor == "whisper":
        extractor = WhisperEmbeddingExtractor(model_name=args.whisper_model)
    elif args.extractor == "mimi":
        extractor = MimiEmbeddingExtractor()
    else:  # prosodic
        extractor = ProsodicEmbeddingExtractor()


    sessions = [d for d in os.listdir(INTERACTION_DIR_BASE) if os.path.isdir(os.path.join(INTERACTION_DIR_BASE, d))]


    for session_id in sessions: # For each session
        logger.info(f"📂 Processing session: {session_id}")
        
        session_path = os.path.join(INTERACTION_DIR_BASE, session_id)

        participants = [d for d in os.listdir(session_path) if 'participant' in d and os.path.isdir(os.path.join(session_path, d))]        
        deep_embeds = {}

        for participant in participants: # For participant, extract embedding per turn
            logger.info(f"📂 Processing participant: {participant}")
            participant_deep_embeds = {}

            participant_turn_path = os.path.join(session_path, participant,  wav_turn_folder)
            participant_audio_files = sorted([f for f in os.listdir(participant_turn_path) if f.endswith('.wav')]) #get all turn _audio_files for the session 

            for audio_file in tqdm.tqdm(participant_audio_files, desc=f'Processing turns for participant: {participant}'): # For each audio file (1 per speaker)
                # ex, utterance_001_0.00s-33.44s.wav
                filename = os.path.basename(audio_file)  
                
                # Remove .wav and rename utterance to turn
                turn_key = filename.replace(".wav", "").replace("utterance", "turn")
                
                speaker = participant
                
                audio_path = os.path.join(participant_turn_path, audio_file)
                audio_data, sr = torchaudio.load(audio_path)
                audio_data = audio_data.squeeze(0).numpy()
                # audio_data, _ = librosa.load(audio_path, sr=args.sr, mono=False)

                # Get deep embedding + save per speaker
                deep_embed = extractor.extract(audio_data)

                if deep_embed is None or np.ndim(deep_embed) != 1:
                    raise ValueError(f"Bad embedding for {session_id}/{particpant}/{turn_key}: shape={getattr(deep_embed,'shape',None)}")

                participant_deep_embeds[turn_key] = deep_embed

            deep_embeds[participant] = participant_deep_embeds
        
        session_save_path = os.path.join(SAVE_DIR, session_id)
        os.makedirs(session_save_path, exist_ok=True)

        embeddings_file = os.path.join(session_save_path, f"deep_embeds_{args.extractor}.pkl")
        pickle.dump(deep_embeds, open(embeddings_file, "wb"))
        logger.info(f"embeddings saved to: {embeddings_file}")

