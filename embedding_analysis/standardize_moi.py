import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch


# ============================================================
# Config
# ============================================================

DEEP_EMB_ROOT = Path("/Users/kaitlinzareno/Downloads/csci535_project/deep_embeddings")
ANNOTATION_ROOT = Path("/Users/kaitlinzareno/Downloads/csci535_project/csci535-project/annotated_interactions_wav/")
WAV_ROOT = Path("/Users/kaitlinzareno/Downloads/csci535_project/csci535-project/annotated_interactions_wav")

MODEL = "whisper"

# If True, only keep events that have pre, during, and post
REQUIRE_FULL_WINDOW = False


# ============================================================
# Utility helpers
# ============================================================

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def normalize_embedding(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.tensor(x)


def canonical_pid(name: str) -> str:
    """
    Normalize speaker IDs to 'P####' if possible.
    Examples:
      'P0737' -> 'P0737'
      'participant_a_P0737' -> 'P0737'
      'participant_b_1093' -> 'P1093'
      '1093' -> 'P1093'
    """
    if name is None:
        return name

    s = str(name)

    m = re.search(r'(P\d{4})', s)
    if m:
        return m.group(1)

    m = re.search(r'participant_[ab]_(\d{4})', s)
    if m:
        return f'P{m.group(1)}'

    m = re.search(r'(\d{4})', s)
    if m:
        return f'P{m.group(1)}'

    return s


def parse_participant_label(participant_key: str) -> Dict[str, Optional[str]]:
    """
    Parse keys like:
      participant_a_P0737
      participant_b_P1093
      participant_a_0737
      participant_b_1093

    Returns:
      {
        'participant_key': original,
        'side': 'a' / 'b' / None,
        'participant_id': 'P0737' / ...
      }
    """
    out = {
        "participant_key": participant_key,
        "side": None,
        "participant_id": None,
    }

    m_side = re.search(r'participant_([ab])_', participant_key)
    if m_side:
        out["side"] = m_side.group(1)

    out["participant_id"] = canonical_pid(participant_key)
    return out


def parse_turn_key(turn_key: str) -> Dict[str, Any]:
    """
    Parse turn keys like:
      turn_001_0.00s-33.44s
      turn_001_45.66-64.33s
      utterance_001_0.00s-33.44s
      utterance_001_45.66-64.33s

    Returns:
      {
        'turn_key': original,
        'turn_idx': 1,
        'start': 0.00,
        'end': 33.44
      }
    """
    pattern = r'^(?:turn|utterance)_(\d+)_([0-9.]+)s?-([0-9.]+)s?$'
    m = re.match(pattern, turn_key)
    if not m:
        raise ValueError(f"Could not parse turn key: {turn_key}")

    turn_idx = int(m.group(1))
    start = float(m.group(2))
    end = float(m.group(3))

    return {
        "turn_key": turn_key,
        "turn_idx": turn_idx,
        "start": start,
        "end": end,
    }


def find_annotation_file(session_name: str) -> Optional[Path]:
    """
    Tries a few likely locations for the session annotation file.
    Edit this if your annotation layout differs.
    """
    candidates = [
        ANNOTATION_ROOT / session_name / "interaction/pre_and_post_moi/turns_pre_post.json",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def load_session_annotations(session_name: str) -> List[dict]:
    ann_path = find_annotation_file(session_name)
    if ann_path is None:
        raise FileNotFoundError(
            f"Could not find annotation file for session {session_name}. "
            f"Please update find_annotation_file()."
        )

    with open(ann_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of events in {ann_path}, got {type(data)}")

    return data


def overlap(turn_start: float, turn_end: float, event_start: float, event_end: float) -> bool:
    return not (turn_end <= event_start or turn_start >= event_end)


def sort_turns(turns: List[dict]) -> List[dict]:
    turns = sorted(turns, key=lambda x: (x["start"], x["end"], x["speaker"]))
    for i, t in enumerate(turns):
        t["global_turn_idx"] = i
    return turns


# ============================================================
# Session preprocessing
# ============================================================

def build_turn_table_from_embedding_pkl(
    embedding_pkl: Dict[str, Dict[str, Any]],
    session_name: str,
    model: str
) -> List[dict]:
    """
    Input expected:
      {
        'participant_a_P0737': {
            'utterance_001_0.00s-33.44s': embedding,
            ...
        },
        'participant_b_P1093': {
            ...
        }
      }

    Output:
      list of standardized turn dicts
    """
    all_turns = []

    for participant_key, turn_map in embedding_pkl.items():
        participant_meta = parse_participant_label(participant_key)

        if not isinstance(turn_map, dict):
            continue

        for turn_key, emb in turn_map.items():
            turn_meta = parse_turn_key(turn_key)
            wav_filepath = build_wav_filepath(session_name, participant_key, turn_key)

            turn = {
                "participant_key": participant_key,
                "speaker": participant_meta["participant_id"],   # canonical P####
                "speaker_side": participant_meta["side"],        # a / b
                "turn_key": turn_key,
                "turn_idx": turn_meta["turn_idx"],               # local per-participant index
                "start": turn_meta["start"],
                "end": turn_meta["end"],
                "wav_filename": f"{turn_key}.wav",
                "wav_filepath": str(wav_filepath),
                "embedding": normalize_embedding(emb),
                "model": model,
            }
            all_turns.append(turn)

    return sort_turns(all_turns)

def build_wav_filepath(session_name: str, participant_key: str, turn_key: str) -> Path:
    return WAV_ROOT / session_name / participant_key / "wavs_by_turn" / f"{turn_key}.wav"


def get_other_participant_id(turns: List[dict], participant_id: str) -> Optional[str]:
    speakers = sorted(list({t["speaker"] for t in turns}))
    others = [s for s in speakers if s != participant_id]
    if len(others) == 1:
        return others[0]
    return others[0] if len(others) > 0 else None


def get_during_turns(turns: List[dict], start_moi: float, end_moi: float) -> List[dict]:
    return [t for t in turns if overlap(t["start"], t["end"], start_moi, end_moi)]


def get_pre_post_turns(turns: List[dict], during_turns: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    if len(during_turns) == 0:
        return None, None

    during_idxs = sorted([t["global_turn_idx"] for t in during_turns])
    first_idx = during_idxs[0]
    last_idx = during_idxs[-1]

    pre_turn = turns[first_idx - 1] if first_idx - 1 >= 0 else None
    post_turn = turns[last_idx + 1] if last_idx + 1 < len(turns) else None

    return pre_turn, post_turn


def infer_role_labels(
    event: dict,
    turns: List[dict],
    during_turns: List[dict]
) -> Dict[str, bool]:
    """
    Rule:
    - use original event-speaker
    - annotations are given per participant
    - if annotated_participant != event_speaker, then annotated participant is listener
    - if annotated_participant == event_speaker, then annotated participant is speaker

    We also compare against the during turn when available.
    """
    annotated = canonical_pid(event.get("annotated_participant"))
    event_speaker = canonical_pid(event.get("event_speaker"))

    annotated_is_speaker = annotated == event_speaker
    annotated_is_listener = annotated != event_speaker

    if len(during_turns) > 0:
        during_speakers = sorted(list({t["speaker"] for t in during_turns}))
        if len(during_speakers) == 1:
            actual_during_speaker = during_speakers[0]
        else:
            actual_during_speaker = None
    else:
        actual_during_speaker = None

    return {
        "annotated_is_speaker": annotated_is_speaker,
        "annotated_is_listener": annotated_is_listener,
        "actual_during_speaker": actual_during_speaker,
    }


def compute_proper_turn_switches(
    pre_turn: Optional[dict],
    post_turn: Optional[dict],
    event_speaker: str
) -> bool:
    """
    True when:
      - closest global prior turn exists and is by the other participant
      - closest global next turn exists and is by the other participant

    Since pre/post are the immediate previous/next global turns around the MoI window,
    this checks whether both are not by the event speaker.
    """
    event_speaker = canonical_pid(event_speaker)

    if pre_turn is None or post_turn is None:
        return False

    return (pre_turn["speaker"] != event_speaker) and (post_turn["speaker"] != event_speaker)


def build_event_record(event: dict, turns: List[dict]) -> Optional[dict]:
    start_moi = float(event["start_moi"])
    end_moi = float(event["end_moi"])

    annotated = canonical_pid(event.get("annotated_participant"))
    event_speaker = canonical_pid(event.get("event_speaker"))

    during_turns = get_during_turns(turns, start_moi, end_moi)
    pre_turn, post_turn = get_pre_post_turns(turns, during_turns)

    role_info = infer_role_labels(event, turns, during_turns)
    other_participant = get_other_participant_id(turns, annotated)
    proper_turn_switches = compute_proper_turn_switches(pre_turn, post_turn, event_speaker)


    event_record = {
        "event_metadata": {
            **event,
            "annotated_participant": annotated,
            "event_speaker": event_speaker,
            "other_participant": other_participant,
            "annotated_is_speaker": role_info["annotated_is_speaker"],
            "annotated_is_listener": role_info["annotated_is_listener"],
            "actual_during_speaker": role_info["actual_during_speaker"],
            "proper_turn_switches": proper_turn_switches,
        },
        "pre": pre_turn,
        "during": during_turns,
        "post": post_turn,
    }

    if REQUIRE_FULL_WINDOW:
        if pre_turn is None or len(during_turns) == 0 or post_turn is None:
            return None

    return event_record


def build_session_moi_dataset(session_name: str, model: str) -> dict:
    emb_path = DEEP_EMB_ROOT / session_name / f"deep_embeds_{model}.pkl"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {emb_path}")

    embedding_pkl = load_pickle(emb_path)
    turns = build_turn_table_from_embedding_pkl(embedding_pkl,session_name=session_name,model=model)
    annotations = load_session_annotations(session_name)

    event_records = []
    for i, event in enumerate(annotations):
        try:
            rec = build_event_record(event, turns)
            if rec is not None:
                rec["event_metadata"]["event_idx"] = i
                rec["event_metadata"]["session_name"] = session_name
                event_records.append(rec)
        except Exception as e:
            print(f"[WARN] Failed event {i} in {session_name}: {e}")

    out = {
        "session_name": session_name,
        "model": model,
        "turns": turns,
        "moi_events": event_records,
    }
    return out


# ============================================================
# Main
# ============================================================

def process_all_sessions(model: str = MODEL):
    session_dirs = sorted([p for p in DEEP_EMB_ROOT.iterdir() if p.is_dir()])

    for session_dir in session_dirs:
        session_name = session_dir.name
        emb_path = session_dir / f"deep_embeds_{model}.pkl"

        if not emb_path.exists():
            continue

        try:
            session_data = build_session_moi_dataset(session_name, model=model)
            out_path = session_dir / f"moi_embeddings_{model}.pkl"
            save_pickle(session_data, out_path)

            print(
                f"[OK] {session_name}: "
                f"{len(session_data['turns'])} turns, "
                f"{len(session_data['moi_events'])} MoI events -> {out_path}"
            )

        except Exception as e:
            print(f"[ERROR] {session_name}: {e}")


if __name__ == "__main__":
    process_all_sessions(model=MODEL)