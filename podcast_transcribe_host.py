import argparse
import csv
import inspect
import json
import os
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def configure_ffmpeg_dll_directory():
    ffmpeg_bin_dir = os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR") or os.getenv("FFMPEG_BIN_DIR")
    if os.name != "nt" or not ffmpeg_bin_dir or not hasattr(os, "add_dll_directory"):
        return

    if os.path.isdir(ffmpeg_bin_dir):
        os.add_dll_directory(ffmpeg_bin_dir)


configure_ffmpeg_dll_directory()

warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    module=r"pyannote\.audio\.core\.io",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Requested Pretrainer collection using symlinks on Windows.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
)

import numpy as np
import torch
import torchaudio
import huggingface_hub
from faster_whisper import WhisperModel


def _patch_huggingface_hub_auth_compat():
    signature = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" in signature.parameters:
        return

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compat_hf_hub_download

    try:
        import huggingface_hub.file_download as file_download

        file_download.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass


_patch_huggingface_hub_auth_compat()

import pyannote.audio as pyannote_audio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


@dataclass
class WordItem:
    start: Optional[float]
    end: Optional[float]
    word: str
    speaker: Optional[str]


@dataclass
class SegmentItem:
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str]
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
    words: List[WordItem]


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe podcasts with diarization and host labeling.")
    parser.add_argument("--input-dir", required=True, help="Directory containing audio files to process.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to input directory.")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name.")
    parser.add_argument("--language", default="en", help="Language code.")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, or cuda.")
    # "auto" can pick CPU paths or unsupported configs. 5070 Ti → float16 is correct and fastest
    # parser.add_argument("--compute-type", default="auto", help="faster-whisper compute type.")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for faster-whisper.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token for pyannote pipeline.")
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote diarization pipeline id.",
    )
    parser.add_argument(
        "--speaker-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="Speaker verification model id.",
    )
    parser.add_argument(
        "--host-reference",
        help="Optional audio file containing a clean sample of the host voice. Strongly recommended for stable host labeling.",
    )
    parser.add_argument(
        "--host-profile-json",
        default="host_profile.json",
        help="Path to a JSON file used to persist a host embedding profile across episodes.",
    )
    parser.add_argument(
        "--known-speakers-dir",
        help="Optional directory containing speakers.json plus named reference audio clips for known speakers.",
    )
    parser.add_argument(
        "--preferred-terms-file",
        help="Optional text file with one preferred term per line. Used as prompt/hotword biasing.",
    )
    parser.add_argument(
        "--replacement-map-json",
        help="Optional JSON file mapping preferred spellings to likely mistranscriptions.",
    )
    parser.add_argument(
        "--assume-dominant-speaker-is-host",
        action="store_true",
        help="When no host reference/profile exists, label the speaker with the most talk time as HOST and bootstrap the profile.",
    )
    parser.add_argument(
        "--host-threshold",
        type=float,
        default=0.45,
        help="Cosine similarity threshold for matching a speaker to the host profile/reference.",
    )
    parser.add_argument(
        "--min-host-seconds",
        type=float,
        default=20.0,
        help="Minimum diarized speech duration required before using a speaker to update the host profile.",
    )
    parser.add_argument(
        "--max-embedding-seconds",
        type=float,
        default=90.0,
        help="Maximum total speech duration per speaker used to build an embedding.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Optional fixed speaker count for pyannote diarization.",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_runtime_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_preferred_terms(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_speaker_verifier(model_id: str, device: str):
    from speechbrain.inference.speaker import SpeakerRecognition

    return SpeakerRecognition.from_hparams(
        source=model_id,
        savedir="pretrained_speaker_model",
        run_opts={"device": normalize_runtime_device(device)},
    )


def build_prompt_bias(terms: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not terms:
        return None, None
    hotwords = ", ".join(terms)
    initial_prompt = (
        "Domain vocabulary and preferred spellings: "
        f"{hotwords}. Use these spellings when they match the audio."
    )
    return initial_prompt, hotwords


def load_replacement_map(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    normalized = {}
    for preferred, aliases in payload.items():
        if isinstance(aliases, list):
            normalized[preferred] = [alias for alias in aliases if isinstance(alias, str) and alias.strip()]
    return normalized


def apply_replacements(text: str, replacement_map: Dict[str, List[str]]) -> str:
    updated = text
    for preferred, aliases in replacement_map.items():
        for alias in aliases:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            updated = pattern.sub(preferred, updated)
    return updated


def detect_replacement_hits(text: str, replacement_map: Dict[str, List[str]]) -> List[Dict[str, str]]:
    hits = []
    for preferred, aliases in replacement_map.items():
        for alias in aliases:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            if pattern.search(text):
                hits.append({"preferred": preferred, "alias": alias})
    return hits


def load_audio_mono_16k(path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.squeeze(0)


def load_host_profile(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    vector = payload.get("embedding")
    if not isinstance(vector, list):
        return None
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return None
    return arr / norm


def load_known_speakers_config(known_speakers_dir: Optional[str]) -> List[Dict[str, object]]:
    if not known_speakers_dir:
        return []

    config_path = Path(known_speakers_dir) / "speakers.json"
    if not config_path.exists():
        return []

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    speakers = payload.get("speakers", [])
    return speakers if isinstance(speakers, list) else []


def save_host_profile(path: Optional[str], embedding: Optional[np.ndarray], source: str):
    if not path or embedding is None:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "updated_from": source,
        "embedding": embedding.tolist(),
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def average_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    merged = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(merged)
    if norm == 0:
        return None
    return merged / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def merge_profile(existing: Optional[np.ndarray], new_embedding: np.ndarray) -> np.ndarray:
    if existing is None:
        merged = new_embedding
    else:
        merged = (existing + new_embedding) / 2.0
    norm = np.linalg.norm(merged)
    if norm == 0:
        return new_embedding
    return merged / norm


def transcribe_audio(
    model: WhisperModel,
    audio_path: str,
    language: str,
    beam_size: int,
    batch_size: int,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
) -> Tuple[List[SegmentItem], Dict[str, object]]:
    transcribe_kwargs = {
        "language": language,
        "beam_size": beam_size,
        "vad_filter": True,
        "word_timestamps": True,
        "condition_on_previous_text": True,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
    }

    transcribe_signature = inspect.signature(model.transcribe)
    if "batch_size" in transcribe_signature.parameters:
        transcribe_kwargs["batch_size"] = batch_size

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    results = []
    total_duration = getattr(info, "duration", None)
    transcription_started = time.perf_counter()
    last_progress_update = 0.0
    for idx, segment in enumerate(segments):
        if total_duration and total_duration > 0:
            progress_ratio = min(1.0, float(segment.end) / float(total_duration))
            progress = min(100, int(progress_ratio * 100))
            elapsed_seconds = time.perf_counter() - transcription_started
            should_print_progress = (
                last_progress_update == 0.0
                or elapsed_seconds - last_progress_update >= 5.0
                or progress >= 100
            )
            if should_print_progress:
                estimated_total_seconds = (
                    elapsed_seconds / progress_ratio if progress_ratio > 0 else None
                )
                bar_width = 40
                filled = min(bar_width, int((progress / 100) * bar_width))
                bar = "━" * filled + " " * (bar_width - filled)
                eta_text = (
                    format_timestamp(estimated_total_seconds)
                    if estimated_total_seconds is not None
                    else "unknown"
                )
                print(
                    f"\r  transcription         {bar} {progress:3d}% "
                    f"{format_timestamp(elapsed_seconds)} / {eta_text}",
                    end="",
                    flush=True,
                )
                last_progress_update = elapsed_seconds
        words = []
        if segment.words:
            for word in segment.words:
                words.append(
                    WordItem(
                        start=getattr(word, "start", None),
                        end=getattr(word, "end", None),
                        word=getattr(word, "word", ""),
                        speaker=None,
                    )
                )

        results.append(
            SegmentItem(
                id=idx,
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                speaker=None,
                avg_logprob=getattr(segment, "avg_logprob", None),
                no_speech_prob=getattr(segment, "no_speech_prob", None),
                words=words,
            )
        )
    if total_duration and total_duration > 0:
        print()

    info_payload = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
    }
    return results, info_payload


def diarize_audio(pipeline: Pipeline, audio_path: str, num_speakers: Optional[int]) -> List[Dict[str, object]]:
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    waveform, sample_rate = torchaudio.load(audio_path)
    diarization_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    with ProgressHook() as hook:
        diarization = pipeline(diarization_input, hook=hook, **kwargs)

    diarization_annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    turns = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return turns


def parse_version_major(version_text: str) -> int:
    match = re.match(r"^(\d+)", version_text or "")
    return int(match.group(1)) if match else 0


def resolve_compatible_diarization_model(model_id: str) -> Tuple[str, Optional[str]]:
    pyannote_version = getattr(pyannote_audio, "__version__", "")
    pyannote_major = parse_version_major(pyannote_version)

    if model_id == "pyannote/speaker-diarization-community-1" and pyannote_major and pyannote_major < 4:
        return (
            "pyannote/speaker-diarization-3.1",
            (
                f"pyannote.audio {pyannote_version} is installed, so switching diarization model from "
                "'pyannote/speaker-diarization-community-1' to the compatible legacy pipeline "
                "'pyannote/speaker-diarization-3.1'."
            ),
        )

    return model_id, None


def load_diarization_pipeline(model_id: str, hf_token: str) -> Tuple[Pipeline, str]:
    resolved_model_id, compatibility_note = resolve_compatible_diarization_model(model_id)
    if compatibility_note:
        print(compatibility_note)

    signature = inspect.signature(Pipeline.from_pretrained)
    parameters = signature.parameters

    if "token" in parameters:
        return Pipeline.from_pretrained(resolved_model_id, token=hf_token), resolved_model_id

    if "use_auth_token" in parameters:
        return Pipeline.from_pretrained(resolved_model_id, use_auth_token=hf_token), resolved_model_id

    raise RuntimeError(
        "Unsupported pyannote.audio installation: Pipeline.from_pretrained accepts neither "
        "'token' nor 'use_auth_token'."
    )


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers_to_segments(segments: List[SegmentItem], diarized_turns: List[Dict[str, object]]):
    for segment in segments:
        overlap_by_speaker = defaultdict(float)
        for turn in diarized_turns:
            overlap = overlap_seconds(segment.start, segment.end, turn["start"], turn["end"])
            if overlap > 0:
                overlap_by_speaker[turn["speaker"]] += overlap

        if overlap_by_speaker:
            segment.speaker = max(overlap_by_speaker.items(), key=lambda item: item[1])[0]

        for word in segment.words:
            if word.start is None or word.end is None:
                word.speaker = segment.speaker
                continue

            word_overlap = defaultdict(float)
            for turn in diarized_turns:
                overlap = overlap_seconds(word.start, word.end, turn["start"], turn["end"])
                if overlap > 0:
                    word_overlap[turn["speaker"]] += overlap

            if word_overlap:
                word.speaker = max(word_overlap.items(), key=lambda item: item[1])[0]
            else:
                word.speaker = segment.speaker


def speaker_durations(diarized_turns: List[Dict[str, object]]) -> Dict[str, float]:
    totals = defaultdict(float)
    for turn in diarized_turns:
        totals[turn["speaker"]] += max(0.0, turn["end"] - turn["start"])
    return dict(totals)


def build_speaker_audio_samples(
    waveform_16k: torch.Tensor,
    diarized_turns: List[Dict[str, object]],
    max_seconds: float,
) -> Dict[str, torch.Tensor]:
    clips = defaultdict(list)
    durations = defaultdict(float)

    total_samples = waveform_16k.shape[0]
    for turn in diarized_turns:
        speaker = turn["speaker"]
        if durations[speaker] >= max_seconds:
            continue

        start = max(0, int(turn["start"] * 16000))
        end = min(total_samples, int(turn["end"] * 16000))
        if end <= start:
            continue

        remaining = max_seconds - durations[speaker]
        clip_samples = int(remaining * 16000)
        segment = waveform_16k[start : min(end, start + clip_samples)]
        if segment.numel() == 0:
            continue

        clips[speaker].append(segment)
        durations[speaker] += segment.shape[0] / 16000.0

    merged = {}
    for speaker, chunks in clips.items():
        merged[speaker] = torch.cat(chunks)
    return merged


def compute_embedding(verifier: Any, waveform_16k: torch.Tensor) -> np.ndarray:
    signal = waveform_16k.unsqueeze(0)
    with torch.no_grad():
        embedding = verifier.encode_batch(signal)
    vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_known_speaker_profiles(
    verifier: Any,
    known_speakers_dir: Optional[str],
) -> Dict[str, Dict[str, object]]:
    config_entries = load_known_speakers_config(known_speakers_dir)
    if not config_entries:
        return {}

    base_dir = Path(known_speakers_dir)
    profiles = {}

    for entry in config_entries:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
        files = entry.get("files", [])
        if not name or not isinstance(files, list):
            continue

        embeddings = []
        resolved_files = []
        for relative_path in files:
            sample_path = base_dir / str(relative_path)
            if not sample_path.exists():
                continue
            waveform = load_audio_mono_16k(str(sample_path))
            embeddings.append(compute_embedding(verifier, waveform))
            resolved_files.append(str(sample_path))

        averaged = average_embeddings(embeddings)
        if averaged is None:
            continue

        profiles[name] = {
            "name": name,
            "embedding": averaged,
            "is_host": bool(entry.get("is_host", False)) or name.upper() == "HOST",
            "sample_files": resolved_files,
        }

    return profiles


def choose_host_speaker(
    verifier: Any,
    waveform_16k: torch.Tensor,
    diarized_turns: List[Dict[str, object]],
    host_reference_path: Optional[str],
    existing_profile: Optional[np.ndarray],
    host_threshold: float,
    assume_dominant: bool,
    max_embedding_seconds: float,
    min_host_seconds: float,
) -> Tuple[Optional[str], Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, float], Dict[str, float]]:
    durations = speaker_durations(diarized_turns)
    if not durations:
        return None, {}, existing_profile, {}, {}

    speaker_audio = build_speaker_audio_samples(waveform_16k, diarized_turns, max_embedding_seconds)
    speaker_embeddings = {}
    for speaker, clip in speaker_audio.items():
        if durations.get(speaker, 0.0) >= min_host_seconds:
            speaker_embeddings[speaker] = compute_embedding(verifier, clip)

    reference_embedding = existing_profile
    if host_reference_path:
        ref_waveform = load_audio_mono_16k(host_reference_path)
        reference_embedding = compute_embedding(verifier, ref_waveform)

    best_match = None
    best_score = -1.0
    similarity_scores = {}

    if reference_embedding is not None:
        for speaker, embedding in speaker_embeddings.items():
            score = cosine_similarity(reference_embedding, embedding)
            similarity_scores[speaker] = score
            if score > best_score:
                best_score = score
                best_match = speaker

        if best_match is not None and best_score >= host_threshold:
            updated_profile = merge_profile(existing_profile, speaker_embeddings[best_match])
            return best_match, speaker_embeddings, updated_profile, durations, similarity_scores

    if assume_dominant:
        dominant_speaker = max(durations.items(), key=lambda item: item[1])[0]
        updated_profile = existing_profile
        if dominant_speaker in speaker_embeddings:
            updated_profile = merge_profile(existing_profile, speaker_embeddings[dominant_speaker])
        return dominant_speaker, speaker_embeddings, updated_profile, durations, similarity_scores

    return None, speaker_embeddings, existing_profile, durations, similarity_scores


def match_known_speakers(
    speaker_embeddings: Dict[str, np.ndarray],
    known_profiles: Dict[str, Dict[str, object]],
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    assignments = {}
    candidates = []

    for diarized_speaker, diarized_embedding in speaker_embeddings.items():
        for known_name, profile in known_profiles.items():
            score = cosine_similarity(diarized_embedding, profile["embedding"])
            if score >= threshold:
                candidates.append((score, diarized_speaker, known_name))

    for score, diarized_speaker, known_name in sorted(candidates, reverse=True):
        if diarized_speaker in assignments:
            continue
        if any(match["known_name"] == known_name for match in assignments.values()):
            continue
        assignments[diarized_speaker] = {
            "known_name": known_name,
            "score": score,
            "is_host": bool(known_profiles[known_name].get("is_host", False)),
        }

    return assignments


def rename_speakers(
    segments: List[SegmentItem],
    diarized_turns: List[Dict[str, object]],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Optional[Dict[str, Dict[str, object]]] = None,
):
    ordered = sorted(durations.items(), key=lambda item: item[1], reverse=True)
    mapping = {}
    guest_index = 1
    known_assignments = known_assignments or {}
    for speaker, _ in ordered:
        if speaker in known_assignments:
            mapping[speaker] = known_assignments[speaker]["known_name"]
        elif speaker == host_speaker:
            mapping[speaker] = "HOST"
        else:
            mapping[speaker] = f"SPEAKER_{guest_index:02d}"
            guest_index += 1

    for segment in segments:
        if segment.speaker in mapping:
            segment.speaker = mapping[segment.speaker]
        for word in segment.words:
            if word.speaker in mapping:
                word.speaker = mapping[word.speaker]

    for turn in diarized_turns:
        if turn["speaker"] in mapping:
            turn["speaker_label"] = mapping[turn["speaker"]]
        else:
            turn["speaker_label"] = turn["speaker"]

    return mapping


def coalesce_segments(
    segments: List[SegmentItem],
    replacement_map: Dict[str, List[str]],
) -> Tuple[List[SegmentItem], List[Dict[str, object]]]:
    cleaned = []
    replacement_events = []
    for segment in segments:
        replacement_hits = detect_replacement_hits(segment.text, replacement_map)
        for hit in replacement_hits:
            replacement_events.append(
                {
                    "issue_type": "glossary_replacement_candidate",
                    "speaker": segment.speaker or "UNKNOWN",
                    "start": format_timestamp(segment.start),
                    "end": format_timestamp(segment.end),
                    "score": "",
                    "details": f"Detected alias '{hit['alias']}' and normalized to '{hit['preferred']}'.",
                    "text": segment.text,
                }
            )

        segment.text = apply_replacements(segment.text, replacement_map).strip()
        if not segment.text:
            continue

        for word in segment.words:
            word.word = apply_replacements(word.word, replacement_map)

        if cleaned and cleaned[-1].speaker == segment.speaker and segment.start - cleaned[-1].end <= 0.8:
            cleaned[-1].text = (cleaned[-1].text + " " + segment.text).strip()
            cleaned[-1].end = segment.end
            cleaned[-1].words.extend(segment.words)
        else:
            cleaned.append(segment)
    return cleaned, replacement_events


def collect_review_rows(
    source_file: str,
    segments: List[SegmentItem],
    replacement_events: List[Dict[str, object]],
    host_speaker: Optional[str],
    host_threshold: float,
    durations: Dict[str, float],
    similarity_scores: Dict[str, float],
    speaker_mapping: Dict[str, str],
) -> List[Dict[str, object]]:
    rows = []

    if host_speaker is None:
        rows.append(
            {
                "issue_type": "host_not_detected",
                "speaker": "",
                "start": "",
                "end": "",
                "score": "",
                "details": "No host speaker met the configured threshold and no fallback label was established.",
                "text": "",
                "source_file": source_file,
            }
        )

    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    if sorted_scores:
        top_speaker, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else None
        margin = top_score - second_score if second_score is not None else None

        if top_score < host_threshold + 0.05:
            rows.append(
                {
                    "issue_type": "host_match_near_threshold",
                    "speaker": speaker_mapping.get(top_speaker, top_speaker),
                    "start": "",
                    "end": "",
                    "score": round(top_score, 4),
                    "details": f"Top host similarity is close to threshold {host_threshold:.2f}.",
                    "text": "",
                    "source_file": source_file,
                }
            )

        if margin is not None and margin < 0.05:
            rows.append(
                {
                    "issue_type": "host_match_ambiguous",
                    "speaker": speaker_mapping.get(top_speaker, top_speaker),
                    "start": "",
                    "end": "",
                    "score": round(top_score, 4),
                    "details": f"Top two host similarity scores are close; margin={margin:.4f}.",
                    "text": "",
                    "source_file": source_file,
                }
            )

    if host_speaker is not None and host_speaker in durations and durations[host_speaker] < 60:
        rows.append(
            {
                "issue_type": "host_low_coverage",
                "speaker": speaker_mapping.get(host_speaker, host_speaker),
                "start": "",
                "end": "",
                "score": round(durations[host_speaker], 2),
                "details": "Detected host has less than 60 seconds of diarized speech in this episode.",
                "text": "",
                "source_file": source_file,
            }
        )

    for event in replacement_events:
        rows.append({**event, "source_file": source_file})

    for segment in segments:
        if segment.speaker == "HOST" and similarity_scores:
            top_score = max(similarity_scores.values())
            if top_score < host_threshold + 0.05:
                rows.append(
                    {
                        "issue_type": "host_segment_review",
                        "speaker": segment.speaker,
                        "start": format_timestamp(segment.start),
                        "end": format_timestamp(segment.end),
                        "score": round(top_score, 4),
                        "details": "Host label came from a weak overall speaker match; review this segment if accuracy is important.",
                        "text": segment.text,
                        "source_file": source_file,
                    }
                )

    return rows


def write_review_csv(path: Path, rows: List[Dict[str, object]]):
    fieldnames = ["issue_type", "speaker", "start", "end", "score", "details", "text", "source_file"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_review_rows(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts[row.get("issue_type", "unknown")] += 1
    return dict(counts)


def build_episode_summary_row(
    audio_path: Path,
    normalized_segments: List[SegmentItem],
    review_rows: List[Dict[str, object]],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    similarity_scores: Dict[str, float],
    speaker_mapping: Dict[str, str],
    known_assignments: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    review_counts = summarize_review_rows(review_rows)
    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1] if sorted_scores else ""
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else ""
    score_margin = (sorted_scores[0][1] - sorted_scores[1][1]) if len(sorted_scores) > 1 else ""
    host_duration = durations.get(host_speaker, 0.0) if host_speaker else 0.0
    total_duration = sum(durations.values())
    host_share = (host_duration / total_duration) if total_duration else 0.0
    review_priority_score = 0.0
    priority_reasons = []

    if host_speaker is None:
        review_priority_score += 100.0
        priority_reasons.append("host not detected")

    if top_score != "":
        review_priority_score += max(0.0, (0.7 - top_score) * 100.0)
        if top_score < 0.55:
            priority_reasons.append(f"low host similarity ({top_score:.2f})")
    else:
        review_priority_score += 30.0
        priority_reasons.append("no host similarity score available")

    if score_margin != "":
        review_priority_score += max(0.0, (0.1 - score_margin) * 120.0)
        if score_margin < 0.05:
            priority_reasons.append(f"ambiguous top speaker margin ({score_margin:.2f})")
    else:
        review_priority_score += 10.0
        priority_reasons.append("only one speaker candidate scored")

    review_priority_score += max(0.0, (0.35 - host_share) * 80.0)
    if host_share < 0.35:
        priority_reasons.append(f"low host share of speech ({host_share:.0%})")

    review_priority_score += review_counts.get("host_match_near_threshold", 0) * 12.0
    review_priority_score += review_counts.get("host_match_ambiguous", 0) * 20.0
    review_priority_score += review_counts.get("host_low_coverage", 0) * 18.0
    review_priority_score += review_counts.get("host_segment_review", 0) * 1.5
    review_priority_score += review_counts.get("host_not_detected", 0) * 40.0
    review_priority_score += min(review_counts.get("glossary_replacement_candidate", 0), 20) * 0.5

    if review_counts.get("host_low_coverage", 0):
        priority_reasons.append("host coverage is low")
    if review_counts.get("host_segment_review", 0):
        priority_reasons.append(f"{review_counts.get('host_segment_review', 0)} host segments need review")
    if review_counts.get("glossary_replacement_candidate", 0) >= 5:
        priority_reasons.append(f"{review_counts.get('glossary_replacement_candidate', 0)} glossary corrections applied")

    if not priority_reasons:
        priority_reasons.append("no major review issues detected")

    return {
        "episode": audio_path.name,
        "review_priority_score": round(review_priority_score, 2),
        "review_priority_reason": "; ".join(dict.fromkeys(priority_reasons)),
        "host_detected": host_speaker is not None,
        "host_label": speaker_mapping.get(host_speaker, "") if host_speaker else "",
        "known_speakers_detected": ", ".join(
            speaker_mapping[speaker_id]
            for speaker_id in sorted(known_assignments.keys(), key=lambda key: speaker_mapping.get(key, key))
        ),
        "host_duration_seconds": round(host_duration, 2),
        "host_share_of_speech": round(host_share, 4),
        "top_host_similarity": round(top_score, 4) if top_score != "" else "",
        "second_host_similarity": round(second_score, 4) if second_score != "" else "",
        "host_similarity_margin": round(score_margin, 4) if score_margin != "" else "",
        "speaker_count": len(durations),
        "transcript_segments": len(normalized_segments),
        "review_row_count": len(review_rows),
        "host_match_near_threshold_count": review_counts.get("host_match_near_threshold", 0),
        "host_match_ambiguous_count": review_counts.get("host_match_ambiguous", 0),
        "host_low_coverage_count": review_counts.get("host_low_coverage", 0),
        "host_segment_review_count": review_counts.get("host_segment_review", 0),
        "glossary_replacement_candidate_count": review_counts.get("glossary_replacement_candidate", 0),
        "host_not_detected_count": review_counts.get("host_not_detected", 0),
    }


def write_episode_summary_csv(path: Path, rows: List[Dict[str, object]]):
    sorted_rows = sorted(rows, key=lambda row: row.get("review_priority_score", 0), reverse=True)
    fieldnames = [
        "episode",
        "review_priority_score",
        "review_priority_reason",
        "host_detected",
        "host_label",
        "known_speakers_detected",
        "host_duration_seconds",
        "host_share_of_speech",
        "top_host_similarity",
        "second_host_similarity",
        "host_similarity_margin",
        "speaker_count",
        "transcript_segments",
        "review_row_count",
        "host_match_near_threshold_count",
        "host_match_ambiguous_count",
        "host_low_coverage_count",
        "host_segment_review_count",
        "glossary_replacement_candidate_count",
        "host_not_detected_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def write_text_transcript(
    path: Path,
    segments: List[SegmentItem],
    host_only: bool = False,
    host_labels: Optional[set[str]] = None,
):
    lines = []
    host_labels = host_labels or {"HOST"}
    for segment in segments:
        if host_only and segment.speaker not in host_labels:
            continue
        label = segment.speaker or "UNKNOWN"
        lines.append(f"[{format_timestamp(segment.start)}][{label}] {segment.text}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_output(
    path: Path,
    source_file: str,
    info_payload: Dict[str, object],
    diarized_turns: List[Dict[str, object]],
    segments: List[SegmentItem],
    speaker_mapping: Dict[str, str],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Dict[str, Dict[str, object]],
):
    payload = {
        "source_file": source_file,
        "transcription": info_payload,
        "host_detected": host_speaker is not None,
        "host_original_speaker_id": host_speaker,
        "speaker_mapping": speaker_mapping,
        "known_speaker_assignments": known_assignments,
        "speaker_durations_seconds": durations,
        "diarization_turns": diarized_turns,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "words": [asdict(word) for word in segment.words],
            }
            for segment in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_file(
    audio_path: Path,
    output_dir: Path,
    whisper_model: WhisperModel,
    diarization_pipeline: Pipeline,
    verifier: Any,
    language: str,
    beam_size: int,
    batch_size: int,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
    replacement_map: Dict[str, List[str]],
    host_reference: Optional[str],
    host_profile_path: Optional[str],
    known_speaker_profiles: Dict[str, Dict[str, object]],
    host_threshold: float,
    assume_dominant: bool,
    max_embedding_seconds: float,
    min_host_seconds: float,
    num_speakers: Optional[int],
) -> Dict[str, object]:
    print(f"Processing {audio_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  stage: transcription")
    transcription_started = time.perf_counter()
    segments, info_payload = transcribe_audio(
        model=whisper_model,
        audio_path=str(audio_path),
        language=language,
        beam_size=beam_size,
        batch_size=batch_size,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    print(
        f"  transcription complete: {len(segments)} raw segments "
        f"in {time.perf_counter() - transcription_started:.1f}s"
    )

    print("  stage: diarization")
    diarization_started = time.perf_counter()
    diarized_turns = diarize_audio(diarization_pipeline, str(audio_path), num_speakers=num_speakers)
    assign_speakers_to_segments(segments, diarized_turns)
    print(
        f"  diarization complete: {len(diarized_turns)} turns "
        f"in {time.perf_counter() - diarization_started:.1f}s"
    )

    print("  stage: speaker matching")
    matching_started = time.perf_counter()
    waveform_16k = load_audio_mono_16k(str(audio_path))
    existing_profile = load_host_profile(host_profile_path)
    host_speaker, speaker_embeddings, updated_profile, durations, similarity_scores = choose_host_speaker(
        verifier=verifier,
        waveform_16k=waveform_16k,
        diarized_turns=diarized_turns,
        host_reference_path=host_reference,
        existing_profile=existing_profile,
        host_threshold=host_threshold,
        assume_dominant=assume_dominant,
        max_embedding_seconds=max_embedding_seconds,
        min_host_seconds=min_host_seconds,
    )

    known_assignments = match_known_speakers(
        speaker_embeddings=speaker_embeddings,
        known_profiles=known_speaker_profiles,
        threshold=host_threshold,
    )
    known_host_speaker = next(
        (speaker_id for speaker_id, assignment in known_assignments.items() if assignment.get("is_host")),
        None,
    )
    if known_host_speaker:
        host_speaker = known_host_speaker

    speaker_mapping = rename_speakers(
        segments,
        diarized_turns,
        host_speaker,
        durations,
        known_assignments=known_assignments,
    )
    normalized_segments, replacement_events = coalesce_segments(segments, replacement_map)
    review_rows = collect_review_rows(
        source_file=str(audio_path),
        segments=normalized_segments,
        replacement_events=replacement_events,
        host_speaker=host_speaker,
        host_threshold=host_threshold,
        durations=durations,
        similarity_scores=similarity_scores,
        speaker_mapping=speaker_mapping,
    )
    print(
        f"  speaker matching complete: {len(speaker_mapping)} labeled speakers, "
        f"{len(review_rows)} review rows in {time.perf_counter() - matching_started:.1f}s"
    )

    print("  stage: writing outputs")
    writing_started = time.perf_counter()
    base_name = audio_path.stem
    resolved_host_label = speaker_mapping.get(host_speaker, "HOST") if host_speaker else "HOST"
    host_output_labels = {resolved_host_label, "HOST"}
    write_text_transcript(output_dir / f"{base_name}_speaker_transcript.txt", normalized_segments, host_only=False)
    write_text_transcript(
        output_dir / f"{base_name}_host_only.txt",
        normalized_segments,
        host_only=True,
        host_labels=host_output_labels,
    )
    write_review_csv(output_dir / f"{base_name}_review.csv", review_rows)
    write_json_output(
        output_dir / f"{base_name}_speaker_transcript.json",
        source_file=str(audio_path),
        info_payload=info_payload,
        diarized_turns=diarized_turns,
        segments=normalized_segments,
        speaker_mapping=speaker_mapping,
        host_speaker=host_speaker,
        durations=durations,
        known_assignments=known_assignments,
    )

    if updated_profile is not None and host_speaker is not None:
        save_host_profile(host_profile_path, updated_profile, str(audio_path))
    print(f"  writing complete in {time.perf_counter() - writing_started:.1f}s")

    total_segments = len(normalized_segments)
    host_segments = sum(1 for segment in normalized_segments if segment.speaker == "HOST")
    print(f"  review rows: {len(review_rows)}")
    print(f"  speaker segments: {total_segments}")
    print(f"  host segments: {host_segments}")
    print(f"  host detected: {host_speaker is not None}")
    return build_episode_summary_row(
        audio_path=audio_path,
        normalized_segments=normalized_segments,
        review_rows=review_rows,
        host_speaker=host_speaker,
        durations=durations,
        similarity_scores=similarity_scores,
        speaker_mapping=speaker_mapping,
        known_assignments=known_assignments,
    )


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not args.hf_token:
        raise RuntimeError(
            "A Hugging Face token is required for pyannote diarization. "
            "Set HF_TOKEN or pass --hf-token."
        )

    preferred_terms = load_preferred_terms(args.preferred_terms_file)
    initial_prompt, hotwords = build_prompt_bias(preferred_terms)
    replacement_map = load_replacement_map(args.replacement_map_json)

    device = get_device(args.device)
    print(f"Using device: {device}")
    whisper_model = WhisperModel(args.model, device=device, compute_type=args.compute_type)

    try:
        diarization_pipeline, resolved_diarization_model = load_diarization_pipeline(
            args.diarization_model, args.hf_token
        )
    except TypeError as exc:
        raise RuntimeError(
            "Failed to load the pyannote diarization model because this environment's pyannote.audio API "
            "does not match the loader call. The code now supports both 'token' and 'use_auth_token', so "
            "this likely indicates an unexpected pyannote.audio version or conflicting installation. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        message = str(exc).lower()
        if any(token_hint in message for token_hint in ["401", "403", "unauthorized", "forbidden", "access denied"]):
            raise RuntimeError(
                "Failed to load the pyannote diarization model because Hugging Face rejected the token or model access. "
                "Confirm the token value and make sure you have accepted access terms for "
                "pyannote/speaker-diarization-community-1."
            ) from exc
        if "plda" in message and "unexpected keyword argument" in message:
            raise RuntimeError(
                "Failed to load the diarization pipeline because the installed pyannote.audio version is not "
                "compatible with 'pyannote/speaker-diarization-community-1'. Upgrade to pyannote.audio 4.x for "
                "community-1, or use the legacy 'pyannote/speaker-diarization-3.1' pipeline with pyannote.audio 3.x."
            ) from exc
        raise RuntimeError(
            f"Failed to load diarization model '{args.diarization_model}'. Original error: {exc}"
        ) from exc
    print(f"Using diarization model: {resolved_diarization_model}")
    if device == "cuda":
        diarization_pipeline.to(torch.device(normalize_runtime_device(device)))

    verifier = load_speaker_verifier(args.speaker_model, device)
    known_speaker_profiles = load_known_speaker_profiles(
        verifier=verifier,
        known_speakers_dir=args.known_speakers_dir,
    )

    audio_files = sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise RuntimeError(f"No supported audio files found in {input_dir}")

    episode_summary_rows = []
    total_files = len(audio_files)
    batch_started = time.perf_counter()
    for index, audio_path in enumerate(audio_files, start=1):
        elapsed = time.perf_counter() - batch_started
        average_seconds = elapsed / (index - 1) if index > 1 else None
        remaining_files = total_files - index + 1
        eta_seconds = average_seconds * remaining_files if average_seconds is not None else None
        if eta_seconds is not None:
            print(
                f"Batch progress: file {index} of {total_files} "
                f"(estimated remaining {format_timestamp(eta_seconds)})"
            )
        else:
            print(f"Batch progress: file {index} of {total_files}")
        episode_summary_rows.append(
            process_file(
            audio_path=audio_path,
            output_dir=output_dir,
            whisper_model=whisper_model,
            diarization_pipeline=diarization_pipeline,
            verifier=verifier,
            language=args.language,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            replacement_map=replacement_map,
            host_reference=args.host_reference,
            host_profile_path=args.host_profile_json,
            known_speaker_profiles=known_speaker_profiles,
            host_threshold=args.host_threshold,
            assume_dominant=args.assume_dominant_speaker_is_host,
            max_embedding_seconds=args.max_embedding_seconds,
            min_host_seconds=args.min_host_seconds,
            num_speakers=args.num_speakers,
        )
        )

    write_episode_summary_csv(output_dir / "_episode_review_summary.csv", episode_summary_rows)
    print(f"Wrote folder summary: {output_dir / '_episode_review_summary.csv'}")


if __name__ == "__main__":
    main()
