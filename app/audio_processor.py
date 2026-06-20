import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from pydub import AudioSegment
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.config import get_config


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    speaker_id: str = "speaker_0"


@dataclass
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    segments: List[AudioSegment] = field(default_factory=list)


def load_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    if file_path.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_path)
        channels = audio.channels
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            samples = samples.reshape((1, -1))
        samples = samples / (2 ** 15)
        sr = audio.frame_rate
        if target_sr and sr != target_sr:
            resampled = []
            for ch in range(channels):
                resampled.append(librosa.resample(samples[ch], orig_sr=sr, target_sr=target_sr))
            samples = np.array(resampled) if channels > 1 else np.array([resampled[0]])
            sr = target_sr
        return samples, sr, channels

    audio_data, sr = sf.read(file_path, dtype="float32")
    if audio_data.ndim == 1:
        channels = 1
        audio_data = audio_data.reshape((1, -1))
    else:
        channels = audio_data.shape[1]
        audio_data = audio_data.T
    if target_sr and sr != target_sr:
        resampled = []
        for ch in range(channels):
            resampled.append(librosa.resample(audio_data[ch], orig_sr=sr, target_sr=target_sr))
        audio_data = np.array(resampled)
        sr = target_sr
    return audio_data, sr, channels


def detect_voice_activity(audio: np.ndarray, sr: int,
                          frame_length: int = 2048, hop_length: int = 512,
                          energy_threshold_db: float = -40) -> List[Tuple[float, float]]:
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    threshold = np.max(rms_db) + energy_threshold_db
    is_voiced = rms_db > threshold
    frame_times = librosa.frames_to_time(np.arange(len(is_voiced)), sr=sr, hop_length=hop_length)
    segments = []
    in_segment = False
    start = 0.0
    for i, voiced in enumerate(is_voiced):
        if voiced and not in_segment:
            start = frame_times[i]
            in_segment = True
        elif not voiced and in_segment:
            end = frame_times[i]
            segments.append((start, end))
            in_segment = False
    if in_segment:
        segments.append((start, frame_times[-1]))
    return segments


def merge_segments(segments: List[Tuple[float, float]],
                   silence_threshold: float = 0.5) -> List[Tuple[float, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= silence_threshold:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def filter_short_segments(segments: List[Tuple[float, float]],
                          min_duration: float = 0.3) -> List[Tuple[float, float]]:
    return [(s, e) for s, e in segments if (e - s) >= min_duration]


def segments_to_audio_segments(audio: np.ndarray, sr: int,
                               time_segments: List[Tuple[float, float]],
                               speaker_id: str = "speaker_0") -> List[AudioSegment]:
    result = []
    for start, end in time_segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        end_sample = min(end_sample, len(audio))
        seg_audio = audio[start_sample:end_sample]
        result.append(AudioSegment(
            start_time=round(start, 3),
            end_time=round(end, 3),
            audio_data=seg_audio,
            speaker_id=speaker_id
        ))
    return result


def process_audio(file_path: str, is_dual_channel: bool = False,
                  model_mode: str = "svm") -> AudioInfo:
    config = get_config()
    target_sr = config["wav2vec_sample_rate"] if model_mode == "wav2vec2" else config["default_sample_rate"]
    audio_data, sr, channels = load_audio(file_path, target_sr=target_sr)
    duration = audio_data.shape[1] / sr
    all_segments = []
    silence_threshold = config["vad_silence_threshold_sec"]
    min_duration = config["vad_min_segment_sec"]

    if is_dual_channel and channels >= 2:
        for ch_idx in range(2):
            ch_audio = audio_data[ch_idx]
            raw_segments = detect_voice_activity(ch_audio, sr)
            merged = merge_segments(raw_segments, silence_threshold)
            filtered = filter_short_segments(merged, min_duration)
            speaker_id = f"speaker_{ch_idx}"
            ch_segments = segments_to_audio_segments(ch_audio, sr, filtered, speaker_id)
            all_segments.extend(ch_segments)
        all_segments.sort(key=lambda s: s.start_time)
    else:
        ch_audio = audio_data[0]
        raw_segments = detect_voice_activity(ch_audio, sr)
        merged = merge_segments(raw_segments, silence_threshold)
        filtered = filter_short_segments(merged, min_duration)
        all_segments = segments_to_audio_segments(ch_audio, sr, filtered, "speaker_0")

    return AudioInfo(
        duration=round(duration, 3),
        sample_rate=sr,
        channels=channels,
        segments=all_segments
    )
