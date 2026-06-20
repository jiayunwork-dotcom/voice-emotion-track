import numpy as np
import librosa
from scipy.stats import linregress
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SegmentFeatures:
    start_time: float
    end_time: float
    speaker_id: str
    mfcc_stats: Dict[str, np.ndarray] = field(default_factory=dict)
    mel_spectrogram: Optional[np.ndarray] = None
    f0_contour: Optional[np.ndarray] = None
    energy_envelope: Optional[np.ndarray] = None
    speech_rate: float = 0.0
    pause_count: int = 0
    avg_pause_duration: float = 0.0
    f0_range: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    aggregated: Dict[str, float] = field(default_factory=dict)


def _aggregate(frame_features: np.ndarray, prefix: str = "") -> Dict[str, float]:
    result = {}
    p = prefix + "_" if prefix else ""
    if frame_features.ndim == 1:
        valid = frame_features[~np.isnan(frame_features)]
        if len(valid) == 0:
            return {f"{p}mean": np.nan, f"{p}std": np.nan,
                    f"{p}min": np.nan, f"{p}max": np.nan, f"{p}slope": np.nan}
        result[f"{p}mean"] = float(np.mean(valid))
        result[f"{p}std"] = float(np.std(valid))
        result[f"{p}min"] = float(np.min(valid))
        result[f"{p}max"] = float(np.max(valid))
        if len(valid) > 1:
            slope = linregress(np.arange(len(valid)), valid).slope
            result[f"{p}slope"] = float(slope)
        else:
            result[f"{p}slope"] = 0.0
    else:
        for i in range(frame_features.shape[0]):
            sub = _aggregate(frame_features[i], f"{prefix}{i}" if prefix else f"dim{i}")
            result.update(sub)
    return result


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta1, delta2])


def extract_mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def extract_f0_pyin(audio: np.ndarray, sr: int) -> np.ndarray:
    f0, voiced_flags, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    return f0


def extract_energy_envelope(audio: np.ndarray, sr: int,
                            frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def estimate_speech_rate(audio: np.ndarray, sr: int) -> float:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(audio) / sr
    if duration <= 0:
        return 0.0
    syllable_count = max(len(onset_frames), 1)
    return syllable_count / duration


def compute_pause_stats(audio: np.ndarray, sr: int,
                        hop_length: int = 512, energy_threshold_db: float = -40) -> tuple:
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_voiced = rms_db > (np.max(rms_db) + energy_threshold_db)
    pause_count = 0
    pause_durations = []
    in_pause = False
    pause_start = 0
    for i, voiced in enumerate(is_voiced):
        if not voiced and not in_pause:
            in_pause = True
            pause_start = i
        elif voiced and in_pause:
            pause_frames = i - pause_start
            if pause_frames >= 2:
                pause_count += 1
                pause_dur = librosa.frames_to_time(pause_frames, sr=sr, hop_length=hop_length)
                pause_durations.append(pause_dur)
            in_pause = False
    avg_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
    return pause_count, avg_pause


def compute_jitter(f0: np.ndarray) -> float:
    valid = f0[~np.isnan(f0)]
    if len(valid) < 2:
        return np.nan
    periods = 1.0 / valid
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs)) if len(diffs) > 0 else np.nan


def compute_shimmer(audio: np.ndarray, sr: int, hop_length: int = 512) -> float:
    S = np.abs(librosa.stft(audio, hop_length=hop_length))
    amplitudes = np.max(S, axis=0)
    if len(amplitudes) < 2:
        return np.nan
    diffs = np.abs(np.diff(amplitudes))
    mean_amp = np.mean(amplitudes[:-1])
    if mean_amp == 0:
        return np.nan
    return float(np.mean(diffs) / mean_amp)


def extract_features(audio: np.ndarray, sr: int,
                     start_time: float, end_time: float,
                     speaker_id: str) -> SegmentFeatures:
    feat = SegmentFeatures(start_time=start_time, end_time=end_time, speaker_id=speaker_id)

    feat.mfcc_stats = extract_mfcc(audio, sr)
    feat.mel_spectrogram = extract_mel_spectrogram(audio, sr)
    feat.f0_contour = extract_f0_pyin(audio, sr)
    feat.energy_envelope = extract_energy_envelope(audio, sr)
    feat.speech_rate = estimate_speech_rate(audio, sr)
    feat.pause_count, feat.avg_pause_duration = compute_pause_stats(audio, sr)

    f0_valid = feat.f0_contour[~np.isnan(feat.f0_contour)] if feat.f0_contour is not None else np.array([])
    if len(f0_valid) > 0:
        feat.f0_range = float(np.max(f0_valid) - np.min(f0_valid))
    else:
        feat.f0_range = np.nan

    feat.jitter = compute_jitter(feat.f0_contour) if feat.f0_contour is not None else np.nan
    feat.shimmer = compute_shimmer(audio, sr)

    aggregated = {}

    mfcc_agg = _aggregate(feat.mfcc_stats, "mfcc")
    aggregated.update(mfcc_agg)

    if feat.energy_envelope is not None:
        energy_agg = _aggregate(feat.energy_envelope, "energy")
        aggregated.update(energy_agg)

    if feat.f0_contour is not None:
        f0_agg = _aggregate(feat.f0_contour, "f0")
        aggregated.update(f0_agg)

    aggregated["speech_rate"] = feat.speech_rate
    aggregated["pause_count"] = float(feat.pause_count)
    aggregated["avg_pause_duration"] = feat.avg_pause_duration
    aggregated["f0_range"] = feat.f0_range if not np.isnan(feat.f0_range) else 0.0
    aggregated["jitter"] = feat.jitter if not (isinstance(feat.jitter, float) and np.isnan(feat.jitter)) else 0.0
    aggregated["shimmer"] = feat.shimmer if not (isinstance(feat.shimmer, float) and np.isnan(feat.shimmer)) else 0.0

    feat.aggregated = aggregated
    return feat


def features_to_vector(feat: SegmentFeatures) -> np.ndarray:
    values = []
    for k, v in sorted(feat.aggregated.items()):
        val = v if not (isinstance(v, float) and np.isnan(v)) else 0.0
        values.append(val)
    return np.array(values, dtype=np.float32)
