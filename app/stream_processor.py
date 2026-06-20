import asyncio
import json
import os
import time
import logging
import numpy as np
import librosa
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from app.config import get_config
from app.feature_extractor import extract_features, features_to_vector
from app.emotion_classifier import get_classifier, EmotionResult, EMOTION_LABELS

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "stream_sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

VALID_MODEL_MODES = {"svm", "rf", "wav2vec2"}
MAX_ACTIVE_SESSIONS = 10
CONFIG_TIMEOUT_SEC = 30
ANALYSIS_TIMEOUT_MS = 500


@dataclass
class FrameResult:
    seq: int
    emotion: str
    confidence: float
    valence: float
    arousal: float
    timestamp_ms: int


@dataclass
class StreamSession:
    session_id: str
    model_mode: str
    status: str = "connected"
    sample_rate: Optional[int] = None
    chunk_duration_ms: Optional[int] = None
    expected_bytes: Optional[int] = None
    frame_count: int = 0
    analyzed_frames: int = 0
    timeout_frames: int = 0
    started_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    results: List[FrameResult] = field(default_factory=list)
    config_received: bool = False
    closed: bool = False


_sessions: Dict[str, StreamSession] = {}
_sessions_lock = asyncio.Lock()


def _expected_byte_count(sample_rate: int, chunk_duration_ms: int) -> int:
    return int(sample_rate * chunk_duration_ms / 1000 * 2)


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def _fast_extract_features(audio: np.ndarray, sr: int,
                           start_time: float, end_time: float,
                           speaker_id: str) -> "SegmentFeatures":
    from app.feature_extractor import SegmentFeatures
    feat = SegmentFeatures(start_time=start_time, end_time=end_time, speaker_id=speaker_id)

    n_fft = min(512, len(audio))
    hop_length = n_fft // 4

    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)[0]
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

    def _quick_stats(arr: np.ndarray) -> Dict[str, float]:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid))
        }

    aggregated = {}
    for prefix, data in [("rms", rms), ("zcr", zcr), ("sc", sc), ("sb", sb)]:
        stats = _quick_stats(data)
        for k, v in stats.items():
            aggregated[f"{prefix}_{k}"] = v

    for i in range(200):
        if f"dim{i}_mean" not in aggregated:
            aggregated[f"dim{i}_mean"] = 0.0
            aggregated[f"dim{i}_std"] = 0.0
            aggregated[f"dim{i}_min"] = 0.0
            aggregated[f"dim{i}_max"] = 0.0

    feat.aggregated = aggregated
    return feat


def _analyze_frame_sync(audio_f32: np.ndarray, sr: int, classifier,
                        seq: int, timestamp_ms: int) -> FrameResult:
    features = _fast_extract_features(audio_f32, sr, 0.0, len(audio_f32) / sr, "speaker_0")
    result = classifier.predict(audio_f32, sr, features)
    return FrameResult(
        seq=seq,
        emotion=result.emotion,
        confidence=result.confidence,
        valence=result.valence,
        arousal=result.arousal,
        timestamp_ms=timestamp_ms
    )


async def analyze_frame_with_timeout(audio_f32: np.ndarray, sr: int, classifier,
                                     seq: int, timestamp_ms: int,
                                     timeout_ms: int = ANALYSIS_TIMEOUT_MS) -> Optional[FrameResult]:
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None, _analyze_frame_sync, audio_f32, sr, classifier, seq, timestamp_ms
            ),
            timeout=timeout_ms / 1000.0
        )
        return result
    except asyncio.TimeoutError:
        return None


async def register_session(session_id: str, model_mode: str) -> Optional[str]:
    async with _sessions_lock:
        if len(_sessions) >= MAX_ACTIVE_SESSIONS:
            return "Server at capacity. Maximum 10 concurrent sessions allowed."
        if session_id in _sessions and not _sessions[session_id].closed:
            return f"Session {session_id} is already connected."
        _sessions[session_id] = StreamSession(
            session_id=session_id,
            model_mode=model_mode
        )
        return None


async def unregister_session(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].closed = True
            _sessions[session_id].status = "disconnected"
            del _sessions[session_id]


async def get_session(session_id: str) -> Optional[StreamSession]:
    async with _sessions_lock:
        return _sessions.get(session_id)


async def list_active_sessions() -> List[dict]:
    async with _sessions_lock:
        now = time.time()
        result = []
        for sid, sess in _sessions.items():
            last_emotion = sess.results[-1].emotion if sess.results else None
            result.append({
                "session_id": sid,
                "status": sess.status,
                "frame_count": sess.frame_count,
                "analyzed_frames": sess.analyzed_frames,
                "timeout_frames": sess.timeout_frames,
                "connection_duration_seconds": round(now - sess.started_at, 2),
                "model_mode": sess.model_mode,
                "last_emotion": last_emotion,
                "last_active_at": datetime.fromtimestamp(sess.last_active_at).isoformat()
            })
        return result


async def get_session_results(session_id: str) -> Optional[List[FrameResult]]:
    async with _sessions_lock:
        sess = _sessions.get(session_id)
        if sess:
            return list(sess.results)
        return None


async def update_session_activity(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].last_active_at = time.time()


async def set_session_config(session_id: str, sample_rate: int, chunk_duration_ms: int) -> bool:
    async with _sessions_lock:
        if session_id not in _sessions:
            return False
        sess = _sessions[session_id]
        sess.sample_rate = sample_rate
        sess.chunk_duration_ms = chunk_duration_ms
        sess.expected_bytes = _expected_byte_count(sample_rate, chunk_duration_ms)
        sess.config_received = True
        return True


async def increment_frame_count(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].frame_count += 1
            _sessions[session_id].last_active_at = time.time()


async def append_frame_result(session_id: str, result: FrameResult):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].results.append(result)
            _sessions[session_id].analyzed_frames += 1


async def increment_timeout_count(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].timeout_frames += 1


def compute_summary(session: StreamSession) -> dict:
    if not session.results:
        return {
            "dominant_emotion": "neutral",
            "valence_mean": 0.0,
            "arousal_mean": 0.0,
            "total_frames": session.frame_count,
            "analyzed_frames": session.analyzed_frames,
            "timeout_frames": session.timeout_frames,
            "duration_seconds": round(session.frame_count * (session.chunk_duration_ms or 0) / 1000.0, 2)
        }

    emotions = [r.emotion for r in session.results]
    counter = Counter(emotions)
    dominant = counter.most_common(1)[0][0]

    valences = [r.valence for r in session.results]
    arousals = [r.arousal for r in session.results]

    duration = session.frame_count * (session.chunk_duration_ms or 0) / 1000.0

    return {
        "dominant_emotion": dominant,
        "valence_mean": round(float(np.mean(valences)), 4),
        "arousal_mean": round(float(np.mean(arousals)), 4),
        "total_frames": session.frame_count,
        "analyzed_frames": session.analyzed_frames,
        "timeout_frames": session.timeout_frames,
        "duration_seconds": round(duration, 2)
    }


def persist_summary(session_id: str, summary: dict, results: List[FrameResult]):
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    data = {
        "session_id": session_id,
        "summary": summary,
        "frames": [
            {
                "seq": r.seq,
                "emotion": r.emotion,
                "confidence": r.confidence,
                "valence": r.valence,
                "arousal": r.arousal,
                "timestamp_ms": r.timestamp_ms
            }
            for r in results
        ],
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Session summary persisted: {file_path}")
    except Exception as e:
        logger.error(f"Failed to persist session {session_id}: {e}")
