import asyncio
import json
import os
import time
import logging
import numpy as np
import librosa
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import tempfile

from app.config import get_config
from app.feature_extractor import extract_features, features_to_vector
from app.emotion_classifier import get_classifier, EmotionResult, EMOTION_LABELS

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "stream_sessions")
OVERFLOW_DIR = os.path.join(BASE_DIR, "data", "_overflow_tmp")
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(OVERFLOW_DIR, exist_ok=True)

VALID_MODEL_MODES = {"svm", "rf", "wav2vec2"}
MAX_ACTIVE_SESSIONS = 10
CONFIG_TIMEOUT_SEC = 30
ANALYSIS_TIMEOUT_MS = 500
RING_BUFFER_SIZE = 100


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
    results_buffer: deque = field(default_factory=lambda: deque(maxlen=RING_BUFFER_SIZE))
    config_received: bool = False
    closed: bool = False
    alert_threshold: int = 3
    alert_count: int = 0
    recent_emotions: List[str] = field(default_factory=list)
    emotion_counter: Counter = field(default_factory=Counter)
    total_valence: float = 0.0
    total_arousal: float = 0.0
    _overflow_file_path: Optional[str] = None


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
            overflow_path = _sessions[session_id]._overflow_file_path
            del _sessions[session_id]
    if overflow_path and os.path.exists(overflow_path):
        try:
            os.unlink(overflow_path)
        except Exception:
            pass


async def get_session(session_id: str) -> Optional[StreamSession]:
    async with _sessions_lock:
        return _sessions.get(session_id)


async def list_active_sessions() -> List[dict]:
    async with _sessions_lock:
        now = time.time()
        result = []
        for sid, sess in _sessions.items():
            last_emotion = sess.results_buffer[-1].emotion if sess.results_buffer else None
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
            return list(sess.results_buffer)
        return None


async def update_session_activity(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].last_active_at = time.time()


async def set_session_config(session_id: str, sample_rate: int, chunk_duration_ms: int,
                             alert_threshold: Optional[int] = None) -> bool:
    async with _sessions_lock:
        if session_id not in _sessions:
            return False
        sess = _sessions[session_id]
        sess.sample_rate = sample_rate
        sess.chunk_duration_ms = chunk_duration_ms
        sess.expected_bytes = _expected_byte_count(sample_rate, chunk_duration_ms)
        sess.config_received = True
        if alert_threshold is not None and alert_threshold > 0:
            sess.alert_threshold = alert_threshold
        return True


async def increment_frame_count(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].frame_count += 1
            _sessions[session_id].last_active_at = time.time()


async def append_frame_result(session_id: str, result: FrameResult) -> Optional[dict]:
    async with _sessions_lock:
        if session_id not in _sessions:
            return None
        sess = _sessions[session_id]
        if len(sess.results_buffer) == RING_BUFFER_SIZE:
            evicted = sess.results_buffer[0]
            _write_overflow(sess, evicted)
        sess.results_buffer.append(result)
        sess.analyzed_frames += 1
        sess.emotion_counter[result.emotion] += 1
        sess.total_valence += result.valence
        sess.total_arousal += result.arousal
        alert = _check_emotion_shift(sess, result)
        return alert


def _write_overflow(session: StreamSession, frame: FrameResult):
    if session._overflow_file_path is None:
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix=f"ovf_{session.session_id}_",
                                     dir=OVERFLOW_DIR)
        os.close(fd)
        session._overflow_file_path = path
    line = json.dumps({
        "seq": frame.seq,
        "emotion": frame.emotion,
        "confidence": frame.confidence,
        "valence": frame.valence,
        "arousal": frame.arousal,
        "timestamp_ms": frame.timestamp_ms
    }, ensure_ascii=False)
    try:
        with open(session._overflow_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.error(f"Failed to write overflow for session {session.session_id}: {e}")


def _read_overflow(session: StreamSession) -> List[FrameResult]:
    if session._overflow_file_path is None or not os.path.exists(session._overflow_file_path):
        return []
    results = []
    try:
        with open(session._overflow_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                results.append(FrameResult(
                    seq=d["seq"],
                    emotion=d["emotion"],
                    confidence=d["confidence"],
                    valence=d["valence"],
                    arousal=d["arousal"],
                    timestamp_ms=d["timestamp_ms"]
                ))
    except Exception as e:
        logger.error(f"Failed to read overflow for session {session.session_id}: {e}")
    return results


def _check_emotion_shift(session: StreamSession, result: FrameResult) -> Optional[dict]:
    threshold = session.alert_threshold
    session.recent_emotions.append(result.emotion)
    max_keep = threshold + 1
    if len(session.recent_emotions) > max_keep:
        session.recent_emotions = session.recent_emotions[-max_keep:]

    if len(session.recent_emotions) < threshold:
        return None

    last_n = session.recent_emotions[-threshold:]
    if len(set(last_n)) != 1:
        return None

    shift_emotion = last_n[0]

    if len(session.recent_emotions) > threshold:
        pre_streak = session.recent_emotions[-(threshold + 1)]
        if pre_streak == shift_emotion:
            return None

    dominant = session.emotion_counter.most_common(1)[0][0]
    if shift_emotion == dominant:
        return None

    session.alert_count += 1
    return {
        "from_emotion": dominant,
        "to_emotion": shift_emotion,
        "trigger_seq": result.seq,
        "sustained_frames": threshold
    }


async def increment_timeout_count(session_id: str):
    async with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].timeout_frames += 1


def compute_summary(session: StreamSession) -> dict:
    n = session.analyzed_frames
    duration = session.frame_count * (session.chunk_duration_ms or 0) / 1000.0

    if n == 0:
        return {
            "dominant_emotion": "neutral",
            "valence_mean": 0.0,
            "arousal_mean": 0.0,
            "total_frames": session.frame_count,
            "analyzed_frames": session.analyzed_frames,
            "timeout_frames": session.timeout_frames,
            "duration_seconds": round(duration, 2),
            "alert_count": session.alert_count
        }

    dominant = session.emotion_counter.most_common(1)[0][0]

    return {
        "dominant_emotion": dominant,
        "valence_mean": round(session.total_valence / n, 4),
        "arousal_mean": round(session.total_arousal / n, 4),
        "total_frames": session.frame_count,
        "analyzed_frames": session.analyzed_frames,
        "timeout_frames": session.timeout_frames,
        "duration_seconds": round(duration, 2),
        "alert_count": session.alert_count
    }


async def get_all_session_results(session_id: str) -> Optional[List[FrameResult]]:
    async with _sessions_lock:
        sess = _sessions.get(session_id)
        if sess is None:
            return None
        overflow = _read_overflow(sess)
        return overflow + list(sess.results_buffer)


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


def cleanup_overflow(session: StreamSession):
    if session._overflow_file_path and os.path.exists(session._overflow_file_path):
        try:
            os.unlink(session._overflow_file_path)
        except Exception:
            pass
