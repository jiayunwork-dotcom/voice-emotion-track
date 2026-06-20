import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

from app.config import get_config
from app.emotion_classifier import EmotionResult, NEGATIVE_EMOTIONS


@dataclass
class TurningPoint:
    time: float
    valence_before: float
    valence_after: float
    delta: float


@dataclass
class EscalationInterval:
    start_time: float
    end_time: float
    avg_arousal: float


@dataclass
class ContagionEvent:
    source_speaker: str
    target_speaker: str
    source_time: float
    target_time: float
    delay_sentences: int
    contagion_strength: float


@dataclass
class DialogueSummary:
    dominant_emotion: str
    valence_std: float
    conflict_density: float
    turning_points: List[TurningPoint]
    escalation_intervals: List[EscalationInterval]
    contagion_events: List[ContagionEvent]


@dataclass
class SentenceEmotionResult:
    start_time: float
    end_time: float
    speaker_id: str
    emotion: str
    confidence: float
    valence: float
    arousal: float


def detect_turning_points(sentences: List[SentenceEmotionResult],
                          threshold: float = 0.4) -> List[TurningPoint]:
    points = []
    for i in range(1, len(sentences)):
        delta = abs(sentences[i].valence - sentences[i - 1].valence)
        if delta >= threshold:
            points.append(TurningPoint(
                time=sentences[i].start_time,
                valence_before=sentences[i - 1].valence,
                valence_after=sentences[i].valence,
                delta=round(delta, 3)
            ))
    return points


def detect_escalation_intervals(sentences: List[SentenceEmotionResult],
                                arousal_threshold: float = 0.6,
                                consecutive_count: int = 3) -> List[EscalationInterval]:
    intervals = []
    i = 0
    while i < len(sentences):
        if sentences[i].arousal > arousal_threshold:
            start_idx = i
            count = 0
            while i < len(sentences) and sentences[i].arousal > arousal_threshold:
                count += 1
                i += 1
            if count >= consecutive_count:
                escal_sentences = sentences[start_idx:i]
                avg_arousal = float(np.mean([s.arousal for s in escal_sentences]))
                intervals.append(EscalationInterval(
                    start_time=escal_sentences[0].start_time,
                    end_time=escal_sentences[-1].end_time,
                    avg_arousal=round(avg_arousal, 3)
                ))
            else:
                i = start_idx + 1
        else:
            i += 1
    return intervals


def detect_contagion(sentences: List[SentenceEmotionResult],
                     speakers: List[str],
                     window: int = 2) -> List[ContagionEvent]:
    if len(speakers) < 2:
        return []
    events = []
    speaker_a, speaker_b = speakers[0], speakers[1]
    for i, sent in enumerate(sentences):
        if sent.emotion not in NEGATIVE_EMOTIONS:
            continue
        source_speaker = sent.speaker_id
        target_speaker = speaker_b if source_speaker == speaker_a else speaker_a
        for j in range(i + 1, min(i + window + 1, len(sentences))):
            candidate = sentences[j]
            if candidate.speaker_id == target_speaker and candidate.emotion in NEGATIVE_EMOTIONS:
                delay = j - i
                strength = round(min(1.0, abs(sent.valence - 0.0) + abs(candidate.valence - 0.0)), 3)
                events.append(ContagionEvent(
                    source_speaker=source_speaker,
                    target_speaker=target_speaker,
                    source_time=sent.start_time,
                    target_time=candidate.start_time,
                    delay_sentences=delay,
                    contagion_strength=strength
                ))
                break
    return events


def compute_dialogue_summary(sentences: List[SentenceEmotionResult],
                             is_dual_speaker: bool = False) -> DialogueSummary:
    config = get_config()
    if not sentences:
        return DialogueSummary(
            dominant_emotion="neutral",
            valence_std=0.0,
            conflict_density=0.0,
            turning_points=[],
            escalation_intervals=[],
            contagion_events=[]
        )

    emotion_counts = Counter([s.emotion for s in sentences])
    dominant = emotion_counts.most_common(1)[0][0]

    valences = [s.valence for s in sentences]
    valence_std = round(float(np.std(valences)), 3) if len(valences) > 1 else 0.0

    threshold = config["turning_point_threshold"]
    turning_points = detect_turning_points(sentences, threshold)

    arousal_thresh = config["escalation_arousal_threshold"]
    consecutive = config["escalation_consecutive_count"]
    escalations = detect_escalation_intervals(sentences, arousal_thresh, consecutive)

    conflict_density = round(len(escalations) / len(sentences), 4) if sentences else 0.0

    contagion_events = []
    if is_dual_speaker:
        speakers = list(set([s.speaker_id for s in sentences]))
        if len(speakers) >= 2:
            window = config["contagion_window_sentences"]
            contagion_events = detect_contagion(sentences, speakers, window)

    return DialogueSummary(
        dominant_emotion=dominant,
        valence_std=valence_std,
        conflict_density=conflict_density,
        turning_points=turning_points,
        escalation_intervals=escalations,
        contagion_events=contagion_events
    )
