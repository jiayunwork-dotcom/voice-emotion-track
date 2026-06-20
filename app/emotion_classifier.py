import numpy as np
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib

from app.config import get_config
from app.feature_extractor import SegmentFeatures, features_to_vector

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]

NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}
POSITIVE_EMOTIONS = {"happy", "surprise"}

EMOTION_VALENCE = {
    "neutral": 0.0, "happy": 0.7, "sad": -0.7,
    "angry": -0.6, "fear": -0.5, "surprise": 0.2, "disgust": -0.6
}
EMOTION_AROUSAL = {
    "neutral": 0.0, "happy": 0.5, "sad": -0.3,
    "angry": 0.7, "fear": 0.6, "surprise": 0.6, "disgust": 0.3
}


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    valence: float
    arousal: float


class BaseEmotionClassifier(ABC):
    @abstractmethod
    def predict(self, audio_data: np.ndarray, sr: int,
                features: SegmentFeatures) -> EmotionResult:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass


class SVMClassifier(BaseEmotionClassifier):
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self._loaded = False
        self._n_features = None
        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            self._init_default()

    def _load(self, model_path: str):
        data = joblib.load(model_path)
        self.model = data.get("model")
        self.scaler = data.get("scaler", StandardScaler())
        self._loaded = True

    def _init_default(self):
        self.model = CalibratedClassifierCV(SVC(kernel='rbf', random_state=42), ensemble=False)
        self.scaler = StandardScaler()
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 200
        X = rng.randn(n_samples, n_features)
        y = np.array([i % 7 for i in range(n_samples)])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._n_features = n_features
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, audio_data: np.ndarray, sr: int,
                features: SegmentFeatures) -> EmotionResult:
        vec = features_to_vector(features)
        if self._n_features and len(vec) < self._n_features:
            vec = np.pad(vec, (0, self._n_features - len(vec)))
        elif self._n_features and len(vec) > self._n_features:
            vec = vec[:self._n_features]
        vec = vec.reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        proba = self.model.predict_proba(vec_scaled)[0]
        class_idx = np.argmax(proba)
        emotion = EMOTION_LABELS[class_idx]
        confidence = float(proba[class_idx])
        valence = EMOTION_VALENCE[emotion] + np.random.uniform(-0.1, 0.1)
        arousal = EMOTION_AROUSAL[emotion] + np.random.uniform(-0.1, 0.1)
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        return EmotionResult(emotion=emotion, confidence=round(confidence, 4),
                             valence=round(valence, 3), arousal=round(arousal, 3))


class RFClassifier(BaseEmotionClassifier):
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self._loaded = False
        self._n_features = None
        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            self._init_default()

    def _load(self, model_path: str):
        data = joblib.load(model_path)
        self.model = data.get("model")
        self.scaler = data.get("scaler", StandardScaler())
        self._loaded = True

    def _init_default(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 200
        X = rng.randn(n_samples, n_features)
        y = np.array([i % 7 for i in range(n_samples)])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._n_features = n_features
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, audio_data: np.ndarray, sr: int,
                features: SegmentFeatures) -> EmotionResult:
        vec = features_to_vector(features)
        if self._n_features and len(vec) < self._n_features:
            vec = np.pad(vec, (0, self._n_features - len(vec)))
        elif self._n_features and len(vec) > self._n_features:
            vec = vec[:self._n_features]
        vec = vec.reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        proba = self.model.predict_proba(vec_scaled)[0]
        class_idx = np.argmax(proba)
        emotion = EMOTION_LABELS[class_idx]
        confidence = float(proba[class_idx])
        valence = EMOTION_VALENCE[emotion] + np.random.uniform(-0.1, 0.1)
        arousal = EMOTION_AROUSAL[emotion] + np.random.uniform(-0.1, 0.1)
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        return EmotionResult(emotion=emotion, confidence=round(confidence, 4),
                             valence=round(valence, 3), arousal=round(arousal, 3))


class Wav2Vec2Classifier(BaseEmotionClassifier):
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        self.model = None
        self.processor = None
        self.classifier_head = None
        self._loaded = False
        self.model_name = model_name
        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"Wav2Vec2 model load failed: {e}. Will lazy-load on first predict.")

    def _load_model(self):
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            import torch
            import torch.nn as nn
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            for param in self.model.parameters():
                param.requires_grad = False
            hidden_size = self.model.config.hidden_size
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 7)
            )
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2: {e}")
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, audio_data: np.ndarray, sr: int,
                features: SegmentFeatures) -> EmotionResult:
        if not self._loaded:
            self._load_model()
        if not self._loaded:
            return EmotionResult(emotion="neutral", confidence=0.0,
                                 valence=0.0, arousal=0.0)
        import torch
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = self.processor(audio_data, sampling_rate=sr, return_tensors="pt",
                                padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            logits = self.classifier_head(pooled)
            proba = torch.softmax(logits, dim=-1).numpy()[0]
        class_idx = int(np.argmax(proba))
        emotion = EMOTION_LABELS[class_idx]
        confidence = float(proba[class_idx])
        valence = EMOTION_VALENCE[emotion] + np.random.uniform(-0.1, 0.1)
        arousal = EMOTION_AROUSAL[emotion] + np.random.uniform(-0.1, 0.1)
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        return EmotionResult(emotion=emotion, confidence=round(confidence, 4),
                             valence=round(valence, 3), arousal=round(arousal, 3))


def get_classifier(mode: str = "svm") -> BaseEmotionClassifier:
    if mode == "svm":
        return SVMClassifier()
    elif mode == "rf":
        return RFClassifier()
    elif mode == "wav2vec2":
        return Wav2Vec2Classifier()
    else:
        raise ValueError(f"Unknown model mode: {mode}")
