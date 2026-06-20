import uuid
import time
import asyncio
import logging
import os
import tempfile
import json
import csv
import io
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from enum import Enum

import numpy as np
from scipy.stats import linregress, pearsonr

from app.config import get_config
from app.audio_processor import process_audio, AudioInfo
from app.feature_extractor import extract_features
from app.emotion_classifier import get_classifier, EMOTION_LABELS
from app.dialogue_tracker import (
    SentenceEmotionResult, compute_dialogue_summary, DialogueSummary
)

logger = logging.getLogger(__name__)


class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchConfig:
    batch_name: str
    dimensions: List[str]
    baseline_file: Optional[str] = None


@dataclass
class BatchFile:
    file_id: str
    filename: str
    temp_path: str
    status: FileStatus = FileStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class BatchTask:
    batch_id: str
    config: BatchConfig
    files: List[BatchFile]
    status: BatchStatus = BatchStatus.QUEUED
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    report: Optional[dict] = None


_batches_store: Dict[str, BatchTask] = {}
_batches_lock = asyncio.Lock()
_file_status_lock = threading.Lock()

MAX_BATCH_CONCURRENT = 3
_batch_semaphore = asyncio.Semaphore(MAX_BATCH_CONCURRENT)
_batch_queue_count = 0
_batch_queue_lock = asyncio.Lock()

MAX_FILE_CONCURRENT = 3

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_file_extension(filename: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(np.where((a != 0) & (b != 0), a * np.log2(a / b), 0))
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(js)


def classify_valence_trend(valences: List[float]) -> Tuple[str, float]:
    if len(valences) < 2:
        return "stationary", 0.0
    x = np.arange(len(valences))
    y = np.array(valences)
    slope, _, _, _, _ = linregress(x, y)
    if slope > 0.01:
        trend = "rising"
    elif slope < -0.01:
        trend = "falling"
    else:
        trend = "stationary"
    return trend, float(slope)


def classify_arousal_pattern(arousals: List[float]) -> Tuple[str, float, List[Tuple[float, float]]]:
    if len(arousals) < 2:
        return "low", 0.0, []
    arousal_arr = np.array(arousals)
    std = float(np.std(arousal_arr))
    if std > 0.4:
        pattern = "high_volatility"
    elif std > 0.2:
        pattern = "medium_volatility"
    else:
        pattern = "low_volatility"
    sharp_intervals = []
    for i in range(1, len(arousals)):
        if abs(arousals[i] - arousals[i-1]) > 0.5:
            sharp_intervals.append((i-1, i))
    return pattern, std, sharp_intervals


def compute_speaker_synchronization(sentences: List[SentenceEmotionResult]) -> Optional[dict]:
    speakers = list(set([s.speaker_id for s in sentences]))
    if len(speakers) < 2:
        return None
    spk_a, spk_b = speakers[0], speakers[1]
    a_sents = [s for s in sentences if s.speaker_id == spk_a]
    b_sents = [s for s in sentences if s.speaker_id == spk_b]
    if len(a_sents) < 3 or len(b_sents) < 3:
        return None
    min_len = min(len(a_sents), len(b_sents))
    a_valence = [s.valence for s in a_sents[:min_len]]
    b_valence = [s.valence for s in b_sents[:min_len]]
    a_arousal = [s.arousal for s in a_sents[:min_len]]
    b_arousal = [s.arousal for s in b_sents[:min_len]]
    try:
        valence_corr, _ = pearsonr(a_valence, b_valence)
        arousal_corr, _ = pearsonr(a_arousal, b_arousal)
    except Exception:
        valence_corr = 0.0
        arousal_corr = 0.0
    sync_score = (valence_corr + arousal_corr) / 2
    return {
        "speaker_a": spk_a,
        "speaker_b": spk_b,
        "valence_correlation": round(float(valence_corr), 4),
        "arousal_correlation": round(float(arousal_corr), 4),
        "synchronization_score": round(float(sync_score), 4),
        "alignment_level": "high" if sync_score > 0.5 else ("medium" if sync_score > 0 else "low")
    }


def get_emotion_distribution(sentences: List[SentenceEmotionResult]) -> Dict[str, float]:
    if not sentences:
        return {e: 0.0 for e in EMOTION_LABELS}
    counts = Counter([s.emotion for s in sentences])
    total = len(sentences)
    return {e: round(counts.get(e, 0) / total, 4) for e in EMOTION_LABELS}


def _process_single_file_sync(
    file_path: str,
    model_mode: str,
    is_dual_channel: bool,
    timeout: float
) -> dict:
    start_time = time.time()
    audio_info = process_audio(
        file_path,
        is_dual_channel=is_dual_channel,
        model_mode=model_mode
    )
    classifier = get_classifier(model_mode)
    sentence_results = []
    is_dual = is_dual_channel and audio_info.channels >= 2
    deadline = start_time + timeout

    for seg in audio_info.segments:
        if time.time() > deadline:
            raise TimeoutError(f"Single file processing timed out after {timeout}s")
        features = extract_features(
            seg.audio_data, audio_info.sample_rate,
            seg.start_time, seg.end_time, seg.speaker_id
        )
        emotion_result = classifier.predict(
            seg.audio_data, audio_info.sample_rate, features
        )
        sentence_results.append(SentenceEmotionResult(
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_id=seg.speaker_id,
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
            valence=emotion_result.valence,
            arousal=emotion_result.arousal
        ))

    dialogue_summary = compute_dialogue_summary(
        sentence_results, is_dual_speaker=is_dual
    )

    return {
        "audio_info": {
            "duration": audio_info.duration,
            "sample_rate": audio_info.sample_rate,
            "channels": audio_info.channels
        },
        "sentences": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "speaker_id": s.speaker_id,
                "emotion": s.emotion,
                "confidence": s.confidence,
                "valence": s.valence,
                "arousal": s.arousal
            }
            for s in sentence_results
        ],
        "sentence_objects": sentence_results,
        "dialogue_summary": dialogue_summary,
        "processing_time": round(time.time() - start_time, 2)
    }


def generate_report(batch: BatchTask) -> dict:
    config = batch.config
    files = batch.files

    completed_files = [f for f in files if f.status == FileStatus.COMPLETED]
    failed_files = [f for f in files if f.status == FileStatus.FAILED]

    meta = {
        "batch_name": config.batch_name,
        "batch_id": batch.batch_id,
        "total_files": len(files),
        "success_count": len(completed_files),
        "failed_count": len(failed_files),
        "total_duration_seconds": round(
            (batch.completed_at or time.time()) - (batch.started_at or time.time()), 2
        ),
        "baseline_file": config.baseline_file,
        "selected_dimensions": config.dimensions,
        "started_at": datetime.fromtimestamp(batch.started_at).isoformat() if batch.started_at else None,
        "completed_at": datetime.fromtimestamp(batch.completed_at).isoformat() if batch.completed_at else None
    }

    file_summaries = {}
    for f in files:
        if f.status == FileStatus.COMPLETED and f.result:
            summary = f.result["dialogue_summary"]
            file_summaries[f.file_id] = {
                "file_id": f.file_id,
                "filename": f.filename,
                "status": f.status.value,
                "processing_time_seconds": f.result.get("processing_time", 0),
                "audio_info": f.result["audio_info"],
                "dialogue_summary": {
                    "dominant_emotion": summary.dominant_emotion,
                    "valence_std": summary.valence_std,
                    "conflict_density": summary.conflict_density,
                    "turning_point_count": len(summary.turning_points),
                    "escalation_interval_count": len(summary.escalation_intervals),
                    "contagion_event_count": len(summary.contagion_events)
                }
            }
        else:
            file_summaries[f.file_id] = {
                "file_id": f.file_id,
                "filename": f.filename,
                "status": f.status.value,
                "error": f.error,
                "available": False
            }

    comparison = {}

    if "emotion_distribution" in config.dimensions:
        dist_data = {}
        for f in completed_files:
            if f.result:
                sents = f.result["sentence_objects"]
                dist_data[f.file_id] = get_emotion_distribution(sents)

        file_ids = list(dist_data.keys())
        js_matrix = {}
        max_js = 0.0
        max_pair = None
        for i in range(len(file_ids)):
            for j in range(i + 1, len(file_ids)):
                fid1, fid2 = file_ids[i], file_ids[j]
                p = np.array([dist_data[fid1][e] for e in EMOTION_LABELS])
                q = np.array([dist_data[fid2][e] for e in EMOTION_LABELS])
                js = compute_js_divergence(p, q)
                js_matrix[f"{fid1}__{fid2}"] = round(js, 6)
                if js > max_js:
                    max_js = js
                    max_pair = (fid1, fid2)

        emotion_result = {
            "distributions": {
                fid: dist
                for fid, dist in dist_data.items()
            },
            "js_divergence_matrix": js_matrix,
            "most_divergent_pair": {
                "file1": max_pair[0] if max_pair else None,
                "file2": max_pair[1] if max_pair else None,
                "js_divergence": round(max_js, 6)
            }
        }

        if config.baseline_file and config.baseline_file in dist_data:
            baseline_dist = np.array([dist_data[config.baseline_file][e] for e in EMOTION_LABELS])
            deviations = {}
            for fid in file_ids:
                if fid != config.baseline_file:
                    f_dist = np.array([dist_data[fid][e] for e in EMOTION_LABELS])
                    js = compute_js_divergence(baseline_dist, f_dist)
                    deviations[fid] = {
                        "js_divergence": round(js, 6),
                        "direction": "more_positive" if f_dist[1] > baseline_dist[1] else "more_negative",
                        "per_emotion_deviation": {
                            e: round(f_dist[i] - baseline_dist[i], 4)
                            for i, e in enumerate(EMOTION_LABELS)
                        }
                    }
            emotion_result["baseline_deviations"] = deviations

        comparison["emotion_distribution"] = emotion_result

    if "valence_trend" in config.dimensions:
        trend_data = {}
        for f in completed_files:
            if f.result:
                sents = f.result["sentence_objects"]
                valences = [s.valence for s in sents]
                trend, slope = classify_valence_trend(valences)
                trend_data[f.file_id] = {
                    "trend": trend,
                    "slope": round(slope, 6),
                    "mean_valence": round(float(np.mean(valences)) if valences else 0.0, 4),
                    "valence_values": valences
                }

        all_slopes = [d["slope"] for d in trend_data.values()]
        mean_slope = float(np.mean(all_slopes)) if all_slopes else 0.0
        std_slope = float(np.std(all_slopes)) if len(all_slopes) > 1 else 0.0
        anomalous = [
            fid for fid, d in trend_data.items()
            if abs(d["slope"] - mean_slope) > 2 * std_slope
        ] if std_slope > 0 else []

        valence_result = {
            "trends": trend_data,
            "mean_slope": round(mean_slope, 6),
            "anomalous_files": anomalous
        }

        if config.baseline_file and config.baseline_file in trend_data:
            baseline_slope = trend_data[config.baseline_file]["slope"]
            deviations = {}
            for fid, d in trend_data.items():
                if fid != config.baseline_file:
                    dev = d["slope"] - baseline_slope
                    deviations[fid] = {
                        "slope_deviation": round(dev, 6),
                        "direction": "steeper" if dev > 0 else "shallower",
                        "relative_deviation_percent": round(abs(dev / baseline_slope * 100), 2) if baseline_slope != 0 else None
                    }
            valence_result["baseline_deviations"] = deviations

        comparison["valence_trend"] = valence_result

    if "arousal_pattern" in config.dimensions:
        arousal_data = {}
        for f in completed_files:
            if f.result:
                sents = f.result["sentence_objects"]
                arousals = [s.arousal for s in sents]
                pattern, std, sharp = classify_arousal_pattern(arousals)
                escalation_count = len(f.result["dialogue_summary"].escalation_intervals)
                arousal_data[f.file_id] = {
                    "pattern": pattern,
                    "arousal_std": round(std, 4),
                    "mean_arousal": round(float(np.mean(arousals)) if arousals else 0.0, 4),
                    "sharp_change_count": len(sharp),
                    "escalation_interval_count": escalation_count,
                    "has_escalation": escalation_count > 0
                }

        arousal_result = {
            "patterns": arousal_data,
            "files_with_escalation": [
                fid for fid, d in arousal_data.items() if d["has_escalation"]
            ]
        }

        if config.baseline_file and config.baseline_file in arousal_data:
            baseline_std = arousal_data[config.baseline_file]["arousal_std"]
            deviations = {}
            for fid, d in arousal_data.items():
                if fid != config.baseline_file:
                    dev = d["arousal_std"] - baseline_std
                    deviations[fid] = {
                        "std_deviation": round(dev, 4),
                        "direction": "more_volatile" if dev > 0 else "less_volatile",
                        "escalation_count_delta": d["escalation_interval_count"] - arousal_data[config.baseline_file]["escalation_interval_count"]
                    }
            arousal_result["baseline_deviations"] = deviations

        comparison["arousal_pattern"] = arousal_result

    if "speaker_similarity" in config.dimensions:
        similarity_data = {}
        for f in completed_files:
            if f.result:
                sents = f.result["sentence_objects"]
                sync = compute_speaker_synchronization(sents)
                similarity_data[f.file_id] = {
                    "is_dual_channel": f.result["audio_info"]["channels"] >= 2,
                    "synchronization": sync
                }

        dual_files = [
            fid for fid, d in similarity_data.items()
            if d["is_dual_channel"] and d["synchronization"] is not None
        ]

        sim_matrix = {}
        if len(dual_files) >= 2:
            for i in range(len(dual_files)):
                for j in range(i + 1, len(dual_files)):
                    fid1, fid2 = dual_files[i], dual_files[j]
                    s1 = similarity_data[fid1]["synchronization"]["synchronization_score"]
                    s2 = similarity_data[fid2]["synchronization"]["synchronization_score"]
                    sim = 1.0 - abs(s1 - s2)
                    sim_matrix[f"{fid1}__{fid2}"] = round(sim, 4)

        similarity_result = {
            "speaker_sync": similarity_data,
            "dual_channel_files": dual_files,
            "single_channel_files": [
                fid for fid, d in similarity_data.items()
                if not d["is_dual_channel"]
            ],
            "cross_file_similarity_matrix": sim_matrix
        }

        if config.baseline_file and config.baseline_file in similarity_data:
            baseline_sync = similarity_data[config.baseline_file]["synchronization"]
            deviations = {}
            if baseline_sync is not None:
                baseline_score = baseline_sync["synchronization_score"]
                for fid, d in similarity_data.items():
                    if fid != config.baseline_file and d["synchronization"] is not None:
                        dev = d["synchronization"]["synchronization_score"] - baseline_score
                        deviations[fid] = {
                            "sync_score_deviation": round(dev, 4),
                            "direction": "more_synced" if dev > 0 else "less_synced"
                        }
            similarity_result["baseline_deviations"] = deviations

        comparison["speaker_similarity"] = similarity_result

    return {
        "meta": meta,
        "file_summaries": file_summaries,
        "comparison": comparison
    }


def generate_csv_report(report: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    meta = report["meta"]
    writer.writerow(["=== 批次元信息 ==="])
    writer.writerow(["批次名", meta["batch_name"]])
    writer.writerow(["批次ID", meta["batch_id"]])
    writer.writerow(["总文件数", meta["total_files"]])
    writer.writerow(["成功数", meta["success_count"]])
    writer.writerow(["失败数", meta["failed_count"]])
    writer.writerow(["总耗时(秒)", meta["total_duration_seconds"]])
    writer.writerow(["基线文件", meta["baseline_file"] or "无"])
    writer.writerow([])

    writer.writerow(["=== 文件摘要 ==="])
    headers = ["文件ID", "文件名", "状态", "处理耗时(秒)", "时长(秒)", "主导情感",
               "效价标准差", "冲突密度", "转折点数量", "激化区间数", "传染事件数"]
    writer.writerow(headers)
    for fid, fsummary in report["file_summaries"].items():
        if fsummary.get("available", True):
            ds = fsummary.get("dialogue_summary", {})
            ai = fsummary.get("audio_info", {})
            writer.writerow([
                fid,
                fsummary["filename"],
                fsummary["status"],
                fsummary.get("processing_time_seconds", ""),
                ai.get("duration", ""),
                ds.get("dominant_emotion", ""),
                ds.get("valence_std", ""),
                ds.get("conflict_density", ""),
                ds.get("turning_point_count", ""),
                ds.get("escalation_interval_count", ""),
                ds.get("contagion_event_count", "")
            ])
        else:
            writer.writerow([
                fid, fsummary["filename"], fsummary["status"],
                "", "", "", "", "", "", "", ""
            ])
    writer.writerow([])

    comparison = report["comparison"]

    if "emotion_distribution" in comparison:
        writer.writerow(["=== 情感分布对比 ==="])
        dist = comparison["emotion_distribution"]["distributions"]
        headers = ["文件ID", "文件名"] + EMOTION_LABELS
        writer.writerow(headers)
        for fid, d in dist.items():
            fname = report["file_summaries"].get(fid, {}).get("filename", fid)
            row = [fid, fname] + [d[e] for e in EMOTION_LABELS]
            writer.writerow(row)
        writer.writerow([])

        if "baseline_deviations" in comparison["emotion_distribution"]:
            writer.writerow(["=== 情感分布基线偏差 ==="])
            headers = ["文件ID", "文件名", "JS散度", "偏差方向"] + [f"{e}_偏差" for e in EMOTION_LABELS]
            writer.writerow(headers)
            for fid, dev in comparison["emotion_distribution"]["baseline_deviations"].items():
                fname = report["file_summaries"].get(fid, {}).get("filename", fid)
                row = [fid, fname, dev["js_divergence"], dev["direction"]]
                row += [dev["per_emotion_deviation"].get(e, 0) for e in EMOTION_LABELS]
                writer.writerow(row)
            writer.writerow([])

    if "valence_trend" in comparison:
        writer.writerow(["=== 效价趋势对比 ==="])
        trends = comparison["valence_trend"]["trends"]
        headers = ["文件ID", "文件名", "趋势", "斜率", "平均效价", "是否异常"]
        writer.writerow(headers)
        anomalous = set(comparison["valence_trend"].get("anomalous_files", []))
        for fid, d in trends.items():
            fname = report["file_summaries"].get(fid, {}).get("filename", fid)
            writer.writerow([
                fid, fname, d["trend"], d["slope"], d["mean_valence"],
                "是" if fid in anomalous else "否"
            ])
        writer.writerow([])

    if "arousal_pattern" in comparison:
        writer.writerow(["=== 唤醒度模式对比 ==="])
        patterns = comparison["arousal_pattern"]["patterns"]
        headers = ["文件ID", "文件名", "波动模式", "标准差", "平均唤醒度",
                   "突变次数", "激化区间数", "有激化"]
        writer.writerow(headers)
        for fid, d in patterns.items():
            fname = report["file_summaries"].get(fid, {}).get("filename", fid)
            writer.writerow([
                fid, fname, d["pattern"], d["arousal_std"], d["mean_arousal"],
                d["sharp_change_count"], d["escalation_interval_count"],
                "是" if d["has_escalation"] else "否"
            ])
        writer.writerow([])

    if "speaker_similarity" in comparison:
        writer.writerow(["=== 说话人同步性对比 ==="])
        sync = comparison["speaker_similarity"]["speaker_sync"]
        headers = ["文件ID", "文件名", "是否双通道", "效价相关", "唤醒相关", "同步分数", "同步等级"]
        writer.writerow(headers)
        for fid, d in sync.items():
            fname = report["file_summaries"].get(fid, {}).get("filename", fid)
            s = d.get("synchronization")
            if s:
                writer.writerow([
                    fid, fname, "是" if d["is_dual_channel"] else "否",
                    s["valence_correlation"], s["arousal_correlation"],
                    s["synchronization_score"], s["alignment_level"]
                ])
            else:
                writer.writerow([
                    fid, fname, "是" if d["is_dual_channel"] else "否",
                    "", "", "", ""
                ])

    return output.getvalue()


async def _acquire_batch_slot() -> bool:
    global _batch_queue_count
    async with _batch_queue_lock:
        if _batch_semaphore._value <= 0:
            if _batch_queue_count >= 10:
                return False
            _batch_queue_count += 1
    try:
        await _batch_semaphore.acquire()
    except Exception:
        async with _batch_queue_lock:
            _batch_queue_count = max(0, _batch_queue_count - 1)
        return False
    async with _batch_queue_lock:
        _batch_queue_count = max(0, _batch_queue_count - 1)
    return True


def _release_batch_slot():
    _batch_semaphore.release()


def _process_single_batch_file(batch_file: BatchFile, timeout: float) -> None:
    try:
        with _file_status_lock:
            batch_file.status = FileStatus.PROCESSING
            batch_file.started_at = time.time()

        result = _process_single_file_sync(
            batch_file.temp_path, "svm",
            batch_file.filename.lower().endswith("_dual.wav"),
            timeout
        )

        with _file_status_lock:
            batch_file.result = result
            batch_file.status = FileStatus.COMPLETED
            batch_file.completed_at = time.time()

    except TimeoutError:
        with _file_status_lock:
            batch_file.status = FileStatus.FAILED
            batch_file.error = f"Processing timed out after {timeout}s"
            batch_file.completed_at = time.time()
    except Exception as e:
        logger.exception(f"Failed to process file {batch_file.filename}")
        with _file_status_lock:
            batch_file.status = FileStatus.FAILED
            batch_file.error = str(e)
            batch_file.completed_at = time.time()
    finally:
        try:
            os.unlink(batch_file.temp_path)
        except Exception:
            pass


async def process_batch_task(batch: BatchTask):
    config = get_config()
    timeout = config["request_timeout_seconds"]

    try:
        if not await _acquire_batch_slot():
            batch.status = BatchStatus.FAILED
            for f in batch.files:
                f.status = FileStatus.FAILED
                f.error = "Batch queue full"
            return

        batch.status = BatchStatus.PROCESSING
        batch.started_at = time.time()

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=MAX_FILE_CONCURRENT) as executor:
            futures = []
            for batch_file in batch.files:
                future = loop.run_in_executor(
                    executor, _process_single_batch_file, batch_file, timeout
                )
                futures.append(future)
            for future in asyncio.as_completed(futures):
                try:
                    await future
                except Exception as e:
                    logger.exception(f"Unexpected error in file processing task")

        all_failed = all(f.status == FileStatus.FAILED for f in batch.files)
        if all_failed and len(batch.files) > 0:
            batch.status = BatchStatus.FAILED
        else:
            batch.status = BatchStatus.COMPLETED
        batch.completed_at = time.time()

        try:
            batch.report = generate_report(batch)
        except Exception as e:
            logger.exception("Failed to generate report")
            batch.report = {
                "error": f"Report generation failed: {str(e)}",
                "meta": {
                    "batch_name": batch.config.batch_name,
                    "batch_id": batch.batch_id,
                    "total_files": len(batch.files),
                    "success_count": len([f for f in batch.files if f.status == FileStatus.COMPLETED]),
                    "failed_count": len([f for f in batch.files if f.status == FileStatus.FAILED]),
                    "total_duration_seconds": round(
                        (batch.completed_at or time.time()) - (batch.started_at or time.time()), 2
                    )
                }
            }

        persist_batch(batch.batch_id)

    finally:
        _release_batch_slot()


def create_batch_task(
    config: BatchConfig,
    files_data: List[Tuple[str, bytes, Optional[str]]]
) -> BatchTask:
    batch_id = str(uuid.uuid4())
    batch_files = []

    for item in files_data:
        if len(item) == 3:
            filename, content, file_id = item
        else:
            filename, content = item
            file_id = None

        if file_id is None:
            file_id = str(uuid.uuid4())
        suffix = os.path.splitext(filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        batch_files.append(BatchFile(
            file_id=file_id,
            filename=filename,
            temp_path=temp_path
        ))

    batch = BatchTask(
        batch_id=batch_id,
        config=config,
        files=batch_files
    )

    _batches_store[batch_id] = batch

    asyncio.create_task(process_batch_task(batch))

    return batch


def get_batch(batch_id: str) -> Optional[BatchTask]:
    return _batches_store.get(batch_id)


def get_batch_status(batch_id: str) -> Optional[dict]:
    batch = _batches_store.get(batch_id)
    if not batch:
        return None

    completed = sum(1 for f in batch.files if f.status in (FileStatus.COMPLETED, FileStatus.FAILED))
    total = len(batch.files)

    return {
        "batch_id": batch_id,
        "status": batch.status.value,
        "progress": {
            "completed": completed,
            "total": total,
            "percentage": round(completed / total * 100, 2) if total > 0 else 0
        },
        "files": [
            {
                "file_id": f.file_id,
                "filename": f.filename,
                "status": f.status.value,
                "error": f.error,
                "started_at": datetime.fromtimestamp(f.started_at).isoformat() if f.started_at else None,
                "completed_at": datetime.fromtimestamp(f.completed_at).isoformat() if f.completed_at else None
            }
            for f in batch.files
        ]
    }


def _get_persistence_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "batch_history.json")


def _batch_to_dict(batch: BatchTask) -> dict:
    files = []
    for f in batch.files:
        files.append({
            "file_id": f.file_id,
            "filename": f.filename,
            "status": f.status.value,
            "error": f.error,
            "started_at": f.started_at,
            "completed_at": f.completed_at,
            "result_summary": None
        })
    return {
        "batch_id": batch.batch_id,
        "config": {
            "batch_name": batch.config.batch_name,
            "dimensions": batch.config.dimensions,
            "baseline_file": batch.config.baseline_file
        },
        "files": files,
        "status": batch.status.value,
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
        "report": batch.report
    }


def _dict_to_batch(data: dict) -> BatchTask:
    config = BatchConfig(
        batch_name=data["config"]["batch_name"],
        dimensions=data["config"]["dimensions"],
        baseline_file=data["config"].get("baseline_file")
    )
    files = []
    for f_data in data["files"]:
        files.append(BatchFile(
            file_id=f_data["file_id"],
            filename=f_data["filename"],
            temp_path="",
            status=FileStatus(f_data["status"]),
            error=f_data.get("error"),
            started_at=f_data.get("started_at"),
            completed_at=f_data.get("completed_at"),
            result=None
        ))
    batch = BatchTask(
        batch_id=data["batch_id"],
        config=config,
        files=files,
        status=BatchStatus(data["status"]),
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        report=data.get("report")
    )
    return batch


def _save_batches_to_disk() -> None:
    try:
        path = _get_persistence_path()
        data = []
        for batch_id, batch in _batches_store.items():
            if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
                data.append(_batch_to_dict(batch))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Failed to save batch history to disk: {e}")


def _load_batches_from_disk() -> None:
    try:
        path = _get_persistence_path()
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for batch_data in data:
            batch = _dict_to_batch(batch_data)
            if batch.batch_id not in _batches_store:
                _batches_store[batch.batch_id] = batch
        logger.info(f"Loaded {len(data)} batches from disk")
    except Exception as e:
        logger.exception(f"Failed to load batch history from disk: {e}")


def list_batches(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 10
) -> dict:
    all_batches = list(_batches_store.values())

    if status:
        try:
            target_status = BatchStatus(status)
            all_batches = [b for b in all_batches if b.status == target_status]
        except ValueError:
            pass

    all_batches.sort(key=lambda b: b.started_at or 0, reverse=True)

    total = len(all_batches)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = all_batches[start:end]

    items = []
    for b in paginated:
        items.append({
            "batch_id": b.batch_id,
            "batch_name": b.config.batch_name,
            "file_count": len(b.files),
            "status": b.status.value,
            "started_at": datetime.fromtimestamp(b.started_at).isoformat() if b.started_at else None,
            "completed_at": datetime.fromtimestamp(b.completed_at).isoformat() if b.completed_at else None,
            "success_count": sum(1 for f in b.files if f.status == FileStatus.COMPLETED),
            "failed_count": sum(1 for f in b.files if f.status == FileStatus.FAILED)
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 0
    }


def persist_batch(batch_id: str) -> None:
    _save_batches_to_disk()


def init_batch_history() -> None:
    _load_batches_from_disk()


SIGNIFICANT_DIFF_THRESHOLD = 0.15


def _get_batch_aggregated_metrics(batch: BatchTask) -> dict:
    if not batch.report or "comparison" not in batch.report:
        return {}

    comparison = batch.report["comparison"]
    metrics = {}

    if "emotion_distribution" in comparison:
        dists = comparison["emotion_distribution"]["distributions"]
        if dists:
            all_emotions = {}
            for emotion in EMOTION_LABELS:
                values = [d[emotion] for d in dists.values() if emotion in d]
                if values:
                    all_emotions[emotion] = float(np.mean(values))
            metrics["emotion_distribution"] = all_emotions

    if "valence_trend" in comparison:
        trends = comparison["valence_trend"]["trends"]
        if trends:
            slopes = [t["slope"] for t in trends.values()]
            mean_valences = [t["mean_valence"] for t in trends.values()]
            metrics["valence_trend"] = {
                "mean_slope": float(np.mean(slopes)) if slopes else 0.0,
                "std_slope": float(np.std(slopes)) if len(slopes) > 1 else 0.0,
                "mean_valence": float(np.mean(mean_valences)) if mean_valences else 0.0
            }

    if "arousal_pattern" in comparison:
        patterns = comparison["arousal_pattern"]["patterns"]
        if patterns:
            stds = [p["arousal_std"] for p in patterns.values()]
            means = [p["mean_arousal"] for p in patterns.values()]
            escalation_counts = [p["escalation_interval_count"] for p in patterns.values()]
            sharp_counts = [p["sharp_change_count"] for p in patterns.values()]
            metrics["arousal_pattern"] = {
                "mean_arousal_std": float(np.mean(stds)) if stds else 0.0,
                "mean_arousal": float(np.mean(means)) if means else 0.0,
                "avg_escalation_count": float(np.mean(escalation_counts)) if escalation_counts else 0.0,
                "avg_sharp_change_count": float(np.mean(sharp_counts)) if sharp_counts else 0.0
            }

    if "speaker_similarity" in comparison:
        sync_data = comparison["speaker_similarity"]["speaker_sync"]
        if sync_data:
            sync_scores = []
            for d in sync_data.values():
                if d.get("synchronization"):
                    sync_scores.append(d["synchronization"]["synchronization_score"])
            if sync_scores:
                metrics["speaker_similarity"] = {
                    "mean_sync_score": float(np.mean(sync_scores)),
                    "std_sync_score": float(np.std(sync_scores)) if len(sync_scores) > 1 else 0.0
                }

    return metrics


def _flatten_metrics(metrics: dict) -> Dict[str, float]:
    flat = {}
    for dim, dim_metrics in metrics.items():
        if isinstance(dim_metrics, dict):
            for key, value in dim_metrics.items():
                if isinstance(value, (int, float)):
                    flat[f"{dim}.{key}"] = float(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            flat[f"{dim}.{key}.{sub_key}"] = float(sub_value)
    return flat


def compare_batches(batch_id_a: str, batch_id_b: str) -> Optional[dict]:
    batch_a = _batches_store.get(batch_id_a)
    batch_b = _batches_store.get(batch_id_b)

    if not batch_a or not batch_b:
        return None

    if batch_a.status != BatchStatus.COMPLETED or batch_b.status != BatchStatus.COMPLETED:
        return None

    metrics_a = _get_batch_aggregated_metrics(batch_a)
    metrics_b = _get_batch_aggregated_metrics(batch_b)

    flat_a = _flatten_metrics(metrics_a)
    flat_b = _flatten_metrics(metrics_b)

    all_keys = sorted(set(list(flat_a.keys()) + list(flat_b.keys())))

    comparisons = []
    significant_diffs = []

    for key in all_keys:
        val_a = flat_a.get(key, 0.0)
        val_b = flat_b.get(key, 0.0)
        diff = val_b - val_a
        abs_diff = abs(diff)
        is_significant = abs_diff > SIGNIFICANT_DIFF_THRESHOLD

        comparison = {
            "metric": key,
            "batch_a_value": round(val_a, 6),
            "batch_b_value": round(val_b, 6),
            "difference": round(diff, 6),
            "abs_difference": round(abs_diff, 6),
            "is_significant": is_significant
        }
        comparisons.append(comparison)

        if is_significant:
            significant_diffs.append(key)

    result = {
        "batch_a": {
            "batch_id": batch_a.batch_id,
            "batch_name": batch_a.config.batch_name,
            "file_count": len(batch_a.files),
            "completed_at": datetime.fromtimestamp(batch_a.completed_at).isoformat() if batch_a.completed_at else None,
            "metrics": metrics_a
        },
        "batch_b": {
            "batch_id": batch_b.batch_id,
            "batch_name": batch_b.config.batch_name,
            "file_count": len(batch_b.files),
            "completed_at": datetime.fromtimestamp(batch_b.completed_at).isoformat() if batch_b.completed_at else None,
            "metrics": metrics_b
        },
        "comparisons": comparisons,
        "significant_diffs": significant_diffs,
        "significant_diff_count": len(significant_diffs),
        "threshold": SIGNIFICANT_DIFF_THRESHOLD
    }

    return result
