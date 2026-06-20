import uuid
import time
import asyncio
import logging
import os
import tempfile
import json
import csv
import io
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import get_config, get_metrics_config
from app.audio_processor import process_audio
from app.feature_extractor import extract_features
from app.emotion_classifier import get_classifier, EmotionResult, EMOTION_LABELS
from app.dialogue_tracker import (
    SentenceEmotionResult, compute_dialogue_summary,
    TurningPoint, EscalationInterval, ContagionEvent, DialogueSummary
)
from app.batch_processor import (
    create_batch_task, get_batch, get_batch_status,
    BatchConfig, validate_file_extension, generate_csv_report,
    FileStatus, BatchStatus, list_batches, init_batch_history,
    compare_batches
)
from app.stream_processor import (
    register_session, unregister_session, get_session, list_active_sessions,
    get_session_results, get_all_session_results, update_session_activity,
    set_session_config, increment_frame_count, append_frame_result,
    analyze_frame_with_timeout, compute_summary, persist_summary,
    _pcm_bytes_to_float32, increment_timeout_count,
    VALID_MODEL_MODES, CONFIG_TIMEOUT_SEC
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Emotion Track API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_results_store: Dict[str, dict] = {}
_result_metadata: Dict[str, dict] = {}

config = get_config()
MAX_FILE_SIZE = config["max_file_size_mb"] * 1024 * 1024
REQUEST_TIMEOUT = config["request_timeout_seconds"]
MAX_CONCURRENT = config["max_concurrent_requests"]
MAX_QUEUE = config["max_queue_size"]

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_count = 0
_queue_lock = asyncio.Lock()


class TurningPointResponse(BaseModel):
    time: float
    valence_before: float
    valence_after: float
    delta: float


class EscalationIntervalResponse(BaseModel):
    start_time: float
    end_time: float
    avg_arousal: float


class ContagionEventResponse(BaseModel):
    source_speaker: str
    target_speaker: str
    source_time: float
    target_time: float
    delay_sentences: int
    contagion_strength: float


class SentenceResultResponse(BaseModel):
    start_time: float
    end_time: float
    speaker_id: str
    emotion: str
    confidence: float
    valence: float
    arousal: float


class DialogueSummaryResponse(BaseModel):
    dominant_emotion: str
    valence_std: float
    conflict_density: float
    turning_points: List[TurningPointResponse]
    escalation_intervals: List[EscalationIntervalResponse]
    contagion_events: List[ContagionEventResponse]


class AudioInfoResponse(BaseModel):
    duration: float
    sample_rate: int
    channels: int


class MetricsResponse(BaseModel):
    uar: float
    f1_scores: Dict[str, float]
    confusion_matrix: List[List[int]]


class AnalyzeResponse(BaseModel):
    task_id: str
    audio_info: AudioInfoResponse
    sentences: List[SentenceResultResponse]
    dialogue_summary: DialogueSummaryResponse
    metrics: MetricsResponse
    model_mode: str
    completed: bool
    incomplete_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]


class QueryResponse(BaseModel):
    task_id: str
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None


async def _acquire_slot():
    global _queue_count
    async with _queue_lock:
        if _semaphore._value <= 0:
            if _queue_count >= MAX_QUEUE:
                raise HTTPException(status_code=503, detail="Server too busy. Queue is full.")
            _queue_count += 1
    acquired = False
    try:
        acquired = await asyncio.wait_for(_semaphore.acquire(), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        async with _queue_lock:
            _queue_count = max(0, _queue_count - 1)
        raise HTTPException(status_code=503, detail="Request timed out waiting in queue.")
    if not acquired:
        async with _queue_lock:
            _queue_count = max(0, _queue_count - 1)
        raise HTTPException(status_code=503, detail="Failed to acquire processing slot.")
    async with _queue_lock:
        _queue_count = max(0, _queue_count - 1)


def _release_slot():
    _semaphore.release()


def _process_audio_sync(file_path: str, model_mode: str,
                        is_dual_channel: bool, output_format: str) -> dict:
    audio_info = process_audio(file_path, is_dual_channel=is_dual_channel,
                               model_mode=model_mode)
    classifier = get_classifier(model_mode)
    sentence_results = []
    is_dual = is_dual_channel and audio_info.channels >= 2
    timeout = time.time() + REQUEST_TIMEOUT
    completed = True
    incomplete_reason = None

    for seg in audio_info.segments:
        if time.time() > timeout:
            completed = False
            incomplete_reason = f"Timeout after {REQUEST_TIMEOUT}s. Partial results returned."
            break
        features = extract_features(seg.audio_data, audio_info.sample_rate,
                                    seg.start_time, seg.end_time, seg.speaker_id)
        emotion_result = classifier.predict(seg.audio_data, audio_info.sample_rate, features)
        sentence_results.append(SentenceEmotionResult(
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_id=seg.speaker_id,
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
            valence=emotion_result.valence,
            arousal=emotion_result.arousal
        ))

    dialogue_summary = compute_dialogue_summary(sentence_results, is_dual_speaker=is_dual)
    metrics_config = get_metrics_config()
    model_metrics = metrics_config.get("metrics", {}).get(model_mode, {})

    return {
        "audio_info": audio_info,
        "sentences": sentence_results,
        "dialogue_summary": dialogue_summary,
        "metrics": model_metrics,
        "completed": completed,
        "incomplete_reason": incomplete_reason
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    model_mode: str = Query("svm", pattern="^(svm|rf|wav2vec2)$"),
    is_dual_channel: bool = Query(False),
    output_format: str = Query("full", pattern="^(full|compact)$")
):
    await _acquire_slot()
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413,
                                detail=f"File too large. Max {config['max_file_size_mb']}MB.")

        suffix = os.path.splitext(file.filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            task_id = str(uuid.uuid4())
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _process_audio_sync,
                                     tmp_path, model_mode, is_dual_channel, output_format),
                timeout=REQUEST_TIMEOUT + 10
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Processing timeout.")
        finally:
            os.unlink(tmp_path)

        audio_info = result["audio_info"]
        sentences = result["sentences"]
        summary = result["dialogue_summary"]
        metrics = result["metrics"]

        sentence_resp = []
        for s in sentences:
            item = SentenceResultResponse(
                start_time=s.start_time, end_time=s.end_time,
                speaker_id=s.speaker_id, emotion=s.emotion,
                confidence=s.confidence, valence=s.valence, arousal=s.arousal
            )
            if output_format == "compact":
                pass
            sentence_resp.append(item)

        tp_resp = [TurningPointResponse(time=t.time, valence_before=t.valence_before,
                    valence_after=t.valence_after, delta=t.delta)
                   for t in summary.turning_points]
        esc_resp = [EscalationIntervalResponse(start_time=e.start_time, end_time=e.end_time,
                    avg_arousal=e.avg_arousal) for e in summary.escalation_intervals]
        cont_resp = [ContagionEventResponse(source_speaker=c.source_speaker,
                     target_speaker=c.target_speaker, source_time=c.source_time,
                     target_time=c.target_time, delay_sentences=c.delay_sentences,
                     contagion_strength=c.contagion_strength)
                     for c in summary.contagion_events]

        summary_resp = DialogueSummaryResponse(
            dominant_emotion=summary.dominant_emotion,
            valence_std=summary.valence_std,
            conflict_density=summary.conflict_density,
            turning_points=tp_resp,
            escalation_intervals=esc_resp,
            contagion_events=cont_resp
        )

        metrics_resp = MetricsResponse(
            uar=metrics.get("uar", 0.0),
            f1_scores=metrics.get("f1_scores", {}),
            confusion_matrix=metrics.get("confusion_matrix", [])
        )

        response = AnalyzeResponse(
            task_id=task_id,
            audio_info=AudioInfoResponse(
                duration=audio_info.duration,
                sample_rate=audio_info.sample_rate,
                channels=audio_info.channels
            ),
            sentences=sentence_resp,
            dialogue_summary=summary_resp,
            metrics=metrics_resp,
            model_mode=model_mode,
            completed=result["completed"],
            incomplete_reason=result["incomplete_reason"]
        )

        result_dict = response.dict()
        _results_store[task_id] = result_dict
        _result_metadata[task_id] = {
            "filename": file.filename or "audio.wav",
            "analyzed_at": datetime.now().isoformat()
        }
        return response

    finally:
        _release_slot()


@app.get("/api/results/{task_id}", response_model=QueryResponse)
async def get_result(task_id: str):
    result = _results_store.get(task_id)
    if result is None:
        return QueryResponse(task_id=task_id, error="Result not found")
    return QueryResponse(task_id=task_id, result=AnalyzeResponse(**result))


def _build_annotation_data(task_id: str, result: dict, metadata: dict) -> dict:
    audio_info = result.get("audio_info", {})
    sentences = result.get("sentences", [])
    summary = result.get("dialogue_summary", {})

    turning_points = summary.get("turning_points", [])
    escalation_intervals = summary.get("escalation_intervals", [])

    tp_times = {tp["time"] for tp in turning_points}

    sentence_annotations = []
    for idx, sent in enumerate(sentences):
        is_turning_point = sent["start_time"] in tp_times or any(
            abs(sent["start_time"] - tp["time"]) < 0.01 for tp in turning_points
        )

        is_escalation = any(
            esc["start_time"] <= sent["start_time"] <= esc["end_time"]
            for esc in escalation_intervals
        )

        sentence_annotations.append({
            "sentence_id": idx + 1,
            "start_time": sent["start_time"],
            "end_time": sent["end_time"],
            "speaker_id": sent["speaker_id"],
            "emotion": sent["emotion"],
            "confidence": sent["confidence"],
            "valence": sent["valence"],
            "arousal": sent["arousal"],
            "is_turning_point": is_turning_point,
            "is_escalation": is_escalation
        })

    dialogue_summary = {
        "dominant_emotion": summary.get("dominant_emotion", "neutral"),
        "valence_std": summary.get("valence_std", 0.0),
        "conflict_density": summary.get("conflict_density", 0.0),
        "turning_point_count": len(turning_points),
        "escalation_interval_count": len(escalation_intervals),
        "contagion_event_count": len(summary.get("contagion_events", []))
    }

    annotation = {
        "audio_metadata": {
            "filename": metadata.get("filename", "unknown.wav"),
            "duration": audio_info.get("duration", 0.0),
            "sample_rate": audio_info.get("sample_rate", 0),
            "channels": audio_info.get("channels", 0),
            "analyzed_at": metadata.get("analyzed_at", "")
        },
        "sentences": sentence_annotations,
        "dialogue_summary": dialogue_summary
    }
    return annotation


def _generate_json_annotation(annotation: dict) -> str:
    return json.dumps(annotation, ensure_ascii=False, indent=2)


def _generate_csv_annotation(annotation: dict) -> str:
    sentences = annotation.get("sentences", [])
    if not sentences:
        return ""

    output = io.StringIO()
    fieldnames = [
        "sentence_id", "start_time", "end_time", "speaker_id",
        "emotion", "confidence", "valence", "arousal",
        "is_turning_point", "is_escalation"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for sent in sentences:
        writer.writerow(sent)

    return output.getvalue()


@app.get("/api/export/{task_id}")
async def export_annotation(
    task_id: str,
    format: str = Query("json", pattern="^(json|csv)$")
):
    result = _results_store.get(task_id)
    metadata = _result_metadata.get(task_id, {})
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    annotation = _build_annotation_data(task_id, result, metadata)

    if format == "json":
        content = _generate_json_annotation(annotation)
        media_type = "application/json"
        filename = f"annotation_{task_id}.json"
    else:
        content = _generate_csv_annotation(annotation)
        media_type = "text/csv"
        filename = f"annotation_{task_id}.csv"

    response = Response(
        content=content,
        media_type=media_type
    )
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    models_loaded = {}
    try:
        from app.emotion_classifier import SVMClassifier, RFClassifier, Wav2Vec2Classifier
        models_loaded["svm"] = SVMClassifier().is_loaded()
        models_loaded["rf"] = RFClassifier().is_loaded()
        try:
            models_loaded["wav2vec2"] = Wav2Vec2Classifier().is_loaded()
        except Exception:
            models_loaded["wav2vec2"] = False
    except Exception:
        models_loaded = {"svm": False, "rf": False, "wav2vec2": False}

    return HealthResponse(
        status="healthy",
        version=config["app_version"],
        models_loaded=models_loaded
    )


class BatchSubmitResponse(BaseModel):
    batch_id: str
    message: str


class BatchFileStatusResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class BatchProgressResponse(BaseModel):
    completed: int
    total: int
    percentage: float


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    progress: BatchProgressResponse
    files: List[BatchFileStatusResponse]


class BatchSubmitConfig(BaseModel):
    batch_name: str
    dimensions: List[str]
    baseline_file: Optional[str] = None


VALID_DIMENSIONS = {"emotion_distribution", "valence_trend", "arousal_pattern", "speaker_similarity"}
MIN_FILES = 2
MAX_FILES = 20


@app.post("/api/batch/submit", response_model=BatchSubmitResponse)
async def submit_batch(
    files: List[UploadFile] = File(...),
    batch_config: str = Query(...)
):
    try:
        config_data = json.loads(batch_config)
        batch_config_obj = BatchSubmitConfig(**config_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in batch_config")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid batch_config: {str(e)}")

    if len(files) < MIN_FILES or len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Number of files must be between {MIN_FILES} and {MAX_FILES}"
        )

    invalid_files = []
    for f in files:
        if not validate_file_extension(f.filename or ""):
            invalid_files.append(f.filename)

    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file formats: {', '.join(invalid_files)}. "
                   f"Supported formats: {', '.join(['.wav', '.mp3', '.flac', '.ogg', '.m4a'])}"
        )

    if not batch_config_obj.dimensions:
        raise HTTPException(
            status_code=400,
            detail=f"At least one dimension must be selected. Valid dimensions: {', '.join(sorted(VALID_DIMENSIONS))}"
        )

    invalid_dims = [d for d in batch_config_obj.dimensions if d not in VALID_DIMENSIONS]
    if invalid_dims:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimensions: {', '.join(invalid_dims)}. "
                   f"Valid dimensions: {', '.join(VALID_DIMENSIONS)}"
        )

    file_ids = {}
    for f in files:
        fid = str(uuid.uuid4())
        file_ids[f.filename or fid] = fid

    baseline_file_id = None
    if batch_config_obj.baseline_file:
        if batch_config_obj.baseline_file not in file_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Baseline file '{batch_config_obj.baseline_file}' not found in uploaded files"
            )
        baseline_file_id = file_ids[batch_config_obj.baseline_file]

    files_data = []
    filename_to_temp_id = {}
    for f in files:
        content = await f.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File {f.filename} too large. Max {config['max_file_size_mb']}MB."
            )
        temp_id = str(uuid.uuid4())
        filename = f.filename or "audio.wav"
        files_data.append((filename, content, temp_id))
        filename_to_temp_id[filename] = temp_id

    resolved_baseline_id = None
    if batch_config_obj.baseline_file:
        resolved_baseline_id = filename_to_temp_id.get(batch_config_obj.baseline_file)

    batch_config_internal = BatchConfig(
        batch_name=batch_config_obj.batch_name,
        dimensions=batch_config_obj.dimensions,
        baseline_file=resolved_baseline_id
    )

    batch = create_batch_task(batch_config_internal, files_data)

    return BatchSubmitResponse(
        batch_id=batch.batch_id,
        message="Batch submitted successfully"
    )


@app.get("/api/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status_api(batch_id: str):
    status = get_batch_status(batch_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    files_resp = [
        BatchFileStatusResponse(**f) for f in status["files"]
    ]

    return BatchStatusResponse(
        batch_id=status["batch_id"],
        status=status["status"],
        progress=BatchProgressResponse(**status["progress"]),
        files=files_resp
    )


@app.get("/api/batch/{batch_id}/report")
async def get_batch_report(
    batch_id: str,
    format: str = Query("json", pattern="^(json|csv)$")
):
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    if batch.status not in (BatchStatus.COMPLETED, BatchStatus.FAILED):
        return Response(
            content=json.dumps({
                "status": "processing",
                "message": "Batch is still processing. Check status endpoint for progress."
            }, ensure_ascii=False),
            status_code=202,
            media_type="application/json"
        )

    if batch.report is None:
        raise HTTPException(status_code=500, detail="Report generation failed")

    if format == "csv":
        csv_content = generate_csv_report(batch.report)
        response = Response(
            content=csv_content,
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f'attachment; filename="batch_report_{batch_id}.csv"'
        return response
    else:
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, (SentenceEmotionResult, DialogueSummary,
                                  TurningPoint, EscalationInterval, ContagionEvent)):
                return clean_for_json(obj.__dict__)
            else:
                return obj

        clean_report = clean_for_json(batch.report)
        return Response(
            content=json.dumps(clean_report, ensure_ascii=False, indent=2),
            media_type="application/json"
        )


class BatchListItemResponse(BaseModel):
    batch_id: str
    batch_name: str
    file_count: int
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    success_count: int
    failed_count: int


class BatchListResponse(BaseModel):
    items: List[BatchListItemResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


@app.get("/api/batch/list", response_model=BatchListResponse)
async def get_batch_list(
    status: Optional[str] = Query(None, pattern="^(completed|failed|processing|queued)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    result = list_batches(status=status, page=page, page_size=page_size)
    items = [BatchListItemResponse(**item) for item in result["items"]]
    return BatchListResponse(
        items=items,
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        total_pages=result["total_pages"]
    )


@app.on_event("startup")
async def startup_event():
    init_batch_history()


class BatchCompareRequest(BaseModel):
    batch_id_a: str
    batch_id_b: str


class BatchCompareMetric(BaseModel):
    metric: str
    batch_a_value: float
    batch_b_value: float
    difference: float
    abs_difference: float
    is_significant: bool


class BatchCompareBatchInfo(BaseModel):
    batch_id: str
    batch_name: str
    file_count: int
    completed_at: Optional[str] = None
    metrics: dict


class BatchCompareResponse(BaseModel):
    batch_a: BatchCompareBatchInfo
    batch_b: BatchCompareBatchInfo
    comparisons: List[BatchCompareMetric]
    significant_diffs: List[str]
    significant_diff_count: int
    threshold: float


@app.post("/api/batch/compare", response_model=BatchCompareResponse)
async def compare_batches_api(request: BatchCompareRequest):
    if request.batch_id_a == request.batch_id_b:
        raise HTTPException(status_code=400, detail="Cannot compare a batch with itself")

    result = compare_batches(request.batch_id_a, request.batch_id_b)
    if result is None:
        raise HTTPException(status_code=404, detail="One or both batches not found or not completed")

    return BatchCompareResponse(
        batch_a=BatchCompareBatchInfo(**result["batch_a"]),
        batch_b=BatchCompareBatchInfo(**result["batch_b"]),
        comparisons=[BatchCompareMetric(**c) for c in result["comparisons"]],
        significant_diffs=result["significant_diffs"],
        significant_diff_count=result["significant_diff_count"],
        threshold=result["threshold"]
    )


class StreamSessionInfo(BaseModel):
    session_id: str
    status: str
    frame_count: int
    analyzed_frames: int
    timeout_frames: int
    connection_duration_seconds: float
    model_mode: str
    last_emotion: Optional[str] = None
    last_active_at: str


class StreamSessionListResponse(BaseModel):
    sessions: List[StreamSessionInfo]
    total: int


@app.get("/api/stream/sessions", response_model=StreamSessionListResponse)
async def get_stream_sessions():
    sessions = await list_active_sessions()
    session_infos = [StreamSessionInfo(**s) for s in sessions]
    return StreamSessionListResponse(sessions=session_infos, total=len(session_infos))


@app.get("/api/stream/sessions/{session_id}/results")
async def get_stream_session_results(session_id: str):
    results = await get_session_results(session_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
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
        ]
    }


async def _send_json(websocket: WebSocket, data: dict):
    try:
        await websocket.send_json(data)
    except Exception as e:
        logger.warning(f"Failed to send message: {e}")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket,
                           session_id: Optional[str] = Query(None),
                           model_mode: Optional[str] = Query(None)):
    await websocket.accept()

    if not session_id:
        await _send_json(websocket, {"type": "error", "message": "Missing required parameter: session_id"})
        await websocket.close()
        return

    if not model_mode or model_mode not in VALID_MODEL_MODES:
        await _send_json(websocket, {
            "type": "error",
            "message": f"Invalid model_mode. Must be one of: {', '.join(sorted(VALID_MODEL_MODES))}"
        })
        await websocket.close()
        return

    err = await register_session(session_id, model_mode)
    if err:
        await _send_json(websocket, {"type": "error", "message": err})
        await websocket.close()
        return

    classifier = get_classifier(model_mode)
    seq_counter = 0
    config_received = False
    try:
        try:
            config_msg = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=CONFIG_TIMEOUT_SEC
            )
            config_data = json.loads(config_msg)
            if config_data.get("type") != "config":
                await _send_json(websocket, {
                    "type": "error",
                    "message": "Expected first message to be config. Format: {\"type\":\"config\",\"sample_rate\":16000,\"chunk_duration_ms\":500}"
                })
                await websocket.close()
                return

            sample_rate = config_data.get("sample_rate")
            chunk_duration_ms = config_data.get("chunk_duration_ms")
            if not sample_rate or not chunk_duration_ms:
                await _send_json(websocket, {
                    "type": "error",
                    "message": "Config must contain sample_rate and chunk_duration_ms"
                })
                await websocket.close()
                return

            alert_threshold = config_data.get("alert_threshold")
            await set_session_config(session_id, sample_rate, chunk_duration_ms,
                                     alert_threshold=alert_threshold)
            config_received = True
            await _send_json(websocket, {
                "type": "config_ack",
                "session_id": session_id,
                "sample_rate": sample_rate,
                "chunk_duration_ms": chunk_duration_ms
            })
        except asyncio.TimeoutError:
            await _send_json(websocket, {
                "type": "error",
                "message": f"Config message not received within {CONFIG_TIMEOUT_SEC}s timeout"
            })
            await websocket.close()
            return
        except json.JSONDecodeError:
            await _send_json(websocket, {"type": "error", "message": "Invalid JSON in config message"})
            await websocket.close()
            return

        while True:
            try:
                data = await websocket.receive()
                if data["type"] == "websocket.disconnect":
                    break

                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "close":
                            break
                    except json.JSONDecodeError:
                        pass
                    continue

                if "bytes" not in data:
                    continue

                pcm_bytes = data["bytes"]
                sess = await get_session(session_id)
                if sess is None:
                    break

                if len(pcm_bytes) != sess.expected_bytes:
                    await _send_json(websocket, {
                        "type": "warning",
                        "message": f"Frame size mismatch. Expected {sess.expected_bytes} bytes, got {len(pcm_bytes)} bytes. Frame discarded."
                    })
                    continue

                seq_counter += 1
                await increment_frame_count(session_id)
                timestamp_ms = (seq_counter - 1) * sess.chunk_duration_ms

                audio_f32 = _pcm_bytes_to_float32(pcm_bytes)
                result = await analyze_frame_with_timeout(
                    audio_f32, sess.sample_rate, classifier, seq_counter, timestamp_ms
                )

                if result is None:
                    await increment_timeout_count(session_id)
                    await _send_json(websocket, {"type": "timeout", "seq": seq_counter})
                else:
                    alert = await append_frame_result(session_id, result)
                    await _send_json(websocket, {
                        "type": "result",
                        "seq": result.seq,
                        "emotion": result.emotion,
                        "confidence": result.confidence,
                        "valence": result.valence,
                        "arousal": result.arousal,
                        "timestamp_ms": result.timestamp_ms
                    })
                    if alert is not None:
                        await _send_json(websocket, {
                            "type": "alert",
                            "alert_type": "emotion_shift",
                            "from_emotion": alert["from_emotion"],
                            "to_emotion": alert["to_emotion"],
                            "trigger_seq": alert["trigger_seq"],
                            "sustained_frames": alert["sustained_frames"]
                        })

                await update_session_activity(session_id)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error in session {session_id}: {e}")
                break

    finally:
        sess = await get_session(session_id)
        if sess is not None:
            all_results = await get_all_session_results(session_id) or []
            summary = compute_summary(sess)
            try:
                await _send_json(websocket, {"type": "summary", **summary})
            except Exception:
                pass
            persist_summary(session_id, summary, all_results)
        await unregister_session(session_id)
        try:
            await websocket.close()
        except Exception:
            pass
