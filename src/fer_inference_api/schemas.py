from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    image: str = Field(min_length=1, max_length=5_000_000)


class EmotionPrediction(BaseModel):
    model_config = {"populate_by_name": True}
    class_: int = Field(alias="class")
    confidence: float


class InferenceTimings(BaseModel):
    total_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    face_detect_ms: Optional[float] = None
    face_crop_ms: Optional[float] = None
    emotion_ms: Optional[float] = None


class InferResponse(BaseModel):
    face_detected: bool
    emotions: Optional[Dict[str, EmotionPrediction]] = None
    num_classes: int
    inference_ms: Optional[int] = None
    bbox: Optional[List[float]] = None
    timings: Optional[InferenceTimings] = None


class MetricsSummary(BaseModel):
    requests: int
    avg_total_ms: Optional[float] = None
    p50_total_ms: Optional[float] = None
    p95_total_ms: Optional[float] = None
    p99_total_ms: Optional[float] = None
    avg_decode_ms: Optional[float] = None
    avg_face_detect_ms: Optional[float] = None
    avg_face_crop_ms: Optional[float] = None
    avg_emotion_ms: Optional[float] = None


class DeviceInfo(BaseModel):
    device: str
    worker_pid: int
