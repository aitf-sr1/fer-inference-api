from typing import Dict, List, Optional

from pydantic import BaseModel


class InferRequest(BaseModel):
    image: str
    model: str


class LoadModelRequest(BaseModel):
    model: str


class EmotionPrediction(BaseModel):
    class_: int
    confidence: float

    model_config = {"fields": {"class_": "class"}}


class InferResponse(BaseModel):
    face_detected: bool
    emotions: Optional[Dict[str, Dict[str, object]]] = None
    num_classes: int
    inference_ms: Optional[int] = None
    bbox: Optional[List[float]] = None


class ModelsResponse(BaseModel):
    models: List[str]
    current: Optional[str] = None


class LoadModelResponse(BaseModel):
    loaded: str


class DeviceInfo(BaseModel):
    device: str
    worker_pid: int
