from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    image: str = Field(min_length=1, max_length=5_000_000)


class EmotionPrediction(BaseModel):
    model_config = {"populate_by_name": True}
    class_: int = Field(alias="class")
    confidence: float


class InferResponse(BaseModel):
    face_detected: bool
    emotions: Optional[Dict[str, EmotionPrediction]] = None
    num_classes: int
    inference_ms: Optional[int] = None
    bbox: Optional[List[float]] = None


class DeviceInfo(BaseModel):
    device: str
    worker_pid: int
