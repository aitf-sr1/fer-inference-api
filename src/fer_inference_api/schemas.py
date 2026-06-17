from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    image: str = Field(min_length=1, max_length=5_000_000)


class InferResponse(BaseModel):
    face_detected: bool
    emotions: Optional[Dict[str, Dict[str, object]]] = None
    num_classes: int
    inference_ms: Optional[int] = None
    bbox: Optional[List[float]] = None


class DeviceInfo(BaseModel):
    device: str
    worker_pid: int
