from typing import Dict, List, Optional

from pydantic import BaseModel


class InferRequest(BaseModel):
    image: str


class InferResponse(BaseModel):
    face_detected: bool
    emotions: Optional[Dict[str, Dict[str, object]]] = None
    num_classes: int
    inference_ms: Optional[int] = None
    bbox: Optional[List[float]] = None


class DeviceInfo(BaseModel):
    device: str
    worker_pid: int
