import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .config import settings
from .face_detector import FaceDetector, detect_and_crop
from .model_loader import run_inference
from .model_registry import ModelRegistry


class InferencePipeline:
    def __init__(self) -> None:
        blazeface_path = settings.resolved_blazeface_path
        if not blazeface_path.exists():
            raise FileNotFoundError(
                f"BlazeFace model not found at {blazeface_path}"
            )
        self._detector = FaceDetector(str(blazeface_path))
        self._registry = ModelRegistry()

    def list_models(self) -> List[str]:
        return self._registry.list_models()

    def load_model(self, model_name: str) -> None:
        self._registry.get(model_name)

    @property
    def current_model(self) -> Optional[str]:
        return self._registry.current_model

    @property
    def device(self) -> str:
        return self._registry.device

    def infer(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        session, num_classes = self._registry.get(
            self._registry.current_model or ""
        )
        face, bbox = detect_and_crop(img_rgb, self._detector)
        if face is None:
            return {
                "face_detected": False,
                "emotions": None,
                "num_classes": num_classes,
                "inference_ms": None,
                "bbox": None,
            }
        t0 = time.perf_counter()
        emotions = run_inference(session, face)
        inference_ms = round((time.perf_counter() - t0) * 1000)
        return {
            "face_detected": True,
            "emotions": emotions,
            "num_classes": num_classes,
            "inference_ms": inference_ms,
            "bbox": list(bbox),
        }

    def infer_base64(self, b64_image: str, model: str) -> Dict[str, Any]:
        self.load_model(model)
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.infer(img_rgb)

    def infer_raw(
        self, raw_data: bytes, width: int, height: int, model: str
    ) -> Dict[str, Any]:
        self.load_model(model)
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 4)
        img_rgb = img[:, :, :3]
        return self.infer(img_rgb)
