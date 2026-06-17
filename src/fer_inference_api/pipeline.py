import base64
import binascii
import time
from typing import Any, Dict

import cv2
import numpy as np

from .config import settings
from .face_detector import FaceDetector, detect_and_crop
from .model_loader import load_model, provider_name, run_inference


class InferencePipeline:
    def __init__(self) -> None:
        blazeface_path = settings.resolved_blazeface_path
        if not blazeface_path.exists():
            raise FileNotFoundError(
                f"BlazeFace model not found at {blazeface_path}"
            )
        self._detector = FaceDetector(str(blazeface_path))

        model_path = settings.resolved_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self._session, self._num_classes = load_model(str(model_path))

    @property
    def device(self) -> str:
        return provider_name()

    def infer(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        face, bbox = detect_and_crop(img_rgb, self._detector)
        if face is None:
            return {
                "face_detected": False,
                "emotions": None,
                "num_classes": self._num_classes,
                "inference_ms": None,
                "bbox": None,
            }
        t0 = time.perf_counter()
        emotions = run_inference(self._session, face)
        inference_ms = round((time.perf_counter() - t0) * 1000)
        return {
            "face_detected": True,
            "emotions": emotions,
            "num_classes": self._num_classes,
            "inference_ms": inference_ms,
            "bbox": list(bbox),
        }

    def infer_base64(self, b64_image: str) -> Dict[str, Any]:
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_image, validate=True)
        except binascii.Error:
            raise ValueError("Invalid base64 encoding.")
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.infer(img_rgb)
