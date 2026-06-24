import base64
import binascii
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from . import model_loader
from .config import settings
from .face_detector import FaceDetector, detect_and_crop

_logger = logging.getLogger(__name__)


def _require_path(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")
    return str(path)


def _random_emotions(num_classes: int) -> Dict[str, dict]:
    return {
        label: {
            "class": random.randint(0, num_classes - 1),
            "confidence": round(random.uniform(10, 95), 1),
        }
        for label in model_loader.EMOTION_LABELS
    }


def _mock_infer(num_classes: int) -> Dict[str, Any]:
    face_detected = random.random() > 0.05
    timings = {
        "total_ms": round(random.uniform(5, 40), 2),
        "decode_ms": round(random.uniform(1, 8), 2),
        "face_detect_ms": round(random.uniform(2, 15), 2),
        "face_crop_ms": round(0, 2),
    }
    if face_detected:
        timings["emotion_ms"] = round(random.uniform(3, 20), 2)
        return {
            "face_detected": True,
            "emotions": _random_emotions(num_classes),
            "num_classes": num_classes,
            "inference_ms": round(timings["emotion_ms"]),
            "bbox": [
                round(random.uniform(0.05, 0.45), 4),
                round(random.uniform(0.05, 0.45), 4),
                round(random.uniform(0.55, 0.95), 4),
                round(random.uniform(0.55, 0.95), 4),
            ],
            "timings": timings,
        }
    return {
        "face_detected": False,
        "emotions": None,
        "num_classes": num_classes,
        "inference_ms": None,
        "bbox": None,
        "timings": timings,
    }


class InferencePipeline:
    def __init__(self) -> None:
        if settings.mock_mode:
            _logger.info("Mock mode enabled — skipping model loading")
            self._mock = True
            self._num_classes = random.randint(2, 4)
            return

        self._mock = False
        blazeface_path = settings.resolved_blazeface_path
        _logger.info("Loading BlazeFace from %s", blazeface_path)
        self._detector = FaceDetector(
            _require_path(blazeface_path, "BlazeFace model")
        )

        model_path = settings.resolved_model_path
        _logger.info("Loading FER model from %s", model_path)
        self._session, self._num_classes = load_model(
            _require_path(model_path, "FER model")
        )

        _logger.info("Warming up models...")
        self._detector.warmup()
        warmup_model(self._session)
        _logger.info("Models ready")

    @property
    def device(self) -> str:
        if self._mock:
            return "mock"
        return provider_name()

    def infer(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        if self._mock:
            return _mock_infer(self._num_classes)

        t_detect = time.perf_counter()
        face, bbox = detect_and_crop(img_rgb, self._detector)
        t_crop = time.perf_counter()

        if face is None:
            return {
                "face_detected": False,
                "emotions": None,
                "num_classes": self._num_classes,
                "inference_ms": None,
                "bbox": None,
                "timings": {
                    "face_detect_ms": round((t_crop - t_detect) * 1000, 2),
                },
            }

        emotions = run_inference(self._session, face)
        t_infer = time.perf_counter()

        face_detect_ms = round((t_crop - t_detect) * 1000, 2)
        face_crop_ms = round(0, 2)
        emotion_ms = round((t_infer - t_crop) * 1000)

        return {
            "face_detected": True,
            "emotions": emotions,
            "num_classes": self._num_classes,
            "inference_ms": emotion_ms,
            "bbox": list(bbox),
            "timings": {
                "face_detect_ms": face_detect_ms,
                "face_crop_ms": face_crop_ms,
                "emotion_ms": emotion_ms,
            },
        }

    def infer_base64(self, b64_image: str) -> Dict[str, Any]:
        t_start = time.perf_counter()

        if self._mock:
            if "," in b64_image:
                b64_image = b64_image.split(",", 1)[1]
            return _mock_infer(self._num_classes)

        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_image, validate=True)
        except binascii.Error:
            raise ValueError("Invalid base64 encoding.")

        t_decode = time.perf_counter()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        decode_ms = round((time.perf_counter() - t_decode) * 1000, 2)

        result = self.infer(img_rgb)
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        result["timings"] = {
            "total_ms": total_ms,
            "decode_ms": decode_ms,
            **(result.get("timings") or {}),
        }

        return result


def load_model(*args, **kwargs):
    return model_loader.load_model(*args, **kwargs)


def provider_name(*args, **kwargs):
    return model_loader.provider_name(*args, **kwargs)


def run_inference(*args, **kwargs):
    return model_loader.run_inference(*args, **kwargs)


def warmup_model(*args, **kwargs):
    return model_loader.warmup(*args, **kwargs)
