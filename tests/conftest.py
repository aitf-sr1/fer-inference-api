import base64
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fer_inference_api.config import Settings


@pytest.fixture
def dummy_rgb_image() -> np.ndarray:
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_face_crop() -> np.ndarray:
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_base64_jpeg() -> str:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    _, buf = __import__("cv2").imencode(".jpg", img)
    return base64.b64encode(buf).decode()


@pytest.fixture
def mock_face_detector_bbox() -> tuple:
    return (100.0, 80.0, 350.0, 400.0)


@pytest.fixture
def mock_emotions() -> Dict[str, Dict[str, Any]]:
    return {
        "Boredom": {"class": 0, "confidence": 85.2},
        "Engagement": {"class": 3, "confidence": 67.1},
        "Confusion": {"class": 1, "confidence": 23.5},
        "Frustration": {"class": 0, "confidence": 12.3},
    }


@pytest.fixture
def mock_pipeline_result(mock_emotions) -> Dict[str, Any]:
    return {
        "face_detected": True,
        "emotions": mock_emotions,
        "num_classes": 4,
        "inference_ms": 15,
        "bbox": [0.15625, 0.16666666666666666, 0.546875, 0.8333333333333334],
        "timings": {
            "total_ms": 25.5,
            "decode_ms": 5.2,
            "face_detect_ms": 10.1,
            "face_crop_ms": 0.0,
            "emotion_ms": 15,
        },
    }


@pytest.fixture
def mock_pipeline_no_face() -> Dict[str, Any]:
    return {
        "face_detected": False,
        "emotions": None,
        "num_classes": 4,
        "inference_ms": None,
        "bbox": None,
        "timings": {
            "total_ms": 8.3,
            "decode_ms": 5.1,
            "face_detect_ms": 3.2,
        },
    }


@pytest.fixture
def test_client(mock_pipeline_result):
    import fer_inference_api.main as mod

    mock_pipeline = MagicMock()
    mock_pipeline.device = "cpu"
    mock_pipeline.infer_base64.return_value = mock_pipeline_result
    mock_pipeline.infer_bytes.return_value = mock_pipeline_result
    mock_pipeline.infer.return_value = mock_pipeline_result

    mod._pipeline = mock_pipeline
    yield TestClient(mod.app)
    mod._pipeline = None


@pytest.fixture
def test_client_no_face(mock_pipeline_no_face):
    import fer_inference_api.main as mod

    mock_pipeline = MagicMock()
    mock_pipeline.device = "cpu"
    mock_pipeline.infer_base64.return_value = mock_pipeline_no_face
    mock_pipeline.infer_bytes.return_value = mock_pipeline_no_face
    mock_pipeline.infer.return_value = mock_pipeline_no_face

    mod._pipeline = mock_pipeline
    yield TestClient(mod.app)
    mod._pipeline = None


@pytest.fixture
def test_settings():
    return Settings()
