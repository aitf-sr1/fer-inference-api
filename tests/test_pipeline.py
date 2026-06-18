import base64

import cv2
import numpy as np
import pytest

from fer_inference_api.face_detector import FaceDetector
from fer_inference_api.pipeline import InferencePipeline


@pytest.fixture
def sample_face_image():
    img = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_base64_jpeg():
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def test_infer_base64_strips_data_url_prefix(monkeypatch):
    class _FakePipeline(InferencePipeline):
        def __init__(self):
            self._mock = False
            self._session = _FakeSession()
            self._detector = _FakeDetector()
            self._num_classes = 4

        @property
        def device(self):
            return "cpu"

    def _mock_init(self2):
        self2._mock = False
        self2._session = _FakeSession()
        self2._detector = _FakeDetector()
        self2._num_classes = 4

    monkeypatch.setattr(InferencePipeline, "__init__", _mock_init)
    monkeypatch.setattr(InferencePipeline, "device", property(lambda self: "cpu"))

    pipeline = InferencePipeline()
    result = pipeline.infer_base64(
        "data:image/jpeg;base64,"
        + base64.b64encode(
            cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1]
        ).decode()
    )
    assert "face_detected" in result
    assert "emotions" in result
    assert "num_classes" in result
    assert "inference_ms" in result
    assert "bbox" in result


def test_infer_base64_raises_on_invalid_base64(monkeypatch):
    def _mock_init(self2):
        self2._mock = False
        self2._session = _FakeSession()
        self2._detector = _FakeDetector()
        self2._num_classes = 4

    monkeypatch.setattr(InferencePipeline, "__init__", _mock_init)

    pipeline = InferencePipeline()
    bad_b64 = base64.b64encode(b"not an image").decode()
    with pytest.raises(ValueError, match="Could not decode image"):
        pipeline.infer_base64(bad_b64)


def test_infer_no_face_returns_none(monkeypatch):
    class _NoFacePipeline(InferencePipeline):
        def infer(self, img_rgb):
            return {
                "face_detected": False,
                "emotions": None,
                "num_classes": 4,
                "inference_ms": None,
                "bbox": None,
            }

    monkeypatch.setattr(
        InferencePipeline, "__init__", lambda self: None
    )

    pipeline = _NoFacePipeline()
    result = pipeline.infer(np.zeros((640, 480, 3), dtype=np.uint8))
    assert result["face_detected"] is False
    assert result["emotions"] is None
    assert result["inference_ms"] is None
    assert result["bbox"] is None


class _FakeSession:
    def get_inputs(self):
        return [_FakeNode("input")]

    def run(self, output_names, feed_dict):
        return [np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)]


class _FakeDetector:
    def detect(self, img_rgb):
        return (100.0, 80.0, 350.0, 400.0)


class _FakeNode:
    def __init__(self, name):
        self.name = name
