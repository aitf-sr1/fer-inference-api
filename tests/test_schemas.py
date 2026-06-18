import pytest
from pydantic import ValidationError

from fer_inference_api.schemas import (
    DeviceInfo,
    EmotionPrediction,
    InferRequest,
    InferResponse,
)


def test_infer_request_with_image():
    req = InferRequest(image="base64data")
    assert req.image == "base64data"


def test_infer_request_with_data_url():
    req = InferRequest(image="data:image/jpeg;base64,/9j/4AAQ")
    assert req.image == "data:image/jpeg;base64,/9j/4AAQ"


def test_infer_request_missing_image():
    with pytest.raises(ValidationError):
        InferRequest()


def test_infer_request_empty_image_rejected():
    with pytest.raises(ValidationError):
        InferRequest(image="")


def test_infer_request_oversized_rejected():
    with pytest.raises(ValidationError):
        InferRequest(image="x" * 5_000_001)


def test_infer_response_face_detected():
    resp = InferResponse(
        face_detected=True,
        emotions={
            "Boredom": {"class": 0, "confidence": 85.2}
        },
        num_classes=4,
        inference_ms=15,
        bbox=[0.1, 0.2, 0.5, 0.6],
    )
    assert resp.face_detected is True
    assert resp.emotions == {
        "Boredom": EmotionPrediction(class_=0, confidence=85.2)
    }
    assert resp.num_classes == 4
    assert resp.inference_ms == 15
    assert resp.bbox == [0.1, 0.2, 0.5, 0.6]


def test_infer_response_no_face():
    resp = InferResponse(
        face_detected=False,
        emotions=None,
        num_classes=4,
        inference_ms=None,
        bbox=None,
    )
    assert resp.face_detected is False
    assert resp.emotions is None
    assert resp.inference_ms is None
    assert resp.bbox is None


def test_infer_response_minimal():
    resp = InferResponse(face_detected=False, num_classes=4)
    assert resp.face_detected is False
    assert resp.emotions is None
    assert resp.inference_ms is None
    assert resp.bbox is None


def test_infer_response_missing_face_detected():
    with pytest.raises(ValidationError):
        InferResponse(num_classes=4)


def test_infer_response_missing_num_classes():
    with pytest.raises(ValidationError):
        InferResponse(face_detected=True)


def test_device_info():
    info = DeviceInfo(device="cpu", worker_pid=12345)
    assert info.device == "cpu"
    assert info.worker_pid == 12345


def test_device_info_cuda():
    info = DeviceInfo(device="cuda", worker_pid=1)
    assert info.device == "cuda"
    assert info.worker_pid == 1
