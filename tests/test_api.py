import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_endpoint(test_client):
    response = test_client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "device" in data
    assert data["device"] == "cpu"
    assert "worker_pid" in data
    assert isinstance(data["worker_pid"], int)


def test_infer_endpoint_face_detected(test_client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    response = test_client.post("/api/infer", json={"image": b64})
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is True
    assert data["emotions"] is not None
    assert data["num_classes"] == 4
    assert isinstance(data["inference_ms"], int)
    assert data["bbox"] is not None


def test_infer_endpoint_no_face(test_client_no_face):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    response = test_client_no_face.post("/api/infer", json={"image": b64})
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is False
    assert data["emotions"] is None
    assert data["inference_ms"] is None
    assert data["bbox"] is None


def test_infer_endpoint_with_data_url_prefix(test_client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    response = test_client.post(
        "/api/infer", json={"image": f"data:image/jpeg;base64,{b64}"}
    )
    assert response.status_code == 200


def test_infer_endpoint_invalid_base64():
    import fer_inference_api.main as mod

    mock_pipeline = MagicMock()
    mock_pipeline.device = "cpu"
    mock_pipeline.infer_base64.side_effect = ValueError(
        "Could not decode image."
    )

    mod._pipeline = mock_pipeline
    client = TestClient(mod.app)
    response = client.post(
        "/api/infer", json={"image": "dGVzdA=="}
    )
    mod._pipeline = None

    assert response.status_code == 400
    assert "Could not decode image" in response.json()["detail"]


def test_infer_endpoint_missing_image_field(test_client):
    response = test_client.post("/api/infer", json={})
    assert response.status_code == 422


def test_infer_endpoint_binary_body_returns_422(test_client):
    # Raw binary (non-UTF-8) to the JSON endpoint must not crash the server.
    response = test_client.post(
        "/api/infer",
        content=b"\xff\xd8\xff\xbe\x00\x10\x00\x00",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422


def test_infer_raw_endpoint_face_detected(test_client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)

    response = test_client.post(
        "/api/infer/raw",
        content=buf.tobytes(),
        headers={"Content-Type": "image/jpeg"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is True
    assert data["emotions"] is not None
    assert data["num_classes"] == 4
    assert isinstance(data["inference_ms"], int)
    assert data["bbox"] is not None


def test_infer_raw_endpoint_no_face(test_client_no_face):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)

    response = test_client_no_face.post("/api/infer/raw", content=buf.tobytes())
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is False
    assert data["emotions"] is None


def test_infer_raw_endpoint_empty_body(test_client):
    response = test_client.post("/api/infer/raw", content=b"")
    assert response.status_code == 400


def test_infer_raw_endpoint_invalid_image():
    import fer_inference_api.main as mod

    mock_pipeline = MagicMock()
    mock_pipeline.device = "cpu"
    mock_pipeline.infer_bytes.side_effect = ValueError("Could not decode image.")

    mod._pipeline = mock_pipeline
    client = TestClient(mod.app)
    response = client.post("/api/infer/raw", content=b"not an image")
    mod._pipeline = None

    assert response.status_code == 400
    assert "Could not decode image" in response.json()["detail"]


def test_infer_endpoint_500_error():
    import fer_inference_api.main as mod

    mock_pipeline = MagicMock()
    mock_pipeline.device = "cpu"
    mock_pipeline.infer_base64.side_effect = RuntimeError("GPU out of memory")

    mod._pipeline = mock_pipeline
    client = TestClient(mod.app)
    response = client.post(
        "/api/infer", json={"image": "dGVzdA=="}
    )
    mod._pipeline = None

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"


def test_metrics_endpoint_initial_empty(test_client):
    import fer_inference_api.main as mod

    mod._metrics.reset()
    response = test_client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["requests"] == 0
    assert data["avg_total_ms"] is None


def test_metrics_endpoint_after_infer(test_client):
    import fer_inference_api.main as mod

    mod._metrics.reset()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    for _ in range(5):
        test_client.post("/api/infer", json={"image": b64})

    response = test_client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["requests"] >= 1
    assert data["avg_total_ms"] is not None
    assert data["p50_total_ms"] is not None
    assert data["avg_decode_ms"] is not None
    assert data["avg_face_detect_ms"] is not None



