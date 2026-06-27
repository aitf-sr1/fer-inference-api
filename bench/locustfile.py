import base64
import logging
import os
import random

import cv2
import numpy as np
from locust import HttpUser, between, events, task

_logger = logging.getLogger(__name__)

_POOL_SIZE = 20
_image_pool: list[str] = []
_raw_image_pool: list[bytes] = []
_target = "both"

_SIZE_PRESETS = {
    "small": [(240, 320), (320, 240)],
    "medium": [(480, 640), (600, 800), (640, 480)],
    "large": [(720, 1280), (1080, 1920), (1920, 1080)],
    "mixed": [
        (480, 640),
        (600, 800),
        (720, 1280),
        (1080, 1920),
        (640, 480),
    ],
}


def _generate_image(height: int, width: int) -> tuple[str, bytes]:
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return b64, bytes(buf)


def _parse_size_spec(spec: str | None) -> list[tuple[int, int]]:
    if not spec:
        return _SIZE_PRESETS["mixed"]
    if spec in _SIZE_PRESETS:
        return _SIZE_PRESETS[spec]
    if "x" in spec:
        try:
            h, w = spec.split("x")
            return [(int(h), int(w))]
        except ValueError:
            pass
    return _SIZE_PRESETS["mixed"]


def _load_face_image(path: str) -> tuple[str, bytes]:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return b64, bytes(buf)


@events.init_command_line_parser.add_listener
def _add_type_arg(parser):
    parser.add_argument(
        "--type",
        choices=["json", "raw", "both"],
        default="both",
        help="Endpoint to test: json (/api/infer), raw (/api/infer/raw), or both",
    )


@events.init.add_listener
def _apply_type(environment, **kw):
    global _target
    _target = environment.parsed_options.type
    _logger.info("Endpoint type: %s", _target)


def _init_pool():
    global _image_pool, _raw_image_pool

    face_path = os.environ.get("FACE_IMAGE_PATH")
    if face_path:
        b64, raw = _load_face_image(face_path)
        _image_pool = [b64] * _POOL_SIZE
        _raw_image_pool = [raw] * _POOL_SIZE
        _logger.info("Loaded face image from %s", face_path)
        return

    size_spec = os.environ.get("IMAGE_SIZE", "mixed")
    sizes = _parse_size_spec(size_spec)
    for _ in range(_POOL_SIZE):
        h, w = random.choice(sizes)
        b64, raw = _generate_image(h, w)
        _image_pool.append(b64)
        _raw_image_pool.append(raw)
    _logger.info(
        "Prepared %d test images (sizes: %s, spec=%s)",
        _POOL_SIZE,
        [f"{h}x{w}" for h, w in sizes],
        size_spec,
    )


_init_pool()


class InferUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(50)
    def infer_json(self):
        if _target not in ("json", "both"):
            return
        b64 = random.choice(_image_pool)
        self.client.post(
            "/api/infer",
            json={"image": b64},
            name="/api/infer",
        )

    @task(50)
    def infer_raw(self):
        if _target not in ("raw", "both"):
            return
        raw = random.choice(_raw_image_pool)
        self.client.post(
            "/api/infer/raw",
            data=raw,
            name="/api/infer/raw",
        )

    @task(1)
    def metrics(self):
        self.client.get("/api/metrics", name="/api/metrics")
