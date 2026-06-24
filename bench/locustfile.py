import base64
import logging
import os
import random

import cv2
import numpy as np
from locust import HttpUser, between, task

_logger = logging.getLogger(__name__)

_POOL_SIZE = 20
_image_pool: list[str] = []

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


def _generate_image(height: int, width: int) -> str:
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


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


def _init_pool():
    global _image_pool
    size_spec = os.environ.get("IMAGE_SIZE", "mixed")
    sizes = _parse_size_spec(size_spec)
    for _ in range(_POOL_SIZE):
        h, w = random.choice(sizes)
        _image_pool.append(_generate_image(h, w))
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
    def infer(self):
        b64 = random.choice(_image_pool)
        self.client.post(
            "/api/infer",
            json={"image": b64},
            name="/api/infer",
        )

    @task(1)
    def metrics(self):
        self.client.get("/api/metrics", name="/api/metrics")
