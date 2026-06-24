import base64
import logging
import random

import cv2
import numpy as np
from locust import HttpUser, between, task

_logger = logging.getLogger(__name__)

_POOL_SIZE = 20
_image_pool: list[str] = []


def _generate_image(height: int, width: int) -> str:
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _init_pool():
    global _image_pool
    sizes = [
        (480, 640),
        (600, 800),
        (720, 1280),
        (1080, 1920),
        (640, 480),
    ]
    for _ in range(_POOL_SIZE):
        h, w = random.choice(sizes)
        _image_pool.append(_generate_image(h, w))
    _logger.info(
        "Prepared %d test images (sizes: %s)",
        _POOL_SIZE,
        [f"{h}x{w}" for h, w in sizes],
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
