import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .model_loader import create_session_options, resolve_providers

_logger = logging.getLogger(__name__)

INPUT_SIZE = 128
OUTPUT_SIZE = 224

_SCORE_THRESHOLD = 0.5
_IOU_THRESHOLD = 0.3


def _build_anchors() -> np.ndarray:
    anchors = []

    for stride, grid_size, num_types, scales_ar in [
        (
            8,
            16,
            2,
            [0.1484375, 0.33371664137865675, 1.0],
        ),
        (
            16,
            8,
            6,
            [0.75, 1.0067663514340038, 1.0, 0.5, 2.0],
        ),
    ]:
        for gy in range(grid_size):
            for gx in range(grid_size):
                cx = (gx + 0.5) * stride
                cy = (gy + 0.5) * stride
                if num_types == 2:
                    for s in scales_ar[:2]:
                        anchors.append((cx, cy, s, 1.0))
                else:
                    for s in scales_ar[:2]:
                        for ar in scales_ar[2:]:
                            anchors.append((cx, cy, s, ar))

    return np.array(anchors, dtype=np.float32)


_ANCHORS = _build_anchors()


def _decode_boxes(raw_boxes: np.ndarray) -> np.ndarray:
    cx = (_ANCHORS[:, 0] + raw_boxes[:, 0]) / INPUT_SIZE
    cy = (_ANCHORS[:, 1] + raw_boxes[:, 1]) / INPUT_SIZE
    w = raw_boxes[:, 2] / INPUT_SIZE
    h = raw_boxes[:, 3] / INPUT_SIZE

    half_w = w * 0.5
    half_h = h * 0.5
    x1 = np.maximum(cx - half_w, 0.0)
    y1 = np.maximum(cy - half_h, 0.0)
    x2 = np.minimum(cx + half_w, 1.0)
    y2 = np.minimum(cy + half_h, 1.0)
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        union = area_i + area_j - inter
        iou = inter / (union + 1e-8)

        remaining = np.where(iou <= _IOU_THRESHOLD)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.intp)


class FaceDetector:
    def __init__(self, model_path: str) -> None:
        _logger.info("Loading BlazeFace: %s", model_path)
        self._session = ort.InferenceSession(
            model_path,
            sess_options=create_session_options(),
            providers=resolve_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def warmup(self) -> None:
        blob = (
            np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            .astype(np.float32)
            / 255.0
        )[np.newaxis]
        with self._lock:
            self._session.run(None, {self._input_name: blob})

    def detect(
        self, img_rgb: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        h, w = img_rgb.shape[:2]

        resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
        blob = (resized.astype(np.float32) / 255.0)[np.newaxis]

        with self._lock:
            regressors, classificators = self._session.run(
                None, {self._input_name: blob}
            )

        raw_scores = np.clip(classificators[0, :, 0], -88.0, 88.0)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))

        mask = scores >= _SCORE_THRESHOLD
        if not mask.any():
            return None

        raw_boxes = regressors[0]
        boxes = _decode_boxes(raw_boxes)
        valid_boxes = boxes[mask]
        valid_scores = scores[mask]

        indices = _nms(valid_boxes, valid_scores)
        if len(indices) == 0:
            return None

        best = indices[0]
        x1, y1, x2, y2 = valid_boxes[best]

        return (
            float(x1 * w),
            float(y1 * h),
            float(x2 * w),
            float(y2 * h),
        )


def detect_and_crop(
    img_rgb: np.ndarray,
    detector: FaceDetector,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    h, w = img_rgb.shape[:2]

    bbox = detector.detect(img_rgb)
    if bbox is None:
        return None, None

    x_min, y_min, x_max, y_max = bbox
    box_w = x_max - x_min
    box_h = y_max - y_min

    pad_x = int(box_w * 0.05)
    pad_top = int(box_h * 0.20)
    pad_bottom = int(box_h * 0.10)

    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_top)
    x2 = min(w, int(x_max) + pad_x)
    y2 = min(h, int(y_max) + pad_bottom)

    cropped = img_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None

    normalized_bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
    return cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE)), normalized_bbox
