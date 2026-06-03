import numpy as np

from fer_inference_api.face_detector import (
    _ANCHORS,
    _build_anchors,
    _decode_boxes,
    _nms,
    INPUT_SIZE,
)


class TestBuildAnchors:
    def test_total_count(self):
        assert _ANCHORS.shape == (896, 4)

    def test_dtype(self):
        assert _ANCHORS.dtype == np.float32

    def test_all_within_input_size(self):
        assert (_ANCHORS[:, 0] >= 0).all()
        assert (_ANCHORS[:, 0] <= INPUT_SIZE).all()
        assert (_ANCHORS[:, 1] >= 0).all()
        assert (_ANCHORS[:, 1] <= INPUT_SIZE).all()

    def test_layer0_shape(self):
        anchors = _build_anchors()
        layer0 = anchors[:512]
        assert layer0.shape == (512, 4)

    def test_layer1_shape(self):
        anchors = _build_anchors()
        layer1 = anchors[512:]
        assert layer1.shape == (384, 4)

    def test_idempotent(self):
        a = _build_anchors()
        b = _build_anchors()
        assert np.array_equal(a, b)


class TestDecodeBoxes:
    def test_shape(self):
        raw = np.random.randn(896, 4).astype(np.float32)
        boxes = _decode_boxes(raw)
        assert boxes.shape == (896, 4)

    def test_normalized_range(self):
        raw = np.zeros((896, 4), dtype=np.float32)
        boxes = _decode_boxes(raw)
        assert (boxes >= 0.0).all()
        assert (boxes <= 1.0).all()

    def test_center_at_anchor_zero_dx(self):
        raw = np.zeros((1, 4), dtype=np.float32)
        boxes = _decode_boxes(raw)
        anchor_cx = _ANCHORS[0, 0] / INPUT_SIZE
        anchor_cy = _ANCHORS[0, 1] / INPUT_SIZE
        assert abs(boxes[0, 0] - anchor_cx) < 1e-4
        assert abs(boxes[0, 1] - anchor_cy) < 1e-4
        assert boxes[0, 2] >= boxes[0, 0]
        assert boxes[0, 3] >= boxes[0, 1]

    def test_positive_dx_shifts_right(self):
        raw = np.array([[20.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        boxes = _decode_boxes(raw)
        anchor_cx = _ANCHORS[0, 0] / INPUT_SIZE
        assert boxes[0, 0] > anchor_cx


class TestNMS:
    def test_no_boxes(self):
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        result = _nms(boxes, scores)
        assert len(result) == 0

    def test_single_box(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        result = _nms(boxes, scores)
        assert len(result) == 1
        assert result[0] == 0

    def test_two_overlapping_keeps_higher_score(self):
        boxes = np.array(
            [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]], dtype=np.float32
        )
        scores = np.array([0.9, 0.6], dtype=np.float32)
        result = _nms(boxes, scores)
        assert len(result) == 1
        assert result[0] == 0

    def test_two_non_overlapping_keeps_both(self):
        boxes = np.array(
            [[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        result = _nms(boxes, scores)
        assert len(result) == 2
        assert set(result.tolist()) == {0, 1}

    def test_iou_at_boundary(self):
        boxes = np.array(
            [[0.0, 0.0, 1.0, 1.0], [0.6, 0.6, 1.6, 1.6]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        result = _nms(boxes, scores)
        assert len(result) == 2
