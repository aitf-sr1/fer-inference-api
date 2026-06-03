import numpy as np

from fer_inference_api.model_loader import (
    EMOTION_LABELS,
    _resolve_providers,
    provider_name,
    run_inference,
)


class TestResolveProviders:
    def test_returns_cpu(self):
        providers = _resolve_providers()
        assert "CPUExecutionProvider" in providers

    def test_cpu_is_first_when_no_cuda(self):
        providers = _resolve_providers()
        assert providers[0] == "CPUExecutionProvider"


class TestProviderName:
    def test_returns_lowercase_no_suffix(self):
        name = provider_name()
        assert name == "cpu"


class TestEmotionLabels:
    def test_four_labels(self):
        assert len(EMOTION_LABELS) == 4

    def test_expected_order(self):
        assert EMOTION_LABELS == [
            "Boredom",
            "Engagement",
            "Confusion",
            "Frustration",
        ]


class TestRunInference:
    def test_binary_output(self):
        logits = np.array([[-2.0, 2.0, -1.0, 3.0]], dtype=np.float32)
        mock_session = _MockSession(logits)
        result = run_inference(mock_session, _dummy_face())
        assert len(result) == 4
        assert result["Boredom"]["class"] == 0
        assert result["Engagement"]["class"] == 1
        assert result["Confusion"]["class"] == 0
        assert result["Frustration"]["class"] == 1
        assert result["Engagement"]["confidence"] > 50
        assert result["Frustration"]["confidence"] > 50

    def test_binary_confidence_range(self):
        logits = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        mock_session = _MockSession(logits)
        result = run_inference(mock_session, _dummy_face())
        for label in EMOTION_LABELS:
            assert 0 <= result[label]["confidence"] <= 100

    def test_multiclass_output(self):
        logits = np.array(
            [[[0.1, 0.2, 0.3, 0.4],
              [1.0, 0.5, 0.2, 0.1],
              [0.1, 0.1, 5.0, 0.1],
              [0.2, 0.3, 0.1, 2.0]]], dtype=np.float32
        )
        mock_session = _MockSession(logits)
        result = run_inference(mock_session, _dummy_face())
        assert result["Boredom"]["class"] == 3
        assert result["Engagement"]["class"] == 0
        assert result["Confusion"]["class"] == 2
        assert result["Frustration"]["class"] == 3

    def test_multiclass_confidence_range(self):
        logits = np.random.randn(1, 4, 4).astype(np.float32)
        mock_session = _MockSession(logits)
        result = run_inference(mock_session, _dummy_face())
        for label in EMOTION_LABELS:
            assert 0 <= result[label]["confidence"] <= 100

    def test_preprocessing_applies_normalization(self):
        face = np.full((224, 224, 3), 128, dtype=np.uint8)
        logits = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        mock_session = _CapturingSession(logits)
        run_inference(mock_session, face)
        captured_input = mock_session.captured_input
        assert captured_input is not None
        assert captured_input.shape == (1, 3, 224, 224)
        assert captured_input.dtype == np.float32
        assert captured_input.max() <= 1.0


def _dummy_face():
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


class _MockSession:
    def __init__(self, output):
        self._output = output

    def get_inputs(self):
        return [_MockNode("input")]

    def run(self, output_names, feed_dict):
        return [self._output]


class _CapturingSession:
    def __init__(self, output):
        self._output = output
        self.captured_input = None

    def get_inputs(self):
        return [_MockNode("input")]

    def run(self, output_names, feed_dict):
        self.captured_input = list(feed_dict.values())[0]
        return [self._output]


class _MockNode:
    def __init__(self, name):
        self.name = name
