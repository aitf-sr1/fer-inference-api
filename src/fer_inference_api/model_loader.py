from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

EMOTION_LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_providers() -> List[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_model(checkpoint_path: str) -> Tuple[ort.InferenceSession, int]:
    """
    Load an ONNX model and infer the number of classes per emotion label.

    The output shape determines the classification mode:
    - 2D [batch, num_labels] → binary (2 classes, sigmoid per label)
    - 3D [batch, num_labels, num_classes] → multiclass (softmax per label)
    """
    session = ort.InferenceSession(checkpoint_path, providers=resolve_providers())
    output_shape = session.get_outputs()[0].shape
    if len(output_shape) >= 3 and isinstance(output_shape[2], int):
        num_classes = int(output_shape[2])
    else:
        num_classes = 2
    return session, num_classes


def run_inference(
    session: ort.InferenceSession,
    face_rgb: np.ndarray,
) -> Dict[str, dict]:
    """
    Run inference on a 224x224 face crop.

    Supports two output modes:
    - 2D binary output: sigmoid activations, threshold at 0.5
    - 3D multiclass output: softmax activations, argmax class selection
    """
    img = face_rgb.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = img.transpose(2, 0, 1)[np.newaxis]

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: img})[0]

    if logits.ndim == 2:
        scores = 1.0 / (1.0 + np.exp(-logits.squeeze(0)))
        return {
            label: {
                "class": int(scores[i] >= 0.5),
                "confidence": round(float(scores[i]) * 100, 1),
            }
            for i, label in enumerate(EMOTION_LABELS)
        }

    exp = np.exp(logits - logits.max(axis=2, keepdims=True))
    probs = exp / exp.sum(axis=2, keepdims=True)
    probs = probs.squeeze(0)
    predicted = logits.argmax(axis=2).squeeze(0).tolist()
    return {
        label: {
            "class": int(cls),
            "confidence": round(float(probs[i, int(cls)]) * 100, 1),
        }
        for i, (label, cls) in enumerate(zip(EMOTION_LABELS, predicted))
    }


def provider_name() -> str:
    return resolve_providers()[0].replace("ExecutionProvider", "").lower()
