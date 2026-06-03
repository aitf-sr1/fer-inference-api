from collections import OrderedDict
from threading import Lock
from typing import List, Optional, Tuple

import onnxruntime as ort

from .config import settings
from .model_loader import load_model, provider_name


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, Tuple[ort.InferenceSession, int]] = OrderedDict()
        self._lock = Lock()
        self._current: Optional[str] = None

    def get(self, model_name: str) -> Tuple[ort.InferenceSession, int]:
        with self._lock:
            if model_name in self._cache:
                self._cache.move_to_end(model_name)
                self._current = model_name
                return self._cache[model_name]

            path = settings.resolved_models_dir / model_name
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")

            session, num_classes = load_model(str(path))

            if len(self._cache) >= settings.max_cached_models:
                self._cache.popitem(last=False)

            self._cache[model_name] = (session, num_classes)
            self._current = model_name
            return session, num_classes

    @property
    def current_model(self) -> Optional[str]:
        return self._current

    def list_models(self) -> List[str]:
        return sorted(
            p.name for p in settings.resolved_models_dir.glob("*.onnx")
        )

    @property
    def device(self) -> str:
        return provider_name()
