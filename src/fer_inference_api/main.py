import collections
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException

from .pipeline import InferencePipeline
from .schemas import DeviceInfo, InferRequest, InferResponse, MetricsSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)

_logger = logging.getLogger(__name__)
_pipeline: Optional[InferencePipeline] = None


class _MetricsCollector:
    def __init__(self, max_samples: int = 2000):
        self._lock = threading.Lock()
        self._samples: collections.deque = collections.deque(maxlen=max_samples)

    def record(self, timings: Dict[str, Optional[float]]) -> None:
        with self._lock:
            self._samples.append(timings)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def stats(self) -> MetricsSummary:
        with self._lock:
            if not self._samples:
                return MetricsSummary(requests=0)
            totals = sorted(s["total_ms"] for s in self._samples if s.get("total_ms"))
            n = len(totals)
            if n == 0:
                return MetricsSummary(requests=len(self._samples))

            def _avg(key: str) -> Optional[float]:
                vals = [s[key] for s in self._samples if s.get(key) is not None]
                return round(sum(vals) / len(vals), 2) if vals else None

            return MetricsSummary(
                requests=n,
                avg_total_ms=round(sum(totals) / n, 2),
                p50_total_ms=totals[int(n * 0.5)],
                p95_total_ms=totals[int(n * 0.95)],
                p99_total_ms=totals[int(n * 0.99)],
                avg_decode_ms=_avg("decode_ms"),
                avg_face_detect_ms=_avg("face_detect_ms"),
                avg_face_crop_ms=_avg("face_crop_ms"),
                avg_emotion_ms=_avg("emotion_ms"),
            )


_metrics = _MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _logger.info("Loading models...")
    _pipeline = InferencePipeline()
    _logger.info("Pipeline ready")
    yield
    _logger.info("Shutting down")
    _pipeline = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/info")
def info() -> DeviceInfo:
    return DeviceInfo(device=_pipeline.device, worker_pid=os.getpid())


@app.get("/api/metrics")
def metrics() -> MetricsSummary:
    return _metrics.stats()


@app.post("/api/infer")
def infer(body: InferRequest) -> InferResponse:
    try:
        result = _pipeline.infer_base64(body.image)
    except ValueError as e:
        _logger.warning("Invalid request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        _logger.exception("Unhandled error during inference")
        raise HTTPException(status_code=500, detail="Internal server error")
    _metrics.record(result.get("timings") or {})
    return InferResponse(**result)
