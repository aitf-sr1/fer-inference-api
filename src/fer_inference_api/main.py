import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from .pipeline import InferencePipeline
from .schemas import DeviceInfo, InferRequest, InferResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)

_logger = logging.getLogger(__name__)
_pipeline: Optional[InferencePipeline] = None


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
    return InferResponse(**result)
