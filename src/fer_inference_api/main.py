import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from .pipeline import InferencePipeline
from .schemas import DeviceInfo, InferRequest, InferResponse

_pipeline: Optional[InferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = InferencePipeline()
    yield
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return InferResponse(**result)
