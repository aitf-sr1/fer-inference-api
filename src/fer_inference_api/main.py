import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .pipeline import InferencePipeline
from .schemas import (
    DeviceInfo,
    InferRequest,
    InferResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelsResponse,
)

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


@app.get("/api/models")
def list_models() -> ModelsResponse:
    models = _pipeline.list_models()
    return ModelsResponse(models=models, current=_pipeline.current_model)


@app.get("/api/info")
def info() -> DeviceInfo:
    return DeviceInfo(device=_pipeline.device, worker_pid=os.getpid())


@app.post("/api/model")
def select_model(body: LoadModelRequest) -> LoadModelResponse:
    try:
        _pipeline.load_model(body.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return LoadModelResponse(loaded=body.model)


@app.post("/api/infer")
def infer(body: InferRequest) -> InferResponse:
    try:
        result = _pipeline.infer_base64(body.image, body.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return InferResponse(**result)


@app.post("/api/infer/raw")
async def infer_raw(request: Request):
    width = int(request.headers.get("X-Width", "0"))
    height = int(request.headers.get("X-Height", "0"))
    model = request.headers.get("X-Model", _pipeline.current_model or "")

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Missing X-Width/X-Height headers")
    if not model:
        raise HTTPException(
            status_code=400, detail="No model specified and no current model loaded"
        )

    raw_data = await request.body()
    try:
        result = _pipeline.infer_raw(raw_data, width, height, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content=result)
