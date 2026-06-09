# fer-inference-api

Facial Expression Recognition inference API.

## Prerequisites

**CPU mode:**
- Docker & Docker Compose

**GPU mode (additional):**
- NVIDIA GPU + driver (`nvidia-smi`)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker restarted after toolkit install

- ONNX model files (see [Model Files](#model-files))

## Quick Start

### CPU

```bash
git clone git@github.com:aitf-sr1/fer-inference-api.git
cd fer-inference-api

# Create your config from the example
cp .env.example .env

# Download model files (see below for details)
# BlazeFace
huggingface-cli download unity/inference-engine-blaze-face \
  blaze_face_short_range.onnx --local-dir assets/

# ConvNeXt V2 FER model (graph + external weights)
huggingface-cli download aitf-ub-2026/ub-sr-01-model-fer-convnextv2-datasetv2 \
  convnextv2_atto.onnx convnextv2_atto.onnx.data --local-dir models/

# Start the API
docker compose up -d --build
```

The API is available at `http://localhost:8001`.

### GPU

```bash
# Use GPU-specific config (fewer workers to fit VRAM)
cp .env.gpu.example .env

# Build and run with GPU support
docker compose -f docker-compose.gpu.yml up -d --build

# Verify GPU is active
curl http://localhost:8001/api/info
# {"device":"cuda","worker_pid":...}
```

To stop: `docker compose -f docker-compose.gpu.yml down`.

## Model Files

Two ONNX model files are required:

| Model     | Files                          | Default Path                         | Description                          |
| --------- | ------------------------------ | ------------------------------------ | ------------------------------------ |
| BlazeFace | `blaze_face_short_range.onnx`  | `assets/blaze_face_short_range.onnx` | Face detection model (128x128 input) |
| ConvNeXt  | `convnextv2_atto.onnx` + `.data` | `models/convnextv2_atto.onnx`        | FER model (224x224 input)            |

### Download

**BlazeFace** — [unity/inference-engine-blaze-face](https://huggingface.co/unity/inference-engine-blaze-face/tree/main/models)

```bash
huggingface-cli download unity/inference-engine-blaze-face \
  blaze_face_short_range.onnx --local-dir assets/
```

**ConvNeXt V2** (FER) — [aitf-ub-2026/ub-sr-01-model-fer-convnextv2-datasetv2](https://huggingface.co/aitf-ub-2026/ub-sr-01-model-fer-convnextv2-datasetv2/tree/main)

```bash
huggingface-cli download aitf-ub-2026/ub-sr-01-model-fer-convnextv2-datasetv2 \
  convnextv2_atto.onnx convnextv2_atto.onnx.data --local-dir models/
```

> The ConvNeXt model uses ONNX external data — the `.onnx` file contains the graph, the `.onnx.data` file contains the weights. Both must be present in the same directory.

Paths can be overridden via environment variables (see [Configuration](#configuration)).

## Endpoints

### `GET /health`

Health check.

```json
{ "status": "ok" }
```

### `GET /api/info`

Inference device and worker PID.

```json
{ "device": "cpu", "worker_pid": 12345 }
```

### `POST /api/infer`

Run inference on a base64-encoded JPEG image.

**Request:**

```json
{ "image": "<base64>" }
```

The `image` field accepts raw base64 or a data URL (`data:image/jpeg;base64,...`).

**Response (face detected):**

```json
{
  "face_detected": true,
  "emotions": {
    "Boredom": { "class": 0, "confidence": 85.2 },
    "Engagement": { "class": 3, "confidence": 67.1 },
    "Confusion": { "class": 1, "confidence": 23.5 },
    "Frustration": { "class": 0, "confidence": 12.3 }
  },
  "num_classes": 4,
  "inference_ms": 15,
  "bbox": [0.12, 0.34, 0.56, 0.78]
}
```

**Response (no face detected):**

```json
{
  "face_detected": false,
  "emotions": null,
  "num_classes": 4,
  "inference_ms": null,
  "bbox": null
}
```

## Configuration

Copy `.env.example` (CPU) or `.env.gpu.example` (GPU) to `.env` and adjust as needed.

| Variable               | CPU default | GPU default | Description                 |
| ---------------------- | ----------- | ----------- | --------------------------- |
| `MODEL_PATH`           | `/app/models/convnextv2_atto.onnx` | same | Path to the FER ONNX model |
| `BLAZEFACE_MODEL_PATH` | `/app/assets/blaze_face_short_range.onnx` | same | Path to BlazeFace ONNX model |
| `NUM_WORKERS`          | `8`         | `2`         | Gunicorn worker processes   |
| `HOST_PORT`            | `8001`      | `8001`      | Host port                   |

## Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
src/fer_inference_api/
├── config.py          # Settings from env
├── face_detector.py   # ONNX BlazeFace, 896 anchors, greedy NMS
├── model_loader.py    # ImageNet normalization + ONNX inference
├── pipeline.py        # Face detect -> crop 224x224 -> classify
├── schemas.py         # Pydantic request/response models
└── main.py            # FastAPI app (3 endpoints)
```
