# fer-inference-api

Facial Expression Recognition inference API.

## Prerequisites

- Docker & Docker Compose
- ONNX model files (see [Model Files](#model-files))

## Quick Start

```bash
git clone git@github.com:aitf-sr1/fer-inference-api.git
cd fer-inference-api

# Place model files (required)
cp /path/to/blaze_face_short_range.onnx assets/
cp /path/to/your_fer_model.onnx models/

# Start the stack (API + nginx with TLS)
docker compose up -d --build
```

The API is available at `https://localhost` (self-signed cert).

To stop: `docker compose down`.

## Model Files

Two ONNX model files are required:

| File      | Default Path                         | Description                                  |
| --------- | ------------------------------------ | -------------------------------------------- |
| BlazeFace | `assets/blaze_face_short_range.onnx` | Face detection model (128x128 input)         |
| FER model | `models/model.onnx`                  | Emotion classification model (224x224 input) |

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

Settings are loaded from environment variables or a `.env` file:

| Variable               | Default                                | Description                      |
| ---------------------- | -------------------------------------- | -------------------------------- |
| `MODEL_PATH`           | `./models/model.onnx`                  | Path to the FER ONNX model       |
| `BLAZEFACE_MODEL_PATH` | `./assets/blaze_face_short_range.onnx` | Path to the BlazeFace ONNX model |
| `LOG_LEVEL`            | `info`                                 | Logging level                    |

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
