#!/usr/bin/env bash
# Run GPU stress test with 128 concurrent users against the CUDA Docker container.
#
# Prerequisites:
#   1. GPU Docker container running on port 8001
#   2. A face test image prepared with: uv run python bench/prep_test_image.py
#
# Usage:
#   ./bench/run_gpu_sweep.sh [--duration 30] [--concurrency "64 128 256"]

set -euo pipefail

DURATION="${DURATION:-30}"
HOST="${HOST:-http://localhost:8001}"
CONCURRENCY="${CONCURRENCY:-64 128 256}"
FACE_IMAGE="${FACE_IMAGE:-bench/face.jpg}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ -n "$FACE_IMAGE" ]] && [[ -f "$FACE_IMAGE" ]]; then
    export FACE_IMAGE_PATH="$(realpath "$FACE_IMAGE")"
    echo "Face image: $FACE_IMAGE_PATH"
else
    echo "WARNING: No face image at $FACE_IMAGE — using random noise (no ConvNeXt path exercised)"
    unset FACE_IMAGE_PATH
fi

echo "Host:        $HOST"
echo "Duration:    ${DURATION}s per level"
echo "Concurrency: $CONCURRENCY"
echo

uv run python bench/run_sweep.py \
    --host "$HOST" \
    --duration "$DURATION" \
    --concurrency $CONCURRENCY \
    ${FACE_IMAGE_PATH:+--face-image "$FACE_IMAGE_PATH"}
