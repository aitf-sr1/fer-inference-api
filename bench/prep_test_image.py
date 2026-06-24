#!/usr/bin/env python3
"""Prepare a test face image for GPU load testing.

Encodes a JPEG image to base64 and saves it as a file that locust
can read via FACE_IMAGE_PATH.

Usage:
    uv run python bench/prep_test_image.py --input /path/to/face.jpg --output bench/face.jpg
    FACE_IMAGE_PATH=bench/face.jpg uv run locust -f bench/locustfile.py ...

The image is also base64-encoded for reference but locust reads the JPEG directly.
"""

import argparse
import base64
import shutil
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test face image for load testing"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to face JPEG/PNG image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bench/face.jpg",
        help="Output JPEG path for locust (default: bench/face.jpg)",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()

    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    img = cv2.imread(str(src))
    if img is None:
        print(f"Error: cannot read {src} — is it a valid image?")
        sys.exit(1)

    h, w = img.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()

    print(f"Image size: {w}x{h}")
    print(f"Saved: {dst}")
    print(f"Base64 length: {len(b64)} chars")
    print()
    print("Ready. Run locust with:")
    print(f"  FACE_IMAGE_PATH={dst} uv run locust -f bench/locustfile.py \\")
    print("    -H http://localhost:8001 -u 128 -r 32 -t 60s --headless")
    print()
    print("Or with the sweep script:")
    print(f"  FACE_IMAGE_PATH={dst} uv run python bench/run_sweep.py \\")
    print("    --host http://localhost:8001 --duration 30")


if __name__ == "__main__":
    main()
