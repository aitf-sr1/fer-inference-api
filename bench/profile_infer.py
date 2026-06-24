#!/usr/bin/env python3
"""Profile the inference pipeline with cProfile.

Usage:
    # Mock mode (no models needed)
    MOCK_MODE=true uv run python bench/profile_infer.py --requests 500 --output profile.pstats

    # Real mode (needs models)
    uv run python bench/profile_infer.py --requests 100 --output profile.pstats

    # View results with snakeviz
    snakeviz profile.pstats

Profiling targets the pipeline directly (no HTTP overhead) to isolate inference costs.
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fer_inference_api.pipeline import InferencePipeline  # noqa: E402

_POOL_SIZE = 30


def _generate_images() -> list[str]:
    sizes = [
        (480, 640),
        (600, 800),
        (720, 1280),
        (1080, 1920),
        (640, 480),
    ]
    pool = []
    for _ in range(_POOL_SIZE):
        h, w = sizes[_ % len(sizes)]
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        import base64
        pool.append(base64.b64encode(buf).decode())
    return pool


def main():
    parser = argparse.ArgumentParser(
        description="Profile the inference pipeline"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=500,
        help="Number of inferences to profile",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write pstats file for visualization (e.g. snakeviz)",
    )
    args = parser.parse_args()

    print("Loading pipeline...")
    pipeline = InferencePipeline()
    images = _generate_images()
    print(f"Pipeline ready (device={pipeline.device})")
    print(f"Profiling {args.requests} inferences over {len(images)} images...")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.perf_counter()
    results = []
    for i in range(args.requests):
        b64 = images[i % len(images)]
        results.append(pipeline.infer_base64(b64))

    profiler.disable()
    elapsed = time.perf_counter() - t0

    _print_timing_breakdown(results)
    _print_profile_stats(profiler, args.output)

    print(f"\nTotal: {args.requests} requests in {elapsed:.1f}s "
          f"({args.requests / elapsed:.1f} RPS locally)")


def _print_timing_breakdown(results: list[dict]):
    timings = [r.get("timings") for r in results if r.get("timings")]
    if not timings:
        print("(No timing data — run without MOCK_MODE for breakdown)")
        return

    n = len(timings)
    fields = ["total_ms", "decode_ms", "face_detect_ms", "face_crop_ms", "emotion_ms"]

    print(f"Per-step timing breakdown ({n} samples):")
    print(f"{'Step':<18} {'Avg(ms)':>9} {'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'Min(ms)':>9} {'Max(ms)':>9}")
    print("-" * 72)

    for field in fields:
        vals = sorted(
            [t[field] for t in timings if t.get(field) is not None]
        )
        if not vals:
            print(f"{field:<18} {'--':>9}")
            continue
        m = len(vals)
        avg = sum(vals) / m
        print(
            f"{field:<18} "
            f"{avg:>9.2f} "
            f"{vals[int(m * 0.5)]:>9.2f} "
            f"{vals[int(m * 0.95)]:>9.2f} "
            f"{vals[int(m * 0.99)]:>9.2f} "
            f"{vals[0]:>9.2f} "
            f"{vals[-1]:>9.2f}"
        )

    face_detected = sum(1 for r in results if r.get("face_detected"))
    print(f"\nFace detected: {face_detected}/{n} "
          f"({face_detected / n * 100:.1f}%)")


def _print_profile_stats(profiler: cProfile.Profile, output_path: str | None):
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(30)

    print()
    print("cProfile — top 30 functions by cumulative time:")
    print(stream.getvalue())

    if output_path:
        stats.dump_stats(output_path)
        print(f"pstats written to: {output_path}")
        print(f"Visualize with: snakeviz {output_path}")


if __name__ == "__main__":
    main()
