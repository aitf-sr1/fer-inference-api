#!/usr/bin/env python3
"""Run Locust load tests across multiple concurrency levels and report results.

Usage:
    uv run bench/run_sweep.py [--host HOST] [--duration SECONDS] [--image-size small|medium|large|mixed|WxH]

Before running, start the API server in another terminal:
    uv run uvicorn fer_inference_api.main:app --port 8001
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_LOCUSTFILE = _BENCH_DIR / "locustfile.py"

_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]


def run_locust(
    host: str, users: int, duration: int, image_size: str | None = None
) -> dict:
    env = os.environ.copy()
    if image_size:
        env["IMAGE_SIZE"] = image_size

    with tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w+", delete=False
    ) as stats_file:
        stats_path = stats_file.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "locust",
                "-f",
                str(_LOCUSTFILE),
                "-H",
                host,
                "-u",
                str(users),
                "-r",
                str(max(1, users // 4)),
                "-t",
                f"{duration}s",
                "--headless",
                "--csv",
                stats_path,
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=duration + 30,
            env=env,
        )

        csv_stats = Path(stats_path + "_stats.csv")
        if csv_stats.exists():
            return _parse_csv_stats(csv_stats)
        return {"error": "no csv output", "stderr": result.stderr}
    finally:
        for p in Path(stats_path).parent.glob(f"{Path(stats_path).name}*"):
            p.unlink(missing_ok=True)


def _parse_csv_stats(csv_path: Path) -> dict:
    import csv

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    infer = next((r for r in rows if r["Name"] == "/api/infer"), None)
    if not infer:
        return {"error": "/api/infer not found in stats", "raw": rows[:5]}

    return {
        "requests": int(infer["Request Count"]),
        "failures": int(infer["Failure Count"]),
        "rps": float(infer["Requests/s"]),
        "avg_ms": float(infer["Average Response Time"]),
        "p50_ms": float(infer["Median Response Time"]),
        "p90_ms": float(infer["90%"]),
        "p95_ms": float(infer["95%"]),
        "p99_ms": float(infer["99%"]),
        "max_ms": float(infer["Max Response Time"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Locust concurrency sweep")
    parser.add_argument("--host", default="http://localhost:8001")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument(
        "--image-size",
        choices=["small", "medium", "large", "mixed"],
        default="mixed",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=_CONCURRENCIES,
        help=f"Concurrency levels to sweep (default: {_CONCURRENCIES})",
    )
    parser.add_argument(
        "--face-image",
        type=str,
        default=None,
        help="Path to JPEG face image for realistic inference testing",
    )
    args = parser.parse_args()

    if args.face_image:
        os.environ["FACE_IMAGE_PATH"] = os.path.abspath(args.face_image)

    concurrency = args.concurrency

    print(f"Target: {args.host}")
    print(f"Duration per level: {args.duration}s")
    print(f"Image size: {args.image_size}")
    print(f"Face image: {args.face_image or 'random noise'}")
    print(f"Concurrency levels: {concurrency}")
    print()
    print(
        f"{'Users':>6} {'RPS':>8} {'Avg(ms)':>9} "
        f"{'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'Max(ms)':>9} "
        f"{'Reqs':>7} {'Fails':>7}"
    )
    print("-" * 85)

    results = []
    for users in concurrency:
        print(f"{users:>6} ", end="", flush=True)
        result = run_locust(args.host, users, args.duration, args.image_size)
        results.append((users, result))

        if "error" in result:
            print(f"ERROR: {result['error']}")
            if "stderr" in result:
                print(result["stderr"])
        else:
            print(
                f"{result['rps']:>8.1f} {result['avg_ms']:>9.1f} "
                f"{result['p50_ms']:>9.1f} {result['p95_ms']:>9.1f} "
                f"{result['p99_ms']:>9.1f} {result['max_ms']:>9.0f} "
                f"{result['requests']:>7} {result['failures']:>7}"
            )

    print()
    print(
        f"{'Users':>6} {'Decode':>8} {'FaceDet':>8} "
        f"{'Crop':>8} {'Emotion':>8} {'Total':>8}"
    )
    print("-" * 55)
    for users, _result in results:
        time.sleep(0.5)
        m = _fetch_metrics(args.host)
        if m:
            print(
                f"{users:>6} {m['avg_decode_ms'] or 0:>8.1f} "
                f"{m['avg_face_detect_ms'] or 0:>8.1f} "
                f"{m['avg_face_crop_ms'] or 0:>8.1f} "
                f"{m['avg_emotion_ms'] or 0:>8.1f} "
                f"{m['avg_total_ms'] or 0:>8.1f}"
            )


def _fetch_metrics(host: str) -> dict | None:
    try:
        import urllib.request

        resp = urllib.request.urlopen(f"{host}/api/metrics", timeout=5)
        return json.loads(resp.read())
    except Exception:
        return None


if __name__ == "__main__":
    main()
