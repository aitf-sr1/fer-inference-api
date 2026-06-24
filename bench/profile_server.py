#!/usr/bin/env python3
"""Profile a running API server process with py-spy.

Captures a CPU flamegraph from a live server under load to visualize
where time is spent across all threads/processes.

Usage:
    # Terminal 1: start the API server
    MOCK_MODE=true uv run uvicorn fer_inference_api.main:app --port 8001
    # Find PID with: pgrep -f "uvicorn fer_inference_api"

    # Terminal 2: profile for 30s while load is running
    sudo env "PATH=$PATH" uv run python bench/profile_server.py \
        --pid $(pgrep -f "uvicorn fer_inference_api") --duration 30

    # Terminal 3 (optional): run load while profiling
    uv run locust -f bench/locustfile.py -H http://localhost:8001 -u 4 -r 4 -t 30s --headless

Outputs:
    profile_flamegraph.svg   — flamegraph (open in browser)
    profile_top.txt          — top functions by CPU time
    profile_speedscope.json  — speedscope-compatible profile

Note: py-spy requires root or CAP_SYS_PTRACE to attach to a process.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent


def _find_py_spy() -> str:
    py_spy = shutil.which("py-spy")
    if py_spy:
        return py_spy
    venv_bin = Path(sys.executable).parent / "py-spy"
    if venv_bin.exists():
        return str(venv_bin)
    raise FileNotFoundError(
        "py-spy not found. Install with: uv add --dev py-spy"
    )


def _check_permissions() -> bool:
    if os.geteuid() == 0:
        return True
    try:
        subprocess.run(
            ["py-spy", "record", "--pid", str(os.getpid()),
             "--duration", "0"],
            capture_output=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Profile a running Python process with py-spy"
    )
    parser.add_argument(
        "--pid", type=int, required=True, help="Process ID to profile"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Profile duration in seconds",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=100,
        help="Samples per second (higher = more resolution)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_BENCH_DIR),
        help=f"Output directory (default: {_BENCH_DIR})",
    )
    args = parser.parse_args()

    if not _check_permissions():
        print(
            "py-spy needs root to attach to process. Re-run with:\n"
            f"  sudo env \"PATH=$PATH\" {' '.join(sys.argv)}"
        )
        sys.exit(1)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    py_spy = _find_py_spy()

    duration_s = str(args.duration)

    outputs = [
        (
            outdir / "profile_flamegraph.svg",
            "flamegraph",
        ),
        (
            outdir / "profile_speedscope.json",
            "speedscope",
        ),
    ]

    for output_path, fmt in outputs:
        print(
            f"Recording {fmt} for {args.duration}s at {args.rate} Hz..."
        )
        try:
            subprocess.run(
                [
                    py_spy, "record",
                    "--pid", str(args.pid),
                    "--rate", str(args.rate),
                    "--duration", duration_s,
                    "--format", fmt,
                    "--output", str(output_path),
                ],
                check=True,
                timeout=args.duration + 15,
            )
            print(f"  -> {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}")

    top_path = outdir / "profile_top.txt"
    print(f"Sampling top functions for {args.duration}s...")
    try:
        proc = subprocess.Popen(
            [
                py_spy, "top",
                "--pid", str(args.pid),
                "--rate", str(args.rate),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(args.duration)
        proc.send_signal(signal.SIGINT)
        stdout, _ = proc.communicate(timeout=5)
        top_path.write_text(stdout)
        print(f"  -> {top_path}")
        print(f"\n--- Top functions ---\n{stdout}")
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception as e:
        print(f"  Failed: {e}")

    print("\nDone. Output files:")
    for p in list(outdir.glob("profile_*")):
        print(f"  {p}")

    flame = outdir / "profile_flamegraph.svg"
    if flame.exists():
        print(f"\nOpen flamegraph: xdg-open {flame}")


if __name__ == "__main__":
    main()
