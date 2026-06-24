#!/usr/bin/env python3
"""Run Locust sweeps across multiple API server configurations and compare results.

Usage:
    # Test different worker counts
    uv run bench/run_config_sweep.py \
        --label workers \
        --var "NUM_WORKERS=1" \
        --var "NUM_WORKERS=2" \
        --var "NUM_WORKERS=4" \
        --var "NUM_WORKERS=8"

    # Test OMP thread counts
    uv run bench/run_config_sweep.py \
        --label omp \
        --var "OMP_NUM_THREADS=1" \
        --var "OMP_NUM_THREADS=2" \
        --var "OMP_NUM_THREADS=4"

    # Test mock vs real
    uv run bench/run_config_sweep.py \
        --label mock \
        --var "MOCK_MODE=true" \
        --var "MOCK_MODE=false"

    # Use presets for common scenarios
    uv run bench/run_config_sweep.py --preset workers
    uv run bench/run_config_sweep.py --preset omp
    uv run bench/run_config_sweep.py --preset mock
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent

_SWEEP_SCRIPT = _BENCH_DIR / "run_sweep.py"
_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
_DEFAULT_DURATION = 30
_HEALTH_TIMEOUT = 30


_PRESETS = {
    "workers": {
        "label": "num_workers",
        "env_vars": [
            {"NUM_WORKERS": "1"},
            {"NUM_WORKERS": "2"},
            {"NUM_WORKERS": "4"},
            {"NUM_WORKERS": "8"},
        ],
    },
    "omp": {
        "label": "omp_threads",
        "env_vars": [
            {"NUM_WORKERS": "4", "OMP_NUM_THREADS": "1"},
            {"NUM_WORKERS": "4", "OMP_NUM_THREADS": "2"},
            {"NUM_WORKERS": "4", "OMP_NUM_THREADS": "4"},
            {"NUM_WORKERS": "4", "OMP_NUM_THREADS": "8"},
        ],
    },
    "mock": {
        "label": "mock_vs_real",
        "env_vars": [
            {"MOCK_MODE": "true"},
            {"MOCK_MODE": "false"},
        ],
    },
}


def _start_server(
    port: int, env_vars: dict[str, str]
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_vars)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "fer_inference_api.main:app",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def _wait_healthy(host: str, timeout: int = _HEALTH_TIMEOUT) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"{host}/health", timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait()


def _fetch_metrics(host: str) -> dict | None:
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{host}/api/metrics", timeout=5)
        return json.loads(resp.read())
    except Exception:
        return None


def _run_sweep(
    host: str, duration: int, concurrency: list[int]
) -> dict[int, dict]:
    results = {}
    for users in concurrency:
        print(f"    users={users:>3} ", end="", flush=True)
        r = _run_locust_single(host, users, duration)
        results[users] = r
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(
                f"RPS={r.get('rps', 0):>8.1f}  "
                f"avg={r.get('avg_ms', 0):>8.1f}ms  "
                f"p95={r.get('p95_ms', 0):>8.1f}ms  "
                f"fails={r.get('failures', 0)}"
            )
    return results


def _run_locust_single(
    host: str, users: int, duration: int
) -> dict:
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
                str(_BENCH_DIR / "locustfile.py"),
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
            capture_output=True,
            text=True,
            timeout=duration + 30,
        )

        csv_stats = Path(stats_path + "_stats.csv")
        if csv_stats.exists():
            return _parse_csv(csv_stats)
        return {"error": "no csv output", "stderr": result.stderr}
    finally:
        for p in Path(stats_path).parent.glob(
            f"{Path(stats_path).name}*"
        ):
            p.unlink(missing_ok=True)


def _parse_csv(csv_path: Path) -> dict:
    import csv

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    infer = next((r for r in rows if r["Name"] == "/api/infer"), None)
    if not infer:
        return {"error": "/api/infer not in stats", "raw": rows[:5]}

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


def _var_label(env: dict[str, str]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(env.items())]
    return " ".join(parts) if parts else "default"


def main():
    parser = argparse.ArgumentParser(
        description="Run Locust sweeps across API configs"
    )
    parser.add_argument(
        "--label", default="config", help="Label for this sweep run"
    )
    parser.add_argument(
        "--var",
        action="append",
        dest="env_specs",
        default=[],
        help="Env var overrides (KEY=VALUE [KEY=VALUE...])",
    )
    parser.add_argument(
        "--preset",
        choices=list(_PRESETS.keys()),
        help="Use a preset config sweep",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=_DEFAULT_DURATION,
        help=f"Duration per concurrency level (default: {_DEFAULT_DURATION}s)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=_CONCURRENCIES,
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18001,
        help="Port for the API server (default: 18001)",
    )
    args = parser.parse_args()

    env_configs = []
    if args.preset:
        preset = _PRESETS[args.preset]
        args.label = preset["label"]
        env_configs = [dict(d) for d in preset["env_vars"]]
    elif args.env_specs:
        for spec in args.env_specs:
            cfg = {}
            for part in spec.split():
                k, v = part.split("=", 1)
                cfg[k] = v
            env_configs.append(cfg)
    else:
        env_configs = [{}]

    host = f"http://localhost:{args.port}"
    concurrency = args.concurrency

    print(f"Config sweep: {args.label}")
    print(f"Host: {host}  Duration: {args.duration}s/level")
    print(f"Concurrency: {concurrency}")
    print(f"Configs to test ({len(env_configs)}):")
    for i, cfg in enumerate(env_configs):
        print(f"  [{i}] {_var_label(cfg)}")
    print()

    all_results: dict[str, dict[int, dict]] = {}

    for cfg in env_configs:
        label = _var_label(cfg)
        print(f"=== {label} ===")

        proc = _start_server(args.port, cfg)
        print(f"  Server starting (PID={proc.pid})...", end="", flush=True)

        if not _wait_healthy(host):
            print(" FAILED to start")
            _stop_server(proc)
            all_results[label] = {}
            continue

        # Fetch device info
        try:
            import urllib.request

            info = json.loads(
                urllib.request.urlopen(
                    f"{host}/api/info", timeout=2
                ).read()
            )
            print(
                f" OK (device={info.get('device')}, "
                f"pid={info.get('worker_pid')})"
            )
        except Exception:
            print(" OK")

        try:
            results = _run_sweep(host, args.duration, concurrency)
        finally:
            _stop_server(proc)
            time.sleep(1)

        all_results[label] = results
        print()

    _print_comparison(all_results, concurrency)


def _print_comparison(
    all_results: dict[str, dict[int, dict]], concurrency: list[int]
):
    print("=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)

    configs = list(all_results.keys())
    max_label = max(len(c) for c in configs)

    print()
    header = f"{'Config':<{max_label}}  {'Users':>5}  {'RPS':>8}  {'Avg':>8}  {'P50':>8}  {'P95':>8}  {'P99':>8}  {'Max':>8}  {'Fails':>6}"
    print(header)
    print("-" * len(header))

    for label in configs:
        results = all_results[label]
        for users in concurrency:
            r = results.get(users, {})
            if not r or "error" in r:
                continue
            print(
                f"{label:<{max_label}}  {users:>5}  "
                f"{r.get('rps', 0):>8.1f}  "
                f"{r.get('avg_ms', 0):>8.1f}  "
                f"{r.get('p50_ms', 0):>8.1f}  "
                f"{r.get('p95_ms', 0):>8.1f}  "
                f"{r.get('p99_ms', 0):>8.1f}  "
                f"{r.get('max_ms', 0):>8.0f}  "
                f"{r.get('failures', 0):>6}"
            )
        if len(concurrency) > 1:
            print()

    print("=" * 90)
    print("MAX RPS ACHIEVED")
    print("=" * 90)
    print(f"{'Config':<{max_label}}  {'Max RPS':>8}  {'At Users':>8}  {'P95(ms)':>8}")
    print("-" * (max_label + 30))

    for label in configs:
        results = all_results[label]
        if not results:
            continue
        best_users = max(
            results,
            key=lambda u: results[u].get("rps", 0)
            if results[u]
            and "error" not in results[u]
            else 0,
        )
        best = results.get(best_users, {})
        if not best or "error" in best:
            continue
        print(
            f"{label:<{max_label}}  "
            f"{best.get('rps', 0):>8.1f}  "
            f"{best_users:>8}  "
            f"{best.get('p95_ms', 0):>8.1f}"
        )


if __name__ == "__main__":
    main()
