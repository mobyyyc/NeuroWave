"""Smoke-test the packaged NeuroWave desktop app."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


DEFAULT_APP = Path("dist/win-unpacked/NeuroWave.exe")
DEFAULT_BACKEND_URL = "http://127.0.0.1:8765"
DEFAULT_MODEL = Path("dist/win-unpacked/resources/models/v3.5_noise_detune_loss.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default=str(DEFAULT_APP), help="Packaged NeuroWave executable to launch.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Backend URL started by the app.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Seconds to wait for backend readiness.")
    parser.add_argument("--audio", help="Optional WAV file for an end-to-end prediction smoke test.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model path for optional prediction smoke test.")
    parser.add_argument("--freq-hz", type=float, default=440.0, help="Frequency context for optional prediction.")
    parser.add_argument("--crop-start", type=float, default=0.0, help="Crop start for optional prediction.")
    parser.add_argument("--crop-end", type=float, default=0.433, help="Crop end for optional prediction.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(os.environ.get("LOCALAPPDATA", "runs")) / "NeuroWave" / "Runs"),
        help="Output directory for optional prediction smoke test.",
    )
    parser.add_argument("--run-id", default="desktop_package_smoke", help="Run ID for optional prediction.")
    return parser.parse_args()


def get_json(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(url, payload, timeout=120.0):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def wait_for_health(base_url, timeout, process=None):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Packaged app exited before backend became healthy: code {process.returncode}")
        try:
            status, payload = get_json(f"{base_url}/health", timeout=2.0)
            if status == 200 and payload.get("status") == "ok":
                return payload
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(f"Backend did not become healthy at {base_url}: {last_error}")


def creation_flags():
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def launch_app(path):
    executable = Path(path).resolve()
    if not executable.exists():
        raise FileNotFoundError(f"Packaged app not found: {executable}")
    env = os.environ.copy()
    env.pop("NEUROWAVE_ELECTRON_SMOKE", None)
    return subprocess.Popen(
        [str(executable)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags(),
    )


def stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def verify_runtime(base_url):
    status, payload = get_json(f"{base_url}/runtime", timeout=30.0)
    if status != 200:
        raise RuntimeError(f"/runtime returned HTTP {status}")
    if "python" not in payload or "device" not in payload:
        raise RuntimeError(f"/runtime payload is missing expected fields: {payload}")
    return payload


def verify_prediction(args):
    audio = Path(args.audio)
    model = Path(args.model)
    if not audio.exists():
        raise FileNotFoundError(f"Smoke audio not found: {audio}")
    if not model.exists():
        raise FileNotFoundError(f"Smoke model not found: {model}")
    payload = {
        "audio_path": str(audio.resolve()),
        "model_path": str(model.resolve()),
        "freq_hz": args.freq_hz,
        "crop_start_seconds": args.crop_start,
        "crop_end_seconds": args.crop_end,
        "output_dir": args.output_dir,
        "run_id": args.run_id,
        "device": "cpu",
    }
    status, result = post_json(f"{args.backend_url}/predict", payload, timeout=180.0)
    if status != 200:
        raise RuntimeError(f"/predict returned HTTP {status}")
    required = [
        "target_crop_wav",
        "predicted_patch_json",
        "predicted_wav",
        "target_spectrogram",
        "predicted_spectrogram",
        "summary",
    ]
    missing = [name for name in required if not Path(result.get(name, "")).exists()]
    if missing:
        raise RuntimeError(f"Prediction missing artifacts: {missing}")
    return result


def main():
    args = parse_args()
    process = launch_app(args.app)
    try:
        health = wait_for_health(args.backend_url, args.timeout, process=process)
        runtime = verify_runtime(args.backend_url)
        result = None
        if args.audio:
            result = verify_prediction(args)
        summary = {
            "status": "ok",
            "health": health,
            "runtime": runtime,
            "prediction_run_dir": result.get("run_dir") if result else None,
        }
        print(json.dumps(summary, indent=2))
    finally:
        stop_process(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
