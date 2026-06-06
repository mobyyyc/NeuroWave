"""Validate the Python runtime that will be bundled with the Windows app."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_RUNTIME_DIR = Path("runtime/python")
REQUIRED_MODULES = (
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "sklearn",
    "torch",
)


PROBE_CODE = r"""
import importlib
import json
import platform
import sys

result = {
    "executable": sys.executable,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "modules": {},
    "torch": {},
    "neurowave_imports": None,
}

missing = []
for name in REQUIRED_MODULES:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {
            "ok": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as error:
        result["modules"][name] = {
            "ok": False,
            "error": str(error),
        }
        missing.append(name)

try:
    import torch

    result["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as error:
    result["torch"] = {
        "error": str(error),
        "cuda_available": False,
        "cuda_device": None,
    }

try:
    from minisynth.app_inference import AppInferenceRequest
    from minisynth.features import mel_spectrogram
    from minisynth.torch_model import predict_patch_from_audio

    result["neurowave_imports"] = {
        "ok": bool(AppInferenceRequest and mel_spectrogram and predict_patch_from_audio),
    }
except Exception as error:
    result["neurowave_imports"] = {
        "ok": False,
        "error": str(error),
    }

result["ok"] = not missing and bool(result["neurowave_imports"].get("ok"))
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ok"] else 1)
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        default=str(DEFAULT_RUNTIME_DIR),
        help="Runtime directory to validate. Looks for python.exe or Scripts/python.exe.",
    )
    parser.add_argument(
        "--backend-root",
        default=".",
        help="Repository or packaged backend root to put on PYTHONPATH during validation.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if torch imports but CUDA is unavailable.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to allow the runtime import probe to run.",
    )
    return parser.parse_args()


def python_candidates(runtime_dir):
    root = Path(runtime_dir)
    return [
        root / "python.exe",
        root / "Scripts" / "python.exe",
        root / "bin" / "python",
    ]


def find_runtime_python(runtime_dir):
    for candidate in python_candidates(runtime_dir):
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in python_candidates(runtime_dir))
    raise FileNotFoundError(f"No Python executable found in runtime. Searched: {searched}")


def run_probe(python, backend_root, timeout):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(backend_root).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    probe = "REQUIRED_MODULES = " + repr(REQUIRED_MODULES) + "\n" + PROBE_CODE
    return subprocess.run(
        [str(python), "-c", probe],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=env,
        timeout=timeout,
    )


def parse_probe_output(stdout):
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Runtime probe did not return JSON: {stdout}") from error


def main():
    args = parse_args()
    python = find_runtime_python(args.runtime_dir)
    completed = run_probe(python, args.backend_root, args.timeout)

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)

    if completed.returncode != 0:
        return completed.returncode

    payload = parse_probe_output(completed.stdout)
    if args.require_cuda and not payload.get("torch", {}).get("cuda_available"):
        print("Runtime validation failed: CUDA is required but unavailable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
