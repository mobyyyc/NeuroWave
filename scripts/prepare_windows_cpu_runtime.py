"""Prepare a reproducible CPU-only Windows runtime for the public installer."""

import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_RUNTIME_DIR = Path("runtime/python-cpu")
DEFAULT_SITE_PACKAGES = (
    Path(".venv/Lib/site-packages")
    if Path(".venv/Lib/site-packages").exists()
    else Path(sys.prefix) / "Lib" / "site-packages"
)
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch==2.11.0+cpu", "torchvision==0.26.0+cpu", "torchaudio==2.11.0+cpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--source-python", default=str(Path(sys.base_prefix)))
    parser.add_argument("--site-packages", default=str(DEFAULT_SITE_PACKAGES))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the CPU staging runtime before preparing it.",
    )
    return parser.parse_args()


def run(command):
    print("+", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    runtime_dir = Path(args.runtime_dir)
    prepare = [
        sys.executable,
        "scripts/prepare_windows_runtime.py",
        "--runtime-dir",
        str(runtime_dir),
        "--source-python",
        args.source_python,
        "--site-packages",
        args.site_packages,
    ]
    if args.replace:
        prepare.append("--replace")
    run(prepare)

    python = runtime_dir / "python.exe"
    if not python.exists():
        raise FileNotFoundError(f"Prepared runtime is missing {python}")

    run([python, "-m", "pip", "uninstall", "--yes", "torch", "torchvision", "torchaudio"])
    run([
        python,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "--index-url",
        CPU_INDEX_URL,
        *TORCH_PACKAGES,
    ])
    print(f"Prepared CPU-only Windows runtime: {runtime_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
