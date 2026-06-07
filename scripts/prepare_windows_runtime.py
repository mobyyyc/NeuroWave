"""Prepare the ignored Python runtime folder for Windows desktop packaging."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


DEFAULT_RUNTIME_DIR = Path("runtime/python")
DEFAULT_SOURCE_PYTHON = Path(sys.base_prefix)
DEFAULT_SITE_PACKAGES = Path(sys.prefix) / "Lib" / "site-packages"
EXCLUDED_ROOT_NAMES = {
    ".gitkeep",
    ".gitignore",
    "pyvenv.cfg",
}
STALE_RUNTIME_MARKERS = {
    "pyvenv.cfg",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        default=str(DEFAULT_RUNTIME_DIR),
        help="Destination runtime directory to prepare.",
    )
    parser.add_argument(
        "--source-python",
        default=str(DEFAULT_SOURCE_PYTHON),
        help="Source Python install root to copy.",
    )
    parser.add_argument(
        "--site-packages",
        default=str(DEFAULT_SITE_PACKAGES),
        help="Working site-packages directory to copy into the runtime.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing runtime directory before preparing it. Use only when no packaged runtime files are locked.",
    )
    return parser.parse_args()


def files_match(source, target):
    if not target.exists() or not target.is_file():
        return False
    source_stat = source.stat()
    target_stat = target.stat()
    return source_stat.st_size == target_stat.st_size


def copytree_contents(source, destination, exclude_root_names=None):
    exclude_root_names = exclude_root_names or set()
    destination.mkdir(parents=True, exist_ok=True)
    for item in Path(source).iterdir():
        if item.name in exclude_root_names:
            continue
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                copytree_contents(item, target)
            else:
                shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            if files_match(item, target):
                continue
            shutil.copy2(item, target)


def reset_permissions(_function, path, _exc_info):
    os.chmod(path, 0o700)
    if Path(path).is_dir():
        shutil.rmtree(path, onerror=reset_permissions)
    else:
        Path(path).unlink()


def write_manifest(runtime_dir, source_python, site_packages):
    payload = {
        "source_python": str(Path(source_python).resolve()),
        "site_packages": str(Path(site_packages).resolve()),
        "python": sys.version,
        "prepared_by": "scripts/prepare_windows_runtime.py",
    }
    (runtime_dir / "__neurowave_runtime__.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def remove_stale_venv_markers(runtime_dir):
    for name in STALE_RUNTIME_MARKERS:
        target = runtime_dir / name
        if not target.exists():
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target, onerror=reset_permissions)
            else:
                target.unlink()
        except OSError as error:
            print(f"Warning: could not remove stale runtime marker {target}: {error}", file=sys.stderr)


def main():
    args = parse_args()
    runtime_dir = Path(args.runtime_dir)
    source_python = Path(args.source_python)
    site_packages = Path(args.site_packages)

    if not source_python.exists():
        raise FileNotFoundError(f"Source Python root not found: {source_python}")
    if not site_packages.exists():
        raise FileNotFoundError(f"site-packages not found: {site_packages}")

    if runtime_dir.exists() and args.replace:
        shutil.rmtree(runtime_dir, onerror=reset_permissions)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / ".gitkeep").write_text("\n", encoding="utf-8")
    remove_stale_venv_markers(runtime_dir)

    copytree_contents(source_python, runtime_dir, exclude_root_names=EXCLUDED_ROOT_NAMES)
    copytree_contents(site_packages, runtime_dir / "Lib" / "site-packages")
    write_manifest(runtime_dir, source_python, site_packages)

    python_exe = runtime_dir / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"Prepared runtime is missing {python_exe}")

    print(f"Prepared Windows runtime: {runtime_dir.resolve()}")
    print(f"Python executable: {python_exe.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
