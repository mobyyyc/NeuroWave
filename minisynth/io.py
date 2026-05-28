"""Patch, audio, and dataset input/output helpers."""

import json
from pathlib import Path


def load_patch(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_patch(patch, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(patch, file, indent=2)
        file.write("\n")
