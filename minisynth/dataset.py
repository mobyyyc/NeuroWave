"""Dataset file generation helpers."""

from pathlib import Path

from minisynth.io import save_patch
from minisynth.randomize import random_patch

DEFAULT_PARAM_DIR = Path("data/generated/v1/params")


def patch_filename(index, seed):
    return f"patch_{index:06d}_seed_{seed}.json"


def write_random_patch_files(output_dir=DEFAULT_PARAM_DIR, seed=0, count=1):
    if count < 1:
        raise ValueError("count must be at least 1")

    destination = Path(output_dir)
    paths = []

    for index in range(count):
        patch_seed = seed + index
        path = destination / patch_filename(index, patch_seed)
        save_patch(random_patch(patch_seed), path)
        paths.append(path)

    return paths
