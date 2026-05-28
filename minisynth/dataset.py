"""Dataset file generation helpers."""

from pathlib import Path

import soundfile as sf

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.io import save_patch
from minisynth.randomize import audio_avoids_clipping, random_patch

DEFAULT_PARAM_DIR = Path("data/generated/v1/params")
DEFAULT_AUDIO_DIR = Path("data/generated/v1/audio")


def patch_filename(index, seed):
    return f"patch_{index:06d}_seed_{seed}.json"


def audio_filename(index, seed):
    return f"patch_{index:06d}_seed_{seed}.wav"


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


def write_random_dataset_files(
    param_dir=DEFAULT_PARAM_DIR,
    audio_dir=DEFAULT_AUDIO_DIR,
    seed=0,
    count=1,
):
    if count < 1:
        raise ValueError("count must be at least 1")

    param_destination = Path(param_dir)
    audio_destination = Path(audio_dir)
    audio_destination.mkdir(parents=True, exist_ok=True)
    records = []

    for index in range(count):
        patch_seed = seed + index
        patch = random_patch(patch_seed)
        audio = render_patch(**patch)

        if not audio_avoids_clipping(audio):
            raise ValueError(f"Generated audio failed clipping constraint: seed {patch_seed}")

        patch_path = param_destination / patch_filename(index, patch_seed)
        audio_path = audio_destination / audio_filename(index, patch_seed)

        save_patch(patch, patch_path)
        sf.write(audio_path, audio, DEFAULT_SAMPLE_RATE)
        records.append(
            {
                "seed": patch_seed,
                "patch_path": patch_path,
                "audio_path": audio_path,
            }
        )

    return records
