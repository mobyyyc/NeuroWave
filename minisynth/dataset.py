"""Dataset file generation helpers."""

import json
from pathlib import Path

import soundfile as sf
import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.features import mel_spectrogram, rms, rms_envelope, spectral_centroid
from minisynth.io import load_patch, save_patch
from minisynth.randomize import audio_avoids_clipping, random_patch
from minisynth.schema import SynthConfig

DEFAULT_PARAM_DIR = Path("data/generated/d1/params")
DEFAULT_AUDIO_DIR = Path("data/generated/d1/audio")
DEFAULT_METADATA_PATH = Path("data/generated/d1/metadata.jsonl")
DEFAULT_DATASET_VERSION = "d1"
DEFAULT_MEL_TENSOR_PATH = Path("data/generated/d1/features/mel_tensors.npz")
DEFAULT_MEL_TENSOR_FRAMES = 256


def generated_dataset_paths(version=DEFAULT_DATASET_VERSION, root=Path("data/generated")):
    dataset_root = Path(root) / version
    return {
        "root": dataset_root,
        "param_dir": dataset_root / "params",
        "audio_dir": dataset_root / "audio",
        "metadata_path": dataset_root / "metadata.jsonl",
    }


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
    metadata_path=DEFAULT_METADATA_PATH,
    seed=0,
    count=1,
):
    if count < 1:
        raise ValueError("count must be at least 1")

    param_destination = Path(param_dir)
    audio_destination = Path(audio_dir)
    audio_destination.mkdir(parents=True, exist_ok=True)
    metadata_destination = Path(metadata_path)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
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
                "index": index,
                "seed": patch_seed,
                "patch_path": patch_path,
                "audio_path": audio_path,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "frames": len(audio),
            }
        )

    write_metadata(records, metadata_destination)

    return records


def write_metadata(records, path=DEFAULT_METADATA_PATH):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(metadata_record(record)) + "\n")


def metadata_record(record):
    return {
        "index": record["index"],
        "seed": record["seed"],
        "patch_path": str(record["patch_path"]),
        "audio_path": str(record["audio_path"]),
        "sample_rate": record["sample_rate"],
        "frames": record["frames"],
    }


def load_metadata(path=DEFAULT_METADATA_PATH):
    source = Path(path)
    rows = []

    with source.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    return rows


def resolve_metadata_path(metadata_path, item_path):
    path = Path(item_path)
    if path.is_absolute():
        return path

    if path.exists():
        return path

    return Path(metadata_path).parent / path


def audio_feature_vector(audio, sample_rate):
    mel = mel_spectrogram(audio, sample_rate=sample_rate)
    envelope = rms_envelope(audio)
    centroid = spectral_centroid(audio, sample_rate=sample_rate)

    return np.array(
        [
            float(np.mean(mel)),
            float(np.std(mel)),
            rms(audio),
            float(np.mean(envelope)),
            float(np.std(envelope)),
            float(np.mean(centroid)),
            float(np.std(centroid)),
        ],
        dtype=float,
    )


def fixed_frame_array(array, frames=DEFAULT_MEL_TENSOR_FRAMES):
    if frames < 1:
        raise ValueError("frames must be at least 1")

    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("array must be 2D")

    if values.shape[1] == frames:
        return values

    if values.shape[1] > frames:
        return values[:, :frames]

    padding = ((0, 0), (0, frames - values.shape[1]))
    return np.pad(values, padding, mode="constant")


def mel_tensor_from_audio(audio, sample_rate, frames=DEFAULT_MEL_TENSOR_FRAMES):
    mel = mel_spectrogram(audio, sample_rate=sample_rate)
    fixed = fixed_frame_array(mel, frames=frames)
    return fixed[np.newaxis, :, :].astype(np.float32)


def load_training_dataset(metadata_path=DEFAULT_METADATA_PATH):
    rows = load_metadata(metadata_path)
    features = []
    targets = []

    for row in rows:
        audio_path = resolve_metadata_path(metadata_path, row["audio_path"])
        patch_path = resolve_metadata_path(metadata_path, row["patch_path"])

        audio, sample_rate = sf.read(audio_path)
        patch = load_patch(patch_path)

        features.append(audio_feature_vector(audio, sample_rate))
        targets.append(SynthConfig(**patch).to_vector())

    if not features:
        raise ValueError("metadata did not contain any training rows")

    return np.vstack(features), np.asarray(targets, dtype=float)


def load_mel_tensor_dataset(metadata_path=DEFAULT_METADATA_PATH, frames=DEFAULT_MEL_TENSOR_FRAMES):
    rows = load_metadata(metadata_path)
    features = []
    targets = []
    indices = []
    seeds = []

    for row in rows:
        audio_path = resolve_metadata_path(metadata_path, row["audio_path"])
        patch_path = resolve_metadata_path(metadata_path, row["patch_path"])

        audio, sample_rate = sf.read(audio_path)
        patch = load_patch(patch_path)

        features.append(mel_tensor_from_audio(audio, sample_rate, frames=frames))
        targets.append(SynthConfig(**patch).to_vector())
        indices.append(row["index"])
        seeds.append(row["seed"])

    if not features:
        raise ValueError("metadata did not contain any training rows")

    return {
        "features": np.stack(features).astype(np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }


def save_mel_tensor_dataset(
    metadata_path=DEFAULT_METADATA_PATH,
    output_path=DEFAULT_MEL_TENSOR_PATH,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
):
    dataset = load_mel_tensor_dataset(metadata_path, frames=frames)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=dataset["features"],
        targets=dataset["targets"],
        indices=dataset["indices"],
        seeds=dataset["seeds"],
        metadata_path=str(metadata_path),
        frames=np.asarray(frames, dtype=np.int64),
    )
    return destination
