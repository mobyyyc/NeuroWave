"""Dataset file generation helpers."""

import concurrent.futures
import json
import os
from pathlib import Path

# Limit BLAS worker threads before importing numerical libraries. This avoids
# multiplying CPU usage when dataset work is split across processes.
CPU_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for env_var in CPU_THREAD_LIMIT_ENV_VARS:
    os.environ.setdefault(env_var, "1")

import numpy as np
import soundfile as sf

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
DEFAULT_WORKER_FRACTION = 0.75


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


def default_worker_count(cpu_count=None, fraction=DEFAULT_WORKER_FRACTION):
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0, 1]")

    return max(1, int(cpu_count * fraction))


def resolve_worker_count(workers):
    if workers is None:
        return 1
    if workers < 0:
        raise ValueError("workers must be 0 or greater")
    if workers == 0:
        return default_worker_count()
    return workers


def print_progress(prefix, completed, total, workers):
    print(
        f"\r{prefix} {completed}/{total} (workers: {workers})",
        end="",
        flush=True,
    )


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


def generate_random_dataset_record(index, seed, param_dir, audio_dir):
    patch_seed = seed + index
    patch = random_patch(patch_seed)
    audio = render_patch(**patch)

    if not audio_avoids_clipping(audio):
        raise ValueError(f"Generated audio failed clipping constraint: seed {patch_seed}")

    patch_path = Path(param_dir) / patch_filename(index, patch_seed)
    audio_path = Path(audio_dir) / audio_filename(index, patch_seed)

    save_patch(patch, patch_path)
    sf.write(audio_path, audio, DEFAULT_SAMPLE_RATE)
    return {
        "index": index,
        "seed": patch_seed,
        "patch_path": patch_path,
        "audio_path": audio_path,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "frames": len(audio),
    }


def _generate_random_dataset_record_task(args):
    return generate_random_dataset_record(*args)


def write_random_dataset_files(
    param_dir=DEFAULT_PARAM_DIR,
    audio_dir=DEFAULT_AUDIO_DIR,
    metadata_path=DEFAULT_METADATA_PATH,
    seed=0,
    count=1,
    workers=1,
    progress=False,
):
    if count < 1:
        raise ValueError("count must be at least 1")

    param_destination = Path(param_dir)
    audio_destination = Path(audio_dir)
    param_destination.mkdir(parents=True, exist_ok=True)
    audio_destination.mkdir(parents=True, exist_ok=True)
    metadata_destination = Path(metadata_path)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_worker_count(workers)

    records = generate_random_dataset_records(
        param_destination,
        audio_destination,
        seed,
        count,
        workers=worker_count,
        progress=progress,
    )

    write_metadata(records, metadata_destination)

    return records


def generate_random_dataset_records(param_dir, audio_dir, seed, count, workers=1, progress=False):
    if workers == 1:
        records = []
        for index in range(count):
            records.append(generate_random_dataset_record(index, seed, param_dir, audio_dir))
            if progress:
                print_progress("Generating audio", len(records), count, workers)
        if progress:
            print()
        return records

    tasks = [(index, seed, str(param_dir), str(audio_dir)) for index in range(count)]
    records = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_random_dataset_record_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                if progress:
                    print_progress("Generating audio", len(records), count, workers)
    except (OSError, PermissionError) as error:
        if progress:
            print(f"\nMultiprocessing unavailable ({error}); falling back to serial generation.")
        return generate_random_dataset_records(
            param_dir,
            audio_dir,
            seed,
            count,
            workers=1,
            progress=progress,
        )

    records.sort(key=lambda record: record["index"])
    if progress:
        print()
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


def mel_tensor_dataset_record(row, metadata_path, frames=DEFAULT_MEL_TENSOR_FRAMES):
    audio_path = resolve_metadata_path(metadata_path, row["audio_path"])
    patch_path = resolve_metadata_path(metadata_path, row["patch_path"])

    audio, sample_rate = sf.read(audio_path)
    patch = load_patch(patch_path)

    return {
        "feature": mel_tensor_from_audio(audio, sample_rate, frames=frames),
        "target": SynthConfig(**patch).to_vector(),
        "index": row["index"],
        "seed": row["seed"],
    }


def _mel_tensor_dataset_record_task(args):
    return mel_tensor_dataset_record(*args)


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


def load_mel_tensor_dataset(
    metadata_path=DEFAULT_METADATA_PATH,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
    workers=1,
    progress=False,
):
    rows = load_metadata(metadata_path)
    if not rows:
        raise ValueError("metadata did not contain any training rows")
    worker_count = resolve_worker_count(workers)
    records = load_mel_tensor_records(
        rows,
        metadata_path,
        frames=frames,
        workers=worker_count,
        progress=progress,
    )

    return {
        "features": np.stack([record["feature"] for record in records]).astype(np.float32),
        "targets": np.asarray([record["target"] for record in records], dtype=np.float32),
        "indices": np.asarray([record["index"] for record in records], dtype=np.int64),
        "seeds": np.asarray([record["seed"] for record in records], dtype=np.int64),
    }


def load_mel_tensor_records(rows, metadata_path, frames=DEFAULT_MEL_TENSOR_FRAMES, workers=1, progress=False):
    if workers == 1:
        records = []
        for row in rows:
            records.append(mel_tensor_dataset_record(row, metadata_path, frames=frames))
            if progress:
                print_progress("Exporting tensors", len(records), len(rows), workers)
        if progress:
            print()
        return records

    tasks = [(row, str(metadata_path), frames) for row in rows]
    records = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_mel_tensor_dataset_record_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                if progress:
                    print_progress("Exporting tensors", len(records), len(rows), workers)
    except (OSError, PermissionError) as error:
        if progress:
            print(f"\nMultiprocessing unavailable ({error}); falling back to serial export.")
        return load_mel_tensor_records(
            rows,
            metadata_path,
            frames=frames,
            workers=1,
            progress=progress,
        )

    records.sort(key=lambda record: record["index"])
    if progress:
        print()
    return records


def save_mel_tensor_dataset(
    metadata_path=DEFAULT_METADATA_PATH,
    output_path=DEFAULT_MEL_TENSOR_PATH,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
    workers=1,
    progress=False,
):
    dataset = load_mel_tensor_dataset(
        metadata_path,
        frames=frames,
        workers=workers,
        progress=progress,
    )
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
