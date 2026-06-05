"""Reusable inference workflow for the future NeuroWave desktop app."""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.features import mel_spectrogram, to_mono
from minisynth.io import save_patch
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import (
    DEFAULT_MEL_TENSOR_FRAMES,
    load_torch_checkpoint,
    predict_patch_from_audio,
)


DEFAULT_APP_OUTPUT_ROOT = Path("runs/app")
TARGET_CROP_FILENAME = "target_crop.wav"
PREDICTED_PATCH_FILENAME = "predicted_patch.json"
PREDICTED_WAV_FILENAME = "predicted.wav"
TARGET_SPECTROGRAM_FILENAME = "target_spectrogram.json"
PREDICTED_SPECTROGRAM_FILENAME = "predicted_spectrogram.json"
SUMMARY_FILENAME = "summary.json"
UI_CROP_END_TOLERANCE_SECONDS = 0.005


@dataclass(frozen=True)
class AppInferenceRequest:
    audio_path: str
    model_path: str
    freq_hz: float
    crop_start_seconds: float = 0.0
    crop_end_seconds: float | None = None
    output_dir: str = str(DEFAULT_APP_OUTPUT_ROOT)
    run_id: str | None = None
    device: str | None = None
    frames: int = DEFAULT_MEL_TENSOR_FRAMES


@dataclass(frozen=True)
class AppInferenceResult:
    run_id: str
    run_dir: str
    input_audio: str
    input_sample_rate: int
    crop_start_seconds: float
    crop_end_seconds: float
    crop_frames: int
    freq_context_hz: float
    model: str
    target_crop_wav: str
    predicted_patch_json: str
    predicted_wav: str
    target_spectrogram: str
    predicted_spectrogram: str
    summary: str
    model_metrics: dict
    warnings: list[str]


def sanitize_run_component(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or "audio"


def default_run_id(audio_path, now=None):
    if now is None:
        now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    stem = sanitize_run_component(Path(audio_path).stem)
    return f"{stamp}_{stem}"


def app_output_paths(output_dir, run_id):
    run_dir = Path(output_dir) / sanitize_run_component(run_id)
    return {
        "run_dir": run_dir,
        "target_crop_wav": run_dir / TARGET_CROP_FILENAME,
        "predicted_patch_json": run_dir / PREDICTED_PATCH_FILENAME,
        "predicted_wav": run_dir / PREDICTED_WAV_FILENAME,
        "target_spectrogram": run_dir / TARGET_SPECTROGRAM_FILENAME,
        "predicted_spectrogram": run_dir / PREDICTED_SPECTROGRAM_FILENAME,
        "summary": run_dir / SUMMARY_FILENAME,
    }


def load_mono_audio(path):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input audio not found: {source}")
    audio, sample_rate = sf.read(source)
    samples = np.asarray(to_mono(audio), dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono audio after conversion, got shape {samples.shape}")
    if len(samples) == 0:
        raise ValueError("Input audio is empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Input audio contains non-finite values")
    return samples, int(sample_rate)


def validate_frequency(freq_hz):
    freq = float(freq_hz)
    if not np.isfinite(freq):
        raise ValueError("freq_hz must be finite")
    if freq <= 0.0:
        raise ValueError("freq_hz must be positive")
    return freq


def crop_frame_range(sample_count, sample_rate, start_seconds=0.0, end_seconds=None):
    if sample_count < 1:
        raise ValueError("audio must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    duration = sample_count / sample_rate
    start = float(start_seconds)
    end = duration if end_seconds is None else float(end_seconds)

    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("crop start and end must be finite")
    if start < 0.0:
        raise ValueError("crop start must be non-negative")
    if end <= start:
        raise ValueError("crop end must be greater than crop start")
    if end > duration and end <= duration + UI_CROP_END_TOLERANCE_SECONDS:
        end = duration
    if end > duration:
        raise ValueError("crop end exceeds audio duration")

    start_frame = int(round(start * sample_rate))
    end_frame = int(round(end * sample_rate))
    start_frame = min(max(start_frame, 0), sample_count)
    end_frame = min(max(end_frame, 0), sample_count)
    if end_frame <= start_frame:
        raise ValueError("crop contains no audio frames")
    return start_frame, end_frame, start, end


def crop_audio(audio, sample_rate, start_seconds=0.0, end_seconds=None):
    samples = np.asarray(audio, dtype=np.float32)
    start_frame, end_frame, start, end = crop_frame_range(
        len(samples),
        int(sample_rate),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    return samples[start_frame:end_frame], start, end


def render_prediction_audio(patch):
    audio = render_patch(**patch)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono rendered audio, got shape {audio.shape}")
    if len(audio) == 0:
        raise ValueError("Rendered audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")
    return audio


def write_wav(path, audio, sample_rate):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.asarray(audio, dtype=np.float32), int(sample_rate))
    return destination


def write_spectrogram_json(path, audio, sample_rate):
    spectrogram = mel_spectrogram(audio, sample_rate=sample_rate)
    payload = {
        "kind": "mel_spectrogram_db",
        "sample_rate": int(sample_rate),
        "n_mels": int(spectrogram.shape[0]),
        "frames": int(spectrogram.shape[1]),
        "min_db": float(np.min(spectrogram)),
        "max_db": float(np.max(spectrogram)),
        "values": spectrogram.astype(float).tolist(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file)
        file.write("\n")
    return destination


def write_summary(path, result):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
        file.write("\n")
    return destination


def run_app_inference(request):
    if not isinstance(request, AppInferenceRequest):
        request = AppInferenceRequest(**request)

    freq_hz = validate_frequency(request.freq_hz)
    input_audio_path = Path(request.audio_path)
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    run_id = sanitize_run_component(request.run_id or default_run_id(input_audio_path))
    paths = app_output_paths(request.output_dir, run_id)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)

    source_audio, source_sample_rate = load_mono_audio(input_audio_path)
    cropped_audio, crop_start, crop_end = crop_audio(
        source_audio,
        source_sample_rate,
        start_seconds=request.crop_start_seconds,
        end_seconds=request.crop_end_seconds,
    )
    write_wav(paths["target_crop_wav"], cropped_audio, source_sample_rate)

    checkpoint = load_torch_checkpoint(model_path, device=request.device)
    predicted_patch = predict_patch_from_audio(
        checkpoint["model"],
        cropped_audio,
        source_sample_rate,
        device=request.device,
        frames=request.frames,
        freq=freq_hz,
    )
    predicted_audio = render_prediction_audio(predicted_patch)

    save_patch(predicted_patch, paths["predicted_patch_json"])
    write_wav(paths["predicted_wav"], predicted_audio, DEFAULT_SAMPLE_RATE)
    write_spectrogram_json(paths["target_spectrogram"], cropped_audio, source_sample_rate)
    write_spectrogram_json(paths["predicted_spectrogram"], predicted_audio, DEFAULT_SAMPLE_RATE)

    result = AppInferenceResult(
        run_id=run_id,
        run_dir=str(paths["run_dir"]),
        input_audio=str(input_audio_path),
        input_sample_rate=source_sample_rate,
        crop_start_seconds=float(crop_start),
        crop_end_seconds=float(crop_end),
        crop_frames=int(len(cropped_audio)),
        freq_context_hz=float(freq_hz),
        model=str(model_path),
        target_crop_wav=str(paths["target_crop_wav"]),
        predicted_patch_json=str(paths["predicted_patch_json"]),
        predicted_wav=str(paths["predicted_wav"]),
        target_spectrogram=str(paths["target_spectrogram"]),
        predicted_spectrogram=str(paths["predicted_spectrogram"]),
        summary=str(paths["summary"]),
        model_metrics=compact_model_metrics(checkpoint.get("metrics", {})),
        warnings=[],
    )
    write_summary(paths["summary"], result)
    return result
