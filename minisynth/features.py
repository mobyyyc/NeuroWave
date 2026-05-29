"""Audio feature extraction for matching, datasets, and ML."""

import numpy as np
from scipy.signal import resample_poly

from minisynth.constants import DEFAULT_SAMPLE_RATE

DEFAULT_TARGET_RMS = 0.1
SILENCE_RMS_THRESHOLD = 1e-12


def to_mono(audio):
    samples = np.asarray(audio)

    if samples.ndim == 1:
        return samples

    if samples.ndim == 2:
        if samples.shape[1] == 0:
            raise ValueError("audio must include at least one channel")

        return np.mean(samples, axis=1)

    raise ValueError(f"expected mono or channel-last audio, got shape {samples.shape}")


def resample_audio(audio, source_sample_rate, target_sample_rate=DEFAULT_SAMPLE_RATE):
    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be positive")

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    samples = np.asarray(audio)

    if source_sample_rate == target_sample_rate:
        return samples

    gcd = np.gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = source_sample_rate // gcd

    return resample_poly(samples, up, down, axis=0)


def rms(audio):
    samples = np.asarray(audio)
    if samples.size == 0:
        raise ValueError("audio must not be empty")

    return float(np.sqrt(np.mean(np.square(samples))))


def normalize_loudness(
    audio,
    target_rms=DEFAULT_TARGET_RMS,
    silence_threshold=SILENCE_RMS_THRESHOLD,
):
    if target_rms <= 0:
        raise ValueError("target_rms must be positive")

    if silence_threshold < 0:
        raise ValueError("silence_threshold must be non-negative")

    samples = np.asarray(audio)
    current_rms = rms(samples)

    if current_rms <= silence_threshold:
        return np.zeros_like(samples, dtype=float)

    return samples * (target_rms / current_rms)
