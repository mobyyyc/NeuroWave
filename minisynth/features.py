"""Audio feature extraction for matching, datasets, and ML."""

import numpy as np
from scipy.signal import resample_poly

from minisynth.constants import DEFAULT_SAMPLE_RATE


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
