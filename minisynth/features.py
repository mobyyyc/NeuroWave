"""Audio feature extraction for matching, datasets, and ML."""

import numpy as np


def to_mono(audio):
    samples = np.asarray(audio)

    if samples.ndim == 1:
        return samples

    if samples.ndim == 2:
        if samples.shape[1] == 0:
            raise ValueError("audio must include at least one channel")

        return np.mean(samples, axis=1)

    raise ValueError(f"expected mono or channel-last audio, got shape {samples.shape}")
