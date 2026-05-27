"""Filter implementations used by the synth engine."""

import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE


def lowpass_filter(audio, cutoff=2000, resonance=0.2, sample_rate=DEFAULT_SAMPLE_RATE):
    cutoff = np.clip(cutoff, 20, sample_rate / 2 - 100)
    resonance = np.clip(resonance, 0.0, 0.99)

    f = 2 * np.sin(np.pi * cutoff / sample_rate)
    q = 1.0 - resonance

    low = 0.0
    band = 0.0
    output = np.zeros_like(audio)

    for i, sample in enumerate(audio):
        low += f * band
        high = sample - low - q * band
        band += f * high
        output[i] = low

    return output
