"""Oscillator and waveform generation utilities."""

import numpy as np
from scipy.signal import sawtooth, square

from minisynth.constants import DEFAULT_SAMPLE_RATE


def oscillator(wave, freq, length, sample_rate=DEFAULT_SAMPLE_RATE):
    t = np.linspace(0, length, int(sample_rate * length), endpoint=False)

    if wave == "sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave == "triangle":
        return sawtooth(2 * np.pi * freq * t, width=0.5)
    elif wave == "saw":
        return sawtooth(2 * np.pi * freq * t)
    elif wave == "square":
        return square(2 * np.pi * freq * t)
    else:
        raise ValueError("Unknown wave type")
