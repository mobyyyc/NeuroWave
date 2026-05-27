"""Oscillator and waveform generation utilities."""

import numpy as np
from scipy.signal import sawtooth, square

from minisynth.constants import DEFAULT_SAMPLE_RATE


BASE_WAVES = ("sine", "triangle", "saw", "square", "noise")


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
    elif wave == "noise":
        return np.random.default_rng(0).uniform(-1.0, 1.0, len(t))
    else:
        raise ValueError("Unknown wave type")


def normalize_wave_mix(weights):
    normalized = {wave: float(weights.get(wave, 0.0)) for wave in BASE_WAVES}

    for wave, weight in normalized.items():
        if weight < 0:
            raise ValueError(f"Wave mix weight must be non-negative: {wave}={weight}")

    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Wave mix must include at least one positive weight")

    return {wave: weight / total for wave, weight in normalized.items()}


def wave_mix(weights, freq, length, sample_rate=DEFAULT_SAMPLE_RATE):
    normalized = normalize_wave_mix(weights)
    output = np.zeros(int(sample_rate * length))

    for wave, weight in normalized.items():
        if weight == 0:
            continue
        output += oscillator(wave, freq, length, sample_rate=sample_rate) * weight

    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak

    return output
