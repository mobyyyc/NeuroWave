"""Envelope generators, including ADSR."""

import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE


def adsr(
    length,
    attack=0.01,
    decay=0.2,
    sustain=0.7,
    release=0.3,
    sample_rate=DEFAULT_SAMPLE_RATE,
):
    n = int(length * sample_rate)
    env = np.ones(n) * sustain

    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)

    if a > 0:
        env[:a] = np.linspace(0, 1, a)

    if d > 0:
        env[a:a + d] = np.linspace(1, sustain, d)

    if r > 0:
        env[-r:] = np.linspace(sustain, 0, r)

    return env
