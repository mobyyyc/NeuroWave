"""Audio rendering entry points for parameterized NeuroWave patches."""

import numpy as np

from minisynth.envelopes import adsr
from minisynth.filters import lowpass_filter
from minisynth.oscillators import oscillator


def render_patch(
    freq=261.63,
    length=1.5,
    osc1_wave="saw",
    osc1_level=0.8,
    osc2_wave="saw",
    osc2_level=0.4,
    osc2_detune=7,
    cutoff=1200,
    resonance=0.2,
    attack=0.01,
    decay=0.2,
    sustain=0.7,
    release=0.3,
):
    osc1 = oscillator(osc1_wave, freq, length) * osc1_level

    detune_ratio = 2 ** (osc2_detune / 1200)
    osc2_freq = freq * detune_ratio
    osc2 = oscillator(osc2_wave, osc2_freq, length) * osc2_level

    audio = osc1 + osc2

    audio = lowpass_filter(audio, cutoff=cutoff, resonance=resonance)

    env = adsr(length, attack, decay, sustain, release)
    audio = audio * env
    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")

    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio

    audio = audio / peak * 0.8

    return audio
