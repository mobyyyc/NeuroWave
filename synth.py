import numpy as np
import soundfile as sf
from scipy.signal import sawtooth, square, butter, lfilter
from patches import PATCHES

SR = 44100


def adsr(length, attack=0.01, decay=0.2, sustain=0.7, release=0.3):
    n = int(length * SR)
    env = np.ones(n) * sustain

    a = int(attack * SR)
    d = int(decay * SR)
    r = int(release * SR)

    if a > 0:
        env[:a] = np.linspace(0, 1, a)

    if d > 0:
        env[a:a + d] = np.linspace(1, sustain, d)

    if r > 0:
        env[-r:] = np.linspace(sustain, 0, r)

    return env


def lowpass_filter(audio, cutoff=2000):
    nyquist = 0.5 * SR
    normal_cutoff = cutoff / nyquist

    b, a = butter(4, normal_cutoff, btype="low")
    return lfilter(b, a, audio)


def render_patch(
    freq=261.63,
    length=1.5,
    wave="sine",
    cutoff=1200,
    level=0.8,
    attack=0.01,
    decay=0.2,
    sustain=0.7,
    release=0.3
):
    t = np.linspace(0, length, int(SR * length), endpoint=False)

    if wave == "sine":
        osc = np.sin(2 * np.pi * freq * t)
    elif wave == "saw":
        osc = sawtooth(2 * np.pi * freq * t)
    elif wave == "square":
        osc = square(2 * np.pi * freq * t)
    else:
        raise ValueError("Unknown wave type")

    osc = lowpass_filter(osc, cutoff=cutoff)

    env = adsr(length, attack, decay, sustain, release)
    audio = osc * env * level

    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio


if __name__ == "__main__":
    patch = PATCHES["dark_saw"]

    audio = render_patch(**patch)

    sf.write("dark_saw.wav", audio, SR)

    print("Saved dark_saw.wav")