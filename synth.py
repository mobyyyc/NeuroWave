import numpy as np
import soundfile as sf
from scipy.signal import sawtooth, square

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
        env[a:a+d] = np.linspace(1, sustain, d)

    if r > 0:
        env[-r:] = np.linspace(sustain, 0, r)

    return env

def render_patch(
    freq=261.63,
    length=1.5,
    wave="sine",
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

    env = adsr(length, attack, decay, sustain, release)
    audio = osc * env * level

    audio = audio / np.max(np.abs(audio)) * 0.8
    return audio

if __name__ == "__main__":
    audio = render_patch(wave="sine")
    sf.write("test_sine.wav", audio, SR)
    print("Saved test_sine.wav")