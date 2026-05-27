import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import sawtooth, square
from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.envelopes import adsr
from patches import PATCHES

SR = DEFAULT_SAMPLE_RATE


def oscillator(wave, freq, length):
    t = np.linspace(0, length, int(SR * length), endpoint=False)

    if wave == "sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave == "saw":
        return sawtooth(2 * np.pi * freq * t)
    elif wave == "square":
        return square(2 * np.pi * freq * t)
    else:
        raise ValueError("Unknown wave type")


def lowpass_filter(audio, cutoff=2000, resonance=0.2):
    cutoff = np.clip(cutoff, 20, SR / 2 - 100)
    resonance = np.clip(resonance, 0.0, 0.99)

    f = 2 * np.sin(np.pi * cutoff / SR)
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


def show_spectrogram(audio):
    import librosa
    import librosa.display

    D = librosa.stft(audio, n_fft=4096, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S_db,
        sr=SR,
        hop_length=512,
        x_axis="time",
        y_axis="log"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Frequency Spectrogram")
    plt.show()


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
    release=0.3
):
    osc1 = oscillator(osc1_wave, freq, length) * osc1_level

    detune_ratio = 2 ** (osc2_detune / 1200)
    osc2_freq = freq * detune_ratio
    osc2 = oscillator(osc2_wave, osc2_freq, length) * osc2_level

    audio = osc1 + osc2

    audio = lowpass_filter(audio, cutoff=cutoff, resonance=resonance)

    env = adsr(length, attack, decay, sustain, release)
    audio = audio * env

    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio


if __name__ == "__main__":
    patch = PATCHES["dark_saw"]

    audio = render_patch(**patch)

    sf.write("dark_saw.wav", audio, SR)

    show_spectrogram(audio)

    print("Saved dark_saw.wav")
