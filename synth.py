"""Compatibility wrapper for the historical synth.py entry point."""

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from scripts.render_patch import render_preset

SR = DEFAULT_SAMPLE_RATE


def show_spectrogram(audio):
    import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    render_preset("presets/dark_saw.json", "dark_saw.wav")
