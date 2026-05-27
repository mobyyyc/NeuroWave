import soundfile as sf
import matplotlib.pyplot as plt
from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from patches import PATCHES

SR = DEFAULT_SAMPLE_RATE


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


if __name__ == "__main__":
    patch = PATCHES["dark_saw"]

    audio = render_patch(**patch)

    sf.write("dark_saw.wav", audio, SR)

    show_spectrogram(audio)

    print("Saved dark_saw.wav")
