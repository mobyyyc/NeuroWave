"""Audio comparison helpers built on NeuroWave feature extraction."""

import numpy as np

from minisynth.features import (
    DEFAULT_SAMPLE_RATE,
    mel_spectrogram,
    multi_resolution_stft_magnitude,
    normalize_loudness,
    resample_audio,
    rms_envelope,
    spectral_centroid,
    to_mono,
)


def preprocess_audio(audio, sample_rate, target_sample_rate=DEFAULT_SAMPLE_RATE):
    mono = to_mono(audio)
    resampled = resample_audio(mono, sample_rate, target_sample_rate)
    return normalize_loudness(resampled)


def mean_absolute_distance(left, right):
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    left_aligned, right_aligned = align_feature_arrays(left_array, right_array)
    return float(np.mean(np.abs(left_aligned - right_aligned)))


def align_feature_arrays(left, right):
    if left.ndim != right.ndim:
        raise ValueError(f"feature ranks must match: {left.ndim} != {right.ndim}")

    slices = tuple(slice(0, min(l_dim, r_dim)) for l_dim, r_dim in zip(left.shape, right.shape))
    return left[slices], right[slices]


def compare_audio_arrays(
    target_audio,
    target_sample_rate,
    candidate_audio,
    candidate_sample_rate,
):
    target = preprocess_audio(target_audio, target_sample_rate)
    candidate = preprocess_audio(candidate_audio, candidate_sample_rate)

    target_stfts = multi_resolution_stft_magnitude(target)
    candidate_stfts = multi_resolution_stft_magnitude(candidate)

    stft_distances = [
        mean_absolute_distance(target_stft, candidate_stft)
        for target_stft, candidate_stft in zip(target_stfts, candidate_stfts)
    ]

    return {
        "mel_distance": mean_absolute_distance(
            mel_spectrogram(target),
            mel_spectrogram(candidate),
        ),
        "rms_envelope_distance": mean_absolute_distance(
            rms_envelope(target),
            rms_envelope(candidate),
        ),
        "spectral_centroid_distance": mean_absolute_distance(
            spectral_centroid(target),
            spectral_centroid(candidate),
        ),
        "stft_magnitude_distances": stft_distances,
    }
