import unittest

import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.features import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS,
    DEFAULT_STFT_RESOLUTIONS,
    DEFAULT_TARGET_RMS,
    mel_spectrogram,
    multi_resolution_stft_magnitude,
    normalize_loudness,
    resample_audio,
    rms,
    rms_envelope,
    stft_magnitude,
    to_mono,
)


class TestAudioFeatures(unittest.TestCase):
    def test_to_mono_returns_one_dimensional_audio_unchanged(self):
        audio = np.array([0.0, 0.25, -0.25])

        mono = to_mono(audio)

        np.testing.assert_array_equal(mono, audio)

    def test_to_mono_averages_stereo_channels(self):
        audio = np.array(
            [
                [1.0, -1.0],
                [0.5, 0.25],
                [-0.5, 0.0],
            ]
        )

        mono = to_mono(audio)

        np.testing.assert_allclose(mono, np.array([0.0, 0.375, -0.25]))

    def test_to_mono_averages_more_than_two_channels(self):
        audio = np.array(
            [
                [1.0, 0.0, -1.0],
                [0.25, 0.25, 0.25],
            ]
        )

        mono = to_mono(audio)

        np.testing.assert_allclose(mono, np.array([0.0, 0.25]))

    def test_to_mono_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            to_mono(np.zeros((2, 2, 2)))

        with self.assertRaises(ValueError):
            to_mono(np.zeros((2, 0)))

    def test_resample_audio_returns_unchanged_audio_when_rates_match(self):
        audio = np.array([0.0, 0.25, -0.25])

        resampled = resample_audio(audio, DEFAULT_SAMPLE_RATE)

        np.testing.assert_array_equal(resampled, audio)

    def test_resample_audio_changes_sample_count_for_new_rate(self):
        audio = np.linspace(-1.0, 1.0, 48000, endpoint=False)

        resampled = resample_audio(audio, 48000, 44100)

        self.assertEqual(len(resampled), 44100)
        self.assertTrue(np.all(np.isfinite(resampled)))

    def test_resample_audio_preserves_channel_last_shape(self):
        audio = np.zeros((48000, 2))

        resampled = resample_audio(audio, 48000, 44100)

        self.assertEqual(resampled.shape, (44100, 2))

    def test_resample_audio_rejects_invalid_sample_rates(self):
        with self.assertRaises(ValueError):
            resample_audio(np.zeros(10), 0)

        with self.assertRaises(ValueError):
            resample_audio(np.zeros(10), 44100, 0)

    def test_rms_computes_root_mean_square(self):
        audio = np.array([1.0, -1.0, 1.0, -1.0])

        self.assertAlmostEqual(rms(audio), 1.0)

    def test_rms_rejects_empty_audio(self):
        with self.assertRaises(ValueError):
            rms(np.array([]))

    def test_normalize_loudness_scales_audio_to_target_rms(self):
        audio = np.array([0.5, -0.5, 0.5, -0.5])

        normalized = normalize_loudness(audio)

        self.assertAlmostEqual(rms(normalized), DEFAULT_TARGET_RMS)

    def test_normalize_loudness_returns_zero_for_silence(self):
        audio = np.zeros(4)

        normalized = normalize_loudness(audio)

        np.testing.assert_array_equal(normalized, np.zeros(4))

    def test_normalize_loudness_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            normalize_loudness(np.ones(4), target_rms=0.0)

        with self.assertRaises(ValueError):
            normalize_loudness(np.ones(4), silence_threshold=-1.0)

    def test_mel_spectrogram_returns_mel_bins_by_frames(self):
        audio = np.sin(
            2 * np.pi * 440.0 * np.arange(DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        )

        spectrogram = mel_spectrogram(audio)

        self.assertEqual(spectrogram.shape[0], DEFAULT_N_MELS)
        self.assertEqual(spectrogram.shape[1], 1 + len(audio) // DEFAULT_HOP_LENGTH)
        self.assertTrue(np.all(np.isfinite(spectrogram)))

    def test_mel_spectrogram_accepts_stereo_input(self):
        mono = np.sin(
            2 * np.pi * 220.0 * np.arange(DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        )
        stereo = np.column_stack([mono, mono])

        spectrogram = mel_spectrogram(stereo, n_mels=32)

        self.assertEqual(spectrogram.shape[0], 32)
        self.assertTrue(np.all(np.isfinite(spectrogram)))

    def test_mel_spectrogram_handles_silence(self):
        spectrogram = mel_spectrogram(np.zeros(DEFAULT_SAMPLE_RATE // 2))

        self.assertEqual(spectrogram.shape[0], DEFAULT_N_MELS)
        self.assertTrue(np.all(np.isfinite(spectrogram)))

    def test_mel_spectrogram_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            mel_spectrogram(np.ones(10), sample_rate=0)

        with self.assertRaises(ValueError):
            mel_spectrogram(np.ones(10), n_mels=0)

        with self.assertRaises(ValueError):
            mel_spectrogram(np.ones(10), n_fft=0)

        with self.assertRaises(ValueError):
            mel_spectrogram(np.ones(10), hop_length=0)

        with self.assertRaises(ValueError):
            mel_spectrogram(np.array([]))

    def test_rms_envelope_returns_one_value_per_frame(self):
        audio = np.ones(DEFAULT_SAMPLE_RATE)

        envelope = rms_envelope(audio)

        self.assertEqual(envelope.shape, (1 + len(audio) // DEFAULT_HOP_LENGTH,))
        self.assertTrue(np.all(np.isfinite(envelope)))

    def test_rms_envelope_tracks_loudness_change(self):
        quiet = np.ones(DEFAULT_SAMPLE_RATE // 2) * 0.1
        loud = np.ones(DEFAULT_SAMPLE_RATE // 2) * 0.5
        envelope = rms_envelope(np.concatenate([quiet, loud]))

        self.assertLess(np.mean(envelope[:10]), np.mean(envelope[-10:]))

    def test_rms_envelope_accepts_stereo_input(self):
        mono = np.ones(DEFAULT_SAMPLE_RATE)
        stereo = np.column_stack([mono, mono])

        envelope = rms_envelope(stereo)

        self.assertEqual(envelope.shape, (1 + len(mono) // DEFAULT_HOP_LENGTH,))

    def test_rms_envelope_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            rms_envelope(np.ones(10), frame_length=0)

        with self.assertRaises(ValueError):
            rms_envelope(np.ones(10), hop_length=0)

        with self.assertRaises(ValueError):
            rms_envelope(np.array([]))

    def test_stft_magnitude_returns_frequency_bins_by_frames(self):
        audio = np.sin(
            2 * np.pi * 440.0 * np.arange(DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        )

        magnitude = stft_magnitude(audio, n_fft=1024, hop_length=256)

        self.assertEqual(magnitude.shape[0], 513)
        self.assertEqual(magnitude.shape[1], 1 + len(audio) // 256)
        self.assertTrue(np.all(magnitude >= 0.0))
        self.assertTrue(np.all(np.isfinite(magnitude)))

    def test_stft_magnitude_accepts_stereo_input(self):
        mono = np.ones(DEFAULT_SAMPLE_RATE)
        stereo = np.column_stack([mono, mono])

        magnitude = stft_magnitude(stereo, n_fft=512, hop_length=128)

        self.assertEqual(magnitude.shape[0], 257)

    def test_stft_magnitude_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            stft_magnitude(np.ones(10), n_fft=0)

        with self.assertRaises(ValueError):
            stft_magnitude(np.ones(10), hop_length=0)

        with self.assertRaises(ValueError):
            stft_magnitude(np.array([]))

    def test_multi_resolution_stft_magnitude_returns_one_array_per_resolution(self):
        audio = np.ones(DEFAULT_SAMPLE_RATE)

        magnitudes = multi_resolution_stft_magnitude(audio)

        self.assertEqual(len(magnitudes), len(DEFAULT_STFT_RESOLUTIONS))
        for magnitude, resolution in zip(magnitudes, DEFAULT_STFT_RESOLUTIONS):
            n_fft, hop_length = resolution
            self.assertEqual(magnitude.shape[0], 1 + n_fft // 2)
            self.assertEqual(magnitude.shape[1], 1 + len(audio) // hop_length)
            self.assertTrue(np.all(np.isfinite(magnitude)))

    def test_multi_resolution_stft_magnitude_rejects_empty_resolutions(self):
        with self.assertRaises(ValueError):
            multi_resolution_stft_magnitude(np.ones(10), resolutions=())


if __name__ == "__main__":
    unittest.main()
