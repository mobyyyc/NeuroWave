import unittest

import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.features import (
    DEFAULT_TARGET_RMS,
    normalize_loudness,
    resample_audio,
    rms,
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


if __name__ == "__main__":
    unittest.main()
