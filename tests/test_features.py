import unittest

import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.features import resample_audio, to_mono


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


if __name__ == "__main__":
    unittest.main()
