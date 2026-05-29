import unittest

import numpy as np

from minisynth.features import to_mono


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


if __name__ == "__main__":
    unittest.main()
