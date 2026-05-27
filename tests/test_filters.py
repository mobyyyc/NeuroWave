import unittest

import numpy as np

from minisynth.filters import lowpass_filter


class TestLowpassFilter(unittest.TestCase):
    def test_lowpass_filter_preserves_length_and_finite_values(self):
        audio = np.linspace(-1.0, 1.0, 1000)
        filtered = lowpass_filter(audio, cutoff=2000, resonance=0.2)

        self.assertEqual(len(filtered), len(audio))
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_lowpass_filter_clips_extreme_parameters(self):
        audio = np.ones(1000)
        filtered = lowpass_filter(audio, cutoff=1_000_000, resonance=10.0)

        self.assertEqual(len(filtered), len(audio))
        self.assertTrue(np.all(np.isfinite(filtered)))


if __name__ == "__main__":
    unittest.main()
