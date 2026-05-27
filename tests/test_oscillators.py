import unittest

import numpy as np

from minisynth.oscillators import BASE_WAVES, normalize_wave_mix, wave_mix


class TestWaveMix(unittest.TestCase):
    def test_normalize_wave_mix_fills_missing_waves_and_sums_to_one(self):
        weights = normalize_wave_mix({"sine": 2, "triangle": 1, "saw": 1})

        self.assertEqual(set(weights), set(BASE_WAVES))
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["sine"], 0.5)
        self.assertAlmostEqual(weights["triangle"], 0.25)
        self.assertAlmostEqual(weights["saw"], 0.25)
        self.assertEqual(weights["square"], 0.0)
        self.assertEqual(weights["noise"], 0.0)

    def test_normalize_wave_mix_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            normalize_wave_mix({"sine": 1, "saw": -0.5})

    def test_normalize_wave_mix_rejects_zero_total(self):
        with self.assertRaises(ValueError):
            normalize_wave_mix({"sine": 0, "triangle": 0})

    def test_wave_mix_length_and_value_range(self):
        audio = wave_mix(
            {"sine": 0.2, "triangle": 0.3, "saw": 0.5, "noise": 0.1},
            freq=440,
            length=1.0,
            sample_rate=44100,
        )

        self.assertEqual(len(audio), 44100)
        self.assertTrue(np.all(np.isfinite(audio)))
        self.assertGreaterEqual(float(audio.min()), -1.0)
        self.assertLessEqual(float(audio.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
