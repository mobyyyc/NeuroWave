import unittest

import numpy as np

from minisynth.compare import (
    DEFAULT_DISTANCE_WEIGHTS,
    align_feature_arrays,
    compare_audio_arrays,
    mean_absolute_distance,
    preprocess_audio,
    weighted_similarity_distance,
)
from minisynth.constants import DEFAULT_SAMPLE_RATE


class TestAudioCompare(unittest.TestCase):
    def test_preprocess_audio_makes_mono_target_rate_loudness_normalized(self):
        audio = np.column_stack([np.ones(48000) * 0.5, np.ones(48000) * -0.5])

        processed = preprocess_audio(audio, 48000)

        self.assertEqual(processed.ndim, 1)
        self.assertEqual(len(processed), DEFAULT_SAMPLE_RATE)
        self.assertTrue(np.all(np.isfinite(processed)))

    def test_mean_absolute_distance_aligns_feature_lengths(self):
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([1.0, 4.0])

        self.assertEqual(mean_absolute_distance(left, right), 1.0)

    def test_align_feature_arrays_rejects_different_ranks(self):
        with self.assertRaises(ValueError):
            align_feature_arrays(np.ones(3), np.ones((3, 1)))

    def test_compare_audio_arrays_returns_feature_distances(self):
        audio = np.sin(
            2 * np.pi * 440.0 * np.arange(DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        )

        result = compare_audio_arrays(
            audio,
            DEFAULT_SAMPLE_RATE,
            audio,
            DEFAULT_SAMPLE_RATE,
        )

        self.assertIn("mel_distance", result)
        self.assertIn("rms_envelope_distance", result)
        self.assertIn("spectral_centroid_distance", result)
        self.assertIn("stft_magnitude_distances", result)
        self.assertIn("weighted_distance", result)
        self.assertEqual(len(result["stft_magnitude_distances"]), 3)

        for value in result.values():
            if isinstance(value, list):
                self.assertTrue(all(distance >= 0.0 for distance in value))
            else:
                self.assertGreaterEqual(value, 0.0)

    def test_weighted_similarity_distance_combines_feature_distances(self):
        distances = {
            "mel_distance": 10.0,
            "rms_envelope_distance": 1.0,
            "spectral_centroid_distance": 5.0,
            "stft_magnitude_distances": [1.0, 2.0, 3.0],
        }

        score = weighted_similarity_distance(distances)

        expected = (
            10.0 * DEFAULT_DISTANCE_WEIGHTS["mel_distance"]
            + 1.0 * DEFAULT_DISTANCE_WEIGHTS["rms_envelope_distance"]
            + 5.0 * DEFAULT_DISTANCE_WEIGHTS["spectral_centroid_distance"]
            + 2.0 * DEFAULT_DISTANCE_WEIGHTS["stft_magnitude_distance"]
        )
        self.assertAlmostEqual(score, expected)

    def test_compare_audio_arrays_has_zero_weighted_distance_for_identical_audio(self):
        audio = np.sin(
            2 * np.pi * 440.0 * np.arange(DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        )

        result = compare_audio_arrays(
            audio,
            DEFAULT_SAMPLE_RATE,
            audio,
            DEFAULT_SAMPLE_RATE,
        )

        self.assertAlmostEqual(result["weighted_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
