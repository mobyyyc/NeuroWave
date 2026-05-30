import unittest

import numpy as np

from minisynth.engine import render_patch
from minisynth.randomize import (
    ENVELOPE_TIME_PARAMETERS,
    MAX_DATASET_AUDIO_PEAK,
    MIN_OSCILLATOR_LEVEL_SUM,
    audio_avoids_clipping,
    random_patch,
)
from minisynth.schema import PARAMETERS, SynthConfig


class TestRandomPatch(unittest.TestCase):
    def test_random_patch_is_reproducible_for_same_seed(self):
        self.assertEqual(random_patch(1000), random_patch(1000))

    def test_random_patch_changes_for_different_seeds(self):
        self.assertNotEqual(random_patch(1000), random_patch(1001))

    def test_random_patch_enforces_minimum_oscillator_level_sum(self):
        for seed in range(100):
            patch = random_patch(seed)
            level_sum = patch["osc1_level"] + patch["osc2_level"]

            self.assertGreaterEqual(level_sum, MIN_OSCILLATOR_LEVEL_SUM)

    def test_random_patch_renders_non_silent_audio(self):
        for seed in range(100):
            patch = random_patch(seed)
            audio = render_patch(**patch)

            self.assertGreater(np.max(np.abs(audio)), 0.0)
            self.assertTrue(np.all(np.isfinite(audio)))

    def test_random_patch_audio_avoids_clipping(self):
        for seed in range(100):
            patch = random_patch(seed)
            audio = render_patch(**patch)

            self.assertTrue(audio_avoids_clipping(audio))
            self.assertLessEqual(np.max(np.abs(audio)), MAX_DATASET_AUDIO_PEAK)

    def test_random_patch_envelope_fits_note_length(self):
        for seed in range(100):
            patch = random_patch(seed)
            envelope_total = patch["attack"] + patch["decay"] + patch["release"]

            self.assertLessEqual(envelope_total, patch["length"])

    def test_random_patch_envelope_times_stay_inside_schema_after_scaling(self):
        parameters = {parameter.name: parameter for parameter in PARAMETERS}

        for seed in range(2000, 2500):
            patch = random_patch(seed)
            vector = SynthConfig(**patch).to_vector()

            for name in ENVELOPE_TIME_PARAMETERS:
                self.assertGreaterEqual(patch[name], parameters[name].minimum, seed)
                self.assertLessEqual(patch[name], parameters[name].maximum, seed)

            for value in vector:
                self.assertGreaterEqual(value, 0.0, seed)
                self.assertLessEqual(value, 1.0, seed)

    def test_random_patch_failed_v2_seeds_keep_decay_inside_schema(self):
        parameters = {parameter.name: parameter for parameter in PARAMETERS}

        for seed in (2054, 2121):
            patch = random_patch(seed)

            self.assertGreaterEqual(patch["decay"], parameters["decay"].minimum)

    def test_audio_avoids_clipping_rejects_bad_audio(self):
        self.assertFalse(audio_avoids_clipping(np.array([1.01])))
        self.assertFalse(audio_avoids_clipping(np.array([np.nan])))
        self.assertFalse(audio_avoids_clipping(np.array([])))


if __name__ == "__main__":
    unittest.main()
