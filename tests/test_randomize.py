import unittest

import numpy as np

from minisynth.engine import render_patch
from minisynth.randomize import MIN_OSCILLATOR_LEVEL_SUM, random_patch


class TestRandomPatch(unittest.TestCase):
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

    def test_random_patch_envelope_fits_note_length(self):
        for seed in range(100):
            patch = random_patch(seed)
            envelope_total = patch["attack"] + patch["decay"] + patch["release"]

            self.assertLessEqual(envelope_total, patch["length"])


if __name__ == "__main__":
    unittest.main()
