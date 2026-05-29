import os
import tempfile
import unittest

import numpy as np

from minisynth.engine import render_patch


class TestRenderPatch(unittest.TestCase):
    def test_render_patch_returns_audio_without_file_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                audio = render_patch(length=1.5)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(audio.ndim, 1)
            self.assertEqual(len(audio), 66150)
            self.assertTrue(np.all(np.isfinite(audio)))
            self.assertLessEqual(float(np.max(np.abs(audio))), 0.8)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "dark_saw.wav")))

    def test_render_patch_is_deterministic_for_same_parameters(self):
        first = render_patch(length=1.5)
        second = render_patch(length=1.5)

        np.testing.assert_array_equal(first, second)

    def test_render_patch_returns_finite_silence_for_zero_oscillator_levels(self):
        audio = render_patch(length=1.5, osc1_level=0.0, osc2_level=0.0)

        self.assertTrue(np.all(np.isfinite(audio)))
        self.assertEqual(float(np.max(np.abs(audio))), 0.0)


if __name__ == "__main__":
    unittest.main()
