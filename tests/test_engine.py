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


if __name__ == "__main__":
    unittest.main()
