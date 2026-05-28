import json
import tempfile
import unittest
from pathlib import Path

from minisynth.io import load_patch, save_patch


class TestPatchIo(unittest.TestCase):
    def test_load_patch_reads_json_patch(self):
        patch = load_patch("presets/dark_saw.json")

        self.assertEqual(patch["osc1_wave"], "saw")
        self.assertEqual(patch["resonance"], 0.7)

    def test_save_patch_writes_json_patch(self):
        patch = {"osc1_wave": "triangle", "osc1_level": 0.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "patch.json"
            save_patch(patch, path)

            self.assertEqual(load_patch(path), patch)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), patch)


if __name__ == "__main__":
    unittest.main()
