import unittest

from minisynth.engine import render_patch
from minisynth.schema import SynthConfig


class TestSynthConfig(unittest.TestCase):
    def test_synth_config_defaults_match_current_renderer_defaults(self):
        config = SynthConfig()

        self.assertEqual(config.freq, 261.63)
        self.assertEqual(config.length, 1.5)
        self.assertEqual(config.osc1_wave, "saw")
        self.assertEqual(config.osc1_level, 0.8)
        self.assertEqual(config.osc2_wave, "saw")
        self.assertEqual(config.osc2_level, 0.4)
        self.assertEqual(config.osc2_detune, 7)
        self.assertEqual(config.cutoff, 1200)
        self.assertEqual(config.resonance, 0.2)
        self.assertEqual(config.attack, 0.01)
        self.assertEqual(config.decay, 0.2)
        self.assertEqual(config.sustain, 0.7)
        self.assertEqual(config.release, 0.3)

    def test_synth_config_converts_to_render_kwargs(self):
        config = SynthConfig(osc1_wave="triangle")
        kwargs = config.to_render_kwargs()

        self.assertEqual(kwargs["length"], 1.5)
        self.assertEqual(kwargs["osc1_wave"], "triangle")

        audio = render_patch(**kwargs)
        self.assertEqual(len(audio), 66150)


if __name__ == "__main__":
    unittest.main()
