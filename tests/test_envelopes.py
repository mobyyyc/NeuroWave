import unittest

from minisynth.envelopes import adsr


class TestAdsr(unittest.TestCase):
    def test_adsr_length_and_value_range(self):
        env = adsr(
            length=1.5,
            attack=0.01,
            decay=0.2,
            sustain=0.7,
            release=0.3,
            sample_rate=44100,
        )

        self.assertEqual(len(env), 66150)
        self.assertGreaterEqual(float(env.min()), 0.0)
        self.assertLessEqual(float(env.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
