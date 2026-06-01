import unittest

from minisynth.oscillator_mix import (
    main_detuned_error_report,
    oscillator_balance,
    oscillator_level_by_wave,
    oscillator_mix_error_report,
    summarize_main_detuned_errors,
    summarize_oscillator_mix_errors,
)


class TestOscillatorMix(unittest.TestCase):
    def test_level_by_wave_is_slot_invariant(self):
        first = {
            "osc1_wave": "saw",
            "osc1_level": 0.2,
            "osc2_wave": "sine",
            "osc2_level": 0.8,
        }
        swapped = {
            "osc1_wave": "sine",
            "osc1_level": 0.8,
            "osc2_wave": "saw",
            "osc2_level": 0.2,
        }

        self.assertEqual(oscillator_level_by_wave(first), oscillator_level_by_wave(swapped))

    def test_balance_reports_second_oscillator_share(self):
        patch = {
            "osc1_wave": "saw",
            "osc1_level": 0.25,
            "osc2_wave": "sine",
            "osc2_level": 0.75,
        }

        self.assertAlmostEqual(oscillator_balance(patch), 0.75)

    def test_mix_error_prefers_swapped_assignment_for_swapped_slots(self):
        target = {
            "osc1_wave": "saw",
            "osc1_level": 0.2,
            "osc2_wave": "sine",
            "osc2_level": 0.8,
            "osc2_detune": 100.0,
        }
        predicted = {
            "osc1_wave": "sine",
            "osc1_level": 0.8,
            "osc2_wave": "saw",
            "osc2_level": 0.2,
            "osc2_detune": -100.0,
        }

        report = oscillator_mix_error_report(target, predicted)

        self.assertEqual(report["best_assignment"], "swapped")
        self.assertEqual(report["total_level"]["absolute_error"], 0.0)
        self.assertEqual(report["per_wave_level"]["saw"]["absolute_error"], 0.0)
        self.assertEqual(report["per_wave_level"]["sine"]["absolute_error"], 0.0)
        self.assertLess(report["best_assignment_error"], report["direct_assignment_error"])

    def test_summarize_oscillator_mix_errors_aggregates_reports(self):
        target = {
            "osc1_wave": "saw",
            "osc1_level": 0.2,
            "osc2_wave": "sine",
            "osc2_level": 0.8,
            "osc2_detune": 0.0,
        }
        predicted = {
            "osc1_wave": "saw",
            "osc1_level": 0.3,
            "osc2_wave": "sine",
            "osc2_level": 0.7,
            "osc2_detune": 0.0,
        }
        report = oscillator_mix_error_report(target, predicted)

        summary = summarize_oscillator_mix_errors(
            [{"oscillator_mix_errors": report}]
        )

        self.assertEqual(summary["count"], 1)
        self.assertAlmostEqual(summary["mean_total_level_error"], 0.0)
        self.assertAlmostEqual(summary["per_wave_level_mae"]["saw"], 0.1)

    def test_main_detuned_errors_preserve_base_and_relative_roles(self):
        target = {
            "osc1_wave": "saw",
            "osc1_level": 0.75,
            "osc2_wave": "sine",
            "osc2_level": 0.25,
            "osc2_detune": 120.0,
        }
        predicted = {
            "osc1_wave": "saw",
            "osc1_level": 0.5,
            "osc2_wave": "triangle",
            "osc2_level": 0.5,
            "osc2_detune": 0.0,
        }

        report = main_detuned_error_report(target, predicted)
        summary = summarize_main_detuned_errors(
            [{"main_detuned_errors": report}]
        )

        self.assertEqual(report["main_wave_error"], 0.0)
        self.assertGreater(report["detuned_wave_error"], 0.0)
        self.assertAlmostEqual(report["detuned_balance"]["absolute_error"], 0.25)
        self.assertAlmostEqual(report["detune"]["normalized_absolute_error"], 0.05)
        self.assertEqual(summary["count"], 1)
        self.assertAlmostEqual(summary["mean_detuned_balance_error"], 0.25)


if __name__ == "__main__":
    unittest.main()
