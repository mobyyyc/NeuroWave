import unittest

from minisynth.engine import render_patch
from minisynth.schema import (
    PARAMETERS,
    VECTOR_PARAMETERS,
    Parameter,
    SynthConfig,
    config_to_vector,
    denormalize_linear,
    denormalize_log,
    normalize_linear,
    normalize_log,
    normalize_parameter_value,
)


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

    def test_parameter_metadata_covers_synth_config_fields(self):
        config_fields = set(SynthConfig().__dataclass_fields__)
        parameter_names = {parameter.name for parameter in PARAMETERS}

        self.assertEqual(parameter_names, config_fields)

    def test_parameter_metadata_includes_required_fields(self):
        parameter = PARAMETERS[0]

        self.assertIsInstance(parameter, Parameter)
        self.assertEqual(parameter.name, "freq")
        self.assertEqual(parameter.kind, "float")
        self.assertEqual(parameter.minimum, 20.0)
        self.assertEqual(parameter.maximum, 20000.0)
        self.assertEqual(parameter.default, 261.63)
        self.assertEqual(parameter.scale, "log")
        self.assertEqual(parameter.group, "global")
        self.assertTrue(parameter.ml_enabled)

    def test_linear_normalization_round_trip(self):
        normalized = normalize_linear(0.495, 0.0, 0.99)
        value = denormalize_linear(normalized, 0.0, 0.99)

        self.assertAlmostEqual(normalized, 0.5)
        self.assertAlmostEqual(value, 0.495)

    def test_linear_normalization_handles_negative_ranges(self):
        normalized = normalize_linear(0.0, -1200.0, 1200.0)
        value = denormalize_linear(normalized, -1200.0, 1200.0)

        self.assertAlmostEqual(normalized, 0.5)
        self.assertAlmostEqual(value, 0.0)

    def test_linear_normalization_rejects_invalid_ranges(self):
        with self.assertRaises(ValueError):
            normalize_linear(1.0, 1.0, 1.0)

        with self.assertRaises(ValueError):
            denormalize_linear(0.5, 1.0, 1.0)

    def test_log_normalization_round_trip(self):
        normalized = normalize_log(200.0, 20.0, 20000.0)
        value = denormalize_log(normalized, 20.0, 20000.0)

        self.assertAlmostEqual(value, 200.0)

    def test_log_normalization_places_geometric_mean_at_half(self):
        normalized = normalize_log(2000.0, 20.0, 200000.0)
        value = denormalize_log(0.5, 20.0, 200000.0)

        self.assertAlmostEqual(normalized, 0.5)
        self.assertAlmostEqual(value, 2000.0)

    def test_log_normalization_rejects_invalid_ranges(self):
        with self.assertRaises(ValueError):
            normalize_log(10.0, 0.0, 100.0)

        with self.assertRaises(ValueError):
            normalize_log(0.0, 1.0, 100.0)

        with self.assertRaises(ValueError):
            denormalize_log(0.5, 0.0, 100.0)

    def test_parameter_defaults_round_trip_through_normalization(self):
        for parameter in PARAMETERS:
            if parameter.kind != "float":
                continue

            if parameter.scale == "linear":
                normalized = normalize_linear(
                    parameter.default,
                    parameter.minimum,
                    parameter.maximum,
                )
                value = denormalize_linear(
                    normalized,
                    parameter.minimum,
                    parameter.maximum,
                )
            elif parameter.scale == "log":
                normalized = normalize_log(
                    parameter.default,
                    parameter.minimum,
                    parameter.maximum,
                )
                value = denormalize_log(
                    normalized,
                    parameter.minimum,
                    parameter.maximum,
                )
            else:
                continue

            self.assertGreaterEqual(normalized, 0.0, parameter.name)
            self.assertLessEqual(normalized, 1.0, parameter.name)
            self.assertAlmostEqual(value, parameter.default, msg=parameter.name)

    def test_vector_parameters_include_ml_enabled_parameters(self):
        self.assertEqual(VECTOR_PARAMETERS, PARAMETERS)

    def test_synth_config_converts_to_normalized_vector(self):
        config = SynthConfig()

        vector = config.to_vector()

        self.assertEqual(len(vector), len(VECTOR_PARAMETERS))
        for value in vector:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_config_to_vector_matches_synth_config_method(self):
        config = SynthConfig(osc1_wave="sine", osc2_wave="noise")

        self.assertEqual(config_to_vector(config), config.to_vector())

    def test_categorical_parameter_values_convert_to_vector_positions(self):
        parameter = next(parameter for parameter in PARAMETERS if parameter.name == "osc1_wave")

        self.assertEqual(normalize_parameter_value(parameter, "sine"), 0.0)
        self.assertEqual(normalize_parameter_value(parameter, "noise"), 1.0)

    def test_categorical_parameter_rejects_unknown_value(self):
        parameter = next(parameter for parameter in PARAMETERS if parameter.name == "osc1_wave")

        with self.assertRaises(ValueError):
            normalize_parameter_value(parameter, "unknown")


if __name__ == "__main__":
    unittest.main()
