import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from minisynth.dataset import write_random_dataset_files
from minisynth.engine import render_patch
from minisynth.ml import (
    create_mlp_regressor,
    load_model_checkpoint,
    parameter_mae,
    predict_patch_from_audio,
    predict_parameter_vectors,
    save_metrics_report,
    save_model_checkpoint,
    train_mlp_from_metadata,
    train_mlp_regressor,
)
from minisynth.schema import SynthConfig


class TestMLBaseline(unittest.TestCase):
    def test_create_mlp_regressor_returns_pipeline(self):
        model = create_mlp_regressor(max_iter=10)

        self.assertIsInstance(model, Pipeline)

    def test_train_mlp_regressor_fits_multi_output_targets(self):
        features = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        targets = np.array(
            [
                [0.0, 0.0],
                [0.25, 0.5],
                [0.5, 0.25],
                [1.0, 1.0],
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = train_mlp_regressor(
                features,
                targets,
                hidden_layer_sizes=(4,),
                max_iter=50,
                random_state=1,
            )
        predictions = predict_parameter_vectors(model, features)

        self.assertEqual(predictions.shape, targets.shape)
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

    def test_train_mlp_regressor_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            train_mlp_regressor(np.ones(3), np.ones((3, 1)))

        with self.assertRaises(ValueError):
            train_mlp_regressor(np.ones((3, 1)), np.ones(3))

        with self.assertRaises(ValueError):
            train_mlp_regressor(np.ones((3, 1)), np.ones((2, 1)))

    def test_parameter_mae_returns_single_distance(self):
        targets = np.array([[0.0, 0.5], [1.0, 0.25]])
        predictions = np.array([[0.25, 0.5], [0.5, 0.75]])

        self.assertAlmostEqual(parameter_mae(targets, predictions), 0.3125)

    def test_parameter_mae_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            parameter_mae(np.ones((2, 1)), np.ones((2, 2)))

    def test_train_mlp_from_metadata_returns_metrics(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=2000,
                count=5,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = train_mlp_from_metadata(
                    metadata_path,
                    hidden_layer_sizes=(4,),
                    max_iter=10,
                    random_state=1,
                    test_size=0.4,
                )

        metrics = result["metrics"]

        self.assertIn("model", result)
        self.assertEqual(metrics["num_samples"], 5)
        self.assertEqual(metrics["num_features"], 7)
        self.assertEqual(metrics["num_targets"], 13)
        self.assertEqual(metrics["train_samples"], 3)
        self.assertEqual(metrics["test_samples"], 2)
        self.assertGreaterEqual(metrics["train_mae"], 0.0)
        self.assertGreaterEqual(metrics["test_mae"], 0.0)

    def test_save_and_load_model_checkpoint(self):
        features = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        targets = np.array(
            [
                [0.0, 0.0],
                [0.25, 0.5],
                [0.5, 0.25],
                [1.0, 1.0],
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = train_mlp_regressor(
                features,
                targets,
                hidden_layer_sizes=(4,),
                max_iter=50,
                random_state=1,
            )

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "models" / "mlp.joblib"
            saved_path = save_model_checkpoint(
                model,
                checkpoint_path,
                metrics={"test_mae": 0.1},
            )
            checkpoint = load_model_checkpoint(saved_path)

        predictions = predict_parameter_vectors(checkpoint["model"], features[:1])

        self.assertEqual(saved_path, checkpoint_path)
        self.assertEqual(checkpoint["metrics"]["test_mae"], 0.1)
        self.assertEqual(predictions.shape, (1, 2))

    def test_save_metrics_report_writes_json(self):
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "runs" / "metrics.json"
            saved_path = save_metrics_report(
                {"num_samples": 500, "test_mae": 0.2},
                report_path,
            )

            text = saved_path.read_text(encoding="utf-8")

        self.assertEqual(saved_path, report_path)
        self.assertIn('"num_samples": 500', text)
        self.assertIn('"test_mae": 0.2', text)

    def test_predict_patch_from_audio_returns_renderable_patch(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=3000,
                count=5,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = train_mlp_from_metadata(
                    metadata_path,
                    hidden_layer_sizes=(4,),
                    max_iter=10,
                    random_state=1,
                    test_size=0.4,
                )

        source_patch = {
            "freq": 261.63,
            "length": 1.0,
            "osc1_wave": "saw",
            "osc1_level": 0.8,
            "osc2_wave": "saw",
            "osc2_level": 0.4,
            "osc2_detune": 7,
            "cutoff": 1200,
            "resonance": 0.2,
            "attack": 0.01,
            "decay": 0.2,
            "sustain": 0.7,
            "release": 0.3,
        }
        audio = render_patch(**source_patch)
        patch = predict_patch_from_audio(result["model"], audio, 44100)
        rendered = render_patch(**patch)

        self.assertIn("osc1_wave", patch)
        self.assertIn("cutoff", patch)
        self.assertGreater(len(rendered), 0)

    def test_predict_patch_from_audio_constrains_envelope_to_predicted_length(self):
        class StaticModel:
            def predict(self, features):
                patch = {
                    "freq": 261.63,
                    "length": 0.05,
                    "osc1_wave": "saw",
                    "osc1_level": 0.8,
                    "osc2_wave": "saw",
                    "osc2_level": 0.4,
                    "osc2_detune": 7,
                    "cutoff": 1200,
                    "resonance": 0.2,
                    "attack": 5.0,
                    "decay": 5.0,
                    "sustain": 0.7,
                    "release": 5.0,
                }
                return np.array([SynthConfig(**patch).to_vector()])

        audio = render_patch(length=1.0)
        patch = predict_patch_from_audio(StaticModel(), audio, 44100)
        rendered = render_patch(**patch)

        envelope_total = patch["attack"] + patch["decay"] + patch["release"]

        self.assertLessEqual(envelope_total, patch["length"])
        self.assertGreater(len(rendered), 0)


if __name__ == "__main__":
    unittest.main()
