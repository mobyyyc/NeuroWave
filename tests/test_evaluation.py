import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
from sklearn.exceptions import ConvergenceWarning

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.dataset import write_random_dataset_files
from minisynth.engine import render_patch
from minisynth.evaluation import evaluate_audio_prediction
from minisynth.evaluation import evaluate_patch_prediction
from minisynth.evaluation import summarize_weighted_distances
from minisynth.ml import train_mlp_from_metadata
from minisynth.torch_model import create_inverse_model, save_torch_checkpoint


class TestPredictionEvaluation(unittest.TestCase):
    def test_evaluate_audio_prediction_returns_patch_audio_and_comparison(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=4000,
                count=5,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                training = train_mlp_from_metadata(
                    metadata_path,
                    hidden_layer_sizes=(4,),
                    max_iter=10,
                    random_state=1,
                    test_size=0.4,
                )

        target_audio = render_patch(length=1.0)
        result = evaluate_audio_prediction(
            training["model"],
            target_audio,
            DEFAULT_SAMPLE_RATE,
            refine_iterations=2,
            refine_seed=1,
            refine_step_size=0.01,
        )

        self.assertIn("freq", result["patch"])
        self.assertEqual(result["rendered_audio"].ndim, 1)
        self.assertIn("weighted_distance", result["comparison"])
        self.assertGreaterEqual(result["comparison"]["weighted_distance"], 0.0)
        self.assertEqual(result["refinement"]["iterations"], 2)
        self.assertLessEqual(
            result["refinement"]["best_score"],
            result["refinement"]["initial_score"],
        )

    def test_summarize_weighted_distances_returns_aggregate_scores(self):
        summary = summarize_weighted_distances(
            [
                {"comparison": {"weighted_distance": 3.0}},
                {"comparison": {"weighted_distance": 1.0}},
                {"comparison": {"weighted_distance": 2.0}},
            ]
        )

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean_weighted_distance"], 2.0)
        self.assertEqual(summary["median_weighted_distance"], 2.0)
        self.assertEqual(summary["min_weighted_distance"], 1.0)
        self.assertEqual(summary["max_weighted_distance"], 3.0)

    def test_summarize_weighted_distances_rejects_empty_results(self):
        with self.assertRaises(ValueError):
            summarize_weighted_distances([])

    def test_evaluate_patch_prediction_returns_weighted_distance(self):
        patch = {
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
        target_audio = render_patch(**patch)

        result = evaluate_patch_prediction(patch, target_audio, DEFAULT_SAMPLE_RATE)

        self.assertIn("weighted_distance", result["comparison"])
        self.assertGreaterEqual(result["comparison"]["weighted_distance"], 0.0)
        self.assertEqual(result["rendered_audio"].ndim, 1)

    def test_evaluate_prediction_torch_cli_writes_report(self):
        from scripts.evaluate_prediction_torch import main

        model = create_inverse_model()
        audio = render_patch(length=1.0)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "target.wav"
            model_path = root / "model.pt"
            output_dir = root / "run"
            sf.write(audio_path, audio, DEFAULT_SAMPLE_RATE)
            save_torch_checkpoint(model, model_path, metrics={"test_mae": 0.2})

            import sys

            original_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_prediction_torch.py",
                    str(audio_path),
                    "--model",
                    str(model_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
                with redirect_stdout(StringIO()):
                    exit_code = main()
            finally:
                sys.argv = original_argv

            report_text = (output_dir / "report.json").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn('"weighted_distance"', report_text)
        self.assertIn('"predicted_patch"', report_text)

    def test_evaluate_dataset_torch_cli_writes_summary_report(self):
        from scripts.evaluate_dataset_torch import main

        model = create_inverse_model()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            model_path = root / "model.pt"
            output_path = root / "eval.json"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=5000,
                count=2,
            )
            save_torch_checkpoint(model, model_path, metrics={"test_mae": 0.2})

            import sys

            original_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_dataset_torch.py",
                    "--metadata",
                    str(metadata_path),
                    "--model",
                    str(model_path),
                    "--count",
                    "2",
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                ]
                with redirect_stdout(StringIO()):
                    exit_code = main()
            finally:
                sys.argv = original_argv

            report_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn('"mean_weighted_distance"', report_text)
        self.assertIn('"failed_count": 0', report_text)


if __name__ == "__main__":
    unittest.main()
