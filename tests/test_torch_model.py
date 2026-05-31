import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES
from minisynth.engine import render_patch
from minisynth.io import load_patch
from minisynth.schema import VECTOR_PARAMETERS
from minisynth.torch_model import (
    DEFAULT_MEL_BINS,
    DEFAULT_WAVEFORM_MODE,
    MelSpectrogramInverseModel,
    create_inverse_model,
    expected_mel_tensor_shape,
    grouped_parameter_mae,
    load_mel_tensor_npz,
    load_torch_checkpoint,
    parameter_mae_by_name,
    parameter_mse_torch,
    predict_patch_from_audio,
    save_torch_checkpoint,
    split_tensor_dataset,
    split_tensor_dataset_with_benchmark,
    train_inverse_model,
    predict_normalized_vectors,
    select_torch_device,
    waveform_accuracy_by_name,
)


class TestTorchInverseModel(unittest.TestCase):
    def test_create_inverse_model_outputs_normalized_parameter_vectors(self):
        model = create_inverse_model()
        inputs = torch.zeros(
            2,
            1,
            DEFAULT_MEL_BINS,
            DEFAULT_MEL_TENSOR_FRAMES,
            dtype=torch.float32,
        )

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (2, len(VECTOR_PARAMETERS)))
        self.assertTrue(torch.all(outputs >= 0.0))
        self.assertTrue(torch.all(outputs <= 1.0))
        self.assertEqual(model.waveform_mode, DEFAULT_WAVEFORM_MODE)

    def test_create_inverse_model_supports_legacy_scalar_waveform_mode(self):
        model = create_inverse_model(waveform_mode="scalar_regression")
        inputs = torch.zeros(
            2,
            1,
            DEFAULT_MEL_BINS,
            DEFAULT_MEL_TENSOR_FRAMES,
            dtype=torch.float32,
        )

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (2, len(VECTOR_PARAMETERS)))
        self.assertEqual(model.waveform_mode, "scalar_regression")

    def test_model_rejects_wrong_input_rank(self):
        model = create_inverse_model()

        with self.assertRaises(ValueError):
            model(torch.zeros(1, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES))

    def test_model_rejects_wrong_channel_count(self):
        model = create_inverse_model()

        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES))

    def test_model_rejects_invalid_output_dim(self):
        with self.assertRaises(ValueError):
            MelSpectrogramInverseModel(output_dim=0)

    def test_model_rejects_invalid_waveform_mode(self):
        with self.assertRaises(ValueError):
            MelSpectrogramInverseModel(waveform_mode="bad-mode")

    def test_predict_normalized_vectors_returns_numpy_array(self):
        model = create_inverse_model()
        inputs = np.zeros(
            (1, 1, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES),
            dtype=np.float32,
        )

        predictions = predict_normalized_vectors(model, inputs, device=torch.device("cpu"))

        self.assertEqual(predictions.shape, (1, len(VECTOR_PARAMETERS)))
        self.assertEqual(predictions.dtype, np.float32)
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

    def test_select_torch_device_returns_cpu_or_mps(self):
        device = select_torch_device()

        self.assertIn(device.type, ("cpu", "mps", "cuda"))

    def test_expected_mel_tensor_shape_matches_export_shape(self):
        self.assertEqual(
            expected_mel_tensor_shape(),
            (1, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES),
        )

    def test_load_mel_tensor_npz_returns_features_and_targets(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mel_tensors.npz"
            np.savez_compressed(
                path,
                features=np.zeros((4, 1, DEFAULT_MEL_BINS, 8), dtype=np.float32),
                targets=np.zeros((4, len(VECTOR_PARAMETERS)), dtype=np.float32),
                metadata_path="metadata.jsonl",
                frames=np.asarray(8, dtype=np.int64),
            )

            dataset = load_mel_tensor_npz(path)

        self.assertEqual(dataset["features"].shape, (4, 1, DEFAULT_MEL_BINS, 8))
        self.assertEqual(dataset["targets"].shape, (4, len(VECTOR_PARAMETERS)))
        self.assertEqual(dataset["metadata_path"], "metadata.jsonl")
        self.assertEqual(dataset["frames"], 8)

    def test_split_tensor_dataset_is_reproducible(self):
        features = np.arange(10 * 2, dtype=np.float32).reshape(10, 1, 1, 2)
        targets = np.arange(10, dtype=np.float32).reshape(10, 1)

        first = split_tensor_dataset(features, targets, test_size=0.2, random_state=4)
        second = split_tensor_dataset(features, targets, test_size=0.2, random_state=4)

        self.assertEqual(len(first["train_features"]), 8)
        self.assertEqual(len(first["test_features"]), 2)
        np.testing.assert_array_equal(first["test_targets"], second["test_targets"])

    def test_split_tensor_dataset_with_benchmark_reserves_fixed_indices(self):
        features = np.arange(20 * 2, dtype=np.float32).reshape(20, 1, 1, 2)
        targets = np.arange(20, dtype=np.float32).reshape(20, 1)

        first = split_tensor_dataset_with_benchmark(
            features,
            targets,
            test_size=0.2,
            benchmark_size=0.1,
            random_state=4,
        )
        second = split_tensor_dataset_with_benchmark(
            features,
            targets,
            test_size=0.2,
            benchmark_size=0.1,
            random_state=4,
        )

        self.assertEqual(len(first["train_features"]), 14)
        self.assertEqual(len(first["test_features"]), 4)
        self.assertEqual(len(first["benchmark_features"]), 2)
        np.testing.assert_array_equal(first["benchmark_indices"], second["benchmark_indices"])

    def test_parameter_mae_by_name_reports_each_target(self):
        targets = np.zeros((2, len(VECTOR_PARAMETERS)), dtype=np.float32)
        predictions = np.full((2, len(VECTOR_PARAMETERS)), 0.25, dtype=np.float32)

        metrics = parameter_mae_by_name(predictions, targets)

        self.assertEqual(set(metrics), {parameter.name for parameter in VECTOR_PARAMETERS})
        self.assertAlmostEqual(metrics["freq"], 0.25)

    def test_grouped_parameter_mae_reports_model_quality_groups(self):
        targets = np.zeros((2, len(VECTOR_PARAMETERS)), dtype=np.float32)
        predictions = np.full((2, len(VECTOR_PARAMETERS)), 0.25, dtype=np.float32)

        metrics = grouped_parameter_mae(predictions, targets)

        self.assertAlmostEqual(metrics["pitch"], 0.25)
        self.assertAlmostEqual(metrics["adsr"], 0.25)
        self.assertAlmostEqual(metrics["oscillator"], 0.25)
        self.assertAlmostEqual(metrics["filter"], 0.25)
        self.assertAlmostEqual(metrics["pitch_conditioned_timbre"], 0.25)

    def test_waveform_accuracy_by_name_decodes_categorical_targets(self):
        targets = np.zeros((2, len(VECTOR_PARAMETERS)), dtype=np.float32)
        predictions = targets.copy()
        wave_indices = {
            parameter.name: index
            for index, parameter in enumerate(VECTOR_PARAMETERS)
            if parameter.kind == "enum"
        }
        targets[:, wave_indices["osc1_wave"]] = [0.0, 1.0]
        predictions[:, wave_indices["osc1_wave"]] = [0.0, 1.0]
        targets[:, wave_indices["osc2_wave"]] = [0.0, 1.0]
        predictions[:, wave_indices["osc2_wave"]] = [1.0, 0.0]

        metrics = waveform_accuracy_by_name(predictions, targets)

        self.assertEqual(metrics["osc1_wave"], 1.0)
        self.assertEqual(metrics["osc2_wave"], 0.0)

    def test_train_inverse_model_returns_metrics(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mel_tensors.npz"
            np.savez_compressed(
                path,
                features=np.zeros((6, 1, DEFAULT_MEL_BINS, 16), dtype=np.float32),
                targets=np.full((6, len(VECTOR_PARAMETERS)), 0.5, dtype=np.float32),
                metadata_path="metadata.jsonl",
                frames=np.asarray(16, dtype=np.int64),
            )

            with redirect_stdout(StringIO()):
                result = train_inverse_model(
                    tensor_path=path,
                    model_id="v_test_pytorch_cnn",
                    epochs=1,
                    batch_size=2,
                    random_state=1,
                    device=torch.device("cpu"),
                    progress=True,
                )

        metrics = result["metrics"]

        self.assertIn("model", result)
        self.assertEqual(metrics["model_id"], "v_test_pytorch_cnn")
        self.assertEqual(metrics["num_samples"], 6)
        self.assertEqual(metrics["train_samples"], 5)
        self.assertEqual(metrics["test_samples"], 1)
        self.assertEqual(metrics["epochs"], 1)
        self.assertEqual(metrics["device"], "cpu")
        self.assertEqual(metrics["waveform_mode"], DEFAULT_WAVEFORM_MODE)
        self.assertGreaterEqual(metrics["train_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_mae"], 0.0)
        self.assertIn("freq", metrics["test_per_parameter_mae"])
        self.assertIn("adsr", metrics["test_grouped_mae"])
        self.assertIn("pitch_conditioned_timbre", metrics["test_grouped_mae"])
        self.assertIn("osc1_wave", metrics["test_waveform_accuracy_by_name"])
        self.assertGreaterEqual(metrics["test_continuous_mae"], 0.0)

    def test_train_inverse_model_can_report_benchmark_metrics(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mel_tensors.npz"
            np.savez_compressed(
                path,
                features=np.zeros((10, 1, DEFAULT_MEL_BINS, 16), dtype=np.float32),
                targets=np.full((10, len(VECTOR_PARAMETERS)), 0.5, dtype=np.float32),
                metadata_path="metadata.jsonl",
                frames=np.asarray(16, dtype=np.int64),
            )

            result = train_inverse_model(
                tensor_path=path,
                model_id="v_test_pytorch_cnn",
                epochs=1,
                batch_size=2,
                random_state=1,
                benchmark_size=0.2,
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["benchmark_samples"], 2)
        self.assertIn("benchmark_loss", metrics)
        self.assertIn("benchmark_per_parameter_mae", metrics)
        self.assertIn("benchmark_grouped_mae", metrics)

    def test_parameter_mse_torch_returns_single_distance(self):
        model = create_inverse_model()
        features = np.zeros(
            (2, 1, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES),
            dtype=np.float32,
        )
        targets = np.zeros((2, len(VECTOR_PARAMETERS)), dtype=np.float32)

        distance = parameter_mse_torch(
            model,
            features,
            targets,
            device=torch.device("cpu"),
        )

        self.assertGreaterEqual(distance, 0.0)

    def test_save_and_load_torch_checkpoint(self):
        model = create_inverse_model()

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            saved_path = save_torch_checkpoint(
                model,
                path,
                metrics={"test_mae": 0.25},
            )
            checkpoint = load_torch_checkpoint(saved_path, device=torch.device("cpu"))

        inputs = np.zeros(
            (1, 1, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES),
            dtype=np.float32,
        )
        predictions = predict_normalized_vectors(checkpoint["model"], inputs)

        self.assertEqual(saved_path, path)
        self.assertEqual(checkpoint["metrics"]["test_mae"], 0.25)
        self.assertEqual(checkpoint["model"].waveform_mode, DEFAULT_WAVEFORM_MODE)
        self.assertEqual(predictions.shape, (1, len(VECTOR_PARAMETERS)))

    def test_predict_patch_from_audio_returns_renderable_patch(self):
        model = create_inverse_model()
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

        patch = predict_patch_from_audio(
            model,
            audio,
            44100,
            device=torch.device("cpu"),
        )
        rendered = render_patch(**patch)

        self.assertIn("osc1_wave", patch)
        self.assertIn("cutoff", patch)
        self.assertGreater(len(rendered), 0)

    def test_predict_patch_torch_cli_writes_patch_json(self):
        from scripts.predict_patch_torch import main

        model = create_inverse_model()
        audio = render_patch(length=1.0)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "target.wav"
            model_path = root / "model.pt"
            output_path = root / "predicted.json"
            sf.write(audio_path, audio, 44100)
            save_torch_checkpoint(model, model_path, metrics={"test_mae": 0.2})

            import sys

            original_argv = sys.argv
            try:
                sys.argv = [
                    "predict_patch_torch.py",
                    str(audio_path),
                    str(output_path),
                    "--model",
                    str(model_path),
                    "--device",
                    "cpu",
                ]
                with redirect_stdout(StringIO()):
                    exit_code = main()
            finally:
                sys.argv = original_argv

            patch = load_patch(output_path)

        self.assertEqual(exit_code, 0)
        self.assertIn("osc1_wave", patch)
        self.assertIn("release", patch)
