import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES
from minisynth.schema import VECTOR_PARAMETERS
from minisynth.torch_model import (
    DEFAULT_MEL_BINS,
    MelSpectrogramInverseModel,
    create_inverse_model,
    expected_mel_tensor_shape,
    load_mel_tensor_npz,
    load_torch_checkpoint,
    parameter_mse_torch,
    save_torch_checkpoint,
    split_tensor_dataset,
    train_inverse_model,
    predict_normalized_vectors,
    select_torch_device,
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

        self.assertIn(device.type, ("cpu", "mps"))

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

            result = train_inverse_model(
                tensor_path=path,
                epochs=1,
                batch_size=2,
                random_state=1,
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertIn("model", result)
        self.assertEqual(metrics["num_samples"], 6)
        self.assertEqual(metrics["train_samples"], 5)
        self.assertEqual(metrics["test_samples"], 1)
        self.assertEqual(metrics["epochs"], 1)
        self.assertEqual(metrics["device"], "cpu")
        self.assertGreaterEqual(metrics["train_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_mae"], 0.0)

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
        self.assertEqual(predictions.shape, (1, len(VECTOR_PARAMETERS)))
