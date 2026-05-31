import unittest

import numpy as np
import torch

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES
from minisynth.schema import VECTOR_PARAMETERS
from minisynth.torch_model import (
    DEFAULT_MEL_BINS,
    MelSpectrogramInverseModel,
    create_inverse_model,
    expected_mel_tensor_shape,
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
