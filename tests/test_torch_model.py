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
    DEFAULT_HEAD_MODE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_POOLING_MODE,
    DEFAULT_TARGET_MODE,
    TARGET_MODE_MAIN_DETUNED_MIX,
    TARGET_MODE_OSCILLATOR_MIX,
    DEFAULT_WAVEFORM_MODE,
    MelSpectrogramInverseModel,
    build_cnn_encoder,
    create_optimizer,
    create_scheduler,
    create_inverse_model,
    dataset_loss_torch,
    expected_mel_tensor_shape,
    grouped_parameter_mae,
    inverse_model_loss,
    LOSS_PRESET_AUDIBLE,
    load_mel_tensor_npz,
    load_torch_checkpoint,
    parameter_mae_by_name,
    parameter_loss_weights,
    parameter_mse_torch,
    prediction_distribution_by_name,
    pooling_shape,
    prepare_model_arrays,
    predict_patch_from_audio,
    patch_from_model_vector,
    save_torch_checkpoint,
    split_tensor_dataset,
    split_tensor_dataset_with_benchmark,
    train_inverse_model,
    predict_normalized_vectors,
    select_torch_device,
    target_parameters_for_mode,
    waveform_accuracy_by_name,
    weighted_mse_loss,
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
        self.assertEqual(model.pooling_mode, DEFAULT_POOLING_MODE)

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

    def test_create_inverse_model_supports_named_model_sizes(self):
        small = create_inverse_model(model_size="small")
        medium = create_inverse_model(model_size="medium")
        large = create_inverse_model(model_size="large")

        self.assertEqual(small.model_size, DEFAULT_MODEL_SIZE)
        self.assertEqual(medium.model_size, "medium")
        self.assertEqual(large.model_size, "large")

    def test_create_inverse_model_supports_grouped_head_mode(self):
        model = create_inverse_model(head_mode="grouped")
        inputs = torch.zeros(
            2,
            1,
            DEFAULT_MEL_BINS,
            DEFAULT_MEL_TENSOR_FRAMES,
            dtype=torch.float32,
        )

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (2, len(VECTOR_PARAMETERS)))
        self.assertEqual(model.head_mode, "grouped")
        self.assertIn("filter", model.continuous_head_groups)

    def test_build_cnn_encoder_ends_with_pooling_layer(self):
        layers = build_cnn_encoder(input_channels=1, channels=(8, 16), pool_shape=(4, 4))

        self.assertEqual(layers[-1].__class__.__name__, "AdaptiveAvgPool2d")
        self.assertEqual(layers[-1].output_size, (4, 4))

    def test_pooling_shape_preserves_time_frequency_structure(self):
        self.assertEqual(pooling_shape("global"), (1, 1))
        self.assertEqual(pooling_shape("time_frequency"), (4, 4))

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

    def test_model_rejects_invalid_model_size(self):
        with self.assertRaises(ValueError):
            MelSpectrogramInverseModel(model_size="huge")

    def test_model_rejects_invalid_pooling_mode(self):
        with self.assertRaises(ValueError):
            MelSpectrogramInverseModel(pooling_mode="wide-open")

    def test_model_rejects_invalid_head_mode(self):
        with self.assertRaises(ValueError):
            MelSpectrogramInverseModel(head_mode="one-head-per-wish")

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

    def test_prepare_model_arrays_can_condition_on_pitch(self):
        features = np.zeros((3, 1, DEFAULT_MEL_BINS, 8), dtype=np.float32)
        targets = np.full((3, len(VECTOR_PARAMETERS)), 0.5, dtype=np.float32)
        targets[:, 0] = [0.1, 0.2, 0.3]

        prepared = prepare_model_arrays(
            features,
            targets,
            target_mode="pitch_conditioned_timbre",
        )

        self.assertEqual(prepared["features"].shape, (3, 2, DEFAULT_MEL_BINS, 8))
        self.assertEqual(prepared["targets"].shape[1], len(VECTOR_PARAMETERS) - 1)
        self.assertNotIn("freq", [parameter.name for parameter in prepared["parameters"]])
        self.assertTrue(np.all(prepared["features"][:, 1, :, :] >= 0.1))

    def test_prepare_model_arrays_can_use_oscillator_mix_targets(self):
        features = np.zeros((1, 1, DEFAULT_MEL_BINS, 8), dtype=np.float32)
        targets = np.full((1, len(VECTOR_PARAMETERS)), 0.5, dtype=np.float32)
        names = [parameter.name for parameter in VECTOR_PARAMETERS]
        targets[0, names.index("osc1_wave")] = 0.5
        targets[0, names.index("osc1_level")] = 0.25
        targets[0, names.index("osc2_wave")] = 0.0
        targets[0, names.index("osc2_level")] = 0.75
        targets[0, names.index("osc2_detune")] = 0.25

        prepared = prepare_model_arrays(
            features,
            targets,
            target_mode=TARGET_MODE_OSCILLATOR_MIX,
        )
        parameter_names = [parameter.name for parameter in prepared["parameters"]]

        self.assertEqual(prepared["features"].shape, (1, 2, DEFAULT_MEL_BINS, 8))
        self.assertIn("osc_total_level", parameter_names)
        self.assertIn("osc_balance", parameter_names)
        self.assertNotIn("freq", parameter_names)
        self.assertNotIn("osc1_level", parameter_names)
        self.assertNotIn("osc2_level", parameter_names)
        self.assertAlmostEqual(
            prepared["targets"][0, parameter_names.index("osc_total_level")],
            0.5,
        )
        self.assertAlmostEqual(
            prepared["targets"][0, parameter_names.index("osc_balance")],
            0.25,
        )

    def test_patch_from_model_vector_reconstructs_oscillator_mix_levels(self):
        parameters = target_parameters_for_mode(TARGET_MODE_OSCILLATOR_MIX)
        values = {parameter.name: 0.5 for parameter in parameters}
        values["osc_total_level"] = 0.5
        values["osc_balance"] = 0.25
        vector = [values[parameter.name] for parameter in parameters]

        patch = patch_from_model_vector(vector, parameters, freq=440.0)

        self.assertEqual(patch["freq"], 440.0)
        self.assertAlmostEqual(patch["osc1_level"], 0.75)
        self.assertAlmostEqual(patch["osc2_level"], 0.25)

    def test_prepare_model_arrays_can_use_main_detuned_mix_targets(self):
        features = np.zeros((1, 1, DEFAULT_MEL_BINS, 8), dtype=np.float32)
        targets = np.full((1, len(VECTOR_PARAMETERS)), 0.5, dtype=np.float32)
        names = [parameter.name for parameter in VECTOR_PARAMETERS]
        targets[0, names.index("osc1_wave")] = 0.75
        targets[0, names.index("osc1_level")] = 0.25
        targets[0, names.index("osc2_wave")] = 0.0
        targets[0, names.index("osc2_level")] = 0.75
        targets[0, names.index("osc2_detune")] = 0.25

        prepared = prepare_model_arrays(
            features,
            targets,
            target_mode=TARGET_MODE_MAIN_DETUNED_MIX,
        )
        parameter_names = [parameter.name for parameter in prepared["parameters"]]

        self.assertEqual(prepared["features"].shape, (1, 2, DEFAULT_MEL_BINS, 8))
        self.assertEqual(
            parameter_names[:6],
            [
                "length",
                "main_wave",
                "osc_total_level",
                "detuned_balance",
                "detuned_wave",
                "detune_amount",
            ],
        )
        self.assertNotIn("freq", parameter_names)
        self.assertNotIn("osc1_level", parameter_names)
        self.assertNotIn("osc2_level", parameter_names)
        self.assertAlmostEqual(
            prepared["targets"][0, parameter_names.index("osc_total_level")],
            0.5,
        )
        self.assertAlmostEqual(
            prepared["targets"][0, parameter_names.index("detuned_balance")],
            0.75,
        )
        self.assertAlmostEqual(
            prepared["targets"][0, parameter_names.index("main_wave")],
            0.75,
        )
        self.assertAlmostEqual(
            prepared["targets"][0, parameter_names.index("detuned_wave")],
            0.0,
        )

    def test_patch_from_model_vector_reconstructs_main_detuned_mix(self):
        parameters = target_parameters_for_mode(TARGET_MODE_MAIN_DETUNED_MIX)
        values = {parameter.name: 0.5 for parameter in parameters}
        values["main_wave"] = 0.75
        values["detuned_wave"] = 0.0
        values["osc_total_level"] = 0.5
        values["detuned_balance"] = 0.25
        values["detune_amount"] = 0.25
        vector = [values[parameter.name] for parameter in parameters]

        patch = patch_from_model_vector(vector, parameters, freq=440.0)

        self.assertEqual(patch["freq"], 440.0)
        self.assertEqual(patch["osc1_wave"], "square")
        self.assertEqual(patch["osc2_wave"], "sine")
        self.assertAlmostEqual(patch["osc1_level"], 0.75)
        self.assertAlmostEqual(patch["osc2_level"], 0.25)
        self.assertAlmostEqual(patch["osc2_detune"], -600.0)

    def test_audible_loss_weights_loud_detuned_wave_more_than_quiet(self):
        parameters = target_parameters_for_mode(TARGET_MODE_MAIN_DETUNED_MIX)
        model = create_inverse_model(
            output_dim=len(parameters),
            input_channels=2,
            waveform_mode="classification",
            parameters=parameters,
        )
        model.eval()
        features = torch.zeros(2, 2, DEFAULT_MEL_BINS, 8)
        targets = torch.full((2, len(parameters)), 0.5)
        names = [parameter.name for parameter in parameters]
        targets[:, names.index("osc_total_level")] = 0.5
        targets[:, names.index("detuned_wave")] = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            model.waveform_heads["detuned_wave"].bias.copy_(
                torch.tensor([10.0, -10.0, -10.0, -10.0, -10.0])
            )
        targets[:, names.index("detuned_balance")] = torch.tensor([0.95, 0.05])
        quiet_wrong_loss = inverse_model_loss(
            model,
            model(features),
            targets,
            features=features,
            loss_preset=LOSS_PRESET_AUDIBLE,
        )

        targets[:, names.index("detuned_balance")] = torch.tensor([0.05, 0.95])
        loud_wrong_loss = inverse_model_loss(
            model,
            model(features),
            targets,
            features=features,
            loss_preset=LOSS_PRESET_AUDIBLE,
        )

        self.assertGreater(loud_wrong_loss.item(), quiet_wrong_loss.item())

    def test_parameter_mae_by_name_reports_each_target(self):
        targets = np.zeros((2, len(VECTOR_PARAMETERS)), dtype=np.float32)
        predictions = np.full((2, len(VECTOR_PARAMETERS)), 0.25, dtype=np.float32)

        metrics = parameter_mae_by_name(predictions, targets)

        self.assertEqual(set(metrics), {parameter.name for parameter in VECTOR_PARAMETERS})
        self.assertAlmostEqual(metrics["freq"], 0.25)

    def test_prediction_distribution_by_name_reports_mean_collapse(self):
        targets = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        predictions = np.asarray(
            [
                [0.5, 0.25],
                [0.5, 0.75],
            ],
            dtype=np.float32,
        )

        metrics = prediction_distribution_by_name(
            predictions,
            targets,
            parameters=VECTOR_PARAMETERS[:2],
        )

        self.assertEqual(metrics["freq"]["predicted_std"], 0.0)
        self.assertLess(metrics["freq"]["std_ratio"], 0.1)
        self.assertGreater(metrics["length"]["std_ratio"], 0.0)

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

    def test_parameter_loss_weights_support_audibility_preset(self):
        weights = parameter_loss_weights(preset="audibility")
        names = [parameter.name for parameter in VECTOR_PARAMETERS]

        self.assertEqual(len(weights), len(VECTOR_PARAMETERS))
        self.assertEqual(float(weights[names.index("freq")]), 0.0)
        self.assertGreater(float(weights[names.index("cutoff")]), 1.0)

    def test_parameter_loss_weights_support_hybrid_preset(self):
        weights = parameter_loss_weights(preset="hybrid")
        names = [parameter.name for parameter in VECTOR_PARAMETERS]

        self.assertEqual(len(weights), len(VECTOR_PARAMETERS))
        self.assertEqual(float(weights[names.index("freq")]), 0.0)
        self.assertGreater(float(weights[names.index("release")]), float(weights[names.index("cutoff")]))
        self.assertGreater(float(weights[names.index("osc2_detune")]), float(weights[names.index("length")]))

    def test_parameter_loss_weights_support_groupbalanced_preset(self):
        weights = parameter_loss_weights(preset="groupbalanced")

        self.assertEqual(len(weights), len(VECTOR_PARAMETERS))
        self.assertTrue(torch.all(weights == 1.0))

    def test_weighted_mse_loss_applies_per_parameter_weights(self):
        predictions = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.zeros_like(predictions)
        weights = torch.tensor([2.0, 0.0], dtype=torch.float32)

        loss = weighted_mse_loss(predictions, targets, weights)

        self.assertAlmostEqual(float(loss), 1.0)

    def test_create_optimizer_supports_adamw(self):
        model = create_inverse_model()

        optimizer = create_optimizer(
            model,
            optimizer_name="adamw",
            learning_rate=0.001,
            weight_decay=0.01,
        )

        self.assertEqual(optimizer.__class__.__name__, "AdamW")
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_create_scheduler_supports_step_scheduler(self):
        model = create_inverse_model()
        optimizer = create_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            scheduler_name="step",
            step_size=2,
            gamma=0.5,
        )

        self.assertEqual(scheduler.step_size, 2)

    def test_dataset_loss_torch_returns_average_loss(self):
        model = create_inverse_model()
        features = np.zeros(
            (2, 1, DEFAULT_MEL_BINS, DEFAULT_MEL_TENSOR_FRAMES),
            dtype=np.float32,
        )
        targets = np.zeros((2, len(VECTOR_PARAMETERS)), dtype=np.float32)

        loss = dataset_loss_torch(
            model,
            features,
            targets,
            device=torch.device("cpu"),
        )

        self.assertGreaterEqual(loss, 0.0)

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
        self.assertEqual(metrics["target_mode"], DEFAULT_TARGET_MODE)
        self.assertEqual(metrics["model_size"], DEFAULT_MODEL_SIZE)
        self.assertEqual(metrics["pooling_mode"], DEFAULT_POOLING_MODE)
        self.assertEqual(metrics["head_mode"], DEFAULT_HEAD_MODE)
        self.assertEqual(metrics["loss_preset"], "flat")
        self.assertIn("cutoff", metrics["loss_weights"])
        self.assertEqual(metrics["optimizer"], "adam")
        self.assertEqual(metrics["scheduler"], "none")
        self.assertEqual(metrics["checkpoint_selection"], "best_validation")
        self.assertEqual(metrics["completed_epochs"], 1)
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertEqual(len(metrics["test_losses"]), 1)
        self.assertEqual(len(metrics["test_objective_losses"]), 1)
        self.assertGreaterEqual(metrics["train_loss"], 0.0)
        self.assertGreaterEqual(metrics["train_objective_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_parameter_mse"], 0.0)
        self.assertGreaterEqual(metrics["test_objective_loss"], 0.0)
        self.assertGreaterEqual(metrics["test_mae"], 0.0)
        self.assertIn("freq", metrics["test_per_parameter_mae"])
        self.assertIn("freq", metrics["test_prediction_distribution"])
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
        self.assertIn("benchmark_parameter_mse", metrics)
        self.assertIn("benchmark_objective_loss", metrics)
        self.assertIn("benchmark_per_parameter_mae", metrics)
        self.assertIn("benchmark_grouped_mae", metrics)

    def test_train_inverse_model_supports_pitch_conditioned_timbre_mode(self):
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
                target_mode="pitch_conditioned_timbre",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["target_mode"], "pitch_conditioned_timbre")
        self.assertEqual(metrics["num_features"][0], 2)
        self.assertNotIn("freq", metrics["target_parameters"])
        self.assertNotIn("freq", metrics["test_per_parameter_mae"])

    def test_train_inverse_model_supports_audibility_loss_preset(self):
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
                loss_preset="audibility",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["loss_preset"], "audibility")
        self.assertEqual(metrics["loss_weights"]["freq"], 0.0)
        self.assertGreater(metrics["loss_weights"]["cutoff"], 1.0)

    def test_train_inverse_model_supports_hybrid_loss_preset(self):
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
                loss_preset="hybrid",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["loss_preset"], "hybrid")
        self.assertGreater(metrics["loss_weights"]["release"], metrics["loss_weights"]["cutoff"])
        self.assertGreater(metrics["loss_weights"]["osc2_detune"], metrics["loss_weights"]["length"])

    def test_train_inverse_model_supports_grouped_head_and_groupbalanced_loss(self):
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
                head_mode="grouped",
                loss_preset="groupbalanced",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["head_mode"], "grouped")
        self.assertEqual(metrics["loss_preset"], "groupbalanced")
        self.assertIn("filter", metrics["loss_groups"])
        self.assertIn("cutoff", metrics["test_prediction_distribution"])

    def test_train_inverse_model_supports_optimizer_controls(self):
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
                epochs=2,
                batch_size=2,
                random_state=1,
                optimizer_name="adamw",
                weight_decay=0.01,
                scheduler_name="step",
                scheduler_step_size=1,
                scheduler_gamma=0.5,
                early_stopping_patience=1,
                checkpoint_selection="best_validation",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["optimizer"], "adamw")
        self.assertEqual(metrics["weight_decay"], 0.01)
        self.assertEqual(metrics["scheduler"], "step")
        self.assertLessEqual(metrics["completed_epochs"], 2)
        self.assertGreaterEqual(metrics["best_test_loss"], 0.0)

    def test_train_inverse_model_supports_model_size(self):
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
                model_size="medium",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["model_size"], "medium")

    def test_train_inverse_model_supports_pooling_mode(self):
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
                pooling_mode="global",
                device=torch.device("cpu"),
            )

        metrics = result["metrics"]

        self.assertEqual(metrics["pooling_mode"], "global")

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
        self.assertEqual(checkpoint["model"].head_mode, DEFAULT_HEAD_MODE)
        self.assertEqual(predictions.shape, (1, len(VECTOR_PARAMETERS)))

    def test_save_and_load_torch_checkpoint_preserves_grouped_head_mode(self):
        model = create_inverse_model(head_mode="grouped")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            saved_path = save_torch_checkpoint(model, path)
            checkpoint = load_torch_checkpoint(saved_path, device=torch.device("cpu"))

        self.assertEqual(checkpoint["model"].head_mode, "grouped")

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

    def test_predict_patch_from_audio_accepts_pitch_context(self):
        model = create_inverse_model(
            output_dim=len(VECTOR_PARAMETERS) - 1,
            input_channels=2,
            parameters=target_parameters_for_mode("pitch_conditioned_timbre"),
        )
        audio = render_patch(length=1.0, freq=440.0)

        patch = predict_patch_from_audio(
            model,
            audio,
            44100,
            device=torch.device("cpu"),
            freq=440.0,
        )

        self.assertEqual(patch["freq"], 440.0)
        self.assertIn("osc1_wave", patch)

    def test_predict_patch_from_audio_requires_pitch_context_when_needed(self):
        model = create_inverse_model(
            output_dim=len(VECTOR_PARAMETERS) - 1,
            input_channels=2,
            parameters=target_parameters_for_mode("pitch_conditioned_timbre"),
        )
        audio = render_patch(length=1.0, freq=440.0)

        with self.assertRaises(ValueError):
            predict_patch_from_audio(
                model,
                audio,
                44100,
                device=torch.device("cpu"),
            )

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
