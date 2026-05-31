"""PyTorch inverse model for mel spectrogram to synth parameter prediction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES, mel_tensor_from_audio
from minisynth.randomize import constrain_envelope_fits_length
from minisynth.schema import SynthConfig, VECTOR_PARAMETERS, categorical_values

DEFAULT_MEL_BINS = 64
DEFAULT_INPUT_CHANNELS = 1
DEFAULT_OUTPUT_DIM = len(VECTOR_PARAMETERS)
DEFAULT_TORCH_TENSOR_PATH = Path("data/generated/d2/features/mel_tensors.npz")
DEFAULT_TORCH_MODEL_PATH = Path("models/v3_pytorch_cnn_500seeds.pt")
DEFAULT_TORCH_METRICS_PATH = Path("runs/training/v3_pytorch_cnn_500seeds_metrics.json")
DEFAULT_TORCH_MODEL_ID = "v3_pytorch_cnn_500seeds"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TEST_SIZE = 0.2
DEFAULT_BENCHMARK_SIZE = 0.0
WAVEFORM_MODE_SCALAR = "scalar_regression"
WAVEFORM_MODE_CLASSIFICATION = "classification"
DEFAULT_WAVEFORM_MODE = WAVEFORM_MODE_CLASSIFICATION
TARGET_MODE_FULL = "full"
TARGET_MODE_PITCH_CONDITIONED_TIMBRE = "pitch_conditioned_timbre"
DEFAULT_TARGET_MODE = TARGET_MODE_FULL
LOSS_PRESET_FLAT = "flat"
LOSS_PRESET_AUDIBILITY = "audibility"
DEFAULT_LOSS_PRESET = LOSS_PRESET_FLAT
PARAMETER_METRIC_GROUPS = {
    "global": ("freq", "length"),
    "pitch": ("freq",),
    "duration": ("length",),
    "timbre": (
        "length",
        "osc1_wave",
        "osc1_level",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
        "cutoff",
        "resonance",
        "attack",
        "decay",
        "sustain",
        "release",
    ),
    "pitch_conditioned_timbre": (
        "length",
        "osc1_wave",
        "osc1_level",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
        "cutoff",
        "resonance",
        "attack",
        "decay",
        "sustain",
        "release",
    ),
    "oscillator": (
        "osc1_wave",
        "osc1_level",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
    ),
    "filter": ("cutoff", "resonance"),
    "adsr": ("attack", "decay", "sustain", "release"),
}
AUDIBILITY_PARAMETER_WEIGHTS = {
    "freq": 0.0,
    "length": 0.75,
    "osc1_wave": 2.0,
    "osc1_level": 1.25,
    "osc2_wave": 2.0,
    "osc2_level": 1.0,
    "osc2_detune": 1.5,
    "cutoff": 2.0,
    "resonance": 1.25,
    "attack": 1.5,
    "decay": 1.25,
    "sustain": 1.0,
    "release": 1.25,
}


def parameters_from_names(names):
    by_name = {parameter.name: parameter for parameter in VECTOR_PARAMETERS}
    return tuple(by_name[name] for name in names)


def target_parameters_for_mode(target_mode=DEFAULT_TARGET_MODE):
    if target_mode == TARGET_MODE_FULL:
        return VECTOR_PARAMETERS
    if target_mode == TARGET_MODE_PITCH_CONDITIONED_TIMBRE:
        return tuple(parameter for parameter in VECTOR_PARAMETERS if parameter.name != "freq")

    raise ValueError(f"Unsupported target mode: {target_mode}")


def parameter_index(name, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    for index, parameter in enumerate(parameters):
        if parameter.name == name:
            return index
    raise ValueError(f"Unknown parameter: {name}")


def enum_parameter_indices(parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    return [
        index
        for index, parameter in enumerate(parameters)
        if parameter.kind == "enum"
    ]


def continuous_parameter_indices(parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    return [
        index
        for index, parameter in enumerate(parameters)
        if parameter.kind != "enum"
    ]


def select_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class MelSpectrogramInverseModel(nn.Module):
    """Small CNN that predicts normalized SynthConfig vectors from mel tensors."""

    def __init__(
        self,
        output_dim=DEFAULT_OUTPUT_DIM,
        input_channels=DEFAULT_INPUT_CHANNELS,
        waveform_mode=DEFAULT_WAVEFORM_MODE,
        parameters=None,
    ):
        super().__init__()
        if parameters is None:
            parameters = VECTOR_PARAMETERS
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1")
        if output_dim != len(parameters):
            raise ValueError("output_dim must match parameter count")
        if input_channels < 1:
            raise ValueError("input_channels must be at least 1")
        if waveform_mode not in (WAVEFORM_MODE_SCALAR, WAVEFORM_MODE_CLASSIFICATION):
            raise ValueError(f"Unsupported waveform mode: {waveform_mode}")

        self.output_dim = output_dim
        self.input_channels = input_channels
        self.waveform_mode = waveform_mode
        self.parameter_names = tuple(parameter.name for parameter in parameters)
        self.parameters_schema = tuple(parameters)
        self.enum_indices = enum_parameter_indices(parameters)
        self.continuous_indices = continuous_parameter_indices(parameters)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        if waveform_mode == WAVEFORM_MODE_SCALAR:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.Sigmoid(),
            )
        else:
            self.shared_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.continuous_head = nn.Sequential(
                nn.Linear(64, len(self.continuous_indices)),
                nn.Sigmoid(),
            )
            self.waveform_heads = nn.ModuleDict(
                {
                    parameters[index].name: nn.Linear(
                        64,
                        len(categorical_values(parameters[index])),
                    )
                    for index in self.enum_indices
                }
            )

    def forward(self, mel_tensors):
        if mel_tensors.ndim != 4:
            raise ValueError("mel_tensors must have shape (batch, channels, mel_bins, frames)")
        if mel_tensors.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {mel_tensors.shape[1]}"
            )

        raw = self.raw_outputs(mel_tensors)
        if self.waveform_mode == WAVEFORM_MODE_SCALAR:
            return raw["vector"]

        return waveform_classification_outputs_to_vector(raw, parameters=self.parameters_schema)

    def raw_outputs(self, mel_tensors):
        if mel_tensors.ndim != 4:
            raise ValueError("mel_tensors must have shape (batch, channels, mel_bins, frames)")
        if mel_tensors.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {mel_tensors.shape[1]}"
            )

        encoded = self.encoder(mel_tensors)
        if self.waveform_mode == WAVEFORM_MODE_SCALAR:
            return {"vector": self.head(encoded)}

        shared = self.shared_head(encoded)
        return {
            "continuous": self.continuous_head(shared),
            "waveforms": {
                name: head(shared)
                for name, head in self.waveform_heads.items()
            },
        }


def create_inverse_model(
    output_dim=DEFAULT_OUTPUT_DIM,
    waveform_mode=DEFAULT_WAVEFORM_MODE,
    input_channels=DEFAULT_INPUT_CHANNELS,
    parameters=None,
):
    return MelSpectrogramInverseModel(
        output_dim=output_dim,
        waveform_mode=waveform_mode,
        input_channels=input_channels,
        parameters=parameters,
    )


def waveform_classification_outputs_to_vector(raw_outputs, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    continuous = raw_outputs["continuous"]
    batch_size = continuous.shape[0]
    vector = torch.zeros(
        batch_size,
        len(parameters),
        dtype=continuous.dtype,
        device=continuous.device,
    )
    continuous_position = 0

    for index, parameter in enumerate(parameters):
        if parameter.kind == "enum":
            logits = raw_outputs["waveforms"][parameter.name]
            probabilities = torch.softmax(logits, dim=1)
            class_values = torch.linspace(
                0.0,
                1.0,
                logits.shape[1],
                dtype=logits.dtype,
                device=logits.device,
            )
            vector[:, index] = torch.sum(probabilities * class_values, dim=1)
        else:
            vector[:, index] = continuous[:, continuous_position]
            continuous_position += 1

    return vector


def load_mel_tensor_npz(path=DEFAULT_TORCH_TENSOR_PATH):
    source = Path(path)
    data = np.load(source)
    features = data["features"].astype(np.float32)
    targets = data["targets"].astype(np.float32)

    if features.ndim != 4:
        raise ValueError("features must have shape (samples, channels, mel_bins, frames)")
    if targets.ndim != 2:
        raise ValueError("targets must have shape (samples, parameters)")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same number of samples")
    if targets.shape[1] != DEFAULT_OUTPUT_DIM:
        raise ValueError(f"Expected {DEFAULT_OUTPUT_DIM} targets, got {targets.shape[1]}")
    if len(features) < 2:
        raise ValueError("at least 2 samples are required")

    return {
        "features": features,
        "targets": targets,
        "metadata_path": str(data["metadata_path"]) if "metadata_path" in data.files else "",
        "frames": int(data["frames"]) if "frames" in data.files else features.shape[-1],
    }


def split_tensor_dataset(features, targets, test_size=DEFAULT_TEST_SIZE, random_state=0):
    return split_tensor_dataset_with_benchmark(
        features,
        targets,
        test_size=test_size,
        benchmark_size=0.0,
        random_state=random_state,
    )


def split_tensor_dataset_with_benchmark(
    features,
    targets,
    test_size=DEFAULT_TEST_SIZE,
    benchmark_size=DEFAULT_BENCHMARK_SIZE,
    random_state=0,
):
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if benchmark_size < 0.0 or benchmark_size >= 1.0:
        raise ValueError("benchmark_size must be in [0, 1)")
    if test_size + benchmark_size >= 1.0:
        raise ValueError("test_size + benchmark_size must leave training samples")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    sample_count = len(x)
    test_count = max(1, int(round(sample_count * test_size)))
    benchmark_count = int(round(sample_count * benchmark_size))
    train_count = sample_count - test_count - benchmark_count

    if train_count < 1:
        raise ValueError("split must leave at least 1 training sample")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(sample_count)
    test_indices = indices[:test_count]
    benchmark_indices = indices[test_count : test_count + benchmark_count]
    train_indices = indices[test_count + benchmark_count :]

    return {
        "train_features": x[train_indices],
        "train_targets": y[train_indices],
        "test_features": x[test_indices],
        "test_targets": y[test_indices],
        "benchmark_features": x[benchmark_indices],
        "benchmark_targets": y[benchmark_indices],
        "train_indices": train_indices,
        "test_indices": test_indices,
        "benchmark_indices": benchmark_indices,
    }


def add_pitch_context_channel(features, full_targets):
    x = np.asarray(features, dtype=np.float32)
    targets = np.asarray(full_targets, dtype=np.float32)
    freq_index = parameter_index("freq")
    pitch_context = targets[:, freq_index].reshape(len(targets), 1, 1, 1)
    pitch_context = np.broadcast_to(
        pitch_context,
        (len(targets), 1, x.shape[2], x.shape[3]),
    ).astype(np.float32)
    return np.concatenate([x, pitch_context], axis=1)


def targets_for_parameters(full_targets, parameters):
    source = np.asarray(full_targets, dtype=np.float32)
    indices = [parameter_index(parameter.name) for parameter in parameters]
    return source[:, indices]


def prepare_model_arrays(features, targets, target_mode=DEFAULT_TARGET_MODE):
    parameters = target_parameters_for_mode(target_mode)
    x = np.asarray(features, dtype=np.float32)
    if target_mode == TARGET_MODE_PITCH_CONDITIONED_TIMBRE:
        x = add_pitch_context_channel(x, targets)

    return {
        "features": x,
        "targets": targets_for_parameters(targets, parameters),
        "parameters": parameters,
    }


def tensor_loader(features, targets, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parameter_loss_weights(parameters=None, preset=DEFAULT_LOSS_PRESET):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    if preset == LOSS_PRESET_FLAT:
        return torch.ones(len(parameters), dtype=torch.float32)
    if preset == LOSS_PRESET_AUDIBILITY:
        weights = [
            AUDIBILITY_PARAMETER_WEIGHTS.get(parameter.name, 1.0)
            for parameter in parameters
        ]
        values = torch.as_tensor(weights, dtype=torch.float32)
        if torch.sum(values) <= 0:
            raise ValueError("loss weights must include at least one positive value")
        return values * (len(values) / torch.sum(values))

    raise ValueError(f"Unsupported loss preset: {preset}")


def weighted_mse_loss(predictions, targets, weights):
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if weights.ndim != 1 or weights.shape[0] != predictions.shape[1]:
        raise ValueError("weights must have one value per target parameter")

    batch_weights = weights.to(device=predictions.device, dtype=predictions.dtype)
    return torch.mean(torch.square(predictions - targets) * batch_weights)


def waveform_classification_loss(model, batch_features, batch_targets, loss_weights=None):
    raw = model.raw_outputs(batch_features)
    if loss_weights is None:
        loss_weights = torch.ones(
            len(model.parameters_schema),
            dtype=batch_targets.dtype,
            device=batch_targets.device,
        )
    else:
        loss_weights = loss_weights.to(device=batch_targets.device, dtype=batch_targets.dtype)

    continuous_loss = weighted_mse_loss(
        raw["continuous"],
        batch_targets[:, model.continuous_indices],
        loss_weights[model.continuous_indices],
    )
    waveform_losses = []
    for index in model.enum_indices:
        parameter = model.parameters_schema[index]
        target_classes = categorical_target_classes(batch_targets[:, index], parameter)
        waveform_losses.append(
            nn.functional.cross_entropy(
                raw["waveforms"][parameter.name],
                target_classes,
            )
            * loss_weights[index]
        )

    if not waveform_losses:
        return continuous_loss

    return continuous_loss + torch.stack(waveform_losses).mean()


def categorical_target_classes(values, parameter):
    choices = categorical_values(parameter)
    scaled = values * (len(choices) - 1)
    return torch.clamp(torch.round(scaled), 0, len(choices) - 1).long()


def inverse_model_loss(
    model,
    predictions,
    targets,
    features=None,
    loss_function=None,
    loss_weights=None,
):
    if model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION:
        if features is None:
            raise ValueError("features are required for waveform classification loss")
        return waveform_classification_loss(
            model,
            features,
            targets,
            loss_weights=loss_weights,
        )

    if loss_function is None:
        loss_function = nn.MSELoss()
    if loss_weights is not None:
        return weighted_mse_loss(predictions, targets, loss_weights)
    return loss_function(predictions, targets)


def parameter_mae_torch(model, features, targets, device=None, batch_size=DEFAULT_BATCH_SIZE):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    loader = tensor_loader(features, targets, batch_size=batch_size, shuffle=False)
    total_absolute_error = 0.0
    total_values = 0

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            total_absolute_error += torch.abs(predictions - batch_targets).sum().item()
            total_values += batch_targets.numel()

    return float(total_absolute_error / total_values)


def predict_dataset_torch(model, features, device=None, batch_size=DEFAULT_BATCH_SIZE):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    loader = tensor_loader(
        features,
        np.zeros((len(features), getattr(model, "output_dim", DEFAULT_OUTPUT_DIM)), dtype=np.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []

    with torch.no_grad():
        for batch_features, _ in loader:
            batch_features = batch_features.to(device)
            predictions.append(model(batch_features).cpu().numpy())

    return np.vstack(predictions).astype(np.float32)


def parameter_mae_by_name(predictions, targets, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    predicted = np.asarray(predictions, dtype=np.float32)
    expected = np.asarray(targets, dtype=np.float32)
    if predicted.shape != expected.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predicted.ndim != 2:
        raise ValueError("predictions and targets must be 2D")
    if predicted.shape[1] != len(parameters):
        raise ValueError(f"Expected {len(parameters)} parameters, got {predicted.shape[1]}")

    absolute_errors = np.abs(predicted - expected)
    return {
        parameter.name: float(absolute_errors[:, index].mean())
        for index, parameter in enumerate(parameters)
    }


def continuous_parameter_mae(predictions, targets, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    continuous_indices = [
        index
        for index, parameter in enumerate(parameters)
        if parameter.kind != "enum"
    ]
    if not continuous_indices:
        return 0.0

    predicted = np.asarray(predictions, dtype=np.float32)
    expected = np.asarray(targets, dtype=np.float32)
    return float(np.abs(predicted[:, continuous_indices] - expected[:, continuous_indices]).mean())


def grouped_parameter_mae(predictions, targets, groups=None, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    if groups is None:
        groups = PARAMETER_METRIC_GROUPS

    predicted = np.asarray(predictions, dtype=np.float32)
    expected = np.asarray(targets, dtype=np.float32)
    if predicted.shape != expected.shape:
        raise ValueError("predictions and targets must have the same shape")

    parameter_indices = {
        parameter.name: index
        for index, parameter in enumerate(parameters)
    }
    metrics = {}

    for group_name, parameter_names in groups.items():
        indices = [
            parameter_indices[name]
            for name in parameter_names
            if name in parameter_indices
        ]
        if not indices:
            continue
        metrics[group_name] = float(np.abs(predicted[:, indices] - expected[:, indices]).mean())

    return metrics


def categorical_predictions(values, parameter):
    choices = categorical_values(parameter)
    scaled = np.asarray(values, dtype=np.float32) * (len(choices) - 1)
    return np.clip(np.rint(scaled), 0, len(choices) - 1).astype(np.int64)


def waveform_accuracy_by_name(predictions, targets, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    predicted = np.asarray(predictions, dtype=np.float32)
    expected = np.asarray(targets, dtype=np.float32)
    accuracies = {}

    for index, parameter in enumerate(parameters):
        if parameter.kind != "enum":
            continue
        predicted_classes = categorical_predictions(predicted[:, index], parameter)
        expected_classes = categorical_predictions(expected[:, index], parameter)
        accuracies[parameter.name] = float(np.mean(predicted_classes == expected_classes))

    return accuracies


def waveform_accuracy(predictions, targets, parameters=None):
    accuracies = waveform_accuracy_by_name(predictions, targets, parameters=parameters)
    if not accuracies:
        return 0.0
    return float(np.mean(list(accuracies.values())))


def parameter_metrics_torch(
    model,
    features,
    targets,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    parameters=None,
):
    if parameters is None:
        parameters = getattr(model, "parameters_schema", VECTOR_PARAMETERS)
    predictions = predict_dataset_torch(
        model,
        features,
        device=device,
        batch_size=batch_size,
    )
    return {
        "mae": float(np.abs(predictions - targets).mean()),
        "continuous_mae": continuous_parameter_mae(predictions, targets, parameters=parameters),
        "grouped_mae": grouped_parameter_mae(predictions, targets, parameters=parameters),
        "per_parameter_mae": parameter_mae_by_name(predictions, targets, parameters=parameters),
        "waveform_accuracy": waveform_accuracy(predictions, targets, parameters=parameters),
        "waveform_accuracy_by_name": waveform_accuracy_by_name(
            predictions,
            targets,
            parameters=parameters,
        ),
    }


def parameter_mse_torch(model, features, targets, device=None, batch_size=DEFAULT_BATCH_SIZE):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    loader = tensor_loader(features, targets, batch_size=batch_size, shuffle=False)
    total_squared_error = 0.0
    total_values = 0

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            total_squared_error += torch.square(predictions - batch_targets).sum().item()
            total_values += batch_targets.numel()

    return float(total_squared_error / total_values)


def train_inverse_model(
    tensor_path=DEFAULT_TORCH_TENSOR_PATH,
    model_id=DEFAULT_TORCH_MODEL_ID,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    test_size=DEFAULT_TEST_SIZE,
    benchmark_size=DEFAULT_BENCHMARK_SIZE,
    waveform_mode=DEFAULT_WAVEFORM_MODE,
    target_mode=DEFAULT_TARGET_MODE,
    loss_preset=DEFAULT_LOSS_PRESET,
    random_state=0,
    device=None,
    progress=False,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if device is None:
        device = select_torch_device()
    else:
        device = torch.device(device)

    torch.manual_seed(random_state)
    dataset = load_mel_tensor_npz(tensor_path)
    prepared = prepare_model_arrays(
        dataset["features"],
        dataset["targets"],
        target_mode=target_mode,
    )
    target_parameters = prepared["parameters"]
    split = split_tensor_dataset_with_benchmark(
        prepared["features"],
        prepared["targets"],
        test_size=test_size,
        benchmark_size=benchmark_size,
        random_state=random_state,
    )
    model = create_inverse_model(
        output_dim=prepared["targets"].shape[1],
        waveform_mode=waveform_mode,
        input_channels=prepared["features"].shape[1],
        parameters=target_parameters,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    loss_weights = parameter_loss_weights(
        target_parameters,
        preset=loss_preset,
    ).to(device)

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        loader = tensor_loader(
            split["train_features"],
            split["train_targets"],
            batch_size=batch_size,
            shuffle=True,
        )
        total_batches = len(loader)
        if progress:
            print(f"Epoch {epoch + 1}/{epochs} starting on {device.type}.")
        for batch_index, (batch_features, batch_targets) in enumerate(loader, start=1):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = inverse_model_loss(
                model,
                predictions,
                batch_targets,
                features=batch_features,
                loss_function=loss_function,
                loss_weights=loss_weights,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_features)
            sample_count += len(batch_features)
            if progress:
                print(
                    f"\r  Batch {batch_index}/{total_batches} - loss {loss.item():.6f}",
                    end="",
                    flush=True,
                )

        epoch_loss = float(running_loss / sample_count)
        epoch_losses.append(epoch_loss)
        if progress:
            print(f"\nEpoch {epoch + 1} complete - average loss {epoch_loss:.6f}\n")

    train_parameter_metrics = parameter_metrics_torch(
        model,
        split["train_features"],
        split["train_targets"],
        device=device,
        batch_size=batch_size,
    )
    test_parameter_metrics = parameter_metrics_torch(
        model,
        split["test_features"],
        split["test_targets"],
        device=device,
        batch_size=batch_size,
    )

    metrics = {
        "model_id": model_id,
        "model_type": "pytorch_cnn",
        "waveform_mode": waveform_mode,
        "target_mode": target_mode,
        "loss_preset": loss_preset,
        "loss_weights": {
            parameter.name: float(weight)
            for parameter, weight in zip(target_parameters, loss_weights.detach().cpu().numpy())
        },
        "tensor_path": str(tensor_path),
        "metadata_path": dataset["metadata_path"],
        "num_samples": int(len(dataset["features"])),
        "num_features": list(prepared["features"].shape[1:]),
        "num_targets": int(prepared["targets"].shape[1]),
        "target_parameters": [parameter.name for parameter in target_parameters],
        "train_samples": int(len(split["train_features"])),
        "test_samples": int(len(split["test_features"])),
        "benchmark_samples": int(len(split["benchmark_features"])),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "test_size": float(test_size),
        "benchmark_size": float(benchmark_size),
        "device": device.type,
        "train_loss": epoch_losses[-1],
        "train_losses": epoch_losses,
        "test_loss": parameter_mse_torch(
            model,
            split["test_features"],
            split["test_targets"],
            device=device,
            batch_size=batch_size,
        ),
        "train_mae": train_parameter_metrics["mae"],
        "test_mae": test_parameter_metrics["mae"],
        "train_continuous_mae": train_parameter_metrics["continuous_mae"],
        "test_continuous_mae": test_parameter_metrics["continuous_mae"],
        "train_per_parameter_mae": train_parameter_metrics["per_parameter_mae"],
        "test_per_parameter_mae": test_parameter_metrics["per_parameter_mae"],
        "train_grouped_mae": train_parameter_metrics["grouped_mae"],
        "test_grouped_mae": test_parameter_metrics["grouped_mae"],
        "train_waveform_accuracy": train_parameter_metrics["waveform_accuracy"],
        "test_waveform_accuracy": test_parameter_metrics["waveform_accuracy"],
        "train_waveform_accuracy_by_name": train_parameter_metrics["waveform_accuracy_by_name"],
        "test_waveform_accuracy_by_name": test_parameter_metrics["waveform_accuracy_by_name"],
        "train_indices": split["train_indices"].astype(int).tolist(),
        "test_indices": split["test_indices"].astype(int).tolist(),
        "benchmark_indices": split["benchmark_indices"].astype(int).tolist(),
    }

    if len(split["benchmark_features"]) > 0:
        benchmark_parameter_metrics = parameter_metrics_torch(
            model,
            split["benchmark_features"],
            split["benchmark_targets"],
            device=device,
            batch_size=batch_size,
        )
        metrics.update(
            {
                "benchmark_loss": parameter_mse_torch(
                    model,
                    split["benchmark_features"],
                    split["benchmark_targets"],
                    device=device,
                    batch_size=batch_size,
                ),
                "benchmark_mae": benchmark_parameter_metrics["mae"],
                "benchmark_continuous_mae": benchmark_parameter_metrics["continuous_mae"],
                "benchmark_per_parameter_mae": benchmark_parameter_metrics["per_parameter_mae"],
                "benchmark_grouped_mae": benchmark_parameter_metrics["grouped_mae"],
                "benchmark_waveform_accuracy": benchmark_parameter_metrics["waveform_accuracy"],
                "benchmark_waveform_accuracy_by_name": benchmark_parameter_metrics[
                    "waveform_accuracy_by_name"
                ],
            }
        )

    return {
        "model": model,
        "metrics": metrics,
    }


def save_torch_checkpoint(model, path=DEFAULT_TORCH_MODEL_PATH, metrics=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "output_dim": getattr(model, "output_dim", DEFAULT_OUTPUT_DIM),
            "input_channels": getattr(model, "input_channels", DEFAULT_INPUT_CHANNELS),
            "waveform_mode": getattr(model, "waveform_mode", WAVEFORM_MODE_SCALAR),
            "parameter_names": getattr(
                model,
                "parameter_names",
                tuple(parameter.name for parameter in VECTOR_PARAMETERS),
            ),
        },
        destination,
    )
    return destination


def load_torch_checkpoint(path=DEFAULT_TORCH_MODEL_PATH, device=None):
    if device is None:
        device = select_torch_device()
    else:
        device = torch.device(device)

    checkpoint = torch.load(Path(path), map_location=device)
    parameter_names = checkpoint.get(
        "parameter_names",
        tuple(parameter.name for parameter in VECTOR_PARAMETERS),
    )
    parameters = parameters_from_names(parameter_names)
    model = MelSpectrogramInverseModel(
        output_dim=checkpoint.get("output_dim", len(parameters)),
        input_channels=checkpoint.get("input_channels", DEFAULT_INPUT_CHANNELS),
        waveform_mode=checkpoint.get("waveform_mode", WAVEFORM_MODE_SCALAR),
        parameters=parameters,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "metrics": checkpoint.get("metrics", {}),
    }


def predict_normalized_vectors(model, mel_tensors, device=None):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    inputs = torch.as_tensor(mel_tensors, dtype=torch.float32, device=device)

    with torch.no_grad():
        predictions = model(inputs)

    return predictions.cpu().numpy()


def predict_patch_from_audio(model, audio, sample_rate, device=None, frames=DEFAULT_MEL_TENSOR_FRAMES):
    mel_tensor = mel_tensor_from_audio(audio, sample_rate, frames=frames)
    vector = tuple(
        float(value)
        for value in predict_normalized_vectors(
            model,
            mel_tensor[np.newaxis, :, :, :],
            device=device,
        )[0]
    )
    patch = SynthConfig.from_vector(vector).to_render_kwargs()
    constrain_envelope_fits_length(patch)
    return patch


def expected_mel_tensor_shape(
    mel_bins=DEFAULT_MEL_BINS,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
):
    return (DEFAULT_INPUT_CHANNELS, mel_bins, frames)
