"""PyTorch inverse model for mel spectrogram to synth parameter prediction."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES, mel_tensor_from_audio
from minisynth.randomize import constrain_envelope_fits_length
from minisynth.oscillator_mix import (
    canonicalize_oscillator_slots,
    oscillator_balance,
    oscillator_total_level,
)
from minisynth.schema import (
    Parameter,
    SynthConfig,
    VECTOR_PARAMETERS,
    categorical_values,
    config_from_vector,
    denormalize_parameter_value,
    normalize_parameter_value,
)

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
TARGET_MODE_OSCILLATOR_MIX = "oscillator_mix"
TARGET_MODE_MAIN_DETUNED_MIX = "main_detuned_mix"
DEFAULT_TARGET_MODE = TARGET_MODE_FULL
LOSS_PRESET_FLAT = "flat"
LOSS_PRESET_AUDIBILITY = "audibility"
LOSS_PRESET_HYBRID = "hybrid"
LOSS_PRESET_GROUP_BALANCED = "groupbalanced"
LOSS_PRESET_AUDIBLE = "audible"
LOSS_PRESET_NOISE_DETUNE = "noise_detune"
DEFAULT_LOSS_PRESET = LOSS_PRESET_FLAT
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAMW = "adamw"
DEFAULT_OPTIMIZER = OPTIMIZER_ADAM
SCHEDULER_NONE = "none"
SCHEDULER_STEP = "step"
DEFAULT_SCHEDULER = SCHEDULER_NONE
CHECKPOINT_FINAL = "final"
CHECKPOINT_BEST_VALIDATION = "best_validation"
DEFAULT_CHECKPOINT_SELECTION = CHECKPOINT_BEST_VALIDATION
MODEL_SIZE_SMALL = "small"
MODEL_SIZE_MEDIUM = "medium"
MODEL_SIZE_LARGE = "large"
DEFAULT_MODEL_SIZE = MODEL_SIZE_SMALL
POOLING_GLOBAL = "global"
POOLING_TIME_FREQUENCY = "time_frequency"
DEFAULT_POOLING_MODE = POOLING_TIME_FREQUENCY
HEAD_MODE_SHARED = "shared"
HEAD_MODE_GROUPED = "grouped"
DEFAULT_HEAD_MODE = HEAD_MODE_SHARED
OSC_TOTAL_LEVEL_PARAMETER = Parameter(
    "osc_total_level",
    "float",
    0.0,
    2.0,
    1.0,
    "linear",
    "oscillator",
)
OSC_BALANCE_PARAMETER = Parameter(
    "osc_balance",
    "float",
    0.0,
    1.0,
    0.5,
    "linear",
    "oscillator",
)
DETUNED_BALANCE_PARAMETER = Parameter(
    "detuned_balance",
    "float",
    0.0,
    1.0,
    0.5,
    "linear",
    "oscillator",
)
MAIN_WAVE_PARAMETER = Parameter(
    "main_wave",
    "enum",
    None,
    None,
    "saw",
    "categorical",
    "oscillator",
)
DETUNED_WAVE_PARAMETER = Parameter(
    "detuned_wave",
    "enum",
    None,
    None,
    "saw",
    "categorical",
    "oscillator",
)
DETUNE_AMOUNT_PARAMETER = Parameter(
    "detune_amount",
    "float",
    -1200.0,
    1200.0,
    7,
    "linear",
    "oscillator",
)
OSCILLATOR_MIX_PARAMETERS = tuple(
    parameter
    for parameter in VECTOR_PARAMETERS
    if parameter.name not in ("freq", "osc1_level", "osc2_level")
)
OSCILLATOR_MIX_PARAMETERS = (
    OSCILLATOR_MIX_PARAMETERS[:2]
    + (OSC_TOTAL_LEVEL_PARAMETER, OSC_BALANCE_PARAMETER)
    + OSCILLATOR_MIX_PARAMETERS[2:]
)
MAIN_DETUNED_MIX_PARAMETERS = (
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "length"),
    MAIN_WAVE_PARAMETER,
    OSC_TOTAL_LEVEL_PARAMETER,
    DETUNED_BALANCE_PARAMETER,
    DETUNED_WAVE_PARAMETER,
    DETUNE_AMOUNT_PARAMETER,
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "cutoff"),
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "resonance"),
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "attack"),
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "decay"),
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "sustain"),
    next(parameter for parameter in VECTOR_PARAMETERS if parameter.name == "release"),
)
MODEL_SIZE_SPECS = {
    MODEL_SIZE_SMALL: {
        "channels": (16, 32, 64),
        "hidden_dim": 64,
    },
    MODEL_SIZE_MEDIUM: {
        "channels": (32, 64, 128),
        "hidden_dim": 128,
    },
    MODEL_SIZE_LARGE: {
        "channels": (32, 64, 128, 256),
        "hidden_dim": 256,
    },
}
PARAMETER_METRIC_GROUPS = {
    "global": ("freq", "length"),
    "pitch": ("freq",),
    "duration": ("length",),
    "timbre": (
        "length",
        "osc1_wave",
        "osc1_level",
        "osc_total_level",
        "osc_balance",
        "main_wave",
        "detuned_wave",
        "detuned_balance",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
        "detune_amount",
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
        "osc_total_level",
        "osc_balance",
        "main_wave",
        "detuned_wave",
        "detuned_balance",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
        "detune_amount",
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
        "osc_total_level",
        "osc_balance",
        "main_wave",
        "detuned_wave",
        "detuned_balance",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
        "detune_amount",
    ),
    "filter": ("cutoff", "resonance"),
    "adsr": ("attack", "decay", "sustain", "release"),
}
CONTINUOUS_HEAD_GROUPS = {
    "duration": ("length",),
    "pitch": ("freq",),
    "oscillator": (
        "osc1_level",
        "osc_total_level",
        "osc_balance",
        "detuned_balance",
        "osc2_level",
        "osc2_detune",
        "detune_amount",
    ),
    "filter": ("cutoff", "resonance"),
    "adsr": ("attack", "decay", "sustain", "release"),
}
GROUP_BALANCED_LOSS_GROUPS = {
    "duration": ("length",),
    "pitch": ("freq",),
    "oscillator": (
        "osc1_wave",
        "osc1_level",
        "osc_total_level",
        "osc_balance",
        "main_wave",
        "detuned_wave",
        "detuned_balance",
        "osc2_wave",
        "osc2_level",
        "osc2_detune",
        "detune_amount",
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
HYBRID_PARAMETER_WEIGHTS = {
    "freq": 0.0,
    "length": 0.9,
    "osc1_wave": 1.65,
    "osc1_level": 1.4,
    "osc2_wave": 1.65,
    "osc2_level": 1.4,
    "osc2_detune": 1.65,
    "cutoff": 1.2,
    "resonance": 1.55,
    "attack": 1.3,
    "decay": 1.3,
    "sustain": 1.55,
    "release": 1.75,
}


def parameters_from_names(names):
    by_name = {
        parameter.name: parameter
        for parameter in (
            tuple(VECTOR_PARAMETERS)
            + tuple(OSCILLATOR_MIX_PARAMETERS)
            + tuple(MAIN_DETUNED_MIX_PARAMETERS)
        )
    }
    return tuple(by_name[name] for name in names)


def target_parameters_for_mode(target_mode=DEFAULT_TARGET_MODE):
    if target_mode == TARGET_MODE_FULL:
        return VECTOR_PARAMETERS
    if target_mode == TARGET_MODE_PITCH_CONDITIONED_TIMBRE:
        return tuple(parameter for parameter in VECTOR_PARAMETERS if parameter.name != "freq")
    if target_mode == TARGET_MODE_OSCILLATOR_MIX:
        return OSCILLATOR_MIX_PARAMETERS
    if target_mode == TARGET_MODE_MAIN_DETUNED_MIX:
        return MAIN_DETUNED_MIX_PARAMETERS

    raise ValueError(f"Unsupported target mode: {target_mode}")


def parameter_index(name, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    for index, parameter in enumerate(parameters):
        if parameter.name == name:
            return index
    raise ValueError(f"Unknown parameter: {name}")


def parameter_by_name(name, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    for parameter in parameters:
        if parameter.name == name:
            return parameter
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


def continuous_head_groups(parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    parameter_names = {parameter.name for parameter in parameters}
    continuous_positions = {
        parameter.name: position
        for position, parameter in enumerate(
            parameter for parameter in parameters if parameter.kind != "enum"
        )
    }
    groups = {}
    assigned = set()

    for group_name, group_parameters in CONTINUOUS_HEAD_GROUPS.items():
        positions = [
            continuous_positions[name]
            for name in group_parameters
            if name in parameter_names and name in continuous_positions
        ]
        if positions:
            groups[group_name] = positions
            assigned.update(positions)

    remaining = [
        position
        for position in range(len(continuous_positions))
        if position not in assigned
    ]
    if remaining:
        groups["other"] = remaining

    return groups


def loss_groups_for_parameters(parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS
    parameter_names = {parameter.name for parameter in parameters}
    return {
        group_name: [
            name
            for name in group_parameters
            if name in parameter_names
        ]
        for group_name, group_parameters in GROUP_BALANCED_LOSS_GROUPS.items()
        if any(name in parameter_names for name in group_parameters)
    }


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
        model_size=DEFAULT_MODEL_SIZE,
        pooling_mode=DEFAULT_POOLING_MODE,
        head_mode=DEFAULT_HEAD_MODE,
    ):
        super().__init__()
        if parameters is None:
            parameters = VECTOR_PARAMETERS
        if model_size not in MODEL_SIZE_SPECS:
            raise ValueError(f"Unsupported model size: {model_size}")
        if pooling_mode not in (POOLING_GLOBAL, POOLING_TIME_FREQUENCY):
            raise ValueError(f"Unsupported pooling mode: {pooling_mode}")
        if head_mode not in (HEAD_MODE_SHARED, HEAD_MODE_GROUPED):
            raise ValueError(f"Unsupported head mode: {head_mode}")
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
        self.model_size = model_size
        self.pooling_mode = pooling_mode
        self.head_mode = head_mode
        self.parameter_names = tuple(parameter.name for parameter in parameters)
        self.parameters_schema = tuple(parameters)
        self.enum_indices = enum_parameter_indices(parameters)
        self.continuous_indices = continuous_parameter_indices(parameters)
        self.continuous_head_groups = continuous_head_groups(parameters)
        spec = MODEL_SIZE_SPECS[model_size]
        channels = spec["channels"]
        hidden_dim = spec["hidden_dim"]
        pool_shape = pooling_shape(pooling_mode)
        self.encoder = nn.Sequential(
            *build_cnn_encoder(
                input_channels,
                channels,
                pool_shape=pool_shape,
            )
        )
        encoder_dim = channels[-1] * pool_shape[0] * pool_shape[1]
        if waveform_mode == WAVEFORM_MODE_SCALAR:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )
        else:
            self.shared_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
            )
            if head_mode == HEAD_MODE_GROUPED:
                self.grouped_continuous_heads = nn.ModuleDict(
                    {
                        group_name: nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, len(positions)),
                            nn.Sigmoid(),
                        )
                        for group_name, positions in self.continuous_head_groups.items()
                    }
                )
            else:
                self.continuous_head = nn.Sequential(
                    nn.Linear(hidden_dim, len(self.continuous_indices)),
                    nn.Sigmoid(),
                )
            self.waveform_heads = nn.ModuleDict(
                {
                    parameters[index].name: nn.Linear(
                        hidden_dim,
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
        if self.head_mode == HEAD_MODE_GROUPED:
            continuous = grouped_continuous_outputs_to_tensor(
                self.grouped_continuous_heads,
                self.continuous_head_groups,
                shared,
            )
        else:
            continuous = self.continuous_head(shared)

        return {
            "continuous": continuous,
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
    model_size=DEFAULT_MODEL_SIZE,
    pooling_mode=DEFAULT_POOLING_MODE,
    head_mode=DEFAULT_HEAD_MODE,
):
    return MelSpectrogramInverseModel(
        output_dim=output_dim,
        waveform_mode=waveform_mode,
        input_channels=input_channels,
        parameters=parameters,
        model_size=model_size,
        pooling_mode=pooling_mode,
        head_mode=head_mode,
    )


def pooling_shape(pooling_mode=DEFAULT_POOLING_MODE):
    if pooling_mode == POOLING_GLOBAL:
        return (1, 1)
    if pooling_mode == POOLING_TIME_FREQUENCY:
        return (4, 4)

    raise ValueError(f"Unsupported pooling mode: {pooling_mode}")


def build_cnn_encoder(input_channels, channels, pool_shape=(1, 1)):
    layers = []
    current_channels = input_channels
    for index, output_channels in enumerate(channels):
        layers.extend(
            [
                nn.Conv2d(current_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            ]
        )
        if index < len(channels) - 1:
            layers.append(nn.MaxPool2d(kernel_size=2))
        current_channels = output_channels

    layers.append(nn.AdaptiveAvgPool2d(pool_shape))
    return layers


def grouped_continuous_outputs_to_tensor(heads, groups, shared_features):
    outputs = [None] * sum(len(positions) for positions in groups.values())
    for group_name, positions in groups.items():
        values = heads[group_name](shared_features)
        for local_index, output_position in enumerate(positions):
            outputs[output_position] = values[:, local_index : local_index + 1]

    if any(output is None for output in outputs):
        raise ValueError("grouped continuous heads did not produce every output")

    return torch.cat(outputs, dim=1)


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


def add_pitch_context_to_mel_tensor(mel_tensor, freq):
    if freq is None:
        raise ValueError("freq is required for pitch-conditioned model prediction")
    normalized_freq = normalize_parameter_value(parameter_by_name("freq"), float(freq))
    x = np.asarray(mel_tensor, dtype=np.float32)
    pitch_context = np.full(
        (1, x.shape[1], x.shape[2]),
        normalized_freq,
        dtype=np.float32,
    )
    return np.concatenate([x, pitch_context], axis=0)


def targets_for_parameters(full_targets, parameters):
    source = np.asarray(full_targets, dtype=np.float32)
    indices = [parameter_index(parameter.name) for parameter in parameters]
    return source[:, indices]


def oscillator_mix_targets(full_targets):
    source = np.asarray(full_targets, dtype=np.float32)
    rows = []
    for row in source:
        patch = config_from_vector(row, parameters=VECTOR_PARAMETERS).to_render_kwargs()
        patch = canonicalize_oscillator_slots(patch)
        values = []
        for parameter in OSCILLATOR_MIX_PARAMETERS:
            if parameter.name == "osc_total_level":
                values.append(oscillator_total_level(patch) / 2.0)
            elif parameter.name == "osc_balance":
                values.append(oscillator_balance(patch))
            else:
                values.append(normalize_parameter_value(parameter, patch[parameter.name]))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def main_detuned_mix_targets(full_targets):
    source = np.asarray(full_targets, dtype=np.float32)
    rows = []
    for row in source:
        patch = config_from_vector(row, parameters=VECTOR_PARAMETERS).to_render_kwargs()
        values = []
        for parameter in MAIN_DETUNED_MIX_PARAMETERS:
            if parameter.name == "main_wave":
                values.append(normalize_parameter_value(parameter, patch["osc1_wave"]))
            elif parameter.name == "detuned_wave":
                values.append(normalize_parameter_value(parameter, patch["osc2_wave"]))
            elif parameter.name == "osc_total_level":
                values.append(oscillator_total_level(patch) / 2.0)
            elif parameter.name == "detuned_balance":
                values.append(oscillator_balance(patch))
            elif parameter.name == "detune_amount":
                values.append(normalize_parameter_value(parameter, patch["osc2_detune"]))
            else:
                values.append(normalize_parameter_value(parameter, patch[parameter.name]))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def prepare_model_arrays(features, targets, target_mode=DEFAULT_TARGET_MODE):
    parameters = target_parameters_for_mode(target_mode)
    x = np.asarray(features, dtype=np.float32)
    if target_mode in (
        TARGET_MODE_PITCH_CONDITIONED_TIMBRE,
        TARGET_MODE_OSCILLATOR_MIX,
        TARGET_MODE_MAIN_DETUNED_MIX,
    ):
        x = add_pitch_context_channel(x, targets)
    if target_mode == TARGET_MODE_OSCILLATOR_MIX:
        target_values = oscillator_mix_targets(targets)
    elif target_mode == TARGET_MODE_MAIN_DETUNED_MIX:
        target_values = main_detuned_mix_targets(targets)
    else:
        target_values = targets_for_parameters(targets, parameters)

    return {
        "features": x,
        "targets": target_values,
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
    if preset == LOSS_PRESET_HYBRID:
        weights = [
            HYBRID_PARAMETER_WEIGHTS.get(parameter.name, 1.0)
            for parameter in parameters
        ]
        values = torch.as_tensor(weights, dtype=torch.float32)
        if torch.sum(values) <= 0:
            raise ValueError("loss weights must include at least one positive value")
        return values * (len(values) / torch.sum(values))
    if preset in (LOSS_PRESET_GROUP_BALANCED, LOSS_PRESET_AUDIBLE, LOSS_PRESET_NOISE_DETUNE):
        return torch.ones(len(parameters), dtype=torch.float32)

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


def group_balanced_loss(model, predictions, targets, features=None):
    if model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION:
        if features is None:
            raise ValueError("features are required for group-balanced classification loss")
        raw = model.raw_outputs(features)
    else:
        raw = None

    parameter_indices = {
        parameter.name: index
        for index, parameter in enumerate(model.parameters_schema)
    }
    continuous_positions = {
        parameter.name: position
        for position, parameter in enumerate(
            parameter for parameter in model.parameters_schema if parameter.kind != "enum"
        )
    }
    group_losses = []

    for group_parameters in GROUP_BALANCED_LOSS_GROUPS.values():
        component_losses = []
        for name in group_parameters:
            if name not in parameter_indices:
                continue

            index = parameter_indices[name]
            parameter = model.parameters_schema[index]
            if model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION and parameter.kind == "enum":
                target_classes = categorical_target_classes(targets[:, index], parameter)
                class_count = len(categorical_values(parameter))
                component_losses.append(
                    nn.functional.cross_entropy(
                        raw["waveforms"][parameter.name],
                        target_classes,
                    )
                    / np.log(class_count)
                )
            elif model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION:
                position = continuous_positions[name]
                component_losses.append(
                    torch.mean(torch.square(raw["continuous"][:, position] - targets[:, index]))
                )
            else:
                component_losses.append(
                    torch.mean(torch.square(predictions[:, index] - targets[:, index]))
                )

        if component_losses:
            group_losses.append(torch.stack(component_losses).mean())

    if not group_losses:
        return torch.mean(torch.square(predictions - targets))

    return torch.stack(group_losses).mean()


def _parameter_indices_by_name(parameters):
    return {
        parameter.name: index
        for index, parameter in enumerate(parameters)
    }


def _continuous_positions_by_name(parameters):
    return {
        parameter.name: position
        for position, parameter in enumerate(
            parameter for parameter in parameters if parameter.kind != "enum"
        )
    }


def _normalized_audibility_weights(levels, floor=0.25, scale=1.75):
    weights = floor + scale * torch.clamp(levels, 0.0, 1.0)
    return weights / torch.clamp(torch.mean(weights.detach()), min=1.0e-6)


def _weighted_sample_mean(losses, weights):
    return torch.mean(losses * weights.to(device=losses.device, dtype=losses.dtype))


def _target_is_noise_wave(targets, parameter_indices, parameters, name):
    if name not in parameter_indices:
        return None
    parameter = parameters[parameter_indices[name]]
    choices = categorical_values(parameter)
    noise_index = choices.index("noise")
    target_classes = categorical_target_classes(targets[:, parameter_indices[name]], parameter)
    return target_classes == noise_index


def _boost_noise_wave_weights(base_weights, noise_mask, boost=1.75):
    if noise_mask is None:
        return base_weights
    boosted = base_weights * torch.where(
        noise_mask.to(device=base_weights.device),
        torch.as_tensor(boost, dtype=base_weights.dtype, device=base_weights.device),
        torch.as_tensor(1.0, dtype=base_weights.dtype, device=base_weights.device),
    )
    return boosted / torch.clamp(torch.mean(boosted.detach()), min=1.0e-6)


def _suppress_noise_detune_weights(base_weights, noise_mask):
    if noise_mask is None:
        return base_weights
    weights = base_weights * (~noise_mask).to(device=base_weights.device, dtype=base_weights.dtype)
    if torch.sum(weights.detach()) <= 0.0:
        return weights
    return weights / torch.clamp(torch.mean(weights.detach()), min=1.0e-6)


def audible_main_detuned_loss(model, predictions, targets, features=None, noise_aware_detune=False):
    required = {"main_wave", "detuned_wave", "osc_total_level", "detuned_balance", "detune_amount"}
    parameter_indices = _parameter_indices_by_name(model.parameters_schema)
    if not required.issubset(parameter_indices):
        return group_balanced_loss(
            model,
            predictions,
            targets,
            features=features,
        )
    if model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION:
        if features is None:
            raise ValueError("features are required for audible classification loss")
        raw = model.raw_outputs(features)
    else:
        raw = None

    continuous_positions = _continuous_positions_by_name(model.parameters_schema)
    total_index = parameter_indices["osc_total_level"]
    balance_index = parameter_indices["detuned_balance"]
    total_level = torch.clamp(targets[:, total_index] * 2.0, 0.0, 2.0)
    detuned_balance = torch.clamp(targets[:, balance_index], 0.0, 1.0)
    main_level = torch.clamp(total_level * (1.0 - detuned_balance), 0.0, 1.0)
    detuned_level = torch.clamp(total_level * detuned_balance, 0.0, 1.0)
    total_audibility = _normalized_audibility_weights(torch.clamp(total_level / 2.0, 0.0, 1.0))
    main_audibility = _normalized_audibility_weights(main_level)
    detuned_audibility = _normalized_audibility_weights(detuned_level)
    quiet_total_weight = _normalized_audibility_weights(1.0 - torch.clamp(targets[:, total_index], 0.0, 1.0))
    main_noise_mask = _target_is_noise_wave(
        targets,
        parameter_indices,
        model.parameters_schema,
        "main_wave",
    )
    detuned_noise_mask = _target_is_noise_wave(
        targets,
        parameter_indices,
        model.parameters_schema,
        "detuned_wave",
    )
    if noise_aware_detune:
        main_wave_weights = _boost_noise_wave_weights(main_audibility, main_noise_mask)
        detuned_wave_weights = _boost_noise_wave_weights(detuned_audibility, detuned_noise_mask)
        detune_weights = _suppress_noise_detune_weights(detuned_audibility, detuned_noise_mask)
    else:
        main_wave_weights = main_audibility
        detuned_wave_weights = detuned_audibility
        detune_weights = detuned_audibility

    group_losses = []
    for group_parameters in GROUP_BALANCED_LOSS_GROUPS.values():
        component_losses = []
        for name in group_parameters:
            if name not in parameter_indices:
                continue

            index = parameter_indices[name]
            parameter = model.parameters_schema[index]
            if model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION and parameter.kind == "enum":
                target_classes = categorical_target_classes(targets[:, index], parameter)
                class_count = len(categorical_values(parameter))
                sample_losses = nn.functional.cross_entropy(
                    raw["waveforms"][parameter.name],
                    target_classes,
                    reduction="none",
                ) / np.log(class_count)
                if name == "main_wave":
                    component_losses.append(_weighted_sample_mean(sample_losses, main_wave_weights))
                elif name == "detuned_wave":
                    component_losses.append(_weighted_sample_mean(sample_losses, detuned_wave_weights))
                else:
                    component_losses.append(torch.mean(sample_losses))
            elif model.waveform_mode == WAVEFORM_MODE_CLASSIFICATION:
                position = continuous_positions[name]
                errors = torch.square(raw["continuous"][:, position] - targets[:, index])
                if name == "osc_total_level":
                    overshoot = torch.clamp(raw["continuous"][:, position] - targets[:, index], min=0.0)
                    component_losses.append(
                        torch.mean(errors)
                        + _weighted_sample_mean(torch.square(overshoot), quiet_total_weight)
                    )
                elif name == "detuned_balance":
                    component_losses.append(_weighted_sample_mean(errors, total_audibility))
                elif name == "detune_amount":
                    component_losses.append(_weighted_sample_mean(errors, detune_weights))
                else:
                    component_losses.append(torch.mean(errors))
            else:
                errors = torch.square(predictions[:, index] - targets[:, index])
                component_losses.append(torch.mean(errors))

        if component_losses:
            group_losses.append(torch.stack(component_losses).mean())

    if not group_losses:
        return torch.mean(torch.square(predictions - targets))

    return torch.stack(group_losses).mean()


def inverse_model_loss(
    model,
    predictions,
    targets,
    features=None,
    loss_function=None,
    loss_weights=None,
    loss_preset=DEFAULT_LOSS_PRESET,
):
    if loss_preset == LOSS_PRESET_GROUP_BALANCED:
        return group_balanced_loss(
            model,
            predictions,
            targets,
            features=features,
        )
    if loss_preset == LOSS_PRESET_AUDIBLE:
        return audible_main_detuned_loss(
            model,
            predictions,
            targets,
            features=features,
        )
    if loss_preset == LOSS_PRESET_NOISE_DETUNE:
        return audible_main_detuned_loss(
            model,
            predictions,
            targets,
            features=features,
            noise_aware_detune=True,
        )

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


def create_optimizer(model, optimizer_name=DEFAULT_OPTIMIZER, learning_rate=DEFAULT_LEARNING_RATE, weight_decay=0.0):
    if optimizer_name == OPTIMIZER_ADAM:
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == OPTIMIZER_ADAMW:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name=DEFAULT_SCHEDULER, step_size=10, gamma=0.5):
    if scheduler_name == SCHEDULER_NONE:
        return None
    if scheduler_name == SCHEDULER_STEP:
        if step_size < 1:
            raise ValueError("scheduler step_size must be at least 1")
        if gamma <= 0.0:
            raise ValueError("scheduler gamma must be positive")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def dataset_loss_torch(
    model,
    features,
    targets,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    loss_weights=None,
    loss_preset=DEFAULT_LOSS_PRESET,
):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    loader = tensor_loader(features, targets, batch_size=batch_size, shuffle=False)
    loss_function = nn.MSELoss()
    running_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = inverse_model_loss(
                model,
                predictions,
                batch_targets,
                features=batch_features,
                loss_function=loss_function,
                loss_weights=loss_weights,
                loss_preset=loss_preset,
            )
            running_loss += loss.item() * len(batch_features)
            sample_count += len(batch_features)

    return float(running_loss / sample_count)


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


def prediction_distribution_by_name(predictions, targets, parameters=None):
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

    metrics = {}
    for index, parameter in enumerate(parameters):
        target_values = expected[:, index]
        predicted_values = predicted[:, index]
        target_std = float(np.std(target_values))
        predicted_std = float(np.std(predicted_values))
        metrics[parameter.name] = {
            "target_mean": float(np.mean(target_values)),
            "predicted_mean": float(np.mean(predicted_values)),
            "mean_delta": float(np.mean(predicted_values) - np.mean(target_values)),
            "target_std": target_std,
            "predicted_std": predicted_std,
            "std_ratio": float(predicted_std / target_std) if target_std > 0.0 else 0.0,
            "target_min": float(np.min(target_values)),
            "target_max": float(np.max(target_values)),
            "predicted_min": float(np.min(predicted_values)),
            "predicted_max": float(np.max(predicted_values)),
        }

    return metrics


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
        "prediction_distribution": prediction_distribution_by_name(
            predictions,
            targets,
            parameters=parameters,
        ),
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



def tensor_shard_paths(path):
    """Return sorted mel tensor shard paths for a directory or shard-like pattern."""
    source = Path(path)
    if source.is_dir():
        shard_paths = sorted(source.glob("mel_tensors_part_*.npz"))
        if not shard_paths:
            raise ValueError(f"No mel_tensors_part_*.npz shards found in {source}")
        return shard_paths
    return []


def is_sharded_tensor_path(path):
    source = Path(path)
    return source.is_dir()


def _npz_scalar(data, name, default=None):
    if name not in data.files:
        return default
    value = data[name]
    if getattr(value, "shape", ()) == ():
        return value.item()
    return value


def load_mel_tensor_shard_source(path):
    """Inspect a directory of mel_tensors_part_*.npz shards without loading all features."""
    shard_paths = tensor_shard_paths(path)
    counts = []
    metadata_path = ""
    frames = DEFAULT_MEL_TENSOR_FRAMES
    feature_shape = None
    target_dim = None

    for shard_index, shard_path in enumerate(shard_paths):
        with np.load(shard_path) as data:
            if "features" not in data.files or "targets" not in data.files:
                raise ValueError(f"Shard missing features/targets: {shard_path}")
            if "indices" in data.files:
                count = int(len(data["indices"]))
            else:
                count = int(data["targets"].shape[0])
            counts.append(count)

            if shard_index == 0:
                feature_shape = tuple(data["features"].shape[1:])
                target_dim = int(data["targets"].shape[1])
                if "metadata_path" in data.files:
                    metadata_path = str(data["metadata_path"])
                if "frames" in data.files:
                    frames = int(data["frames"])

    if target_dim != DEFAULT_OUTPUT_DIM:
        raise ValueError(f"Expected {DEFAULT_OUTPUT_DIM} targets, got {target_dim}")
    if not counts or sum(counts) < 2:
        raise ValueError("at least 2 samples are required")

    cumulative_counts = np.cumsum(np.asarray(counts, dtype=np.int64))
    return {
        "sharded": True,
        "path": str(path),
        "shard_paths": shard_paths,
        "counts": counts,
        "cumulative_counts": cumulative_counts,
        "sample_count": int(cumulative_counts[-1]),
        "feature_shape": feature_shape,
        "target_dim": target_dim,
        "metadata_path": metadata_path,
        "frames": frames,
    }


def split_sample_indices(
    sample_count,
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
        "train_indices": train_indices,
        "test_indices": test_indices,
        "benchmark_indices": benchmark_indices,
    }


def _shard_index_for_position(cumulative_counts, position):
    return int(np.searchsorted(cumulative_counts, int(position), side="right"))


def _shard_start(cumulative_counts, shard_index):
    if shard_index == 0:
        return 0
    return int(cumulative_counts[shard_index - 1])


def _group_global_indices_by_shard(source, indices, shuffle=False, rng=None):
    ordered_indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        ordered_indices = ordered_indices.copy()
        rng.shuffle(ordered_indices)

    groups = {}
    cumulative = source["cumulative_counts"]
    for global_index in ordered_indices:
        shard_index = _shard_index_for_position(cumulative, global_index)
        local_index = int(global_index) - _shard_start(cumulative, shard_index)
        groups.setdefault(shard_index, []).append(local_index)

    shard_order = list(groups.keys())
    if shuffle:
        rng.shuffle(shard_order)
    return shard_order, groups


def sharded_batch_count(source, sample_indices, batch_size=DEFAULT_BATCH_SIZE):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    _, groups = _group_global_indices_by_shard(
        source,
        sample_indices,
        shuffle=False,
        rng=np.random.default_rng(0),
    )
    return int(sum(np.ceil(len(local_indices) / batch_size) for local_indices in groups.values()))


def _load_tensor_shard_arrays(source, shard_index):
    shard_path = source["shard_paths"][shard_index]
    with np.load(shard_path) as data:
        features = data["features"].astype(np.float32)
        targets = data["targets"].astype(np.float32)

    if features.ndim != 4:
        raise ValueError(f"features must have shape (samples, channels, mel_bins, frames): {shard_path}")
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (samples, parameters): {shard_path}")
    if len(features) != len(targets):
        raise ValueError(f"features and targets must have the same number of samples: {shard_path}")
    if targets.shape[1] != DEFAULT_OUTPUT_DIM:
        raise ValueError(f"Expected {DEFAULT_OUTPUT_DIM} targets, got {targets.shape[1]}: {shard_path}")

    return features, targets


def iter_sharded_tensor_batches(
    source,
    sample_indices,
    batch_size=DEFAULT_BATCH_SIZE,
    target_mode=DEFAULT_TARGET_MODE,
    shuffle=True,
    random_state=0,
):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    rng = np.random.default_rng(random_state)
    shard_order, groups = _group_global_indices_by_shard(
        source,
        sample_indices,
        shuffle=shuffle,
        rng=rng,
    )

    for shard_index in shard_order:
        local_indices = np.asarray(groups[shard_index], dtype=np.int64)
        if shuffle:
            rng.shuffle(local_indices)

        shard_features, shard_targets = _load_tensor_shard_arrays(source, shard_index)
        selected_features = shard_features[local_indices]
        selected_targets = shard_targets[local_indices]
        prepared = prepare_model_arrays(
            selected_features,
            selected_targets,
            target_mode=target_mode,
        )

        order = np.arange(len(local_indices))
        if shuffle:
            rng.shuffle(order)

        for batch_start in range(0, len(order), batch_size):
            batch_order = order[batch_start : batch_start + batch_size]
            yield (
                torch.as_tensor(prepared["features"][batch_order], dtype=torch.float32),
                torch.as_tensor(prepared["targets"][batch_order], dtype=torch.float32),
            )


def sharded_sample_loss_torch(
    model,
    source,
    sample_indices,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    target_mode=DEFAULT_TARGET_MODE,
    loss_weights=None,
    loss_preset=DEFAULT_LOSS_PRESET,
):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    loss_function = nn.MSELoss()
    running_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_features, batch_targets in iter_sharded_tensor_batches(
            source,
            sample_indices,
            batch_size=batch_size,
            target_mode=target_mode,
            shuffle=False,
        ):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = inverse_model_loss(
                model,
                predictions,
                batch_targets,
                features=batch_features,
                loss_function=loss_function,
                loss_weights=loss_weights,
                loss_preset=loss_preset,
            )
            running_loss += loss.item() * len(batch_features)
            sample_count += len(batch_features)

    return float(running_loss / sample_count)


def sharded_parameter_mse_torch(
    model,
    source,
    sample_indices,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    target_mode=DEFAULT_TARGET_MODE,
):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    total_squared_error = 0.0
    total_values = 0

    with torch.no_grad():
        for batch_features, batch_targets in iter_sharded_tensor_batches(
            source,
            sample_indices,
            batch_size=batch_size,
            target_mode=target_mode,
            shuffle=False,
        ):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            total_squared_error += torch.square(predictions - batch_targets).sum().item()
            total_values += batch_targets.numel()

    return float(total_squared_error / total_values)


def sharded_parameter_metrics_torch(
    model,
    source,
    sample_indices,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    target_mode=DEFAULT_TARGET_MODE,
    parameters=None,
):
    if device is None:
        device = select_torch_device()
    if parameters is None:
        parameters = getattr(model, "parameters_schema", VECTOR_PARAMETERS)

    model = model.to(device)
    model.eval()

    parameter_count = len(parameters)
    total_samples = 0
    total_absolute_error = 0.0
    per_parameter_abs_sum = np.zeros(parameter_count, dtype=np.float64)
    pred_sum = np.zeros(parameter_count, dtype=np.float64)
    target_sum = np.zeros(parameter_count, dtype=np.float64)
    pred_sq_sum = np.zeros(parameter_count, dtype=np.float64)
    target_sq_sum = np.zeros(parameter_count, dtype=np.float64)
    pred_min = np.full(parameter_count, np.inf, dtype=np.float64)
    pred_max = np.full(parameter_count, -np.inf, dtype=np.float64)
    target_min = np.full(parameter_count, np.inf, dtype=np.float64)
    target_max = np.full(parameter_count, -np.inf, dtype=np.float64)
    waveform_correct = {
        parameter.name: 0
        for parameter in parameters
        if parameter.kind == "enum"
    }
    waveform_total = {
        parameter.name: 0
        for parameter in parameters
        if parameter.kind == "enum"
    }

    with torch.no_grad():
        for batch_features, batch_targets in iter_sharded_tensor_batches(
            source,
            sample_indices,
            batch_size=batch_size,
            target_mode=target_mode,
            shuffle=False,
        ):
            batch_features = batch_features.to(device)
            predictions = model(batch_features).cpu().numpy().astype(np.float32)
            targets = batch_targets.numpy().astype(np.float32)
            absolute_errors = np.abs(predictions - targets)

            batch_size_actual = len(predictions)
            total_samples += batch_size_actual
            total_absolute_error += float(absolute_errors.sum())
            per_parameter_abs_sum += absolute_errors.sum(axis=0)
            pred_sum += predictions.sum(axis=0)
            target_sum += targets.sum(axis=0)
            pred_sq_sum += np.square(predictions).sum(axis=0)
            target_sq_sum += np.square(targets).sum(axis=0)
            pred_min = np.minimum(pred_min, predictions.min(axis=0))
            pred_max = np.maximum(pred_max, predictions.max(axis=0))
            target_min = np.minimum(target_min, targets.min(axis=0))
            target_max = np.maximum(target_max, targets.max(axis=0))

            for index, parameter in enumerate(parameters):
                if parameter.kind != "enum":
                    continue
                predicted_classes = categorical_predictions(predictions[:, index], parameter)
                expected_classes = categorical_predictions(targets[:, index], parameter)
                waveform_correct[parameter.name] += int(np.sum(predicted_classes == expected_classes))
                waveform_total[parameter.name] += int(len(expected_classes))

    if total_samples < 1:
        raise ValueError("sample_indices did not contain any rows")

    per_parameter_mae_values = per_parameter_abs_sum / total_samples
    per_parameter_mae = {
        parameter.name: float(per_parameter_mae_values[index])
        for index, parameter in enumerate(parameters)
    }
    continuous_indices = [
        index
        for index, parameter in enumerate(parameters)
        if parameter.kind != "enum"
    ]
    if continuous_indices:
        continuous_mae = float(
            per_parameter_abs_sum[continuous_indices].sum()
            / (total_samples * len(continuous_indices))
        )
    else:
        continuous_mae = 0.0

    parameter_indices = {
        parameter.name: index
        for index, parameter in enumerate(parameters)
    }
    grouped_mae = {}
    for group_name, parameter_names in PARAMETER_METRIC_GROUPS.items():
        indices = [
            parameter_indices[name]
            for name in parameter_names
            if name in parameter_indices
        ]
        if not indices:
            continue
        grouped_mae[group_name] = float(
            per_parameter_abs_sum[indices].sum() / (total_samples * len(indices))
        )

    pred_mean = pred_sum / total_samples
    target_mean = target_sum / total_samples
    pred_var = np.maximum(pred_sq_sum / total_samples - np.square(pred_mean), 0.0)
    target_var = np.maximum(target_sq_sum / total_samples - np.square(target_mean), 0.0)
    pred_std = np.sqrt(pred_var)
    target_std = np.sqrt(target_var)
    prediction_distribution = {}
    for index, parameter in enumerate(parameters):
        prediction_distribution[parameter.name] = {
            "target_mean": float(target_mean[index]),
            "predicted_mean": float(pred_mean[index]),
            "mean_delta": float(pred_mean[index] - target_mean[index]),
            "target_std": float(target_std[index]),
            "predicted_std": float(pred_std[index]),
            "std_ratio": float(pred_std[index] / target_std[index]) if target_std[index] > 0.0 else 0.0,
            "target_min": float(target_min[index]),
            "target_max": float(target_max[index]),
            "predicted_min": float(pred_min[index]),
            "predicted_max": float(pred_max[index]),
        }

    waveform_accuracy_by_parameter = {
        name: float(waveform_correct[name] / waveform_total[name])
        for name in waveform_correct
        if waveform_total[name] > 0
    }
    waveform_acc = (
        float(np.mean(list(waveform_accuracy_by_parameter.values())))
        if waveform_accuracy_by_parameter
        else 0.0
    )

    return {
        "mae": float(total_absolute_error / (total_samples * parameter_count)),
        "continuous_mae": continuous_mae,
        "grouped_mae": grouped_mae,
        "per_parameter_mae": per_parameter_mae,
        "prediction_distribution": prediction_distribution,
        "waveform_accuracy": waveform_acc,
        "waveform_accuracy_by_name": waveform_accuracy_by_parameter,
    }


def train_inverse_model_sharded(
    tensor_path=DEFAULT_TORCH_TENSOR_PATH,
    validation_tensor_path=None,
    model_id=DEFAULT_TORCH_MODEL_ID,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    test_size=DEFAULT_TEST_SIZE,
    benchmark_size=DEFAULT_BENCHMARK_SIZE,
    waveform_mode=DEFAULT_WAVEFORM_MODE,
    target_mode=DEFAULT_TARGET_MODE,
    loss_preset=DEFAULT_LOSS_PRESET,
    optimizer_name=DEFAULT_OPTIMIZER,
    weight_decay=0.0,
    scheduler_name=DEFAULT_SCHEDULER,
    scheduler_step_size=10,
    scheduler_gamma=0.5,
    early_stopping_patience=0,
    checkpoint_selection=DEFAULT_CHECKPOINT_SELECTION,
    model_size=DEFAULT_MODEL_SIZE,
    pooling_mode=DEFAULT_POOLING_MODE,
    head_mode=DEFAULT_HEAD_MODE,
    random_state=0,
    device=None,
    progress=False,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if checkpoint_selection not in (CHECKPOINT_FINAL, CHECKPOINT_BEST_VALIDATION):
        raise ValueError(f"Unsupported checkpoint selection: {checkpoint_selection}")
    if head_mode not in (HEAD_MODE_SHARED, HEAD_MODE_GROUPED):
        raise ValueError(f"Unsupported head mode: {head_mode}")
    if device is None:
        device = select_torch_device()
    else:
        device = torch.device(device)

    torch.manual_seed(random_state)
    source = load_mel_tensor_shard_source(tensor_path)
    validation_source = None
    if validation_tensor_path is not None:
        validation_source = load_mel_tensor_shard_source(validation_tensor_path)
        if validation_source["feature_shape"] != source["feature_shape"]:
            raise ValueError("validation tensor feature shape must match the training tensor source")
        if validation_source["target_dim"] != source["target_dim"]:
            raise ValueError("validation tensor target dimension must match the training tensor source")
    target_parameters = target_parameters_for_mode(target_mode)
    input_channels = int(source["feature_shape"][0])
    if target_mode in (
        TARGET_MODE_PITCH_CONDITIONED_TIMBRE,
        TARGET_MODE_OSCILLATOR_MIX,
        TARGET_MODE_MAIN_DETUNED_MIX,
    ):
        input_channels += 1

    if validation_source is None:
        split = split_sample_indices(
            source["sample_count"],
            test_size=test_size,
            benchmark_size=benchmark_size,
            random_state=random_state,
        )
        test_source = source
    else:
        split = {
            "train_indices": np.arange(source["sample_count"], dtype=np.int64),
            "test_indices": np.arange(validation_source["sample_count"], dtype=np.int64),
            "benchmark_indices": np.asarray([], dtype=np.int64),
        }
        test_source = validation_source
    model = create_inverse_model(
        output_dim=len(target_parameters),
        waveform_mode=waveform_mode,
        input_channels=input_channels,
        parameters=target_parameters,
        model_size=model_size,
        pooling_mode=pooling_mode,
        head_mode=head_mode,
    ).to(device)
    optimizer = create_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )
    loss_function = nn.MSELoss()
    loss_weights = parameter_loss_weights(
        target_parameters,
        preset=loss_preset,
    ).to(device)

    epoch_losses = []
    test_losses = []
    learning_rates = []
    best_epoch = 0
    best_test_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    completed_epochs = 0
    total_train_batches = sharded_batch_count(
        source,
        split["train_indices"],
        batch_size=batch_size,
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        if progress:
            print(f"Epoch {epoch + 1}/{epochs} starting on {device.type}.")

        for batch_index, (batch_features, batch_targets) in enumerate(
            iter_sharded_tensor_batches(
                source,
                split["train_indices"],
                batch_size=batch_size,
                target_mode=target_mode,
                shuffle=True,
                random_state=random_state + epoch,
            ),
            start=1,
        ):
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
                loss_preset=loss_preset,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_features)
            sample_count += len(batch_features)
            if progress:
                print(
                    f"\r  Batch {batch_index}/{total_train_batches} - loss {loss.item():.6f}",
                    end="",
                    flush=True,
                )

        epoch_loss = float(running_loss / sample_count)
        epoch_losses.append(epoch_loss)
        completed_epochs = epoch + 1
        test_epoch_loss = sharded_sample_loss_torch(
            model,
            test_source,
            split["test_indices"],
            device=device,
            batch_size=batch_size,
            target_mode=target_mode,
            loss_weights=loss_weights,
            loss_preset=loss_preset,
        )
        test_losses.append(test_epoch_loss)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        if test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if progress:
            print(
                f"\nEpoch {epoch + 1} complete - "
                f"train loss {epoch_loss:.6f}, test loss {test_epoch_loss:.6f}\n"
            )
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            if progress:
                print(f"Early stopping after epoch {epoch + 1}.")
            break

    if checkpoint_selection == CHECKPOINT_BEST_VALIDATION:
        model.load_state_dict(best_state_dict)

    train_parameter_metrics = sharded_parameter_metrics_torch(
        model,
        source,
        split["train_indices"],
        device=device,
        batch_size=batch_size,
        target_mode=target_mode,
        parameters=target_parameters,
    )
    test_parameter_metrics = sharded_parameter_metrics_torch(
        model,
        test_source,
        split["test_indices"],
        device=device,
        batch_size=batch_size,
        target_mode=target_mode,
        parameters=target_parameters,
    )

    metrics = {
        "model_id": model_id,
        "model_type": "pytorch_cnn",
        "waveform_mode": waveform_mode,
        "target_mode": target_mode,
        "model_size": model_size,
        "pooling_mode": pooling_mode,
        "head_mode": head_mode,
        "loss_preset": loss_preset,
        "loss_weights": {
            parameter.name: float(weight)
            for parameter, weight in zip(target_parameters, loss_weights.detach().cpu().numpy())
        },
        "loss_groups": loss_groups_for_parameters(target_parameters)
        if loss_preset in (LOSS_PRESET_GROUP_BALANCED, LOSS_PRESET_AUDIBLE, LOSS_PRESET_NOISE_DETUNE)
        else {},
        "tensor_path": str(tensor_path),
        "validation_tensor_path": str(validation_tensor_path) if validation_tensor_path else None,
        "tensor_sharded": True,
        "tensor_shard_count": len(source["shard_paths"]),
        "metadata_path": source["metadata_path"],
        "num_samples": int(source["sample_count"]),
        "num_features": [input_channels, int(source["feature_shape"][1]), int(source["feature_shape"][2])],
        "num_targets": int(len(target_parameters)),
        "target_parameters": [parameter.name for parameter in target_parameters],
        "train_samples": int(len(split["train_indices"])),
        "test_samples": int(len(split["test_indices"])),
        "benchmark_samples": int(len(split["benchmark_indices"])),
        "epochs": int(epochs),
        "completed_epochs": int(completed_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "learning_rates": learning_rates,
        "optimizer": optimizer_name,
        "weight_decay": float(weight_decay),
        "scheduler": scheduler_name,
        "scheduler_step_size": int(scheduler_step_size),
        "scheduler_gamma": float(scheduler_gamma),
        "early_stopping_patience": int(early_stopping_patience),
        "checkpoint_selection": checkpoint_selection,
        "best_epoch": int(best_epoch),
        "best_test_loss": float(best_test_loss),
        "best_test_objective_loss": float(best_test_loss),
        "test_size": float(test_size) if validation_source is None else None,
        "benchmark_size": float(benchmark_size) if validation_source is None else 0.0,
        "device": device.type,
        "train_loss": epoch_losses[-1],
        "train_objective_loss": epoch_losses[-1],
        "train_losses": epoch_losses,
        "train_objective_losses": epoch_losses,
        "test_losses": test_losses,
        "test_objective_losses": test_losses,
        "test_objective_loss": test_losses[-1],
        "train_parameter_mse": sharded_parameter_mse_torch(
            model,
            source,
            split["train_indices"],
            device=device,
            batch_size=batch_size,
            target_mode=target_mode,
        ),
        "test_loss": sharded_parameter_mse_torch(
            model,
            test_source,
            split["test_indices"],
            device=device,
            batch_size=batch_size,
            target_mode=target_mode,
        ),
        "train_mae": train_parameter_metrics["mae"],
        "test_mae": test_parameter_metrics["mae"],
        "train_continuous_mae": train_parameter_metrics["continuous_mae"],
        "test_continuous_mae": test_parameter_metrics["continuous_mae"],
        "train_per_parameter_mae": train_parameter_metrics["per_parameter_mae"],
        "test_per_parameter_mae": test_parameter_metrics["per_parameter_mae"],
        "train_prediction_distribution": train_parameter_metrics["prediction_distribution"],
        "test_prediction_distribution": test_parameter_metrics["prediction_distribution"],
        "train_grouped_mae": train_parameter_metrics["grouped_mae"],
        "test_grouped_mae": test_parameter_metrics["grouped_mae"],
        "train_waveform_accuracy": train_parameter_metrics["waveform_accuracy"],
        "test_waveform_accuracy": test_parameter_metrics["waveform_accuracy"],
        "train_waveform_accuracy_by_name": train_parameter_metrics["waveform_accuracy_by_name"],
        "test_waveform_accuracy_by_name": test_parameter_metrics["waveform_accuracy_by_name"],
    }
    metrics["test_parameter_mse"] = metrics["test_loss"]

    if len(split["benchmark_indices"]) > 0:
        benchmark_parameter_metrics = sharded_parameter_metrics_torch(
            model,
            source,
            split["benchmark_indices"],
            device=device,
            batch_size=batch_size,
            target_mode=target_mode,
            parameters=target_parameters,
        )
        metrics.update(
            {
                "benchmark_objective_loss": sharded_sample_loss_torch(
                    model,
                    source,
                    split["benchmark_indices"],
                    device=device,
                    batch_size=batch_size,
                    target_mode=target_mode,
                    loss_weights=loss_weights,
                    loss_preset=loss_preset,
                ),
                "benchmark_loss": sharded_parameter_mse_torch(
                    model,
                    source,
                    split["benchmark_indices"],
                    device=device,
                    batch_size=batch_size,
                    target_mode=target_mode,
                ),
                "benchmark_mae": benchmark_parameter_metrics["mae"],
                "benchmark_continuous_mae": benchmark_parameter_metrics["continuous_mae"],
                "benchmark_per_parameter_mae": benchmark_parameter_metrics["per_parameter_mae"],
                "benchmark_prediction_distribution": benchmark_parameter_metrics[
                    "prediction_distribution"
                ],
                "benchmark_grouped_mae": benchmark_parameter_metrics["grouped_mae"],
                "benchmark_waveform_accuracy": benchmark_parameter_metrics["waveform_accuracy"],
                "benchmark_waveform_accuracy_by_name": benchmark_parameter_metrics[
                    "waveform_accuracy_by_name"
                ],
            }
        )
        metrics["benchmark_parameter_mse"] = metrics["benchmark_loss"]

    return {
        "model": model,
        "metrics": metrics,
    }

def train_inverse_model(
    tensor_path=DEFAULT_TORCH_TENSOR_PATH,
    validation_tensor_path=None,
    model_id=DEFAULT_TORCH_MODEL_ID,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    test_size=DEFAULT_TEST_SIZE,
    benchmark_size=DEFAULT_BENCHMARK_SIZE,
    waveform_mode=DEFAULT_WAVEFORM_MODE,
    target_mode=DEFAULT_TARGET_MODE,
    loss_preset=DEFAULT_LOSS_PRESET,
    optimizer_name=DEFAULT_OPTIMIZER,
    weight_decay=0.0,
    scheduler_name=DEFAULT_SCHEDULER,
    scheduler_step_size=10,
    scheduler_gamma=0.5,
    early_stopping_patience=0,
    checkpoint_selection=DEFAULT_CHECKPOINT_SELECTION,
    model_size=DEFAULT_MODEL_SIZE,
    pooling_mode=DEFAULT_POOLING_MODE,
    head_mode=DEFAULT_HEAD_MODE,
    random_state=0,
    device=None,
    progress=False,
):
    if is_sharded_tensor_path(tensor_path):
        return train_inverse_model_sharded(
            tensor_path=tensor_path,
            validation_tensor_path=validation_tensor_path,
            model_id=model_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            test_size=test_size,
            benchmark_size=benchmark_size,
            waveform_mode=waveform_mode,
            target_mode=target_mode,
            loss_preset=loss_preset,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            early_stopping_patience=early_stopping_patience,
            checkpoint_selection=checkpoint_selection,
            model_size=model_size,
            pooling_mode=pooling_mode,
            head_mode=head_mode,
            random_state=random_state,
            device=device,
            progress=progress,
        )

    if validation_tensor_path is not None:
        raise ValueError("validation_tensor_path requires sharded tensor directories")

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if checkpoint_selection not in (CHECKPOINT_FINAL, CHECKPOINT_BEST_VALIDATION):
        raise ValueError(f"Unsupported checkpoint selection: {checkpoint_selection}")
    if head_mode not in (HEAD_MODE_SHARED, HEAD_MODE_GROUPED):
        raise ValueError(f"Unsupported head mode: {head_mode}")
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
        model_size=model_size,
        pooling_mode=pooling_mode,
        head_mode=head_mode,
    ).to(device)
    optimizer = create_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )
    loss_function = nn.MSELoss()
    loss_weights = parameter_loss_weights(
        target_parameters,
        preset=loss_preset,
    ).to(device)

    epoch_losses = []
    test_losses = []
    learning_rates = []
    best_epoch = 0
    best_test_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    completed_epochs = 0
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
                loss_preset=loss_preset,
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
        completed_epochs = epoch + 1
        test_epoch_loss = dataset_loss_torch(
            model,
            split["test_features"],
            split["test_targets"],
            device=device,
            batch_size=batch_size,
            loss_weights=loss_weights,
            loss_preset=loss_preset,
        )
        test_losses.append(test_epoch_loss)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        if test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if progress:
            print(
                f"\nEpoch {epoch + 1} complete - "
                f"train loss {epoch_loss:.6f}, test loss {test_epoch_loss:.6f}\n"
            )
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            if progress:
                print(f"Early stopping after epoch {epoch + 1}.")
            break

    if checkpoint_selection == CHECKPOINT_BEST_VALIDATION:
        model.load_state_dict(best_state_dict)

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
        "model_size": model_size,
        "pooling_mode": pooling_mode,
        "head_mode": head_mode,
        "loss_preset": loss_preset,
        "loss_weights": {
            parameter.name: float(weight)
            for parameter, weight in zip(target_parameters, loss_weights.detach().cpu().numpy())
        },
        "loss_groups": loss_groups_for_parameters(target_parameters)
        if loss_preset in (LOSS_PRESET_GROUP_BALANCED, LOSS_PRESET_AUDIBLE, LOSS_PRESET_NOISE_DETUNE)
        else {},
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
        "completed_epochs": int(completed_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "learning_rates": learning_rates,
        "optimizer": optimizer_name,
        "weight_decay": float(weight_decay),
        "scheduler": scheduler_name,
        "scheduler_step_size": int(scheduler_step_size),
        "scheduler_gamma": float(scheduler_gamma),
        "early_stopping_patience": int(early_stopping_patience),
        "checkpoint_selection": checkpoint_selection,
        "best_epoch": int(best_epoch),
        "best_test_loss": float(best_test_loss),
        "best_test_objective_loss": float(best_test_loss),
        "test_size": float(test_size),
        "benchmark_size": float(benchmark_size),
        "device": device.type,
        "train_loss": epoch_losses[-1],
        "train_objective_loss": epoch_losses[-1],
        "train_losses": epoch_losses,
        "train_objective_losses": epoch_losses,
        "test_losses": test_losses,
        "test_objective_losses": test_losses,
        "test_objective_loss": test_losses[-1],
        "train_parameter_mse": parameter_mse_torch(
            model,
            split["train_features"],
            split["train_targets"],
            device=device,
            batch_size=batch_size,
        ),
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
        "train_prediction_distribution": train_parameter_metrics["prediction_distribution"],
        "test_prediction_distribution": test_parameter_metrics["prediction_distribution"],
        "train_grouped_mae": train_parameter_metrics["grouped_mae"],
        "test_grouped_mae": test_parameter_metrics["grouped_mae"],
        "train_waveform_accuracy": train_parameter_metrics["waveform_accuracy"],
        "test_waveform_accuracy": test_parameter_metrics["waveform_accuracy"],
        "train_waveform_accuracy_by_name": train_parameter_metrics["waveform_accuracy_by_name"],
        "test_waveform_accuracy_by_name": test_parameter_metrics["waveform_accuracy_by_name"],
    }
    metrics["test_parameter_mse"] = metrics["test_loss"]

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
                "benchmark_objective_loss": dataset_loss_torch(
                    model,
                    split["benchmark_features"],
                    split["benchmark_targets"],
                    device=device,
                    batch_size=batch_size,
                    loss_weights=loss_weights,
                    loss_preset=loss_preset,
                ),
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
                "benchmark_prediction_distribution": benchmark_parameter_metrics[
                    "prediction_distribution"
                ],
                "benchmark_grouped_mae": benchmark_parameter_metrics["grouped_mae"],
                "benchmark_waveform_accuracy": benchmark_parameter_metrics["waveform_accuracy"],
                "benchmark_waveform_accuracy_by_name": benchmark_parameter_metrics[
                    "waveform_accuracy_by_name"
                ],
            }
        )
        metrics["benchmark_parameter_mse"] = metrics["benchmark_loss"]

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
            "model_size": getattr(model, "model_size", DEFAULT_MODEL_SIZE),
            "pooling_mode": getattr(model, "pooling_mode", POOLING_GLOBAL),
            "head_mode": getattr(model, "head_mode", DEFAULT_HEAD_MODE),
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
        model_size=checkpoint.get("model_size", DEFAULT_MODEL_SIZE),
        pooling_mode=checkpoint.get("pooling_mode", POOLING_GLOBAL),
        head_mode=checkpoint.get("head_mode", DEFAULT_HEAD_MODE),
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


def patch_from_model_vector(vector, parameters, freq=None):
    values = {}
    osc_total_level = None
    osc_balance_value = None
    detuned_balance_value = None
    for parameter, normalized in zip(parameters, vector):
        value = float(np.clip(normalized, 0.0, 1.0))
        if parameter.name == "osc_total_level":
            osc_total_level = value * 2.0
        elif parameter.name == "osc_balance":
            osc_balance_value = value
        elif parameter.name == "detuned_balance":
            detuned_balance_value = value
        elif parameter.name == "main_wave":
            values["osc1_wave"] = denormalize_parameter_value(parameter, value)
        elif parameter.name == "detuned_wave":
            values["osc2_wave"] = denormalize_parameter_value(parameter, value)
        elif parameter.name == "detune_amount":
            values["osc2_detune"] = denormalize_parameter_value(parameter, value)
        else:
            values[parameter.name] = denormalize_parameter_value(parameter, value)

    balance_values = [
        value for value in (osc_balance_value, detuned_balance_value) if value is not None
    ]
    if osc_total_level is not None or balance_values:
        if osc_total_level is None or len(balance_values) != 1:
            raise ValueError("oscillator mix predictions require total level and one balance")
        balance = balance_values[0]
        values["osc1_level"] = float(np.clip(osc_total_level * (1.0 - balance), 0.0, 1.0))
        values["osc2_level"] = float(np.clip(osc_total_level * balance, 0.0, 1.0))

    if "freq" not in values:
        if freq is None:
            raise ValueError("freq is required to render pitch-conditioned predictions")
        values["freq"] = float(freq)

    return SynthConfig(**values).to_render_kwargs()


def predict_patch_from_audio(
    model,
    audio,
    sample_rate,
    device=None,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
    freq=None,
):
    mel_tensor = mel_tensor_from_audio(audio, sample_rate, frames=frames)
    if getattr(model, "input_channels", DEFAULT_INPUT_CHANNELS) == 2:
        mel_tensor = add_pitch_context_to_mel_tensor(mel_tensor, freq)
    elif mel_tensor.shape[0] != getattr(model, "input_channels", DEFAULT_INPUT_CHANNELS):
        raise ValueError(
            f"model expects {model.input_channels} input channels, got {mel_tensor.shape[0]}"
        )
    vector = tuple(
        float(value)
        for value in predict_normalized_vectors(
            model,
            mel_tensor[np.newaxis, :, :, :],
            device=device,
        )[0]
    )
    parameters = getattr(model, "parameters_schema", VECTOR_PARAMETERS)
    patch = patch_from_model_vector(vector, parameters, freq=freq)
    constrain_envelope_fits_length(patch)
    return patch


def expected_mel_tensor_shape(
    mel_bins=DEFAULT_MEL_BINS,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
):
    return (DEFAULT_INPUT_CHANNELS, mel_bins, frames)
