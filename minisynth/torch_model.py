"""PyTorch inverse model for mel spectrogram to synth parameter prediction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES, mel_tensor_from_audio
from minisynth.randomize import constrain_envelope_fits_length
from minisynth.schema import SynthConfig, VECTOR_PARAMETERS

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
    ):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1")
        if input_channels < 1:
            raise ValueError("input_channels must be at least 1")

        self.output_dim = output_dim
        self.input_channels = input_channels
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, mel_tensors):
        if mel_tensors.ndim != 4:
            raise ValueError("mel_tensors must have shape (batch, channels, mel_bins, frames)")
        if mel_tensors.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {mel_tensors.shape[1]}"
            )

        return self.head(self.encoder(mel_tensors))


def create_inverse_model(output_dim=DEFAULT_OUTPUT_DIM):
    return MelSpectrogramInverseModel(output_dim=output_dim)


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
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be between 0 and 1")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    sample_count = len(x)
    test_count = max(1, int(round(sample_count * test_size)))
    train_count = sample_count - test_count

    if train_count < 1:
        raise ValueError("train/test split must leave at least 1 training sample")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(sample_count)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return {
        "train_features": x[train_indices],
        "train_targets": y[train_indices],
        "test_features": x[test_indices],
        "test_targets": y[test_indices],
    }


def tensor_loader(features, targets, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    split = split_tensor_dataset(
        dataset["features"],
        dataset["targets"],
        test_size=test_size,
        random_state=random_state,
    )
    model = create_inverse_model(output_dim=dataset["targets"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

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
            loss = loss_function(predictions, batch_targets)
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

    metrics = {
        "model_id": model_id,
        "model_type": "pytorch_cnn",
        "tensor_path": str(tensor_path),
        "metadata_path": dataset["metadata_path"],
        "num_samples": int(len(dataset["features"])),
        "num_features": list(dataset["features"].shape[1:]),
        "num_targets": int(dataset["targets"].shape[1]),
        "train_samples": int(len(split["train_features"])),
        "test_samples": int(len(split["test_features"])),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
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
        "train_mae": parameter_mae_torch(
            model,
            split["train_features"],
            split["train_targets"],
            device=device,
            batch_size=batch_size,
        ),
        "test_mae": parameter_mae_torch(
            model,
            split["test_features"],
            split["test_targets"],
            device=device,
            batch_size=batch_size,
        ),
    }

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
    model = MelSpectrogramInverseModel(
        output_dim=checkpoint.get("output_dim", DEFAULT_OUTPUT_DIM),
        input_channels=checkpoint.get("input_channels", DEFAULT_INPUT_CHANNELS),
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
