"""PyTorch inverse model for mel spectrogram to synth parameter prediction."""

import torch
from torch import nn

from minisynth.dataset import DEFAULT_MEL_TENSOR_FRAMES
from minisynth.schema import VECTOR_PARAMETERS

DEFAULT_MEL_BINS = 64
DEFAULT_INPUT_CHANNELS = 1
DEFAULT_OUTPUT_DIM = len(VECTOR_PARAMETERS)


def select_torch_device():
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


def predict_normalized_vectors(model, mel_tensors, device=None):
    if device is None:
        device = select_torch_device()

    model = model.to(device)
    model.eval()
    inputs = torch.as_tensor(mel_tensors, dtype=torch.float32, device=device)

    with torch.no_grad():
        predictions = model(inputs)

    return predictions.cpu().numpy()


def expected_mel_tensor_shape(
    mel_bins=DEFAULT_MEL_BINS,
    frames=DEFAULT_MEL_TENSOR_FRAMES,
):
    return (DEFAULT_INPUT_CHANNELS, mel_bins, frames)
