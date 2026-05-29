"""Prediction evaluation helpers for NeuroWave models."""

from minisynth.compare import compare_audio_arrays
from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.ml import predict_patch_from_audio


def evaluate_audio_prediction(model, target_audio, target_sample_rate):
    patch = predict_patch_from_audio(model, target_audio, target_sample_rate)
    rendered_audio = render_patch(**patch)
    comparison = compare_audio_arrays(
        target_audio,
        target_sample_rate,
        rendered_audio,
        DEFAULT_SAMPLE_RATE,
    )

    return {
        "patch": patch,
        "rendered_audio": rendered_audio,
        "comparison": comparison,
    }
