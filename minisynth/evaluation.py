"""Prediction evaluation helpers for NeuroWave models."""

from minisynth.compare import compare_audio_arrays
from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.ml import predict_patch_from_audio
from minisynth.schema import SynthConfig
from minisynth.search import local_refinement_search


def evaluate_audio_prediction(
    model,
    target_audio,
    target_sample_rate,
    refine_iterations=0,
    refine_seed=0,
    refine_step_size=0.05,
):
    patch = predict_patch_from_audio(model, target_audio, target_sample_rate)
    rendered_audio = render_patch(**patch)
    comparison = compare_audio_arrays(
        target_audio,
        target_sample_rate,
        rendered_audio,
        DEFAULT_SAMPLE_RATE,
    )
    result = {
        "patch": patch,
        "rendered_audio": rendered_audio,
        "comparison": comparison,
    }

    if refine_iterations > 0:
        initial_score = comparison["weighted_distance"]
        initial_vector = SynthConfig(**patch).to_vector()
        refined = local_refinement_search(
            target_audio,
            target_sample_rate,
            initial_vector,
            iterations=refine_iterations,
            seed=refine_seed,
            step_size=refine_step_size,
        )
        result.update(
            {
                "patch": refined["config"].to_render_kwargs(),
                "rendered_audio": refined["audio"],
                "comparison": refined["distances"],
                "refinement": {
                    "initial_score": initial_score,
                    "best_score": refined["score"],
                    "iterations": refined["evaluations"],
                    "attempts": refined["attempts"],
                    "seed": refine_seed,
                    "step_size": refine_step_size,
                },
            }
        )

    return result
