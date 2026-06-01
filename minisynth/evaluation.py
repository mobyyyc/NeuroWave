"""Prediction evaluation helpers for NeuroWave models."""

import numpy as np

from minisynth.compare import compare_audio_arrays
from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.ml import predict_patch_from_audio
from minisynth.schema import (
    SynthConfig,
    VECTOR_PARAMETERS,
    categorical_values,
    normalize_parameter_value,
)
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


def evaluate_patch_prediction(
    patch,
    target_audio,
    target_sample_rate,
    refine_iterations=0,
    refine_seed=0,
    refine_step_size=0.05,
):
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


def summarize_weighted_distances(results):
    if not results:
        raise ValueError("results must not be empty")

    scores = np.asarray(
        [result["comparison"]["weighted_distance"] for result in results],
        dtype=float,
    )

    return {
        "count": int(len(scores)),
        "mean_weighted_distance": float(np.mean(scores)),
        "median_weighted_distance": float(np.median(scores)),
        "min_weighted_distance": float(np.min(scores)),
        "max_weighted_distance": float(np.max(scores)),
    }


def parameter_error_report(target_patch, predicted_patch, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    errors = {}
    for parameter in parameters:
        target_value = target_patch[parameter.name]
        predicted_value = predicted_patch[parameter.name]
        if parameter.kind == "enum" and parameter.scale == "categorical":
            choices = categorical_values(parameter)
            target_index = choices.index(target_value)
            predicted_index = choices.index(predicted_value)
            if len(choices) == 1:
                normalized_error = 0.0
                target_normalized = 0.0
                predicted_normalized = 0.0
            else:
                normalized_error = abs(target_index - predicted_index) / (len(choices) - 1)
                target_normalized = target_index / (len(choices) - 1)
                predicted_normalized = predicted_index / (len(choices) - 1)
            errors[parameter.name] = {
                "target": target_value,
                "predicted": predicted_value,
                "match": target_value == predicted_value,
                "target_normalized": float(target_normalized),
                "predicted_normalized": float(predicted_normalized),
                "normalized_error": float(normalized_error),
            }
            continue

        target_normalized = normalize_parameter_value(parameter, target_value)
        predicted_normalized = normalize_parameter_value(parameter, predicted_value)
        errors[parameter.name] = {
            "target": float(target_value),
            "predicted": float(predicted_value),
            "target_normalized": float(target_normalized),
            "predicted_normalized": float(predicted_normalized),
            "normalized_error": float(abs(target_normalized - predicted_normalized)),
        }

    return errors


def patch_prediction_distribution(results, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    metrics = {}
    for parameter in parameters:
        target_values = []
        predicted_values = []
        for result in results:
            errors = result.get("parameter_errors")
            if not errors or parameter.name not in errors:
                continue
            details = errors[parameter.name]
            target_values.append(details["target_normalized"])
            predicted_values.append(details["predicted_normalized"])

        if not target_values:
            continue

        target_array = np.asarray(target_values, dtype=np.float32)
        predicted_array = np.asarray(predicted_values, dtype=np.float32)
        target_std = float(np.std(target_array))
        predicted_std = float(np.std(predicted_array))
        metrics[parameter.name] = {
            "target_mean": float(np.mean(target_array)),
            "predicted_mean": float(np.mean(predicted_array)),
            "mean_delta": float(np.mean(predicted_array) - np.mean(target_array)),
            "target_std": target_std,
            "predicted_std": predicted_std,
            "std_ratio": float(predicted_std / target_std) if target_std > 0.0 else 0.0,
            "target_min": float(np.min(target_array)),
            "target_max": float(np.max(target_array)),
            "predicted_min": float(np.min(predicted_array)),
            "predicted_max": float(np.max(predicted_array)),
        }

    return metrics


def worst_clip_diagnostics(results, top_n=10, include_full=False):
    if top_n < 1:
        return []

    diagnostic_results = [
        result
        for result in results
        if "comparison" in result and "parameter_errors" in result
    ]
    diagnostic_results.sort(
        key=lambda result: result["comparison"]["weighted_distance"],
        reverse=True,
    )

    worst = []
    for result in diagnostic_results[:top_n]:
        parameter_errors = result["parameter_errors"]
        ranked_errors = sorted(
            parameter_errors.items(),
            key=lambda item: item[1]["normalized_error"],
            reverse=True,
        )
        diagnostic = {
            "index": result["index"],
            "seed": result["seed"],
            "audio_path": result["audio_path"],
            "weighted_distance": result["comparison"]["weighted_distance"],
            "comparison": result["comparison"],
            "largest_parameter_errors": [
                {"parameter": name, **details}
                for name, details in ranked_errors[:5]
            ],
        }
        if include_full:
            diagnostic.update(
                {
                    "parameter_errors": parameter_errors,
                    "target_patch": result["target_patch"],
                    "predicted_patch": result["predicted_patch"],
                }
            )
        worst.append(diagnostic)

    return worst
