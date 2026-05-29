"""Search utilities for matching target audio with synth parameters."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from minisynth.compare import compare_audio_arrays
from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.io import save_patch
from minisynth.schema import SynthConfig, VECTOR_PARAMETERS

DEFAULT_RUNS_DIR = Path("runs")


def random_vector(rng, size=None):
    if size is None:
        size = len(VECTOR_PARAMETERS)

    return tuple(float(value) for value in rng.random(size))


def render_config_audio(config):
    return render_patch(**config.to_render_kwargs())


def save_search_result(result, run_dir):
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)

    patch_path = destination / "best_patch.json"
    audio_path = destination / "best.wav"
    report_path = destination / "report.json"

    save_patch(result["config"].to_render_kwargs(), patch_path)
    sf.write(audio_path, result["audio"], DEFAULT_SAMPLE_RATE)
    save_search_report(result, report_path, patch_path, audio_path)

    return {
        "patch_path": patch_path,
        "audio_path": audio_path,
        "report_path": report_path,
    }


def save_search_report(result, path, patch_path, audio_path):
    report = search_report(result, patch_path, audio_path)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def search_report(result, patch_path, audio_path):
    return {
        "score": result["score"],
        "iteration": result.get("iteration"),
        "attempt": result.get("attempt"),
        "evaluations": result.get("evaluations"),
        "attempts": result.get("attempts"),
        "distances": result.get("distances", {}),
        "vector": list(result["vector"]),
        "config": result["config"].to_render_kwargs(),
        "patch_path": str(patch_path),
        "audio_path": str(audio_path),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "frames": len(result["audio"]),
    }


def format_search_progress(progress):
    best_score = progress["best_score"]
    if best_score is None:
        best_score_text = "none"
    else:
        best_score_text = f"{best_score:.6f}"

    return (
        f"evaluation {progress['evaluations']}/{progress['iterations']} "
        f"attempt {progress['attempts']} "
        f"score {progress['score']:.6f} "
        f"best {best_score_text}"
    )


def print_search_progress(progress):
    print(format_search_progress(progress))


def random_search(
    target_audio,
    target_sample_rate=DEFAULT_SAMPLE_RATE,
    iterations=100,
    seed=0,
    renderer=render_config_audio,
    comparer=compare_audio_arrays,
    progress_callback=None,
    progress_interval=1,
):
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1")

    rng = np.random.default_rng(seed)
    best = None
    attempts = 0
    evaluated = 0
    max_attempts = iterations * 10

    while evaluated < iterations and attempts < max_attempts:
        attempts += 1
        vector = random_vector(rng)
        config = SynthConfig.from_vector(vector)
        try:
            candidate_audio = renderer(config)
            distances = comparer(
                target_audio,
                target_sample_rate,
                candidate_audio,
                DEFAULT_SAMPLE_RATE,
            )
        except ValueError:
            continue

        score = distances["weighted_distance"]
        if not np.isfinite(score):
            continue

        improved = best is None or score < best["score"]
        if best is None or score < best["score"]:
            best = {
                "iteration": evaluated,
                "attempt": attempts - 1,
                "score": score,
                "distances": distances,
                "config": config,
                "vector": vector,
                "audio": candidate_audio,
            }

        evaluated += 1
        if progress_callback is not None and evaluated % progress_interval == 0:
            progress_callback(
                {
                    "evaluations": evaluated,
                    "iterations": iterations,
                    "attempts": attempts,
                    "score": score,
                    "best_score": best["score"] if best is not None else None,
                    "improved": improved,
                }
            )

    if best is None:
        raise ValueError("random search did not produce any valid candidates")

    best["evaluations"] = evaluated
    best["attempts"] = attempts
    return best
