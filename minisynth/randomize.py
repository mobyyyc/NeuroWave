"""Random patch generation for dataset creation."""

import random

import numpy as np

from minisynth.oscillators import BASE_WAVES
from minisynth.schema import PARAMETERS, denormalize_linear, denormalize_log

MIN_OSCILLATOR_LEVEL_SUM = 0.2
MAX_DATASET_AUDIO_PEAK = 0.95
ENVELOPE_TIME_PARAMETERS = ("attack", "decay", "release")

RANDOM_PARAMETER_RANGES = {
    "freq": (55.0, 1760.0),
    "length": (0.5, 3.0),
    "cutoff": (80.0, 8000.0),
    "resonance": (0.0, 0.8),
    "attack": (0.001, 0.8),
    "decay": (0.001, 0.8),
    "sustain": (0.05, 1.0),
    "release": (0.001, 0.8),
}


def random_patch(seed):
    rng = random.Random(seed)
    patch = {}

    for parameter in PARAMETERS:
        minimum, maximum = random_range(parameter)

        if parameter.kind == "enum":
            patch[parameter.name] = rng.choice(BASE_WAVES)
        elif parameter.scale == "log":
            patch[parameter.name] = denormalize_log(
                rng.random(),
                minimum,
                maximum,
            )
        elif parameter.scale == "linear":
            patch[parameter.name] = denormalize_linear(
                rng.random(),
                minimum,
                maximum,
            )
        else:
            raise ValueError(f"Unsupported parameter scale: {parameter.scale}")

    constrain_not_silent(patch)
    constrain_envelope_fits_length(patch)

    return patch


def constrain_not_silent(patch):
    level_sum = patch["osc1_level"] + patch["osc2_level"]
    if level_sum >= MIN_OSCILLATOR_LEVEL_SUM:
        return patch

    patch["osc1_level"] += MIN_OSCILLATOR_LEVEL_SUM - level_sum
    return patch


def constrain_envelope_fits_length(patch):
    minimums = {
        parameter.name: parameter.minimum
        for parameter in PARAMETERS
        if parameter.name in ENVELOPE_TIME_PARAMETERS
    }
    minimum_total = sum(minimums[name] for name in ENVELOPE_TIME_PARAMETERS)
    max_total = patch["length"] * 0.95

    if max_total < minimum_total:
        raise ValueError("length is too short for envelope minimums")

    for name in ENVELOPE_TIME_PARAMETERS:
        patch[name] = max(patch[name], minimums[name])

    envelope_total = sum(patch[name] for name in ENVELOPE_TIME_PARAMETERS)

    if envelope_total <= max_total:
        return patch

    adjustable_total = sum(
        patch[name] - minimums[name]
        for name in ENVELOPE_TIME_PARAMETERS
    )
    target_adjustable_total = max_total - minimum_total
    scale = target_adjustable_total / adjustable_total

    for name in ENVELOPE_TIME_PARAMETERS:
        patch[name] = minimums[name] + (patch[name] - minimums[name]) * scale

    return patch


def random_range(parameter):
    return RANDOM_PARAMETER_RANGES.get(
        parameter.name,
        (parameter.minimum, parameter.maximum),
    )


def audio_avoids_clipping(audio, peak_limit=MAX_DATASET_AUDIO_PEAK):
    if len(audio) == 0:
        return False

    if not np.all(np.isfinite(audio)):
        return False

    return np.max(np.abs(audio)) <= peak_limit
