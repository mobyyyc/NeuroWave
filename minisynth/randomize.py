"""Random patch generation for dataset creation."""

import random

from minisynth.oscillators import BASE_WAVES
from minisynth.schema import PARAMETERS, denormalize_linear, denormalize_log


def random_patch(seed):
    rng = random.Random(seed)
    patch = {}

    for parameter in PARAMETERS:
        if parameter.kind == "enum":
            patch[parameter.name] = rng.choice(BASE_WAVES)
        elif parameter.scale == "log":
            patch[parameter.name] = denormalize_log(
                rng.random(),
                parameter.minimum,
                parameter.maximum,
            )
        elif parameter.scale == "linear":
            patch[parameter.name] = denormalize_linear(
                rng.random(),
                parameter.minimum,
                parameter.maximum,
            )
        else:
            raise ValueError(f"Unsupported parameter scale: {parameter.scale}")

    return patch
