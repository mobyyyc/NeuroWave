"""Parameter schema and normalization helpers for ML-ready synth configs."""

from dataclasses import asdict, dataclass
import math

from minisynth.oscillators import BASE_WAVES


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    minimum: float | None
    maximum: float | None
    default: object
    scale: str
    group: str
    ml_enabled: bool = True


@dataclass(frozen=True)
class SynthConfig:
    freq: float = 261.63
    length: float = 1.5
    osc1_wave: str = "saw"
    osc1_level: float = 0.8
    osc2_wave: str = "saw"
    osc2_level: float = 0.4
    osc2_detune: float = 7
    cutoff: float = 1200
    resonance: float = 0.2
    attack: float = 0.01
    decay: float = 0.2
    sustain: float = 0.7
    release: float = 0.3

    def to_render_kwargs(self):
        return asdict(self)

    def to_vector(self):
        return config_to_vector(self)

    @classmethod
    def from_vector(cls, vector):
        return config_from_vector(vector)


def normalize_linear(value, minimum, maximum):
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    return (value - minimum) / (maximum - minimum)


def denormalize_linear(normalized, minimum, maximum):
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    return minimum + normalized * (maximum - minimum)


def normalize_log(value, minimum, maximum):
    if minimum <= 0:
        raise ValueError("minimum must be positive for logarithmic normalization")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    if value <= 0:
        raise ValueError("value must be positive for logarithmic normalization")

    return (math.log(value) - math.log(minimum)) / (math.log(maximum) - math.log(minimum))


def denormalize_log(normalized, minimum, maximum):
    if minimum <= 0:
        raise ValueError("minimum must be positive for logarithmic denormalization")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    return math.exp(math.log(minimum) + normalized * (math.log(maximum) - math.log(minimum)))


def config_to_vector(config, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    values = asdict(config)
    return tuple(normalize_parameter_value(parameter, values[parameter.name]) for parameter in parameters)


def config_from_vector(vector, parameters=None):
    if parameters is None:
        parameters = VECTOR_PARAMETERS

    if len(vector) != len(parameters):
        raise ValueError(f"Expected vector length {len(parameters)}, got {len(vector)}")

    values = {}
    for parameter, normalized in zip(parameters, vector):
        values[parameter.name] = denormalize_parameter_value(parameter, normalized)

    return SynthConfig(**values)


def normalize_parameter_value(parameter, value):
    if parameter.kind == "float" and parameter.scale == "linear":
        return normalize_linear(value, parameter.minimum, parameter.maximum)

    if parameter.kind == "float" and parameter.scale == "log":
        return normalize_log(value, parameter.minimum, parameter.maximum)

    if parameter.kind == "enum" and parameter.scale == "categorical":
        choices = categorical_values(parameter)
        if value not in choices:
            raise ValueError(f"Unknown categorical value for {parameter.name}: {value}")

        if len(choices) == 1:
            return 0.0

        return choices.index(value) / (len(choices) - 1)

    raise ValueError(f"Unsupported parameter for vector conversion: {parameter.name}")


def denormalize_parameter_value(parameter, normalized):
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"normalized value must be in [0, 1] for {parameter.name}")

    if parameter.kind == "float" and parameter.scale == "linear":
        return denormalize_linear(normalized, parameter.minimum, parameter.maximum)

    if parameter.kind == "float" and parameter.scale == "log":
        return denormalize_log(normalized, parameter.minimum, parameter.maximum)

    if parameter.kind == "enum" and parameter.scale == "categorical":
        choices = categorical_values(parameter)
        index = round(normalized * (len(choices) - 1))
        return choices[index]

    raise ValueError(f"Unsupported parameter for vector reconstruction: {parameter.name}")


def categorical_values(parameter):
    if parameter.name in ("osc1_wave", "osc2_wave", "main_wave", "detuned_wave"):
        return BASE_WAVES

    raise ValueError(f"No categorical values configured for {parameter.name}")


PARAMETERS = (
    Parameter("freq", "float", 20.0, 20000.0, 261.63, "log", "global"),
    Parameter("length", "float", 0.05, 10.0, 1.5, "linear", "global"),
    Parameter("osc1_wave", "enum", None, None, "saw", "categorical", "oscillator"),
    Parameter("osc1_level", "float", 0.0, 1.0, 0.8, "linear", "oscillator"),
    Parameter("osc2_wave", "enum", None, None, "saw", "categorical", "oscillator"),
    Parameter("osc2_level", "float", 0.0, 1.0, 0.4, "linear", "oscillator"),
    Parameter("osc2_detune", "float", -1200.0, 1200.0, 7, "linear", "oscillator"),
    Parameter("cutoff", "float", 20.0, 20000.0, 1200, "log", "filter"),
    Parameter("resonance", "float", 0.0, 0.99, 0.2, "linear", "filter"),
    Parameter("attack", "float", 0.001, 5.0, 0.01, "log", "envelope"),
    Parameter("decay", "float", 0.001, 5.0, 0.2, "log", "envelope"),
    Parameter("sustain", "float", 0.0, 1.0, 0.7, "linear", "envelope"),
    Parameter("release", "float", 0.001, 5.0, 0.3, "log", "envelope"),
)

VECTOR_PARAMETERS = tuple(parameter for parameter in PARAMETERS if parameter.ml_enabled)
