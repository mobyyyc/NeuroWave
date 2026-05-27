"""Parameter schema and normalization helpers for ML-ready synth configs."""

from dataclasses import asdict, dataclass


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
