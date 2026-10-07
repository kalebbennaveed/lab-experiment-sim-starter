"""Slow, smooth 3D Lissajous curves with integer frequency ratios."""

from dataclasses import replace
import numpy as np

from .config import ExperimentConfig

# A base frequency of 0.15 rad/s gives all curves a 41.89 s common period.
# Configured offsets and yaw settings remain editable in config/generic.toml.
PATTERNS = {
    "orbit": {"amplitude": [2.4, 2.0, 1.2], "ratio": [2, 3, 4], "phase": [np.pi/2, 0.0, np.pi/4]},
    "butterfly": {"amplitude": [2.5, 2.0, 1.0], "ratio": [3, 2, 4], "phase": [0.0, np.pi/2, np.pi/4]},
    "ribbon": {"amplitude": [2.6, 1.5, 1.3], "ratio": [1, 2, 3], "phase": [0.0, np.pi/2, 0.0]},
    "crown": {"amplitude": [2.2, 2.2, 0.8], "ratio": [1, 1, 5], "phase": [np.pi/2, 0.0, np.pi/4]},
}


def apply_pattern(config: ExperimentConfig, name: str) -> ExperimentConfig:
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern {name!r}; choose from {', '.join(PATTERNS)}")
    pattern = PATTERNS[name]
    config.trajectory = replace(config.trajectory, amplitude=pattern["amplitude"].copy(), frequency=(0.15*np.asarray(pattern["ratio"])).tolist(), phase=pattern["phase"].copy())
    config.simulation.duration = 2*np.pi/0.15
    return config
