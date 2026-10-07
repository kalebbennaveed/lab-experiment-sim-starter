"""Analytic Lissajous reference and its derivatives, ported from meSch."""

import numpy as np

from .config import TrajectoryConfig
from .flatness import flat_state_to_reference
from .types import Reference, State


class Lissajous:
    def __init__(self, config: TrajectoryConfig, gravity: float = 9.81):
        self.config = config
        self.gravity = gravity
        self.amplitude = np.asarray(config.amplitude, dtype=float)
        self.frequency = np.asarray(config.frequency, dtype=float)
        self.phase = np.asarray(config.phase, dtype=float)
        self.offset = np.asarray(config.offset, dtype=float)

    def derivative(self, t: float, order: int = 0) -> np.ndarray:
        value = self.amplitude * self.frequency**order * np.sin(self.frequency * t + self.phase + np.pi * order / 2)
        return value + self.offset if order == 0 else value

    def __call__(self, t: float, state: State) -> Reference:
        return flat_state_to_reference(self.derivative(t), self.derivative(t, 1), self.derivative(t, 2), self.gravity, self.config.yaw + self.config.yaw_rate*t, self.config.yaw_rate)
