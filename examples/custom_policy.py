"""Add your algorithm here and keep the geometric tracker/plant underneath.

Run: uv run python examples/custom_policy.py
Preview: uv run lab-demo --example examples/custom_policy.py
Edit this file while the demo is running; it reruns in a fresh Python process.
"""

import numpy as np

from lab_sim import ExperimentConfig, Lissajous, Reference, State
from lab_sim.cli import example_main
from lab_sim.flatness import flat_state_to_reference


class MyPolicy:
    def __init__(self, config: ExperimentConfig):
        self.nominal = Lissajous(config.trajectory, config.vehicle.gravity)
        self.gravity = config.vehicle.gravity
        self.yaw_rate = config.trajectory.yaw_rate
        self.amplitude = 0.15 # m; try changing this with the browser open
        self.frequency = 0.4 # rad/s

    def __call__(self, t: float, state: State) -> Reference:
        reference = self.nominal(t, state)
        # A small vertical oscillation demonstrates a custom planning layer.
        # Replace this block with your algorithm. Supply consistent derivatives.
        a, w = self.amplitude, self.frequency
        reference.position += np.array([0.0, 0.0, a * np.sin(w * t)])
        reference.velocity += np.array([0.0, 0.0, a * w * np.cos(w * t)])
        reference.acceleration += np.array([0.0, 0.0, -a * w**2 * np.sin(w * t)])
        return flat_state_to_reference(reference.position, reference.velocity, reference.acceleration, self.gravity, reference.yaw, self.yaw_rate)


if __name__ == "__main__":
    example_main(policy_factory=MyPolicy)
