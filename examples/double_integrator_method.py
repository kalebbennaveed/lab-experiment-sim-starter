"""Replace the nominal method with your double-integrator acceleration law.

Guide: examples/README.md
Run: uv run python examples/double_integrator_method.py
Preview: uv run lab-demo --example examples/double_integrator_method.py --port 8001
Free space: add --config config/generic.toml to either command.
"""

import numpy as np

from lab_sim import DoubleIntegratorPolicy, ExperimentConfig, Lissajous, Reference
from lab_sim.cli import example_main
from lab_sim.double_integrator import double_integrator_reference


class MyAccelerationMethod:
    """A working PD example; replace __call__ with your MPC/planner/algorithm."""

    def __init__(self, config: ExperimentConfig):
        self.goal = Lissajous(config.trajectory, config.vehicle.gravity)
        self.kp = 2.0
        self.kd = 3.0
        self.max_acceleration = 2.0 # per-axis m/s²; a method setting

    def __call__(self, t: float, planned_state: np.ndarray, measured_state: np.ndarray) -> np.ndarray:
        # Both arrays: [px, py, pz, vx, vy, vz], in the world frame.
        # planned_state: DI reference integrated from your previous commands.
        # measured_state: quadrotor position/velocity; available for feedback.
        # This example controls the virtual DI reference toward a Lissajous goal.
        position, velocity = planned_state[:3], planned_state[3:]
        goal_p = self.goal.derivative(t)
        goal_v = self.goal.derivative(t, 1)
        goal_a = self.goal.derivative(t, 2)

        # REPLACE THIS BLOCK with your method. Return exactly [ax, ay, az].
        acceleration = goal_a + self.kp*(goal_p-position) + self.kd*(goal_v-velocity)
        return np.clip(acceleration, -self.max_acceleration, self.max_acceleration)


def initial_di_state(config: ExperimentConfig) -> np.ndarray:
    goal = Lissajous(config.trajectory, config.vehicle.gravity)
    return np.concatenate((goal.derivative(0), goal.derivative(0, 1)))


def make_policy(config: ExperimentConfig) -> DoubleIntegratorPolicy:
    # Your method runs at 20 Hz; flatness and the geometric tracker run at
    # the simulation's 100 Hz. Choose a command_dt that is a multiple of dt.
    return DoubleIntegratorPolicy(
        method=MyAccelerationMethod(config),
        initial_state=initial_di_state(config),
        command_dt=5*config.simulation.dt,
        gravity=config.vehicle.gravity,
        yaw=config.trajectory.yaw,
        yaw_rate=config.trajectory.yaw_rate,
    )


def make_initial_reference(config: ExperimentConfig) -> Reference:
    # Initialize independently so the method first sees the real vehicle state.
    # simulate() applies simulation.initial_position_error to this position.
    return double_integrator_reference(initial_di_state(config), np.zeros(3), gravity=config.vehicle.gravity, yaw=config.trajectory.yaw, yaw_rate=config.trajectory.yaw_rate)


if __name__ == "__main__":
    example_main(policy_factory=make_policy, initial_reference_factory=make_initial_reference)
