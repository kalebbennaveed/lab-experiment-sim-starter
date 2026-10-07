"""Connect a double-integrator method to the meSch tracking pipeline."""

from typing import Protocol

import numpy as np

from .flatness import flat_state_to_reference
from .types import Reference, State, Vector


def _array(value, size: int, name: str) -> Vector:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def double_integrator_step(state: Vector, acceleration: Vector, dt: float) -> Vector:
    """Exact step for p_dot = v, v_dot = u with u held constant."""
    state = _array(state, 6, "Double-integrator state [px, py, pz, vx, vy, vz]")
    acceleration = _array(acceleration, 3, "Acceleration command [ax, ay, az]")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("Double-integrator dt must be finite and nonnegative")
    return np.concatenate((state[:3] + dt*state[3:] + 0.5*dt**2*acceleration, state[3:] + dt*acceleration))


def double_integrator_reference(state: Vector, acceleration: Vector, *, gravity: float = 9.81, yaw: float = 0.0, yaw_rate: float = 3/8, yaw_acceleration: float = 0.0) -> Reference:
    """Map a planner's desired six-state and three-input output into flatness."""
    state = _array(state, 6, "Desired state [px, py, pz, vx, vy, vz]")
    acceleration = _array(acceleration, 3, "Acceleration command [ax, ay, az]")
    return flat_state_to_reference(state[:3], state[3:], acceleration, gravity, yaw, yaw_rate, yaw_acceleration)


class AccelerationMethod(Protocol):
    def __call__(self, t: float, planned_state: Vector, measured_state: Vector) -> Vector:
        """Return world-frame acceleration in m/s², without gravity or thrust."""
        ...


class DoubleIntegratorPolicy:
    """Roll out a method's acceleration commands and apply meSch flatness.

The planned state is a virtual DI reference, separate from the quadrotor's
measured position/velocity. Commands are held between method updates; the
reference is sampled at each geometric controller step. Call with increasing
times. Repeated calls at the same time do not integrate or rerun the method.

Pass initial_reference to simulate() so the method's first measured state
comes from the initialized vehicle rather than the initialization placeholder.
"""

    def __init__(self, method: AccelerationMethod, initial_state: Vector, *, command_dt: float = 0.05, gravity: float = 9.81, yaw: float = 0.0, yaw_rate: float = 3/8):
        if not np.isfinite(command_dt) or command_dt <= 0:
            raise ValueError("command_dt must be finite and positive")
        self.method = method
        self.planned_state = _array(initial_state, 6, "Initial double-integrator state").copy()
        self.command_dt = command_dt
        self.gravity, self.yaw, self.yaw_rate = gravity, yaw, yaw_rate
        self.time = 0.0
        self.acceleration: Vector | None = None
        self.next_command_time = 0.0

    def __call__(self, t: float, state: State) -> Reference:
        if not np.isfinite(t) or t < self.time:
            raise ValueError("DoubleIntegratorPolicy requires finite, nondecreasing times")
        if self.acceleration is not None:
            self.planned_state = double_integrator_step(self.planned_state, self.acceleration, t-self.time)
        self.time = t
        if self.acceleration is None or t + 1e-9 >= self.next_command_time:
            measured = np.concatenate((state.position, state.velocity))
            command = self.method(t, self.planned_state.copy(), measured.copy())
            self.acceleration = _array(command, 3, "Method acceleration command [ax, ay, az]").copy()
            self.next_command_time = t + self.command_dt
        return double_integrator_reference(self.planned_state.copy(), self.acceleration.copy(), gravity=self.gravity, yaw=self.yaw+self.yaw_rate*t, yaw_rate=self.yaw_rate)
