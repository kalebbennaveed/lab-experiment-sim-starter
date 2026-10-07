"""22-state rigid body with motor lag, drag and rotor angular momentum.

Ported from meSch/src/ExpDynamicsLibrary.jl: quadrotor3D!, excluding battery
discharge. RK4 holds motor commands constant during each controller step.
"""

import numpy as np

from .config import VehicleConfig
from .controller import allocation_matrix, hat
from .types import State


class Quadrotor:
    def __init__(self, vehicle: VehicleConfig, wind: np.ndarray, external_force: np.ndarray):
        self.vehicle = vehicle
        self.inertia = np.asarray(vehicle.inertia)
        self.allocation = allocation_matrix(vehicle)
        self.motor_direction = np.array([1.0, 1.0, -1.0, -1.0])
        self.wind = np.asarray(wind, dtype=float)
        self.external_force = np.asarray(external_force, dtype=float)

    def derivative(self, packed: np.ndarray, command: np.ndarray) -> np.ndarray:
        s, p = State.unpack(packed), self.vehicle
        motor_dot = (np.clip(command, 0, p.max_motor_speed) - s.motor_speed) / p.motor_time_constant
        thrust, mx, my, mz = self.allocation @ (s.motor_speed * np.abs(s.motor_speed))
        air_velocity = s.velocity - self.wind
        force = s.rotation[:, 2] * thrust - p.linear_drag * np.linalg.norm(air_velocity) * air_velocity + self.external_force
        force[2] -= p.mass * p.gravity
        moment = np.array([mx, my, mz])
        moment[2] -= p.rotor_inertia * (self.motor_direction @ motor_dot)
        moment -= p.angular_drag * np.linalg.norm(s.angular_velocity) * s.angular_velocity
        momentum = self.inertia * s.angular_velocity
        momentum[2] -= p.rotor_inertia * (self.motor_direction @ s.motor_speed)
        omega_dot = (moment - np.cross(s.angular_velocity, momentum)) / self.inertia
        return np.concatenate((s.velocity, force / p.mass, (s.rotation @ hat(s.angular_velocity)).ravel(), omega_dot, motor_dot))

    def step(self, state: State, command: np.ndarray, dt: float) -> State:
        x = state.pack()
        k1 = self.derivative(x, command)
        k2 = self.derivative(x + dt * k1 / 2, command)
        k3 = self.derivative(x + dt * k2 / 2, command)
        k4 = self.derivative(x + dt * k3, command)
        next_state = State.unpack(x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6)
        # Keep R on SO(3) after the numerical integration.
        u, _, vt = np.linalg.svd(next_state.rotation)
        correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
        next_state.rotation = u @ correction @ vt
        return next_state
