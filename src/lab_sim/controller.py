"""SO(3) geometric tracker with position and attitude feedback.

References include the heading and angular feedforward obtained from
flat-state conversion.
"""

import numpy as np

from .config import ControllerConfig, VehicleConfig
from .types import Reference, State


def hat(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def desired_rotation(force: np.ndarray, yaw: float, heading: np.ndarray | None = None) -> np.ndarray:
    magnitude = np.linalg.norm(force)
    if magnitude < 1e-9:
        raise ValueError("Desired force is zero; attitude is undefined")
    b3 = force / magnitude
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0]) if heading is None else np.asarray(heading, dtype=float)
    b2 = np.cross(b3, heading)
    if np.linalg.norm(b2) < 1e-9:
        # Avoid the heading/force singularity for custom references.
        heading = np.eye(3)[np.argmin(np.abs(b3))]
        b2 = np.cross(b3, heading)
    b2 /= np.linalg.norm(b2)
    return np.column_stack((np.cross(b2, b3), b2, b3))


def allocation_matrix(vehicle: VehicleConfig) -> np.ndarray:
    """Motor order: (+x,-y), (-x,+y), (+x,+y), (-x,-y)."""
    kf, km = vehicle.thrust_coefficient, vehicle.torque_coefficient
    lx, ly = vehicle.arm_x, vehicle.arm_y
    return np.array([[kf, kf, kf, kf], [-ly*kf, ly*kf, ly*kf, -ly*kf], [-lx*kf, lx*kf, -lx*kf, lx*kf], [-km, -km, km, km]])


class GeometricController:
    def __init__(self, vehicle: VehicleConfig, gains: ControllerConfig):
        self.vehicle = vehicle
        self.gains = gains
        self.inertia = np.diag(vehicle.inertia)
        self.inverse_allocation = np.linalg.inv(allocation_matrix(vehicle))

    def wrench(self, state: State, reference: Reference) -> np.ndarray:
        p, gains = self.vehicle, self.gains
        e3 = np.array([0.0, 0.0, 1.0])
        force = -gains.kx * (state.position - reference.position) - gains.kv * (state.velocity - reference.velocity) + p.mass * (p.gravity * e3 + reference.acceleration)
        rd = desired_rotation(force, reference.yaw, reference.heading)
        rotation, omega = state.rotation, state.angular_velocity
        e_rotation = 0.5 * vee(rd.T @ rotation - rotation.T @ rd)
        e_omega = omega - rotation.T @ rd @ reference.angular_velocity
        thrust = float(force @ rotation[:, 2])
        moment = -gains.kR * e_rotation - gains.kOmega * e_omega + np.cross(omega, self.inertia @ omega) - self.inertia @ (hat(omega) @ rotation.T @ rd @ reference.angular_velocity - rotation.T @ rd @ reference.angular_acceleration)
        return np.concatenate(([thrust], moment))

    def __call__(self, state: State, reference: Reference) -> np.ndarray:
        squared_speed = self.inverse_allocation @ self.wrench(state, reference)
        # Limit commands to forward rotation and the maximum motor speed.
        return np.sqrt(np.clip(squared_speed, 0.0, self.vehicle.max_motor_speed**2))
