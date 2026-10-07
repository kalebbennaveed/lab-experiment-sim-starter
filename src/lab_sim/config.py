"""Load an explicit experiment configuration; no global controller settings."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

import numpy as np


@dataclass
class VehicleConfig:
    mass: float = 0.680
    inertia: list[float] = field(default_factory=lambda: [0.005, 0.005, 0.003])
    gravity: float = 9.81
    arm_x: float = 0.08
    arm_y: float = 0.08
    motor_time_constant: float = 0.02
    rotor_inertia: float = 6.62e-6
    thrust_coefficient: float = 1.91e-6
    torque_coefficient: float = 2.7e-7
    linear_drag: float = 0.1
    angular_drag: float = 0.003
    max_motor_speed: float = 2520.0

    def validate(self) -> None:
        for name in ("mass", "gravity", "arm_x", "arm_y", "motor_time_constant", "thrust_coefficient", "torque_coefficient", "max_motor_speed"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"vehicle.{name} must be finite and positive")
        for name in ("rotor_inertia", "linear_drag", "angular_drag"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"vehicle.{name} must be finite and nonnegative")
        _vector("vehicle.inertia", self.inertia)
        if np.min(self.inertia) <= 0:
            raise ValueError("vehicle.inertia must be positive")


@dataclass
class ControllerConfig:
    kx: float = 3.0
    kv: float = 5.0
    kR: float = 0.90
    kOmega: float = 0.120


@dataclass
class TrajectoryConfig:
    amplitude: list[float] = field(default_factory=lambda: [1.1, 1.4, 0.0])
    frequency: list[float] = field(default_factory=lambda: [0.625, 0.5, 0.75])
    phase: list[float] = field(default_factory=lambda: [np.pi / 2, 0.0, 0.0])
    offset: list[float] = field(default_factory=lambda: [-2.0, 0.0, 7.0])
    yaw: float = 0.0
    yaw_rate: float = 3.0 / 8.0


@dataclass
class SimulationConfig:
    duration: float = 30.0
    dt: float = 0.01
    sample_dt: float = 0.05
    initial_position_error: list[float] = field(default_factory=lambda: [0.2, -0.2, -0.1])
    wind: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    external_force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class LabConfig:
    enabled: bool = True
    name: str = "Flylab"
    floor: list[list[float]] = field(default_factory=lambda: [[-3.60, -4.80], [0.0, -4.80], [2.40, -4.80], [2.40, 0.0], [2.40, 1.80], [0.0, 1.80], [-3.60, 1.80], [-3.40, -1.40]])
    z_min: float = 0.0
    z_max: float = 8.0
    vehicle_radius: float = 0.15


def _vector(name: str, value: list[float]) -> None:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain three finite values")


@dataclass
class ExperimentConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    lab: LabConfig = field(default_factory=LabConfig)

    def validate(self) -> None:
        self.vehicle.validate()
        for name, value in vars(self.controller).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"controller.{name} must be finite and nonnegative")
        for name in ("amplitude", "frequency", "phase", "offset"):
            _vector(f"trajectory.{name}", getattr(self.trajectory, name))
        if not np.isfinite([self.trajectory.yaw, self.trajectory.yaw_rate]).all():
            raise ValueError("trajectory.yaw and yaw_rate must be finite")
        sim = self.simulation
        for name in ("duration", "dt", "sample_dt"):
            value = getattr(sim, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"simulation.{name} must be finite and positive")
        if sim.dt > min(0.01, self.vehicle.motor_time_constant / 2):
            raise ValueError("dt must be <= 0.01 s and <= motor_time_constant / 2 for this RK4 model")
        if sim.sample_dt < sim.dt or sim.duration / sim.dt > 200_000:
            raise ValueError("sample_dt must be >= dt; limit an experiment to 200,000 steps")
        for name in ("initial_position_error", "wind", "external_force"):
            _vector(f"simulation.{name}", getattr(sim, name))
        lab = self.lab
        polygon = np.asarray(lab.floor, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError("lab.floor must be a polygon with at least three finite [x, y] vertices")
        signed_area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
        if abs(signed_area) < 1e-9:
            raise ValueError("lab.floor must enclose a nonzero area")
        if not np.isfinite([lab.z_min, lab.z_max, lab.vehicle_radius]).all() or lab.z_max <= lab.z_min or lab.vehicle_radius < 0:
            raise ValueError("lab requires z_max > z_min and a nonnegative vehicle_radius")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    sections = {"vehicle": VehicleConfig, "controller": ControllerConfig, "trajectory": TrajectoryConfig, "simulation": SimulationConfig, "lab": LabConfig}
    unknown = raw.keys() - sections.keys()
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    cfg = ExperimentConfig(**{name: cls(**raw.get(name, {})) for name, cls in sections.items()})
    cfg.validate()
    return cfg
