"""Run a policy through a replaceable tracker and the quadrotor plant."""

from dataclasses import asdict
import time

import numpy as np

from .bounds import lab_clearance
from .config import ExperimentConfig
from .controller import GeometricController
from .dynamics import Quadrotor
from .trajectory import Lissajous
from .types import Policy, Reference, State, Tracker


def simulate(config: ExperimentConfig, policy: Policy | None = None, tracker: Tracker | None = None, *, initial_reference: Reference | None = None) -> dict:
    """Track a policy; supply initial_reference for stateful/feedback methods.

    Without it, the policy is evaluated once on a placeholder to choose the
    initial position/velocity. With it, every policy call observes the vehicle.
    """
    config.validate()
    started = time.perf_counter()
    policy = policy if policy is not None else Lissajous(config.trajectory, config.vehicle.gravity)
    tracker = tracker if tracker is not None else GeometricController(config.vehicle, config.controller)
    sim, vehicle = config.simulation, config.vehicle
    plant = Quadrotor(vehicle, np.asarray(sim.wind), np.asarray(sim.external_force))
    placeholder = State(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), np.zeros(4))
    initial = initial_reference if initial_reference is not None else policy(0.0, placeholder)
    hover_speed = np.sqrt(vehicle.mass * vehicle.gravity / (4 * vehicle.thrust_coefficient))
    state = State(np.asarray(initial.position, dtype=float) + np.asarray(sim.initial_position_error), np.asarray(initial.velocity, dtype=float).copy(), np.eye(3), np.zeros(3), np.full(4, hover_speed))
    history = {key: [] for key in ("time", "position", "reference", "reference_velocity", "reference_acceleration", "velocity", "rotation", "motor_speed", "error", "clearance", "reference_clearance")}
    errors, clearances, reference_clearances = [], [], []
    saturated_steps = 0
    saturation_count = 0
    next_sample = 0.0
    t = 0.0
    while True:
        reference = policy(t, state)
        target = np.concatenate((reference.position, reference.velocity, reference.acceleration, [reference.yaw], reference.angular_velocity, reference.angular_acceleration))
        if target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError(f"Policy returned an invalid Reference at t={t:.3f}")
        if reference.heading is not None and (np.asarray(reference.heading).shape != (3,) or not np.isfinite(reference.heading).all() or np.linalg.norm(reference.heading) < 1e-9):
            raise ValueError(f"Policy returned an invalid heading at t={t:.3f}")
        command = np.asarray(tracker(state, reference), dtype=float)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError(f"Tracker must return four finite motor speeds at t={t:.3f}")
        error = float(np.linalg.norm(state.position - reference.position))
        clearance = lab_clearance(state.position, config.lab, config.lab.vehicle_radius) if config.lab.enabled else None
        reference_clearance = lab_clearance(reference.position, config.lab, config.lab.vehicle_radius) if config.lab.enabled else None
        errors.append(error)
        clearances.append(clearance)
        reference_clearances.append(reference_clearance)
        if t + 1e-9 >= next_sample or t >= sim.duration:
            values = (t, state.position.tolist(), reference.position.tolist(), reference.velocity.tolist(), reference.acceleration.tolist(), state.velocity.tolist(), state.rotation.tolist(), state.motor_speed.tolist(), error, clearance, reference_clearance)
            for key, value in zip(history, values):
                history[key].append(value)
            next_sample += sim.sample_dt
        if t >= sim.duration:
            break
        saturated_steps += int(np.any((command <= 0) | (command >= vehicle.max_motor_speed)))
        saturation_count += 1
        h = min(sim.dt, sim.duration - t)
        state = plant.step(state, command, h)
        if not np.isfinite(state.pack()).all() or np.linalg.norm(state.position) > 1e5:
            raise ValueError(f"Simulation diverged at t={t:.3f}; check the policy and gains")
        t = min(sim.duration, t + h)
        if sim.duration - t < 1e-9:
            t = sim.duration
    metrics = {
        "rms_error": float(np.sqrt(np.mean(np.square(errors)))),
        "max_error": max(errors), "final_error": errors[-1],
        "minimum_clearance": min(clearances) if config.lab.enabled else None,
        "boundary_violations": sum(value is not None and value < 0 for value in clearances),
        "reference_violations": sum(value is not None and value < 0 for value in reference_clearances),
        "saturation_fraction": saturated_steps / max(1, saturation_count),
        "steps": len(errors), "compute_seconds": time.perf_counter() - started,
    }
    return {"schema_version": 1, "config": asdict(config), "history": history, "metrics": metrics}
