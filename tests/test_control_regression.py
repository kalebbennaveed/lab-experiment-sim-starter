"""Fixed numerical regression cases for the control pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from lab_sim import GeometricController, Lissajous, State, load_config
from lab_sim.dynamics import Quadrotor

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/control_reference.json").read_text())


def fixture_state():
    state = FIXTURE["state"]
    return State(np.array(state["position"]), np.array(state["velocity"]), np.array(state["rotation_column_major"]).reshape(3,3,order="F"), np.array(state["angular_velocity"]), np.array(state["motor_speed"]))


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: f"t={case['time']}")
def test_flat_state_and_geometric_wrench_match_reference(case):
    config = load_config(ROOT / "config/lab.toml")
    reference = Lissajous(config.trajectory,config.vehicle.gravity)(case["time"],fixture_state())
    flattened = np.concatenate((reference.position,reference.velocity,reference.acceleration,reference.heading,reference.angular_velocity,reference.angular_acceleration))
    np.testing.assert_allclose(flattened,case["flat_state"],rtol=1e-12,atol=1e-12)
    wrench = GeometricController(config.vehicle,config.controller).wrench(fixture_state(),reference)
    np.testing.assert_allclose(wrench,case["wrench"],rtol=1e-12,atol=1e-12)


def test_rigid_body_and_motor_derivatives_match_reference():
    config = load_config(ROOT / "config/lab.toml")
    fixture = FIXTURE["dynamics"]
    plant = Quadrotor(config.vehicle,np.array(fixture["wind"]),np.array(fixture["external_force"]))
    derivative = State.unpack(plant.derivative(fixture_state().pack(),np.array(fixture["command"])))
    column_major = np.concatenate((derivative.position,derivative.velocity,derivative.rotation.ravel(order="F"),derivative.angular_velocity,derivative.motor_speed))
    np.testing.assert_allclose(column_major,fixture["derivative_column_major"],rtol=1e-12,atol=1e-12)
