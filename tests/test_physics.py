from dataclasses import replace

import numpy as np
import pytest

from lab_sim.config import ControllerConfig, VehicleConfig
from lab_sim.controller import GeometricController, allocation_matrix, desired_rotation, hat
from lab_sim.dynamics import Quadrotor
from lab_sim.flatness import flat_state_to_reference
from lab_sim.types import Reference, State


def hover_state(vehicle):
    motor = np.sqrt(vehicle.mass * vehicle.gravity / (4 * vehicle.thrust_coefficient))
    return State(np.array([0., 0., 2.]), np.zeros(3), np.eye(3), np.zeros(3), np.full(4, motor))


def test_hover_is_a_plant_equilibrium_and_tracker_commands_hover():
    vehicle = VehicleConfig()
    state = hover_state(vehicle)
    ref = Reference(state.position.copy(), np.zeros(3), np.zeros(3))
    tracker = GeometricController(vehicle, ControllerConfig())
    np.testing.assert_allclose(tracker.wrench(state, ref), [6.6708, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(tracker(state, ref), state.motor_speed, atol=1e-12)
    plant = Quadrotor(vehicle, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(plant.derivative(state.pack(), state.motor_speed), 0, atol=1e-12)


def test_motor_order_and_yaw_torque():
    vehicle = VehicleConfig()
    # Hand-computed: increasing motor 1 produces negative roll, pitch and yaw.
    wrench = allocation_matrix(vehicle) @ np.array([1e6, 0, 0, 0])
    np.testing.assert_allclose(wrench, [1.91, -0.1528, -0.1528, -0.27])
    # Motors 3 and 4 produce positive yaw.
    assert (allocation_matrix(vehicle) @ np.array([0, 0, 1e6, 1e6]))[3] == pytest.approx(0.54)


def test_rotor_acceleration_torque_and_drag():
    vehicle = VehicleConfig()
    state = hover_state(vehicle)
    state.velocity = np.array([2., 0., 0.])
    state.angular_velocity = np.array([0., 0., .5])
    command = state.motor_speed.copy()
    command[0] += 20.0 # motor 1 acceleration = 1000 rad/s²
    plant = Quadrotor(vehicle, np.zeros(3), np.zeros(3))
    derivative = State.unpack(plant.derivative(state.pack(), command))
    assert derivative.velocity[0] == pytest.approx(-.4 / .680)
    assert derivative.motor_speed[0] == pytest.approx(1000.)
    # Rotor acceleration torque -J_m*1000, plus quadratic yaw drag -.003*.5².
    assert derivative.angular_velocity[2] == pytest.approx((-6.62e-3 - .00075) / .003)
    np.testing.assert_allclose(derivative.rotation, hat(state.angular_velocity))


def test_rotor_momentum_contributes_to_body_cross_term():
    vehicle = replace(VehicleConfig(), angular_drag=0)
    state = hover_state(vehicle)
    state.motor_speed = np.array([100., 200., 300., 400.])
    state.angular_velocity = np.array([1., 2., 3.])
    plant = Quadrotor(vehicle, np.zeros(3), np.zeros(3))
    derivative = State.unpack(plant.derivative(state.pack(), state.motor_speed))
    # Literal rotor thrust/lever-arm sums, independent of the allocation matrix.
    f = 1.91e-6 * np.array([10000., 40000., 90000., 160000.])
    torque = np.array([.08*(-f[0]+f[1]+f[2]-f[3]), .08*(-f[0]+f[1]-f[2]+f[3]), 2.7e-7*(-10000.-40000.+90000.+160000.)])
    angular_momentum = np.array([.005, .010, .009 + 6.62e-6*400.])
    expected = (torque - np.cross([1., 2., 3.], angular_momentum)) / [.005, .005, .003]
    np.testing.assert_allclose(derivative.angular_velocity, expected, atol=1e-12)


def test_flat_hover_with_rotating_yaw_and_feedforward():
    ref = flat_state_to_reference([0,0,2], [0,0,0], [0,0,0], yaw=0, yaw_rate=3/8)
    np.testing.assert_allclose(ref.heading, [1,0,0], atol=1e-12)
    np.testing.assert_allclose(ref.angular_velocity, [0,0,.375], atol=1e-12)
    np.testing.assert_allclose(ref.angular_acceleration, 0, atol=1e-12)
    tracker = GeometricController(VehicleConfig(), ControllerConfig())
    np.testing.assert_allclose(tracker.wrench(hover_state(VehicleConfig()), ref), [6.6708, 0, 0, .045], atol=1e-12)


def test_rotation_stays_on_so3_and_motor_commands_are_bounded():
    vehicle = VehicleConfig()
    state = hover_state(vehicle)
    state.angular_velocity = np.array([.3, -.2, .4])
    plant = Quadrotor(vehicle, np.zeros(3), np.zeros(3))
    for _ in range(100):
        state = plant.step(state, state.motor_speed, .01)
    np.testing.assert_allclose(state.rotation.T @ state.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(state.rotation) == pytest.approx(1)
    ref = Reference(np.array([100., 100., 100.]), np.zeros(3), np.zeros(3))
    command = GeometricController(vehicle, ControllerConfig())(state, ref)
    assert np.min(command) >= 0
    assert np.max(command) <= vehicle.max_motor_speed


def test_desired_rotation_handles_heading_singularity():
    rotation = desired_rotation(np.array([1.,0.,0.]), yaw=0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)
