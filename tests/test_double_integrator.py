from pathlib import Path
import runpy

import numpy as np
import pytest

from lab_sim import DoubleIntegratorPolicy, State, double_integrator_reference, double_integrator_step, load_config, simulate

ROOT = Path(__file__).resolve().parents[1]


def vehicle_state():
    return State(np.array([10., 20., 30.]), np.array([1., 2., 3.]), np.eye(3), np.zeros(3), np.zeros(4))


def test_constant_acceleration_step_matches_known_motion():
    # Half a second from rest: 2 m/s² in x, -4 m/s² in y.
    state = double_integrator_step(np.array([1., 2., 3., 0., 0., 0.]), np.array([2., -4., 0.]), .5)
    np.testing.assert_allclose(state, [1.25, 1.5, 3., 1., -2., 0.])


def test_command_hold_and_method_timing_preserve_reference_motion():
    calls = []

    def method(t, planned, measured):
        calls.append((t, planned.copy(), measured.copy()))
        planned[:] = -999 # inputs are copies
        measured[:] = -999
        return np.array([2., 0., 0.]) if len(calls) == 1 else np.array([-2., 0., 0.])

    policy = DoubleIntegratorPolicy(method, np.array([0., 0., 3., 0., 0., 0.]), command_dt=.05)
    state = vehicle_state()
    policy(0, state)
    policy(0, state) # no duplicate method call or integration
    for t in [.01, .02, .03, .04]:
        policy(t, state)
    assert len(calls) == 1
    np.testing.assert_allclose(policy.planned_state, [.0016, 0., 3., .08, 0., 0.])
    at_update = policy(.05, state)
    assert len(calls) == 2
    np.testing.assert_allclose(at_update.position, [.0025, 0., 3.])
    np.testing.assert_allclose(at_update.velocity, [.1, 0., 0.])
    np.testing.assert_allclose(at_update.acceleration, [-2., 0., 0.])
    afterward = policy(.06, state)
    np.testing.assert_allclose(afterward.position, [.0034, 0., 3.])
    np.testing.assert_allclose(afterward.velocity, [.08, 0., 0.])
    np.testing.assert_allclose(state.position, [10., 20., 30.])
    np.testing.assert_allclose(calls[0][2], [10., 20., 30., 1., 2., 3.])


def test_stateful_method_initializes_from_real_vehicle_and_commands_hover():
    config = load_config(ROOT / "config/generic.toml")
    config.simulation.duration = .2
    config.simulation.initial_position_error = [0., 0., 0.]
    observed = []
    initial_state = np.array([20., -15., 12., 0., 0., 0.])

    def method(t, planned, measured):
        observed.append((t, measured.copy()))
        return np.zeros(3)

    policy = DoubleIntegratorPolicy(method, initial_state, command_dt=.05, yaw_rate=0)
    initial = double_integrator_reference(initial_state, np.zeros(3), yaw_rate=0)
    result = simulate(config, policy=policy, initial_reference=initial)
    assert len(observed) == 5 # 0, .05, .10, .15, .20, with no placeholder call
    np.testing.assert_allclose(observed[0][1], initial_state)
    assert result["metrics"]["max_error"] < 1e-12
    np.testing.assert_allclose(result["history"]["reference_acceleration"], 0)
    np.testing.assert_allclose(result["history"]["reference_velocity"], 0)
    assert len(result["history"]["reference_acceleration"]) == len(result["history"]["time"])


@pytest.mark.parametrize("config_name", ["lab", "generic"])
def test_worked_example_tracks_with_default_mesch_controller(config_name):
    example = runpy.run_path(str(ROOT / "examples/double_integrator_method.py"))
    config = load_config(ROOT / f"config/{config_name}.toml")
    config.simulation.duration = 5
    result = simulate(config, policy=example["make_policy"](config), initial_reference=example["make_initial_reference"](config))
    assert result["metrics"]["rms_error"] < .2
    assert result["metrics"]["final_error"] < .08
    assert result["metrics"]["boundary_violations"] == 0


@pytest.mark.parametrize("command", [np.array([1., 2.]), np.array([0., 0., float('nan')])])
def test_invalid_acceleration_command_fails_before_flatness(command):
    policy = DoubleIntegratorPolicy(lambda *args: command, np.zeros(6))
    with pytest.raises(ValueError, match="3 finite values"):
        policy(0, vehicle_state())


def test_backward_time_is_rejected():
    policy = DoubleIntegratorPolicy(lambda *args: np.zeros(3), np.zeros(6))
    policy(0, vehicle_state())
    policy(.01, vehicle_state())
    with pytest.raises(ValueError, match="nondecreasing"):
        policy(0, vehicle_state())
