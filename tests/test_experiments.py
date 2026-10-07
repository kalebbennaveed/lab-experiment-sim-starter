from pathlib import Path

import numpy as np
import pytest

from lab_sim import Lissajous, load_config, simulate
from lab_sim.bounds import lab_clearance
from lab_sim.presets import PATTERNS, apply_pattern

ROOT = Path(__file__).resolve().parents[1]


def test_lissajous_derivatives_and_mesch_initial_point():
    config = load_config(ROOT / "config/lab.toml")
    trajectory = Lissajous(config.trajectory)
    np.testing.assert_allclose(trajectory.derivative(0), [-.9,0,7], atol=1e-12)
    np.testing.assert_allclose(trajectory.derivative(0,1), [0,.7,0], atol=1e-12)
    np.testing.assert_allclose(trajectory.derivative(0,2), [-.4296875,0,0], atol=1e-12)
    t, h = 2., 1e-4
    np.testing.assert_allclose((trajectory.derivative(t+h)-trajectory.derivative(t-h))/(2*h), trajectory.derivative(t,1), atol=1e-8)


def test_lab_polygon_notch_height_and_vehicle_clearance():
    lab = load_config(ROOT / "config/lab.toml").lab
    assert lab_clearance(np.array([0.,0.,7.]),lab,.15) == pytest.approx(.85)
    assert lab_clearance(np.array([3.,0.,7.]),lab) < 0
    assert lab_clearance(np.array([0.,0.,8.1]),lab) < 0
    assert lab_clearance(np.array([-3.55,-1.4,1.]),lab) < 0 # left-wall notch
    assert lab_clearance(np.array([2.4,0.,1.]),lab) == pytest.approx(0)


def test_nominal_tracking_converges_and_remains_inside_lab():
    config = load_config(ROOT / "config/lab.toml")
    config.simulation.duration = 5.013 # include a final partial integration step
    result = simulate(config)
    assert result["metrics"]["rms_error"] < .2
    assert result["metrics"]["final_error"] < .08
    assert result["metrics"]["boundary_violations"] == 0
    assert result["history"]["time"][-1] == 5.013
    assert len(result["history"]["time"]) == len(result["history"]["rotation"])


def test_generic_tracking_is_translation_invariant_without_bounds():
    config = load_config(ROOT / "config/generic.toml")
    config.simulation.duration = 2
    original = simulate(config)
    translation = np.array([100., -70., -30.])
    config.trajectory.offset = (np.asarray(config.trajectory.offset)+translation).tolist()
    translated = simulate(config)
    np.testing.assert_allclose(np.asarray(translated["history"]["position"])-translation, original["history"]["position"], atol=1e-10)
    assert translated["metrics"]["minimum_clearance"] is None
    assert translated["metrics"]["boundary_violations"] == 0
    assert all(value is None for value in translated["history"]["clearance"])


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_gallery_curves_close_and_track(pattern):
    config = apply_pattern(load_config(ROOT / "config/generic.toml"), pattern)
    path = Lissajous(config.trajectory)
    np.testing.assert_allclose(path.derivative(0),path.derivative(config.simulation.duration),atol=1e-12)
    result = simulate(config)
    assert result["metrics"]["rms_error"] < .3
    assert result["metrics"]["final_error"] < .2
    assert not result["config"]["lab"]["enabled"]


def test_invalid_time_step_is_rejected():
    config = load_config(ROOT / "config/lab.toml")
    config.simulation.dt = .03
    with pytest.raises(ValueError,match="dt must be"):
        simulate(config)
