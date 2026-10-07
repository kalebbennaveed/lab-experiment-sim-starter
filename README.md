# Lab experiment simulation starter

A self-contained Python starter for quadrotor tracking experiments, with a uv environment,
plain Python examples, and a local web preview that updates when you save code.
It includes quadrotor dynamics, flat-state conversion, and a geometric controller.

Two demo modes share the same vehicle and controller:

- **Lab:** a Flylab Lissajous reference, floor polygon, and 0–8 m height
  range. Shows vehicle clearance and reference boundary crossings.
- **Generic:** free-space tracking with **no bounds**, floor collision, or
  geofence. Place the trajectory anywhere; the camera fits the full path.

## Start with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```bash
uv sync --locked
uv run lab-demo --open
```

Open **http://127.0.0.1:8000**. Use **Lab demo / Free space** to switch modes.
In free space, choose a curve from the Lissajous menu. The 3D view supports
orbit, zoom, top/side views, play/pause, scrubbing, playback speed, and looping.
It displays reference and actual flight paths, tracking error, motor saturation,
and lab clearance where applicable. You can download the latest run as JSON.

`uv sync` creates `.venv/`; `uv run` uses it without manual activation. The
tracked `uv.lock` pins dependencies. Python 3.12 is selected by `.python-version`;
the package supports Python 3.11 and later. NumPy is the only runtime dependency.
The web UI uses local HTML, CSS, and canvas JavaScript with no build tool or CDN.

## Run the examples

```bash
# Lab reference and configured Flylab bounds
uv run python examples/lab_demo.py --output outputs/lab.json

# Generic trajectory directly from editable config/generic.toml
uv run python examples/generic_demo.py --output outputs/generic.json

# Four different 3D Lissajous curves
uv run python examples/generic_demo.py --pattern orbit
uv run python examples/generic_demo.py --pattern butterfly
uv run python examples/generic_demo.py --pattern ribbon
uv run python examples/generic_demo.py --pattern crown

# Export all four trajectories and their diagnostics
uv run python examples/gallery.py

# A small example showing where to add your algorithm
uv run python examples/custom_policy.py

# Replace the nominal method with double-integrator acceleration commands
uv run python examples/double_integrator_method.py
```

| Curve | Frequency ratio x:y:z | Shape |
| --- | --- | --- |
| Orbit | 2:3:4 | Interlaced 3D loops |
| Butterfly | 3:2:4 | Symmetric crossing lobes |
| Ribbon | 1:2:3 | A twisting figure-eight ribbon |
| Crown | 1:1:5 | A circular sweep with five vertical waves |

The gallery uses a base frequency of 0.15 rad/s and a common period of about
41.89 seconds. Offsets and yaw remain configurable. Presets replace amplitude,
frequency, phase, and duration; choose **From config** in the browser to use
every trajectory value from `config/generic.toml`. `--duration` overrides the
run duration, including a preset's duration.

```bash
uv run lab-demo --mode generic --pattern crown --open
uv run lab-demo --example examples/custom_policy.py --open
uv run lab-demo --mode generic --port 8001
uv run lab-sim --config config/generic.toml --duration 5
```

## Add your code

For a method that returns double-integrator commands, follow the
[integration guide](examples/README.md) and the working
[double-integrator example](examples/double_integrator_method.py).

```text
your method -> acceleration [ax, ay, az]
    -> desired double-integrator [position, velocity]
    -> flat-state conversion -> geometric controller -> quadrotor
```

Replace the marked acceleration-law block in `MyAccelerationMethod`. The
adapter integrates held commands into a desired trajectory; the same guide
shows how to pass desired states and accelerations directly from a planner
that already produces a trajectory. Preview either config with:

```bash
uv run lab-demo --example examples/double_integrator_method.py --port 8001 --open
uv run lab-demo --example examples/double_integrator_method.py --config config/generic.toml --port 8001 --open
```

The experiment pipeline is:

```text
policy(t, state) -> Reference
    -> GeometricController(state, reference) -> four motor speed commands
    -> Quadrotor dynamics -> next State
```

Start with [examples/custom_policy.py](examples/custom_policy.py). Replace its
`MyPolicy.__call__` body with your planning or control algorithm. The policy can
inspect position, velocity, body-to-world rotation, body angular velocity, and
motor speeds. Return desired position, velocity, and acceleration through
`flat_state_to_reference` to compute the heading and angular feedforward.
Use consistent derivatives when changing a reference trajectory.

For a different low-level controller, pass `tracker=` to `simulate`. It receives
`(state, reference)` and returns four desired motor speeds in **rad/s**, in the
documented motor order. A custom example can call `simulate(config,
policy=my_policy, tracker=my_tracker)` and serialize the result with
`json.dumps(result, allow_nan=False)`. The shared `example_main` helper provides
`--config`, `--duration`, `--output`, and `--json` for examples using the default
tracker. Live-preview scripts must accept `--config` and `--output`.

To track a path elsewhere, edit `trajectory.offset` in `config/generic.toml`.
For example, `[20.0, -15.0, 12.0]` translates the entire curve. This mode never
checks the Flylab polygon. It still applies the same physical motor limits.
The grid shown in this mode is a visual guide, not a floor or constraint.

While `lab-demo` is running, saving `src/**/*.py`, `examples/**/*.py`, or
`config/**/*.toml` reruns the selected experiment in a fresh Python process.
Saving `web/` assets reloads the browser. An invalid edit displays the traceback
and keeps the last successful flight visible; fixing the edit automatically
retries. Changes to server networking code require restarting `lab-demo`, and
dependency changes require `uv sync` and a restart. File watching is local to
this checkout, rather than a hosted deployment service.

## Models and conventions

The implementation is included in `src/lab_sim/`:

- `dynamics.py` simulates a 22-state rigid body with gravity, thrust, quadratic
  linear/angular drag, motor lag, rotor acceleration torque, and rotor angular
  momentum. It uses fixed-step RK4 with commands held for each controller step
  and projects the rotation onto SO(3) after integration.
- `controller.py` computes position/velocity feedback, SO(3) attitude and
  angular-velocity errors, thrust, moment feedback/feedforward, and motor allocation.
- `flatness.py` converts desired position, velocity, acceleration, and yaw into
  heading and angular feedforward, assuming zero jerk and snap.
- `trajectory.py` supplies analytic Lissajous position, velocity, and acceleration.
- `config/lab.toml` contains the vehicle parameters, controller gains, reference
  trajectory, lab polygon, and height limits. `config/generic.toml` defines a
  free-space experiment using the same vehicle and controller defaults.

World z points up, rotation maps body vectors into world coordinates, angular
velocity uses the body frame, and all values use SI units. Motor positions are
`(+x,-y)`, `(-x,+y)`, `(+x,+y)`, `(-x,-y)`, with directions `[1,1,-1,-1]`.
The default gains are `kx=3`, `kv=5`, `kR=0.90`, `kOmega=0.120`. Yaw is
`3*t/8` by default. Allocated motor commands are limited to forward rotation
and the configured maximum motor speed.

The nominal demo tracks an analytic path with a 0.01 s controller step.
The initial state has a small position offset to make convergence visible.
Lab bounds diagnose crossings; they do not constrain the geometric controller
or stop the vehicle.

## Repository layout

```text
config/        lab and generic experiment settings
examples/      executable experiments and extension examples
src/lab_sim/   reference, geometric tracker, plant, runner, and live server
web/           local browser visualizer
tests/         physical invariants, tracking, modes, and live reload checks
```

Run the checks with `uv run pytest`. The tests cover hover, motor torque signs,
angular feedforward, rigid-body rotation, lab geometry, free-space translation,
all four presets, and edit/error/recovery behavior in the experiment runner.
The numerical regression cases in `tests/fixtures/control_reference.json`
cover flat-state conversion, geometric control, and dynamics for a tilted,
moving vehicle. Tests compare the outputs against fixed values at a tolerance
of 1e-12. All tests run in the project's Python environment.
