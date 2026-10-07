"""Shared command-line arguments for plain Python experiment examples."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .simulation import simulate
from .types import Policy


def example_main(policy_factory=None, argv=None, default_config="config/lab.toml", config_transform=None, initial_reference_factory=None) -> None:
    parser = argparse.ArgumentParser(description="Run a quadrotor Lissajous tracking experiment.")
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--duration", type=float, help="Override duration in seconds")
    parser.add_argument("--output", type=Path, help="Save trajectory and diagnostics as JSON")
    parser.add_argument("--json", action="store_true", help="Write the complete result to stdout")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config_transform is not None:
        config = config_transform(config)
    if args.duration is not None:
        config.simulation.duration = args.duration
    policy: Policy | None = policy_factory(config) if policy_factory else None
    initial_reference = initial_reference_factory(config) if initial_reference_factory else None
    result = simulate(config, policy=policy, initial_reference=initial_reference)
    payload = json.dumps(result, allow_nan=False, separators=(",", ":"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if args.json:
        print(payload)
    else:
        metrics = result["metrics"]
        print(f"RMS error: {metrics['rms_error']:.4f} m | max: {metrics['max_error']:.4f} m | final: {metrics['final_error']:.4f} m")
        if config.lab.enabled:
            print(f"Minimum clearance: {metrics['minimum_clearance']:.3f} m | boundary violations: {metrics['boundary_violations']} integration samples")
        else:
            print("Free-space experiment; boundary diagnostics disabled")
        print(f"Computed {metrics['steps']} steps in {metrics['compute_seconds']:.2f} s")
        if args.output:
            print(f"Saved {args.output}")


def main() -> None:
    example_main()
