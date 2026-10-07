"""Free-space demo with no bounds; choose a pattern or edit generic.toml.

uv run python examples/generic_demo.py --pattern butterfly
uv run lab-demo --mode generic --pattern crown

Without --pattern, the trajectory comes directly from config/generic.toml.
"""

import argparse

from lab_sim.cli import example_main
from lab_sim.presets import PATTERNS, apply_pattern


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pattern", choices=list(PATTERNS))
    args, remaining = parser.parse_known_args()
    example_main(argv=remaining, default_config="config/generic.toml", config_transform=(lambda config: apply_pattern(config, args.pattern)) if args.pattern else None)
