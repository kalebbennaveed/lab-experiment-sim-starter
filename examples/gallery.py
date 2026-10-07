"""Export four fancy 3D Lissajous experiments as ordinary JSON files.

uv run python examples/gallery.py
Preview any one: uv run lab-demo --mode generic --pattern ribbon
"""

import argparse
import json
from pathlib import Path

from lab_sim import load_config, simulate
from lab_sim.presets import PATTERNS, apply_pattern


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gallery"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in PATTERNS:
        config = apply_pattern(load_config("config/generic.toml"), name)
        result = simulate(config)
        destination = args.output_dir / f"{name}.json"
        destination.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        print(f"{name}: RMS {result['metrics']['rms_error']:.4f} m -> {destination}")
