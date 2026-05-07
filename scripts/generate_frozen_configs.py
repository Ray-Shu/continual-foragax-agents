#!/usr/bin/env python3
"""Generate frozen config files by adding freeze_steps to metaParameters."""

import argparse
import json
from pathlib import Path


def generate_frozen_configs(config_dir):
    """Generate frozen config files for all JSON configs in the specified directory."""
    # Directory containing the config files
    config_path = Path(config_dir)

    # Process each JSON file
    for config_file in config_path.glob("*.json"):
        if "_frozen" in config_file.stem:
            continue

        with open(config_file, "r") as f:
            config = json.load(f)

        # Generate configs for 5M freeze steps
        for freeze_steps, suffix in [(5000000, "5M")]:
            frozen_config = config.copy()
            frozen_config["agent"] += f"_frozen_{suffix}"

            # Add freeze_steps to metaParameters
            if "metaParameters" not in frozen_config:
                frozen_config["metaParameters"] = {}
            frozen_config["metaParameters"]["freeze_steps"] = freeze_steps

            # Create output filename in the same directory
            output_file = config_path / f"{config_file.stem}_frozen_{suffix}.json"

            # Write the modified config
            with open(output_file, "w") as f:
                json.dump(frozen_config, f, indent=4)

            print(f"Generated {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate frozen config files.")
    parser.add_argument("config_dir", help="Directory containing the config files")
    args = parser.parse_args()
    generate_frozen_configs(args.config_dir)
