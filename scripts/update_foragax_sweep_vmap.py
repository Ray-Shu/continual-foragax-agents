#!/usr/bin/env python3
"""Update foragax-sweep slurm.sh files to use vmap cluster configs."""

import glob
import re


def main():
    # Find all foragax-sweep slurm.sh files
    pattern = "experiments/**/foragax-sweep/**/slurm.sh"
    for slurm_path in glob.glob(pattern, recursive=True):
        print(f"Processing {slurm_path}")

        with open(slurm_path, "r") as f:
            content = f.read()

        # Replace CPU cluster configs with the canonical vmap config (3h time).
        updated_content = re.sub(
            r"clusters/vulcan-cpu(?:-c8)?\.json(?: --time=\d+:\d+:\d+)?",
            "clusters/vulcan-gpu-vmap.json --time=3:00:00",
            content,
        )

        if updated_content != content:
            with open(slurm_path, "w") as f:
                f.write(updated_content)
            print(f"Updated {slurm_path}")
        else:
            print(f"No changes needed for {slurm_path}")


if __name__ == "__main__":
    main()
