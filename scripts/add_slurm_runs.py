#!/usr/bin/env python3
"""Add DQN_CReLU runs to slurm.sh files."""

import glob
import os


def main():
    # Find all slurm.sh in experiments with search-limited-fov, mitigations, or weather
    patterns = [
        "experiments/*-search-limited-fov/**/slurm.sh",
        "experiments/*-mitigations/**/slurm.sh",
        "experiments/*-weather/**/slurm.sh",
    ]

    for pattern in patterns:
        for slurm_path in glob.glob(pattern, recursive=True):
            print(f"Processing {slurm_path}")
            # Determine the config path
            config_dir = os.path.dirname(slurm_path)

            with open(slurm_path, "r") as f:
                content = f.read()

            # Determine runs and cluster based on whether it's foragax or foragax-sweep
            is_foragax = "foragax/" in slurm_path and "foragax-sweep/" not in slurm_path
            runs = 30 if is_foragax else 5
            cluster_frozen, cluster_time = (
                ("vulcan-cpu.json", "3:00:00")
                if is_foragax
                else ("vulcan-gpu-vmap.json", "3:00:00")
            )

            updated = False
            # Add runs for all frozen configs in 9/ and 15/
            for subdir in ["9", "15"]:
                config_subdir = os.path.join(config_dir, subdir)
                if os.path.exists(config_subdir):
                    for config_file in os.listdir(config_subdir):
                        if config_file.endswith(".json") and "_frozen_" in config_file:
                            config_path = os.path.join(config_subdir, config_file)
                            time_arg = (
                                f" --time={cluster_time}" if cluster_time else ""
                            )
                            slurm_line = f"python scripts/slurm.py --cluster clusters/{cluster_frozen}{time_arg} --runs {runs} --entry src/continuing_main.py --force -e {config_path}"
                            if slurm_line not in content:
                                content += f"{slurm_line}\n"
                                updated = True
                                print(f"Added run for {config_path}")

            if updated:
                with open(slurm_path, "w") as f:
                    f.write(content)
                print(f"Updated {slurm_path}")
            else:
                print(f"No new runs added for {slurm_path}")


if __name__ == "__main__":
    main()
