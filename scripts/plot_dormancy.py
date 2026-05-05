"""Plot per-stage dormancy fraction across env steps for DQN_ReDo runs.

Usage:
  python scripts/plot_dormancy.py \
      --results-dir results/XN33/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_ReDo \
      --seeds 0 1 2 \
      --out results/XN33/dormancy_3seeds.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


STAGES = [
    ("dormancy_pre_core", "pre_core", "tab:blue"),
    ("dormancy_core", "core", "tab:orange"),
    ("dormancy_post_core", "post_core", "tab:green"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--downsample", type=int, default=1000,
                    help="Plot every Nth env step (default 1000).")
    args = ap.parse_args()

    per_seed = {}
    for s in args.seeds:
        path = os.path.join(args.results_dir, "data", f"{s}.npz")
        d = np.load(path)
        per_seed[s] = {k: d[k].astype(np.float32) for k, _, _ in STAGES}
    n_steps = len(per_seed[args.seeds[0]][STAGES[0][0]])
    x = np.arange(n_steps)[:: args.downsample]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, label, color in STAGES:
        traces = np.stack(
            [per_seed[s][key][:: args.downsample] for s in args.seeds], axis=0
        )
        for s_i in range(traces.shape[0]):
            ax.plot(x, traces[s_i], color=color, alpha=0.25, linewidth=1)
        mean = traces.mean(axis=0)
        ax.plot(x, mean, color=color, alpha=1.0, linewidth=2.0, label=label)

    ax.set_xlabel("env step")
    ax.set_ylabel("dormancy fraction")
    ax.set_title(
        f"DQN_ReDo dormancy per stage ({len(args.seeds)} seeds, "
        f"{n_steps:,} steps)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
