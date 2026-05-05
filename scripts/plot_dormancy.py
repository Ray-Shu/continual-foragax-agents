"""Plot per-stage dormancy fraction across env steps for DQN_ReDo runs.

Single-run usage:
  python scripts/plot_dormancy.py \
      --results-dir results/.../DQN_ReDo \
      --seeds 0 1 2 \
      --out results/dormancy.png

Compare two runs (e.g. ReDo vs vanilla DQN observed):
  python scripts/plot_dormancy.py \
      --results-dir results/.../DQN_ReDo \
      --compare-dir results/.../DQN_ReDo_observe \
      --seeds 0 1 2 \
      --out results/dormancy_compare.png
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


def load_run(results_dir, seeds):
    out = {}
    for s in seeds:
        path = os.path.join(results_dir, "data", f"{s}.npz")
        d = np.load(path)
        out[s] = {k: d[k].astype(np.float32) for k, _, _ in STAGES}
    return out


def plot_run_stage(ax, x, run, key, downsample, color_a, color_b=None, label_a=None, label_b=None):
    """Plot all seed traces (low alpha) and mean (full alpha) for one stage."""
    traces = np.stack(
        [run[s][key][::downsample] for s in run.keys()], axis=0
    )
    color = color_a if color_b is None else color_a
    for s_i in range(traces.shape[0]):
        ax.plot(x, traces[s_i], color=color, alpha=0.22, linewidth=1)
    mean = traces.mean(axis=0)
    ax.plot(x, mean, color=color, alpha=1.0, linewidth=2.0, label=label_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--compare-dir", default=None)
    ap.add_argument("--label-a", default="DQN_ReDo")
    ap.add_argument("--label-b", default="DQN (observe)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--downsample", type=int, default=1000)
    args = ap.parse_args()

    run_a = load_run(args.results_dir, args.seeds)
    n_steps = len(run_a[args.seeds[0]][STAGES[0][0]])
    x = np.arange(n_steps)[:: args.downsample]

    run_b = load_run(args.compare_dir, args.seeds) if args.compare_dir else None

    n_panels = len(STAGES)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9.0, 2.5 * n_panels), sharex=True, sharey=True
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (key, label, color) in zip(axes, STAGES):
        plot_run_stage(ax, x, run_a, key, args.downsample,
                       color_a=color, label_a=args.label_a)
        if run_b is not None:
            plot_run_stage(ax, x, run_b, key, args.downsample,
                           color_a="tab:gray", label_a=args.label_b)
        ax.set_title(f"stage: {label}")
        ax.set_ylabel("dormancy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("env step")
    suptitle = (
        f"{args.label_a} vs {args.label_b} dormancy "
        f"({len(args.seeds)} seeds, {n_steps:,} steps)"
        if run_b is not None
        else f"{args.label_a} dormancy ({len(args.seeds)} seeds, {n_steps:,} steps)"
    )
    fig.suptitle(suptitle)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
