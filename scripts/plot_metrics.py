"""Plot NTK metrics (rank, condition number, churn) over training.

Examples:
    python scripts/plot_metrics.py \
        -e experiments/E136-big/foragax/ForagaxBig-v5 \
        -a DQN -f 9

    python scripts/plot_metrics.py \
        -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 \
        -a PPO_LN_128 -f 9
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # Go up from scripts/ to repo root
sys.path.insert(0, str(ROOT / "src"))


def _normalize_exp(exp: str) -> str:
    """Strip a leading experiments/ or results/ prefix and trailing slash."""
    exp = exp.strip().rstrip("/")
    for prefix in ("experiments/", "results/"):
        if exp.startswith(prefix):
            exp = exp[len(prefix) :]
    return exp


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-e",
        "--exp",
        required=True,
        help="Experiment path, e.g. experiments/E136-big/foragax/ForagaxBig-v5 "
        "(a leading experiments/ or results/ prefix is optional)",
    )
    p.add_argument(
        "-a",
        "--agent",
        required=True,
        help="Agent / algorithm name (e.g. DQN, PPO_LN_128, ActorCriticMLP)",
    )
    p.add_argument(
        "-f", "--fov", required=True, type=int, help="Field-of-view / aperture size"
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Write the plot to the repo root instead of the experiment's "
        "metrics/ folder (handy for quick one-off checks).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    exp = _normalize_exp(args.exp)
    data_path = ROOT / "results" / exp / str(args.fov) / args.agent / "data"
    metrics_dir = ROOT / "experiments" / exp / "metrics"

    # Each panel lists every column that could supply it, paired with a series
    # label.  DQN runs write `churn_norm` / `ntk_rank` / `ntk_cond`; PPO runs write
    # separate `value_*` / `policy_*` columns (hard `*_ntk_rank` plus effective
    # `*_ntk_eff_rank`, no condition number).  Only the columns actually present in
    # the loaded data are plotted, so this works unchanged for either agent (or a
    # mix), drawing one line per available series.
    # Panels are laid out on a grid by their `row`: row 0 holds the churn / NTK
    # metrics, row 1 holds the weight metrics.  Unused cells in a shorter row are
    # hidden.
    PANELS = [
        {
            "row": 0,
            "ylabel": "NTK Rank",
            "title": "NTK Rank Over Time",
            "log": False,
            "series": [
                ("ntk_rank", "DQN"),
                ("value_ntk_rank", "Critic (PPO)"),
                ("policy_ntk_rank", "Actor (PPO)"),
            ],
        },
        {
            "row": 0,
            "ylabel": "NTK Condition Number",
            "title": "NTK Condition Number Over Time",
            "log": True,  # condition numbers span many orders of magnitude
            "series": [
                ("ntk_cond", "DQN"),
                ("value_ntk_cond", "Value (PPO)"),
                ("policy_ntk_cond", "Policy (PPO)"),
            ],
        },
        {
            "row": 0,
            "ylabel": "NTK Effective Rank",
            "title": "NTK Effective (Stable) Rank Over Time",
            "log": False,
            "series": [
                ("value_ntk_eff_rank", "Value (PPO)"),
                ("policy_ntk_eff_rank", "Policy (PPO)"),
            ],
        },
        {
            "row": 1,
            "ylabel": "Churn Norm",
            "title": "Churn Over Time",
            "log": False,
            "series": [
                ("churn_norm", "DQN"),
                ("value_churn", "Value (PPO)"),
                ("policy_churn", "Policy (PPO)"),
            ],
        },
        {
            "row": 1,
            "ylabel": "Weight Norm",
            "title": "Weight Norm Over Time",
            "log": False,
            "series": [
                ("weight_norm", "PPO"),
            ],
        },
        {
            "row": 1,
            "ylabel": "Weight Drift",
            "title": "Weight Drift Over Time",
            "log": False,
            "series": [
                ("weight_drift_total", "Total (PPO)"),
                ("weight_drift_pi", "Actor (PPO)"),
                ("weight_drift_vf", "Critic (PPO)"),
            ],
        },
    ]

    print(f"Data path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")

    # Load the raw per-run npz files from the /data/ folder.  We read these directly
    # (rather than via data.parquet) to access metrics at their native resolution:
    # DQN writes one value per env step (NaN except every ntk_freq), while PPO
    # writes one value per *update*.  The parquet reader would repeat PPO per-update
    # arrays up to per-step length, which makes constant-valued series (e.g. NTK rank)
    # indistinguishable from a single repeated point.
    runs = []
    for f in sorted(Path(data_path).glob("*.npz")):
        with np.load(f) as d:
            runs.append({k: np.asarray(d[k]) for k in d.keys()})
    print(f"Loaded {len(runs)} run(s)")

    def metric_series(col: str):
        """Seed-averaged (x_steps, values) for `col`, or None if absent/empty.

        A metric array has one entry per *measurement* (per env step for DQN, per
        update for PPO).  Its x-axis is recovered in env steps by comparing its
        length to the per-step `rewards` array: `steps_per_point = len(rewards) /
        len(metric)` (1 for DQN, rollout_steps for PPO).  NaN entries (non-metric
        steps/updates) are dropped *after* averaging across seeds.
        """
        arrays = []
        steps_per_point = 1.0
        for r in runs:
            if col not in r:
                continue
            v = np.asarray(r[col], dtype=float).reshape(-1)
            if v.shape[0] == 0:
                continue
            base_len = (
                r["rewards"].reshape(-1).shape[0] if "rewards" in r else v.shape[0]
            )
            steps_per_point = base_len / v.shape[0]
            arrays.append(v)

        if not arrays:
            return None

        # Align seeds to the shortest run, average (ignoring NaN measurements).
        m = min(a.shape[0] for a in arrays)
        stacked = np.stack([a[:m] for a in arrays], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
            mean = np.nanmean(stacked, axis=0)

        x = np.arange(m) * steps_per_point
        finite = np.isfinite(mean)
        if not finite.any():
            return None
        return x[finite], mean[finite]

    # Report which metric columns are present and have computed (finite) values.
    all_metric_cols = [col for panel in PANELS for col, _ in panel["series"]]
    present_cols = [c for c in all_metric_cols if any(c in r for r in runs)]
    for metric in all_metric_cols:
        series = metric_series(metric)
        if metric not in present_cols:
            print(f"{metric}: NOT FOUND in data")
        elif series is None:
            print(f"{metric}: present but no finite values")
        else:
            x, y = series
            print(
                f"{metric}: {len(y)} finite point(s) | "
                f"min={y.min():.4g} max={y.max():.4g} mean={y.mean():.4g}"
            )

    if all(metric_series(c) is None for c in present_cols):
        print("\nWARNING: No finite metric values found!")
        print("The metrics may not have been computed during training.")
        return

    # Plot on a (n_rows x n_cols) grid, where each panel sits in its assigned
    # `row` and columns are filled left-to-right within a row.  n_cols is the
    # width of the widest row; leftover cells are hidden.
    rows = sorted({panel["row"] for panel in PANELS})
    n_rows = len(rows)
    n_cols = max(sum(p["row"] == r for p in PANELS) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    used = set()
    for r in rows:
        row_panels = [p for p in PANELS if p["row"] == r]
        for col_idx, panel in enumerate(row_panels):
            ax = axes[r][col_idx]
            used.add((r, col_idx))
            n_plotted = 0
            for col, label in panel["series"]:
                series = metric_series(col)
                if series is None:
                    continue
                x, y = series
                ax.plot(x, y, marker="o", label=label)
                n_plotted += 1

            ax.set_ylabel(panel["ylabel"])
            ax.set_title(panel["title"])
            if panel["log"] and n_plotted > 0:
                ax.set_yscale("log")

            # Format x-axis with dynamic scaling for readability
            xlim = ax.get_xlim()
            x_max = xlim[1]
            if x_max > 0:
                # Find the order of magnitude
                magnitude = 10 ** int(np.floor(np.log10(x_max)))
                # Determine appropriate power of 10 for scaling
                scaled_max = x_max / magnitude
                if scaled_max > 50:
                    scale = magnitude * 10
                    power = int(np.log10(scale))
                else:
                    scale = magnitude
                    power = int(np.log10(scale))

                # Format ticks with the scale
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/scale:.10g}"))
                ax.set_xlabel(f"Time steps ($\\times 10^{{{power}}}$)")
            else:
                ax.set_xlabel("Step")

            # Legend only matters when multiple series share a panel (e.g. PPO
            # value vs policy); a single DQN line doesn't need one.
            if n_plotted > 1:
                ax.legend()

    # Hide any grid cells left empty by a shorter row.
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in used:
                axes[r][c].axis("off")

    plt.tight_layout()
    # --test dumps the figure in the repo root for a quick look; otherwise it
    # lands in the experiment's metrics/ folder alongside the other outputs.
    if args.test:
        out_dir = ROOT
    else:
        out_dir = metrics_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"metrics_{args.agent}_{args.fov}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
