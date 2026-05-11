from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import bootstrap

from plotting_utils import (
    PlottingArgumentParser,
    load_data,
    parse_plotting_args,
    save_plot,
)

METRIC = "ewm_reward_5"
LAST_PERCENT = 0.1

COLOR_MAP_FOV = {
    "DQN": "#1F77B4",
    "Search-Oracle": "#2CA02C",
    "Search-Nearest": "#D62728",
    "Random": "#000000",
}


def compute_last_percent_auc(df: pl.DataFrame, metric: str, percent: float = 0.1) -> tuple:
    """Compute AUC from the last N% of training data."""
    if df.is_empty():
        return np.nan, (np.nan, np.nan)

    # Group by seed and compute mean over the last N% of frames
    def get_last_percent_mean(group):
        frames = group["frame"].to_numpy()
        if len(frames) == 0:
            return np.nan
        last_idx = int((1 - percent) * len(frames))
        values = group[metric][last_idx:].to_numpy()
        return float(np.mean(values)) if len(values) > 0 else np.nan

    last_percent_means = []
    for seed in df["seed"].unique():
        seed_data = df.filter(pl.col("seed") == seed)
        mean_val = get_last_percent_mean(seed_data)
        if not np.isnan(mean_val):
            last_percent_means.append(mean_val)

    last_percent_means = np.array(last_percent_means)
    if len(last_percent_means) == 0:
        return np.nan, (np.nan, np.nan)

    # Compute bootstrap CI
    mean_val = np.mean(last_percent_means)
    if len(last_percent_means) > 1:
        res = bootstrap((last_percent_means,), np.mean, confidence_level=0.95)
        ci = res.confidence_interval
    else:
        ci = (mean_val, mean_val)

    return mean_val, ci


def plot_auc_fov(args, metric: str = METRIC, last_percent: float = LAST_PERCENT):
    """Plot Last 10% AUC vs Field of View."""
    # Load data from preprocessed parquet files
    df = load_data(args.experiment_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    is_dqn = pl.col("alg").str.starts_with("DQN")
    dqn_df = df.filter(is_dqn & pl.col("aperture").is_not_null())
    baseline_df = df.filter(~is_dqn)

    sorted_apertures = sorted(
        a for a in dqn_df["aperture"].unique().to_list() if a is not None
    )
    dqn_values = []
    dqn_ci_low = []
    dqn_ci_high = []

    for aperture in sorted_apertures:
        alg_df = dqn_df.filter(pl.col("aperture") == aperture)
        mean_val, (ci_low, ci_high) = compute_last_percent_auc(
            alg_df, metric, last_percent
        )
        dqn_values.append(mean_val)
        dqn_ci_low.append(ci_low)
        dqn_ci_high.append(ci_high)

    baselines = {}
    for alg in baseline_df["alg"].unique().to_list():
        alg_df = baseline_df.filter(pl.col("alg") == alg)
        baselines[alg] = compute_last_percent_auc(alg_df, metric, last_percent)

    # Plot DQN
    if dqn_values:
        ax.plot(sorted_apertures, dqn_values, label='DQN', color=COLOR_MAP_FOV["DQN"], linewidth=2, marker='o')
        ax.fill_between(sorted_apertures, dqn_ci_low, dqn_ci_high, color=COLOR_MAP_FOV["DQN"], alpha=0.2)

    # Plot baselines as horizontal lines
    baseline_order = ["Random", "Search-Nearest", "Search-Oracle"]
    for alg in baseline_order:
        if alg in baselines:
            mean_val, (ci_low, ci_high) = baselines[alg]
            color = COLOR_MAP_FOV.get(alg, "#000000")
            ax.plot(sorted_apertures, [mean_val] * len(sorted_apertures), label=alg, color=color, linewidth=2)
            ax.fill_between(sorted_apertures, ci_low, ci_high, color=color, alpha=0.3)

    ax.set_xlabel('Field of View', fontsize=22)
    ax.set_ylabel('Last 10% Average Reward AUC', fontsize=22)
    if sorted_apertures:
        ax.set_xticks(sorted_apertures)
        ax.set_xticklabels([str(int(x)) for x in sorted_apertures])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plot_name = args.plot_name or 'auc_fov'
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


def main():
    parser = PlottingArgumentParser(description="Plot Last 10% AUC vs Field of View.")
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC,
        help="Metric to compute AUC for.",
    )
    parser.add_argument(
        "--last-percent",
        type=float,
        default=LAST_PERCENT,
        help="Percentage of training to compute AUC over (default: 0.1 = last 10%).",
    )
    args = parse_plotting_args(parser)

    plot_auc_fov(args, metric=args.metric, last_percent=args.last_percent)


if __name__ == "__main__":
    main()
