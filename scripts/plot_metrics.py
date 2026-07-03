"""Plot plasticity metrics (NTK rank/cond/eff-rank, churn, weight norms) over
training, comparing one or more agents on the same axes.

Like ``src/learning_curve.py``, the hue is the *agent*: pass several agents and
each metric panel overlays one line per agent (mean over seeds), so different
agents' metrics can be compared directly. Pass a single ``--metric`` to emit one
figure with just that panel; pass ``--metrics a b c`` to emit an auto grid; pass
neither to emit every metric that has data.

Each friendly metric name resolves to whichever underlying column an agent
actually wrote: DQN runs store ``ntk_rank`` / ``ntk_cond`` / ``churn_norm``,
while PPO runs store separate ``value_*`` / ``policy_*`` columns. So e.g.
``--metric ntk_rank`` plots ``value_ntk_rank`` for a PPO agent and ``ntk_rank``
for a DQN agent on the same axis.

Examples:
    # one panel per metric, one line per agent
    python scripts/plot_metrics.py \
        -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 \
        -a PPO_LN_128 PPO_LN_256 -f 9

    # a single metric -> a single graph
    python scripts/plot_metrics.py -e <exp> -a DQN PPO_LN_128 -f 9 \
        --metric critic_churn --plot-name dqn_vs_ppo_critic_churn

    # list the metric names you can pass to --metric / --metrics
    python scripts/plot_metrics.py --list-metrics
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # Go up from scripts/ to repo root
sys.path.insert(0, str(ROOT / "src"))

try:  # optional: prettier legend labels, but don't hard-depend on it
    from plotting_utils import LABEL_MAP, get_mapped_label
except Exception:  # pragma: no cover - fallback when src isn't importable
    LABEL_MAP = {}

    def get_mapped_label(label, label_map=None, disable_fov=False):
        return label


# Friendly metric name -> panel spec.  A panel holds one or more `series`; each
# series has a `name` (a sub-label, e.g. actor/critic/total — "" for a lone
# series) and a `cols` *priority list* of underlying parquet/npz column names.
# For each agent the first column that has finite data is used, so one friendly
# name works across agent types (DQN's single head vs the actor/critic heads).
#
# Panels with multiple series (ntk_rank / ntk_eff_rank / ntk_cond / weight_drift)
# overlay actor and critic on one axis: color encodes the *agent* (so an agent's
# actor/critic lines share a color) and linestyle encodes the *series*, with a
# legend spelling out "agent (series)".  `log` uses a log y-axis (condition
# numbers span many orders of magnitude).
METRICS = {
    "ntk_rank": {
        "label": "NTK Rank",
        "title": "NTK Rank Over Time",
        "log": False,
        "series": [
            {"name": "critic", "cols": ["value_ntk_rank", "ntk_rank"]},
            {"name": "actor", "cols": ["policy_ntk_rank"]},
        ],
    },
    "ntk_eff_rank": {
        "label": "NTK Effective Rank",
        "title": "NTK Effective Rank Over Time",
        "log": False,
        "series": [
            {"name": "critic", "cols": ["value_ntk_eff_rank"]},
            {"name": "actor", "cols": ["policy_ntk_eff_rank"]},
        ],
    },
    "ntk_cond": {
        "label": "NTK Condition Number",
        "title": "NTK Condition Number Over Time",
        "log": True,
        "series": [
            {"name": "critic", "cols": ["value_ntk_cond", "ntk_cond"]},
            {"name": "actor", "cols": ["policy_ntk_cond"]},
        ],
    },
    "critic_churn": {
        "label": "Relative Critic Churn",
        "title": "Critic Churn Over Time (magnitude)",
        "log": False,
        "series": [
            {"name": "", "cols": ["value_churn", "churn_norm"]},
        ],
    },
    "actor_churn": {
        "label": "Actor Churn",
        "title": "Actor Churn Over Time (KL divergence)",
        "log": False,
        "series": [
            {"name": "", "cols": ["policy_churn"]},
        ],
    },
    "weight_drift": {
        "label": "Weight Drift",
        "title": "Weight Drift Over Time",
        "log": False,
        "series": [
            {"name": "total", "cols": ["weight_drift_total"]},
            {"name": "actor", "cols": ["weight_drift_pi"]},
            {"name": "critic", "cols": ["weight_drift_vf"]},
        ],
    },
    "weight_update_norm": {
        "label": "Weight Update Norm",
        "title": "Weight Update Norm Over Time",
        "log": False,
        "series": [
            {"name": "", "cols": ["weight_update_norm"]},
        ],
    },
    "weight_norm": {
        "label": "Weight Norm",
        "title": "Weight Norm Over Time",
        "log": False,
        "series": [
            {"name": "", "cols": ["weight_norm"]},
        ],
    },
}

# Linestyle per series index within a combined panel (actor vs critic vs total).
SERIES_LINESTYLES = ["-", "--", ":", "-."]


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
        help="Experiment path, e.g. experiments/E136-big/foragax/ForagaxBig-v5 "
        "(a leading experiments/ or results/ prefix is optional)",
    )
    p.add_argument(
        "-a",
        "--agent",
        nargs="+",
        help="One or more agent / algorithm names (e.g. DQN PPO_LN_128). Each "
        "is drawn as its own line in every metric panel.",
    )
    p.add_argument(
        "-f",
        "--fov",
        nargs="+",
        type=int,
        help="Field-of-view / aperture size. Pass one value (applied to every "
        "agent) or one per agent.",
    )
    p.add_argument(
        "--metric",
        help="Single metric to plot -> a single graph. See --list-metrics.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to plot as a grid of panels. See --list-metrics. "
        "If neither --metric nor --metrics is given, every metric with data "
        "is plotted.",
    )
    p.add_argument(
        "--plot-name",
        help="Output filename stem (without extension). Defaults to a name "
        "derived from the metric(s) and agent(s).",
    )
    p.add_argument("--save-type", default="png", help="Image format (png, pdf, ...).")
    p.add_argument(
        "--no-legend", action="store_true", help="Suppress the per-agent legend."
    )
    p.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the available metric names and exit.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Write the plot to the repo root instead of the experiment's "
        "metrics/ folder (handy for quick one-off checks).",
    )
    return p.parse_args()


def load_runs(data_path: Path) -> list:
    """All per-run npz dicts under `data_path` (one file per seed).

    Read the raw npz (not data.parquet) to keep metrics at native resolution:
    DQN writes one value per env step (NaN except every ntk_freq), PPO writes one
    value per *update*.  The parquet reader repeats PPO per-update arrays to
    per-step length, which makes constant series indistinguishable from a single
    repeated point.
    """
    runs = []
    for f in sorted(Path(data_path).glob("*.npz")):
        with np.load(f) as d:
            runs.append({k: np.asarray(d[k]) for k in d.keys()})
    return runs


def metric_series(runs: list, col: str):
    """Seed-averaged (x_steps, values) for `col`, or None if absent/empty.

    A metric array has one entry per *measurement* (per env step for DQN, per
    update for PPO).  Its x-axis is recovered in env steps by comparing its
    length to the per-step `rewards` array: `steps_per_point = len(rewards) /
    len(metric)` (1 for DQN, rollout_steps for PPO).  NaN entries are dropped
    *after* averaging across seeds.
    """
    arrays = []
    steps_per_point = 1.0
    for r in runs:
        if col not in r:
            continue
        v = np.asarray(r[col], dtype=float).reshape(-1)
        if v.shape[0] == 0:
            continue
        base_len = r["rewards"].reshape(-1).shape[0] if "rewards" in r else v.shape[0]
        steps_per_point = base_len / v.shape[0]
        arrays.append(v)

    if not arrays:
        return None

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


def resolve_series(runs: list, cols: list):
    """First (col, (x, y)) in the priority list `cols` that has finite data for
    this agent, or None."""
    for col in cols:
        series = metric_series(runs, col)
        if series is not None:
            return col, series
    return None


def _format_xaxis(ax):
    """Scale the x-axis to a readable power of ten (e.g. Time steps x10^6)."""
    x_max = ax.get_xlim()[1]
    if x_max <= 0:
        ax.set_xlabel("Step")
        return
    magnitude = 10 ** int(np.floor(np.log10(x_max)))
    scale = magnitude * 10 if x_max / magnitude > 50 else magnitude
    power = int(np.log10(scale))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / scale:.10g}"))
    ax.set_xlabel(f"Time steps ($\\times 10^{{{power}}}$)")


def main():
    args = parse_args()

    if args.list_metrics:
        print("Available metrics (pass to --metric / --metrics):\n")
        for name, spec in METRICS.items():
            parts = [
                f"{s['name']}: {'/'.join(s['cols'])}"
                if s["name"]
                else "/".join(s["cols"])
                for s in spec["series"]
            ]
            print(f"  {name:<20} {spec['label']}  <- {'; '.join(parts)}")
        return

    if not (args.exp and args.agent and args.fov):
        sys.exit(
            "-e/--exp, -a/--agent and -f/--fov are required (unless --list-metrics)."
        )

    # Resolve which metrics to draw and validate names up front.
    if args.metric and args.metrics:
        sys.exit("Pass either --metric or --metrics, not both.")
    requested = [args.metric] if args.metric else args.metrics
    if requested:
        unknown = [m for m in requested if m not in METRICS]
        if unknown:
            sys.exit(
                f"Unknown metric(s): {unknown}. Run --list-metrics to see valid names."
            )

    # Broadcast a single fov to every agent, else require one fov per agent.
    if len(args.fov) == 1:
        fovs = args.fov * len(args.agent)
    elif len(args.fov) == len(args.agent):
        fovs = args.fov
    else:
        sys.exit(
            f"--fov must be one value or one per agent "
            f"({len(args.agent)} agents, got {len(args.fov)} fovs)."
        )

    exp = _normalize_exp(args.exp)
    metrics_dir = ROOT / "experiments" / exp / "metrics"

    # Load every agent's runs once.
    agents = []  # list of (agent, fov, runs)
    for agent, fov in zip(args.agent, fovs):
        data_path = ROOT / "results" / exp / str(fov) / agent / "data"
        runs = load_runs(data_path)
        print(f"{agent} (fov {fov}): loaded {len(runs)} run(s) from {data_path}")
        if not runs:
            print(f"  WARNING: no npz files found for {agent} — skipping.")
            continue
        agents.append((agent, fov, runs))

    if not agents:
        sys.exit("No data loaded for any agent.")

    # Decide the metric list.  Default: every metric that at least one agent has
    # finite data for, in registry order.
    if requested:
        metric_names = requested
    else:
        metric_names = [
            name
            for name, spec in METRICS.items()
            if any(
                resolve_series(runs, s["cols"])
                for _, _, runs in agents
                for s in spec["series"]
            )
        ]
        if not metric_names:
            sys.exit("None of the known metrics have finite data for these agents.")

    # One color per agent, consistent across panels.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    agent_colors = {
        agent: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, (agent, _, _) in enumerate(agents)
    }

    # Auto grid: roughly square, one panel per metric.
    n = len(metric_names)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    flat_axes = axes.flatten()

    for idx, name in enumerate(metric_names):
        spec = METRICS[name]
        ax = flat_axes[idx]
        n_plotted = 0
        for agent, fov, runs in agents:
            base = get_mapped_label(agent, LABEL_MAP)
            for s_idx, series in enumerate(spec["series"]):
                resolved = resolve_series(runs, series["cols"])
                if resolved is None:
                    continue
                _, (x, y) = resolved
                # Same color per agent, linestyle per series (actor/critic/total),
                # with an explicit label so the legend maps line -> agent+series.
                label = f"{base} ({series['name']})" if series["name"] else base
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3,
                    label=label,
                    color=agent_colors[agent],
                    linestyle=SERIES_LINESTYLES[s_idx % len(SERIES_LINESTYLES)],
                )
                n_plotted += 1

        ax.set_ylabel(spec["label"])
        ax.set_title(spec["title"])
        if spec["log"] and n_plotted > 0:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        if n_plotted == 0:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
        else:
            _format_xaxis(ax)
        # Legend on each panel so a single-metric figure is self-contained.
        if n_plotted > 1 and not args.no_legend:
            ax.legend(frameon=False)

    # Hide unused grid cells.
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()

    # Filename: prefer --plot-name, else derive from metric(s) + agent(s).
    agents_str = "_".join(agent for agent, _, _ in agents)
    if args.plot_name:
        stem = args.plot_name
    elif len(metric_names) == 1:
        stem = f"metrics_{metric_names[0]}_{agents_str}"
    else:
        stem = f"metrics_{agents_str}"

    out_dir = ROOT if args.test else metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.{args.save_type}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
