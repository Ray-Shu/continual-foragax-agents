"""Normalized, parameter-count-weighted gradient-norm curves.

Reads the per-seed .npz files written by ``rtu_ppo.py`` (keys
``grad_{l0,l1,l2}_<layer>`` and ``grad_nparams_<layer>``) and reproduces the
Figure-4c-style construction:

  1. per layer, per seed: divide the norm time series by its first-rollout
     value (first rollout -> 1.0 by construction);
  2. combine layers with a parameter-count-weighted average
     ``A(t) = sum_l n_l r_l(t) / sum_l n_l``;
  3. average across seeds (mean + bootstrap 95% CI).

Reads the raw .npz directly (not the downsampled parquet) so the first-rollout
baseline is always present. One subplot per norm type (l0, l1, l2); within each,
a curve for the full network and one each for the actor / critic sub-networks.

Usage:
    python src/grad_norm_curve.py <experiment_path> [--out <pdf>] [--norms l0 l1 l2]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NORM_LABELS = {"l0": r"$\ell_0$", "l1": r"$\ell_1$", "l2": r"$\ell_2$"}
EPS = 1e-12


def _results_root(experiment_path: Path) -> Path:
    """Map an experiments/... path to its results/... root (or pass through)."""
    parts = experiment_path.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        return Path("results", *parts[idx + 1 :])
    return experiment_path


def _find_alg_dirs(root: Path):
    """Every directory containing a data/ subdir with .npz seed files."""
    alg_dirs = []
    for data_dir in sorted(root.rglob("data")):
        if data_dir.is_dir() and any(data_dir.glob("*.npz")):
            alg_dirs.append(data_dir.parent)
    return alg_dirs


def _layers_in(npz) -> list:
    return sorted(
        k[len("grad_l2_") :] for k in npz.files if k.startswith("grad_l2_")
    )


def _combined_curve(npz, norm: str, layers: list) -> np.ndarray:
    """Per-seed: normalize each layer by its first rollout, param-weight-combine."""
    num = None
    den = 0.0
    for layer in layers:
        g = np.asarray(npz[f"grad_{norm}_{layer}"], dtype=np.float64)
        n = float(np.atleast_1d(npz[f"grad_nparams_{layer}"])[0])
        r = g / (g[0] + EPS)  # first rollout -> 1.0
        num = n * r if num is None else num + n * r
        den += n
    assert num is not None, "no layers to combine"
    return num / den


def _seed_curves(alg_dir: Path, norm: str):
    """Stack combined/actor/critic curves across all seeds. Returns dict of
    (num_seeds, num_rollouts) arrays plus the frames x-axis."""
    seed_paths = sorted(alg_dir.glob("data/*.npz"), key=lambda p: int(p.stem))
    groups = {"full": [], "actor": [], "critic": []}
    frames = None
    for sp in seed_paths:
        npz = np.load(sp)
        layers = _layers_in(npz)
        subsets = {
            "full": layers,
            "actor": [name for name in layers if name.startswith("actor")],
            "critic": [name for name in layers if name.startswith("critic")],
        }
        for name, subset in subsets.items():
            if subset:
                groups[name].append(_combined_curve(npz, norm, subset))
        if frames is None:
            n_roll = len(np.asarray(npz[f"grad_{norm}_{layers[0]}"]))
            rollout_steps = len(np.asarray(npz["rewards"])) // n_roll
            frames = (np.arange(n_roll) + 1) * rollout_steps
    return {k: np.stack(v) for k, v in groups.items() if v}, frames


def _mean_ci(curves: np.ndarray, n_boot: int = 1000):
    """Mean and bootstrap 95% CI across seeds (axis 0)."""
    mean = curves.mean(axis=0)
    n = curves.shape[0]
    if n < 2:
        return mean, None, None
    rng = np.random.default_rng(0)
    boot = curves[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return mean, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_path", type=str)
    parser.add_argument("--norms", nargs="+", default=["l0", "l1", "l2"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    root = _results_root(Path(args.experiment_path))
    alg_dirs = _find_alg_dirs(root)
    if not alg_dirs:
        sys.exit(f"No data/*.npz found under {root}")

    colors = {"full": "tab:orange", "actor": "tab:blue", "critic": "tab:red"}
    fig, axes = plt.subplots(
        len(args.norms), 1, figsize=(7, 3.2 * len(args.norms)), squeeze=False
    )

    for alg_dir in alg_dirs:
        alg = alg_dir.name
        for ax, norm in zip(axes[:, 0], args.norms, strict=True):
            curves, frames = _seed_curves(alg_dir, norm)
            assert frames is not None, f"no seeds in {alg_dir}"
            x = frames / 1e6
            for name, arr in curves.items():
                mean, lo, hi = _mean_ci(arr)
                style = "-" if name == "full" else "--"
                label = f"{alg}:{name}" if len(alg_dirs) > 1 else name
                ax.plot(x, mean, style, color=colors[name], label=label, lw=1.6)
                if lo is not None:
                    ax.fill_between(x, lo, hi, color=colors[name], alpha=0.15)
            ax.axhline(1.0, color="grey", lw=0.6, ls=":")
            ax.set_ylabel(f"{NORM_LABELS.get(norm, norm)} grad norm\n(norm. to rollout 1)")
            ax.legend(fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel(r"Time steps ($\times 10^6$)")
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "plots" / "grad_norm_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
