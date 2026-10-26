import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

def _set_plot_style() -> None:
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.5,
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 1.0,
            "grid.alpha": 0.5,
        },
    )

def _plot_series(
    ax: plt.Axes,
    series: list[float],
    label: str,
    *,
    color: str,
    linestyle: str = "-",
    marker: str = None,
) -> None:
    xs = np.arange(len(series))
    ax.plot(
        xs,
        np.asarray(series, dtype=np.float64),
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=max(len(series) // 10, 1),
        markersize=8,
    )

def main():
    _set_plot_style()
    out_dir = Path("data/internal/conf_discrete_synth2")
    
    with open(out_dir / "stats_aggregate_seed0.pkl", "rb") as f:
        agg = pickle.load(f)
    with open(out_dir / "stats_single_fixed_seed0.pkl", "rb") as f:
        single = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    _plot_series(axes[0], agg["policy_kl_to_uniform"], "Aggregate IRL (Ours)", color="#0072B2", marker="o")
    _plot_series(axes[0], single["policy_kl_to_uniform"], "Single Fixed IRL", color="#D55E00", linestyle="--", marker="s")
    axes[0].set_title("Policy KL to Uniform")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("KL Divergence")
    axes[0].legend()

    _plot_series(axes[1], agg["theta"], "Aggregate IRL (Ours)", color="#0072B2", marker="o")
    _plot_series(axes[1], single["theta"], "Single Fixed IRL", color="#D55E00", linestyle="--", marker="s")
    axes[1].set_title(r"Configuration Parameter $\theta$ Schedule")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\theta$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    fig.tight_layout()
    diagnostics_path = out_dir / "conf_discrete_kl_theta_only.pdf"
    fig.savefig(str(diagnostics_path), bbox_inches="tight")
    print(f"Saved plot: {diagnostics_path}")

if __name__ == "__main__":
    main()
