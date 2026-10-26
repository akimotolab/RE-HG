import matplotlib.pyplot as plt
import numpy as np


def _set_plot_style() -> None:
    """Use a single matplotlib style."""
    plt.style.use("default")

    plt.rcParams.update(
        {
            "font.size": 25,
            "axes.titlesize": 25,
            "axes.titleweight": "bold",
            "axes.labelsize": 25,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 21,
            "lines.linewidth": 2.8,
            "lines.markersize": 7.5,
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "axes.spines.top": True,
            "axes.spines.right": True,
        },
    )


_METHOD_STYLES = [
    {"color": "black", "linestyle": "-", "marker": "o"},
    {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    {"color": "#D55E00", "linestyle": "-.", "marker": "^"},
    {"color": "#009E73", "linestyle": ":", "marker": "D"},
    {"color": "#CC79A7", "linestyle": (0, (5, 1)), "marker": "v"},
    {"color": "#E69F00", "linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
    {"color": "#56B4E9", "linestyle": (0, (7, 2)), "marker": "X"},
    {"color": "#A6761D", "linestyle": (0, (1, 1)), "marker": "<"},
    {"color": "#F0E442", "linestyle": (0, (2, 1)), "marker": ">"},
    {"color": "#C44E52", "linestyle": (0, (4, 2, 1, 2)), "marker": "*"},
    {"color": "#4C72B0", "linestyle": (0, (6, 2, 2, 2)), "marker": "h"},
    {"color": "#55A868", "linestyle": (0, (8, 2, 1, 2)), "marker": "8"},
]


def _style_for_name(name: str, default_idx: int = 0) -> dict:
    name_lower = name.lower()

    # Oracle variants
    if "oracle" in name_lower or "softvi" in name_lower or "softql" in name_lower:
        return {"color": "black", "linestyle": "--", "marker": "o"}

    has_p = "proposed" in name_lower or "(p)" in name_lower or " p)" in name_lower or ",p)" in name_lower or ", p)" in name_lower

    # Single
    if "single" in name_lower:
        if has_p:
            return {"color": "#56B4E9", "linestyle": "-.", "marker": "p"} # Light Blue
        else:
            return {"color": "#0072B2", "linestyle": "-.", "marker": "^"} # Blue

    # Every iter
    if "every iter" in name_lower:
        if has_p:
            return {"color": "#E69F00", "linestyle": "-", "marker": "x"} # Yellow
        else:
            return {"color": "#CC79A7", "linestyle": "-", "marker": "v"} # Purple

    # RE-HG (baseline without P/Single/Every iter)
    if "re-hg" in name_lower:
        if has_p:
            return {"color": "#CC79A7", "linestyle": "-", "marker": "s"} # Orange
        else:
            return {"color": "#D55E00", "linestyle": "--", "marker": "v"} # Purple

    # No-Influence
    if "no-influence" in name_lower or "no influence" in name_lower:
        return {"color": "#009E73", "linestyle": ":", "marker": "D"}

    # Fallback
    return dict(_METHOD_STYLES[(default_idx % len(_METHOD_STYLES))])


def _markevery(num_points: int, target_marks: int = 10) -> int:
    return max(num_points // target_marks, 1)


def _plot_metric(ax, stats, baselines, metric_key, title, ylabel, main_label="Proposed (IRL)"):
    """Helper to plot a single metric with standard deviations across methods."""
    ALPHA_MAIN = 0.7
    ALPHA_BASELINE = 0.3

    # Main Method
    if metric_key in stats:
        means = np.array(stats[metric_key])
        iterations = range(len(means))
        main_style = _style_for_name(main_label)
        ax.plot(
            iterations,
            means,
            label=main_label,
            markevery=_markevery(len(means), 12),
            **main_style,
        )
        std_key = f"{metric_key}_std"
        if std_key in stats:
            stds = np.array(stats[std_key])
            ax.fill_between(iterations, means - stds, means + stds, color=main_style["color"], alpha=ALPHA_MAIN)

    # Baselines
    if baselines:
        for i, (name, base_stats) in enumerate(baselines.items()):
            if metric_key in base_stats:
                means = np.array(base_stats[metric_key])
                iterations = range(len(means))
                style = _style_for_name(name, i + 1)
                ax.plot(
                    iterations,
                    means,
                    label=name,
                    markevery=_markevery(len(means), 12),
                    **style,
                )
                std_key = f"{metric_key}_std"
                if std_key in base_stats:
                    stds = np.array(base_stats[std_key])
                    ax.fill_between(iterations, means - stds, means + stds, color=style["color"], alpha=ALPHA_BASELINE)

    ax.set_xlabel("Leader Iteration", fontsize=25, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=25, labelpad=8)
    ax.set_title(title)
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, borderaxespad=0.0)


def plot_components(stats, save_path=None, baselines=None, main_label="Proposed (IRL)"):
    """Plot the individual components of the leader reward."""
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))

    _plot_metric(axes[0, 0], stats, baselines, "stability", "Discounted Stability", "Discounted Stability", main_label=main_label)
    _plot_metric(axes[0, 1], stats, baselines, "energy_cost", "Discounted HVAC Energy Cost", "Discounted Energy Cost", main_label=main_label)
    _plot_metric(axes[1, 0], stats, baselines, "insulation_cost", "Discounted Insulation Cost (α)", "Discounted Insulation Cost", main_label=main_label)
    _plot_metric(axes[1, 1], stats, baselines, "airflow_cost", "Discounted Airflow Cost (β)", "Discounted Airflow Cost", main_label=main_label)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Component curves saved to: {save_path}")
    return fig


def plot_learning_curves(stats, save_path=None, baselines=None, main_label="Proposed (IRL)"):
    """Plot comprehensive learning curves with comparison.

    Args:
        stats: Statistics dictionary for the main method (Proposed IRL)
        save_path: Path to save the plot
        baselines: Dictionary of baseline stats {name: stats_dict}

    """
    _set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))

    # --- Plot 1: Leader Performance Comparison [MAIN] ---
    ax = axes[0, 0]

    # 1. Main Method
    ret_key = "disc_leader_return" if "disc_leader_return" in stats else "leader_return"
    if ret_key in stats:
        means = np.array(stats[ret_key])
        iterations = range(len(means))
        main_style = _style_for_name(main_label)
        ax.plot(
            iterations,
            means,
            label=main_label,
            markevery=_markevery(len(means), 12),
            **main_style,
        )

        std_key = "disc_leader_return_std" if "disc_leader_return_std" in stats else "leader_return_std"
        if std_key in stats:
            stds = np.array(stats[std_key])
            ax.fill_between(iterations, means - stds, means + stds, color=main_style["color"], alpha=0.15)

    # 2. Baselines
    if baselines:
        for i, (name, base_stats) in enumerate(baselines.items()):
            ret_key = "disc_leader_return" if "disc_leader_return" in base_stats else "leader_return"
            if ret_key in base_stats:
                means = np.array(base_stats[ret_key])
                iterations = range(len(means))
                style = _style_for_name(name, i + 1)
                ax.plot(
                    iterations,
                    means,
                    label=name,
                    markevery=_markevery(len(means), 12),
                    **style,
                )

                std_key = "disc_leader_return_std" if "disc_leader_return_std" in base_stats else "leader_return_std"
                if std_key in base_stats:
                    stds = np.array(base_stats[std_key])
                    ax.fill_between(iterations, means - stds, means + stds, color=style["color"], alpha=0.1)

    ax.set_xlabel("Leader Iteration", fontsize=25, labelpad=10)
    ax.set_ylabel("Disc. Leader Return", fontsize=25, labelpad=10)
    ax.set_title("Leader Performance Comparison")
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.legend(loc="lower right")

    # --- Plot 2: IRL Delta FEM (Main Method only) ---
    ax = axes[0, 1]
    if "irl_delta_fem" in stats and len(stats["irl_delta_fem"]) > 0:
        ax.plot(
            stats["irl_delta_fem"],
            color="#0072B2",
            linestyle="-",
            marker="s",
            markevery=_markevery(len(stats["irl_delta_fem"]), 8),
        )
        ax.axhline(y=0.01, color="r", linestyle="--", alpha=0.5, label="Threshold (0.01)")
        ax.set_yscale("log")
        ax.set_xlabel("IRL Iteration", fontsize=25)
        ax.set_ylabel("Delta FEM", fontsize=25)
        ax.set_title("IRL Feature Matching Error")
        ax.tick_params(axis="both", which="major", labelsize=25)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No IRL Delta Data", ha="center", va="center", fontsize=18)

    # --- Plot 3: IRL Gradient Norm (Main Method only) ---
    ax = axes[0, 2]
    if "irl_gradient_norm" in stats and len(stats["irl_gradient_norm"]) > 0:
        ax.plot(
            stats["irl_gradient_norm"],
            color="#D55E00",
            linestyle="-.",
            marker="^",
            markevery=_markevery(len(stats["irl_gradient_norm"]), 8),
        )
        ax.set_yscale("log")
        ax.set_xlabel("IRL Iteration", fontsize=25)
        ax.set_ylabel("Gradient Norm", fontsize=25)
        ax.set_title("IRL Gradient Norm")
        ax.tick_params(axis="both", which="major", labelsize=25)
    else:
        ax.text(0.5, 0.5, "No IRL Gradient Data", ha="center", va="center", fontsize=18)

    # --- Plot 4: Leader Gradient Norm (Main Method) ---
    ax = axes[1, 0]
    if "leader_gradient_norm" in stats and len(stats["leader_gradient_norm"]) > 0:
        ax.plot(
            stats["leader_gradient_norm"],
            color="black",
            linestyle="-",
            marker="o",
            markevery=_markevery(len(stats["leader_gradient_norm"]), 8),
        )
        ax.set_yscale("log")
        ax.set_xlabel("Leader Iteration", fontsize=25)
        ax.set_title("Leader Gradient Norm (Proposed)")
        ax.tick_params(axis="both", which="major", labelsize=25)
    else:
        ax.text(0.5, 0.5, "No Gradient Data", ha="center", va="center", fontsize=18)

    # --- Plot 5: IRL Q-Values (Main Method) ---
    ax = axes[1, 1]
    if "irl_q_value_mean" in stats and len(stats["irl_q_value_mean"]) > 0:
        ax.plot(
            stats["irl_q_value_mean"],
            color="#009E73",
            linestyle=":",
            marker="D",
            markevery=_markevery(len(stats["irl_q_value_mean"]), 8),
            label="Mean",
        )
        if "irl_q_value_min" in stats and len(stats["irl_q_value_min"]) > 0:
            ax.plot(
                stats["irl_q_value_min"],
                color="#009E73",
                linestyle="--",
                linewidth=1.4,
                alpha=0.7,
                label="Min",
            )
        if "irl_q_value_max" in stats and len(stats["irl_q_value_max"]) > 0:
            ax.plot(
                stats["irl_q_value_max"],
                color="#D55E00",
                linestyle="--",
                linewidth=1.4,
                alpha=0.7,
                label="Max",
            )
        ax.set_xlabel("IRL Iteration", fontsize=25)
        ax.set_ylabel("Q-Value", fontsize=25)
        ax.set_title("IRL Q-Values (Follower)")
        ax.tick_params(axis="both", which="major", labelsize=25)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No IRL Q-Value Data", ha="center", va="center", fontsize=18)

    # --- Plot 6: Summary Text ---
    ax = axes[1, 2]
    ax.axis("off")
    summary = "Experiment Summary\n------------------\n\n"

    # Proposed
    ret_key = "disc_leader_return" if "disc_leader_return" in stats else "leader_return"
    if ret_key in stats and len(stats[ret_key]) > 0:
        final_val = stats[ret_key][-1]
        summary += f"Proposed (IRL): {final_val:.4f}\n"

    # Baselines
    if baselines:
        for name, base_stats in baselines.items():
            b_ret_key = "disc_leader_return" if "disc_leader_return" in base_stats else "leader_return"
            if b_ret_key in base_stats and len(base_stats[b_ret_key]) > 0:
                final_val = base_stats[b_ret_key][-1]
                summary += f"{name}: {final_val:.4f}\n"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=18, verticalalignment="top", family="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to: {save_path}")
    return fig


def plot_leader_performance_only(stats, save_path=None, baselines=None, main_label="Proposed (IRL)"):

    _set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))

    # 1. Main Method
    ret_key = "disc_leader_return" if "disc_leader_return" in stats else "leader_return"
    if ret_key in stats:
        means = np.array(stats[ret_key])
        iterations = range(len(means))
        main_style = _style_for_name(main_label)
        ax.plot(
            iterations,
            means,
            label=main_label,
            markevery=_markevery(len(means), 12),
            **main_style,
        )

        std_key = "disc_leader_return_std" if "disc_leader_return_std" in stats else "leader_return_std"
        if std_key in stats:
            stds = np.array(stats[std_key])
            ax.fill_between(iterations, means - stds, means + stds, color=main_style["color"], alpha=0.2)

    # 2. Baselines
    if baselines:
        for i, (name, base_stats) in enumerate(baselines.items()):
            ret_key = "disc_leader_return" if "disc_leader_return" in base_stats else "leader_return"
            if ret_key in base_stats:
                means = np.array(base_stats[ret_key])
                iterations = range(len(means))
                style = _style_for_name(name, i + 1)
                ax.plot(
                    iterations,
                    means,
                    label=name,
                    markevery=_markevery(len(means), 12),
                    **style,
                )

                std_key = "disc_leader_return_std" if "disc_leader_return_std" in base_stats else "leader_return_std"
                if std_key in base_stats:
                    stds = np.array(base_stats[std_key])
                    ax.fill_between(iterations, means - stds, means + stds, color=style["color"], alpha=0.15)

    ax.set_xlabel("Leader Iteration", fontsize=25, labelpad=10)
    ax.set_ylabel("Disc. Leader Return", fontsize=25, labelpad=10)
    ax.set_title("Leader Performance Comparison")
    ax.tick_params(axis="both", which="major", labelsize=25)
    _, legend_labels = ax.get_legend_handles_labels()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=max(1, len(legend_labels)), borderaxespad=0.0)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Leader performance plot saved to: {save_path}")
    return fig



def plot_gradient_norm_comparison(softql_stats, softvi_stats, save_path=None):
    """Plot gradient norm comparison between SoftQL and SoftVI.

    Args:
        softql_stats: Statistics dictionary for SoftQL
        softvi_stats: Statistics dictionary for SoftVI
        save_path: Path to save the plot
    """
    _set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(13, 8.5))

    # SoftQL
    if "leader_gradient_norm" in softql_stats and len(softql_stats["leader_gradient_norm"]) > 0:
        softql_grads = np.array(softql_stats["leader_gradient_norm"])
        softql_iterations = range(len(softql_grads))
        ax.plot(
            softql_iterations,
            softql_grads,
            color="#0072B2",
            linestyle="--",
            marker="s",
            markevery=_markevery(len(softql_grads), 10),
            label="SoftQL",
            alpha=0.9,
        )

    # SoftVI
    if "leader_gradient_norm" in softvi_stats and len(softvi_stats["leader_gradient_norm"]) > 0:
        softvi_grads = np.array(softvi_stats["leader_gradient_norm"])
        softvi_iterations = range(len(softvi_grads))
        ax.plot(
            softvi_iterations,
            softvi_grads,
            color="#D55E00",
            linestyle="-.",
            marker="^",
            markevery=_markevery(len(softvi_grads), 10),
            label="SoftVI",
            alpha=0.9,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Leader Iteration", fontsize=25)
    ax.set_ylabel("Gradient Norm", fontsize=25)
    ax.set_title("Gradient Norm Comparison: SoftQL vs SoftVI")
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gradient norm comparison plot saved to: {save_path}")
    return fig


def plot_leader_performance_split(results_unknown, results_known, save_path=None):
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    all_names = list(set(list(results_unknown.keys()) + list(results_known.keys())))
    order = ["Oracle","RE-HG (Single)", "RE-HG", "No-Influence"]
    all_names_sorted = [n for n in order if n in all_names] + [n for n in all_names if n not in order]

    style_map = {}
    for i, name in enumerate(all_names_sorted):
        style_map[name] = _style_for_name(name, i)

    def _plot_dict(ax, results, title):
        for name, stats in results.items():
            ret_key = "disc_leader_return" if "disc_leader_return" in stats else "leader_return"
            if ret_key in stats:
                means = np.array(stats[ret_key])
                iterations = range(len(means))
                style = style_map[name]
                ax.plot(
                    iterations,
                    means,
                    label=name,
                    markevery=_markevery(len(means), 12),
                    **style,
                )
                std_key = "disc_leader_return_std" if "disc_leader_return_std" in stats else "leader_return_std"
                if std_key in stats:
                    stds = np.array(stats[std_key])
                    alpha = 0.2 if name.startswith("Oracle") else 0.15
                    ax.fill_between(iterations, means - stds, means + stds, color=style["color"], alpha=alpha)
        ax.set_xlabel("Leader Iteration", fontsize=25, labelpad=10)
        ax.set_title(title, fontsize=25, pad=15)
        ax.tick_params(axis="both", which="major", labelsize=25)

    _plot_dict(axes[0], results_unknown, "Unknown Dynamics")
    axes[0].set_ylabel("Disc. Leader Return", fontsize=25, labelpad=10)

    _plot_dict(axes[1], results_known, "Known Dynamics")

    # Unified Legend
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()

    by_label = dict(zip(labels, handles))
    by_label.update(dict(zip(labels2, handles2)))

    final_labels = [n for n in all_names_sorted if n in by_label]
    final_handles = [by_label[n] for n in final_labels]
    axes[0].legend(final_handles, final_labels, loc='center right', fontsize=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Split leader performance plot saved to: {save_path}")
    return fig
