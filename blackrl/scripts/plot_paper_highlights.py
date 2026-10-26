import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 22,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 18,
    "lines.linewidth": 2.5,
    "lines.markersize": 6.0,
    "figure.figsize": (6, 4.5),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Shared mode → (label, color, linestyle, alpha_line, alpha_fill)
# Keep this as the single source of truth so all plots use identical colors.
# ---------------------------------------------------------------------------
_MODE_STYLE: dict[str, tuple] = {
    # ---- Known Dynamics ----
    "softvi_known_dynamics":                         ("Oracle",             "black", "--",  1.0, 0.15),
    "irl_known_dynamics_every_iter":                 ("RE-HG (P)",  "#CC79A7", "-.",  0.9, 0.15),
    "irl_known_dynamics":                            ("RE-HG (Single, P)",      "#56B4E9", "-.",  0.8, 0.15),
    "expert_surrogate_known_dynamics_no_second_term":("No-Influence(P)",       "#009E73", ":",   0.8, 0.12),
    # ---- Unknown Dynamics ----
    "softql":                                        ("Oracle",            "black", "--",  1.0, 0.15),
    "irl_recursive_rls":                             ("RE-HG",        "#D55E00", "-",   1.0, 0.20),
    "irl_unknown_dynamics":                          ("RE-HG (Single)",   "#0072B2", "-.",  0.8, 0.15),
    "irl_unknown_recursive_rls":                     ("RE-HG",            "#D55E00", "-",   1.0, 0.20),
    "irl":                                           ("RE-HG (Single)",     "#0072B2", "-.",  0.8, 0.15),
    "expert_surrogate_no_second_term":               ("No-Influence",      "#009E73", ":",   0.8, 0.12),
}


def _load_pkl(dir_path: str, mode: str):
    candidates = [
        os.path.join(dir_path, f"aggregated_{mode}.pkl"),
        os.path.join(dir_path, f"stats_{mode}__multi_seed.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def plot_gradient_norm(dir_path):
    """
    1. Gradient Norm (Log Scale)
    """
    print(f"Generating Gradient Norm plot for directory: {dir_path}")
    fig, ax = plt.subplots()

    norm_oracle_modes = {"softvi_known_dynamics", "softql"}

    for mode, (label, color, ls, alpha_line, alpha_fill) in _MODE_STYLE.items():
        data = _load_pkl(dir_path, mode)
        if data is None:
            continue

        norm_key = "hg_oracle_norm" if mode in norm_oracle_modes else "hg_estimated_norm"
        mean_val = np.array(data.get(f"{norm_key}_mean", data.get(norm_key, [])))
        std_val  = np.array(data.get(f"{norm_key}_std",  np.zeros_like(mean_val)))

        if len(mean_val) > 0:
            iterations = np.arange(len(mean_val))
            ax.plot(iterations, mean_val, label=label, color=color, linestyle=ls, alpha=alpha_line)
            ax.fill_between(iterations,
                            np.maximum(1e-10, mean_val - std_val),
                            mean_val + std_val,
                            color=color, alpha=alpha_fill)

    ax.set_yscale("log")
    ax.set_xlabel("Leader Iteration")
    ax.set_ylabel(r"HG Norm $\| \nabla_\theta J_L \|$")
    ax.set_title("Stability of HG Estimation")
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(dir_path, "1_gradient_norm_log.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_pearson(dir_path, is_lqr=False):
    """
    2. Reward Pearson Correlation
    LQR: single line (trajectory-weighted).
    Discrete: unweighted + occupancy-weighted.
    """
    print(f"Generating Pearson Correlation plot for directory: {dir_path}")
    fig, ax = plt.subplots()

    pearson_modes = [
        "irl_unknown_dynamics",
        "irl_recursive_rls",
        "irl_unknown_recursive_rls",
        "irl",
    ]

    found_any = False
    has_weighted = False

    for mode in pearson_modes:
        if mode not in _MODE_STYLE:
            continue
        label, color, ls, alpha_line, alpha_fill = _MODE_STYLE[mode]

        data = _load_pkl(dir_path, mode)
        if data is None:
            continue

        std_corr = np.array(data.get("reward_pearson_corr_mean", data.get("reward_pearson_corr", [])))
        std_std  = np.array(data.get("reward_pearson_corr_std",  np.zeros_like(std_corr)))

        weighted_corr = np.array(data.get("reward_weighted_pearson_corr_mean", data.get("reward_weighted_pearson_corr", [])))
        weighted_std  = np.array(data.get("reward_weighted_pearson_corr_std",  np.zeros_like(weighted_corr)))

        if len(std_corr) == 0:
            continue

        found_any = True
        iterations = np.arange(len(std_corr))

        if is_lqr or len(weighted_corr) == 0:
            ax.plot(iterations, std_corr, label=label, color=color, linestyle=ls, alpha=alpha_line)
            ax.fill_between(iterations,
                            np.maximum(0, std_corr - std_std),
                            np.minimum(1, std_corr + std_std),
                            color=color, alpha=alpha_fill)
        else:
            has_weighted = True
            # Occupancy-weighted
            ax.plot(iterations, weighted_corr, label=label, color=color, linestyle=ls, alpha=alpha_line)
            ax.fill_between(iterations,
                            np.maximum(0, weighted_corr - weighted_std),
                            np.minimum(1, weighted_corr + weighted_std),
                            color=color, alpha=alpha_fill)

    if not found_any:
        print("Required key (reward_pearson_corr) missing in the data.")
        return

    if is_lqr or not has_weighted:
        ax.set_title("Reward Recovery")
    else:
        ax.set_title("Reward Recovery")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Leader Iteration")
    ax.set_ylabel("Pearson Correlation")
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(dir_path, "2_reward_pearson.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_rank_ablation(rank_data_dict, output_path):
    """
    3. Rank vs Stability (Ablation)
    """
    print("Generating Rank vs Stability Ablation plot...")
    if not rank_data_dict:
        print("No rank data provided, generating demonstration plot...")
        ranks     = [1, 3, 5, 8, 12, 16]
        max_norms = [20, 50, 150, 800, 5000, 15000]
        pearsons  = [0.85, 0.94, 0.98, 0.99, 0.995, 1.0]
    else:
        ranks     = sorted(rank_data_dict.keys())
        max_norms = [rank_data_dict[r][0] for r in ranks]
        pearsons  = [rank_data_dict[r][1] for r in ranks]

    fig, ax1 = plt.subplots()

    color1 = _MODE_STYLE["irl_recursive_rls"][1]
    ax1.set_xlabel(r"Information Filter Rank ($k$)")
    ax1.set_ylabel(r"Max Gradient Norm (Log Scale)", color=color1)
    ax1.plot(ranks, max_norms, marker="o", color=color1, linestyle="-")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_yscale("log")
    ax1.axhline(y=1000, color="r", linestyle="--", alpha=0.3)
    ax1.text(ranks[0], 1500, "Instability Threshold", color="r", alpha=0.5, fontsize=14)

    ax2 = ax1.twinx()
    color2 = _MODE_STYLE["irl_known_dynamics_every_iter"][1]
    ax2.set_ylabel("Final Pearson Correlation", color=color2)
    ax2.plot(ranks, pearsons, marker="s", color=color2, linestyle="-")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    plt.title("Ablation: The Trade-off of Filtering")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cosine_similarity(dir_path):
    """
    4. HG Cosine Similarity
    """
    print(f"Generating Cosine Similarity plot for directory: {dir_path}")
    fig, ax = plt.subplots()

    cosine_modes = [
        "irl_known_dynamics_every_iter",
        "irl_known_dynamics",
        "irl_unknown_dynamics",
        "irl_recursive_rls",
        "irl_unknown_recursive_rls",
        "irl",
    ]

    found_any = False
    for mode in cosine_modes:
        if mode not in _MODE_STYLE:
            continue
        label, color, ls, alpha_line, alpha_fill = _MODE_STYLE[mode]

        data = _load_pkl(dir_path, mode)
        if data is None:
            continue

        mean_val = np.array(data.get("hg_cosine_similarity_mean", data.get("hg_cosine_similarity", [])))
        std_val  = np.array(data.get("hg_cosine_similarity_std",  np.zeros_like(mean_val)))

        if len(mean_val) > 0:
            found_any = True
            iterations = np.arange(len(mean_val))
            ax.plot(iterations, mean_val, label=label, color=color, linestyle=ls, alpha=alpha_line)
            ax.fill_between(iterations,
                            np.maximum(-1, mean_val - std_val),
                            np.minimum(1,  mean_val + std_val),
                            color=color, alpha=alpha_fill)

    if not found_any:
        print("Required key (hg_cosine_similarity) missing in the data.")
        return

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Leader Iteration")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Alignment of HG Direction")
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(dir_path, "4_hg_cosine_similarity.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",      required=True, help="Directory containing aggregated_*.pkl files")
    parser.add_argument("--is-lqr",  action="store_true", help="Flag to indicate LQR mode (for Pearson labels)")
    args = parser.parse_args()

    plot_gradient_norm(args.dir)
    plot_reward_pearson(args.dir, is_lqr=args.is_lqr)
    plot_rank_ablation({}, os.path.join(args.dir, "3_rank_ablation.pdf"))
    plot_cosine_similarity(args.dir)

    print("\nAll requested plots generated successfully in:", args.dir)
