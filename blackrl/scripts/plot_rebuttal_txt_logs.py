from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.titlesize": 20, "figure.titlesize": 20})


EVAL_LINE_RE = re.compile(r"iter\s+(\d+):\s+train=.*?eval=([-+]?\d*\.?\d+)")
ITERATION_RE = re.compile(r"Leader Iteration (\d+)")
CAO_EVAL_RE = re.compile(r"Leader Performance \(vs Best Response\):\s+([-+]?\d*\.?\d+)")
FILE_RE = re.compile(r"logs_(.+?)(?:__seed_|_seed)(\d+)\.txt$")


def parse_eval_curve(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    iterations: list[int] = []
    eval_values: list[float] = []

    current_cao_iteration: int | None = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "(pre-update eval)" in line:
                continue
            m = EVAL_LINE_RE.search(line)
            if m:
                iterations.append(int(m.group(1)))
                eval_values.append(float(m.group(2)))
                continue

            cao_iteration = ITERATION_RE.search(line)
            if cao_iteration:
                current_cao_iteration = int(cao_iteration.group(1))
                continue

            cao_eval = CAO_EVAL_RE.search(line)
            if cao_eval and current_cao_iteration is not None:
                iterations.append(current_cao_iteration)
                eval_values.append(float(cao_eval.group(1)))

    if not iterations:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(iterations, dtype=int), np.array(eval_values, dtype=float)


def aggregate_method_curves(
    exp_dir: Path, excluded_seeds: tuple[int, ...] = ()
) -> dict[str, dict[str, np.ndarray]]:
    by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for path in sorted(exp_dir.glob("logs_*_seed*.txt")):
        m = FILE_RE.search(path.name)
        if not m:
            continue
        if int(m.group(2)) in excluded_seeds:
            continue
        method = m.group(1)
        iters, vals = parse_eval_curve(path)
        if len(iters) == 0:
            continue
        by_method[method].append((iters, vals))

    summary: dict[str, dict[str, np.ndarray]] = {}
    for method, series in by_method.items():
        common_iters = sorted(set.intersection(*(set(x.tolist()) for x, _ in series)))
        if not common_iters:
            continue

        common_arr = np.array(common_iters, dtype=int)
        aligned = []
        for iters, vals in series:
            idx = {int(i): j for j, i in enumerate(iters)}
            aligned.append([vals[idx[int(i)]] for i in common_arr])
        stacked = np.array(aligned, dtype=float)

        summary[method] = {
            "iters": common_arr,
            "mean": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
            "n": np.array([stacked.shape[0]]),
        }

    return summary


def pretty_method_name(method: str) -> str:
    labels = {
        "irl_recursive_rls": "RE-HG (IRL-RLS)",
        "irl_known_dynamics": "CAO (known dynamics)",
        "irl_known_dynamics_every_iter": "CAO (known dynamics)",
        "softvi": "Oracle (SoftVI)",
    }
    return labels.get(method, method)


def method_style(method: str) -> dict:
    if method == "irl_recursive_rls":
        return {"color": "#D55E00", "linestyle": "-", "marker": "o"}
    if method == "softvi":
        return {"color": "black", "linestyle": "--", "marker": "s"}
    if method.startswith("irl_known_dynamics"):
        return {"color": "#0072B2", "linestyle": "-.", "marker": "^"}
    return {"color": "#0072B2", "linestyle": "-.", "marker": "^"}


def plot_single_experiment(exp_dir: Path, output_dir: Path) -> Path | None:
    stats = aggregate_method_curves(exp_dir)
    if not stats:
        return None

    plt.figure(figsize=(7.2, 4.6), dpi=180)
    for method, v in sorted(stats.items()):
        style = method_style(method)
        iters = v["iters"]
        mean = v["mean"]
        std = v["std"]

        plt.plot(
            iters,
            mean,
            label=f"{pretty_method_name(method)} (n={int(v['n'][0])})",
            linewidth=2.4,
            markersize=5.8,
            markevery=max(len(iters) // 12, 1),
            **style,
        )
        plt.fill_between(iters, mean - std, mean + std, color=style["color"], alpha=0.16)

    plt.title(exp_dir.name, fontsize=20)
    plt.xlabel("Leader Iteration", fontsize=20)
    plt.ylabel("Disc. Leader Return", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.grid(alpha=0.25)
    plt.legend(loc="best", frameon=True, fontsize=17)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{exp_dir.name}_eval_curve.pdf"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_overview(exp_dirs: list[Path], output_dir: Path) -> Path | None:
    if not exp_dirs:
        return None

    n = len(exp_dirs)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.6 * rows), dpi=160, squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for i, exp_dir in enumerate(exp_dirs):
        ax = axes[i // cols][i % cols]
        ax.axis("on")

        stats = aggregate_method_curves(exp_dir)
        if not stats:
            ax.set_title(f"{exp_dir.name} (no parsed data)")
            continue

        for method, v in sorted(stats.items()):
            style = method_style(method)
            iters = v["iters"]
            mean = v["mean"]
            std = v["std"]
            ax.plot(iters, mean, label=pretty_method_name(method), linewidth=1.8, **style)
            ax.fill_between(iters, mean - std, mean + std, color=style["color"], alpha=0.12)

        ax.set_title(exp_dir.name, fontsize=20)
        ax.set_xlabel("Leader Iteration", fontsize=20)
        ax.set_ylabel("Disc. Leader Return", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=True, fontsize=17)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rebuttal_overview_eval_curves.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _setting_label(exp_dir: Path, group: str) -> str:
    if group == "ognorm":
        match = re.search(r"_lr([^_]+)", exp_dir.name)
        return f"lr={match.group(1)}" if match else exp_dir.name
    if group == "lr":
        match = re.search(r"oracle_lr([^_]+)", exp_dir.name)
        if match:
            return f"lr={match.group(1)}"
        return "RE-HG"
    if group == "beta":
        match = re.search(r"beta(.+)$", exp_dir.name)
        return f"beta={match.group(1)}" if match else exp_dir.name
    if group == "T":
        match = re.search(r"T(.+)$", exp_dir.name)
        return match.group(1) if match else exp_dir.name
    if group == "a":
        match = re.search(r"a(.+)$", exp_dir.name)
        return match.group(1) if match else exp_dir.name
    if group == "sweep":
        match = re.search(r"sweeps?(.+)$", exp_dir.name)
        return match.group(1) if match else exp_dir.name
    if group == "d":
        match = re.search(r"d(.+)$", exp_dir.name)
        return match.group(1) if match else exp_dir.name
    if group == "sev":
        match = re.search(r"sev(.+)$", exp_dir.name)
        if match:
            severity = float(match.group(1))
            if severity == 0.0:
                return "top-3 weighted features fixed at true values"
            if severity == 1.0:
                return "top-3 weighted features fixed"
            return f"top-3 fixed (freeze error={100 * severity:g}%)"
        return exp_dir.name
    if group == "excitation":
        return "least-excited-3 fixed"
    if group == "excitation":
        return "least-excited-3 fixed"
    if group == "p":
        match = re.fullmatch(r"exp2c_overspec_p(.+)", exp_dir.name)
        return f"redundant={match.group(1)}" if match else exp_dir.name
    return exp_dir.name


def plot_setting_group(
    group: str,
    title: str,
    exp_dirs: list[Path],
    output_dir: Path,
) -> Path | None:
    """Plot one learning-curve figure for a single rebuttal setting axis."""
    if not exp_dirs:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(exp_dirs)))
    for color, exp_dir in zip(colors, exp_dirs):
        stats = aggregate_method_curves(exp_dir)
        for method, values in sorted(stats.items()):
            label = _setting_label(exp_dir, group)
            if len(stats) > 1:
                label = f"{label} ({pretty_method_name(method)})"
            ax.plot(
                values["iters"],
                values["mean"],
                label=label,
                color=color,
                linestyle="--" if method == "softvi" else "-",
                linewidth=2.2,
                marker="s" if method == "softvi" else "o",
                markevery=max(len(values["iters"]) // 12, 1),
                markersize=4.5,
            )
            ax.fill_between(
                values["iters"],
                values["mean"] - values["std"],
                values["mean"] + values["std"],
                color=color,
                alpha=0.12,
            )

    ax.set_title(title)
    ax.set_xlabel("Leader Iteration", fontsize=20)
    ax.set_ylabel("Disc. Leader Return", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=17)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"exp3_{group}_eval_curves.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_setting_groups(root: Path, output_dir: Path) -> list[Path]:
    groups = {
        "ognorm": (
            "Exp 3b: Oracle with gradient clipping",
            sorted(root.glob("exp3b_oracle_gnorm_lr*")),
        ),
        "lr": (
            "Exp 3b: Oracle learning-rate setting",
            sorted(root.parent.joinpath("rebuttal1").glob("exp3_oracle_lr*")),
        ),
        "beta": (
            r"Exp 3c: influence scale $\beta$ setting",
            sorted(root.glob("exp3c_oracle_beta*")),
        ),
    }
    saved = []
    for group, (title, exp_dirs) in groups.items():
        out = plot_setting_group(group, title, exp_dirs, output_dir)
        if out is not None:
            saved.append(out)
    return saved


def _plot_stats(ax, stats: dict[str, dict[str, np.ndarray]], methods: tuple[str, ...],
                label_fn, baseline: dict[str, np.ndarray] | None = None) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, max(len(stats), 1)))
    for color, (label, values) in zip(colors, stats.items()):
        ax.plot(values["iters"], values["mean"], label=label, color=color,
                linewidth=2.2, marker="o", markersize=4.5,
                markevery=max(len(values["iters"]) // 12, 1))
        ax.fill_between(values["iters"], values["mean"] - values["std"],
                        values["mean"] + values["std"], color=color, alpha=0.12)
    if baseline is not None:
        max_iteration = max((int(values["iters"][-1]) for values in stats.values()), default=None)
        baseline_iters = baseline["iters"]
        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]
        if max_iteration is not None:
            keep = baseline_iters <= max_iteration
            baseline_iters = baseline_iters[keep]
            baseline_mean = baseline_mean[keep]
            baseline_std = baseline_std[keep]
        ax.plot(baseline_iters, baseline_mean, label="RE-HG",
                color="#D55E00", linewidth=2.4, linestyle="--")
        ax.fill_between(baseline_iters, baseline_mean - baseline_std,
                        baseline_mean + baseline_std, color="#D55E00", alpha=0.10)
    ax.set_xlabel("Leader Iteration", fontsize=20)
    ax.set_ylabel("Disc. Leader Return", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=17)


def _plot_reference_curves(
    ax, reference: dict[str, dict[str, np.ndarray]], max_iteration: int | None = None
) -> None:
    """Add Oracle, No-Influence, and matched RE-HG reference curves."""
    styles = {
        "softvi": ("Oracle", "black", "--"),
        "non_irl": ("No-Influence", "#009E73", ":"),
        "irl_recursive_rls": ("RE-HG (matched setting)", "#D55E00", "--"),
    }
    for method, (label, color, linestyle) in styles.items():
        values = reference.get(method)
        if values is None:
            continue
        keep = np.ones(len(values["iters"]), dtype=bool)
        if max_iteration is not None:
            keep = values["iters"] <= max_iteration
        iters = values["iters"][keep]
        mean = values["mean"][keep]
        std = values["std"][keep]
        ax.plot(iters, mean, label=label, color=color, linestyle=linestyle,
                linewidth=2.4)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.10)


def _group_method_curves(exp_dirs: list[Path], method: str, group: str) -> dict[str, dict[str, np.ndarray]]:
    curves = {}
    for exp_dir in exp_dirs:
        values = aggregate_method_curves(exp_dir).get(method)
        if values is not None:
            curves[_setting_label(exp_dir, group)] = values
    return curves


def plot_exp3_comparison(root: Path, output_dir: Path, baseline_dir: Path) -> Path | None:
    baseline = aggregate_method_curves(baseline_dir).get("irl_recursive_rls")
    lr_dirs = sorted(root.parent.joinpath("rebuttal1").glob("exp3_oracle_lr*"))
    groups = [
        ("Oracle: learning-rate adjustment", _group_method_curves(lr_dirs, "softvi", "lr")),
        ("Oracle: gradient clipping", _group_method_curves(sorted(root.glob("exp3b_oracle_gnorm_lr*")), "softvi", "ognorm")),
        ("Oracle: influence-term scaling", _group_method_curves(sorted(root.glob("exp3c_oracle_beta*")), "softvi", "beta")),
    ]
    if not any(curves for _, curves in groups):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8), dpi=180, squeeze=False)
    for ax, (title, curves) in zip(axes[0], groups):
        ax.set_title(title, fontsize=26)
        _plot_stats(ax, curves, ("softvi",), lambda x: x, baseline)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "exp3_oracle_comparison_with_q30_rehg.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_exp12_grouped(root: Path, output_dir: Path, baseline_dir: Path) -> list[Path]:
    exp2b_dirs = sorted(root.glob("exp2b_lowspec_excitation_full"))
    exp2b_cost_dir = root / "exp2b_lowspec_cost_sev0.0"
    reference = aggregate_method_curves(baseline_dir)
    definitions = {
        "exp1": [
            ("Myopic follower", sorted(root.glob("exp1a_*")), "T"),
            ("Under-reacting follower", sorted(root.glob("exp1b_*")), "a"),
        ],
        "exp2": [
            ("Basis mismatch", sorted(root.glob("exp2_misspec_*")), "d"),
            ("Representation-limited reward model", exp2b_dirs, "excitation"),
            ("Redundant features", sorted(root.glob("exp2c_*")), "p"),
        ],
    }
    saved = []
    for name, panels in definitions.items():
        available = [(title, dirs, label_group) for title, dirs, label_group in panels if dirs]
        if not available:
            continue
        fig, axes = plt.subplots(1, len(available), figsize=(7.2 * len(available), 5.8),
                                 dpi=180, squeeze=False)
        for ax, (title, dirs, label_group) in zip(axes[0], available):
            curves = _group_method_curves(dirs, "irl_recursive_rls", label_group)
            if title.startswith("Representation-limited") and exp2b_cost_dir.is_dir():
                top3 = aggregate_method_curves(exp2b_cost_dir).get("irl_recursive_rls")
                if top3 is not None:
                    curves["top-3 weighted features fixed at true values"] = top3
            ax.set_title(title, fontsize=26)
            max_iteration = max(
                (int(values["iters"][-1]) for values in curves.values()),
                default=None,
            )
            _plot_stats(ax, curves, ("irl_recursive_rls",), lambda x: x)
            _plot_reference_curves(ax, reference, max_iteration)
            ax.legend(loc="best", frameon=True, fontsize=17)
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{name}_grouped_with_q30_rehg.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(out_path)
    return saved


def plot_cao_comparison(exp_dir: Path, output_dir: Path) -> Path | None:
    """Plot the discrete toy environment CAO and RE-HG curves together."""
    stats = aggregate_method_curves(exp_dir)
    methods = ("irl_known_dynamics_every_iter", "irl_recursive_rls")
    if not all(method in stats for method in methods):
        return None

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    for method in methods:
        values = stats[method]
        style = method_style(method)
        ax.plot(
            values["iters"],
            values["mean"],
            label="CAO" if method == "irl_known_dynamics_every_iter" else "RE-HG",
            linewidth=2.4,
            markersize=5.0,
            markevery=max(len(values["iters"]) // 12, 1),
            **style,
        )
        ax.fill_between(
            values["iters"],
            values["mean"] - values["std"],
            values["mean"] + values["std"],
            color=style["color"],
            alpha=0.14,
        )

    ax.set_title("CAO vs RE-HG (discrete toy environment)")
    ax.set_xlabel("Leader Iteration", fontsize=20)
    ax.set_ylabel("Disc. Leader Return", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=17)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cao_vs_rehg_discrete_toyenv_eval_curve.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval curves from rebuttal txt logs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/internal/rebuttal"),
        help="Root directory containing exp* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/internal/rebuttal/plots"),
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--cao-root",
        type=Path,
        default=Path("data/internal/discrete_toyenv/test30sq50gamma0.9beta0.25randomstart"),
        help="Discrete toy environment directory containing CAO and RE-HG logs.",
    )
    args = parser.parse_args()

    exp_dirs = sorted(
        p for p in args.root.iterdir()
        if p.is_dir()
        and not p.name.endswith("_ep800")
        and not p.name.startswith("exp1c_")
    )
    if not exp_dirs:
        raise SystemExit(f"No experiment directories found under: {args.root}")

    saved = []
    for exp_dir in exp_dirs:
        out = plot_single_experiment(exp_dir, args.output_dir)
        if out is not None:
            saved.append(out)

    saved.extend(plot_setting_groups(args.root, args.output_dir))
    baseline_dir = Path("data/internal/conf_continuous/next1/test18secondthirdbaseQ30")
    exp3_comparison = plot_exp3_comparison(args.root, args.output_dir, baseline_dir)
    if exp3_comparison is not None:
        saved.append(exp3_comparison)
    saved.extend(plot_exp12_grouped(args.root, args.output_dir, baseline_dir))
    cao = plot_cao_comparison(args.cao_root, args.output_dir)
    if cao is not None:
        saved.append(cao)

    if not saved:
        raise SystemExit("No figures generated. Check log format.")

    print("Generated figures:")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()