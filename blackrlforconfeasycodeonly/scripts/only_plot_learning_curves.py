import argparse
import pickle
from pathlib import Path

from plot_learning_curves import plot_components, plot_learning_curves, plot_leader_performance_only


MODE_LABELS = {
    # discrete toy
    "irl": "Proposed (IRL)",
    "expert_surrogate_no_second_term": " Baseline (Expert surrogate, no 2nd term)",
    "irl_known_dynamics": "Proposed (IRL, known dyn)",
    "irl_known_dynamics_every_iter": "Proposed (IRL, known dyn, every iter)",
    "softql": "Proposed (SoftQL)",
    "softvi": "Oracle (SoftVI)",
    "softvi_known_dynamics": "Oracle (SoftVI, known dyn)",
    # conf continuous
    "irl_recursive_rls": "Proposed (IRL aggregate)",
    "non_irl": "Baseline Non-IRL",
}


def _to_plot_stats(data: dict) -> dict:
    """Normalize either raw stats_*.pkl or aggregated_*.pkl to plotting format."""
    if "leader_return_mean" in data or "eval_leader_return_mean" in data:
        prefix = "eval_" if "eval_leader_return_mean" in data else ""
        return {
            "leader_return": data.get(f"{prefix}leader_return_mean", []),
            "leader_return_std": data.get(f"{prefix}leader_return_std", []),
            "stability": data.get(f"{prefix}stability_mean", []),
            "stability_std": data.get(f"{prefix}stability_std", []),
            "energy_cost": data.get(f"{prefix}energy_cost_mean", []),
            "energy_cost_std": data.get(f"{prefix}energy_cost_std", []),
            "insulation_cost": data.get(f"{prefix}insulation_cost_mean", []),
            "insulation_cost_std": data.get(f"{prefix}insulation_cost_std", []),
            "airflow_cost": data.get(f"{prefix}airflow_cost_mean", []),
            "airflow_cost_std": data.get(f"{prefix}airflow_cost_std", []),
            "mode": data.get("mode", ""),
        }
    return data


def _load_mode_stats(data_dir: Path, mode: str) -> dict | None:
    cands = [
        data_dir / f"aggregated_{mode}.pkl",
        data_dir / f"stats_{mode}.pkl",
        data_dir / f"stats_{mode}_seed0.pkl",
    ]
    for p in cands:
        if p.exists():
            with open(p, "rb") as f:
                return _to_plot_stats(pickle.load(f))
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot only selected modes from an output directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--modes",
        help="Comma-separated mode keys, e.g. softvi,irl_known_dynamics. If omitted, all detected modes are plotted.",
    )
    parser.add_argument("--main-mode", default=None, help="Main mode key used as highlighted curve")
    parser.add_argument(
        "--plot-type",
        choices=["leader", "all", "components"],
        default="leader",
        help="leader: only leader return, all: 2x3 summary, components: 2x2 component plots",
    )
    args = parser.parse_args()

    data_dir = Path(args.output_dir)
    
    if args.modes:
        mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        # Auto-detect modes from stats_*.pkl or aggregated_*.pkl files
        print("No modes specified. Detecting all modes in output directory...")
        mode_list = []
        for p in data_dir.glob("*.pkl"):
            name = p.stem
            if name.startswith("stats_"):
                mode = name[6:]
                if mode.endswith("_seed0"): # Handle seed0 separately if needed
                    mode = mode[:-6]
                if mode not in mode_list:
                    mode_list.append(mode)
            elif name.startswith("aggregated_"):
                mode = name[11:]
                if mode not in mode_list:
                    mode_list.append(mode)
        print(f"Detected modes: {', '.join(mode_list)}")

    loaded: dict[str, dict] = {}
    for mode in mode_list:
        s = _load_mode_stats(data_dir, mode)
        if s is None:
            print(f"Warning: no stats found for mode={mode}")
            continue
        loaded[mode] = s
        print(f"Loaded mode={mode}")

    if not loaded:
        print("No results found to plot.")
        return

    main_mode = args.main_mode if args.main_mode in loaded else next(iter(loaded.keys()))
    main_label = MODE_LABELS.get(main_mode, main_mode)
    main_stats = loaded[main_mode]

    baselines = {
        MODE_LABELS.get(k, k): v
        for k, v in loaded.items()
        if k != main_mode
    }

    if args.plot_type == "leader":
        save_path = data_dir / "leader_performance_only_selected.pdf"
        plot_leader_performance_only(main_stats, save_path=save_path, baselines=baselines, main_label=main_label)
    elif args.plot_type == "all":
        save_path = data_dir / "comparison_curves_selected.pdf"
        plot_learning_curves(main_stats, save_path=save_path, baselines=baselines, main_label=main_label)
    else:
        save_path = data_dir / "components_curves_selected.pdf"
        plot_components(main_stats, save_path=save_path, baselines=baselines, main_label=main_label)

    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
