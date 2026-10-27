"""Training script for Bi-level RL with Baselines (Parallel Ready)."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnvPaper
from plot_learning_curves import plot_learning_curves

COMMON_PARAMS = {
    "discount_leader": 0.99,
    "discount_follower": 0.9,
    "learning_rate_leader_actor": 1e-2,
    "learning_rate_leader_critic": 1e-3,
    "learning_rate_follower": 0.01,
    "mdce_irl_config": {
        "max_iterations": 300,
        "tolerance": 0.05,
        "n_soft_q_iterations": 300,
        "learning_rate": 0.005,
        "lr_decay": 0.002,
        "irl_rollout_episodes": 100,
    },
    "soft_q_config": {
        "learning_rate": 0.1,
        "temperature": 0.25,
        "epsilon": 0,
        "oracle_softql_full_batch": True,
        "oracle_softql_batch_updates": 5,
        "adapt_softql_full_batch": True,
        "adapt_softql_batch_updates": 5,
        "adaptation_episodes": 50,
    },
}

TRAIN_PARAMS = {
    "n_leader_iterations": 1000,
    "n_episodes_per_iteration": 10,
    "n_expert_episodes_per_iteration": 200,
    "mdce_irl_frequency": 10,
    "verbose": True,
}


def create_simple_leader_policy():
    return np.random.choice([0, 1], p=[0.5, 0.5])


def feature_fn(state, leader_action, follower_action):
    num_states, num_leader_actions, num_follower_actions = 3, 2, 3
    total_dim = num_states * num_leader_actions * num_follower_actions

    # Assume scalar-compatible inputs and map (s, a, b) to a flat index.
    s = int(np.asarray(state).item())
    a = int(np.asarray(leader_action).item())
    b = int(np.asarray(follower_action).item())

    index = s * (num_leader_actions * num_follower_actions) + a * num_follower_actions + b
    feature = np.zeros(total_dim, dtype=np.float32)
    feature[index] = 1.0
    return feature


def run_experiment(mode, output_dir):
    """Run a single experiment for the given mode and save the results."""
    print(f"\n=== Starting Experiment: {mode} ===")

    env = DiscreteToyEnvPaper(random_start=True)
    # Initialize the algorithm
    algo = BilevelRL(
        env_spec=env.spec,
        leader_policy=create_simple_leader_policy(),
        feature_fn=feature_fn,
        transition_fn=env.transition_fn if mode in ("irl_known_dynamics", "irl_known_dynamics_every_iter", "expert_surrogate_known_dynamics_no_second_term", "softvi_known_dynamics") else None,
        **COMMON_PARAMS,
    )

    # Run training
    # irl uses oracle_mode="none" internally
    # expert_surrogate_no_second_term uses the true SoftVI follower and is a
    # baseline that only disables the second term (no empirical surrogate)
    # irl_known_dynamics uses oracle_mode="irl_known_dynamics"
    # irl_known_dynamics_every_iter runs MDCE IRL at every iteration
    if mode == "irl":
        oracle_arg = "none"
    elif mode == "expert_surrogate_no_second_term":
        oracle_arg = "expert_surrogate_no_second_term"
    elif mode == "expert_surrogate_known_dynamics_no_second_term":
        oracle_arg = "expert_surrogate_known_dynamics_no_second_term"
    else:
        oracle_arg = mode  # pass softvi / softql / irl_known_dynamics / softvi_known_dynamics through as is
    use_second_term = mode not in ("expert_surrogate_no_second_term", "expert_surrogate_known_dynamics_no_second_term")

    stats = algo.train(env=env, oracle_mode=oracle_arg, use_second_term=use_second_term, **TRAIN_PARAMS)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"stats_{mode}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"=== Finished {mode}. Saved to {save_path} ===")


def merge_and_plot(output_dir):
    """Load saved results and plot them."""
    output_dir = Path(output_dir)
    print(f"\n=== Merging Results and Plotting (from {output_dir}) ===")
    results = {}

    # Loading map
    modes = {
        "irl": "Proposed IRL",
        "irl_recursive_rls": "Proposed Aggregate IRL",
        "expert_surrogate_no_second_term": "Baseline No-Influence term",
        "irl_known_dynamics": "Proposed IRL (P)",
        "expert_surrogate_known_dynamics_no_second_term": "Baseline No-Influence term (P)",
        "irl_known_dynamics_every_iter": "Proposed Aggregate IRL (P)",
        "softql": "Oracle (SoftQL)",
        "softvi": "Oracle (SoftVI)",
        "softvi_known_dynamics": "Oracle (SoftVI, P)",
    }

    for mode_key, mode_name in modes.items():
        path = output_dir / f"stats_{mode_key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                results[mode_name] = pickle.load(f)
            print(f"Loaded: {mode_name}")
        else:
            print(f"Warning: Result file for {mode_name} not found at {path}")

    if not results:
        print("No results found to plot.")
        return

    # Plot
    # Assume Proposed (IRL) exists; use it as the main curve and the others as baselines
    main_stats = results.get("Proposed (IRL)", {})
    baselines = {k: v for k, v in results.items() if k != "Proposed (IRL)"}

    save_path = output_dir / "comparison_curves.pdf"
    from plot_learning_curves import plot_leader_performance_only
    plot_leader_performance_only(main_stats, save_path=save_path, baselines=baselines)
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "irl",
            "irl_recursive_rls",
            "expert_surrogate_no_second_term",
            "irl_known_dynamics",
            "expert_surrogate_known_dynamics_no_second_term",
            "irl_known_dynamics_every_iter",
            "softql",
            "softvi",
            "softvi_known_dynamics",
            "plot",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for stats and plots")
    args = parser.parse_args()

    if args.mode == "plot":
        merge_and_plot(args.output_dir)
    else:
        run_experiment(args.mode, args.output_dir)


if __name__ == "__main__":
    main()
