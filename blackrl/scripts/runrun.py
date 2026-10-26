"""Training script for Bi-level RL with Baselines (Parallel Ready)."""

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path
import random

import numpy as np
import torch
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnvPaper
from plot_learning_curves import plot_leader_performance_split


COMMON_PARAMS = {
    "discount_leader": 0.99,
    "discount_follower": 0.9,
    "learning_rate_leader_actor": 1e-2,
    "learning_rate_leader_critic": 1e-3,
    "learning_rate_follower": 0.01,
    "mdce_irl_config": {
        "max_iterations": 30,
        "tolerance": 0.05,
        "n_soft_q_iterations": 50,
        "learning_rate": 0.05,
        "lr_decay": 0.0,
        "irl_rollout_episodes": 100,
    },
    "soft_q_config": {
        "learning_rate": 0.1,
        "temperature": 0.25,
        "epsilon": 0,
        "oracle_softql_full_batch": True,
        "oracle_softql_batch_updates": 5,
        "adapt_softql_full_batch": True,
        "adapt_softql_batch_updates": 50,
        "adaptation_episodes": 50,
    },
}

TRAIN_PARAMS = {
    "n_leader_iterations": 300,
    "n_episodes_per_iteration": 50,
    "n_expert_episodes_per_iteration": 200,
    "mdce_irl_frequency": 10,
    "recursive_rls_irl_iterations": [iter for iter in range(100) if iter%10==0],
    "reward_info_eigen_threshold": 1e-23,
    "verbose": True,
}


def create_simple_leader_policy():
    return np.random.choice([0, 1], p=[0.5, 0.5])


def feature_fn(state, leader_action, follower_action):
    num_states, num_leader_actions, num_follower_actions = 3, 2, 3
    total_dim = num_states * num_leader_actions * num_follower_actions

    s = int(np.asarray(state).item())
    a = int(np.asarray(leader_action).item())
    b = int(np.asarray(follower_action).item())

    index = s * (num_leader_actions * num_follower_actions) + a * num_follower_actions + b
    feature = np.zeros(total_dim, dtype=np.float32)
    feature[index] = 1.0
    return feature


def _aggregate_seed_stats(stats_by_seed: dict) -> dict:
    """Aggregate per-iteration returns over seeds for plotting compatibility."""
    if not stats_by_seed:
        return {}

    first_stats = next(iter(stats_by_seed.values()))
    n_iters = min(len(v.get("leader_return", [])) for v in stats_by_seed.values())
    if n_iters == 0:
        return {"by_seed": stats_by_seed}

    returns = np.array(
        [np.asarray(v["leader_return"][:n_iters], dtype=np.float64) for v in stats_by_seed.values()],
        dtype=np.float64,
    )
    stds = []
    for v in stats_by_seed.values():
        if "leader_return_std" in v and len(v["leader_return_std"]) >= n_iters:
            stds.append(np.asarray(v["leader_return_std"][:n_iters], dtype=np.float64))
    std_arr = np.array(stds, dtype=np.float64) if stds else np.zeros_like(returns)

    agg = dict(first_stats)
    agg["leader_return"] = returns.mean(axis=0).tolist()
    agg["leader_return_std"] = returns.std(axis=0).tolist()
    agg["leader_return_seed_sem"] = (
        returns.std(axis=0, ddof=0) / max(np.sqrt(returns.shape[0]), 1.0)
    ).tolist()
    agg["leader_return_episode_std_mean"] = std_arr.mean(axis=0).tolist() if std_arr.size > 0 else []

    if "disc_leader_return" in first_stats:
        disc_returns = np.array(
            [np.asarray(v["disc_leader_return"][:n_iters], dtype=np.float64) for v in stats_by_seed.values()],
            dtype=np.float64,
        )
        agg["disc_leader_return"] = disc_returns.mean(axis=0).tolist()
        agg["disc_leader_return_std"] = disc_returns.std(axis=0).tolist()

    keys_to_aggregate = [
        "reward_pearson_corr", "reward_weighted_pearson_corr",
        "hg_cosine_similarity", "hg_oracle_norm", "hg_estimated_norm", "reward_var_diff"
    ]
    for key in keys_to_aggregate:
        if key in first_stats:
            try:
                arrs = [np.asarray(v[key][:n_iters], dtype=np.float64) for v in stats_by_seed.values()]
                arrs_stacked = np.array(arrs, dtype=np.float64)
                agg[f"{key}_mean"] = arrs_stacked.mean(axis=0).tolist()
                agg[f"{key}_std"] = arrs_stacked.std(axis=0).tolist()
            except Exception:
                pass

    agg["seeds"] = list(stats_by_seed.keys())
    agg["by_seed"] = stats_by_seed
    return agg


def _resolve_mode(mode: str) -> tuple[str, bool]:
    """Resolve oracle_mode and second-term flag from mode name."""
    if mode == "irl":
        oracle_arg = "none"
    elif mode == "expert_surrogate_no_second_term":
        oracle_arg = "expert_surrogate_no_second_term"
    elif mode == "expert_surrogate_known_dynamics_no_second_term":
        oracle_arg = "expert_surrogate_known_dynamics_no_second_term"
    else:
        oracle_arg = mode
    use_second_term = mode not in (
        "expert_surrogate_no_second_term",
        "expert_surrogate_known_dynamics_no_second_term",
    )
    return oracle_arg, use_second_term


def _run_single_seed(
    mode: str,
    output_dir_str: str,
    seed: int,
    oracle_arg: str,
    use_second_term: bool,
    train_params: dict,
    follower_softvi_iters: int = 0,
) -> tuple[int, dict]:
    """Run a single seed experiment and persist per-seed artifacts."""
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_log_path = output_dir / f"logs_{mode}__seed_{seed}.txt"
    with open(seed_log_path, "w", buffering=1) as seed_log_file:
        with contextlib.redirect_stdout(seed_log_file), contextlib.redirect_stderr(seed_log_file):
            print(f"=== Mode: {mode} / Seed: {seed} ===")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = DiscreteToyEnvPaper(random_start=True)
            env.seed(seed)

            algo = BilevelRL(
                env_spec=env.spec,
                leader_policy=create_simple_leader_policy(),
                feature_fn=feature_fn,
                transition_fn=env.transition_fn,
                **COMMON_PARAMS,
            )
            # Suboptimal-follower experiment (0 = exact best response, i.e. the
            # published setting). Truncating soft value iteration gives a myopic
            # follower; the discrete case isolates the optimal-follower
            # assumption because one-hot features make Assumption 1 exact.
            algo.follower_softvi_iters = int(follower_softvi_iters)

            stats = algo.train(
                env=env,
                oracle_mode=oracle_arg,
                use_second_term=use_second_term,
                run_seed=seed,
                **train_params,
            )

    seed_save_path = output_dir / f"stats_{mode}__seed_{seed}.pkl"
    with open(seed_save_path, "wb") as f:
        pickle.dump(stats, f)

    return seed, stats


def run_experiment(mode, output_dir, seeds, seed_parallel=False, max_seed_workers=1,
                   follower_softvi_iters=0):
    print(f"\n=== Starting Experiment: {mode} ===")

    oracle_arg, use_second_term = _resolve_mode(mode)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_by_seed = {}

    do_parallel = bool(seed_parallel) and len(seeds) > 1
    if do_parallel:
        max_workers = max(1, min(int(max_seed_workers), len(seeds)))
        print(f"Running seeds in parallel (workers={max_workers})")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_seed,
                    mode,
                    str(output_dir),
                    int(seed),
                    oracle_arg,
                    use_second_term,
                    TRAIN_PARAMS,
                    int(follower_softvi_iters),
                )
                for seed in seeds
            ]
            for future in as_completed(futures):
                seed, stats = future.result()
                stats_by_seed[f"seed_{seed}"] = stats
                print(f"Saved per-seed stats: {output_dir / f'stats_{mode}__seed_{seed}.pkl'}")
    else:
        for seed in seeds:
            print(f"--- Seed: {seed} ---")
            seed_done, stats = _run_single_seed(
                mode=mode,
                output_dir_str=str(output_dir),
                seed=int(seed),
                oracle_arg=oracle_arg,
                use_second_term=use_second_term,
                train_params=TRAIN_PARAMS,
                follower_softvi_iters=int(follower_softvi_iters),
            )
            stats_by_seed[f"seed_{seed_done}"] = stats
            print(f"Saved per-seed stats: {output_dir / f'stats_{mode}__seed_{seed_done}.pkl'}")

    if len(stats_by_seed) == 1:
        combined_stats = next(iter(stats_by_seed.values()))
    else:
        combined_stats = _aggregate_seed_stats(stats_by_seed)
        combined_stats["seeds"] = list(seeds)

    save_path = output_dir / f"stats_{mode}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(combined_stats, f)

    if len(stats_by_seed) > 1:
        multi_save_path = output_dir / f"stats_{mode}__multi_seed.pkl"
        with open(multi_save_path, "wb") as f:
            pickle.dump(combined_stats, f)
        print(f"Saved multi-seed aggregate: {multi_save_path}")

    print(f"=== Finished {mode}. Saved to {save_path} ===")


def merge_and_plot(output_dir):
    output_dir = Path(output_dir)
    print(f"\n=== Merging Results and Plotting (from {output_dir}) ===")

    modes = {
        "irl": "RE-HG (Single)",
        "irl_recursive_rls": "RE-HG",
        "expert_surrogate_no_second_term": "No-Influence",
        "irl_known_dynamics": "RE-HG (Single)",
        "expert_surrogate_known_dynamics_no_second_term": "No-Influence",
        "irl_known_dynamics_every_iter": "RE-HG",
        "softql": "Oracle",
        "softvi": "Oracle (SoftVI)",
        "softvi_known_dynamics": "Oracle",
    }

    unknown_modes = ["softql", "irl", "irl_recursive_rls",  "expert_surrogate_no_second_term"]
    known_modes = ["softvi_known_dynamics","irl_known_dynamics", "irl_known_dynamics_every_iter", "expert_surrogate_known_dynamics_no_second_term"]

    results_unknown = {}
    for mode_key in unknown_modes:
        if mode_key not in modes:
            continue
        mode_name = modes[mode_key]
        path = output_dir / f"stats_{mode_key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                results_unknown[mode_name] = pickle.load(f)
            print(f"Loaded (Unknown Dynamics): {mode_name} from {mode_key}")
        else:
            print(f"Warning: Result file for {mode_key} not found at {path}")

    results_known = {}
    for mode_key in known_modes:
        if mode_key not in modes:
            continue
        mode_name = modes[mode_key]
        path = output_dir / f"stats_{mode_key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                results_known[mode_name] = pickle.load(f)
            print(f"Loaded (Known Dynamics): {mode_name} from {mode_key}")
        else:
            print(f"Warning: Result file for {mode_key} not found at {path}")

    if not results_unknown and not results_known:
        print("No results found to plot.")
        return

    save_path = output_dir / "comparison_curves.pdf"
    plot_leader_performance_split(results_unknown, results_known, save_path=save_path)
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
    parser.add_argument(
        "--reward-info-update",
        type=str,
        default=None,
        choices=["eigenvalue", "analytic", "joint_cao"],
        help="Reward aggregation rule (Xcsj-W2/Q3). 'eigenvalue' is RE-HG's "
             "projection-and-truncation (default). 'joint_cao' replaces it by the "
             "pooled multi-environment baseline in the formulation of Cao, Cohen and "
             "Szpruch (2021, Sec. 7.2): one shared reward and per-environment value "
             "functions fitted jointly to the soft-Bellman residual on the SAME "
             "accumulated datasets D_1..D_k, with no extra environment interaction "
             "and no forward solve",
    )
    parser.add_argument("--joint-cao-chunk", type=int, default=None,
                        help="Re-solve the pooled baseline every N outer iterations (default 1)")
    parser.add_argument("--joint-cao-iters", type=int, default=None,
                        help="Adam steps per pooled re-solve (default 2000, as in Sec. 7.2)")
    parser.add_argument("--joint-cao-ridge", type=float, default=None,
                        help="Tikhonov penalty for the regularized arm of the pooled baseline")
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="Override the number of leader iterations (smoke tests)")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds. Examples: '0' or '0,1,2,3,4'",
    )
    parser.add_argument(
        "--seed-parallel",
        action="store_true",
        help="Run seeds in parallel worker processes.",
    )
    parser.add_argument(
        "--max-seed-workers",
        type=int,
        default=1,
        help="Maximum parallel workers for seed runs when --seed-parallel is set.",
    )
    parser.add_argument(
        "--follower-softvi-iters",
        type=int,
        default=0,
        help="Suboptimal-follower experiment: truncate the follower's soft value "
             "iteration after this many sweeps (0 = exact best response, the "
             "published setting). Only the follower's behaviour is truncated; "
             "each method's internal belief stays exact.",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    # Pooled-baseline overrides. Untouched keys leave TRAIN_PARAMS exactly as
    # published, so the default run is bit-identical to before.
    if args.reward_info_update is not None:
        TRAIN_PARAMS["reward_info_update"] = args.reward_info_update
        print(f"[reward-aggregation] reward_info_update = {args.reward_info_update}")
    if args.joint_cao_chunk is not None:
        TRAIN_PARAMS["joint_cao_chunk"] = int(args.joint_cao_chunk)
    if args.joint_cao_iters is not None:
        TRAIN_PARAMS["joint_cao_iters"] = int(args.joint_cao_iters)
    if args.joint_cao_ridge is not None:
        TRAIN_PARAMS["joint_cao_ridge"] = float(args.joint_cao_ridge)
    if args.n_iterations is not None:
        TRAIN_PARAMS["n_leader_iterations"] = int(args.n_iterations)
        # The IRL schedule is a fixed list of iteration indices; clip it so a
        # short smoke run still triggers the reward updates.
        TRAIN_PARAMS["recursive_rls_irl_iterations"] = [
            i for i in TRAIN_PARAMS.get("recursive_rls_irl_iterations", [])
            if i < int(args.n_iterations)
        ]
        print(f"[smoke] n_leader_iterations = {args.n_iterations}")

    if args.mode == "plot":
        merge_and_plot(args.output_dir)
    else:
        run_experiment(
            args.mode,
            args.output_dir,
            seeds=seeds,
            seed_parallel=args.seed_parallel,
            max_seed_workers=args.max_seed_workers,
            follower_softvi_iters=args.follower_softvi_iters,
        )


if __name__ == "__main__":
    main()