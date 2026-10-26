"""Leader evaluation utilities for bilevel RL."""

import numpy as np


def normalize_obs_action(obs, leader_act):
    """Normalize observation and leader action to integer values."""
    if hasattr(obs, "cpu"):
        obs = obs.cpu().numpy()
    if isinstance(obs, np.ndarray):
        flat = obs.flatten()
        obs_val = int(flat[0]) if flat.size == 1 else int(np.argmax(flat))
    else:
        obs_val = int(obs)

    if hasattr(leader_act, "cpu"):
        leader_act = leader_act.cpu().numpy()
    if isinstance(leader_act, np.ndarray):
        flat = leader_act.flatten()
        leader_act_val = int(flat[0]) if flat.size == 1 else int(np.argmax(flat))
    else:
        leader_act_val = int(leader_act)

    return obs_val, leader_act_val


class LeaderEvaluator:
    """Handles leader policy evaluation and display."""

    def __init__(self, env_spec):
        self.env_spec = env_spec

    def evaluate(self, env, leader_policy_obj, follower_policy_fn,
                 n_episodes=10, return_trajectories=False, gamma=1.0):
        total_returns = []
        disc_returns = []
        all_trajectories = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            disc_return = 0.0
            step_count = 0

            trajectory = {
                "observations": [],
                "leader_actions": [],
                "follower_actions": [],
                "rewards": [],
                "leader_rewards": [],
            }

            while True:
                # Get joint action
                leader_act = leader_policy_obj.sample_action(obs)

                if isinstance(obs, np.ndarray):
                    obs_for_follower = (
                        obs.astype(np.int32) if obs.dtype != np.int32 else obs
                    )

                if isinstance(leader_act, np.ndarray):
                    leader_act_for_follower = (
                        leader_act.astype(np.int32) if leader_act.dtype != np.int32 else leader_act
                    )
                else:
                    leader_act_for_follower = np.array(int(leader_act), dtype=np.int32)

                follower_act = follower_policy_fn(obs_for_follower, leader_act_for_follower)

                env_step = env.step(leader_act, follower_act)
                leader_reward = env_step.env_info.get("leader_reward", env_step.reward)

                trajectory["observations"].append(int(obs))
                trajectory["leader_actions"].append(int(leader_act))
                trajectory["follower_actions"].append(int(follower_act))
                trajectory["rewards"].append(float(env_step.reward))
                trajectory["leader_rewards"].append(float(leader_reward))

                episode_return += leader_reward
                disc_return += leader_reward * (gamma ** step_count)
                step_count += 1
                obs = env_step.observation

                if env_step.last:
                    break

            total_returns.append(episode_return)
            disc_returns.append(disc_return)
            all_trajectories.append(trajectory)

        result = {
            "mean": np.mean(total_returns),
            "std": np.std(total_returns),
            "disc_mean": np.mean(disc_returns),
            "disc_std": np.std(disc_returns),
        }

        if return_trajectories:
            result["trajectories"] = all_trajectories

        return result

    def display_leader_policy(self, iteration, leader_q_table, leader_policy_obj):
        """Display leader policy probabilities."""
        print(f"\n--- Leader Policy at Iteration {iteration} ---")

        num_states = leader_q_table.shape[0]

        for s in range(num_states):
            probs = leader_policy_obj.policy_table[s]
            probs_str = "  ".join([f"P(a={a})={p:.4f}" for a, p in enumerate(probs)])
            print(f"State {s}: {probs_str}")
        print("-" * 60)

    @staticmethod
    def make_true_follower_eval_fn(true_follower_model):
        """Create an evaluation follower policy function from a true follower model."""
        def policy_fn(obs, leader_act):
            obs_val, leader_act_val = normalize_obs_action(obs, leader_act)
            return true_follower_model.sample_action(obs_val, leader_act_val)
        return policy_fn
