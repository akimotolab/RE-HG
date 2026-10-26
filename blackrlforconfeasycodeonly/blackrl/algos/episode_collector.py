"""Episode collection utilities for bilevel RL."""

import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch


def _create_env_from_info(env_class_name, env_module_path, env_init_kwargs):
    module = importlib.import_module(env_module_path)
    env_class = getattr(module, env_class_name)
    if env_init_kwargs:
        return env_class(**env_init_kwargs)
    return env_class()


def _get_joint_action_from_state(
    obs,
    leader_policy_table,
    follower_q_values,
    num_follower_actions,
    temperature,
):
    state_int = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
    leader_probs = leader_policy_table[state_int]
    leader_act = int(np.random.choice(len(leader_probs), p=leader_probs))

    q_vals = []
    for b in range(num_follower_actions):
        state_key = (state_int,)
        leader_key = (leader_act,)
        follower_key = (b,)
        q_val = follower_q_values.get(state_key, {}).get(leader_key, {}).get(follower_key, 0.0)
        q_vals.append(q_val)

    q_tensor = torch.tensor(q_vals)
    probs = torch.softmax(q_tensor / temperature, dim=0).numpy()
    follower_act = int(np.random.choice(num_follower_actions, p=probs))

    return leader_act, follower_act


def _collect_single_episode(
    env_class_name,
    env_module_path,
    env_init_kwargs,
    leader_policy_table,
    follower_q_values,
    num_follower_actions,
    temperature,
):
    env = _create_env_from_info(env_class_name, env_module_path, env_init_kwargs)
    obs, _ = env.reset()
    time_step = 0
    observations = []
    leader_actions = []
    follower_actions = []
    rewards = []
    leader_rewards = []
    next_observations = []
    terminals = []
    last_flags = []
    time_steps = []

    while True:
        leader_act, follower_act = _get_joint_action_from_state(
            obs,
            leader_policy_table,
            follower_q_values,
            num_follower_actions,
            temperature,
        )

        env_step = env.step(leader_act, follower_act)

        leader_reward = env_step.env_info.get("leader_reward", env_step.reward)

        observations.append(obs)
        leader_actions.append(leader_act)
        follower_actions.append(follower_act)
        rewards.append(env_step.reward)
        leader_rewards.append(leader_reward)
        next_observations.append(env_step.observation)
        terminals.append(1 if env_step.terminal else 0)
        last_flags.append(1 if env_step.last else 0)
        time_steps.append(time_step)

        obs = env_step.observation
        time_step += 1

        if env_step.last:
            break

    return {
        "observation": observations,
        "leader_action": leader_actions,
        "action": follower_actions,
        "reward": rewards,
        "leader_reward": leader_rewards,
        "next_observation": next_observations,
        "terminal": terminals,
        "last": last_flags,
        "time_step": time_steps,
    }


class EpisodeCollector:
    """Handles parallel episode collection."""

    def __init__(self, env_spec, temperature):
        self.env_spec = env_spec
        self.temperature = temperature

    def _get_env_info(self, env):
        """Extract serializable environment info for parallel collection."""
        env_class = type(env)
        return {
            "env_class_name": env_class.__name__,
            "env_module_path": env_class.__module__,
            "env_init_kwargs": getattr(env, "_init_kwargs", {}),
        }

    def _serialize_q_values(self, model):
        """Extract Q-values from a model with Q attribute as serializable dict."""
        q_values = {}
        if model is not None and hasattr(model, "Q"):
            for sk, ld in model.Q.items():
                q_values[sk] = {}
                for lk, fd in ld.items():
                    q_values[sk][lk] = dict(fd)
        return q_values

    def collect_episodes(self, env, leader_policy_table, follower_model, n_episodes):
        """Collect episodes in parallel.

        Returns:
            dict with transition data lists (observation, leader_action, action,
            reward, leader_reward, next_observation, terminal, last, time_step).
        """
        env_info = self._get_env_info(env)
        follower_q_values = self._serialize_q_values(follower_model)
        n_fa = self.env_spec.action_space.n
        leader_policy_copy = leader_policy_table.copy()

        max_workers = min(n_episodes, 8)

        all_data = {
            "observation": [], "leader_action": [], "action": [],
            "reward": [], "leader_reward": [], "next_observation": [],
            "terminal": [], "last": [], "time_step": [],
        }

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _collect_single_episode,
                    env_info["env_class_name"], env_info["env_module_path"],
                    env_info["env_init_kwargs"], leader_policy_copy,
                    follower_q_values, n_fa, self.temperature,
                )
                for _ in range(n_episodes)
            ]
            for future in as_completed(futures):
                data = future.result()
                for key in all_data:
                    all_data[key].extend(data[key])

        return all_data

    def generate_expert_trajectories(self, env, leader_policy_table, true_follower_model,
                                     n_trajectories, verbose=False):
        """Generate expert trajectories for IRL."""
        env_info = self._get_env_info(env)
        leader_policy_copy = leader_policy_table.copy()
        expert_q_values = self._serialize_q_values(true_follower_model)
        n_fa = self.env_spec.action_space.n
        max_workers = min(n_trajectories, 8)

        if verbose:
            print(f"  Generating {n_trajectories} expert trajectories with {max_workers} workers...")

        trajectories = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _collect_single_episode,
                    env_info["env_class_name"], env_info["env_module_path"],
                    env_info["env_init_kwargs"], leader_policy_copy,
                    expert_q_values, n_fa, self.temperature,
                )
                for _ in range(n_trajectories)
            ]
            for future in as_completed(futures):
                raw_traj = future.result()
                formatted_traj = {
                    "observations": raw_traj["observation"],
                    "leader_actions": raw_traj["leader_action"],
                    "follower_actions": raw_traj["action"],
                    "rewards": raw_traj["reward"],
                }
                trajectories.append(formatted_traj)

        return trajectories
