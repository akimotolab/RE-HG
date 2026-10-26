"""Leader critic (Q-table) management for bilevel RL."""

import numpy as np
import torch


class LeaderCritic:
    """Manages leader's Q-table and critic updates."""

    def __init__(self, env_spec, discount_leader, learning_rate):
        self.env_spec = env_spec
        self.discount_leader = discount_leader
        self.learning_rate = learning_rate
        self.q_table = None

    def initialize(self):
        """Initialize Q-table with zeros."""
        obs_space = self.env_spec.observation_space
        leader_action_space = self.env_spec.leader_action_space
        follower_action_space = self.env_spec.action_space

        num_states = obs_space.n
        num_leader_actions = leader_action_space.n
        num_follower_actions = follower_action_space.n

        self.q_table = np.zeros(
            (num_states, num_leader_actions, num_follower_actions),
            dtype=np.float32,
        )

    def update(
        self,
        replay_buffer,
        leader_policy_obj,
        get_follower_probs_fn,
        n_updates=100,
        batch_size=None,
        replace=False,
    ):
        """Update leader's Q-table using sampled transitions.

        Args:
            replay_buffer: Replay buffer to sample from.
            leader_policy_obj: Leader policy object with policy_table.
            get_follower_probs_fn: Callable(state, leader_action) -> np.ndarray of probs.
            n_updates: Number of update steps.
            batch_size: Number of transitions per update. None means full buffer.
            replace: Whether sampling allows replacement.
        """
        if replay_buffer.current_size <= 0:
            return

        full_batch_size = int(replay_buffer.current_size)
        use_batch_size = full_batch_size if batch_size is None else int(batch_size)
        use_batch_size = max(1, min(use_batch_size, full_batch_size))

        for _ in range(n_updates):
            samples = replay_buffer.sample_transitions(
                batch_size=use_batch_size,
                replace=replace,
                discount=False,
                with_subsequence=False,
            )

            obs = samples["observation"]
            leader_acts = samples["leader_action"]
            follower_acts = samples["action"]
            if "leader_reward" in samples:
                rewards = samples["leader_reward"]
            else:
                rewards = samples.get("reward", np.zeros_like(obs))
            next_obs = samples["next_observation"]
            terminals = samples["terminal"]

            # Convert to numpy arrays if needed
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            if isinstance(leader_acts, torch.Tensor):
                leader_acts = leader_acts.cpu().numpy()
            if isinstance(follower_acts, torch.Tensor):
                follower_acts = follower_acts.cpu().numpy()
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.cpu().numpy()
            if isinstance(terminals, torch.Tensor):
                terminals = terminals.cpu().numpy()

            # Flatten arrays
            obs = obs.flatten().astype(int)
            leader_acts = leader_acts.flatten().astype(int)
            follower_acts = follower_acts.flatten().astype(int)
            rewards = rewards.flatten()
            next_obs = next_obs.flatten().astype(int)
            terminals = terminals.flatten().astype(bool)

            # Compute target Q-values
            target_q_values = np.zeros(len(obs))
            for i in range(len(obs)):
                if terminals[i]:
                    target_q_values[i] = rewards[i]
                else:
                    next_state = next_obs[i]
                    v_next = 0.0

                    num_leader_actions = self.env_spec.leader_action_space.n
                    num_follower_actions = self.env_spec.action_space.n

                    for leader_act_next in range(num_leader_actions):
                        leader_prob = leader_policy_obj.policy_table[next_state, leader_act_next]
                        follower_probs = get_follower_probs_fn(next_state, leader_act_next)

                        for follower_act_next in range(num_follower_actions):
                            follower_prob = follower_probs[follower_act_next]
                            q_val = self.q_table[next_state, leader_act_next, follower_act_next]
                            v_next += leader_prob * follower_prob * q_val

                    target_q_values[i] = rewards[i] + self.discount_leader * v_next

                # Q-Learning update
                state = obs[i]
                leader_act = leader_acts[i]
                follower_act = follower_acts[i]

                current_q = self.q_table[state, leader_act, follower_act]
                self.q_table[state, leader_act, follower_act] = current_q + self.learning_rate * (
                    target_q_values[i] - current_q
                )

    def update_model_based(
        self,
        leader_policy_obj,
        get_follower_probs_fn,
        transition_fn,
        leader_reward_fn,
        n_updates=100,
    ):
        """Update leader Q-table with known dynamics (tabular Bellman backup).

        This performs synchronous value-iteration-style backups:
            Q(s,a,b) <- r_L(s,a,b) + gamma * E_{s'|s,a,b}[V(s')]
            V(s')    = E_{a'~pi_L, b'~pi_F}[Q(s',a',b')]
        """
        n_s = self.env_spec.observation_space.n
        n_a = self.env_spec.leader_action_space.n
        n_b = self.env_spec.action_space.n

        for _ in range(n_updates):
            q_old = self.q_table.copy()

            for s in range(n_s):
                for a in range(n_a):
                    for b in range(n_b):
                        reward = float(leader_reward_fn(s, a, b))

                        trans = transition_fn(s, a, b)
                        if isinstance(trans, dict):
                            next_dist = {int(sn): float(p) for sn, p in trans.items()}
                        else:
                            next_dist = {int(trans): 1.0}

                        v_next = 0.0
                        for s_next, p_trans in next_dist.items():
                            state_value = 0.0
                            for a_next in range(n_a):
                                pi_l = float(leader_policy_obj.policy_table[s_next, a_next])
                                pi_f = get_follower_probs_fn(s_next, a_next)
                                state_value += pi_l * float(np.dot(pi_f, q_old[s_next, a_next]))
                            v_next += p_trans * state_value

                        self.q_table[s, a, b] = reward + self.discount_leader * v_next
