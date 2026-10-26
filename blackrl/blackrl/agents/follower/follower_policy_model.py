from collections import defaultdict

import numpy as np
import torch


class FollowerPolicyModel:
    def __init__(
        self,
        env_spec,
        temperature: float = 1.0,
    ):
        self.env_spec = env_spec
        self.temperature = temperature

        # Q-function: Q[state][leader_action][follower_action]
        self.Q: dict | None = None
        self._initialize_q_function()

    def _initialize_q_function(self):
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def set_q_values(self, q_values: dict):
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for (s, a, b), q_val in q_values.items():
            # Convert keys to match get_q_value's key format
            state_key = self._state_to_key(s)
            leader_key = self._action_to_key(a)
            follower_key = self._action_to_key(b)
            self.Q[state_key][leader_key][follower_key] = q_val

    def get_q_value(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
        follower_action: np.ndarray | float,
    ) -> float:
        # Convert to hashable key
        state_key = self._state_to_key(state)
        leader_key = self._action_to_key(leader_action)
        follower_key = self._action_to_key(follower_action)

        return self.Q[state_key][leader_key][follower_key]

    def compute_soft_value(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
    ) -> float:
        # Get Q-values for all follower actions
        num_follower_actions = self.env_spec.action_space.n
        q_values = []

        for b in range(num_follower_actions):
            q_val = self.get_q_value(state, leader_action, b)
            q_values.append(q_val / self.temperature)

        if len(q_values) == 0:
            return 0.0

        # Softmax: log Σ exp(q / temp)
        q_values_tensor = torch.tensor(q_values)
        soft_value = self.temperature * torch.logsumexp(q_values_tensor, dim=0).item()

        return soft_value

    def get_policy(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
    ) -> dict:
        # Compute soft value
        soft_value = self.compute_soft_value(state, leader_action)

        # Compute probabilities for all follower actions
        num_follower_actions = self.env_spec.action_space.n
        probs = {}

        for b in range(num_follower_actions):
            q_val = self.get_q_value(state, leader_action, b)
            log_prob = (q_val - soft_value) / self.temperature
            probs[b] = np.exp(log_prob)

        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            for key in probs:
                probs[key] /= total_prob
        else:
            # Fallback to uniform if underflow occurs
            for key in probs:
                probs[key] = 1.0 / num_follower_actions

        return probs

    def sample_action(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
    ) -> np.ndarray:
        policy = self.get_policy(state, leader_action)

        actions = list(policy.keys())
        probs = np.array(list(policy.values()), dtype=np.float64)

        sampled_action = int(np.random.choice(actions, p=probs))
        return np.array(sampled_action, dtype=np.int32)

    def _to_key(self, value) -> tuple:
        """Convert state or action to hashable key (always int tuple for discrete envs)."""
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return (int(value.item()),)
            return tuple(int(v) for v in value.flatten())
        return (int(value),)

    def _state_to_key(self, state) -> tuple:
        return self._to_key(state)

    def _action_to_key(self, action) -> tuple:
        return self._to_key(action)
