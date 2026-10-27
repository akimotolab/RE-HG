from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch


class SoftQLearning:
    def __init__(
        self,
        env_spec,
        reward_fn: Callable | None = None,
        leader_policy: Callable | None = None,
        discount: float = 0.99,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ):
        self.env_spec = env_spec
        self.reward_fn = reward_fn  # Optional: only used if computing rewards internally
        self.leader_policy = leader_policy
        self.discount = discount
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.epsilon = epsilon

        # Q-function network (can be replaced with tabular or neural network)
        self.Q: dict | None = None
        self._initialize_q_function()

    def _initialize_q_function(self):
        """Initialize Q-function."""
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def compute_soft_value(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
    ) -> float:
        # Get Q-values for all follower actions
        q_values = []
        for b in self._get_follower_actions():
            q_val = self.get_q_value(state, leader_action, b)
            q_values.append(q_val / self.temperature)

        if len(q_values) == 0:
            return 0.0

        # Softmax: log Σ exp(q / temp)
        q_values_tensor = torch.tensor(q_values)
        soft_value = self.temperature * torch.logsumexp(q_values_tensor, dim=0).item()

        return soft_value

    def get_q_value(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
        follower_action: np.ndarray | float,
    ) -> float:
        # Convert to hashable key (simplified)
        state_key = self._state_to_key(state)
        leader_key = self._action_to_key(leader_action)
        follower_key = self._action_to_key(follower_action)

        return self.Q[state_key][leader_key][follower_key]

    def set_q_value(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
        follower_action: np.ndarray | float,
        value: float,
    ):
        state_key = self._state_to_key(state)
        leader_key = self._action_to_key(leader_action)
        follower_key = self._action_to_key(follower_action)

        self.Q[state_key][leader_key][follower_key] = value

    def _to_key(self, value: np.ndarray | float) -> tuple:
        """Convert value (state or action) to hashable key."""
        if isinstance(value, np.ndarray):
            flat = value.flatten()
            if flat.size == 1:
                val = flat.item()
                return (int(val),) if isinstance(val, (float, np.floating)) else (val,)
            return tuple(flat)

        if isinstance(value, (float, np.floating)):
            return (int(value),)

        if isinstance(value, (int, np.integer)):
            return (int(value),)

        return tuple(value) if hasattr(value, "__iter__") else (value,)

    def _state_to_key(self, state: np.ndarray | float) -> tuple:
        """Convert state to hashable key."""
        return self._to_key(state)

    def _action_to_key(self, action: np.ndarray | float) -> tuple:
        """Convert action to hashable key."""
        return self._to_key(action)

    def _get_follower_actions(self):
        action_space = self.env_spec.follower_policy_env_spec.action_space
        return list(range(action_space.n))

    def update(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
        follower_action: np.ndarray | float,
        reward: float,
        next_state: np.ndarray | float,
        done: bool,
        learning_rate: float | None = None,
    ):
        lr = learning_rate or self.learning_rate

        # Current Q-value
        current_q = self.get_q_value(state, leader_action, follower_action)

        if done:
            target_q = reward
        else:
            expected_soft_value = 0.0
            leader_action_probs = self.leader_policy(next_state)

            num_leader_actions = self.env_spec.leader_policy_env_spec.action_space.n

            for next_leader_act in range(num_leader_actions):
                if next_leader_act < len(leader_action_probs):
                    # Compute V_F^soft(s', a')
                    soft_value = self.compute_soft_value(next_state, next_leader_act)

                    # Add to the expectation: p(a'|s') * V(s', a')
                    expected_soft_value += leader_action_probs[next_leader_act] * soft_value

            target_q = reward + self.discount * expected_soft_value

        # Update Q-value
        new_q = current_q + lr * (target_q - current_q)
        self.set_q_value(state, leader_action, follower_action, new_q)

    def get_policy(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
    ) -> dict:
        """Get optimal Max-Ent policy g^*(b|s, a).

        g^*(b|s, a) = exp((Q_F^soft(s, a, b) - V_F^soft(s, a)) / temperature)

        Args:
            state: State s
            leader_action: Leader action a

        Returns:
            Dictionary mapping follower actions to probabilities

        """
        # Compute soft value
        soft_value = self.compute_soft_value(state, leader_action)

        # Compute probabilities for all follower actions
        follower_actions = self._get_follower_actions()
        probs = {}

        for b in follower_actions:
            q_val = self.get_q_value(state, leader_action, b)
            log_prob = (q_val - soft_value) / self.temperature
            # Normalize keys to int (for discrete actions)
            action_key = int(b) if isinstance(b, (int, np.integer)) else b
            probs[action_key] = np.exp(log_prob)

        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            for key in probs:
                probs[key] /= total_prob

        return probs

    def sample_action(
        self,
        state: np.ndarray | float,
        leader_action: np.ndarray | float,
    ) -> np.ndarray:
        # ε-greedy: random with probability ε, softmax policy with probability 1-ε
        if self.epsilon > 0.0 and np.random.random() < self.epsilon:
            n_actions = self.env_spec.action_space.n
            sampled_action = np.random.randint(n_actions)
            return np.array(sampled_action, dtype=np.int32)

        policy = self.get_policy(state, leader_action)
        actions = list(policy.keys())
        probs = list(policy.values())

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(actions), p=probs)
        sampled_action = actions[sampled_idx]

        # Always return a numpy array (consistent type)
        if isinstance(sampled_action, tuple):
            return np.array(sampled_action, dtype=np.int32)
        return np.array(sampled_action, dtype=np.int32)
