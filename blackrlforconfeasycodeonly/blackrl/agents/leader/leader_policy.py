"""Leader policy implementation for bilevel RL."""

from collections.abc import Callable

import numpy as np


class LeaderPolicy:
    """Leader policy that can be either tabular or callable.

    This class manages the leader's policy π_L(a|s), which can be:
    - Tabular policy: π_L[state, action] (probability table)
    - Callable policy: function(state) -> action or probabilities

    Args:
        env_spec: Environment specification
        policy: Callable policy function (optional)

    """

    def __init__(
        self,
        env_spec,
        policy: Callable | None = None,
    ):
        self.env_spec = env_spec
        self.policy = policy

        # Initialize tabular policy (always used in this codebase)
        self._initialize_tabular_policy()

    def _initialize_tabular_policy(self):
        """Initialize tabular policy with uniform distribution."""
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2

        # Initialize with uniform distribution
        self.policy_table = np.ones((num_states, num_leader_actions), dtype=np.float32) / num_leader_actions

    def sample_action(self, state: np.ndarray) -> int:
        state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
        probs = self.policy_table[state_int]
        return int(np.random.choice(len(probs), p=probs))

    def get_probability(self, state: np.ndarray, action: int) -> float:
        state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
        return float(self.policy_table[state_int, action])

    def get_probabilities(self, state: np.ndarray) -> list[float]:
        """Get probability distribution for all actions in given state.

        Args:
            state: State s

        Returns:
            List of probabilities [π_L(0|s), π_L(1|s), ...]

        """
        state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
        return self.policy_table[state_int].tolist()

    def update(self, gradients: np.ndarray, learning_rate: float):
        # Update policy: π_L^{n+1} ← π_L^n + α_L * ∇_{π_L} J_L
        self.policy_table = self.policy_table + learning_rate * gradients

        # Normalize to ensure valid probability distribution
        # Add small epsilon to avoid numerical issues
        self.policy_table = np.maximum(self.policy_table, 1e-8)

        # Normalize per state
        for state in range(self.policy_table.shape[0]):
            state_sum = np.sum(self.policy_table[state])
            if state_sum > 0:
                self.policy_table[state] = self.policy_table[state] / state_sum
            else:
                # If all probabilities are zero or negative, reset to uniform
                num_actions = self.policy_table.shape[1]
                self.policy_table[state] = np.ones(num_actions) / num_actions
