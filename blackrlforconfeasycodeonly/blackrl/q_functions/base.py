"""Base Q-function classes."""

import abc

import numpy as np
import torch


class QFunction(abc.ABC):
    """Base class for Q-functions.

    Q-functions estimate the value of taking an action in a given state.
    In bilevel RL, we have separate Q-functions for leader and follower.
    """

    def __init__(self, env_spec, name="QFunction"):
        """Initialize Q-function.

        Args:
            env_spec: Environment specification
            name: Name of the Q-function

        """
        self._env_spec = env_spec
        self._name = name

    @property
    def env_spec(self):
        """Get environment specification."""
        return self._env_spec

    @abc.abstractmethod
    def forward(self, observations, actions):
        """Compute Q-values.

        Args:
            observations: Batch of observations
            actions: Batch of actions

        Returns:
            Q-values for the given (observation, action) pairs

        """

    def __call__(self, observations, actions):
        """Call forward method."""
        return self.forward(observations, actions)


class DiscreteQFunction(QFunction):
    """Q-function for discrete action spaces.

    This can be implemented as a table (tabular) or neural network.
    """

    def __init__(self, env_spec, name="DiscreteQFunction"):
        """Initialize discrete Q-function.

        Args:
            env_spec: Environment specification with discrete action space
            name: Name of the Q-function

        """
        super().__init__(env_spec, name)
        self._action_space = env_spec.action_space

    @property
    def num_actions(self):
        """Get number of discrete actions."""
        if hasattr(self._action_space, "n"):
            return self._action_space.n
        raise ValueError("Action space does not have 'n' attribute")

    @abc.abstractmethod
    def forward(self, observations, actions=None):
        """Compute Q-values for discrete actions.

        Args:
            observations: Batch of observations
            actions: Optional batch of actions. If None, returns Q-values
                     for all actions.

        Returns:
            If actions is provided: Q-values for the given actions
            If actions is None: Q-values for all actions (batch_size, num_actions)

        """


class ContinuousQFunction(QFunction):
    """Q-function for continuous action spaces.

    This is typically implemented as a neural network.
    """

    def __init__(self, env_spec, name="ContinuousQFunction"):
        """Initialize continuous Q-function.

        Args:
            env_spec: Environment specification with continuous action space
            name: Name of the Q-function

        """
        super().__init__(env_spec, name)
        self._action_space = env_spec.action_space

    @abc.abstractmethod
    def forward(self, observations, actions):
        """Compute Q-values for continuous actions.

        Args:
            observations: Batch of observations
            actions: Batch of actions

        Returns:
            Q-values for the given (observation, action) pairs

        """


class TabularQFunction(DiscreteQFunction):
    """Tabular Q-function for discrete state and action spaces.

    This stores Q-values in a table: Q[state, action] = value

    For bilevel RL with discrete environments, this is the primary Q-function
    implementation. It supports both leader and follower Q-functions.
    """

    def __init__(self, env_spec, name="TabularQFunction", initial_value=0.0):
        """Initialize tabular Q-function.

        Args:
            env_spec: Environment specification (can be leader or follower spec)
            name: Name of the Q-function
            initial_value: Initial Q-value for all (state, action) pairs

        """
        super().__init__(env_spec, name)
        self._initial_value = initial_value

        # Initialize Q-table
        # For discrete state space
        if hasattr(env_spec.observation_space, "n"):
            num_states = env_spec.observation_space.n
        elif hasattr(env_spec.observation_space, "shape"):
            # If observation space has shape, assume it's one-hot encoded
            # The number of states is the size of the one-hot vector
            num_states = env_spec.observation_space.shape[0] if len(env_spec.observation_space.shape) > 0 else 1
        else:
            # Fallback: try to get from discrete space
            num_states = getattr(env_spec.observation_space, "n", 1000)

        num_actions = self.num_actions
        self._q_table = np.full((num_states, num_actions), initial_value, dtype=np.float32)

    def forward(self, observations, actions=None):
        """Compute Q-values from table.

        Args:
            observations: Batch of states (integers or one-hot encoded)
            actions: Optional batch of actions (integers)

        Returns:
            If actions is None: Q-values for all actions (batch_size, num_actions)
            If actions provided: Q-values for specific actions (batch_size, 1)

        """
        # Convert observations to state indices
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()

        # Handle different observation formats
        if len(observations.shape) > 1:
            if observations.shape[-1] > 1:
                # One-hot encoding: take argmax
                states = np.argmax(observations, axis=-1).astype(int)
            else:
                # Flatten and convert to int
                states = observations.flatten().astype(int)
        else:
            # Single dimension: direct state indices
            states = observations.astype(int)

        # Ensure states are within valid range
        states = np.clip(states, 0, self._q_table.shape[0] - 1)

        if actions is None:
            # Return Q-values for all actions
            q_values = self._q_table[states]  # (batch_size, num_actions)
            return torch.from_numpy(q_values).float()
        # Return Q-values for specific actions
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Handle action format (can be integers or one-hot)
        if len(actions.shape) > 1 and actions.shape[-1] > 1:
            # One-hot encoded actions
            actions = np.argmax(actions, axis=-1).astype(int)
        else:
            actions = actions.flatten().astype(int)

        # Ensure actions are within valid range
        actions = np.clip(actions, 0, self._q_table.shape[1] - 1)

        q_values = self._q_table[states, actions]  # (batch_size,)
        return torch.from_numpy(q_values).float().unsqueeze(-1)

    def update(self, states, actions, values):
        """Update Q-table values.

        Args:
            states: State indices (can be integers or one-hot)
            actions: Action indices (can be integers or one-hot)
            values: New Q-values (can be scalar, array, or tensor)

        """
        # Convert to numpy arrays
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()

        # Convert states to indices
        if len(states.shape) > 1 and states.shape[-1] > 1:
            states = np.argmax(states, axis=-1).astype(int)
        else:
            states = states.flatten().astype(int)

        # Convert actions to indices
        if len(actions.shape) > 1 and actions.shape[-1] > 1:
            actions = np.argmax(actions, axis=-1).astype(int)
        else:
            actions = actions.flatten().astype(int)

        values = values.flatten()

        # Ensure valid indices
        states = np.clip(states, 0, self._q_table.shape[0] - 1)
        actions = np.clip(actions, 0, self._q_table.shape[1] - 1)

        # Update Q-table
        self._q_table[states, actions] = values

    def get_q_table(self):
        """Get a copy of the Q-table.

        Returns:
            Copy of the Q-table array

        """
        return self._q_table.copy()

    def get_value(self, state, action=None):
        """Get Q-value for a single state-action pair.

        Args:
            state: State (integer or one-hot)
            action: Optional action (integer or one-hot). If None, returns all Q-values.

        Returns:
            Q-value(s) as numpy array

        """
        # Convert state to index
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        if isinstance(state, np.ndarray) and len(state.shape) > 0 and state.shape[-1] > 1:
            state_idx = np.argmax(state).item()
        else:
            state_idx = int(state)

        state_idx = np.clip(state_idx, 0, self._q_table.shape[0] - 1)

        if action is None:
            return self._q_table[state_idx].copy()
        # Convert action to index
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        if isinstance(action, np.ndarray) and len(action.shape) > 0 and action.shape[-1] > 1:
            action_idx = np.argmax(action).item()
        else:
            action_idx = int(action)

        action_idx = np.clip(action_idx, 0, self._q_table.shape[1] - 1)
        return self._q_table[state_idx, action_idx].item()

    def set_value(self, state, action, value):
        """Set Q-value for a single state-action pair.

        Args:
            state: State (integer or one-hot)
            action: Action (integer or one-hot)
            value: Q-value to set

        """
        # Convert state to index
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        if isinstance(state, np.ndarray) and len(state.shape) > 0 and state.shape[-1] > 1:
            state_idx = np.argmax(state).item()
        else:
            state_idx = int(state)

        # Convert action to index
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        if isinstance(action, np.ndarray) and len(action.shape) > 0 and action.shape[-1] > 1:
            action_idx = np.argmax(action).item()
        else:
            action_idx = int(action)

        # Ensure valid indices
        state_idx = np.clip(state_idx, 0, self._q_table.shape[0] - 1)
        action_idx = np.clip(action_idx, 0, self._q_table.shape[1] - 1)

        self._q_table[state_idx, action_idx] = float(value)
