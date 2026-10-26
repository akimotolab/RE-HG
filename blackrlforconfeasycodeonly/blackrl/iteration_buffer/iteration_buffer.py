"""Iteration rollout buffer for bilevel RL."""

import numpy as np

from blackrl.iteration_buffer.base import ReplayBufferBase


class IterationRolloutBuffer(ReplayBufferBase):
    """Iteration-scoped rollout buffer with optional gamma-weighted sampling.

    This replay buffer supports discount-weighted sampling, which is useful
    for MDCE IRL and other algorithms that need to sample transitions
    according to their time step in the episode.

    Args:
        size: Maximum number of transitions to store
        gamma: Discount factor for weighted sampling (default: 1.0)

    """

    def __init__(self, size, gamma=1.0):
        """Initialize iteration rollout buffer.

        Args:
            size: Maximum size of transitions in the buffer
            gamma: Discount factor for weighted sampling

        """
        super().__init__(size)
        self._current_size = 0
        self._current_ptr = 0
        self._n_transitions_stored = 0
        self._initialized_buffer = False
        self._buffer = {}
        self.gamma = gamma

    def sample_transitions(
        self,
        batch_size,
        replace=True,
        discount=False,
        with_subsequence=False,
    ):
        """Sample a batch of transitions.

        Args:
            batch_size: The number of transitions to be sampled
            replace: Whether to sample with replacement
            discount: Whether to use gamma-weighted sampling probabilities
            with_subsequence: Whether to include subsequence trajectories

        Returns:
            Dictionary of sampled transitions

        """
        if self._current_size == 0:
            raise ValueError("Buffer is empty. Cannot sample transitions.")

        if discount and "time_step" not in self._buffer.keys():
            raise ValueError("time_step is not stored in the replay buffer.")

        if discount:
            # Gamma-weighted sampling
            probabilities = self.gamma ** self._buffer["time_step"][: self._current_size]
            probabilities /= probabilities.sum()
            indices = np.random.choice(
                self._current_size,
                batch_size,
                replace=replace,
                p=probabilities,
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                self._current_size,
                batch_size,
                replace=replace,
            )

        sampled_transitions = {key: self._buffer[key][indices] for key in self._buffer.keys()}

        if with_subsequence:
            subseqs = {key: [] for key in self._buffer.keys()}
            for i in indices:
                j = i
                while not self._buffer["last"][j] and j != self._current_ptr:
                    j = (j + 1) % self._size
                if j < i:
                    idx_a = np.arange(i, self._size)
                    idx_b = np.arange(0, j + 1)
                    idx = np.concatenate([idx_a, idx_b])
                else:
                    idx = np.arange(i, j + 1)
                for key in self._buffer.keys():
                    subseqs[key].append(self._buffer[key][idx])
            sampled_transitions["subsequence"] = subseqs

        return sampled_transitions

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer.

        Args:
            **kwargs: Dictionary that holds the transition data.
                Each value should be a single array or scalar.

        """
        transition = {k: [v] for k, v in kwargs.items()}
        self.add_transitions(**transition)

    def add_transitions(self, **kwargs):
        """Add multiple transitions into the replay buffer.

        Args:
            **kwargs: Dictionary that holds the transitions.
                Each value should be a list of arrays.

        """
        if not self._initialized_buffer:
            self._initialize_buffer(**kwargs)

        assert self._buffer.keys() == kwargs.keys(), "Keys of the buffer and transitions do not match."

        num_transitions = len(kwargs["observation"])
        idx = self._get_storage_idx(num_transitions)

        for key, value in kwargs.items():
            self._buffer[key][idx] = np.asarray(value)

        self._n_transitions_stored = min(
            self._size,
            self._n_transitions_stored + num_transitions,
        )

    def _initialize_buffer(self, **kwargs):
        """Initialize the buffer with the first transition.

        Args:
            **kwargs: Dictionary that holds the first transition(s)

        """
        for key, value in kwargs.items():
            values = np.array(value)
            self._buffer[key] = np.zeros(
                [self._size, *values.shape[1:]],
                dtype=values.dtype,
            )
        self._initialized_buffer = True

    def _get_storage_idx(self, size_increment=1):
        """Get the storage index for new transitions.

        Args:
            size_increment: The number of storage indices needed

        Returns:
            numpy.ndarray: The indices to store transitions at

        """
        # Ensure current_ptr is within bounds (circular buffer)
        self._current_ptr = self._current_ptr % self._size

        if self._current_ptr + size_increment <= self._size:
            idx = np.arange(self._current_ptr, self._current_ptr + size_increment)
            self._current_ptr = (self._current_ptr + size_increment) % self._size
        else:
            # Wrap around: need to split across buffer boundary
            remaining = self._size - self._current_ptr
            overflow = size_increment - remaining
            idx_a = np.arange(self._current_ptr, self._size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self._current_ptr = overflow % self._size

        # Update replay size
        self._current_size = min(self._size, self._current_size + size_increment)

        if size_increment == 1:
            idx = idx[0]
        return idx

    def clear(self):
        self._buffer.clear()
        self._current_size = 0
        self._current_ptr = 0
        self._n_transitions_stored = 0
        self._initialized_buffer = False

    @property
    def full(self):
        return self._current_size == self._size

    @property
    def n_transitions_stored(self):
        return self._n_transitions_stored

    @property
    def current_size(self):
        return self._current_size

