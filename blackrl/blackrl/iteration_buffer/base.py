"""Base replay buffer classes."""

import abc


class ReplayBufferBase(abc.ABC):
    """Abstract class for Replay Buffer.

    Replay buffer stores transitions in a memory buffer of fixed size.
    When the buffer is full, oldest memory will be discarded.
    """

    def __init__(self, size):
        """Initialize replay buffer.

        Args:
            size: Maximum size of the buffer

        """
        self._size = size

    @abc.abstractmethod
    def sample_transitions(self, batch_size, **kwargs):
        """Sample a batch of transitions.

        Args:
            batch_size: The number of transitions to be sampled.
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary of sampled transitions.

        """

    @abc.abstractmethod
    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer.

        Args:
            **kwargs: Dictionary that holds the transition data.

        """

    @abc.abstractmethod
    def add_transitions(self, **kwargs):
        """Add multiple transitions into the replay buffer.

        Args:
            **kwargs: Dictionary that holds the transitions.
                Each value should be a list of arrays.

        """

    @abc.abstractmethod
    def clear(self):
        """Clear the buffer."""

    @property
    def size(self):
        """Get the maximum buffer size."""
        return self._size
