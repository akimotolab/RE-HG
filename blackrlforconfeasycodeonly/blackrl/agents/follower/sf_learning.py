import numpy as np


class SuccessorFeatureLearning:
    """Successor Feature (SF) Learning for Follower."""

    def __init__(
        self,
        env_spec,
        feature_dim: int,
        feature_fn,
        discount: float = 0.99,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        self.n_s = env_spec.observation_space.n if hasattr(env_spec.observation_space, "n") else 3
        self.n_la = env_spec.leader_action_space.n if hasattr(env_spec.leader_action_space, "n") else 2
        self.n_fa = env_spec.action_space.n if hasattr(env_spec.action_space, "n") else 3

        self.feature_dim = feature_dim
        self.feature_fn = feature_fn
        self.discount = discount
        self.learning_rate = learning_rate
        self.temperature = temperature

        # SFテーブル: ψ(s, leader_a, follower_b) -> vector
        self.sf_table = np.zeros((self.n_s, self.n_la, self.n_fa, feature_dim), dtype=np.float32)
