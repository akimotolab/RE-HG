"""LQR environment (configurable MDP) for bilevel RL.

Dynamics: x_{t+1} = A x_t + B u_t + C a_t
  - x: state (observation)
  - u: follower action
  - a: leader action

Follower cost: x' Q x + u' R u  ->  follower_reward = -cost
Leader cost: configurable (default adds penalty for x[0] < 0).
"""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from blackrl.envs.base import (
    Environment,
    EnvStep,
    GlobalEnvSpec,
    StepType,
)


class LQRConfigurableMDPEnv(Environment):
    """LQR as a configurable MDP with leader (a) and follower (u).

    Parameters
    ----------
    A : array-like, shape (n_state, n_state)
        State transition matrix.
    B : array-like, shape (n_state, n_follower_action)
        Follower action matrix.
    C : array-like, shape (n_state, n_leader_action)
        Leader action matrix.
    Q : array-like, shape (n_state, n_state)
        State cost matrix (positive semi-definite).
    R : array-like, shape (n_follower_action, n_follower_action)
        Follower control cost (positive definite).
    max_episode_steps : int
        Episode length limit. Default 500.
    initial_state : array-like or None
        Fixed initial state; if None, sampled from observation_space.
    target_state : array-like or None
        Target state for the leader.
    leader_reward_weights : array-like or None
        Weighting matrix for the leader's reward/cost function.

    """

    def __init__(
        self,
        A,
        B,
        C,
        Q,
        R,
        max_episode_steps: int,
        initial_state,
        target_state,
        leader_reward_weights,
        env_seed: int,
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self._max_episode_steps = max_episode_steps
        self._initial_state = np.asarray(initial_state, dtype=np.float64)
        self._target_state = np.asarray(target_state, dtype=np.float64)
        self._leader_reward_weights = np.asarray(leader_reward_weights, dtype=np.float64)

        n_state = self.A.shape[0]
        n_follower = self.B.shape[1]
        n_leader = self.C.shape[1]

        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_state,),
            dtype=np.float32,
        )
        self._action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_follower,),
            dtype=np.float32,
        )
        self._leader_action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_leader,),
            dtype=np.float32,
        )

        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.state = None
        self.steps_n = 0
        self.np_random = None
        self.seed(env_seed)

    def seed(self, seed_val):
        self.np_random, seed_val = seeding.np_random(seed_val)
        return [seed_val]

    @property
    def action_space(self):
        return self._action_space

    @property
    def leader_action_space(self):
        return self._leader_action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spec(self) -> GlobalEnvSpec:
        return self._spec

    @property
    def render_modes(self):
        return ["human"]

    def next_state(self, x: np.ndarray, a: np.ndarray, u: np.ndarray) -> np.ndarray:
        """x_{t+1} = A x + B u + C a."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        return (self.A @ x + self.B @ u + self.C @ a).astype(np.float32)

    def follower_reward(self, x: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
        """Follower reward: -(x' Q x + u' R u)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        return float(-(x.T @ self.Q @ x + u.T @ self.R @ u))

    def leader_reward(self, x: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
        """Leader reward. Computes exp(- sum((x - target_state)^2 / 2 / xstd^2)).
        Assumes leader_reward_weights acts as 1/xstd^2 when target_state is given.
        """
        x_diff = (np.asarray(x) - self._target_state).reshape(-1, 1)
        weighted_sq_dist = float(x_diff.T @ self._leader_reward_weights @ x_diff)
        r = np.exp(-weighted_sq_dist / 2.0)

        return float(r)

    def reset(self):
        self.state = np.asarray(self._initial_state, dtype=np.float32).copy()
        self.steps_n = 0
        episode_info = {}
        return np.asarray(self.state, dtype=np.float32), episode_info

    def step(self, leader_action, action):
        a = np.asarray(leader_action, dtype=np.float64).flatten()
        u = np.asarray(action, dtype=np.float64).flatten()

        follower_reward_val = self.follower_reward(self.state, a, u)
        leader_reward_val = self.leader_reward(self.state, a, u)

        self.state = self.next_state(self.state, a, u)
        self.steps_n += 1

        if self.steps_n >= self._max_episode_steps:
            step_type = StepType.TIMEOUT
        else:
            step_type = StepType.MID

        env_info = {
            "leader_action": np.asarray(a, dtype=np.float32),
            "leader_reward": float(leader_reward_val),
            "follower_reward": float(follower_reward_val),
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.asarray(u, dtype=np.float32),
            reward=float(follower_reward_val),
            observation=np.asarray(self.state, dtype=np.float32),
            env_info=env_info,
            step_type=step_type,
        )

    def render(self, mode):
        if self.state is None:
            return None
        return f"LQR state={self.state} step={self.steps_n}"

    def close(self):
        pass
