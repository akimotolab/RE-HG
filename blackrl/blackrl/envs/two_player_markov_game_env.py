
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from blackrl.envs.base import (
    Environment,
    EnvStep,
    GlobalEnvSpec,
    StepType,
)


class TwoPlayerMarkovGameEnv(Environment):
    """Configurable two-player (leader–follower) Markov game.

    The leader outputs a **continuous** action of shape ``(n_leader_actions,)``
    representing an (unnormalized) probability distribution over discrete leader
    actions.  The environment normalizes this vector and samples a discrete
    leader action index internally (mixed-strategy commitment).

    Parameters
    ----------
    n_states : int
        Number of states.
    n_leader_actions : int
        Number of discrete leader actions (= dimension of leader's output vector).
    n_follower_actions : int
        Number of follower actions.
    transition : np.ndarray, shape (n_states, n_leader_actions, n_follower_actions)
        transition[s, a_l, a_f] = next state index (deterministic).
        Use -1 for "no transition" (stay in s).
    follower_rewards : np.ndarray, shape (n_states, n_leader_actions, n_follower_actions)
        follower_rewards[s, a_l, a_f] = follower reward.
    leader_rewards : np.ndarray, shape (n_states, n_leader_actions, n_follower_actions)
        leader_rewards[s, a_l, a_f] = leader reward.
    max_episode_steps : int
        Episode length limit.
    random_start : bool
        If True, initial state is uniform over states; else state 0.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_leader_actions: int = 2,
        n_follower_actions: int = 3,
        transition=None,
        follower_rewards=None,
        leader_rewards=None,
        max_episode_steps: int = 150,
        random_start: bool = False,
    ):
        self._n_states = n_states
        self._n_leader_actions = n_leader_actions
        self._n_follower_actions = n_follower_actions
        self._max_episode_steps = max_episode_steps
        self.random_start = random_start

        if transition is None:
            transition = np.full(
                (n_states, n_leader_actions, n_follower_actions), -1, dtype=np.int32
            )
        if follower_rewards is None:
            follower_rewards = np.zeros(
                (n_states, n_leader_actions, n_follower_actions), dtype=np.float64
            )
        if leader_rewards is None:
            leader_rewards = np.zeros(
                (n_states, n_leader_actions, n_follower_actions), dtype=np.float64
            )

        self.transition = np.asarray(transition, dtype=np.int32)
        self.follower_rewards = np.asarray(follower_rewards, dtype=np.float64)
        self.leader_rewards = np.asarray(leader_rewards, dtype=np.float64)

        assert self.transition.shape == (
            n_states,
            n_leader_actions,
            n_follower_actions,
        ), "transition shape mismatch"
        assert self.follower_rewards.shape == self.transition.shape
        assert self.leader_rewards.shape == self.transition.shape

        self._observation_space = spaces.Discrete(n_states)
        self._action_space = spaces.Discrete(n_follower_actions)
        # Leader outputs a continuous probability vector over discrete actions
        self._leader_action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_leader_actions,), dtype=np.float32
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
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def transition_fn(self, state: int, leader_action: int, follower_action: int) -> int:
        """Next state given *sampled* discrete leader action; -1 means stay."""
        s_next = self.transition[state, leader_action, follower_action].item()
        if s_next < 0:
            return state
        return s_next

    def follower_reward_fn(
        self, state: int, leader_action: int, follower_action: int
    ) -> float:
        return float(self.follower_rewards[state, leader_action, follower_action])

    def leader_reward_fn(
        self, state: int, leader_action: int, follower_action: int
    ) -> float:
        return float(self.leader_rewards[state, leader_action, follower_action])

    def reset(self, init_state=None):
        if init_state is not None:
            self.state = int(init_state)
        elif self.random_start:
            self.state = self.np_random.integers(0, self._n_states)
        else:
            self.state = 0
        self.steps_n = 0
        episode_info = {}
        return np.array(self.state, dtype=np.int32), episode_info

    def step(self, leader_action, action):
        """Step the environment.

        Parameters
        ----------
        leader_action : int
            Leader's discrete action index.
        action : int
            Follower's discrete action.
        """
        sampled_la = int(leader_action)
        follower_action = int(action)

        next_state = self.transition_fn(self.state, sampled_la, follower_action)
        reward = self.follower_reward_fn(self.state, sampled_la, follower_action)
        leader_reward = self.leader_reward_fn(
            self.state, sampled_la, follower_action
        )

        self.state = next_state
        self.steps_n += 1

        if self.steps_n >= self._max_episode_steps:
            step_type = StepType.TIMEOUT
        else:
            step_type = StepType.MID

        env_info = {
            "leader_action": sampled_la,
            "leader_reward": leader_reward,
            "follower_reward": reward,
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.array(follower_action, dtype=np.int32),
            reward=float(reward),
            observation=np.array(self.state, dtype=np.int32),
            env_info=env_info,
            step_type=step_type,
        )

    def set_state(self, state: int):
        """Set current state (e.g. for planning)."""
        self.state = int(state)

    def render(self, mode="human"):
        if self.state is None:
            return None
        return f"TwoPlayerMarkovGame state={self.state} step={self.steps_n}"

    def close(self):
        pass
