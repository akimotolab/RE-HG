"""Discrete toy environments for bilevel RL."""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from blackrl.envs.base import (
    Environment,
    EnvStep,
    GlobalEnvSpec,
    StepType,
)


class DiscreteToyEnvBase(Environment):
    """Base class for discrete toy environments.

    This is a simple discrete environment with 3 states (S, A, B),
    2 leader actions (0, 1), and 3 follower actions (s, a, b).
    """

    STATE_TYPE = ["S0", "S1", "S2"]  # 0, 1, 2
    LEADER_ACT_TYPE = ["a0", "a1"]  # 0, 1
    FOLLOWER_ACT_TYPE = ["b0", "b1", "b2"]  # 0, 1, 2

    def __init__(self, random_start=False):
        """Initialize the discrete toy environment."""
        # Transition function: transition[state][leader_action][follower_action]
        # = 0: Sへ遷移, = 1: Aへ遷移, = 2: Bへ遷移, = -1: 遷移しない
        self.transition = np.ones((3, 2, 3), dtype=np.int32) * -1

        # Reward function: rewards[state][leader_action][follower_action]
        self.follower_rewards = np.zeros((3, 2, 3))
        self.follower_r_range = (-np.inf, np.inf)

        # Target reward function: leader_rewards[state][leader_action][follower_action]
        self.leader_rewards = np.zeros((3, 2, 3))
        self.leader_r_range = (-np.inf, np.inf)

        # Optimal leader policy configuration
        self.key_state = None
        self.optimal_action = 1

        # Leader action costs: costs[state][leader_action]
        self.costs = np.zeros((3, 2))

        # Episode length
        self._max_episode_steps = 150

        # Optimal follower action table: opt_follower_action_table[leader_action][state][follower_action]
        # Value 1 indicates optimal action, 0 indicates non-optimal
        self.opt_follower_action_table = np.zeros((2, 3, 3))

        # Optimal follower Q-table: opt_follower_q_table[leader_action][state][follower_action]
        # Contains actual Q-values (continuous values) for optimal policy
        self.opt_follower_q_table = np.zeros((2, 3, 3), dtype=np.float64)

        # Define spaces
        self._observation_space = spaces.Discrete(3)
        self._action_space = spaces.Discrete(3)  # Follower action space
        self._leader_action_space = spaces.Discrete(2)

        self.random_start = random_start

        # Create GlobalEnvSpec
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        # Initialize state
        self.seed()
        self.state = None
        self.steps_n = 0
        self.actions = [None, None]
        self.render_rewards = [0, 0]

    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):
        """Get follower's action space."""
        return self._action_space

    @property
    def leader_action_space(self):
        """Get leader's action space."""
        return self._leader_action_space

    @property
    def observation_space(self):
        """Get observation space."""
        return self._observation_space

    @property
    def spec(self):
        """Get global environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """Get supported render modes."""
        return ["human"]

    def transition_fn(self, state, leader_action, follower_action):
        """Compute next state from transition function.

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Next state

        """
        next_state = self.transition[state, leader_action, follower_action].item()
        if next_state == -1:
            return state
        return next_state

    def reward_fn(self, state, leader_action, follower_action):
        """Compute reward.

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Reward value

        """
        return self.follower_rewards[state, leader_action, follower_action].item()

    def leader_reward_fn(self, state, leader_action, follower_action):
        """Compute target reward.

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Target reward value

        """
        return self.leader_rewards[state, leader_action, follower_action].item()

    def reset(self, init_state=None):
        """Reset the environment."""
        if init_state is not None:
            assert self.observation_space.contains(init_state), f"Invalid state {init_state} passed to reset"
            self.state = int(init_state)
        elif self.random_start:
            self.state = self.observation_space.sample()
        else:
            self.state = 0

        self.actions = [None, None]
        self.render_rewards = [0, 0]
        self.steps_n = 0

        episode_info = {}
        return np.array(self.state, dtype=np.int32), episode_info

    def step(self, leader_action, action):
        """Step the environment.

        Args:
            leader_action: Leader's action
            action: Follower's action

        Returns:
            EnvStep: Environment step result

        """
        follower_action = action

        next_state = self.transition_fn(self.state, leader_action, follower_action)
        reward = self.reward_fn(self.state, leader_action, follower_action)
        leader_reward = self.leader_reward_fn(self.state, leader_action, follower_action)

        self.steps_n += 1
        self.state = next_state

        # Determine step type
        if self.steps_n >= self._max_episode_steps:
            step_type = StepType.TIMEOUT
        else:
            step_type = StepType.MID

        self.actions = [leader_action, follower_action]
        self.render_rewards = [leader_reward, reward]

        env_info = {
            "leader_action": leader_action,
            "leader_reward": leader_reward,
            "follower_reward": reward,
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.array(follower_action, dtype=np.int32),
            reward=reward,
            observation=np.array(self.state, dtype=np.int32),
            env_info=env_info,
            step_type=step_type,
        )

    def set_state(self, state):
        """Set the current state.

        Args:
            state: State to set

        """
        self.state = state

    def render(self, mode="human"):
        """Render the environment.

        Args:
            mode: Render mode (only 'human' supported)

        Returns:
            str: String representation of current state

        """
        if mode != "human":
            raise ValueError(f"Unsupported render mode: {mode}")

        if self.actions[0] is None or self.actions[1] is None:
            print(f"Step {self.steps_n}: state={self.STATE_TYPE[self.state]}")
        else:
            print(
                f"leader action={self.LEADER_ACT_TYPE[self.actions[0]]}, "
                f"follower action={self.FOLLOWER_ACT_TYPE[self.actions[1]]}, "
                f"reward={self.render_rewards[1]}, "
                f"leader_reward={self.render_rewards[0]}",
            )
            print(f"Step {self.steps_n}: state={self.STATE_TYPE[self.state]}")
        return self.STATE_TYPE[self.state]

    def close(self):
        """Close the environment."""

    def get_opt_follower_act_array(self):
        """Get optimal follower action array.

        Returns:
            Array of optimal actions

        """
        return self.opt_follower_action_table.argmax(axis=2)


# Environment variants (same as ptia)
class DiscreteToyEnvPaper(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(50, -50)
    leader action=1 -> best response return (follower, leader)=(50, 50)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transition[0, :, 0] = 0
        self.transition[0, :, 1] = 1
        self.transition[0, :, 2] = 2
        self.transition[1, 0, 0] = 0
        self.transition[1, 0, 1] = 1
        self.transition[1, 0, 2] = 2
        self.transition[1, 1, :] = 0
        self.transition[2, :, 0] = 0
        self.transition[2, :, 1] = 2
        self.transition[2, :, 2] = 2

        self.leader_rewards[0, :, 1] = 1
        self.leader_r_range = (0, 1)

        self.follower_rewards[0, :, 1] = 1
        self.follower_rewards[0, :, 2] = 1
        self.follower_rewards[1, 1, :] = -1
        self.follower_rewards[1, 0, 2] = 1
        self.follower_r_range = (-1, 1)
