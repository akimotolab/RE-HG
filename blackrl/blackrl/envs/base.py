"""Base Environment API for Bilevel RL."""

import abc
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class StepType:
    """Step type enumeration."""

    FIRST = 0
    MID = 1
    TERMINAL = 2
    TIMEOUT = 3


@dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""

    input_space: "akro.Space"  # type: ignore
    output_space: "akro.Space"  # type: ignore


@dataclass(frozen=True, init=False)
class EnvSpec(InOutSpec):
    """Describes the observations, actions, and time horizon of an MDP.

    Args:
        observation_space: The observation space of the env.
        action_space: The action space of the env.
        max_episode_length: The maximum number of steps allowed in an episode.

    """

    def __init__(
        self,
        observation_space,
        action_space,
        max_episode_length: int | None = None,
    ):
        object.__setattr__(self, "max_episode_length", max_episode_length)
        super().__init__(input_space=action_space, output_space=observation_space)

    max_episode_length: int | None = None

    @property
    def action_space(self):
        """Get action space."""
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space."""
        return self.output_space


@dataclass(frozen=True, init=False)
class GlobalEnvSpec:
    """Global environment specification for bilevel RL.

    This spec manages separate observation/action spaces for leader and follower,
    and provides utilities to construct inputs for each agent's policy/Q-function.
    """

    _observation_space: "akro.Space"  # type: ignore
    _action_space: "akro.Space"  # type: ignore
    _leader_action_space: "akro.Space"  # type: ignore
    _leader_policy_env_spec: EnvSpec
    _follower_policy_env_spec: EnvSpec
    _leader_qf_env_spec: EnvSpec
    _follower_qf_env_spec: EnvSpec
    _leader_policy_obs_info: list
    _follower_policy_obs_info: list
    _leader_qf_obs_info: list
    _follower_qf_obs_info: list

    def __init__(
        self,
        observation_space,
        action_space,
        leader_action_space,
        max_episode_length: int | None = None,
    ):
        object.__setattr__(self, "_observation_space", observation_space)
        object.__setattr__(self, "_action_space", action_space)
        object.__setattr__(self, "_leader_action_space", leader_action_space)
        object.__setattr__(self, "max_episode_length", max_episode_length)
        self.set_env_specs_for_agents()

    max_episode_length: int | None = None

    def set_env_specs_for_agents(self):
        """Set environment specifications for leader and follower agents."""
        # For policies
        l_policy_es = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.leader_action_space,
            max_episode_length=self.max_episode_length,
        )
        object.__setattr__(self, "_leader_policy_env_spec", l_policy_es)
        object.__setattr__(self, "_leader_policy_obs_info", ["observation"])

        # Follower policy observes: [observation, leader_action]
        # Note: This requires akro.concat, simplified here
        f_policy_es = EnvSpec(
            observation_space=self.observation_space,  # Simplified
            action_space=self.action_space,
            max_episode_length=self.max_episode_length,
        )
        object.__setattr__(self, "_follower_policy_env_spec", f_policy_es)
        object.__setattr__(self, "_follower_policy_obs_info", ["observation", "leader_action"])

        # For Q-functions
        l_qf_es = EnvSpec(
            observation_space=self.observation_space,  # Simplified
            action_space=self.leader_action_space,
            max_episode_length=self.max_episode_length,
        )
        object.__setattr__(self, "_leader_qf_env_spec", l_qf_es)
        object.__setattr__(self, "_leader_qf_obs_info", ["observation", "follower_action"])

        f_qf_es = EnvSpec(
            observation_space=self.observation_space,  # Simplified
            action_space=self.action_space,
            max_episode_length=self.max_episode_length,
        )
        object.__setattr__(self, "_follower_qf_env_spec", f_qf_es)
        object.__setattr__(self, "_follower_qf_obs_info", ["observation", "leader_action"])

    def get_inputs_for(
        self,
        agent: str,
        module: str,
        obs=None,
        leader_act=None,
        follower_act=None,
        obs_info=None,
    ):
        """Get inputs for agent's policy or Q-function.

        Args:
            agent: 'leader' or 'follower'
            module: 'policy' or 'qf'
            obs: Observations
            leader_act: Leader actions
            follower_act: Follower actions
            obs_info: Override observation info

        Returns:
            Concatenated input tensor

        """
        assert agent in ["leader", "follower"] and module in ["policy", "qf"]
        if obs_info is None:
            if module == "policy":
                obs_info = self.leader_policy_obs_info if agent == "leader" else self.follower_policy_obs_info
            elif module == "qf":
                obs_info = self.leader_qf_obs_info if agent == "leader" else self.follower_qf_obs_info

        inputs = {
            "observation": obs,
            "leader_action": leader_act,
            "follower_action": follower_act,
        }

        flatten_tensors = []
        for k in obs_info:
            if k not in inputs:
                raise KeyError(f"Key {k} not found in inputs")
            v = inputs[k]

            if v is None:
                continue

            if isinstance(v, torch.Tensor):
                v_tensor = v.float()
                v_tensor = v_tensor.view(v_tensor.shape[0], -1)
            elif isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v).float()
                v_tensor = v_tensor.view(v_tensor.shape[0], -1)
            else:
                raise ValueError(f"Unsupported type for {k}: {type(v)}")

            flatten_tensors.append(v_tensor)

        return torch.cat(flatten_tensors, dim=1) if flatten_tensors else None

    @property
    def action_space(self):
        """Get action space."""
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
    def leader_policy_env_spec(self):
        """Get leader's policy environment specification."""
        return self._leader_policy_env_spec

    @property
    def follower_policy_env_spec(self):
        """Get follower's policy environment specification."""
        return self._follower_policy_env_spec

    @property
    def leader_qf_env_spec(self):
        """Get leader's Q-function environment specification."""
        return self._leader_qf_env_spec

    @property
    def follower_qf_env_spec(self):
        """Get follower's Q-function environment specification."""
        return self._follower_qf_env_spec

    @property
    def leader_policy_obs_info(self):
        """Get leader's policy observation information."""
        return self._leader_policy_obs_info

    @property
    def follower_policy_obs_info(self):
        """Get follower's policy observation information."""
        return self._follower_policy_obs_info

    @property
    def leader_qf_obs_info(self):
        """Get leader's Q-function observation information."""
        return self._leader_qf_obs_info

    @property
    def follower_qf_obs_info(self):
        """Get follower's Q-function observation information."""
        return self._follower_qf_obs_info


@dataclass
class EnvStep:
    """A tuple representing a single step returned by the environment.

    Attributes:
        env_spec: Specification for the environment.
        action: Action for this time step.
        reward: Reward for taking the action.
        observation: Observation after applying the action.
        env_info: Environment state information.
        step_type: StepType enum value.

    """

    env_spec: EnvSpec
    action: np.ndarray
    reward: float
    observation: np.ndarray
    env_info: dict[str, np.ndarray | dict]
    step_type: StepType

    @property
    def first(self):
        """Whether this step is the first of a sequence."""
        return self.step_type == StepType.FIRST

    @property
    def mid(self):
        """Whether this step is in the middle of a sequence."""
        return self.step_type == StepType.MID

    @property
    def terminal(self):
        """Whether this step records a termination condition."""
        return self.step_type == StepType.TERMINAL

    @property
    def timeout(self):
        """Whether this step records a timeout condition."""
        return self.step_type == StepType.TIMEOUT

    @property
    def last(self):
        """Whether this step is the last of a sequence."""
        return self.step_type == StepType.TERMINAL or self.step_type == StepType.TIMEOUT


class Environment(abc.ABC):
    """The main API for bilevel RL environments.

    This environment supports two-agent (leader-follower) interactions.
    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """The action space specification (follower's action space)."""

    @property
    @abc.abstractmethod
    def leader_action_space(self):
        """The leader's action space specification."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """The observation space specification."""

    @property
    @abc.abstractmethod
    def spec(self) -> GlobalEnvSpec:
        """The global environment specification."""

    @abc.abstractmethod
    def reset(self, init_state=None):
        """Reset the environment.

        Returns:
            tuple: (observation, episode_info)

        """

    @abc.abstractmethod
    def step(self, leader_action, action):
        """Step the environment with leader and follower actions.

        Args:
            leader_action: Leader's action.
            action: Follower's action.

        Returns:
            EnvStep: The environment step resulting from the actions.

        """

    @abc.abstractmethod
    def render(self, mode="human"):
        """Render the environment."""

    @abc.abstractmethod
    def close(self):
        """Close the environment."""
