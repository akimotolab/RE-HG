"""Joint Policy for Leader and Follower."""

import numpy as np


class JointPolicy:
    """Joint Policy combining leader and follower policies.

    This policy manages the interaction between leader and follower agents
    in a bilevel RL setting.

    Args:
        env_spec: Global environment specification
        leader_policy: Leader's policy f_θ_L(a|s)
        follower_policy: Follower's policy g(b|s, a)

    """

    def __init__(
        self,
        env_spec,
        leader_policy,
        follower_policy,
    ):
        self.env_spec = env_spec
        self.leader_policy = leader_policy
        self.follower_policy = follower_policy

    def get_action(
        self,
        observation: np.ndarray,
        deterministic_leader: bool = False,
        deterministic_follower: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get joint action (leader_action, follower_action).

        Args:
            observation: Current observation
            deterministic_leader: Whether to use deterministic leader policy
            deterministic_follower: Whether to use deterministic follower policy

        Returns:
            Tuple of (leader_action, follower_action)

        """
        actions, _ = self.get_actions(
            [observation],
            deterministic_leader=deterministic_leader,
            deterministic_follower=deterministic_follower,
        )
        return actions[0]

    def get_actions(
        self,
        observations: list[np.ndarray],
        deterministic_leader: bool = False,
        deterministic_follower: bool = False,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
        """Get joint actions for a batch of observations.

        Args:
            observations: List of observations
            deterministic_leader: Whether to use deterministic leader policy
            deterministic_follower: Whether to use deterministic follower policy

        Returns:
            Tuple of (list of (leader_action, follower_action), agent_info dict)

        """
        leader_actions = []
        follower_actions = []
        agent_infos = {}

        for obs in observations:
            # Get leader action
            leader_act = self.leader_policy(obs, deterministic=deterministic_leader)
            leader_actions.append(leader_act)

            # Convert to numpy arrays for get_inputs_for
            obs_array = np.array([obs]) if not isinstance(obs, np.ndarray) else np.array([obs])
            leader_act_array = np.array([leader_act]) if not isinstance(leader_act, np.ndarray) else np.array([leader_act])

            # Construct follower observation: [observation, leader_action]
            follower_obs = self.env_spec.get_inputs_for(
                "follower",
                "policy",
                obs=obs_array,
                leader_act=leader_act_array,
            )

            # Get follower action
            # follower_obs contains [observation, leader_action] concatenated
            # But follower_policy expects (obs, leader_act) separately
            # So we pass obs and leader_act separately
            follower_act = self.follower_policy(
                obs,
                leader_act,
                deterministic=deterministic_follower,
            )
            follower_actions.append(follower_act)

        return (
            list(zip(leader_actions, follower_actions, strict=False)),
            agent_infos,
        )

    def reset(self, do_resets: list[bool] | None = None):
        """Reset the policy.

        Args:
            do_resets: Optional list of reset flags

        """
        if hasattr(self.leader_policy, "reset"):
            self.leader_policy.reset(do_resets)
        if hasattr(self.follower_policy, "reset"):
            self.follower_policy.reset(do_resets)
