"""Follower agent modules."""

from blackrl.agents.follower.feature_fns import (
    lqr_irl_feature_fn,
    make_lqr_irl_feature_fn,
)
from blackrl.agents.follower.follower_policy_model import FollowerPolicyModel
from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.follower.sf_learning import SuccessorFeatureLearning
from blackrl.agents.follower.soft_q_learning import SoftQLearning

__all__ = [
    "FollowerPolicyModel",
    "lqr_irl_feature_fn",
    "make_lqr_irl_feature_fn",
    "MDCEIRL",
    "SoftQLearning",
    "SuccessorFeatureLearning",
]
