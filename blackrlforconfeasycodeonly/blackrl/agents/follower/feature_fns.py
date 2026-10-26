"""Feature functions for follower IRL (when follower reward is unknown).

For discrete MDPs use one-hot over (s, a, b). For LQR, use state squared and
follower action squared so that r = w·φ can represent quadratic cost x'Qx + u'Ru.
"""

from collections.abc import Callable

import numpy as np


def lqr_irl_feature_fn(
    state,
    leader_action,
    follower_action,
    *,
    include_leader_squared: bool = False,
) -> np.ndarray:
    """Feature for LQR IRL: 状態の2乗, 行動の2乗 (and optionally leader action squared).

    φ = [ state.ravel()**2, follower_action.ravel()**2 ]
    so that r = w·φ can express diagonal quadratic cost (e.g. x'Qx + u'Ru with diagonal Q,R).

    Parameters
    ----------
    state : array-like
        Observation (state) vector.
    leader_action : array-like
        Leader action vector.
    follower_action : array-like
        Follower action vector (the "行動" in 行動の2乗).
    include_leader_squared : bool
        If True, append leader_action.ravel()**2 to the feature (for cost terms in a).

    Returns
    -------
    np.ndarray, dtype float32, shape (n_state_dim + n_follower_dim [+ n_leader_dim],)
    """
    x = np.atleast_1d(np.asarray(state, dtype=np.float64)).ravel()
    a = np.atleast_1d(np.asarray(leader_action, dtype=np.float64)).ravel()
    u = np.atleast_1d(np.asarray(follower_action, dtype=np.float64)).ravel()

    phi = np.concatenate([np.square(x), np.square(u)])
    if include_leader_squared:
        phi = np.concatenate([phi, np.square(a)])
    return phi.astype(np.float32)


def make_lqr_irl_feature_fn(
    state_dim: int,
    follower_action_dim: int,
    leader_action_dim: int | None = None,
    *,
    include_leader_squared: bool = False,
) -> Callable:
    """Return a feature_fn for LQR IRL with fixed dimensions (状態の2乗, 行動の2乗).

    Convenient when you need a callable with a known feature dimension for MDCEIRL.

    Parameters
    ----------
    state_dim : int
        Length of state vector.
    follower_action_dim : int
        Length of follower action vector.
    leader_action_dim : int, optional
        Length of leader action vector (used only if include_leader_squared).
    include_leader_squared : bool
        If True, feature includes leader_action**2; leader_action_dim must be set.

    Returns
    -------
    feature_fn(s, a, b) -> np.ndarray
    """
    if include_leader_squared and leader_action_dim is None:
        raise ValueError("leader_action_dim required when include_leader_squared=True")

    dim = state_dim + follower_action_dim
    if include_leader_squared:
        dim += leader_action_dim

    def feature_fn(state, leader_action, follower_action):
        out = lqr_irl_feature_fn(
            state,
            leader_action,
            follower_action,
            include_leader_squared=include_leader_squared,
        )
        return out

    feature_fn.feature_dim = dim  # type: ignore[attr-defined]
    return feature_fn
