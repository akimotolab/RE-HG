"""Pooled multi-environment IRL baseline, in the formulation of Cao, Cohen and
Szpruch (2021), "Identifiability in inverse reinforcement learning", Section 7.2.

Why this file exists
--------------------
Reviewer Xcsj asks for a joint multi-environment IRL baseline that uses exactly
the datasets D_1, ..., D_k the leader has already accumulated, so that the value
of the projection-and-truncation aggregation (Algorithm 2) can be separated from
the value of merely having demonstrations from several environments.

The continuous experiment already has such a baseline, but it matches feature
expectations, which requires re-solving the follower's forward problem for every
environment at every optimizer step. Cao et al.'s Section 7.2 loss avoids that
entirely by treating the per-environment soft value functions as free variables:

    L(R, v_1, v_2) = 1/2 sum_a sum_s [ a' Pi_1 s - exp{a' R s + g_1 s' P_a v_1 - v_1' s} ]^2
                   + 1/2 sum_a sum_s [ a' Pi_2 s - exp{a' R s + g_2 s' P_a v_2 - v_2' s} ]^2

Their setting varies the discount factor; ours varies the transition kernel,
which the same identifiability argument covers ("or under sufficiently different
environments"). This module is the direct transcription to our two-player MG.

Setting and notation
--------------------
From the follower's perspective the state is the pair (s, a) of environment
state and leader action, and the transition is

    P_theta((s', a') | (s, a, b)) = p(s' | s, a, b) * f_theta(a' | s'),

so the leader parameter enters only through f_theta. Each outer iteration k
therefore contributes one environment, with the SAME reward and a DIFFERENT
kernel -- exactly the situation Cao et al. require.

With temperature tau the entropy-regularized optimality condition is

    pi(b | s, a) = exp( [ r(s,a,b) + gamma (P_theta v)(s,a,b) - v(s,a) ] / tau ),

and the residual of that identity is what we minimize, jointly over the shared
reward omega and the per-environment v_k. No forward solve appears anywhere.

Indexing (must match ``feature_fn`` in scripts/runrun.py)
--------------------------------------------------------
    flat (s, a, b) -> s * (n_a * n_b) + a * n_b + b   [equivalently (s*n_a+a)*n_b+b]
    flat (s, a)    -> s * n_a + a

Identifiability note
--------------------
Adding a constant c to every reward and c/(1-gamma) to every v_k leaves each
z(s,a,b) unchanged, because gamma*(P(v + c/(1-gamma))) - (v + c/(1-gamma))
= gamma*Pv - v - c. The constant offset is therefore exactly the direction this
objective cannot see -- the same limit as Theorem 1, reached by a different
route. Comparisons must be made after removing the mean.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_transition_tensor",
    "empirical_follower_policy",
    "solve_joint_cao",
    "center",
]

# exp() guard. At the solution z/tau <= 0 because the left-hand side is a
# probability, so an upper clip at +30 can only ever bite on a wild iterate,
# where it prevents an overflow warning turning into inf/nan.
_Z_CLIP = 30.0


def center(omega: np.ndarray) -> np.ndarray:
    """Remove the unidentifiable constant offset, for comparing reward vectors."""
    omega = np.asarray(omega, dtype=np.float64)
    return omega - float(np.mean(omega))


def build_transition_tensor(
    transition_fn,
    leader_probs: np.ndarray,
    n_s: int,
    n_a: int,
    n_b: int,
) -> np.ndarray:
    """T[(s,a,b), (s',a')] = p(s'|s,a,b) * f_theta(a'|s').

    ``transition_fn(s, a, b)`` may return either a dict {s_next: prob} or a
    single next state, matching the convention used elsewhere in the codebase.
    ``leader_probs`` is the (n_s, n_a) policy table at this outer iteration.
    """
    leader_probs = np.asarray(leader_probs, dtype=np.float64)
    if leader_probs.shape != (n_s, n_a):
        raise ValueError(
            f"leader_probs must have shape {(n_s, n_a)}, got {leader_probs.shape}"
        )

    T = np.zeros((n_s * n_a * n_b, n_s * n_a), dtype=np.float64)
    for s in range(n_s):
        for a in range(n_a):
            for b in range(n_b):
                row = (s * n_a + a) * n_b + b
                result = transition_fn(s, a, b)
                if isinstance(result, dict):
                    items = [(int(sn), float(p)) for sn, p in result.items()]
                else:
                    items = [(int(result), 1.0)]
                for s_next, p_trans in items:
                    if p_trans == 0.0:
                        continue
                    T[row, s_next * n_a: s_next * n_a + n_a] += (
                        p_trans * leader_probs[s_next]
                    )
    return T


def empirical_follower_policy(
    trajectories: list[dict],
    n_s: int,
    n_a: int,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical follower policy from expert trajectories.

    Returns (pi_hat, counts) with shapes (n_s*n_a, n_b) and (n_s*n_a,). Rows for
    unvisited (s, a) pairs are left at zero and are excluded from the objective
    via the count mask: the pooled solver must be given exactly the data the
    leader actually collected, not the analytic policy, otherwise it would be
    compared against RE-HG's local IRL on strictly better information.
    """
    counts_sab = np.zeros((n_s * n_a, n_b), dtype=np.float64)
    for traj in trajectories:
        obs = traj["observations"]
        las = traj["leader_actions"]
        fas = traj["follower_actions"]
        for s_raw, a_raw, b_raw in zip(obs, las, fas):
            s = int(np.asarray(s_raw).item())
            a = int(np.asarray(a_raw).item())
            b = int(np.asarray(b_raw).item())
            counts_sab[s * n_a + a, b] += 1.0

    counts = counts_sab.sum(axis=1)
    pi_hat = np.zeros_like(counts_sab)
    visited = counts > 0
    pi_hat[visited] = counts_sab[visited] / counts[visited, None]
    return pi_hat, counts


def _loss_and_grad(
    omega: np.ndarray,
    vs: list[np.ndarray],
    env_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    gamma: float,
    tau: float,
    n_b: int,
    ridge: float,
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """Objective value and analytic gradients w.r.t. omega and every v_k."""
    grad_omega = np.zeros_like(omega)
    grad_vs = [np.zeros_like(v) for v in vs]
    total = 0.0

    for k, (pi_flat, row_mask, T) in enumerate(env_data):
        v = vs[k]
        # z = r_omega + gamma * (T v) - v repeated over b
        z = omega + gamma * (T @ v) - np.repeat(v, n_b)
        E = np.exp(np.clip(z / tau, -np.inf, _Z_CLIP))
        res = (pi_flat - E) * row_mask          # zero on unvisited (s, a)
        total += 0.5 * float(res @ res)

        # dL/dz = -res * E / tau   (already masked through res)
        dz = -(res * E) / tau
        grad_omega += dz
        # d/dv: gamma * T^T dz  from the look-ahead term, minus the per-(s,a) sum
        grad_vs[k] = gamma * (T.T @ dz) - dz.reshape(-1, n_b).sum(axis=1)

    if ridge > 0.0:
        total += 0.5 * ridge * float(omega @ omega)
        grad_omega = grad_omega + ridge * omega

    return total, grad_omega, grad_vs


def solve_joint_cao(
    env_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_s: int,
    n_a: int,
    n_b: int,
    gamma: float,
    tau: float = 1.0,
    n_iters: int = 2000,
    lr: float = 0.005,
    ridge: float = 0.0,
    warm_omega: np.ndarray | None = None,
    warm_vs: list[np.ndarray] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Minimize the pooled soft-Bellman residual over (omega, {v_k}).

    ``env_data`` is a list of (pi_flat, row_mask, T) triples, one per accumulated
    outer iteration:
        pi_flat   (n_s*n_a*n_b,)  empirical follower policy, flattened
        row_mask  (n_s*n_a*n_b,)  1.0 on rows whose (s, a) pair was visited
        T         (n_s*n_a*n_b, n_s*n_a)

    Adam with the settings reported in Cao et al. Section 7.2
    (alpha = 0.005, betas = (0.5, 0.9), 2000 steps) by default.

    Returns (omega, vs, final_loss).
    """
    if not env_data:
        raise ValueError("solve_joint_cao requires at least one environment")

    dim = n_s * n_a * n_b
    n_sa = n_s * n_a
    for idx, (pi_flat, row_mask, T) in enumerate(env_data):
        if pi_flat.shape != (dim,):
            raise ValueError(f"env {idx}: pi_flat shape {pi_flat.shape} != {(dim,)}")
        if row_mask.shape != (dim,):
            raise ValueError(f"env {idx}: row_mask shape {row_mask.shape} != {(dim,)}")
        if T.shape != (dim, n_sa):
            raise ValueError(f"env {idx}: T shape {T.shape} != {(dim, n_sa)}")

    omega = (np.zeros(dim, dtype=np.float64) if warm_omega is None
             else np.asarray(warm_omega, dtype=np.float64).copy())
    if warm_vs is not None and len(warm_vs) >= len(env_data):
        vs = [np.asarray(warm_vs[k], dtype=np.float64).copy() for k in range(len(env_data))]
    else:
        vs = [np.zeros(n_sa, dtype=np.float64) for _ in env_data]
        if warm_vs is not None:                       # reuse what we have
            for k in range(min(len(warm_vs), len(vs))):
                vs[k] = np.asarray(warm_vs[k], dtype=np.float64).copy()

    # --- Adam state (one moment pair for omega, one per v_k) ---
    b1, b2, eps = 0.5, 0.9, 1e-8
    m_o = np.zeros_like(omega); s_o = np.zeros_like(omega)
    m_v = [np.zeros_like(v) for v in vs]
    s_v = [np.zeros_like(v) for v in vs]

    loss = float("nan")
    for t in range(1, int(n_iters) + 1):
        loss, g_o, g_v = _loss_and_grad(
            omega, vs, env_data, gamma, tau, n_b, ridge,
        )

        m_o = b1 * m_o + (1.0 - b1) * g_o
        s_o = b2 * s_o + (1.0 - b2) * (g_o * g_o)
        omega -= lr * (m_o / (1.0 - b1 ** t)) / (np.sqrt(s_o / (1.0 - b2 ** t)) + eps)

        for k in range(len(vs)):
            m_v[k] = b1 * m_v[k] + (1.0 - b1) * g_v[k]
            s_v[k] = b2 * s_v[k] + (1.0 - b2) * (g_v[k] * g_v[k])
            vs[k] -= lr * (m_v[k] / (1.0 - b1 ** t)) / (
                np.sqrt(s_v[k] / (1.0 - b2 ** t)) + eps
            )

        if verbose and (t % 500 == 0 or t == 1):
            print(f"    [Joint-Cao] step {t:5d}: loss = {loss:.6e} "
                  f"over {len(env_data)} env(s)")

    return omega, vs, float(loss)