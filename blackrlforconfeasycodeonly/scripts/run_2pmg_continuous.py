"""Training script for 2PMG Continuous (LQR bilevel) with parallel collection.

irl_known_dynamics / irl_unknown_dynamics
------------------
Under the setting where the leader's state-transition dynamics are known (A, B, C) or unknown (estimated by regression from data),
Feature Expectation Matching (FEM) IRL is used to
estimate the follower's cost matrices (Q_hat, R_hat).

  1. Compute μ_expert = E[φ(s,b)] from expert trajectories (φ = quadratic features)
     Note: no OLS or inverse LQR is used. Only (s, a, b) trajectories are observed.
  2. IRL loop:
       w → diag(exp(w)) = (Q,R) → solve_2pmg_lqr → π* → E_π*[φ(s,b)] = μ_policy
       w += α (μ_expert - μ_policy)   ← max-ent IRL gradient
  3. Keep the converged (Q_hat, R_hat), and at every subsequent iteration
       solve_2pmg_lqr(A, B, C, Q_hat, R_hat, gamma_F, K_theta_current)
     re-solve the DARE to re-derive the follower policy (not BC).

Modes
-----
softvi             : Follower = analytical LQR optimal policy (true reward known).
irl_known_dynamics : Follower cost (Q,R) estimated via FEM-IRL from expert trajectories (dynamics known).
irl_unknown_dynamics : Dynamics estimated via LS, Follower cost (Q,R) estimated via FEM-IRL.
non_irl            : Follower policy fit from on-policy data (no IRL, no expert).

Leader policy
-------------
Linear Gaussian:  a_t = K_theta @ s_t + N(0, sigma^2 I)
Updated by REINFORCE (vanilla policy gradient with baseline).

Run
---
uv run python scripts/run_2pmg_continuous.py --mode softvi --output-dir data/2pmg_cont/test1
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from blackrl.algos.lqr_analytical import value_function_2pmg
from scripts.plot_learning_curves import plot_leader_performance_only, plot_learning_curves  # type: ignore
import matplotlib.pyplot as plt

import numpy as np
import torch
from blackrl.algos.lqr_analytical import solve_2pmg_lqr
from scipy.linalg import solve_discrete_lyapunov
from torch import nn, optim

# ---------------------------------------------------------------------------
# Environment constants (top-level for worker pickling)
# ---------------------------------------------------------------------------
DELTA_T = 0.1
_A = np.array([[1, DELTA_T], [0, 1]], dtype=np.float64)
_B = np.array([[0.0], [DELTA_T]], dtype=np.float64)  # follower input
_C = np.array([[DELTA_T], [0.0]], dtype=np.float64)  # leader influence
_W = np.eye(2, dtype=np.float64)  # follower state cost
_U = np.array([[1]], dtype=np.float64)  # follower action cost

_ENV_CLASS_NAME = "LQRConfigurableMDPEnv"
_ENV_MODULE_PATH = "blackrl.envs.lqr_env"
_ENV_INIT_KWARGS: dict = {
    "A": _A.tolist(),
    "B": _B.tolist(),
    "C": _C.tolist(),
    "Q": _W.tolist(),
    "R": _U.tolist(),
    "max_episode_steps": 100,
    "initial_state": [1.0, 0.0],
    "target_state": [-0.8, 0.0],
    "leader_reward_weights": [[0.5, 0.0], [0.0, 0.5]],
    "env_seed": 42,
}

PARAMS = {
    "n_leader_iterations": 1500,
    "n_episodes_per_iteration": 7,
    "learning_rate_leader": 0.05,
    "discount_leader": 0.99,
    "discount_follower": 0.99,
    "beta": 0.01,
    "leader_sigma": 1e-3,
    "max_update_norm": 1.0,
    "max_workers": 16,
    "verbose": True,
    # irl_recursive_rls mode
    "rls_reg": 1e-6,
    "rls_prior_precision": 1e-1,
    "rls_eigmin_gate": 1e-4,
    "rls_local_irl_iters": 200,
    "rls_local_irl_lr": 0.01,
    "rls_gd_lr": 0.01,
    "benefit_baseline_mode": "online-online",  # online-online | online-target | target-target
    # diagnostics
    "debug_benefit_compare": False,
    "debug_benefit_log_every": 10,
    "debug_trace": False,
    "debug_trace_every": 10,
    "debug_trace_jsonl": False,
    # if True, update actor-critic with all trajectories accumulated so far
    "full_batch_cumulative": False,
}


def make_env():
    from blackrl.envs.lqr_env import LQRConfigurableMDPEnv

    return LQRConfigurableMDPEnv(**_ENV_INIT_KWARGS)


# ---------------------------------------------------------------------------
# Top-level picklable worker — no closures, no lambdas
# b_mean = Ks_f @ s + Ka_f @ a + bias_f   (signs already absorbed into Ks_f, Ka_f)
# ---------------------------------------------------------------------------


def _worker_episode_2pmg(
    env_class_name: str,
    env_module_path: str,
    env_init_kwargs: dict,
    K_leader: np.ndarray,  # (p, n)
    sigma_leader: float,
    Ks_f: np.ndarray,  # (m, n)  mean = Ks_f @ s + Ka_f @ a + bias_f
    Ka_f: np.ndarray,  # (m, p)
    bias_f: np.ndarray,  # (m,)
    Sigma_f: np.ndarray,  # (m, m)
    discount_leader: float,
    discount_follower: float,
    seed: int,
    initial_state_override: np.ndarray | None = None,
) -> dict:
    """Single-episode rollout. Top-level so ProcessPoolExecutor can pickle it."""
    module = importlib.import_module(env_module_path)
    env_cls = getattr(module, env_class_name)
    kwargs = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in env_init_kwargs.items()}
    kwargs["env_seed"] = seed
    env = env_cls(**kwargs)

    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    if initial_state_override is not None:
        env.state = np.asarray(initial_state_override, dtype=np.float32).copy()
        obs = env.state

    s_list, a_list, b_list, rL_list, rF_list = [], [], [], [], []

    while True:
        s = obs.astype(np.float64)
        a = K_leader @ s + rng.normal(0.0, sigma_leader, size=(K_leader.shape[0],))
        b_mean = Ks_f @ s + Ka_f @ a + bias_f
        b = rng.multivariate_normal(b_mean, Sigma_f)

        step = env.step(a, b)
        r_L = step.env_info.get("leader_reward", float(step.reward))
        r_F = step.env_info.get("follower_reward", float(step.reward))
        s_list.append(s.tolist())
        a_list.append(a.tolist())
        b_list.append(b.tolist())
        rL_list.append(r_L)
        rF_list.append(r_F)
        obs = step.observation
        if step.last:
            break

    T = len(rL_list)
    G = [0.0] * T
    G[-1] = rL_list[-1]
    G_F = [0.0] * T
    G_F[-1] = rF_list[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rL_list[t] + discount_leader * G[t + 1]
        G_F[t] = rF_list[t] + discount_follower * G_F[t + 1]

    total = sum(rL_list)  # Calculate undiscounted cumulative reward
    return {
        "s": s_list,
        "a": a_list,
        "b": b_list,
        "r_L": rL_list,
        "r_F": rF_list,
        "G_L": G,
        "G_F": G_F,
        "total_leader_return": float(total),
    }


# ---------------------------------------------------------------------------
# Parallel collection
# ---------------------------------------------------------------------------


def _collect_episodes_parallel(
    leader_policy,
    Ks_f: np.ndarray,
    Ka_f: np.ndarray,
    bias_f: np.ndarray,
    Sigma_f,
    n_episodes: int,
    discount_leader: float,
    discount_follower: float,
    base_seed: int,
    max_workers: int,
    initial_state_overrides: list[np.ndarray] | None = None,
) -> tuple[dict, list]:
    """Collect n_episodes in parallel. Returns (pool, trajectories)."""
    max_w = min(n_episodes, max_workers)
    seeds = [base_seed * 100_000 + i for i in range(n_episodes)]

    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {
            ex.submit(
                _worker_episode_2pmg,
                _ENV_CLASS_NAME,
                _ENV_MODULE_PATH,
                _ENV_INIT_KWARGS,
                leader_policy.K.copy(),
                leader_policy.sigma,
                Ks_f,
                Ka_f,
                bias_f,
                Sigma_f,
                discount_leader,
                discount_follower,
                seeds[i],
                initial_state_overrides[i] if initial_state_overrides is not None else None,
            ): i
            for i in range(n_episodes)
        }
        ordered = [None] * n_episodes
        for fut in as_completed(futs):
            ordered[futs[fut]] = fut.result()

    pool = {"s": [], "a": [], "b": []}
    trajs = []
    for r in ordered:
        s_arr = [np.asarray(s) for s in r["s"]]
        a_arr = [np.asarray(a) for a in r["a"]]
        b_arr = [np.asarray(b) for b in r["b"]]
        pool["s"].extend(s_arr)
        pool["a"].extend(a_arr)
        pool["b"].extend(b_arr)
        trajs.append(
            {
                "s": s_arr,
                "a": a_arr,
                "b": b_arr,
                "r_L": r["r_L"],
                "r_F": r["r_F"],
                "G_L": np.asarray(r["G_L"]),
                "G_F": np.asarray(r["G_F"]),
                "total_leader_return": r["total_leader_return"],
            },
        )
    return pool, trajs


# ---------------------------------------------------------------------------
# Leader policy: a_t = K_theta @ s_t + N(0, σ²I)
# ---------------------------------------------------------------------------


class LinearLeaderPolicy:
    def __init__(self, n_leader: int, n_state: int, sigma: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.K = self.rng.standard_normal((n_leader, n_state), dtype=np.float64)
        self.sigma = float(sigma)

    def log_prob_grad_K(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        diff = (a - self.K @ s).reshape(-1, 1)
        return (diff @ s.reshape(1, -1)) / (self.sigma**2)

    def update(self, grad_K: np.ndarray, lr: float, max_update_norm: float | None = None) -> None:
        delta = lr * grad_K
        if max_update_norm is not None and max_update_norm > 0:
            dnorm = np.linalg.norm(delta)
            if dnorm > max_update_norm:
                delta = delta * (max_update_norm / (dnorm + 1e-12))
        self.K += delta


# ---------------------------------------------------------------------------
# Follower surrogate: analytical solution → worker params
# ---------------------------------------------------------------------------


def _sol_to_follower_params(sol: dict, m: int, beta: float):
    """Analytical solution → (Ks_f, Ka_f, bias_f, Sigma_f) ready for the worker."""
    Ks_f = -sol["Ks"]
    Ka_f = -sol["Ka"]
    bias_f = np.zeros(m, dtype=np.float64)
    Sigma_f = (beta / 2.0) * sol["S_inv"]
    return Ks_f, Ka_f, bias_f, Sigma_f


# ---------------------------------------------------------------------------
# Feature Expectation Matching (FEM) IRL  ← correct continuous IRL implementation
# ---------------------------------------------------------------------------


def _quad_features(s: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quadratic feature vector φ(s, a, b) = [s₁², ..., sₙ², b₁², ..., bₘ²].

    Assumption 1 of the paper: φ: S×A×B → R^K
    The leader action a is taken as input, but since the follower cost depends only on s and b,
    a² is not included in the features (when the follower cost is s'Qs + b'Rb).
    If needed, a² can be added with include_leader=True.

    Corresponding reward function: r_F(s,a,b;w) = φ(s,a,b)·w
    When w holds the diagonal entries: r_F = -Σ wᵢ sᵢ² - Σ wⱼ bⱼ²  = -(s'Qs + b'Rb)
    where Q = diag(w[:n]), R = diag(w[n:]).
    """
    return np.concatenate([s**2, b**2])  # φ(s,a,b): a independent cost


def compute_expert_fev(
    expert_trajs: list[dict],
    gamma_F: float,
) -> np.ndarray:
    """Compute the discounted feature expectation μ_expert = E[Σ γ^t φ(s_t,a_t,b_t)] from expert trajectories.

    Following the FEV definition of the paper (MDCE IRL), apply γ^t discounting:
        φ̄_BR^{γ_F} = E^{f,g*}[ Σ_t γ_F^t φ(s_t, a_t, b_t) ]

    Key point of IRL: no OLS or inverse LQR. Only observed (s, a, b) trajectories are used.

    Parameters
    ----------
    expert_trajs : list of dicts, each with 's', 'a', 'b' keys
                   (each element is the time-series list of one episode)
    gamma_F : float  follower discount factor

    Returns
    -------
    mu : (n_s + m,)  discounted feature expectation vector

    """
    traj_fevs = []
    for traj in expert_trajs:
        s_list = traj["s"]
        a_list = traj["a"]
        b_list = traj["b"]
        phi0 = _quad_features(
            np.asarray(s_list[0]),
            np.asarray(a_list[0]),
            np.asarray(b_list[0]),
        )
        traj_fev = np.zeros_like(phi0, dtype=np.float64)
        for t, (s, a, b) in enumerate(zip(s_list, a_list, b_list)):
            phi_t = _quad_features(np.asarray(s), np.asarray(a), np.asarray(b))
            traj_fev += (gamma_F**t) * phi_t
        traj_fevs.append(traj_fev)
    return np.mean(traj_fevs, axis=0)


def compute_policy_fev(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    gamma_F: float,
    beta: float,
    K_theta: np.ndarray,
    sigma_leader: float,
    Sigma_0: np.ndarray,
) -> np.ndarray:
    """Analytic μ_policy via the discounted Lyapunov equation (2PMG version, no sampling).

    Continuous analogue of compute_analytic_sf (successor features) in the discrete version.

    Derivation
    ------
    Closed-loop matrix:
        K_eff = Ks + Ka K_θ                     (m, n)
        A_cl  = A_tilde - B K_eff               (n, n)  (stable closed-loop matrix)

    Process noise covariance:
        Q_noise = (C - B Ka)σ_L²(C-BKa)' + B(β/2)S⁻¹B' + σ_env² I

    Discounted Lyapunov equation (continuous analogue of the SF recursion):
        M = γ A_cl M A_cl' + Σ_0 + γ/(1-γ) Q_noise
        ↔ solve_discrete_lyapunov(√γ A_cl,  Σ_0 + γ/(1-γ) Q_noise)

    FEV (no sampling):
        μ_s = diag(M)
        μ_b = diag(K_eff M K_eff') + (1/(1-γ)) diag(Cov_b_eff)

    Parameters
    ----------
    sigma_leader : float  std of the leader policy (match the value used for episode collection)
    Sigma_0      : (n, n)  initial state covariance (None → I)

    """
    try:
        sol = solve_2pmg_lqr(A, B, C, Q, R, gamma_F, K_theta)
    except Exception:
        n_s = A.shape[0]
        m = B.shape[1]
        return np.ones(n_s + m) * 1e6

    n_s = A.shape[0]
    n_p = C.shape[1]
    m = B.shape[1]
    Ks = sol["Ks"]  # (m, n)
    Ka = sol["Ka"]  # (m, p)
    S_inv = sol["S_inv"]  # (m, m)
    A_tilde = sol["A_tilde"]  # (n, n) = A + C K_θ

    # ------------------------------------------------------------------
    # Step 1: closed-loop matrix  (continuous version of Soft VI / SF)
    # b_t = -(Ks + Ka K_θ) s_t + ε_b_eff  =  -K_eff s_t + ε_b_eff
    # s_{t+1} = A_tilde s_t + B b_t + ε_s
    #          = (A_tilde - B K_eff) s_t + noise
    # ------------------------------------------------------------------
    K_eff = Ks + Ka @ K_theta  # (m, n)
    A_cl = A_tilde - B @ K_eff  # (n, n)

    # ------------------------------------------------------------------
    # Step 2: process noise covariance
    #   ε_a  ~ N(0, σ_L² I)  ← stochastic actions of the leader policy
    #   ε_b  ~ N(0, (β/2) S⁻¹)  ← entropy of the MaxEnt follower
    #   ε_b_eff = -Ka ε_a + ε_b
    #   w_t = (C - B Ka) ε_a + B ε_b + σ_env ε_s
    # ------------------------------------------------------------------
    C_eff = C - B @ Ka  # (n, p)
    Cov_a = sigma_leader**2 * np.eye(n_p)  # from the leader std
    Cov_b_entropy = (beta / 2.0) * S_inv  # from the MaxEnt entropy
    Cov_b_eff = Ka @ Cov_a @ Ka.T + Cov_b_entropy  # local follower noise

    Q_noise = (
        C_eff @ Cov_a @ C_eff.T  # propagation of leader noise
        + B @ Cov_b_entropy @ B.T  # propagation of follower noise
    )

    # ------------------------------------------------------------------
    # Step 3: solve the discounted Lyapunov equation (continuous analogue of the discrete SF)
    #
    #   M = γ A_cl M A_cl' + Σ_0 + γ/(1-γ) Q_noise
    #
    # solve_discrete_lyapunov(A, Q) solves X = A X A' + Q, so
    #   A := √γ A_cl
    #   Q := Σ_0 + γ/(1-γ) Q_noise
    # ------------------------------------------------------------------
    sq_g = np.sqrt(gamma_F)
    Q_lyap = Sigma_0 + (gamma_F / (1.0 - gamma_F)) * Q_noise
    try:
        M = solve_discrete_lyapunov(sq_g * A_cl, Q_lyap)
    except Exception:
        return np.ones(n_s + m) * 1e6

    # ------------------------------------------------------------------
    # Step 4: read the FEV off the diagonal of the covariance
    #   E[Σ_t γ^t s_i(t)^2] = M[i,i]
    #   E[Σ_t γ^t b_j(t)^2] = (K_eff M K_eff')[j,j] + (1/(1-γ)) Cov_b_eff[j,j]
    # ------------------------------------------------------------------
    mu_s = np.diag(M)  # (n_s,)
    Mb = K_eff @ M @ K_eff.T + (1.0 / (1.0 - gamma_F)) * Cov_b_eff
    mu_b = np.diag(Mb)  # (m,)

    return np.concatenate([mu_s, mu_b])


def fem_irl_loop(
    expert_mu: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    gamma_F: float,
    beta: float,
    K_theta_0: np.ndarray,
    sigma_leader: float,
    Sigma_0: np.ndarray,
    n_irl_iters: int,
    lr_irl: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature Expectation Matching IRL for 2PMG (LQR setting).

    Correct IRL implementation (fully analytic via the discounted Lyapunov equation):
      - Input: expert feature expectation μ_expert
      - Output: estimated cost matrices (Q_hat, R_hat)
      - No OLS / inverse LQR / Ks_exp / sampling needed
      - Each iteration: DARE (↔Soft VI) → Lyapunov (↔SF) → discounted FEV

    Parameters
    ----------
    sigma_leader : float  set to the same value as for trajectory collection
    n_irl_iters  : number of IRL gradient updates
    lr_irl       : learning rate

    Returns
    -------
    Q_hat : (n_s, n_s)  diagonal estimated state cost matrix
    R_hat : (m, m)      diagonal estimated action cost matrix

    """
    n_s = A.shape[0]
    m = B.shape[1]
    # Parametrization: w = [log q₀, ..., log qₙ, log r₀, ..., log rₘ]
    # → Q = diag(exp(w[:n_s])), R = diag(exp(w[n_s:]))
    # log guarantees positivity (no Nelder-Mead needed)
    log_w = np.zeros(n_s + m, dtype=np.float64)  # init: Q=R=I

    # Use negative features for reward view: phi_r = -phi_cost.
    # We keep cost parameterization (Q,R) positive via exp(log_w), and apply
    # the sign convention only in feature-matching residual.
    expert_mu_neg = -expert_mu

    for i in range(n_irl_iters):
        # Limit log_w to prevent exp overflow
        log_w = np.clip(log_w, -15.0, 15.0)
        Q_diag = np.exp(log_w[:n_s])
        R_diag = np.exp(log_w[n_s:])

        Q = np.diag(Q_diag)
        R = np.diag(R_diag)

        mu_policy = compute_policy_fev(
            A,
            B,
            C,
            Q,
            R,
            gamma_F,
            beta,
            K_theta_0,
            sigma_leader=sigma_leader,
            Sigma_0=Sigma_0,
        )
        mu_policy_neg = -mu_policy

        # With phi_r = -phi_cost, reward-style feature matching is:
        #   grad_theta = mu_expert_r - mu_policy_r
        # Under cost-parameterization with log_w, the resulting cost update
        # matches this residual through chain rule below.
        grad_w = expert_mu_neg - mu_policy_neg
        grad_logw = grad_w * np.exp(log_w)  # chain rule

        # Stability: clip gradients
        grad_logw = np.clip(grad_logw, -1.0, 1.0)
        log_w += lr_irl * grad_logw

        if verbose and (i % 20 == 0 or i == n_irl_iters - 1):
            diff_norm = float(np.linalg.norm(expert_mu - mu_policy))
            print(
                f"    [FEM-IRL] iter {i:4d}: ||μ_e - μ_π|| = {diff_norm:.6f} | "
                f"Q_diag={np.exp(log_w[:n_s]).round(4)}, "
                f"R_diag={np.exp(log_w[n_s:]).round(4)}",
            )

    Q_hat = np.diag(np.exp(log_w[:n_s]))
    R_hat = np.diag(np.exp(log_w[n_s:]))
    return Q_hat, R_hat


# ---------------------------------------------------------------------------
# RLS helpers (same sign convention as conf: Q_diag = exp(-ω_q), R_diag = exp(-ω_r))
# ---------------------------------------------------------------------------


def _omega_from_diag_cost_2pmg(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Map diagonal costs Q, R to ω where diag = exp(-ω)."""
    q = np.clip(np.diag(Q), 1e-3, 1e2)
    r = np.clip(np.diag(R), 1e-3, 1e2)
    return np.concatenate([-np.log(q), -np.log(r)]).astype(np.float64)


def _diag_from_reward_param_2pmg(omega: np.ndarray, n_s: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack ω into positive diagonal cost matrices (Q, R)."""
    omega = np.asarray(omega, dtype=np.float64)
    omega_q = np.clip(omega[:n_s], -8.0, 8.0)
    omega_r = np.clip(omega[n_s : n_s + m], -8.0, 8.0)
    q = np.clip(np.exp(-omega_q), 1e-3, 1e2)
    r = np.clip(np.exp(-omega_r), 1e-3, 1e2)
    return np.diag(q), np.diag(r)


def _2pmg_info_matrix_from_trajs(
    trajs: list[dict],
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    K_theta: np.ndarray,
    omega: np.ndarray,
    gamma_F: float,
    n_s: int,
    m: int,
    beta: float,
    reg: float,
) -> np.ndarray:
    """Analytic estimate of Σ_k for the 2PMG.

    Corresponds to _conf_info_matrix_from_trajs in the conf version, but in the 2PMG
    the follower acts conditioned on (s, a), so we approximate it by:
        b_mean  = -(Ks + Ka K_θ) s - Ka (a - K_θ s)
                = -K_eff s - Ka ε_a
    and build the closed-loop matrix with K_eff = Ks + Ka @ K_theta.

    Compute ψ_i(s, b) = ∂Q_soft(s, b, a) / ∂ω_i analytically:
        dP_i: Lyapunov equation (discrete time, based on A_cl)
        dS_i = dR_i + γ B' dP_i B
        dKs_i, dKa_i: sensitivity of the optimal solution

    Center using (s, a, b) in the trajectories by subtracting the Gaussian mean E_b[ψ_i | s, a].
    """
    dim = n_s + m
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    K_theta = np.asarray(K_theta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    if len(trajs) == 0:
        return reg * np.eye(dim, dtype=np.float64)

    Q0, R0 = _diag_from_reward_param_2pmg(omega, n_s, m)
    try:
        sol0 = solve_2pmg_lqr(A, B, C, Q0, R0, gamma_F, K_theta)
    except Exception:
        return reg * np.eye(dim, dtype=np.float64)

    S0 = np.asarray(sol0["S"], dtype=np.float64)
    S0_inv = np.asarray(sol0["S_inv"], dtype=np.float64)
    Ks0 = np.asarray(sol0["Ks"], dtype=np.float64)  # (m, n)
    Ka0 = np.asarray(sol0["Ka"], dtype=np.float64)  # (m, p)
    P0 = np.asarray(sol0["P"], dtype=np.float64)  # (n, n)
    A_tilde = np.asarray(sol0["A_tilde"], dtype=np.float64)  # (n, n)

    K_eff = Ks0 + Ka0 @ K_theta  # (m, n)  mean, ignoring the stochastic leader noise
    A_cl = A_tilde - B @ K_eff  # (n, n)

    # Local follower noise covariance (MaxEnt): Cov_b_eff ≈ (β/2) S⁻¹
    Cov_b = (beta / 2.0) * S0_inv  # (m, m)

    # ---- derivative coefficients for each ω_i ----
    omega_q = np.clip(omega[:n_s], -8.0, 8.0)
    omega_r = np.clip(omega[n_s : n_s + m], -8.0, 8.0)
    q_raw = np.exp(-omega_q)
    r_raw = np.exp(-omega_r)
    dq = np.where((q_raw > 1e-3) & (q_raw < 1e2), -q_raw, 0.0)
    dr = np.where((r_raw > 1e-3) & (r_raw < 1e2), -r_raw, 0.0)

    dS = np.zeros((dim, m, m), dtype=np.float64)
    dKs = np.zeros((dim, m, n_s), dtype=np.float64)
    dKa = np.zeros((dim, m, C.shape[1]), dtype=np.float64)
    sq_g = np.sqrt(gamma_F)

    for i in range(dim):
        dQ_i = np.zeros((n_s, n_s), dtype=np.float64)
        dR_i = np.zeros((m, m), dtype=np.float64)
        if i < n_s:
            dQ_i[i, i] = dq[i]
        else:
            j = i - n_s
            dR_i[j, j] = dr[j]

        # Envelope theorem: dP_i solves a Lyapunov equation
        # dP_i = (dQ_i + K_eff' dR_i K_eff) + γ A_cl' dP_i A_cl
        M_i = dQ_i + K_eff.T @ dR_i @ K_eff
        dP_i = solve_discrete_lyapunov(sq_g * A_cl.T, M_i)

        dS_i = dR_i + gamma_F * (B.T @ dP_i @ B)
        # Use np.linalg.solve instead of explicitly computing the inverse (numerically stable)
        # dKs_i = γ (-S⁻¹ dS_i S⁻¹ B' P A + S⁻¹ B' dP_i A)
        #      = γ S⁻¹ (-dS_i S⁻¹ B' P A + B' dP_i A)
        rhs_Ks = -dS_i @ np.linalg.solve(S0, B.T @ P0 @ A) + B.T @ dP_i @ A
        dKs_i = gamma_F * np.linalg.solve(S0, rhs_Ks)

        # dKa_i = γ (-S⁻¹ dS_i S⁻¹ B' P C + S⁻¹ B' dP_i C)
        #      = γ S⁻¹ (-dS_i S⁻¹ B' P C + B' dP_i C)
        rhs_Ka = -dS_i @ np.linalg.solve(S0, B.T @ P0 @ C) + B.T @ dP_i @ C
        dKa_i = gamma_F * np.linalg.solve(S0, rhs_Ka)

        dS[i] = dS_i
        dKs[i] = dKs_i
        dKa[i] = dKa_i

    # ---- accumulate Σ_k from the trajectories ----
    Sigma_acc = np.zeros((dim, dim), dtype=np.float64)
    w_sum = 0.0

    for traj in trajs:
        for t, (s, a, b) in enumerate(zip(traj["s"], traj["a"], traj["b"])):
            s = np.asarray(s, dtype=np.float64).ravel()
            a = np.asarray(a, dtype=np.float64).ravel()
            b = np.asarray(b, dtype=np.float64).ravel()
            # Conditional mean of the follower b | s, a
            mu_b = -(Ks0 @ s + Ka0 @ a)
            w_t = float(gamma_F**t)

            psi_obs = np.zeros(dim, dtype=np.float64)
            psi_mean = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                dSi = dS[i]
                dKsi = dKs[i]
                dKai = dKa[i]
                # dK_eff_i = dKs_i + dKa_i @ K_theta  (could be approximated by 0, but computed exactly)
                dK_eff_i = dKsi + dKai @ K_theta
                Mi = dSi @ K_eff + S0 @ dK_eff_i  # d(S K_eff)/dω_i

                psi_obs[i] = -(float(b @ dSi @ b) + 2.0 * float(b @ Mi @ s))

                term_quad = float(np.trace(dSi @ Cov_b) + mu_b @ dSi @ mu_b)
                term_cross = 2.0 * float(mu_b @ Mi @ s)
                psi_mean[i] = -(term_quad + term_cross)

            delta = psi_obs - psi_mean
            Sigma_acc += w_t * np.outer(delta, delta)
            w_sum += w_t

    if w_sum <= 0.0:
        return reg * np.eye(dim, dtype=np.float64)

    Sigma_k = Sigma_acc / w_sum
    Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)
    Sigma_k += reg * np.eye(dim, dtype=np.float64)
    return Sigma_k


# ---------------------------------------------------------------------------
# STDPG Actor-Critic update
# ---------------------------------------------------------------------------


class Critic(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a, b):
        return self.net(torch.cat([s, a, b], dim=-1))


def stdpg_actor_critic_update(
    leader_policy: LinearLeaderPolicy,
    leader_critic: Critic,
    leader_target_critic: Critic,
    lc_opt: optim.Optimizer,
    trajectories: list[dict],
    Ks_f: np.ndarray,
    Ka_f: np.ndarray,
    bias_f: np.ndarray,
    Sigma_f: np.ndarray,
    learning_rate: float,
    max_update_norm: float,
    beta: float,
    gamma_L: float,
    gamma_F: float,
    mode: str,
    A: np.ndarray | None = None,
    C: np.ndarray | None = None,
    W_hat: np.ndarray | None = None,
    sol_f: dict | None = None,
    iteration: int | None = None,
) -> tuple[float, dict]:
    s_ts = torch.tensor(np.concatenate([np.asarray(t["s"]) for t in trajectories]), dtype=torch.float32)
    a_ts = torch.tensor(np.concatenate([np.asarray(t["a"]) for t in trajectories]), dtype=torch.float32)
    b_ts = torch.tensor(np.concatenate([np.asarray(t["b"]) for t in trajectories]), dtype=torch.float32)
    gL_ts = torch.tensor(np.concatenate([np.asarray(t["G_L"]) for t in trajectories]), dtype=torch.float32).unsqueeze(1)

    # --- Critic update: SARSA with target network bootstrap ---
    s_curr, a_curr, b_curr = [], [], []
    r_curr = []
    s_next, a_next, b_next = [], [], []
    done = []
    for traj in trajectories:
        T = len(traj["s"])
        for t in range(T):
            s_curr.append(np.asarray(traj["s"][t], dtype=np.float32))
            a_curr.append(np.asarray(traj["a"][t], dtype=np.float32))
            b_curr.append(np.asarray(traj["b"][t], dtype=np.float32))
            r_curr.append(float(traj["r_L"][t]))

            if t < T - 1:
                s_next.append(np.asarray(traj["s"][t + 1], dtype=np.float32))
                a_next.append(np.asarray(traj["a"][t + 1], dtype=np.float32))
                b_next.append(np.asarray(traj["b"][t + 1], dtype=np.float32))
                done.append(0.0)
            else:
                # terminal sample (bootstrap masked by done=1)
                s_next.append(np.asarray(traj["s"][t], dtype=np.float32))
                a_next.append(np.asarray(traj["a"][t], dtype=np.float32))
                b_next.append(np.asarray(traj["b"][t], dtype=np.float32))
                done.append(1.0)

    s_curr_ts = torch.tensor(np.asarray(s_curr), dtype=torch.float32)
    a_curr_ts = torch.tensor(np.asarray(a_curr), dtype=torch.float32)
    b_curr_ts = torch.tensor(np.asarray(b_curr), dtype=torch.float32)
    r_curr_ts = torch.tensor(np.asarray(r_curr), dtype=torch.float32).unsqueeze(1)
    s_next_ts = torch.tensor(np.asarray(s_next), dtype=torch.float32)
    a_next_ts = torch.tensor(np.asarray(a_next), dtype=torch.float32)
    b_next_ts = torch.tensor(np.asarray(b_next), dtype=torch.float32)
    done_ts = torch.tensor(np.asarray(done), dtype=torch.float32).unsqueeze(1)

    for _ in range(7):
        lc_opt.zero_grad()
        q_curr = leader_critic(s_curr_ts, a_curr_ts, b_curr_ts)
        with torch.no_grad():
            q_next = leader_target_critic(s_next_ts, a_next_ts, b_next_ts)
            td_target = r_curr_ts + gamma_L * (1.0 - done_ts) * q_next
        l_loss = nn.functional.mse_loss(q_curr, td_target)
        l_loss.backward()
        lc_opt.step()

        # Target network smoothing (Polyak averaging)
        tau = 0.01
        for target_param, param in zip(leader_target_critic.parameters(), leader_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    with torch.no_grad():
        s_np = s_ts.numpy()
        a_np = a_ts.numpy()
        q_L = leader_critic(s_ts, a_ts, b_ts).squeeze(-1).numpy()
        q_target = leader_target_critic(s_ts, a_ts, b_ts).squeeze(-1).numpy()
        v_L = np.zeros(len(s_np))
        v_benefit = np.zeros(len(s_np))
        v_benefit_target = np.zeros(len(s_np))
        n_samples = 5
        for _ in range(n_samples):
            a_samp_np = (leader_policy.K @ s_np.T).T + np.random.normal(
                0,
                leader_policy.sigma,
                size=(len(s_np), a_ts.shape[1]),
            )
            b_mean = (Ks_f @ s_np.T).T + (Ka_f @ a_samp_np.T).T + bias_f
            b_samp_np = b_mean + np.random.multivariate_normal(np.zeros(len(bias_f)), Sigma_f, size=len(s_np))
            v_L += (
                leader_critic(s_ts, torch.tensor(a_samp_np, dtype=torch.float32), torch.tensor(b_samp_np, dtype=torch.float32))
                .squeeze(-1)
                .numpy()
            )

            b_mean_given_a = (Ks_f @ s_np.T).T + (Ka_f @ a_np.T).T + bias_f
            b_samp_given_a = b_mean_given_a + np.random.multivariate_normal(np.zeros(len(bias_f)), Sigma_f, size=len(s_np))
            v_benefit += leader_critic(s_ts, a_ts, torch.tensor(b_samp_given_a, dtype=torch.float32)).squeeze(-1).numpy()
            v_benefit_target += (
                leader_target_critic(s_ts, a_ts, torch.tensor(b_samp_given_a, dtype=torch.float32)).squeeze(-1).numpy()
            )

        v_L /= n_samples
        v_benefit /= n_samples
        v_benefit_target /= n_samples

        # advantage uses empirical return baseline, benefit uses online Q baseline
        gL_flattened = gL_ts.flatten().numpy()
        advantage = gL_flattened - v_L
        benefit_mode = str(PARAMS.get("benefit_baseline_mode", "online-online"))
        if benefit_mode == "online-target":
            benefit = q_L - v_benefit_target
        elif benefit_mode == "target-target":
            benefit = q_target - v_benefit_target
        else:
            benefit = q_L - v_benefit

        # Optional diagnostics: compare online-online vs online-target variants.
        if PARAMS.get("debug_benefit_compare", False):
            benefit_ot = q_L - v_benefit_target
            benefit_tt = q_target - v_benefit_target
            log_every = int(PARAMS.get("debug_benefit_log_every", 10))
            if iteration is None or (log_every > 0 and iteration % log_every == 0):
                print(
                    "  [BenefitDiag] "
                    f"iter={iteration if iteration is not None else -1} | "
                    f"std(oo)={np.std(benefit):.6f}, "
                    f"std(ot)={np.std(benefit_ot):.6f}, "
                    f"std(tt)={np.std(benefit_tt):.6f}, "
                    f"mean|oo|={np.mean(np.abs(benefit)):.6f}, "
                    f"mean|ot|={np.mean(np.abs(benefit_ot)):.6f}, "
                    f"mean|tt|={np.mean(np.abs(benefit_tt)):.6f}",
                )


    grad_K = np.zeros_like(leader_policy.K)
    idx = 0
    m_dim = Sigma_f.shape[0] if Sigma_f is not None else 1
    total_steps = 0
    grad1_norm_sum = 0.0
    grad2_norm_sum = 0.0
    influence_norm_sum = 0.0
    influence_count = 0

    for traj in trajectories:
        T = len(traj["s"])
        total_steps += T
        log_pi_grads = []
        for s, a in zip(traj["s"], traj["a"]):
            log_pi_grads.append(leader_policy.log_prob_grad_K(np.array(s), np.array(a)))

        for t in range(T):
            grad_1 = log_pi_grads[t] * advantage[idx + t]
            grad1_norm_sum += float(np.linalg.norm(grad_1))

            # --- Influence gradient (2nd term) ---
            # non_irl mode or missing analytical info skips the influence term (Benefit term)
            if mode == "non_irl" or sol_f is None or A is None or C is None or W_hat is None:
                grad_2 = np.zeros_like(grad_K)
            else:
                influ_grad = np.zeros_like(grad_K)
                for t_prime in range(t, T):
                    # PTIA approach: use analytical expectation V_F(s, a) instead of trajectory G_F
                    s_prime = np.array(traj["s"][t_prime])
                    a_prime = np.array(traj["a"][t_prime])

                    v_F_exact = value_function_2pmg(
                        s=s_prime,
                        a=a_prime,
                        A=A,
                        C=C,
                        W=W_hat,
                        sol=sol_f,
                        beta=beta,
                        m=m_dim,
                        Sigma_theta=np.eye(leader_policy.K.shape[0]) * (leader_policy.sigma**2),
                    )
                    influ_grad += (gamma_F ** (t_prime - t)) * log_pi_grads[t_prime] * v_F_exact

                grad_2 = (1.0 / beta) * benefit[idx + t] * influ_grad
                influence_norm_sum += float(np.linalg.norm(influ_grad))
                influence_count += 1

            grad2_norm_sum += float(np.linalg.norm(grad_2))
            grad_K += grad_1 + grad_2
        idx += T

    grad_K /= total_steps

    # Clip gradient to a maximum norm of 1.0
    grad_norm = np.linalg.norm(grad_K)
    if grad_norm > 1.0:
        grad_K = grad_K / grad_norm

    leader_policy.update(grad_K, learning_rate, max_update_norm=max_update_norm)

    debug = {
        "benefit_mode": benefit_mode,
        "mean_q_online": float(np.mean(q_L)),
        "std_q_online": float(np.std(q_L)),
        "mean_q_target": float(np.mean(q_target)),
        "std_q_target": float(np.std(q_target)),
        "mean_v_online": float(np.mean(v_benefit)),
        "std_v_online": float(np.std(v_benefit)),
        "mean_v_target": float(np.mean(v_benefit_target)),
        "std_v_target": float(np.std(v_benefit_target)),
        "mean_adv": float(np.mean(advantage)),
        "std_adv": float(np.std(advantage)),
        "mean_benefit": float(np.mean(benefit)),
        "std_benefit": float(np.std(benefit)),
        "mean_abs_benefit": float(np.mean(np.abs(benefit))),
        "grad1_norm_mean": float(grad1_norm_sum / max(total_steps, 1)),
        "grad2_norm_mean": float(grad2_norm_sum / max(total_steps, 1)),
        "influence_norm_mean": float(influence_norm_sum / max(influence_count, 1)),
        "gradK_norm": float(grad_norm),
    }

    return float(np.mean([t["total_leader_return"] for t in trajectories])), debug


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def estimate_dynamics_lsq(trajectories: list[dict], n_s: int, n_p: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate A, B, C matrices using ordinary least squares from trajectories."""
    X_list = []
    Y_list = []
    for traj in trajectories:
        T = len(traj["s"]) - 1
        for t in range(T):
            s_t = np.asarray(traj["s"][t])
            a_t = np.asarray(traj["a"][t])
            b_t = np.asarray(traj["b"][t])
            s_next = np.asarray(traj["s"][t + 1])
            x_t = np.concatenate([s_t, a_t, b_t])
            X_list.append(x_t)
            Y_list.append(s_next)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Solve Y = X @ M -> M = pinv(X) @ Y
    M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    M_T = M.T  # shape: (n_s, n_s + n_p + m)

    A_hat = M_T[:, :n_s]
    C_hat = M_T[:, n_s : n_s + n_p]
    B_hat = M_T[:, n_s + n_p :]

    return A_hat, B_hat, C_hat


def run_experiment(mode: str, output_dir: str, seed: int) -> dict:
    print(f"\n=== 2PMG Continuous | mode={mode} | seed={seed} ===")
    n_s = _A.shape[0]
    n_p = _C.shape[1]
    m = _B.shape[1]

    leader_policy = LinearLeaderPolicy(n_p, n_s, sigma=PARAMS["leader_sigma"], seed=seed)

    gamma_F = PARAMS["discount_follower"]
    gamma_L = PARAMS["discount_leader"]
    lr = PARAMS["learning_rate_leader"]
    max_upd = PARAMS["max_update_norm"]
    n_ep = PARAMS["n_episodes_per_iteration"]
    beta = PARAMS["beta"]
    verbose = PARAMS["verbose"]
    max_w = PARAMS["max_workers"]

    stats = {
        "leader_return": [],
        "leader_return_std": [],
        "mode": mode,
        "eval_trajectories": None,
        "best_eval_trajectories": None,
        "best_iteration": None,
        "best_leader_K": None,
        "best_leader_return": -np.inf,
        "debug_trace": [],
    }
    cumulative_trajectories: list[dict] = []

    # Surrogate / IRL state
    Ks_f = np.zeros((m, n_s))
    Ka_f = np.zeros((m, n_p))
    bias_f = np.zeros(m)
    Sigma_f = 0.01 * np.eye(m)
    Q_hat: np.ndarray | None = None  # estimated by inverse LQR (irl_known_dynamics)
    R_hat: np.ndarray | None = None
    surrogate_fitted = False

    # For unknown dynamics mode
    A_hat = np.copy(_A)
    B_hat = np.copy(_B)
    C_hat = np.copy(_C)

    # RLS accumulators for irl_recursive_rls / irl_unknown_recursive_rls
    rls_sigma_bar: np.ndarray | None = None
    rls_v_acc: np.ndarray | None = None
    rls_omega_bar: np.ndarray | None = None
    if mode in ("irl_recursive_rls", "irl_unknown_recursive_rls"):
        dim = n_s + m
        prior_prec = float(PARAMS["rls_prior_precision"])
        rls_sigma_bar = prior_prec * np.eye(dim, dtype=np.float64)
        rls_v_acc = np.zeros(dim, dtype=np.float64)
        rls_omega_bar = np.zeros(dim, dtype=np.float64)

    lc_net = Critic(n_s + n_p + m)
    leader_target_critic = copy.deepcopy(lc_net)
    leader_target_critic.eval()
    lc_opt = optim.Adam(lc_net.parameters(), lr=1e-3)

    # ---- Critic Pre-training with Uniform State Distribution ----
    if verbose:
        print(f"  [Pre-training] Initializing critic with uniform state distribution ({n_ep * 100} episodes)...")

    # Sample uniformly over roughly [-1.5, 1.5] x [-1.5, 1.5] of the state space
    n_pre_ep = n_ep * 10
    rng_pre = np.random.default_rng(seed)
    # The position range covers the target (-0.8) and the initial value (1.0)
    low = np.array([-1.5, -1.0])
    high = np.array([1.5, 1.0])
    random_starts = [rng_pre.uniform(low, high) for _ in range(n_pre_ep)]

    # Collect with the current (Oracle/IRL) follower parameters
    if mode == "softvi" or mode == "non_irl":
        sol_pre = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)
    else:  # irl
        sol_pre = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)  # use W, U on the first call

    Ksf_pre, Kaf_pre, bf_pre, Sf_pre = _sol_to_follower_params(sol_pre, m, beta)

    _, pre_trajs = _collect_episodes_parallel(
        leader_policy,
        Ksf_pre,
        Kaf_pre,
        bf_pre,
        Sf_pre,
        n_pre_ep,
        gamma_L,
        gamma_F,
        base_seed=seed * 777,
        max_workers=max_w,
        initial_state_overrides=random_starts,
    )

    # Train the critic on the collected data for about 50 epochs (SARSA TD)
    pre_s_curr, pre_a_curr, pre_b_curr = [], [], []
    pre_r_curr = []
    pre_s_next, pre_a_next, pre_b_next = [], [], []
    pre_done = []
    for traj in pre_trajs:
        T = len(traj["s"])
        for t in range(T):
            pre_s_curr.append(np.asarray(traj["s"][t], dtype=np.float32))
            pre_a_curr.append(np.asarray(traj["a"][t], dtype=np.float32))
            pre_b_curr.append(np.asarray(traj["b"][t], dtype=np.float32))
            pre_r_curr.append(float(traj["r_L"][t]))
            if t < T - 1:
                pre_s_next.append(np.asarray(traj["s"][t + 1], dtype=np.float32))
                pre_a_next.append(np.asarray(traj["a"][t + 1], dtype=np.float32))
                pre_b_next.append(np.asarray(traj["b"][t + 1], dtype=np.float32))
                pre_done.append(0.0)
            else:
                pre_s_next.append(np.asarray(traj["s"][t], dtype=np.float32))
                pre_a_next.append(np.asarray(traj["a"][t], dtype=np.float32))
                pre_b_next.append(np.asarray(traj["b"][t], dtype=np.float32))
                pre_done.append(1.0)

    pre_s_curr_ts = torch.tensor(np.asarray(pre_s_curr), dtype=torch.float32)
    pre_a_curr_ts = torch.tensor(np.asarray(pre_a_curr), dtype=torch.float32)
    pre_b_curr_ts = torch.tensor(np.asarray(pre_b_curr), dtype=torch.float32)
    pre_r_curr_ts = torch.tensor(np.asarray(pre_r_curr), dtype=torch.float32).unsqueeze(1)
    pre_s_next_ts = torch.tensor(np.asarray(pre_s_next), dtype=torch.float32)
    pre_a_next_ts = torch.tensor(np.asarray(pre_a_next), dtype=torch.float32)
    pre_b_next_ts = torch.tensor(np.asarray(pre_b_next), dtype=torch.float32)
    pre_done_ts = torch.tensor(np.asarray(pre_done), dtype=torch.float32).unsqueeze(1)

    tau_pre = 0.01
    for epoch in range(50):
        lc_opt.zero_grad()
        pre_q_curr = lc_net(pre_s_curr_ts, pre_a_curr_ts, pre_b_curr_ts)
        with torch.no_grad():
            pre_q_next = leader_target_critic(pre_s_next_ts, pre_a_next_ts, pre_b_next_ts)
            pre_td_target = pre_r_curr_ts + gamma_L * (1.0 - pre_done_ts) * pre_q_next
        loss = nn.functional.mse_loss(pre_q_curr, pre_td_target)
        loss.backward()
        lc_opt.step()

        # target critic update (Polyak)
        for target_param, param in zip(leader_target_critic.parameters(), lc_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau_pre) + param.data * tau_pre)

        if verbose and epoch % 10 == 0:
            print(f"    Pre-train epoch {epoch:2d}: loss = {loss.item():.6f}")

    leader_target_critic = copy.deepcopy(lc_net)
    leader_target_critic.eval()

    for iteration in range(PARAMS["n_leader_iterations"]):
        # ---- Determine follower params for this iteration ----
        if mode == "softvi":
            sol = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)
            Ks_f, Ka_f, bias_f, Sigma_f = _sol_to_follower_params(sol, m, beta)

        elif mode == "irl_known_dynamics" or mode == "irl_unknown_dynamics":
            if not surrogate_fitted:
                # === Estimation with the correct FEM-IRL ===

                # Step 1: collect expert data
                # Generate expert trajectories with the analytic follower solution under the true (W, U)
                # (in practice this corresponds to "just observing expert trajectories")
                sol0 = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)
                Ks0, Ka0, b0, Sf0 = _sol_to_follower_params(sol0, m, beta)

                collect_n_ep = n_ep * 10 if mode == "irl_unknown_dynamics" else n_ep
                _, expert_trajs = _collect_episodes_parallel(
                    leader_policy,
                    Ks0,
                    Ka0,
                    b0,
                    Sf0,
                    collect_n_ep,
                    gamma_L,
                    gamma_F,
                    base_seed=seed * 1000,
                    max_workers=max_w,
                )

                if mode == "irl_unknown_dynamics":
                    if verbose:
                        print("  [Dynamics Estimator] Estimating A, B, C from expert trajectories via OLS...")
                    A_hat, B_hat, C_hat = estimate_dynamics_lsq(expert_trajs, n_s, n_p, m)
                    if verbose:
                        print(f"    Estimated A:\n{A_hat.round(4)}")
                        print(f"    Estimated B:\n{B_hat.round(4)}")
                        print(f"    Estimated C:\n{C_hat.round(4)}")
                else:
                    A_hat, B_hat, C_hat = _A, _B, _C

                # Step 2: compute the expert feature expectation
                # Ks_exp is not used. Just accumulate φ(s_t, a_t, b_t) with γ^t discounting
                # from the observed (s, a, b) trajectories (FEV definition in the paper)
                mu_expert = compute_expert_fev(expert_trajs, gamma_F)
                if verbose:
                    print(
                        f"  [FEM-IRL] μ_expert = {mu_expert.round(4)}",
                    )

                # Step 3: FEM-IRL loop → estimate (Q_hat, R_hat)
                # No OLS / inverse LQR / Nelder-Mead needed
                if verbose:
                    print("  [FEM-IRL] Starting feature expectation matching...")
                Q_hat, R_hat = fem_irl_loop(
                    expert_mu=mu_expert,
                    A=A_hat,
                    B=B_hat,
                    C=C_hat,
                    gamma_F=gamma_F,
                    beta=beta,
                    K_theta_0=leader_policy.K.copy(),
                    sigma_leader=PARAMS["leader_sigma"],
                    Sigma_0=np.outer(np.array(_ENV_INIT_KWARGS["initial_state"]), np.array(_ENV_INIT_KWARGS["initial_state"])),
                    n_irl_iters=200,
                    lr_irl=0.01,
                    verbose=verbose,
                )
                if verbose:
                    print(
                        f"  [FEM-IRL] Q_hat diag={np.diag(Q_hat).round(4)}, R_hat diag={np.diag(R_hat).round(4)}",
                    )
                surrogate_fitted = True

            # Step 4: re-solve the DARE with the estimated (A_hat, B_hat, C_hat, Q_hat, R_hat) and the current K_theta
            sol = solve_2pmg_lqr(A_hat, B_hat, C_hat, Q_hat, R_hat, gamma_F, leader_policy.K)
            Ks_f, Ka_f, bias_f, Sigma_f = _sol_to_follower_params(sol, m, beta)

        elif mode in ("irl_recursive_rls", "irl_unknown_recursive_rls"):
            # Mode that runs FEM-IRL every iteration, accumulates Σ_k with RLS, and updates ω̄_k.

            # Step 1: run the true follower with the current K_theta and collect expert trajectories
            sol_exp = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)
            Ks_exp, Ka_exp, b_exp, Sf_exp = _sol_to_follower_params(sol_exp, m, beta)

            collect_n_ep = n_ep * 10 if mode == "irl_unknown_recursive_rls" else n_ep
            _, expert_trajs = _collect_episodes_parallel(
                leader_policy,
                Ks_exp,
                Ka_exp,
                b_exp,
                Sf_exp,
                collect_n_ep,
                gamma_L,
                gamma_F,
                base_seed=seed * 300 + iteration,
                max_workers=max_w,
            )

            if mode == "irl_unknown_recursive_rls":
                if verbose and iteration % 10 == 0:
                    print("  [Dynamics Estimator] Estimating A, B, C via OLS...")
                A_hat, B_hat, C_hat = estimate_dynamics_lsq(expert_trajs, n_s, n_p, m)
            else:
                A_hat, B_hat, C_hat = _A, _B, _C

            # Step 2: local IRL → ω_k
            mu_expert_k = compute_expert_fev(expert_trajs, gamma_F)
            Q_k, R_k = fem_irl_loop(
                expert_mu=mu_expert_k,
                A=A_hat,
                B=B_hat,
                C=C_hat,
                gamma_F=gamma_F,
                beta=beta,
                K_theta_0=leader_policy.K.copy(),
                sigma_leader=PARAMS["leader_sigma"],
                Sigma_0=np.outer(
                    np.array(_ENV_INIT_KWARGS["initial_state"]),
                    np.array(_ENV_INIT_KWARGS["initial_state"]),
                ),
                n_irl_iters=PARAMS["rls_local_irl_iters"],
                lr_irl=PARAMS["rls_local_irl_lr"],
                verbose=False,
            )
            omega_k = _omega_from_diag_cost_2pmg(Q_k, R_k)

            # Step 3: compute Σ_k analytically
            Sigma_k = _2pmg_info_matrix_from_trajs(
                expert_trajs,
                A=A_hat,
                B=B_hat,
                C=C_hat,
                K_theta=leader_policy.K,
                omega=omega_k,
                gamma_F=gamma_F,
                n_s=n_s,
                m=m,
                beta=beta,
                reg=PARAMS["rls_reg"],
            )

            # Step 4: accumulate information from all samples (full-gradient method, so eigmin-based rejection was removed)
            assert rls_sigma_bar is not None and rls_v_acc is not None and rls_omega_bar is not None
            eigvals_k = np.linalg.eigvalsh(Sigma_k)

            rls_sigma_bar = rls_sigma_bar + Sigma_k
            rls_v_acc = rls_v_acc + Sigma_k @ omega_k

            # Step 4.5: Gradient Descent step for true common reward parameter
            eta = float(PARAMS.get("rls_gd_lr", 0.01))

            # Divide by (iteration + 1) so the gradient scale doesn't grow linearly over time
            grad_omega = 2.0 * (rls_sigma_bar @ rls_omega_bar - rls_v_acc) / (iteration + 1)

            # Gradient clipping to prevent explosive updates
            grad_norm = np.linalg.norm(grad_omega)
            if grad_norm > 10.0:
                grad_omega = grad_omega * (10.0 / grad_norm)

            rls_omega_bar = rls_omega_bar - eta * grad_omega

            # Ensure no NaNs/Infs propagate, keeping within a robust numerical bound
            rls_omega_bar = np.nan_to_num(rls_omega_bar, nan=0.0, posinf=10.0, neginf=-10.0)
            rls_omega_bar = np.clip(rls_omega_bar, -10.0, 10.0)

            Q_hat, R_hat = _diag_from_reward_param_2pmg(rls_omega_bar, n_s, m)

            # Step 5: solve the DARE with the estimated reward and update the follower parameters
            sol = solve_2pmg_lqr(A_hat, B_hat, C_hat, Q_hat, R_hat, gamma_F, leader_policy.K)
            Ks_f, Ka_f, bias_f, Sigma_f = _sol_to_follower_params(sol, m, beta)

            if verbose and iteration % 10 == 0:
                eigvals_bar = np.linalg.eigvalsh(rls_sigma_bar)
                print(
                    "  [IRL-RLS 2pmg] "
                    f"||ω_k||={np.linalg.norm(omega_k):.4f}, "
                    f"||ω̄||={np.linalg.norm(rls_omega_bar):.4f}, "
                    f"eigmin(Σ_k)={eigvals_k.min():.3e}, "
                    f"eigmin(Σ̄)={eigvals_bar.min():.3e}, "
                    f"Q_hat diag={np.diag(Q_hat).round(6)}, "
                    f"R_hat diag={np.diag(R_hat).round(6)}",
                )

        elif mode == "non_irl":
            # For non_irl baseline, we do not estimate the follower's cost function or behavior via OLS/IRL.
            # We simply let the follower act according to their true optimal policy.
            # However, the influence/benefit term (Term 3) will be skipped during gradient update.
            sol = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)
            Ks_f, Ka_f, bias_f, Sigma_f = _sol_to_follower_params(sol, m, beta)

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # ---- Parallel episode collection ----
        pool, trajectories = _collect_episodes_parallel(
            leader_policy,
            Ks_f,
            Ka_f,
            bias_f,
            Sigma_f,
            n_ep,
            gamma_L,
            gamma_F,
            base_seed=iteration,
            max_workers=max_w,
        )

        # ---- STDPG Leader update ----
        if PARAMS.get("full_batch_cumulative", False):
            cumulative_trajectories.extend(trajectories)
            ac_trajectories = cumulative_trajectories
        else:
            ac_trajectories = trajectories

        mean_ret, debug_trace = stdpg_actor_critic_update(
            leader_policy=leader_policy,
            leader_critic=lc_net,
            leader_target_critic=leader_target_critic,
            lc_opt=lc_opt,
            trajectories=ac_trajectories,
            Ks_f=Ks_f,
            Ka_f=Ka_f,
            bias_f=bias_f,
            Sigma_f=Sigma_f,
            learning_rate=lr,
            max_update_norm=max_upd,
            beta=beta,
            gamma_L=gamma_L,
            gamma_F=gamma_F,
            mode=mode,
            A=_A if mode not in ("irl_unknown_dynamics", "irl_unknown_recursive_rls") else A_hat,
            C=_C if mode not in ("irl_unknown_dynamics", "irl_unknown_recursive_rls") else C_hat,
            W_hat=_W if mode == "softvi" else Q_hat,
            sol_f=sol
            if mode
            in ("softvi", "irl_known_dynamics", "irl_unknown_dynamics", "irl_recursive_rls", "irl_unknown_recursive_rls")
            else None,
            iteration=iteration,
        )
        std_ret = float(np.std([t["total_leader_return"] for t in trajectories]))
        true_sol = solve_2pmg_lqr(_A, _B, _C, _W, _U, gamma_F, leader_policy.K)
        true_Ks_f, true_Ka_f, true_b_f, true_Sigma_f = _sol_to_follower_params(true_sol, m, beta)

        # Collect evaluation episodes (e.g. 10 episodes, following the paper's experimental setup)
        _, eval_trajectories = _collect_episodes_parallel(
            leader_policy,
            true_Ks_f,
            true_Ka_f,
            true_b_f,
            true_Sigma_f,
            n_episodes=10,
            discount_leader=gamma_L,
            discount_follower=gamma_F,
            base_seed=iteration * 10000,
            max_workers=max_w,
        )

        mean_ret_true = float(np.mean([t["total_leader_return"] for t in eval_trajectories]))
        std_ret_true = float(np.std([t["total_leader_return"] for t in eval_trajectories]))

        stats["leader_return"].append(mean_ret_true)
        stats["leader_return_std"].append(std_ret_true)
        stats["eval_trajectories"] = eval_trajectories
        if mean_ret_true > stats["best_leader_return"]:
            stats["best_leader_return"] = mean_ret_true
            stats["best_iteration"] = iteration
            stats["best_eval_trajectories"] = eval_trajectories
            stats["best_leader_K"] = leader_policy.K.copy()

        if PARAMS.get("debug_trace", False):
            trace_row = {
                "iteration": int(iteration),
                "mode": mode,
                "mean_ret_sample": float(mean_ret),
                "mean_ret_eval": float(mean_ret_true),
                "std_ret_eval": float(std_ret_true),
                "ac_batch_trajs": int(len(ac_trajectories)),
            }
            trace_row.update(debug_trace)
            stats["debug_trace"].append(trace_row)

            log_every = int(PARAMS.get("debug_trace_every", 10))
            if log_every > 0 and iteration % log_every == 0:
                print(
                    "  [Trace] "
                    f"iter={iteration} | mode={trace_row['benefit_mode']} | "
                    f"ret_eval={trace_row['mean_ret_eval']:.4f} | "
                    f"batch={trace_row['ac_batch_trajs']} trajs | "
                    f"std_benefit={trace_row['std_benefit']:.6f} | "
                    f"|benefit|={trace_row['mean_abs_benefit']:.6f} | "
                    f"g1={trace_row['grad1_norm_mean']:.6f}, g2={trace_row['grad2_norm_mean']:.6f}, "
                    f"infl={trace_row['influence_norm_mean']:.6f}, gradK={trace_row['gradK_norm']:.6f}",
                )

            if PARAMS.get("debug_trace_jsonl", False):
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                trace_path = out / f"trace_{mode}.jsonl"
                with open(trace_path, "a", encoding="utf-8") as tf:
                    tf.write(json.dumps(trace_row, ensure_ascii=True) + "\n")

        if verbose and iteration % 10 == 0:
            print(
                f"  iter {iteration:4d}: return = {mean_ret:9.4f} ± {std_ret:.4f} | K = {leader_policy.K.flatten().round(4)}",
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"stats_{mode}.pkl"
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    if verbose and stats["best_iteration"] is not None:
        print(
            f"=== Best iter {stats['best_iteration']} | "
            f"best return = {stats['best_leader_return']:.4f} | "
            f"K = {np.asarray(stats['best_leader_K']).flatten().round(4)} ===",
        )
    print(f"=== Saved to {path} ===")
    return stats


def merge_and_plot(output_dir: str) -> None:

    out = Path(output_dir)
    modes = {
        "softvi": "Oracle (SoftVI)",
        "irl_known_dynamics": "IRL (known dyn)",
        "irl_unknown_dynamics": "IRL (unknown dyn)",
        "irl_recursive_rls": "IRL (recursive RLS)",
        "irl_unknown_recursive_rls": "IRL (unk. dyn + RLS)",
        "non_irl": "Non-IRL Surrogate",
    }
    results = {}
    for key, name in modes.items():
        p = out / f"stats_{key}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                results[name] = pickle.load(f)
    if not results:
        print("No results found.")
        return
    main_name = "Oracle (SoftVI)" if "Oracle (SoftVI)" in results else list(results.keys())[0]
    main_stats = results.get(main_name)
    blines = {k: v for k, v in results.items() if k != main_name}

    plot_learning_curves(main_stats, save_path=out / "comparison_curves.pdf", baselines=blines, main_label=main_name)
    print(f"Plot saved to {out / 'comparison_curves.pdf'}")

    try:
        plot_leader_performance_only(
            main_stats,
            save_path=out / "leader_performance.pdf",
            baselines=blines,
            main_label=main_name,
        )
        print(f"Performance plot saved to {out / 'leader_performance.pdf'}")
    except Exception as e:
        print(f"Failed to plot performance: {e}")

    # --- add trajectory plots ---
    try:
        plot_eval_trajectories(results, save_path=out / "eval_trajectories.pdf")
        print(f"Trajectory plot saved to {out / 'eval_trajectories.pdf'}")
    except Exception as e:
        print(f"Failed to plot trajectories: {e}")


def plot_eval_trajectories(results: dict, save_path: Path):
    

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(results))]

    for i, (name, data) in enumerate(results.items()):
        trajs = data.get("best_eval_trajectories") or data.get("eval_trajectories")
        if not trajs:
            continue

        # Plot the first evaluation episode as the representative
        traj = trajs[0]
        s_arr = np.array(traj["s"])  # (T, n_s)
        time = np.arange(len(s_arr))

        axes[0].plot(time, s_arr[:, 0], label=name, color=colors[i], linewidth=2)
        axes[1].plot(time, s_arr[:, 1], color=colors[i], linewidth=2)

    axes[0].set_ylabel("Position (s[0])")
    axes[1].set_ylabel("Velocity (s[1])")
    axes[1].set_xlabel("Time Step")
    axes[0].legend(loc="upper right", fontsize="small")
    axes[0].set_title("Evaluation Trajectories (Final Iteration)")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2PMG Continuous training")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "softvi",
            "irl_known_dynamics",
            "irl_unknown_dynamics",
            "irl_recursive_rls",
            "irl_unknown_recursive_rls",
            "non_irl",
            "plot",
        ],
    )
    parser.add_argument("--output-dir", default="data/internal/2pmg_continuous")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--benefit-baseline-mode",
        choices=["online-online", "online-target", "target-target"],
        default="online-online",
    )
    parser.add_argument("--debug-benefit-compare", action="store_true")
    parser.add_argument("--debug-benefit-log-every", type=int, default=10)
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--debug-trace-every", type=int, default=10)
    parser.add_argument("--debug-trace-jsonl", action="store_true")
    parser.add_argument("--full-batch-cumulative", action="store_true")
    args = parser.parse_args()
    PARAMS["benefit_baseline_mode"] = args.benefit_baseline_mode
    PARAMS["debug_benefit_compare"] = bool(args.debug_benefit_compare)
    PARAMS["debug_benefit_log_every"] = int(args.debug_benefit_log_every)
    PARAMS["debug_trace"] = bool(args.debug_trace)
    PARAMS["debug_trace_every"] = int(args.debug_trace_every)
    PARAMS["debug_trace_jsonl"] = bool(args.debug_trace_jsonl)
    PARAMS["full_batch_cumulative"] = bool(args.full_batch_cumulative)
    if args.mode == "plot":
        merge_and_plot(args.output_dir)
    else:
        run_experiment(args.mode, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
