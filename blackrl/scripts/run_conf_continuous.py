from __future__ import annotations

import argparse
import importlib
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from blackrl.algos.lqr_analytical import solve_confmdp_lqr
from blackrl.algos.conf_mdp_theorem1_gradient import theorem1_episode_gradient
from scipy.linalg import solve_discrete_lyapunov

_ENV_CLASS_NAME  = "BuildingThermalControlEnv"
_ENV_MODULE_PATH = "blackrl.envs.configurable_mdp_continuous_env"

_h_cc = 0.08 
_h_cr = 0.05 
_h_rc = 0.03 

_A8 = np.zeros((8, 8), dtype=np.float64)
_A8[0, 1] = _h_cc;  _A8[1, 0] = _h_cc
_A8[0, 2] = _h_cr;  _A8[2, 0] = _h_rc
_A8[0, 3] = _h_cr;  _A8[3, 0] = _h_rc
_A8[0, 5] = _h_cr;  _A8[5, 0] = _h_rc
_A8[1, 4] = _h_cr;  _A8[4, 1] = _h_rc
_A8[1, 6] = _h_cr;  _A8[6, 1] = _h_rc
_A8[1, 7] = _h_cr;  _A8[7, 1] = _h_rc

_ENV_INIT_KWARGS: dict = {
    "num_zones":      8,
    "num_hvac_units": 4,
    "max_episode_steps": 100,
    "A": _A8,
    "B": np.array([
        [0.0, 0.0, 0.0, 0.0],   
        [0.0, 0.0, 0.0, 0.0],   
        [0.8, 0.0, 0.0, 0.0],   
        [0.0, 0.0, 0.0, 0.0],   
        [0.0, 0.8, 0.0, 0.0],   
        [0.0, 0.0, 0.8, 0.0],   
        [0.0, 0.0, 0.0, 0.0],   
        [0.0, 0.0, 0.0, 0.8],   
    ], dtype=np.float64),
    "Q": np.diag([0.1, 0.1, 8.0, 30.0, 8.0, 3.0, 5.0, 3.0]).astype(np.float64),
    "R": 0.01 * np.eye(4, dtype=np.float64),
    "insulation_effect": np.array(
        [0.01, 0.01, 0.06, 0.08, 0.06, 0.03, 0.04, 0.03], dtype=np.float64
    ),
    "adjacency": [
        (0, 1, 0),   
        (0, 2, 2),   
        (0, 3, 3),   
        (0, 5, 5),  
        (1, 4, 4),   
        (1, 6, 6),   
        (1, 7, 7),   
    ],
    "transition_std":        float(np.sqrt(0.02)),
    "init_temp_std":         5.0,
    "temperature_range":     (-30.0, 30.0),
    "hvac_range":            (-10.0, 10.0),
    "stability_weight":      1.0,
    "energy_weight":         0.5,
    "insulation_cost_weight": 1.0,
    "airflow_cost_weight":   1.0,
    "init_temp_mean":        np.zeros(8, dtype=np.float64),
}


def make_env():
    from blackrl.envs.configurable_mdp_continuous_env import BuildingThermalControlEnv
    return BuildingThermalControlEnv(**_ENV_INIT_KWARGS)

def _worker_episode_conf(
    env_class_name: str,
    env_module_path: str,
    env_init_kwargs: dict,
    theta: np.ndarray,  # (2n,)  leader action, constant for this episode
    Ks_f: np.ndarray,   # (m, n)
    bias_f: np.ndarray, # (m,)
    Sigma_f,            # (m, m) ndarray or None (deterministic)
    discount_leader: float,
    seed: int,
) -> dict:
    module  = importlib.import_module(env_module_path)
    env_cls = getattr(module, env_class_name)
    env     = env_cls(**env_init_kwargs)
    env.seed(seed)

    rng     = np.random.default_rng(seed)
    obs, _  = env.reset()
    s_list, b_list, rL_list = [], [], []
    stab_list, nrg_list, ins_list, air_list = [], [], [], []

    while True:
        s      = obs.astype(np.float64)
        b_mean = Ks_f @ s + bias_f
        b      = rng.multivariate_normal(b_mean, Sigma_f)

        step = env.step(theta, b)
        r_L  = step.env_info["leader_reward"]
        stab_list.append(step.env_info["stability"])
        nrg_list.append(step.env_info["energy_cost"])
        ins_list.append(step.env_info["insulation_cost"])
        air_list.append(step.env_info["airflow_cost"])

        s_list.append(s.tolist())
        b_list.append(b.tolist())
        rL_list.append(r_L)
        obs = step.observation
        if step.last:
            break

    total       = sum(discount_leader ** t * r for t, r in enumerate(rL_list))
    total_stab  = sum(discount_leader ** t * r for t, r in enumerate(stab_list))
    total_nrg   = sum(discount_leader ** t * r for t, r in enumerate(nrg_list))
    total_ins   = sum(discount_leader ** t * r for t, r in enumerate(ins_list))
    total_air   = sum(discount_leader ** t * r for t, r in enumerate(air_list))

    return {
        "theta":                 theta.tolist(),
        "s":                     s_list,
        "b":                     b_list,
        "r_L":                   rL_list,
        "total_leader_return":   float(total),
        "total_stability":       float(total_stab),
        "total_energy_cost":     float(total_nrg),
        "total_insulation_cost": float(total_ins),
        "total_airflow_cost":    float(total_air),
    }


def _collect_episodes_parallel(
    per_episode_params: list[tuple],
    discount_leader: float,
    base_seed: int,
    max_workers: int,
) -> tuple[dict, list]:
    n_ep  = len(per_episode_params)
    max_w = min(n_ep, max_workers)
    seeds = [base_seed * 100_000 + i for i in range(n_ep)]

    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {
            ex.submit(
                _worker_episode_conf,
                _ENV_CLASS_NAME, _ENV_MODULE_PATH, _ENV_INIT_KWARGS,
                per_episode_params[i][0],  # theta
                per_episode_params[i][1],  # Ks_f
                per_episode_params[i][2],  # bias_f
                per_episode_params[i][3],  # Sigma_f
                discount_leader, seeds[i],
            ): i
            for i in range(n_ep)
        }
        ordered = [None] * n_ep
        for fut in as_completed(futs):
            ordered[futs[fut]] = fut.result()

    pool  = {"s": [], "b": []}
    trajs = []
    for r in ordered:
        pool["s"].extend([np.asarray(s) for s in r["s"]])
        pool["b"].extend([np.asarray(b) for b in r["b"]])
        trajs.append({
            "theta":                 np.asarray(r["theta"]),
            "s":                     [np.asarray(s) for s in r["s"]],
            "b":                     [np.asarray(b) for b in r["b"]],
            "r_L":                   r["r_L"],
            "total_leader_return":   r["total_leader_return"],
            "total_stability":       r["total_stability"],
            "total_energy_cost":     r["total_energy_cost"],
            "total_insulation_cost": r["total_insulation_cost"],
            "total_airflow_cost":    r["total_airflow_cost"],
        })
    return pool, trajs


class ConstantEpisodeLeaderPolicy:

    def __init__(self, n_theta: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.mu = rng.uniform(0.5, 1.0, size=n_theta).astype(np.float64)

    def get_theta(self) -> np.ndarray:
        return np.clip(self.mu, 0.0, 1.0)

    def update(self, grad_mu: np.ndarray, lr: float) -> None:
        self.mu += lr * grad_mu
        self.mu  = np.clip(self.mu, 0.0, 1.0)


def _sol_to_follower_params_conf(sol: dict, m: int, beta: float):
    """Analytical solution → (Ks_f, bias_f, Sigma_f)."""
    Ks_f    = -sol["Ks"]
    bias_f  = np.zeros(m, dtype=np.float64)
    Sigma_f = (beta / 2.0) * sol["S_inv"]
    return Ks_f, bias_f, Sigma_f


# ---------------------------------------------------------------------------
# Rebuttal experiments: knobs and helpers
#   Exp 1 (suboptimal follower): --follower-riccati-iters T  (0 = exact = paper)
#   Exp 2 (feature misspec.):    --misspec-delta d           (0 = exact = paper)
#   Exp 3 (Oracle LR/clip):      --lr / --clip overrides
#   All defaults reproduce the original paper setup exactly.
# ---------------------------------------------------------------------------

REBUTTAL = {
    "follower_riccati_iters": 0,  # 0 = exact DARE; T>0 = T-step Riccati (myopic/suboptimal follower)
    "follower_gain_shrink": 0.0,  # alpha: behavioral gain is (1-alpha) K*; 0 = exact
    "misspec_delta": 0.0,         # off-diagonal coupling added to TRUE Q; IRL model stays diagonal
    # --- Under-specification (lowspec): remove features from the IRL model ---
    "lowspec_drop": 0,            # number of state-cost features removed from the model
    "lowspec_order": "excitation",# which ones: "excitation" | "cost" | "controllable"
    "lowspec_severity": 1.0,      # how wrong the frozen guess is (0 = correct, 1 = default 1.0)
    "lowspec_mask": None,         # bool array (n_s+m,), True = free parameter
    "lowspec_fixed_logw": None,   # values the dropped components are frozen at
    "lowspec_amount": None,       # measured irreducible misspecification (occupancy-weighted)
    # --- Over-specification (overspec): add cross-term features s_i s_j -----
    "overspec_pairs": (),         # tuple of (i, j) zone pairs added to the MODEL
    "overspec_param": "tanh",     # "tanh" (bounded, scale-matched) | "additive" (legacy)
}


# Candidate cross-term pairs for the over-specification experiment. These are
# *redundant* under the default setup (the true Q is diagonal, so their true
# coefficients are exactly zero) -- the point is to verify that the recovered
# coefficients collapse to zero and that truncation does not get confused.
_OVERSPEC_CANDIDATE_PAIRS = ((0, 1), (2, 3), (3, 4), (5, 7), (1, 6))

# Over-specification diagnostic counters (only ever touched when
# REBUTTAL["overspec_pairs"] is non-empty, so the default path is unaffected).
#
#   calls       : how many times a cross-term Q was assembled
#   fired       : how many of those needed the PSD eigenvalue floor
#   eigmin_last : smallest eigenvalue of Q before the last projection
#   eigmin_min  : smallest eigenvalue ever seen before projection
#
# Why this matters: the projection rebuilds Q from its eigen-decomposition, so
# it writes non-zero values into off-diagonal entries that were never declared
# as model features. If it fires, "the recovered cross coefficient is ~0" can no
# longer be read off Q_hat[i, j], and any degradation is a feedback-loop effect
# rather than an ill-conditioned Sigma_k.
_PSD_PROJ = {"calls": 0, "fired": 0, "eigmin_last": float("nan"),
             "eigmin_min": float("inf"),
             "calls_infomat": 0, "fired_infomat": 0}


def _cross_rho(pairs) -> float:
    """Safety factor for the cross-term bound, from the pair incidence degree.

    With bound_k = rho * min(q_i, q_j) for pair k = (i, j), row i of Q obeys

        sum_{j != i} |Q_ij| <= deg(i) * rho * q_i < q_i   iff   deg(i)*rho < 1,

    i.e. Q is strictly diagonally dominant with a positive diagonal, hence
    positive definite -- by construction, for ANY parameter value. Disjoint
    pairs give deg = 1 and rho = 0.99; the 5-pair candidate list has deg = 2 on
    the shared indices, giving rho = 0.495.
    """
    if not pairs:
        return 0.0
    deg: dict[int, int] = {}
    for (i, j) in pairs:
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    return 0.99 / float(max(deg.values()))


def _cross_bounds(pairs, q_diag) -> np.ndarray:
    """Per-pair bound on |Q_ij|, on the SAME multiplicative scale as the diagonal.

    The diagonal costs are parameterized multiplicatively (q = exp(-w), so
    dq/dw = -q), while the original cross terms were additive (dQ/dv = -1). On
    this task the diagonal costs span 1e-3..1e2, so the two Jacobians differ by
    orders of magnitude, the information matrix Sigma_k is dominated by the
    cross directions, and the eigenvalue truncation keeps the redundant
    directions while discarding the genuine ones.

    Scaling the bound by min(q_i, q_j) makes the cross Jacobian proportional to
    a diagonal cost as well, so both blocks live on one scale and a single
    absolute eigenvalue threshold is meaningful for both.

    ``q_diag`` must be the diagonal that is actually being assembled into Q (the
    caller's own clipping already applied), otherwise the diagonal-dominance
    guarantee above does not hold. The log_w -> Q -> omega -> Q round trip is
    therefore exact whenever the diagonal lies in the common [1e-3, 1e2] range,
    and saturates outside it exactly as the diagonal parameter itself does.
    """
    q = np.maximum(np.asarray(q_diag, dtype=np.float64), 0.0)
    rho = _cross_rho(pairs)
    return np.array([rho * min(float(q[i]), float(q[j])) for (i, j) in pairs],
                    dtype=np.float64)


def _cross_values(param, pairs, q_diag) -> np.ndarray:
    """Map free cross parameters to off-diagonal entries of Q.

    ``tanh`` (default): Q_ij = bound_k * tanh(param_k). Bounded by construction,
    so the PSD projection can never fire, and the true value 0 sits at the
    centre where the derivative is largest.

    ``additive`` reproduces the original unbounded behaviour; kept so the
    failure mode it produces stays reproducible.
    """
    param = np.asarray(param, dtype=np.float64)
    if REBUTTAL.get("overspec_param", "tanh") == "additive":
        return param
    return _cross_bounds(pairs, q_diag) * np.tanh(param)


def _cross_dvalues(param, pairs, q_diag) -> np.ndarray:
    """d(Q_ij)/d(param_k) for the mapping above."""
    param = np.asarray(param, dtype=np.float64)
    if REBUTTAL.get("overspec_param", "tanh") == "additive":
        return np.ones_like(param)
    return _cross_bounds(pairs, q_diag) * (1.0 - np.tanh(param) ** 2)


def _cross_to_param(vals, pairs, q_diag) -> np.ndarray:
    """Inverse of ``_cross_values`` (used when reading Q back into parameters)."""
    vals = np.asarray(vals, dtype=np.float64)
    if REBUTTAL.get("overspec_param", "tanh") == "additive":
        return vals
    bnd = np.maximum(_cross_bounds(pairs, q_diag), 1e-12)
    return np.arctanh(np.clip(vals / bnd, -1.0 + 1e-12, 1.0 - 1e-12))


def _psd_proj_snapshot() -> dict:
    """Read out and reset the per-iteration PSD-projection counters."""
    snap = {
        "calls": int(_PSD_PROJ["calls"]),
        "fired": int(_PSD_PROJ["fired"]),
        "calls_infomat": int(_PSD_PROJ["calls_infomat"]),
        "fired_infomat": int(_PSD_PROJ["fired_infomat"]),
        "eigmin_last": float(_PSD_PROJ["eigmin_last"]),
        "eigmin_min": (float(_PSD_PROJ["eigmin_min"])
                       if np.isfinite(_PSD_PROJ["eigmin_min"]) else float("nan")),
    }
    _PSD_PROJ["calls"] = 0
    _PSD_PROJ["fired"] = 0
    _PSD_PROJ["calls_infomat"] = 0
    _PSD_PROJ["fired_infomat"] = 0
    _PSD_PROJ["eigmin_min"] = float("inf")
    return snap


def _feature_excitation_order(env, theta: np.ndarray, gamma_F: float, beta: float) -> np.ndarray:
    """Rank state-cost features by how weakly they are excited under the
    follower's closed loop (ascending: least excited first).

    For a zero-mean Gaussian occupancy with second-moment matrix M, the
    feature s_i^2 has Var[s_i^2] = 2 M_ii^2, so the excitation ordering is the
    ordering of M_ii. Weakly excited features are the ones the follower
    regulates tightly (or that the leader's dynamics barely move), and they are
    exactly the directions the eigenvalue truncation removes first.
    """
    n_s = env.num_zones
    il, aa = theta[:n_s], theta[n_s:]
    A_theta = env.compute_parameterized_A(il, aa)
    sol = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
    Ks = sol["Ks"]
    A_cl = A_theta - env.B @ Ks
    Cov_b = (beta / 2.0) * sol["S_inv"]
    Q_noise = env.B @ Cov_b @ env.B.T + (env.transition_std ** 2) * np.eye(n_s)
    Q_lyap = (env.init_temp_std ** 2) * np.eye(n_s) + (gamma_F / (1.0 - gamma_F)) * Q_noise
    M = solve_discrete_lyapunov(np.sqrt(gamma_F) * A_cl, Q_lyap)
    excitation = 2.0 * np.diag(M) ** 2
    return np.argsort(excitation)  # ascending: least excited first


def _occupancy_moments(env, theta: np.ndarray, gamma_F: float, beta: float):
    """Discounted second moments of the follower's closed-loop occupancy.

    Returns (M, Mb): state and action second-moment matrices. For a zero-mean
    Gaussian occupancy the quadratic features have covariance
    Cov(s_i^2, s_j^2) = 2 M_ij^2, which is the inner product used to measure
    how much of the true reward a restricted model class can still express.
    """
    n_s = env.num_zones
    il, aa = theta[:n_s], theta[n_s:]
    A_theta = env.compute_parameterized_A(il, aa)
    sol = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
    Ks = sol["Ks"]
    A_cl = A_theta - env.B @ Ks
    Cov_b = (beta / 2.0) * sol["S_inv"]
    Q_noise = env.B @ Cov_b @ env.B.T + (env.transition_std ** 2) * np.eye(n_s)
    Q_lyap = (env.init_temp_std ** 2) * np.eye(n_s) + (gamma_F / (1.0 - gamma_F)) * Q_noise
    M = solve_discrete_lyapunov(np.sqrt(gamma_F) * A_cl, Q_lyap)
    Mb = Ks @ M @ Ks.T + (1.0 / (1.0 - gamma_F)) * Cov_b
    return M, Mb


def measure_lowspec_amount(env, dropped: np.ndarray, frozen_q: np.ndarray,
                           theta_ref: np.ndarray, gamma_F: float, beta: float) -> float:
    """Irreducible misspecification imposed by an under-specified model class.

    The dropped state-cost coefficients are stuck at ``frozen_q`` instead of
    their true values. The remaining free coefficients can partially absorb the
    discrepancy, because the quadratic features are correlated under the
    occupancy measure; the *irreducible* part is the residual after that
    compensation, i.e. the omitted-variable projection residual.

    Returned as an occupancy-weighted RMS reward error relative to the RMS of
    the true reward -- a dimensionless "amount of misspecification" in [0, ~1]
    that serves as the x-axis of the robustness curve (analogous to the TV
    distance used for the suboptimal-follower experiment).
    """
    n_s = env.num_zones
    M, Mb = _occupancy_moments(env, theta_ref, gamma_F, beta)
    C = 2.0 * (M ** 2)          # Cov of state features
    Cb = 2.0 * (Mb ** 2)        # Cov of action features
    q_true = np.diag(env.Q).copy()
    r_true = np.diag(env.R).copy()

    dropped = np.asarray(dropped, dtype=int)
    free = np.array([i for i in range(n_s) if i not in set(dropped.tolist())], dtype=int)

    # Coefficient error before compensation.
    d = np.zeros(n_s, dtype=np.float64)
    d[dropped] = q_true[dropped] - np.asarray(frozen_q, dtype=np.float64)

    # Best compensation by the free features under the occupancy inner product.
    resid = d.copy()
    if len(free) > 0:
        Cff = C[np.ix_(free, free)]
        coef = np.linalg.solve(
            Cff + 1e-12 * np.eye(len(free)), C[np.ix_(free, dropped)] @ d[dropped]
        )
        resid[free] -= coef

    rms_resid = float(np.sqrt(max(resid @ C @ resid, 0.0)))
    rms_reward = float(np.sqrt(max(q_true @ C @ q_true + r_true @ Cb @ r_true, 1e-300)))
    return rms_resid / rms_reward


def measure_span_misspec_amount(env, delta: float, theta_ref: np.ndarray,
                                gamma_F: float, beta: float) -> float:
    """Irreducible misspecification for the span-mismatch experiment (Exp 2a).

    The true reward carries off-diagonal terms -2 delta s_i s_j that the
    diagonal model cannot express. The diagonal coefficients can absorb part of
    them (the quadratic features are correlated under the occupancy measure);
    what remains is the irreducible part, reported on the same occupancy-weighted
    relative-RMS scale as ``measure_lowspec_amount`` so the two experiments are
    directly comparable.
    """
    n_s = env.num_zones
    M, Mb = _occupancy_moments(env, theta_ref, gamma_F, beta)
    C = 2.0 * (M ** 2)
    Cb = 2.0 * (Mb ** 2)
    q_true = np.diag(env.Q).copy()
    r_true = np.diag(env.R).copy()

    pairs = list(_MISSPEC_PAIRS)
    if not pairs or float(delta) == 0.0:
        return 0.0
    weights = np.array([_MISSPEC_COUPLING[i, j] for (i, j) in pairs], dtype=np.float64)

    # Cross-feature inner products under the Gaussian occupancy:
    #   <s_i s_j, s_p s_q> = M_ip M_jq + M_iq M_jp
    #   <s_i s_j, s_p^2>   = 2 M_ip M_jp
    n_c = len(pairs)
    Ccc = np.empty((n_c, n_c), dtype=np.float64)
    for a, (i, j) in enumerate(pairs):
        for b, (p, q) in enumerate(pairs):
            Ccc[a, b] = M[i, p] * M[j, q] + M[i, q] * M[j, p]
    Cdc = np.empty((n_s, n_c), dtype=np.float64)
    for d in range(n_s):
        for a, (i, j) in enumerate(pairs):
            Cdc[d, a] = 2.0 * M[d, i] * M[d, j]

    # True cross coefficients: reward contribution is -2 delta w_ij s_i s_j.
    c_coef = 2.0 * float(delta) * weights

    # Best compensation of the cross terms by the diagonal features.
    coef = np.linalg.solve(C + 1e-12 * np.eye(n_s), Cdc @ c_coef)
    # Residual squared norm: ||c - projection||^2 in the occupancy inner product
    sq = float(c_coef @ Ccc @ c_coef - 2.0 * coef @ (Cdc @ c_coef) + coef @ C @ coef)
    rms_resid = float(np.sqrt(max(sq, 0.0)))

    # Reward RMS includes the true cross terms as well.
    rms_reward = float(np.sqrt(max(
        q_true @ C @ q_true + r_true @ Cb @ r_true
        + c_coef @ Ccc @ c_coef + 2.0 * q_true @ Cdc @ c_coef,
        1e-300,
    )))
    return rms_resid / rms_reward


def setup_lowspec(env, n_drop: int, order: str, theta_ref: np.ndarray,
                  gamma_F: float, beta: float, severity: float = 1.0) -> None:
    """Under-specify the IRL model by freezing ``n_drop`` state-cost features.

    The frozen components are held at a fixed guess the model cannot revise, so
    the model class can no longer represent the true reward: this is the "fewer
    features than necessary" regime.

    ``order`` selects which features are frozen:

      "excitation"   least-excited first  (what eigenvalue truncation itself
                     would remove -- the cheapest directions to lose)
      "cost"         largest true cost first (the most semantically important)
      "controllable" zones with direct HVAC actuation first

    ``severity`` in [0, 1] controls *how wrong* the frozen guess is, by
    interpolating the frozen log-weight between the true value (0) and the
    uninformative default cost 1.0 (1). This is what makes the sweep
    well-calibrated: freezing at the default outright is catastrophic for the
    dominant zone (its cost alone is ~80% of the reward) and negligible for the
    tightly regulated ones, so ``n_drop`` on its own cannot produce a gradual
    degradation curve. The resulting misspecification amount is measured and
    reported, and is the quantity to use as the x-axis.

    Action-cost features (b^2) are never dropped: without them the surrogate
    LQR problem is ill-posed (R must stay positive definite).
    """
    n_s, m = env.num_zones, env.num_hvac_units
    n_drop = int(n_drop)
    severity = float(severity)
    if n_drop <= 0:
        return
    if n_drop >= n_s:
        raise ValueError(f"lowspec_drop={n_drop} must be < n_zones={n_s}")
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"lowspec_severity={severity} must lie in [0, 1]")

    if order == "excitation":
        ranked = _feature_excitation_order(env, theta_ref, gamma_F, beta)
    elif order == "cost":
        ranked = np.argsort(-np.diag(env.Q))
    elif order == "controllable":
        controllable = np.array([1 if np.any(env.B[i] != 0) else 0 for i in range(n_s)])
        # controllable zones first, ties broken by descending cost
        ranked = np.lexsort((-np.diag(env.Q), -controllable))
    else:
        raise ValueError(f"unknown lowspec_order: {order!r}")

    dropped = np.asarray(ranked[:n_drop], dtype=int)
    # Dimension includes optional cross-term parameters (over-specification),
    # so the two misspecification experiments can be combined.
    dim = _logw_dim(env)
    mask = np.ones(dim, dtype=bool)
    mask[dropped] = False

    # Frozen guess: interpolate in log-cost space between the true value
    # (severity 0) and the uninformative default cost 1.0 (severity 1).
    q_true = np.diag(env.Q).copy()
    logw_true = -np.log(np.clip(q_true, 1e-8, None))
    fixed = np.zeros(dim, dtype=np.float64)
    fixed[:n_s] = (1.0 - severity) * logw_true
    frozen_q = np.exp(-fixed[dropped])

    amount = measure_lowspec_amount(env, dropped, frozen_q, theta_ref, gamma_F, beta)

    REBUTTAL["lowspec_drop"] = n_drop
    REBUTTAL["lowspec_order"] = order
    REBUTTAL["lowspec_severity"] = severity
    REBUTTAL["lowspec_mask"] = mask.tolist()
    REBUTTAL["lowspec_fixed_logw"] = fixed.tolist()
    REBUTTAL["lowspec_dropped_idx"] = dropped.tolist()
    REBUTTAL["lowspec_amount"] = float(amount)
    print(
        f"[lowspec] freezing {n_drop} state-cost feature(s) by '{order}' "
        f"(severity {severity:.2f}): zones {dropped.tolist()}, "
        f"true costs {q_true[dropped].round(2).tolist()} -> frozen at "
        f"{np.round(frozen_q, 3).tolist()}. Model dim {int(mask.sum())}/{dim}. "
        f"Irreducible misspecification = {amount:.4f}"
    )


def _apply_lowspec_mask(log_w: np.ndarray) -> np.ndarray:
    """Freeze the dropped components of the log-weight vector (no-op if off)."""
    mask = REBUTTAL.get("lowspec_mask")
    if mask is None:
        return log_w
    mask = np.asarray(mask, dtype=bool)
    fixed = np.asarray(REBUTTAL["lowspec_fixed_logw"], dtype=np.float64)
    out = np.asarray(log_w, dtype=np.float64).copy()
    out[~mask] = fixed[~mask]
    return out

# Off-diagonal coupling pattern for the span-mismatch experiment.
# Cross terms s_i s_j lie OUTSIDE the span of the diagonal quadratic features
# [s^2, b^2], so the IRL model class can no longer represent the true reward.
#
# The pairs are disjoint and the coupling is normalized by sqrt(Q_ii Q_jj), so
# each 2x2 block is [[q_i, d sqrt(q_i q_j)], [d sqrt(q_i q_j), q_j]], which is
# positive definite exactly for |d| < 1. That makes ``delta`` a dimensionless
# coupling strength on a fixed scale (an unnormalized coupling would be capped
# by the cheapest coupled zone, limiting the reachable misspecification).
_MISSPEC_PAIRS = ((2, 4), (3, 6), (5, 7))
_MISSPEC_COUPLING = np.zeros((8, 8), dtype=np.float64)
for _i, _j in _MISSPEC_PAIRS:
    _w = float(np.sqrt(_ENV_INIT_KWARGS["Q"][_i, _i] * _ENV_INIT_KWARGS["Q"][_j, _j]))
    _MISSPEC_COUPLING[_i, _j] = _w
    _MISSPEC_COUPLING[_j, _i] = _w


def apply_misspecification(delta: float) -> None:
    """Add off-diagonal coupling delta*C to the TRUE follower cost Q (Exp 2).

    The follower's true reward becomes -s'(Q_diag + delta*C)s - b'Rb, while
    the IRL model class remains diagonal quadratic -> the true reward is no
    longer in the span of the model features (feature misspecification).
    """
    if float(delta) == 0.0:
        return
    Q_new = _ENV_INIT_KWARGS["Q"] + float(delta) * _MISSPEC_COUPLING
    eigmin = float(np.linalg.eigvalsh(Q_new).min())
    if eigmin <= 1e-6:
        raise ValueError(
            f"misspec delta={delta} makes true Q non-PD (eigmin={eigmin:.2e})"
        )
    _ENV_INIT_KWARGS["Q"] = Q_new
    REBUTTAL["misspec_delta"] = float(delta)
    rel = float(delta) * np.linalg.norm(_MISSPEC_COUPLING) / np.linalg.norm(Q_new)
    print(
        f"[misspec] TRUE Q += {delta} * C (off-diag, eigmin={eigmin:.4f}, "
        f"rel. misspec ||dQ||/||Q|| = {rel:.4f}); IRL model remains diagonal."
    )
    # Occupancy-weighted irreducible amount, on the SAME scale as the
    # under-specification experiment, so the two curves share an x-axis.
    try:
        _env = make_env()
        _theta = np.full(2 * _env.num_zones, 0.75)
        REBUTTAL["misspec_amount"] = measure_span_misspec_amount(
            _env, float(delta), _theta,
            PARAMS["discount_follower"], PARAMS["beta"],
        )
        print(f"[misspec] irreducible misspecification = {REBUTTAL['misspec_amount']:.4f}")
    except Exception as exc:  # diagnostics must never break a run
        print(f"[misspec] amount measurement skipped ({exc})")


def solve_confmdp_lqr_early(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    gamma_F: float,
    n_iters: int,
) -> dict:
    """Early-stopped Riccati value iteration -> suboptimal follower (Exp 1).

    Runs the discounted Riccati recursion for ``n_iters`` steps from P = W
    instead of solving the DARE exactly. This equals the T-step finite-horizon
    (myopic) follower; as n_iters -> inf it converges to solve_confmdp_lqr.
    Returns the same dict structure as solve_confmdp_lqr.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    P = W.copy()
    for _ in range(int(n_iters)):
        S = U + gamma_F * (B.T @ P @ B)
        S_inv = np.linalg.inv(S)
        P = (
            W
            + gamma_F * (A.T @ P @ A)
            - gamma_F ** 2 * (A.T @ P @ B @ S_inv @ B.T @ P @ A)
        )
        P = 0.5 * (P + P.T)

    S = U + gamma_F * (B.T @ P @ B)
    S_inv = np.linalg.inv(S)
    Ks = gamma_F * S_inv @ B.T @ P @ A
    Pww = gamma_F * P - gamma_F ** 2 * P @ B @ S_inv @ B.T @ P

    return {
        "P": P,
        "S": S,
        "S_inv": S_inv,
        "Ks": Ks,
        "Pww": Pww,
        "gamma_F": gamma_F,
    }


def _solve_follower_behavior(
    A_theta: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    gamma_F: float,
) -> dict:
    """The follower's *behavioral* solve.

    Exact best response by default. Two suboptimality knobs are available; they
    apply only to the follower's actual behavior (expert demonstrations,
    training rollouts, evaluation rollouts), while each method's internal belief
    (the Oracle's exact solve, RE-HG's surrogate re-derivation) stays exact.

    ``follower_riccati_iters`` T: T-step Riccati recursion, i.e. a T-horizon
        myopic follower. Interpretable, but its severity depends strongly on the
        leader's configuration -- as the leader stabilizes the building the
        recursion converges faster and the follower becomes near-optimal, so the
        perturbation fades exactly as the leader converges.

    ``follower_gain_shrink`` alpha: the follower under-reacts, applying
        (1 - alpha) K* instead of the optimal gain (alpha = 1 means no control
        at all). The action noise is left at the optimal policy's covariance, so
        only the mean response is suboptimal. Severity is essentially
        configuration-independent, which makes it the knob to use when a
        *persistent* level of suboptimality is required.
    """
    T = int(REBUTTAL.get("follower_riccati_iters", 0) or 0)
    if T > 0:
        sol = solve_confmdp_lqr_early(A_theta, B, W, U, gamma_F, T)
    else:
        sol = solve_confmdp_lqr(A_theta, B, W, U, gamma_F)

    alpha = float(REBUTTAL.get("follower_gain_shrink", 0.0) or 0.0)
    if alpha != 0.0:
        sol = dict(sol)
        sol["Ks"] = (1.0 - alpha) * np.asarray(sol["Ks"], dtype=np.float64)
    return sol


def _policy_value_matrix(
    A_theta: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    gamma_F: float,
    Ks: np.ndarray,
) -> np.ndarray:
    """Discounted cost matrix of the linear policy b = -Ks s.

    Solves  P = (W + Ks' U Ks) + gamma (A - B Ks)' P (A - B Ks).

    Needed because neither suboptimality knob leaves a usable value matrix in
    the solver output: the early-stopped recursion returns an intermediate P
    that is not the value of the policy it induces, and the shrunk-gain policy
    keeps the optimal P. Evaluating the actual policy value is what makes the
    reported suboptimality gap meaningful.
    """
    A_cl = np.asarray(A_theta, dtype=np.float64) - np.asarray(B, dtype=np.float64) @ Ks
    cost = np.asarray(W, dtype=np.float64) + Ks.T @ np.asarray(U, dtype=np.float64) @ Ks
    P = solve_discrete_lyapunov(np.sqrt(gamma_F) * A_cl.T, 0.5 * (cost + cost.T))
    return 0.5 * (P + P.T)


def _gaussian_tv_mc(
    mu1: np.ndarray,
    L1: np.ndarray,
    logdet1: float,
    P1: np.ndarray,
    mu0: np.ndarray,
    logdet0: float,
    P0: np.ndarray,
    z: np.ndarray,
) -> float:
    """Monte-Carlo estimate of TV( N(mu1,C1) || N(mu0,C0) ).

    Uses the identity
        TV(p, q) = E_{x~q}[ max(p(x)/q(x) - 1, 0) ]
                 = 1 - E_{x~q}[ min(p(x)/q(x), 1) ],
    estimated here with the symmetric two-sided average of the equivalent
    forms under p and under q, which is unbiased and lower-variance than
    either one alone. Common random numbers ``z`` (shared across all states
    and all sweep settings) keep the estimate smooth across iterations.

    Parameters are pre-factorized: ``L1`` is the Cholesky factor of C1,
    ``P0``/``P1`` the precision matrices, ``logdet0``/``logdet1`` the log
    determinants. Returns a value in [0, 1].
    """
    # x ~ N(mu1, C1)  via the shared standard-normal draws z
    x = mu1 + z @ L1.T
    d1 = x - mu1
    d0 = x - mu0
    # log p1(x) - log p0(x)
    log_ratio = 0.5 * (
        logdet0 - logdet1
        + np.einsum("ij,jk,ik->i", d0, P0, d0)
        - np.einsum("ij,jk,ik->i", d1, P1, d1)
    )
    # E_{x~p1}[ max(1 - p0/p1, 0) ] = TV
    tv = float(np.mean(np.maximum(1.0 - np.exp(-log_ratio), 0.0)))
    return float(np.clip(tv, 0.0, 1.0))


def _follower_policy_divergence(
    sol_sub: dict,
    sol_exact: dict,
    states: list,
    beta: float,
    n_mc: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """Average TV distance (primary) and KL (auxiliary) between the follower's
    actual behavior g_tilde(.|s) and the exact best response g_dagger(.|s).

    Both are Gaussians N(-Ks s, (beta/2) S^{-1}) with state-dependent means and
    state-independent covariances. TV has no closed form for multivariate
    Gaussians, so it is estimated by Monte Carlo with common random numbers
    (the same standard-normal draws are reused for every state and every
    configuration, making the sweep curves comparable and smooth).

    TV is the primary x-axis for the suboptimal-follower experiment: it is a
    proper metric, symmetric, and bounded in [0, 1]. KL is reported alongside
    only as a diagnostic (it is asymmetric and unbounded, and its Pinsker
    bound saturates for strongly suboptimal followers).

    Returns (mean TV in [0,1], mean KL(g_tilde || g_dagger)).
    """
    K1 = np.asarray(sol_sub["Ks"], dtype=np.float64)
    C1 = (beta / 2.0) * np.asarray(sol_sub["S_inv"], dtype=np.float64)
    K0 = np.asarray(sol_exact["Ks"], dtype=np.float64)
    C0 = (beta / 2.0) * np.asarray(sol_exact["S_inv"], dtype=np.float64)
    m = C0.shape[0]

    if not states:
        return 0.0, 0.0

    # Symmetrize for numerical safety, then factorize once.
    C1 = 0.5 * (C1 + C1.T)
    C0 = 0.5 * (C0 + C0.T)
    L1 = np.linalg.cholesky(C1)
    P1 = np.linalg.inv(C1)
    P0 = np.linalg.inv(C0)
    _, logdet1 = np.linalg.slogdet(C1)
    _, logdet0 = np.linalg.slogdet(C0)

    # Common random numbers: identical draws for every state / setting.
    z = np.random.default_rng(seed).standard_normal((int(n_mc), m))

    # Closed-form KL constant part (covariance mismatch), reused per state.
    kl_const = 0.5 * (float(np.trace(P0 @ C1)) - m + float(logdet0 - logdet1))

    dK = K0 - K1  # mean difference: (-K1 s) - (-K0 s) = (K0 - K1) s
    tvs, kls = [], []
    for s in states:
        s = np.asarray(s, dtype=np.float64).ravel()
        mu1 = -(K1 @ s)
        mu0 = -(K0 @ s)
        tvs.append(
            _gaussian_tv_mc(mu1, L1, logdet1, P1, mu0, logdet0, P0, z)
        )
        dmu = dK @ s
        kls.append(kl_const + 0.5 * float(dmu @ P0 @ dmu))

    return float(np.mean(tvs)), float(np.mean(kls))


def _conf_quad_features(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Model features: [s^2, b^2] plus optional redundant cross terms s_i s_j.

    The cross terms are appended only when the over-specification experiment is
    enabled; their true coefficients are zero under the default (diagonal) true
    reward, so they are pure redundancy.
    """
    base = np.concatenate([s ** 2, b ** 2])
    pairs = REBUTTAL.get("overspec_pairs", ())
    if not pairs:
        return base
    cross = np.array([2.0 * s[i] * s[j] for (i, j) in pairs], dtype=np.float64)
    return np.concatenate([base, cross])


# The leader's cost matrices are _reward_scale() * exp(-w). The reward it
# recovers is r_F / beta, so with _reward_scale() = 1 / beta the log-weights w
# (and omega), their clips, initial value and step sizes keep exactly their
# original r_F-unit meaning. Any fixed scale is a valid parameterization; this
# one reproduces the tuned constants.
def _reward_scale() -> float:
    return 1.0 / float(PARAMS["beta"])


def _reward_matrices_from_logw(log_w: np.ndarray, n_s: int, m: int,
                               lo: float = 1e-8, hi: float = np.inf
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Build (Q, R) from log-weights, including optional cross-term entries.

    Diagonal entries use the positive parameterization diag = exp(-log_w),
    clamped to [lo, hi].

    The DEFAULT bounds reproduce the original per-iteration solver exactly
    (floor at 1e-8, no ceiling) and must stay that way: the local IRL's
    trajectory depends on the Q it feeds into the policy feature expectation,
    so imposing a ceiling here silently changes RE-HG's reward estimates, its
    information matrices, and hence the aggregation.

    The joint pooled solver passes the tighter range [1e-3, 1e2] -- the same
    dynamic range as ``_omega_from_diag_cost`` -- because it returns these
    matrices directly rather than round-tripping through that function, and
    without a ceiling a saturated log-weight (the iterate is clipped at +-20)
    yields a cost of exp(20) ~ 5e8 that destroys the surrogate LQR solve.

    Cross terms (over-specification) go through ``_cross_values``: bounded by
    min(q_i, q_j) times a tanh, which keeps Q strictly diagonally dominant (so
    positive definite for any parameter value) and puts the cross Jacobian on
    the same multiplicative scale as the diagonal one. The eigenvalue floor
    below is kept only as a safety net -- under the tanh map it must never fire,
    and ``_PSD_PROJ`` records whether that holds.
    """
    log_w = np.asarray(log_w, dtype=np.float64)
    q_diag = np.clip(np.exp(-log_w[:n_s]), lo, hi)
    Q = np.diag(q_diag)
    R = np.diag(np.clip(np.exp(-log_w[n_s:n_s + m]), lo, hi))

    pairs = REBUTTAL.get("overspec_pairs", ())
    if pairs:
        extra = _cross_values(log_w[n_s + m:], pairs, q_diag)
        for (i, j), v in zip(pairs, extra):
            Q[i, j] += float(v)
            Q[j, i] += float(v)
        # Project onto the PSD cone (floor eigenvalues) to keep DARE solvable.
        # Under the bounded 'tanh' map Q is diagonally dominant, hence PD, so the
        # floor is dropped to 0: it may only ever correct round-off. Keeping the
        # old 1e-4 floor here would fire spuriously, because the caller's
        # diagonal floor is 1e-8 and legitimate small costs sit below 1e-4.
        evals, evecs = np.linalg.eigh(0.5 * (Q + Q.T))
        floor = 1e-4 if REBUTTAL.get("overspec_param", "tanh") == "additive" else 0.0
        # Diagnostic: the projection rewrites *all* of Q along its eigenvectors,
        # not just the declared cross entries, so a firing projection invalidates
        # the "recovered cross coefficient" reading. Count it. Guarded by
        # ``pairs`` so the default RE-HG path never touches these counters.
        _PSD_PROJ["calls"] += 1
        _PSD_PROJ["eigmin_last"] = float(evals.min())
        _PSD_PROJ["eigmin_min"] = min(_PSD_PROJ["eigmin_min"], float(evals.min()))
        if evals.min() < floor:
            _PSD_PROJ["fired"] += 1
            Q = evecs @ np.diag(np.maximum(evals, floor)) @ evecs.T
            Q = 0.5 * (Q + Q.T)
    c = _reward_scale()
    return c * Q, c * R


def _logw_dim(env) -> int:
    """Dimension of the IRL model parameter vector (with optional cross terms)."""
    return env.num_zones + env.num_hvac_units + len(REBUTTAL.get("overspec_pairs", ()))


def compute_expert_fev_conf(
    expert_trajs: list[dict],
    gamma_F: float,
) -> np.ndarray:
    traj_fevs = []
    for traj in expert_trajs:
        s_list = traj["s"]
        b_list = traj["b"]
        phi0 = _conf_quad_features(np.asarray(s_list[0]), np.asarray(b_list[0]))
        traj_fev = np.zeros_like(phi0, dtype=np.float64)
        for t, (s, b) in enumerate(zip(s_list, b_list)):
            traj_fev += (gamma_F ** t) * _conf_quad_features(
                np.asarray(s), np.asarray(b)
            )
        traj_fevs.append(traj_fev)
    return np.mean(traj_fevs, axis=0)


def compute_policy_fev_conf(
    env,
    A_theta_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    gamma_F: float,
    beta: float,
) -> np.ndarray:
  
    sol = solve_confmdp_lqr(A_theta_ref, env.B, Q, R, gamma_F)


    n_s = A_theta_ref.shape[0]
    Ks    = sol["Ks"]      # (m, n)
    S_inv = sol["S_inv"]   # (m, m)


    A_cl = A_theta_ref - env.B @ Ks

    Cov_b = (beta / 2.0) * S_inv          
    Q_noise = (
        env.B @ Cov_b @ env.B.T           
        + (env.transition_std ** 2) * np.eye(n_s)
    )

    Sigma_0 = (env.init_temp_std ** 2) * np.eye(n_s)

    sq_g   = np.sqrt(gamma_F)
    Q_lyap = Sigma_0 + (gamma_F / (1.0 - gamma_F)) * Q_noise
    M = solve_discrete_lyapunov(sq_g * A_cl, Q_lyap)
    mu_s = np.diag(M)
    Mb   = Ks @ M @ Ks.T + (1.0 / (1.0 - gamma_F)) * Cov_b
    mu_b = np.diag(Mb)

    pairs = REBUTTAL.get("overspec_pairs", ())
    if pairs:
        # Feature 2 s_i s_j has expectation 2 M_ij under the same occupancy.
        mu_cross = np.array([2.0 * M[i, j] for (i, j) in pairs], dtype=np.float64)
        return np.concatenate([mu_s, mu_b, mu_cross])
    return np.concatenate([mu_s, mu_b])


def fem_irl_loop_conf(
    expert_mu: np.ndarray,
    env,
    A_theta_ref: np.ndarray,
    gamma_F: float,
    beta: float,
    n_irl_iters: int = 200,
    lr_irl: float = 0.01,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
  
    n_s = A_theta_ref.shape[0]
    m   = env.B.shape[1]
    dim = _logw_dim(env)
    log_w = _apply_lowspec_mask(np.zeros(dim, dtype=np.float64))
    n_pairs = len(REBUTTAL.get("overspec_pairs", ()))

    for i in range(n_irl_iters):
        Q, R = _reward_matrices_from_logw(log_w, n_s, m)

        mu_policy = compute_policy_fev_conf(
            env, A_theta_ref, Q, R, gamma_F, beta,
        )

        grad_reward = expert_mu - mu_policy
        # Chain rule (jac = -dQ/dparam throughout): diagonal entries use
        # diag = exp(-w); cross terms go through the bounded tanh map, whose
        # derivative is bound_k * sech^2(w_k). The bound depends on the diagonal
        # too, but that coupling term is proportional to tanh(w_k) and so
        # vanishes exactly at the true value w_k = 0; ignoring it leaves the
        # fixed point unchanged.
        jac = np.ones(dim, dtype=np.float64)
        jac[:n_s + m] = np.exp(-log_w[:n_s + m])
        if n_pairs:
            _pairs = REBUTTAL.get("overspec_pairs", ())
            jac[n_s + m:] = -_cross_dvalues(
                log_w[n_s + m:], _pairs, np.diag(Q) / _reward_scale()
            )
        grad_logw   = grad_reward * jac
        grad_logw   = np.clip(grad_logw, -50.0, 50.0)
        log_w      += lr_irl * grad_logw
        log_w       = np.clip(log_w, -20.0, 20.0)
        # Under-specification: frozen components never move (no-op if disabled).
        log_w       = _apply_lowspec_mask(log_w)

        if verbose and (i % 20 == 0 or i == n_irl_iters - 1):
            diff_norm = float(np.linalg.norm(expert_mu - mu_policy))
            q_diag = _reward_scale() * np.maximum(np.exp(-log_w[:n_s]), 1e-8)
            r_diag = _reward_scale() * np.maximum(np.exp(-log_w[n_s:n_s + m]), 1e-8)
            msg = (
                f"    [FEM-IRL conf] iter {i:4d}: ||μ_e - μ_π|| = {diff_norm:.6f} | "
                f"Q_diag={q_diag.round(3)}, R_diag={r_diag.round(3)}"
            )
            if n_pairs:
                msg += f", cross={log_w[n_s + m:].round(4)}"
            print(msg)

    Q_hat, R_hat = _reward_matrices_from_logw(log_w, n_s, m)
    return Q_hat, R_hat


def fem_irl_loop_joint(
    env_data: list[tuple[np.ndarray, np.ndarray]],
    env,
    gamma_F: float,
    beta: float,
    n_irl_iters: int = 100,
    lr_irl: float = 0.01,
    warm_logw: np.ndarray | None = None,
    ridge: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint multi-environment FEM-IRL baseline (Xcsj-Q3 / meta-review).

    Estimates ONE shared reward directly from ALL accumulated demonstrations
    by minimizing the sum of per-environment feature-matching objectives:

        sum_k || mu_expert_k - mu_policy(Q, R; A_k) ||^2  ->  min over (Q, R)

    ``env_data`` is the accumulated list of (mu_expert_k, A_theta_k) pairs,
    one per leader iteration. No spectral aggregation, no standardization, no
    eigenvalue truncation: this is the "pool everything, solve once" strategy
    the reviewer proposes, re-solved every chunk of leader iterations.

    Two regularizers are available so the comparison can separate the effect of
    *pooling versus sequential aggregation* from the effect of *regularized
    versus unregularized estimation* -- without them the baseline differs from
    RE-HG in two ways at once. ``ridge`` adds a Tikhonov penalty; sweeping
    ``n_irl_iters`` gives early stopping, which is itself a spectral filter of
    the same family and costs nothing to implement. Report the baseline at its
    best setting.

    Note on the theoretical relation: any exact solution reproduces the expert
    in every environment, hence lies in the intersection of the per-environment
    IRL solution sets -- exactly the set characterized by Theorem 1. The two
    approaches share the same identification limit; they differ in finite-
    sample weighting (occupancy-weighted here vs. standardized in Algorithm 2)
    and in cost (re-solve over all data vs. O(1) recursive update).

    Warm-started with ``warm_logw`` from the previous chunk. Returns
    (Q_hat, R_hat, log_w).
    """
    n_s = env.num_zones
    m = env.B.shape[1]
    dim = _logw_dim(env)
    n_pairs = len(REBUTTAL.get("overspec_pairs", ()))
    if warm_logw is not None:
        log_w = np.asarray(warm_logw, dtype=np.float64).copy()
    else:
        log_w = np.zeros(dim, dtype=np.float64)
    log_w = _apply_lowspec_mask(log_w)

    best_logw, best_res = log_w.copy(), np.inf

    for i in range(n_irl_iters):
        # Tighter range than the per-iteration solver: see the note in
        # _reward_matrices_from_logw. The joint solver needs the ceiling.
        Q, R = _reward_matrices_from_logw(log_w, n_s, m, lo=1e-3, hi=1e2)

        grad_reward = np.zeros(dim, dtype=np.float64)
        for mu_expert_k, A_k in env_data:
            mu_policy_k = compute_policy_fev_conf(
                env, A_k, Q, R, gamma_F, beta,
            )
            grad_reward += mu_expert_k - mu_policy_k
        grad_reward /= len(env_data)

        # Track the best residual for reporting only. Returning that iterate
        # was measured to be WORSE than returning the last one (recovery
        # correlation 0.52 vs 0.65 on noisy empirical moments): the residual is
        # computed against noisy feature expectations, so its minimizer fits
        # the noise, whereas the decaying step size acts as implicit
        # regularization. Non-monotonicity is expected with a single
        # environment anyway -- the solution set is then an affine subspace
        # (Prop. 1) and the iterate drifts along the unidentified directions.
        res = float(np.linalg.norm(grad_reward))
        if res < best_res:
            best_res, best_logw = res, log_w.copy()

        jac = np.ones(dim, dtype=np.float64)
        jac[:n_s + m] = np.exp(-log_w[:n_s + m])
        if n_pairs:
            jac[n_s + m:] = -_cross_dvalues(
                log_w[n_s + m:], REBUTTAL.get("overspec_pairs", ()),
                np.diag(Q) / _reward_scale(),
            )
        grad_logw = grad_reward * jac

        # Optional Tikhonov (ridge) regularization, for the fairness control
        # that isolates "pooling vs sequential" from "regularized vs not".
        # Ridge and the eigenvalue truncation are the soft and hard members of
        # the same family of spectral filters on the information matrix
        # (filter factors sigma^2/(sigma^2+lambda) and 1[sigma > eps]), and
        # they shrink toward the same point: the minimum-norm solve used by the
        # recursive aggregation sets the truncated components of omega to zero,
        # which is exactly where this penalty pulls.
        #
        # The strength is expressed RELATIVE to the data term. Because the step
        # is normalized, an absolute penalty would barely tilt the direction
        # (the data gradient is O(1e2) here), and its relative weight would
        # drift over the solve as the residual shrinks. With this scaling,
        # ridge = 0.1 always means "the shrinkage pull is a tenth of the data
        # pull", independent of the residual scale.
        if ridge > 0.0:
            wn = float(np.linalg.norm(log_w))
            if wn > 1e-8:
                scale = float(ridge) * float(np.linalg.norm(grad_logw)) / wn
                grad_logw = grad_logw - scale * log_w

        # Normalized, decaying steps. The raw feature-matching residual has
        # magnitude set by the state second moments (O(1e2) here), so a fixed
        # learning rate either saturates the iterate within a handful of steps
        # or crawls. To keep this baseline competitive rather than crippled we
        # step a fixed distance along the gradient direction, with a 1/sqrt(t)
        # decay so the iterate settles instead of oscillating around the
        # optimum (without the decay the residual was observed to rise again
        # mid-solve).
        gn = float(np.linalg.norm(grad_logw))
        if gn > 1e-12:
            step = lr_irl / np.sqrt(1.0 + i / 50.0)
            log_w += step * grad_logw / gn
        # Same range as _diag_from_reward_param's omega clip: beyond +-8 the
        # implied cost saturates the [1e-3, 1e2] clamp anyway, and letting the
        # iterate run to +-20 only makes the solve diverge.
        log_w = np.clip(log_w, -8.0, 8.0)
        log_w = _apply_lowspec_mask(log_w)

        if verbose and (i % 100 == 0 or i == n_irl_iters - 1):
            # This is the DATA residual. With ridge > 0 the solver descends a
            # penalized objective, so this number is not expected to decrease
            # monotonically -- trading fit for variance is the point.
            print(
                f"    [Joint-IRL] iter {i:4d}: ||mean residual|| = {res:.6f} "
                f"(best {best_res:.6f}) over {len(env_data)} envs"
            )

    Q_hat, R_hat = _reward_matrices_from_logw(log_w, n_s, m, lo=1e-3, hi=1e2)
    return Q_hat, R_hat, log_w


def _omega_from_diag_cost(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Map cost matrices to ω where diag = exp(-ω), with robust bounds.

    Keep local IRL outputs in the same dynamic range as _diag_from_reward_param
    so a single saturated IRL step does not destabilize recursive aggregation.
    Cross-term coefficients (over-specification) are carried through as-is.
    """
    Q = np.asarray(Q, dtype=np.float64) / _reward_scale()
    R = np.asarray(R, dtype=np.float64) / _reward_scale()
    q = np.clip(np.diag(Q), 1e-3, 1e2)
    r = np.clip(np.diag(R), 1e-3, 1e2)
    omega = np.concatenate([-np.log(q), -np.log(r)]).astype(np.float64)
    pairs = REBUTTAL.get("overspec_pairs", ())
    if pairs:
        # Invert the bounded map, then apply the same sign flip the diagonal
        # gets (omega is a NEGATIVE log-weight), so that
        # _diag_from_reward_param(_omega_from_diag_cost(Q)) reproduces Q.
        vals = np.array([Q[i, j] for (i, j) in pairs], dtype=np.float64)
        omega = np.concatenate([omega, -_cross_to_param(vals, pairs, q)])
    return omega


def _diag_from_reward_param(omega: np.ndarray, n_s: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack unconstrained ω into positive diagonal cost matrices.

    We use the same positive parameterization as FEM-IRL conf mode:
        Q_diag = exp(-ω_q), R_diag = exp(-ω_r)
    """
    omega = _apply_lowspec_mask(np.asarray(omega, dtype=np.float64))
    omega_q = np.clip(omega[:n_s], -8.0, 8.0)
    omega_r = np.clip(omega[n_s:n_s + m], -8.0, 8.0)
    q = np.clip(np.exp(-omega_q), 1e-3, 1e2)
    r = np.clip(np.exp(-omega_r), 1e-3, 1e2)
    Q, R = np.diag(q), np.diag(r)

    pairs = REBUTTAL.get("overspec_pairs", ())
    if pairs:
        cross = np.clip(omega[n_s + m:n_s + m + len(pairs)], -1e2, 1e2)
        # Sign flip mirrors the diagonal convention (omega is a negative
        # log-weight); the bound is built from the SAME clipped q that goes into
        # Q, so the result is strictly diagonally dominant and PD.
        vals = _cross_values(-cross, pairs, q)
        for (i, j), v in zip(pairs, vals):
            Q[i, j] += float(v)
            Q[j, i] += float(v)
        evals, evecs = np.linalg.eigh(0.5 * (Q + Q.T))
        floor = 1e-4 if REBUTTAL.get("overspec_param", "tanh") == "additive" else 0.0
        # Second projection site (surrogate Q used for Sigma_k and the HG).
        # Counted separately so a firing here can be told apart from the one in
        # ``_reward_matrices_from_logw``. Under 'tanh' it must never fire.
        _PSD_PROJ["calls_infomat"] += 1
        if evals.min() < floor:
            _PSD_PROJ["fired_infomat"] += 1
            Q = evecs @ np.diag(np.maximum(evals, floor)) @ evecs.T
            Q = 0.5 * (Q + Q.T)
    c = _reward_scale()
    return c * Q, c * R


def _conf_info_matrix_from_trajs(
    trajs: list[dict],
    A_theta: np.ndarray,
    B: np.ndarray,
    omega: np.ndarray,
    gamma_F: float,
    n_s: int,
    m: int,
    beta: float,
) -> np.ndarray:
    pairs = tuple(REBUTTAL.get("overspec_pairs", ()))
    dim = n_s + m + len(pairs)
    A_theta = np.asarray(A_theta, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)


    Q0, R0 = _diag_from_reward_param(omega, n_s, m)
    sol0 = solve_confmdp_lqr(A_theta, B, Q0, R0, gamma_F)
    S0 = np.asarray(sol0["S"], dtype=np.float64)
    S0_inv = np.asarray(sol0["S_inv"], dtype=np.float64)
    Ks0 = np.asarray(sol0["Ks"], dtype=np.float64)
    P0 = np.asarray(sol0["P"], dtype=np.float64)
    Cov_b = (beta / 2.0) * S0_inv
    A_cl = A_theta - B @ Ks0

    omega_q = np.clip(omega[:n_s], -8.0, 8.0)
    omega_r = np.clip(omega[n_s:n_s + m], -8.0, 8.0)
    q_raw = np.exp(-omega_q)
    r_raw = np.exp(-omega_r)
    c = _reward_scale()
    dq = c * np.where((q_raw > 1e-3) & (q_raw < 1e2), -q_raw, 0.0)
    dr = c * np.where((r_raw > 1e-3) & (r_raw < 1e2), -r_raw, 0.0)
    # Cross-term sensitivity, consistent with the map applied in
    # _diag_from_reward_param: Q_ij = bound_k * tanh(-omega_k), so
    # dQ_ij/domega_k = -bound_k * sech^2(omega_k). Under the legacy additive
    # parameterization this collapses to the old constant -1.
    if pairs:
        _q_clip = np.clip(q_raw, 1e-3, 1e2)
        _cross_om = np.clip(omega[n_s + m:n_s + m + len(pairs)], -1e2, 1e2)
        dcross = -c * _cross_dvalues(-_cross_om, pairs, _q_clip)
    else:
        dcross = np.zeros(0, dtype=np.float64)

    dS = np.zeros((dim, m, m), dtype=np.float64)
    dKs = np.zeros((dim, m, n_s), dtype=np.float64)
    sq_g = np.sqrt(gamma_F)
    for i in range(dim):
        dQ_i = np.zeros((n_s, n_s), dtype=np.float64)
        dR_i = np.zeros((m, m), dtype=np.float64)
        if i < n_s:
            dQ_i[i, i] = dq[i]
        elif i < n_s + m:
            j = i - n_s
            dR_i[j, j] = dr[j]
        else:
            # Cross-term parameter (over-specification), matching
            # _diag_from_reward_param's sign convention and bounded map.
            k = i - (n_s + m)
            p, q = pairs[k]
            dQ_i[p, q] = dcross[k]
            dQ_i[q, p] = dcross[k]

        # Envelope-theorem form for optimal LQR sensitivity.
        M_i = dQ_i + Ks0.T @ dR_i @ Ks0
        dP_i = solve_discrete_lyapunov(sq_g * A_cl.T, M_i)

        dS_i = dR_i + gamma_F * (B.T @ dP_i @ B)
        dS_inv_i = -S0_inv @ dS_i @ S0_inv
        dKs_i = gamma_F * (
            dS_inv_i @ B.T @ P0 @ A_theta
            + S0_inv @ B.T @ dP_i @ A_theta
        )
        dS[i] = dS_i
        dKs[i] = dKs_i

    Sigma_acc = np.zeros((dim, dim), dtype=np.float64)
    w_sum = 0.0

    for traj in trajs:
        for t, (s, b) in enumerate(zip(traj["s"], traj["b"])):
            s = np.asarray(s, dtype=np.float64).ravel()
            b = np.asarray(b, dtype=np.float64).ravel()
            mu_b = -(Ks0 @ s)
            w_t = float(gamma_F ** t)

            psi_obs = np.zeros(dim, dtype=np.float64)
            psi_mean = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                dSi = dS[i]
                dKsi = dKs[i]
                Mi = dSi @ Ks0 + S0 @ dKsi

                # ψ_i(s,b) = ∂Q_soft(s,b)/∂ω_i (b-dependent part)
                psi_obs[i] = -(
                    float(b @ dSi @ b)
                    + 2.0 * float(b @ Mi @ s)
                )

                # E_b[ψ_i(s,b)] under Gaussian follower policy
                term_quad = float(np.trace(dSi @ Cov_b) + mu_b @ dSi @ mu_b)
                term_cross = 2.0 * float(mu_b @ Mi @ s)
                psi_mean[i] = -(term_quad + term_cross)

            delta = psi_obs - psi_mean
            Sigma_acc += w_t * np.outer(delta, delta)
            w_sum += w_t

    Sigma_k = Sigma_acc / w_sum
    Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)

    return Sigma_k

def theorem1_update(
    leader_policy: ConstantEpisodeLeaderPolicy,
    trajs: list[dict],
    learning_rate: float,
    grad_clip_max_norm: float,
    normalize: bool = False,
) -> tuple[float, dict]:
    """Apply one clipped gradient-ascent step to the leader.

    Also returns clipping diagnostics: clipping is a positively homogeneous
    operation, so it rescales the update without changing its direction. The
    fraction of iterations in which it is active, together with the pre- and
    post-clip norms, is what shows that every method's effective step size is
    the same and that any performance difference comes from direction.

    ``normalize=True`` replaces clipping by full gradient normalization,
    g <- g / ||g||, so every step has exactly the same length regardless of the
    gradient magnitude. This is the strictest magnitude-only control: clipping
    leaves small gradients untouched, whereas normalization removes the
    magnitude channel entirely. Note that g/||g|| is homogeneous of degree zero
    and therefore preserves the direction exactly; the diagnostic
    ``cos_pre_post`` records this and should stay at 1.

    The default (normalize=False) reproduces the original behaviour exactly.
    """
    grad_mu = np.zeros_like(leader_policy.mu)
    for traj in trajs:
        grad_mu += np.asarray(traj["grad_theta"], dtype=np.float64)
    grad_mu /= len(trajs)

    grad_norm = float(np.linalg.norm(grad_mu))
    raw = grad_mu.copy()

    if normalize:
        # Unit-norm step. Guard against a numerically zero gradient.
        if grad_norm > 1e-12:
            grad_mu = grad_mu / grad_norm
        clipped = True          # the magnitude is always overridden
    else:
        clipped = grad_norm > grad_clip_max_norm
        if clipped:
            grad_mu = grad_mu * (grad_clip_max_norm / grad_norm)
    post_norm = float(np.linalg.norm(grad_mu))

    # Direction check: any positively homogeneous rescaling (clipping,
    # normalization, a smaller learning rate) leaves this at exactly 1.
    if grad_norm > 1e-12 and post_norm > 1e-12:
        cos_pre_post = float(raw @ grad_mu / (grad_norm * post_norm))
    else:
        cos_pre_post = float("nan")

    leader_policy.update(grad_mu, learning_rate)
    returns = [t["total_leader_return"] for t in trajs]
    info = {
        "grad_norm_preclip": grad_norm,
        "grad_norm_postclip": post_norm,
        "grad_clipped": 1.0 if clipped else 0.0,
        "effective_step": learning_rate * post_norm,
        "grad_normalized": 1.0 if normalize else 0.0,
        "cos_pre_post": cos_pre_post,
    }
    return float(np.mean(returns)), info


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

PARAMS = {
    "n_leader_iterations":      1000,
    "n_episodes_per_iteration":  100,  
    "n_eval_episodes":           50,   
    "learning_rate_leader":      1e-1,
    "discount_leader":           0.9,
    "discount_follower":         0.9,
    "beta":                      1e-1,
    "max_workers":               32,
    "grad_clip_max_norm":        1.0,       # iterative IRL + RLS mode
    "grad_normalize":            False,     # Exp 3b: unit-norm steps instead of clipping
    "influence_scale":           1.0,       # Exp 3c: multiply the HG influence term by beta
    "rls_eig_threshold":         1e-2,    # Sigma_k is in (r_F/beta)-units: 1/beta^2 = 100x the old 1e-4
    "rls_local_irl_iters":       200,
    "rls_local_irl_lr":          0.01,
    "rls_max_update_iters":      10,
    "tv_mc_samples":             4000,  # MC samples for the follower TV estimate (Exp 1)
    "joint_irl_chunk":           10,    # re-solve joint pooled IRL every M leader iterations
    "joint_irl_iters":           400,   # FEM iterations per joint re-solve (warm-started)
    "joint_irl_lr":              0.05,  # step length (gradient is normalized)
    "joint_irl_ridge":           0.0,   # Tikhonov penalty for the regularized joint arm
    "verbose":                   True,
}

def _evaluate_leader_mu(
    env,
    mu_theta: np.ndarray,
    n_eval_eps: int,
    discount_leader: float,
    gamma_F: float,
    beta: float,
    base_seed: int,
    max_workers: int,
) -> tuple[float, float, dict]:
    n_zones = env.num_zones
    n_hvac  = env.num_hvac_units
    il = mu_theta[:n_zones]
    aa = mu_theta[n_zones:]
    A_theta = env.compute_parameterized_A(il, aa)
    # Evaluation rolls out the follower's actual BEHAVIOR (early-stopped when
    # the suboptimal-follower experiment is enabled; exact otherwise).
    sol     = _solve_follower_behavior(A_theta, env.B, env.Q, env.R, gamma_F)
    Ks_f, bias_f, Sigma_f = _sol_to_follower_params_conf(sol, n_hvac, beta)

    per_ep = [
        (mu_theta, Ks_f, bias_f, Sigma_f)
        for _ in range(n_eval_eps)
    ]
    _, trajs = _collect_episodes_parallel(
        per_ep, discount_leader, base_seed=base_seed, max_workers=max_workers,
    )

    returns  = [t["total_leader_return"]     for t in trajs]
    stabs    = [t["total_stability"]         for t in trajs]
    nrgs     = [t["total_energy_cost"]       for t in trajs]
    inss     = [t["total_insulation_cost"]   for t in trajs]
    airs     = [t["total_airflow_cost"]      for t in trajs]

    components = {
        "eval_stability":         float(np.mean(stabs)),
        "eval_stability_std":     float(np.std(stabs)),
        "eval_energy_cost":       float(np.mean(nrgs)),
        "eval_energy_cost_std":   float(np.std(nrgs)),
        "eval_insulation_cost":   float(np.mean(inss)),
        "eval_insulation_cost_std": float(np.std(inss)),
        "eval_airflow_cost":      float(np.mean(airs)),
        "eval_airflow_cost_std":  float(np.std(airs)),
    }
    return float(np.mean(returns)), float(np.std(returns)), components


def run_experiment(mode: str, output_dir: str, seed: int = 42) -> dict:
    print(f"\n=== Conf Continuous | mode={mode} | seed={seed} ===")
    env = make_env()

    n_zones  = env.num_zones
    n_hvac   = env.num_hvac_units
    n_theta  = 2 * n_zones

    leader_policy = ConstantEpisodeLeaderPolicy(n_theta, seed=seed)

    gamma_F = PARAMS["discount_follower"]
    gamma_L = PARAMS["discount_leader"]
    lr      = PARAMS["learning_rate_leader"]
    n_ep    = PARAMS["n_episodes_per_iteration"]
    # beta of the TRUE follower: simulates it (demonstrations, evaluation) and
    # is used by the white-box arms that know the true reward (softvi, non_irl).
    # IRL arms never touch it: the recovered reward is r_F / beta, so their
    # follower model, IRL and hypergradient all run at temperature 1.
    beta    = PARAMS["beta"]
    beta_leader = 1.0
    verbose = PARAMS["verbose"]
    max_w   = PARAMS["max_workers"]

    stats = {
        "leader_return":          [], "leader_return_std":          [],
        "stability":              [], "stability_std":              [],
        "energy_cost":            [], "energy_cost_std":            [],
        "insulation_cost":        [], "insulation_cost_std":        [],
        "airflow_cost":           [], "airflow_cost_std":           [],
        "eval_leader_return":     [], "eval_leader_return_std":     [],
        "eval_stability":         [], "eval_stability_std":         [],
        "eval_energy_cost":       [], "eval_energy_cost_std":       [],
        "eval_insulation_cost":   [], "eval_insulation_cost_std":   [],
        "eval_airflow_cost":      [], "eval_airflow_cost_std":      [],
        "reward_Q_diag":          [],  # estimated Q diagonal at each iteration
        "reward_R_diag":          [],  # estimated R diagonal at each iteration
        "reward_omega_bar":       [],  # aggregated omega vector (RLS) or single estimate
        "reward_omega_k":         [],  # local omega_k (RLS only; None for other modes)
        "mode": mode,
        # Full experiment configuration for reproducibility of rebuttal runs.
        "params_config":   {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in PARAMS.items()},
        "rebuttal_config": dict(REBUTTAL),
        "true_Q":          np.asarray(env.Q, dtype=np.float64).tolist(),
        "true_R":          np.asarray(env.R, dtype=np.float64).tolist(),
    }

    # Surrogate / IRL state
    Q_hat: np.ndarray | None = None
    R_hat: np.ndarray | None = None
    surrogate_fitted = False
    rls_sigma_bar: np.ndarray | None = None
    rls_v_acc: np.ndarray | None = None
    omega_bar: np.ndarray | None = None
    rls_update_count: int = 0

    if mode == "irl_recursive_rls":
        dim = _logw_dim(env)
        rls_sigma_bar = np.zeros((dim, dim), dtype=np.float64)
        rls_v_acc = np.zeros(dim, dtype=np.float64)
        omega_bar = np.zeros(dim, dtype=np.float64)

    # Joint pooled IRL baseline state (Xcsj-Q3): accumulated per-environment
    # feature expectations and dynamics references, plus warm-start weights.
    joint_env_data: list[tuple[np.ndarray, np.ndarray]] = []
    joint_logw: np.ndarray | None = None

    # Save explicit iteration 0 performance before any leader update.
    init_eval_mean, init_eval_std, init_eval_comps = _evaluate_leader_mu(
        env, leader_policy.mu,
        n_eval_eps=PARAMS["n_eval_episodes"],
        discount_leader=gamma_L,
        gamma_F=gamma_F,
        beta=beta,
        base_seed=500_000 + seed * 10_000,
        max_workers=max_w,
    )
    stats["eval_leader_return"].append(init_eval_mean)
    stats["eval_leader_return_std"].append(init_eval_std)
    for k, v in init_eval_comps.items():
        stats[k].append(v)

    if verbose:
        print(
            f"  iter {0:4d} (pre-update eval): "
            f"eval={init_eval_mean:8.4f}±{init_eval_std:.4f} | "
            f"μ = {np.round(leader_policy.mu, 3)}"
        )

    for iteration in range(PARAMS["n_leader_iterations"]):

        theta = leader_policy.get_theta()
        thetas = [theta for _ in range(n_ep)]

        # ---- Build per-episode follower params ----
        per_ep = []

        if mode == "softvi":
            il      = theta[:n_zones]
            aa      = theta[n_zones:]
            A_theta = env.compute_parameterized_A(il, aa)
            # Oracle's BELIEF: exact best response under the true reward.
            sol     = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
            # Follower's actual BEHAVIOR (may be early-stopped in Exp 1).
            sol_beh = _solve_follower_behavior(A_theta, env.B, env.Q, env.R, gamma_F)
            Ks_f_i, bias_f_i, Sigma_f_i = _sol_to_follower_params_conf(sol_beh, n_hvac, beta)
            for _ in thetas:
                per_ep.append((theta, Ks_f_i, bias_f_i, Sigma_f_i))
            sol_follower  = sol
            sol_grad      = sol
            include_term3 = True
            beta_model    = beta

        elif mode == "irl_known_dynamics":
            if not surrogate_fitted:
                theta_c    = np.full(n_theta, 0.5)
                il_c, aa_c = theta_c[:n_zones], theta_c[n_zones:]
                A_c        = env.compute_parameterized_A(il_c, aa_c)
                # Expert demonstrations follow the follower's actual BEHAVIOR.
                sol_c      = _solve_follower_behavior(A_c, env.B, env.Q, env.R, gamma_F)
                Ks_exp_c, bias_exp_c, Sigma_exp_c = _sol_to_follower_params_conf(
                    sol_c, n_hvac, beta,
                )
                expert_ep = [
                    (theta_c, Ks_exp_c, bias_exp_c, Sigma_exp_c)
                ] * n_ep
                _, expert_trajs = _collect_episodes_parallel(
                    expert_ep, gamma_L,
                    base_seed=seed * 1000, max_workers=max_w,
                )
                mu_expert = compute_expert_fev_conf(expert_trajs, gamma_F)
                if verbose:
                    print(f"  [FEM-IRL conf] μ_expert = {mu_expert.round(4)}")
                if verbose:
                    print("  [FEM-IRL conf] Starting feature expectation matching...")
                Q_hat, R_hat = fem_irl_loop_conf(
                    expert_mu   = mu_expert,
                    env         = env,
                    A_theta_ref = A_c,
                    gamma_F     = gamma_F,
                    beta        = beta_leader,
                    n_irl_iters = PARAMS["rls_local_irl_iters"],
                    lr_irl      = 0.1,
                    verbose     = verbose,
                )
                if verbose:
                    print(
                        f"  [FEM-IRL conf] "
                        f"Q_hat diag={np.diag(Q_hat).round(4)}, "
                        f"R_hat diag={np.diag(R_hat).round(4)}"
                    )
                surrogate_fitted = True

            il       = theta[:n_zones]
            aa       = theta[n_zones:]
            A_theta  = env.compute_parameterized_A(il, aa)
            sol_hat  = solve_confmdp_lqr(A_theta, env.B, Q_hat, R_hat, gamma_F)
            Ks_hat, bias_hat, Sigma_hat = _sol_to_follower_params_conf(sol_hat, n_hvac, beta_leader)
            for _ in thetas:
                per_ep.append((theta, Ks_hat, bias_hat, Sigma_hat))
            sol_follower  = sol_hat
            sol_grad      = sol_hat
            include_term3 = True
            beta_model    = beta_leader

        elif mode == "irl_recursive_rls":
            il       = theta[:n_zones]
            aa       = theta[n_zones:]
            A_theta  = env.compute_parameterized_A(il, aa)

            if verbose:
                print(f"  [IRL-RLS conf] Updating reward parameters (iter {iteration})...")

            # Expert trajectories in the current environment: the follower's
            # actual BEHAVIOR (early-stopped in Exp 1; exact otherwise).
            sol_true = _solve_follower_behavior(A_theta, env.B, env.Q, env.R, gamma_F)
            Ks_exp, bias_exp, Sigma_exp = _sol_to_follower_params_conf(sol_true, n_hvac, beta)
            expert_ep = [(theta, Ks_exp, bias_exp, Sigma_exp)] * n_ep
            _, expert_trajs = _collect_episodes_parallel(
                expert_ep, gamma_L,
                base_seed=seed * 2000 + iteration, max_workers=max_w,
            )

            # Local IRL (current environment): FEM solve -> local reward estimate
            mu_expert_k = compute_expert_fev_conf(expert_trajs, gamma_F)
            Q_k, R_k = fem_irl_loop_conf(
                expert_mu=mu_expert_k,
                env=env,
                A_theta_ref=A_theta,
                gamma_F=gamma_F,
                beta=beta_leader,
                n_irl_iters=PARAMS["rls_local_irl_iters"],
                lr_irl=PARAMS["rls_local_irl_lr"],
                verbose=False,
            )
            omega_k = _omega_from_diag_cost(Q_k, R_k)

            # Information matrix proxy from trajectory feature sums
            Sigma_k = _conf_info_matrix_from_trajs(
                expert_trajs,
                A_theta=A_theta,
                B=env.B,
                omega=omega_k,
                gamma_F=gamma_F,
                n_s=n_zones,
                m=n_hvac,
                beta=beta_leader,
            )

            # Reward identification accumulation (thresholded covariance)
            assert rls_sigma_bar is not None and rls_v_acc is not None
            assert omega_bar is not None

            eigvals_k, eigvecs_k = np.linalg.eigh(Sigma_k)
            eig_threshold = float(PARAMS.get("rls_eig_threshold", 1e-2))
            mask_k = (eigvals_k >= eig_threshold).astype(np.float64)
            U_k = eigvecs_k @ np.diag(mask_k) @ eigvecs_k.T
            U_k = 0.5 * (U_k + U_k.T)

            # Truncation diagnostics (epsilon sensitivity / over-specification):
            # how many directions survive the threshold, and how ill-conditioned
            # the per-iteration information matrix is.
            stats.setdefault("Uk_rank", []).append(int(mask_k.sum()))
            stats.setdefault("Sigma_k_eigmin", []).append(float(eigvals_k.min()))
            stats.setdefault("Sigma_k_eigmax", []).append(float(eigvals_k.max()))
            stats.setdefault("Sigma_k_cond", []).append(
                float(eigvals_k.max() / (abs(eigvals_k.min()) + 1e-30))
            )
            stats.setdefault("Sigma_k_eigvals", []).append(eigvals_k.tolist())

            _t_rls = time.perf_counter()
            rls_update_count += 1
            rls_sigma_bar = rls_sigma_bar + U_k
            rls_v_acc = rls_v_acc + U_k @ omega_k

            omega_prev = omega_bar.copy()

            rls_sigma_bar_arr = np.asarray(rls_sigma_bar, dtype=np.float64)
            bar_u = rls_sigma_bar_arr
            omega_bar = np.linalg.lstsq(bar_u, rls_v_acc, rcond=None)[0]

            # Rank/eigen-spectrum trace of the aggregated matrix bar_U
            # (empirical check of the rank condition in Thm. 1 / Cor. 1:
            #  rank is non-decreasing since bar_U is a cumulative PSD sum).
            # Aggregation cost: O(1) per iteration, to be compared against the
            # joint pooled IRL baseline's re-solve time (joint_irl_solve_time).
            stats.setdefault("rls_update_time", []).append(
                time.perf_counter() - _t_rls
            )

            _bar_eigs = np.linalg.eigvalsh(bar_u)
            stats.setdefault("barU_eigvals", []).append(_bar_eigs.tolist())
            stats.setdefault("barU_rank", []).append(int(np.sum(_bar_eigs > 1e-8)))
            stats.setdefault("barU_eigmin_nonzero", []).append(
                float(_bar_eigs[_bar_eigs > 1e-8].min()) if np.any(_bar_eigs > 1e-8) else 0.0
            )

            Q_hat, R_hat = _diag_from_reward_param(omega_bar, n_zones, n_hvac)

            # Use aggregated reward estimate to compute follower response
            sol_hat  = solve_confmdp_lqr(A_theta, env.B, Q_hat, R_hat, gamma_F)
            Ks_hat, bias_hat, Sigma_hat = _sol_to_follower_params_conf(sol_hat, n_hvac, beta_leader)
            for _ in thetas:
                per_ep.append((theta, Ks_hat, bias_hat, Sigma_hat))
            sol_follower  = sol_hat
            sol_grad      = sol_hat
            include_term3 = True
            beta_model    = beta_leader

            if verbose:
                diff_omega = np.linalg.norm(omega_k - omega_prev)
                eig_u = np.linalg.eigvalsh(U_k)
                print(f"  [IRL-RLS Diagnostics] iter {iteration}")
                print(f"    Sigma_k eigmin: {eigvals_k.min():.2e}, eigmax: {eigvals_k.max():.2e}, cond: {eigvals_k.max()/(eigvals_k.min()+1e-12):.2e}")
                print(f"    U_k     eigmin: {eig_u.min():.2e}, eigmax: {eig_u.max():.2e}, rank: {np.linalg.matrix_rank(U_k)}")
                print(f"    ||omega_k||: {np.linalg.norm(omega_k):.4f}, ||omega_bar||: {np.linalg.norm(omega_bar):.4f}")
                print(f"    ||omega_k - omega_prev||: {diff_omega:.4f}")
                print(f"    bar_U condition number: {np.linalg.cond(bar_u):.4e}")
                print(f"    Q_hat diag={np.diag(Q_hat).round(4)}")
                print(f"    R_hat diag={np.diag(R_hat).round(4)}")

        elif mode == "irl_joint_pooled":
            il       = theta[:n_zones]
            aa       = theta[n_zones:]
            A_theta  = env.compute_parameterized_A(il, aa)

            # Expert trajectories: identical data budget to irl_recursive_rls.
            sol_beh_exp = _solve_follower_behavior(A_theta, env.B, env.Q, env.R, gamma_F)
            Ks_exp, bias_exp, Sigma_exp = _sol_to_follower_params_conf(sol_beh_exp, n_hvac, beta)
            expert_ep = [(theta, Ks_exp, bias_exp, Sigma_exp)] * n_ep
            _, expert_trajs = _collect_episodes_parallel(
                expert_ep, gamma_L,
                base_seed=seed * 2000 + iteration, max_workers=max_w,
            )

            # Accumulate this environment's feature expectation, then re-solve
            # the JOINT pooled IRL every `joint_irl_chunk` iterations.
            mu_expert_k = compute_expert_fev_conf(expert_trajs, gamma_F)
            joint_env_data.append((mu_expert_k, A_theta.copy()))

            _chunk = int(PARAMS.get("joint_irl_chunk", 10))
            _t_solve = 0.0
            if iteration % _chunk == 0 or Q_hat is None:
                _t0 = time.perf_counter()
                Q_hat, R_hat, joint_logw = fem_irl_loop_joint(
                    joint_env_data, env, gamma_F, beta_leader,
                    n_irl_iters=int(PARAMS.get("joint_irl_iters", 400)),
                    lr_irl=float(PARAMS.get("joint_irl_lr", 0.05)),
                    warm_logw=joint_logw,
                    ridge=float(PARAMS.get("joint_irl_ridge", 0.0)),
                    verbose=verbose,
                )
                _t_solve = time.perf_counter() - _t0
                if verbose:
                    print(
                        f"  [Joint-IRL] re-solved over {len(joint_env_data)} envs "
                        f"in {_t_solve:.2f}s | Q_hat diag={np.diag(Q_hat).round(4)}"
                    )
            stats.setdefault("joint_irl_solve_time", []).append(_t_solve)
            stats.setdefault("joint_irl_n_envs", []).append(len(joint_env_data))

            sol_hat  = solve_confmdp_lqr(A_theta, env.B, Q_hat, R_hat, gamma_F)
            Ks_hat, bias_hat, Sigma_hat = _sol_to_follower_params_conf(sol_hat, n_hvac, beta_leader)
            for _ in thetas:
                per_ep.append((theta, Ks_hat, bias_hat, Sigma_hat))
            sol_follower  = sol_hat
            sol_grad      = sol_hat
            include_term3 = True
            beta_model    = beta_leader

        elif mode == "non_irl":
            il      = theta[:n_zones]
            aa      = theta[n_zones:]
            A_theta = env.compute_parameterized_A(il, aa)
            sol     = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
            sol_beh = _solve_follower_behavior(A_theta, env.B, env.Q, env.R, gamma_F)
            Ks_f_i, bias_f_i, Sigma_f_i = _sol_to_follower_params_conf(sol_beh, n_hvac, beta)
            for _ in thetas:
                per_ep.append((theta, Ks_f_i, bias_f_i, Sigma_f_i))
            sol_follower  = sol
            sol_grad      = None
            include_term3 = False
            beta_model    = beta

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        _, trajs = _collect_episodes_parallel(
            per_ep, gamma_L, base_seed=iteration, max_workers=max_w,
        )

        il = theta[:n_zones]
        aa = theta[n_zones:]

        A_theta_true = env.compute_parameterized_A(il, aa)
        sol_true = solve_confmdp_lqr(A_theta_true, env.B, env.Q, env.R, gamma_F)

        # Exp 1 diagnostics: distance between the follower's actual behavior
        # and the exact best response, averaged over visited states.
        if (int(REBUTTAL.get("follower_riccati_iters", 0) or 0) > 0
                or float(REBUTTAL.get("follower_gain_shrink", 0.0) or 0.0) != 0.0):
            sol_beh_diag = _solve_follower_behavior(
                A_theta_true, env.B, env.Q, env.R, gamma_F
            )
            _div_states = [s for traj in trajs[:5] for s in traj["s"][:40]]
            f_tv, f_kl = _follower_policy_divergence(
                sol_beh_diag, sol_true, _div_states, beta,
                n_mc=PARAMS.get("tv_mc_samples", 4000), seed=0,
            )
            stats.setdefault("follower_tv", []).append(f_tv)
            stats.setdefault("follower_kl", []).append(f_kl)

            # Value-based suboptimality gap: the follower's own objective loss
            # relative to its optimum. Complements TV (behavioral distance) with
            # the quantity the theory's optimal-follower assumption is about.
            # Both policies are evaluated by solving for their actual value, so
            # the gap is exact for either knob.
            _S0 = (env.init_temp_std ** 2) * np.eye(n_zones)
            _P_ex = _policy_value_matrix(
                A_theta_true, env.B, env.Q, env.R, gamma_F,
                np.asarray(sol_true["Ks"], dtype=np.float64),
            )
            _P_sb = _policy_value_matrix(
                A_theta_true, env.B, env.Q, env.R, gamma_F,
                np.asarray(sol_beh_diag["Ks"], dtype=np.float64),
            )
            _v_ex = float(np.trace(_P_ex @ _S0))
            _v_sb = float(np.trace(_P_sb @ _S0))
            stats.setdefault("follower_value_gap_rel", []).append(
                abs(_v_sb - _v_ex) / max(abs(_v_ex), 1e-12)
            )

            if verbose and iteration % 10 == 0:
                print(
                    f"    [Subopt Follower] TV(g_beh, g_exact) = {f_tv:.4f} | "
                    f"KL (aux) = {f_kl:.4f} | rel. value gap = "
                    f"{stats['follower_value_gap_rel'][-1]:.4f}"
                )

        est_grads = []
        oracle_grads = []
        est_direct_grads, est_infl_grads = [], []
        ora_direct_grads, ora_infl_grads = [], []
        r_true_list = []
        r_est_list = []

        for traj in trajs:
            g_direct, g_infl = theorem1_episode_gradient(
                traj, il, aa, env,
                gamma_L, gamma_F, beta_model,
                sol_follower, sol_grad, include_term3,
                return_terms=True,
            )
            # Exp 3c: explicit rescaling of the influence term only. This is the
            # one realization of "matched regularization" that is operationally
            # well defined here: RE-HG's truncation is claimed to act by
            # suppressing this term, and beta lets a white-box run reproduce the
            # same term ratio directly. Unlike clipping, normalization or a
            # smaller learning rate, g_direct + beta*g_infl is NOT positively
            # homogeneous in the full gradient, so it does change the direction
            # -- which is exactly the effect we attribute to truncation.
            _beta = float(PARAMS.get("influence_scale", 1.0))
            grad = g_direct + g_infl if _beta == 1.0 else g_direct + _beta * g_infl
            traj["grad_theta"] = grad.tolist()
            est_grads.append(grad)
            est_direct_grads.append(g_direct)
            est_infl_grads.append(g_infl)

            go_direct, go_infl = theorem1_episode_gradient(
                traj, il, aa, env,
                gamma_L, gamma_F, beta,
                sol_true, sol_true, include_term3=True,
                return_terms=True,
            )
            oracle_grads.append(go_direct + go_infl)
            ora_direct_grads.append(go_direct)
            ora_infl_grads.append(go_infl)

            for s, b in zip(traj["s"], traj["b"]):
                r_true = -np.dot(s, env.Q @ s) - np.dot(b, env.R @ b)
                if Q_hat is not None:
                    # IRL estimates are in r_F / beta units.
                    r_true = r_true / beta
                Q_eval = Q_hat if Q_hat is not None else env.Q
                R_eval = R_hat if R_hat is not None else env.R
                r_est = -np.dot(s, Q_eval @ s) - np.dot(b, R_eval @ b)
                r_true_list.append(r_true)
                r_est_list.append(r_est)

        mean_est_grad = np.mean(est_grads, axis=0)
        mean_oracle_grad = np.mean(oracle_grads, axis=0)
        norm_est = float(np.linalg.norm(mean_est_grad))
        norm_oracle = float(np.linalg.norm(mean_oracle_grad))
        cosine_sim = float(np.dot(mean_est_grad.flatten(), mean_oracle_grad.flatten()) / (norm_est * norm_oracle)) if (norm_est > 1e-12 and norm_oracle > 1e-12) else 0.0

        r_true_arr = np.array(r_true_list)
        r_est_arr = np.array(r_est_list)
        var_diff = float(np.var(r_est_arr - r_true_arr))
        std_true, std_est = np.std(r_true_arr), np.std(r_est_arr)
        correlation = float(np.corrcoef(r_true_arr, r_est_arr)[0, 1]) if (std_true > 1e-8 and std_est > 1e-8) else 0.0

        stats.setdefault("hg_cosine_similarity", []).append(cosine_sim)
        stats.setdefault("hg_oracle_norm", []).append(norm_oracle)
        stats.setdefault("hg_estimated_norm", []).append(norm_est)
        stats.setdefault("reward_pearson_corr", []).append(correlation)
        stats.setdefault("reward_var_diff", []).append(var_diff)

        # ---- Term-wise HG diagnostics (Xcsj-Q2 / TReH-W3) -------------------
        # Eq. (5) is a SUM: direct (partial-derivative) term + influence term.
        # A steep follower reward inflates only the influence term, so the
        # ratio ||influence|| / ||direct|| controls the direction of the sum.
        # Gradient clipping is positively homogeneous: it rescales the sum and
        # leaves this ratio -- hence the direction -- unchanged.
        #
        # IMPORTANT, on interpreting the "oracle" fields below. They evaluate
        # the exact-reward HG formula on THIS run's trajectories, which in any
        # IRL mode are generated by the surrogate follower. They are therefore
        # off-policy for the Oracle and its influence term is inflated by
        # orders of magnitude (measured: ratio ~2e2 off-policy versus ~5e-1 for
        # a genuine `softvi` run on its own trajectories). Use them only for
        # the *cosine* comparisons, which are fair because both gradients are
        # evaluated on the same states. To compare term ratios ACROSS methods,
        # take `hg_term_ratio` from a `softvi` run and from an IRL run --
        # never `hg_term_ratio` against `hg_oracle_term_ratio` within one run.
        def _cos(u, v):
            nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            return float(np.dot(u, v) / (nu * nv)) if (nu > 1e-12 and nv > 1e-12) else 0.0

        m_est_dir = np.mean(est_direct_grads, axis=0)
        m_est_inf = np.mean(est_infl_grads, axis=0)
        m_ora_dir = np.mean(ora_direct_grads, axis=0)
        m_ora_inf = np.mean(ora_infl_grads, axis=0)

        n_est_dir = float(np.linalg.norm(m_est_dir))
        n_est_inf = float(np.linalg.norm(m_est_inf))
        n_ora_dir = float(np.linalg.norm(m_ora_dir))
        n_ora_inf = float(np.linalg.norm(m_ora_inf))

        stats.setdefault("hg_direct_norm", []).append(n_est_dir)
        stats.setdefault("hg_influence_norm", []).append(n_est_inf)
        stats.setdefault("hg_term_ratio", []).append(n_est_inf / (n_est_dir + 1e-12))
        stats.setdefault("hg_oracle_direct_norm", []).append(n_ora_dir)
        stats.setdefault("hg_oracle_influence_norm", []).append(n_ora_inf)
        stats.setdefault("hg_oracle_term_ratio", []).append(n_ora_inf / (n_ora_dir + 1e-12))
        # Direction agreement per term: is the influence term itself pointing
        # the right way, independently of its magnitude?
        stats.setdefault("hg_influence_cosine", []).append(_cos(m_est_inf, m_ora_inf))
        stats.setdefault("hg_direct_cosine", []).append(_cos(m_est_dir, m_ora_dir))

        if PARAMS.get("verbose", False):
            print(f"    [Diagnostics] HG Cosine Sim: {cosine_sim:.4f} | Norm(Est): {norm_est:.4f} | Norm(Oracle): {norm_oracle:.4f}")
            print(f"    [Diagnostics] Reward Pearson Corr: {correlation:.4f} | Var(Δr): {var_diff:.4f}")
            print(
                f"    [Term-wise] Est: ||dir||={n_est_dir:.3e} ||infl||={n_est_inf:.3e} "
                f"ratio={n_est_inf/(n_est_dir+1e-12):.3e} | "
                f"Oracle: ||dir||={n_ora_dir:.3e} ||infl||={n_ora_inf:.3e} "
                f"ratio={n_ora_inf/(n_ora_dir+1e-12):.3e}"
            )
            print(
                f"    [Term-wise] cos(infl_est, infl_oracle)={_cos(m_est_inf, m_ora_inf):.4f} | "
                f"cos(dir_est, dir_oracle)={_cos(m_est_dir, m_ora_dir):.4f}"
            )

        mean_ret, clip_info = theorem1_update(
            leader_policy, trajs, lr, PARAMS["grad_clip_max_norm"],
            normalize=bool(PARAMS.get("grad_normalize", False)),
        )
        std_ret  = float(np.std([t["total_leader_return"] for t in trajs]))

        stats["leader_return"].append(mean_ret)
        stats["leader_return_std"].append(std_ret)
        for _k, _v in clip_info.items():
            stats.setdefault(_k, []).append(_v)
        # Leader parameter trajectory: needed to tell "converged to a different
        # basin" apart from "did not converge" when comparing against Oracle.
        stats.setdefault("leader_theta", []).append(leader_policy.mu.tolist())

        _q_rec = np.diag(Q_hat).tolist() if Q_hat is not None else np.diag(env.Q).tolist()
        _r_rec = np.diag(R_hat).tolist() if R_hat is not None else np.diag(env.R).tolist()
        stats["reward_Q_diag"].append(_q_rec)
        stats["reward_R_diag"].append(_r_rec)

        # Over-specification diagnostics: the redundant cross-term coefficients
        # have TRUE value zero, so their recovered magnitude measures how much
        # spurious structure the redundant features absorb.
        _pairs = REBUTTAL.get("overspec_pairs", ())
        if _pairs:
            _Qe = Q_hat if Q_hat is not None else env.Q
            _cross = [float(_Qe[i, j]) for (i, j) in _pairs]
            stats.setdefault("overspec_cross_coeffs", []).append(_cross)
            stats.setdefault("overspec_cross_absmax", []).append(
                float(np.max(np.abs(_cross))) if _cross else 0.0
            )

            # Leakage: off-diagonal mass in entries that were NEVER declared as
            # model features. Under the intended behaviour this is exactly 0.
            # A non-zero value means the PSD projection rebuilt Q from its
            # eigenvectors and smeared the cross terms across the whole matrix,
            # which invalidates reading the recovered coefficients off Q_hat.
            _declared = {tuple(sorted(p)) for p in _pairs}
            _n = _Qe.shape[0]
            _leak = [abs(float(_Qe[i, j]))
                     for i in range(_n) for j in range(i + 1, _n)
                     if (i, j) not in _declared]
            stats.setdefault("overspec_offdiag_leak", []).append(
                max(_leak) if _leak else 0.0
            )

            # PSD-projection counters for this leader iteration.
            _psd = _psd_proj_snapshot()
            stats.setdefault("psd_proj_calls", []).append(_psd["calls"])
            stats.setdefault("psd_proj_fired", []).append(_psd["fired"])
            stats.setdefault("psd_proj_calls_infomat", []).append(_psd["calls_infomat"])
            stats.setdefault("psd_proj_fired_infomat", []).append(_psd["fired_infomat"])
            stats.setdefault("psd_proj_eigmin_min", []).append(_psd["eigmin_min"])

        if mode == "irl_recursive_rls" and omega_bar is not None:
            stats["reward_omega_bar"].append(omega_bar.tolist())
            stats["reward_omega_k"].append(omega_k.tolist())
        else:
            _unit = _reward_scale() if Q_hat is not None else 1.0
            _q_arr = np.array(_q_rec) / _unit
            _r_arr = np.array(_r_rec) / _unit
            _omega = np.concatenate([
                -np.log(np.clip(_q_arr, 1e-3, 1e2)),
                -np.log(np.clip(_r_arr, 1e-3, 1e2)),
            ])
            stats["reward_omega_bar"].append(_omega.tolist())
            stats["reward_omega_k"].append(None)


        if "total_stability" in trajs[0]:
            stabs = [t["total_stability"] for t in trajs]
            nrgs  = [t["total_energy_cost"] for t in trajs]
            inss  = [t["total_insulation_cost"] for t in trajs]
            airs  = [t["total_airflow_cost"] for t in trajs]

            stats["stability"].append(float(np.mean(stabs)))
            stats["stability_std"].append(float(np.std(stabs)))
            stats["energy_cost"].append(float(np.mean(nrgs)))
            stats["energy_cost_std"].append(float(np.std(nrgs)))
            stats["insulation_cost"].append(float(np.mean(inss)))
            stats["insulation_cost_std"].append(float(np.std(inss)))
            stats["airflow_cost"].append(float(np.mean(airs)))
            stats["airflow_cost_std"].append(float(np.std(airs)))

        eval_mean, eval_std, eval_comps = _evaluate_leader_mu(
            env, leader_policy.mu,
            n_eval_eps    = PARAMS["n_eval_episodes"],
            discount_leader = gamma_L,
            gamma_F       = gamma_F,
            beta          = beta,
            base_seed     = 1_000_000 + iteration * 1000,
            max_workers   = max_w,
        )
        stats["eval_leader_return"].append(eval_mean)
        stats["eval_leader_return_std"].append(eval_std)
        for k, v in eval_comps.items():
            stats[k].append(v)

        if verbose and iteration % 10 == 0:
            print(
                f"  iter {iteration:4d}: train={mean_ret:8.4f}±{std_ret:.4f} "
                f"eval={eval_mean:8.4f}±{eval_std:.4f} | "
                f"μ = {np.round(leader_policy.mu, 3)}"
            )

    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"stats_{mode}_seed{seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"=== Saved to {path} ===")
    return stats


_METRIC_KEYS = [
    "eval_leader_return",
    "eval_stability", "eval_energy_cost", "eval_insulation_cost", "eval_airflow_cost",
    "leader_return",
    "stability", "energy_cost", "insulation_cost", "airflow_cost",
    "hg_cosine_similarity", "hg_oracle_norm", "hg_estimated_norm",
    "reward_pearson_corr", "reward_var_diff",
    # Exp 1: suboptimal follower
    "follower_tv", "follower_kl", "follower_value_gap_rel",
    # Exp 3: clipping / effective step diagnostics
    "grad_norm_preclip", "grad_norm_postclip", "grad_clipped", "effective_step",
    "grad_normalized", "cos_pre_post",
    # Term-wise HG decomposition (Xcsj-Q2 / TReH-W3)
    "hg_direct_norm", "hg_influence_norm", "hg_term_ratio",
    "hg_oracle_direct_norm", "hg_oracle_influence_norm", "hg_oracle_term_ratio",
    "hg_influence_cosine", "hg_direct_cosine",
    # Exp 4 / 2c: truncation and identifiability
    "barU_rank", "barU_eigmin_nonzero", "Uk_rank",
    "Sigma_k_eigmin", "Sigma_k_eigmax", "Sigma_k_cond",
    # Exp 5: cost comparison against joint pooled IRL
    "joint_irl_solve_time", "joint_irl_n_envs", "rls_update_time",
    # Exp 2c: redundant coefficients should collapse to zero
    "overspec_cross_absmax",
    # Exp 2c: is a firing PSD projection contaminating the above?
    "overspec_offdiag_leak",
    "psd_proj_calls", "psd_proj_fired",
    "psd_proj_calls_infomat", "psd_proj_fired_infomat",
    "psd_proj_eigmin_min",
]


def aggregate_seeds(
    output_dir: str,
    mode: str,
    n_seeds: int = 10,
) -> dict:
    out = Path(output_dir)
    all_data: dict[str, list[list[float]]] = {k: [] for k in _METRIC_KEYS}

    loaded = 0
    for s in range(n_seeds):
        p = out / f"stats_{mode}_seed{s}.pkl"
        if not p.exists():
            print(f"[aggregate_seeds] {p} not found, skipping.")
            continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        for k in _METRIC_KEYS:
            if k in d:
                all_data[k].append(d[k])
        loaded += 1

    if loaded == 0:
        print(f"[aggregate_seeds] No data found for mode={mode}")
        return {}

    print(f"[aggregate_seeds] mode={mode}: loaded {loaded} seeds")

    result: dict = {"mode": mode, "n_seeds": loaded}
    for k in _METRIC_KEYS:
        if not all_data[k]:
            continue
        arr = np.array(all_data[k])  # shape: (n_seeds, n_iterations)

        result[f"{k}_mean"] = arr.mean(axis=0).tolist()
        result[f"{k}_std"]  = arr.std(axis=0).tolist()
    agg_path = out / f"aggregated_{mode}.pkl"
    with open(agg_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[aggregate_seeds] Saved aggregated result -> {agg_path}")
    return result


def merge_and_plot(output_dir: str, n_seeds: int = 10) -> None:
    from scripts.plot_learning_curves import plot_learning_curves, plot_components, plot_leader_performance_only
    out   = Path(output_dir)
    modes = {
        "softvi": "Oracle",
        "irl_known_dynamics": "RE-HG (Single)",
        "irl_recursive_rls": "RE-HG",
        "irl_joint_pooled": "Joint-IRL (pooled)",
        "non_irl": "No-Influence",
    }

    agg_results = {}
    for key, name in modes.items():
        agg = aggregate_seeds(str(out), key, n_seeds=n_seeds)
        if agg:
            agg_results[name] = agg

    if not agg_results:
        print("No aggregated results. Falling back to single-seed files.")
        _merge_and_plot_single_seed(output_dir)
        return

    def _to_plot_stats(agg: dict, use_eval: bool = True) -> dict:
        prefix = "eval_" if use_eval else ""
        return {
            "leader_return":        agg.get(f"{prefix}leader_return_mean", []),
            "leader_return_std":    agg.get(f"{prefix}leader_return_std",  []),
            "stability":            agg.get(f"{prefix}stability_mean",     []),
            "stability_std":        agg.get(f"{prefix}stability_std",      []),
            "energy_cost":          agg.get(f"{prefix}energy_cost_mean",   []),
            "energy_cost_std":      agg.get(f"{prefix}energy_cost_std",    []),
            "insulation_cost":      agg.get(f"{prefix}insulation_cost_mean", []),
            "insulation_cost_std":  agg.get(f"{prefix}insulation_cost_std",  []),
            "airflow_cost":         agg.get(f"{prefix}airflow_cost_mean",  []),
            "airflow_cost_std":     agg.get(f"{prefix}airflow_cost_std",   []),
            "mode":                 agg.get("mode", ""),
        }

    main_name  = "Oracle"
    main_stats = _to_plot_stats(agg_results[main_name]) if main_name in agg_results else \
                 _to_plot_stats(next(iter(agg_results.values())))
    blines     = {
        k: _to_plot_stats(v)
        for k, v in agg_results.items() if k != main_name
    }
    plot_learning_curves(main_stats, save_path=out / "comparison_curves.pdf", baselines=blines, main_label=main_name)
    plot_leader_performance_only(main_stats, save_path=out / "comparison_curves_leader_only.pdf", baselines=blines, main_label=main_name)
    plot_components(main_stats, save_path=out / "components_curves.pdf", baselines=blines, main_label=main_name)
    print(f"Plots saved to {out}")


def _merge_and_plot_single_seed(output_dir: str) -> None:
    from scripts.plot_learning_curves import plot_learning_curves, plot_components, plot_leader_performance_only
    out   = Path(output_dir)
    modes = {
        "softvi": "Oracle",
        "irl_known_dynamics": "RE-HG (Single)",
        "irl_recursive_rls": "RE-HG",
        "non_irl": "No-Influence",
    }
    results = {}
    for key, name in modes.items():
        for cand in [out / f"stats_{key}_seed0.pkl", out / f"stats_{key}.pkl"]:
            if cand.exists():
                with open(cand, "rb") as f:
                    results[name] = pickle.load(f)
                break
    if not results:
        print("No results found."); return
    main_stats = results.get("Oracle (SoftVI)", next(iter(results.values())))
    blines     = {k: v for k, v in results.items() if k != "Oracle (SoftVI)"}
    plot_learning_curves(main_stats, save_path=out / "comparison_curves.pdf", baselines=blines)
    plot_leader_performance_only(main_stats, save_path=out / "comparison_curves_leader_only.pdf", baselines=blines)
    plot_components(main_stats, save_path=out / "components_curves.pdf", baselines=blines)
    print(f"Plots saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Conf Continuous training")
    parser.add_argument("--mode", required=True,
                        choices=["softvi", "irl_known_dynamics", "irl_recursive_rls",
                                 "irl_joint_pooled", "non_irl", "plot", "aggregate"])
    parser.add_argument("--output-dir", default="data/internal/conf_continuous")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds to aggregate (for plot/aggregate mode)")
    # ---- Rebuttal experiment overrides (defaults reproduce the paper) ----
    parser.add_argument("--lr", type=float, default=None,
                        help="Override leader learning rate (Exp 3: Oracle LR sweep)")
    parser.add_argument("--clip", type=float, default=None,
                        help="Override gradient clip max norm (Exp 3: clip sweep)")
    parser.add_argument("--eig-threshold", type=float, default=None,
                        help="Override RLS eigenvalue truncation threshold epsilon")
    parser.add_argument("--follower-riccati-iters", type=int, default=0,
                        help="Exp 1a: T-step Riccati follower behavior (0 = exact best response). "
                             "Severity depends on the leader's configuration")
    parser.add_argument("--follower-gain-shrink", type=float, default=0.0,
                        help="Exp 1b: under-reacting follower, behavioral gain (1-alpha) K*. "
                             "Gives a persistent, configuration-independent level of "
                             "suboptimality (1.0 = no control at all)")
    parser.add_argument("--misspec-delta", type=float, default=0.0,
                        help="Exp 2a: off-diagonal coupling added to the TRUE follower cost Q "
                             "(span mismatch: true reward leaves the model span)")
    parser.add_argument("--lowspec-drop", type=int, default=0,
                        help="Exp 2b: number of state-cost features REMOVED from the IRL model "
                             "(under-specification)")
    parser.add_argument("--lowspec-order", type=str, default="excitation",
                        choices=["excitation", "cost", "controllable"],
                        help="Exp 2b: which features to remove first")
    parser.add_argument("--lowspec-severity", type=float, default=1.0,
                        help="Exp 2b: how wrong the frozen guess is, 0 (correct) to 1 "
                             "(uninformative default). Controls the amount of "
                             "under-specification continuously")
    parser.add_argument("--influence-scale", type=float, default=None,
                        help="Exp 3c: multiply the hypergradient's influence term (the second "
                             "term, carrying grad_theta Q_F) by this factor. beta<1 reproduces, "
                             "on a white-box run, the term-ratio suppression we attribute to "
                             "eigenvalue truncation. Default 1.0 leaves the gradient unchanged")
    parser.add_argument("--grad-normalize", action="store_true",
                        help="Exp 3b: normalize the hypergradient to unit norm before the "
                             "step, instead of clipping. Strictest magnitude-only control: "
                             "unlike clipping it also rescales small gradients, removing the "
                             "magnitude channel entirely")
    parser.add_argument("--overspec-pairs", type=int, default=0,
                        help="Exp 2c: number of redundant cross-term features s_i s_j ADDED to "
                             "the IRL model (over-specification; true coefficients are zero)")
    parser.add_argument("--overspec-param", type=str, default="tanh",
                        choices=["tanh", "additive"],
                        help="Exp 2c: cross-term parameterization. 'tanh' bounds Q_ij by "
                             "rho*min(q_i,q_j), keeping Q diagonally dominant (PD by "
                             "construction) and putting the cross Jacobian on the same "
                             "multiplicative scale as the diagonal. 'additive' is the legacy "
                             "unbounded map, kept to reproduce its failure mode")
    parser.add_argument("--n-iters", type=int, default=None,
                        help="Override number of leader iterations (smoke tests)")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Override episodes per iteration (smoke tests)")
    parser.add_argument("--n-eval-episodes", type=int, default=None,
                        help="Override evaluation episodes (smoke tests)")
    parser.add_argument("--joint-chunk", type=int, default=None,
                        help="Exp 5: re-solve joint pooled IRL every M leader iterations")
    parser.add_argument("--joint-iters", type=int, default=None,
                        help="Exp 5: FEM iterations per joint re-solve (warm-started)")
    parser.add_argument("--joint-lr", type=float, default=None,
                        help="Exp 5: joint solver step length (gradient is normalized)")
    parser.add_argument("--joint-ridge", type=float, default=None,
                        help="Exp 5: Tikhonov penalty for the regularized joint arm, "
                             "relative to the data term (0 = unregularized, 0.1 = the "
                             "shrinkage pull is a tenth of the data pull). Together "
                             "with --joint-iters (early stopping) this separates "
                             "'pooled vs sequential' from 'regularized vs not'")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    if args.lr is not None:
        PARAMS["learning_rate_leader"] = float(args.lr)
    if args.clip is not None:
        PARAMS["grad_clip_max_norm"] = float(args.clip)
    if args.influence_scale is not None:
        PARAMS["influence_scale"] = float(args.influence_scale)
        print(
            f"[influence-scale] HG influence term multiplied by "
            f"beta = {PARAMS['influence_scale']} (direct term unchanged)"
        )
    if args.grad_normalize:
        PARAMS["grad_normalize"] = True
        print(
            "[grad-norm] hypergradient normalized to unit norm each step "
            f"(clipping bypassed); effective step = lr = {PARAMS['learning_rate_leader']}"
        )
    if args.eig_threshold is not None:
        PARAMS["rls_eig_threshold"] = float(args.eig_threshold)
    if args.n_iters is not None:
        PARAMS["n_leader_iterations"] = int(args.n_iters)
    if args.n_episodes is not None:
        PARAMS["n_episodes_per_iteration"] = int(args.n_episodes)
    if args.n_eval_episodes is not None:
        PARAMS["n_eval_episodes"] = int(args.n_eval_episodes)
    if args.max_workers is not None:
        PARAMS["max_workers"] = int(args.max_workers)
    if args.joint_chunk is not None:
        PARAMS["joint_irl_chunk"] = int(args.joint_chunk)
    if args.joint_iters is not None:
        PARAMS["joint_irl_iters"] = int(args.joint_iters)
    if args.joint_lr is not None:
        PARAMS["joint_irl_lr"] = float(args.joint_lr)
    if args.joint_ridge is not None:
        PARAMS["joint_irl_ridge"] = float(args.joint_ridge)
    REBUTTAL["follower_riccati_iters"] = int(args.follower_riccati_iters)
    REBUTTAL["follower_gain_shrink"] = float(args.follower_gain_shrink)
    apply_misspecification(args.misspec_delta)
    if int(args.overspec_pairs) > 0:
        n_pairs = int(args.overspec_pairs)
        if n_pairs > len(_OVERSPEC_CANDIDATE_PAIRS):
            raise ValueError(
                f"--overspec-pairs at most {len(_OVERSPEC_CANDIDATE_PAIRS)}"
            )
        REBUTTAL["overspec_pairs"] = _OVERSPEC_CANDIDATE_PAIRS[:n_pairs]
        REBUTTAL["overspec_param"] = str(args.overspec_param)
        if REBUTTAL["overspec_param"] == "additive":
            # Legacy map, kept only to reproduce the failure it causes: the
            # diagonal costs are multiplicative (dQ/dw = -q, spanning 1e-3..1e2)
            # while these cross terms are additive (dQ/dw = -1). Sigma_k is then
            # dominated by the cross directions -- measured eigmax ~1.5e2 versus
            # ~1e-4 normally -- the truncation keeps them and discards the
            # genuine parameters, and the unbounded entries drive Q indefinite
            # so the PSD projection rewrites the diagonal as well.
            print(
                "[overspec] WARNING: --overspec-param additive uses the legacy "
                "unbounded cross terms; the spectral aggregation becomes "
                "ill-conditioned. Diagnostic use only."
            )
        _rho = _cross_rho(REBUTTAL["overspec_pairs"])
        print(
            f"[overspec] adding {n_pairs} redundant cross-term feature(s) to the IRL "
            f"model: {list(REBUTTAL['overspec_pairs'])} (true coefficients are 0); "
            f"param={REBUTTAL['overspec_param']}, rho={_rho:.4f}."
        )
    if int(args.lowspec_drop) > 0:
        _env_tmp = make_env()
        _theta_ref = np.full(2 * _env_tmp.num_zones, 0.75)
        setup_lowspec(
            _env_tmp, args.lowspec_drop, args.lowspec_order, _theta_ref,
            PARAMS["discount_follower"], PARAMS["beta"],
            severity=args.lowspec_severity,
        )

    if args.mode == "plot":
        merge_and_plot(args.output_dir, n_seeds=args.n_seeds)
    elif args.mode == "aggregate":
        for m in ["softvi", "irl_known_dynamics", "irl_recursive_rls",
                  "irl_joint_pooled", "non_irl"]:
            aggregate_seeds(args.output_dir, m, n_seeds=args.n_seeds)
    else:
        run_experiment(args.mode, args.output_dir, args.seed)


if __name__ == "__main__":
    main()