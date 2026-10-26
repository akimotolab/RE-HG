from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from blackrl.algos.lqr_analytical import solve_confmdp_lqr


# ---------------------------------------------------------------------------
# ∂A(θ)/∂θ  — Jacobian of the parameterised transition matrix
# ---------------------------------------------------------------------------

def _dA_dtheta(env) -> np.ndarray:
    n  = env.num_zones
    ie = env.insulation_effect  # (n,)
    dA = np.zeros((n * n, 2 * n), dtype=np.float64)

    if getattr(env, "adjacency", None) is not None:
        # ---- Hub (adjacency-list) topology ----
        h = env.A  # conductance matrix; diagonal = 0

        # Insulation: ∂A_θ[i,i] / ∂il[i] = +ie[i]
        for i in range(n):
            row_ii = i * n + i
            dA[row_ii, i] = ie[i]

        # Airflow: for each edge (ci, cj, aa_k) → column (n + aa_k)
        for ci, cj, k in env.adjacency:
            col_aa   = n + k
            row_ij   = ci * n + cj   # A_θ[ci, cj]
            row_ji   = cj * n + ci   # A_θ[cj, ci]
            row_ii   = ci * n + ci   # diagonal ci
            row_jj   = cj * n + cj   # diagonal cj

            dA[row_ij, col_aa]  = h[ci, cj]         # off-diag (ci→cj)
            dA[row_ji, col_aa]  = h[cj, ci]         # off-diag (cj→ci)
            dA[row_ii, col_aa] -= h[ci, cj]         # outflow from row ci
            dA[row_jj, col_aa] -= h[cj, ci]         # outflow from row cj

    else:
        # ---- Legacy ring topology (n=4, backward-compatible) ----
        A = env.A  # (n, n) full base matrix including diagonal

        for i in range(n):
            j_next = (i + 1) % n
            j_prev = (i - 1 + n) % n

            # ∂A_θ[i,i] / ∂il[i]  = -ie[i]
            row_ii = i * n + i
            dA[row_ii, i] = -ie[i]

            # ∂A_θ[i,i]      / ∂aa[i]  = -A[i, j_next]
            # ∂A_θ[i,j_next] / ∂aa[i]  = +A[i, j_next]
            # ∂A_θ[j_next,i] / ∂aa[i]  = +A[j_next, i]
            col_aa_i     = n + i
            row_ij_next  = i      * n + j_next
            row_jnext_i  = j_next * n + i

            dA[row_ii,      col_aa_i] -= A[i, j_next]
            dA[row_ij_next, col_aa_i]  = A[i, j_next]
            dA[row_jnext_i, col_aa_i]  = A[j_next, i]

            # ∂A_θ[i,i] / ∂aa[j_prev]  = -A[i, j_prev]
            col_aa_jp = n + j_prev
            dA[row_ii, col_aa_jp] -= A[i, j_prev]

    return dA


# ---------------------------------------------------------------------------
# ∂P(θ)/∂θ  — sensitivity of DARE solution via implicit function theorem
# ---------------------------------------------------------------------------

def _dP_dtheta_vec(
    A_theta: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    S_inv: np.ndarray,
    Ks: np.ndarray,
    gamma: float,
    dA_dtheta: np.ndarray,
) -> np.ndarray:
    n      = A_theta.shape[0]
    n_theta = dA_dtheta.shape[1]
    sq_g   = np.sqrt(gamma)

    # Closed-loop matrix A_cl = A(θ) - B Ks  (standard; DARE uses √γ A_cl)
    A_cl   = A_theta - B @ Ks           # (n, n)
    Abar   = sq_g * A_cl                # Lyapunov matrix: Abar' dP Abar - dP = RHS

    dP_dtheta = np.zeros((n * n, n_theta), dtype=np.float64)

    for k in range(n_theta):
        # dA_k = ∂A/∂θ_k  reshaped to (n, n)
        dA_k = dA_dtheta[:, k].reshape(n, n)


        T1 = gamma * dA_k.T @ P @ A_theta
        T2 = gamma * A_theta.T @ P @ dA_k
        T3 = gamma**2 * dA_k.T @ P @ B @ S_inv @ B.T @ P @ A_theta
        T4 = gamma**2 * A_theta.T @ P @ B @ S_inv @ B.T @ P @ dA_k

        dRHS = T1 + T2 - T3 - T4   # (n, n)

        dP_k = solve_discrete_lyapunov(Abar.T, dRHS)
        dP_dtheta[:, k] = dP_k.ravel()

    return dP_dtheta


# ---------------------------------------------------------------------------
# Leader value function  V_L(s; θ)  via closed-loop Lyapunov
# ---------------------------------------------------------------------------

def compute_leader_value_lyapunov(
    env,
    il: np.ndarray,
    aa: np.ndarray,
    A_theta: np.ndarray,
    Ks: np.ndarray,
    S_inv: np.ndarray,
    gamma_F: float,
    gamma_L: float,
    beta: float,
    Sigma_0: np.ndarray,
) -> np.ndarray:
    """Compute the (expected) discounted leader value matrix M_L.

    Under the closed-loop system:
        b_t = -Ks s_t + ε_b,   ε_b ~ N(0, (β/2) S⁻¹)
        s_{t+1} = A_cl s_t + B ε_b + ε_env

    The leader value is:
        V_L(s) = s' M_s s + const

    We compute M_s via the discounted Lyapunov equation:
        M_s = γ_L A_cl' M_s A_cl + Q_L_eff

    where Q_L_eff is the effective per-step leader state cost.

    NOTE: We return the matrix M_s and the noise covariance M_noise
    so that V_L and B_L can be evaluated at any (s, b).

    Returns
    -------
    dict with:
        M_s     : (n, n)  leader value matrix (state quadratic)
        A_cl    : (n, n)  closed-loop matrix
        Q_noise : (n, n)  per-step noise propagation to s
    """
    n     = A_theta.shape[0]
    B     = env.B
    A_cl  = A_theta - B @ Ks

    Cov_b   = (beta / 2.0) * S_inv
    Q_noise = (
        B @ Cov_b @ B.T
        + (env.transition_std ** 2) * np.eye(n)
    )

    n_z = env.num_zones
    Q_stab = env.stability_weight * (
        np.eye(n_z) / n_z - np.ones((n_z, n_z)) / (n_z ** 2)
    )
    Q_energy = env.energy_weight * (Ks.T @ Ks)
    Q_L_state = Q_stab + Q_energy

    sq_g = np.sqrt(gamma_L)

    M_s = solve_discrete_lyapunov(sq_g * A_cl.T, Q_L_state)

    c_const = (
        -env.energy_weight * np.trace(Cov_b)
        -env.insulation_cost_weight * np.sum(il ** 2)
        -env.airflow_cost_weight * np.sum(aa ** 2)
    )
    v_const = (c_const - gamma_L * float(np.trace(M_s @ Q_noise))) / (1.0 - gamma_L)

    return {"M_s": M_s, "V_const": v_const, "A_cl": A_cl, "Q_noise": Q_noise, "Cov_b": Cov_b}

def _grad_rL_theta(
    env,
    il: np.ndarray,
    aa: np.ndarray,
    energy_cost: float,
) -> np.ndarray:
    n = env.num_zones
    grad = np.zeros(2 * n, dtype=np.float64)
    grad[:n]  = -2.0 * env.insulation_cost_weight * il
    grad[n:]  = -2.0 * env.airflow_cost_weight * aa
    return grad

def _grad_log_p_theta(
    s: np.ndarray,
    b: np.ndarray,
    s_next: np.ndarray,
    env,
    A_theta: np.ndarray,
    dA_dtheta: np.ndarray,
) -> np.ndarray:
    sigma2 = env.transition_std ** 2
    residual = s_next - A_theta @ s - env.B @ b   # (n,)

    kron_vec = np.kron(residual, s)  # (n*n,)
    grad = kron_vec @ dA_dtheta / sigma2  # (2n,)
    return grad

def _grad_QF_theta_full(
    s: np.ndarray,
    b: np.ndarray,
    env_B: np.ndarray,
    A_theta: np.ndarray,
    sol: dict,
    dP_dtheta: np.ndarray,
    dA_dtheta: np.ndarray,
    gamma_F: float,
    sigma2: float,
    beta: float,
) -> np.ndarray:
    n       = A_theta.shape[0]
    n_theta = dP_dtheta.shape[1]
    B       = env_B
    P       = sol["P"]
    S_inv   = sol["S_inv"]

    As       = A_theta @ s   # (n,)
    BSinvBt  = B @ S_inv @ B.T   # (n, n)
    coeff_t6 = gamma_F * sigma2 / (1.0 - gamma_F)
    coeff_t7 = gamma_F**2 * beta / (2.0 * (1.0 - gamma_F))
    grad = np.zeros(n_theta, dtype=np.float64)

    for k in range(n_theta):
        dP_k = dP_dtheta[:, k].reshape(n, n)
        dA_k = dA_dtheta[:, k].reshape(n, n)

        t1 = gamma_F * float(b @ B.T @ dP_k @ B @ b)
        t2 = 2.0 * gamma_F * float(b @ B.T @ dP_k @ As)
        t3 = gamma_F * float(As @ dP_k @ As)

        dA_k_s = dA_k @ s   # (n,)
        t4 = 2.0 * gamma_F * float(b @ B.T @ P @ dA_k_s)
        t5 = 2.0 * gamma_F * float(As @ P @ dA_k_s)

        tr_dP_k        = float(np.trace(dP_k))
        tr_BSinvBt_dPk = float(np.trace(BSinvBt @ dP_k))
        t6 = coeff_t6 * tr_dP_k
        t7 = coeff_t7 * tr_BSinvBt_dPk

        grad[k] = t1 + t2 + t3 + t4 + t5 + t6 + t7

    return -grad

def _grad_log_pi_theta_full(
    s: np.ndarray,
    b: np.ndarray,
    env_B: np.ndarray,
    A_theta: np.ndarray,
    sol: dict,
    dP_dtheta: np.ndarray,
    dA_dtheta: np.ndarray,
    gamma_F: float,
    beta: float,
) -> np.ndarray:
    n       = A_theta.shape[0]
    n_theta = dP_dtheta.shape[1]
    B       = env_B
    P       = sol["P"]
    S       = sol["S"]
    S_inv   = sol["S_inv"]
    Ks      = sol["Ks"]

    e = b + Ks @ s
    grad = np.zeros(n_theta, dtype=np.float64)

    for k in range(n_theta):
        dP_k = dP_dtheta[:, k].reshape(n, n)
        dA_k = dA_dtheta[:, k].reshape(n, n)

        dS_k = gamma_F * B.T @ dP_k @ B
        dK_k = - S_inv @ dS_k @ Ks + gamma_F * S_inv @ B.T @ dP_k @ A_theta + gamma_F * S_inv @ B.T @ P @ dA_k

        term1 = float(e @ dS_k @ e)
        term2 = 2.0 * float(e @ S @ dK_k @ s)
        term3 = 0.5 * float(np.trace(S_inv @ dS_k))

        grad[k] = - (term1 + term2) / beta + term3

    return grad


def _leader_value(s: np.ndarray, M_s: np.ndarray, V_const: float) -> float:
    """V_L(s) = -s' M_s s + V_const."""
    return -float(s @ M_s @ s) + V_const

def theorem1_step_gradient(
    s: np.ndarray,
    b: np.ndarray,
    s_next: np.ndarray,
    il: np.ndarray,
    aa: np.ndarray,
    r_L: float,
    env,
    A_theta: np.ndarray,
    sol_grad: dict | None,         
    dA_dtheta: np.ndarray,
    dP_dtheta: np.ndarray | None,  
    M_s: np.ndarray,
    V_const: float,
    gamma_F: float,
    gamma_L: float,
    beta: float,
    include_term3: bool,
    return_terms: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """One-step HG contribution.

    With ``return_terms=True`` returns the pair (direct, influence) where

        direct    = g1 + g2  -> the partial-derivative term  ∂J_L/∂θ of Eq. (5)
        influence = g3       -> the influence term  (B_L/β) ∇θ Q_F of Eq. (5)

    Keeping the two separate lets us measure their relative magnitude, which is
    what determines the direction of their sum (see Sec. 5): a steep follower
    reward inflates only ``influence``, so the sum's direction is dominated by
    it. Gradient clipping rescales the sum and hence preserves this ratio.
    """
    energy_cost = float(np.sum(b ** 2))
    g1 = _grad_rL_theta(env, il, aa, energy_cost)

    g_log_p = _grad_log_p_theta(s, b, s_next, env, A_theta, dA_dtheta)
    V_L_s_next = _leader_value(s_next, M_s, V_const)

    s_next_mean = A_theta @ s + env.B @ b
    V_L_s_next_mean = _leader_value(s_next_mean, M_s, V_const) - float(np.trace(M_s) * (env.transition_std**2))
    g2 = (V_L_s_next - V_L_s_next_mean) * g_log_p

    direct = g1 + g2

    if not include_term3:
        zero = np.zeros_like(direct)
        return (direct, zero) if return_terms else direct

    V_L_s  = _leader_value(s, M_s, V_const)
    Q_L_sb = r_L + gamma_L * V_L_s_next
    B_L    = Q_L_sb - V_L_s

    g_log_pi = _grad_log_pi_theta_full(
        s, b, env.B, A_theta, sol_grad, dP_dtheta, dA_dtheta,
        gamma_F, beta=beta,
    )
    g3     = B_L * g_log_pi
    return (direct, g3) if return_terms else direct + g3


# ---------------------------------------------------------------------------
# Episode-level gradient (sum over timesteps with γ_L^t weighting)
# ---------------------------------------------------------------------------

def theorem1_episode_gradient(
    traj: dict,
    il: np.ndarray,
    aa: np.ndarray,
    env,
    gamma_L: float,
    gamma_F: float,
    beta: float,
    sol_follower: dict,
    sol_grad: dict | None,
    include_term3: bool,
    return_terms: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Episode-level HG.

    With ``return_terms=True`` returns (direct, influence) accumulated over the
    episode; their sum equals the default return value.
    """
    s_list  = [np.asarray(x, dtype=np.float64) for x in traj["s"]]
    b_list  = [np.asarray(x, dtype=np.float64) for x in traj["b"]]
    rL_list = list(traj["r_L"])
    T       = len(rL_list)

    A_theta = env.compute_parameterized_A(il, aa)

    dA_dth = _dA_dtheta(env)  # (n*n, 2n)

    if include_term3:
        dP_dth = _dP_dtheta_vec(
            A_theta, env.B, sol_grad["P"], sol_grad["S_inv"], sol_grad["Ks"],
            gamma_F, dA_dth,
        )
    else:
        dP_dth = None

    Sigma_0 = (env.init_temp_std ** 2) * np.eye(env.num_zones)
    lv_info = compute_leader_value_lyapunov(
        env, il, aa, A_theta, sol_follower["Ks"], sol_follower["S_inv"],
        gamma_F, gamma_L, beta, Sigma_0,
    )
    M_s = lv_info["M_s"]
    V_const = lv_info["V_const"]

    n_theta   = 2 * env.num_zones
    direct    = np.zeros(n_theta, dtype=np.float64)
    influence = np.zeros(n_theta, dtype=np.float64)

    for t in range(T - 1):
        s_t      = s_list[t]
        b_t      = b_list[t]
        s_next_t = s_list[t + 1]
        r_L_t    = float(rL_list[t])

        d_t, i_t = theorem1_step_gradient(
            s_t, b_t, s_next_t, il, aa, r_L_t,
            env, A_theta, sol_grad, dA_dth, dP_dth, M_s, V_const,
            gamma_F, gamma_L, beta, include_term3, return_terms=True,
        )
        w = gamma_L ** t
        direct    += w * d_t
        influence += w * i_t

    return (direct, influence) if return_terms else direct + influence


# ---------------------------------------------------------------------------
# Batch gradient (across multiple episodes per iteration)
# ---------------------------------------------------------------------------

def theorem1_batch_gradient(
    trajs: list[dict],
    env,
    leader_mu: np.ndarray,
    leader_sigma: float,
    gamma_L: float,
    gamma_F: float,
    beta: float,
) -> np.ndarray:
    n_theta   = 2 * env.num_zones
    grad_mu   = np.zeros(n_theta, dtype=np.float64)

    for traj in trajs:
        theta = np.asarray(traj["theta"], dtype=np.float64)
        il    = theta[:env.num_zones]
        aa    = theta[env.num_zones:]
        A_theta = env.compute_parameterized_A(il, aa)
        sol_f   = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)

        try:
            grad_theta = theorem1_episode_gradient(
                traj, il, aa, env, gamma_L, gamma_F, beta,
                sol_follower=sol_f, sol_grad=sol_f, include_term3=True,
            )
        except Exception:
            continue

        grad_mu += grad_theta

    if len(trajs) > 0:
        grad_mu /= len(trajs)

    return grad_mu