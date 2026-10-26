"""Theorem 1 gradient for Configurable MDP (LQR / continuous setting).

Theorem (Conf-MDP):

    ∇_θ J_L = E_τ [ Σ_{t≥0} γ_L^t (
        ∇_θ r_L^θ                        # Term 1: reward gradient
      + V_L^{θ†} ∇_θ log p^θ            # Term 2: score × leader value
      + (B_L^{θ†} / β) ∇_θ Q_F^{θ†}    # Term 3: benefit × follower Q gradient
    )]

where
    θ = (il, aa)  ∈ R^{2n}   (insulation_level ‖ airflow_adjustment)
    A(θ)          parameterises the state-transition matrix
    p^θ(s'|s,b)  = N(s' | A(θ)s + Bb,  σ²I)
    Q_F^{θ†}(s,b) = b'S(θ)b + 2b'(S Ks)(θ)s + ξ(s; θ)
                   where S, Ks come from the DARE solution
    V_L           = leader value function (Lyapunov-based)
    B_L           = Q_L(s,b) - V_L  (advantage / benefit)

All three terms are computed analytically using:
  - ∂A/∂θ  (closed form, see _dA_dtheta)
  - ∂P/∂θ  via the discrete Lyapunov equation on the DARE sensitivity
             (Implicit Function Theorem, see _dP_dtheta_vec)
  - ∂S/∂θ  derived from ∂P/∂θ
  - V_L     from solve_discrete_lyapunov (closed-loop Lyapunov)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from blackrl.algos.lqr_analytical import solve_confmdp_lqr


# ---------------------------------------------------------------------------
# ∂A(θ)/∂θ  — Jacobian of the parameterised transition matrix
# ---------------------------------------------------------------------------

def _dA_dtheta(env) -> np.ndarray:
    """Compute ∂vec(A(θ))/∂θ  for BuildingThermalControlEnv.

    Uses BuildingThermalControl_3 parameterisation (ring topology):
        A_θ[i,i]     = A[i,i] + ie[i]*(1-il[i]) - A[i,j_next]*aa[i] - A[i,j_prev]*aa[j_prev]
        A_θ[i,j_next] = A[i,j_next] * aa[i]   (symmetric entry A_θ[j_next,i] = same)

    θ = [il_0, …, il_{n-1}, aa_0, …, aa_{n-1}]  shape (2n,)

    Returns
    -------
    dA_dtheta : np.ndarray, shape (n*n, 2*n)
        Each column k is vec(∂A/∂θ_k), row-major vectorisation.
    """
    n   = env.num_zones
    A   = env.A          # (n, n)  base matrix
    ie  = env.insulation_effect  # (n,)

    dA = np.zeros((n * n, 2 * n), dtype=np.float64)

    for i in range(n):
        j_next = (i + 1) % n
        j_prev = (i - 1 + n) % n

        # ∂A_θ[i,i] / ∂il[i]  = -ie[i]
        row_ii = i * n + i
        dA[row_ii, i] = -ie[i]

        # ∂A_θ[i,i]     / ∂aa[i]      = -A[i, j_next]
        # ∂A_θ[i,j_next] / ∂aa[i]     = +A[i, j_next]
        # ∂A_θ[j_next,i] / ∂aa[i]     = +A[j_next, i]
        col_aa_i = n + i

        row_ij_next  = i      * n + j_next
        row_jnext_i  = j_next * n + i

        dA[row_ii,       col_aa_i] -= A[i, j_next]
        dA[row_ij_next,  col_aa_i]  = A[i, j_next]
        dA[row_jnext_i,  col_aa_i]  = A[j_next, i]

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
    """Compute ∂vec(P)/∂θ via the DARE implicit function theorem.

    The DARE  P = Q + γ A'PA - γ² A'PB S⁻¹ B'PA  defines P(θ) implicitly.
    Taking the total derivative w.r.t. θ_k (one column of dA_dtheta):

        vec(dP) = (∂F/∂P)⁻¹  (∂F/∂A ⊗ dA/dθ_k)

    where  F(P, A) = Q + γA'PA - γ²A'PB S⁻¹ B'PA - P = 0.

    The sensitivity equation turns into a discrete Lyapunov equation:
        A_cl' dP A_cl - dP = -dRHS
    with  A_cl = √γ (A - B Ks)  (closed-loop, discounted).

    Since the Lyapunov equation is linear in dP, we solve it once per θ_k.

    Parameters
    ----------
    A_theta  : (n, n)  current A(θ)
    B        : (n, m)
    P        : (n, n)  current DARE solution
    S_inv    : (m, m)  = (U + γ B'PB)^{-1}
    Ks       : (m, n)  = γ S⁻¹ B'PA  (optimal feedback gain)
    gamma    : float   follower discount factor
    dA_dtheta: (n*n, 2n)  Jacobian ∂vec(A)/∂θ

    Returns
    -------
    dP_dtheta : np.ndarray, shape (n*n, 2n)
        Each column k is vec(∂P/∂θ_k).
    """
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

        # ∂F/∂A  applied to dA_k (while holding everything else fixed):
        # F = Q + γA'PA - γ²A'PB S⁻¹ B'PA
        # dF/dA[δA] = γ (δA)'PA + γ A'P(δA)
        #            - γ² [(δA)'PB S⁻¹ B'PA + A'PB S⁻¹ B'P(δA)]
        #
        # = γ (δA)'PA + γ A'P(δA) - γ(Ks'(γδA)'PA + γ A'P(δA)Ks)
        # Simplifying via Ks = γ S⁻¹ B'PA:
        #   dRHS = γ dA_k' P A_theta + γ A_theta' P dA_k
        #         - γ² dA_k' P B S⁻¹ B' P A_theta
        #         - γ² A_theta' P B S⁻¹ B' P dA_k

        T1 = gamma * dA_k.T @ P @ A_theta
        T2 = gamma * A_theta.T @ P @ dA_k
        T3 = gamma**2 * dA_k.T @ P @ B @ S_inv @ B.T @ P @ A_theta
        T4 = gamma**2 * A_theta.T @ P @ B @ S_inv @ B.T @ P @ dA_k

        dRHS = T1 + T2 - T3 - T4   # (n, n)

        # Lyapunov:  Abar' dP Abar - dP = -dRHS
        # scipy convention: A X A' - X = Q  →  solve_discrete_lyapunov(A, Q)
        # Our equation:  Abar' X Abar - X = -dRHS
        # Rearranging:  Abar X Abar' - X = -dRHS  (after transposing both sides,
        # valid because dP is symmetric)
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

    # Leader's Q_L weighting on state (stability):  -Var(s) = -diag(I)·s²
    # The per-step leader reward is:
    #   r_L = w_stab*(-Var(s)) - w_nrg*||b||² - w_ins*||il||² - w_air*||aa||²
    # The state-quadratic term for expected V_L is driven by:
    #   -E[Var(s)] ≈ -(1/n) s'(I - (1/n)11') s   (exact for Gaussian)
    n_z = env.num_zones
    Q_stab = env.stability_weight * (
        np.eye(n_z) / n_z - np.ones((n_z, n_z)) / (n_z ** 2)
    )
    # The energy cost state-dependent term: w_nrg * E[|| -Ks s + \epsilon_b ||^2]
    # yields a state-dependent cost of w_nrg * Ks' Ks.
    Q_energy = env.energy_weight * (Ks.T @ Ks)
    Q_L_state = Q_stab + Q_energy

    sq_g = np.sqrt(gamma_L)
    # M_s for V_L(s) = -s' M_s s + V_const
    M_s = solve_discrete_lyapunov(sq_g * A_cl.T, Q_L_state)

    # Constant cost component from action penalties and follower noise
    c_const = (
        -env.energy_weight * np.trace(Cov_b)
        -env.insulation_cost_weight * np.sum(il ** 2)
        -env.airflow_cost_weight * np.sum(aa ** 2)
    )
    # Expected next state value const: V_const = C_const - gamma_L * Tr(M_s Q_noise) + gamma_L * V_const
    v_const = (c_const - gamma_L * float(np.trace(M_s @ Q_noise))) / (1.0 - gamma_L)

    return {"M_s": M_s, "V_const": v_const, "A_cl": A_cl, "Q_noise": Q_noise, "Cov_b": Cov_b}


# ---------------------------------------------------------------------------
# Term 1:  ∇_θ r_L^θ  (reward gradient)
# ---------------------------------------------------------------------------

def _grad_rL_theta(
    env,
    il: np.ndarray,
    aa: np.ndarray,
    energy_cost: float,
) -> np.ndarray:
    """Compute ∂r_L/∂θ evaluated at (il, aa).

    Leader reward decomposition:
        r_L = w_stab*(-Var(s)) - w_nrg*||b||² - w_ins*||il||² - w_air*||aa||²

    Only the insulation/airflow cost terms depend on θ:
        ∂r_L/∂il_i = -2 w_ins il_i
        ∂r_L/∂aa_i = -2 w_air aa_i

    The stability and energy terms depend on θ only *indirectly* through
    the trajectory (caught by Terms 2 and 3).  Here we return the
    *direct* partial derivative.

    Returns
    -------
    grad : np.ndarray, shape (2*n,)
    """
    n = env.num_zones
    grad = np.zeros(2 * n, dtype=np.float64)
    grad[:n]  = -2.0 * env.insulation_cost_weight * il
    grad[n:]  = -2.0 * env.airflow_cost_weight * aa
    return grad


# ---------------------------------------------------------------------------
# Term 2:  V_L^{θ†} ∇_θ log p^θ
# ---------------------------------------------------------------------------

def _grad_log_p_theta(
    s: np.ndarray,
    b: np.ndarray,
    s_next: np.ndarray,
    env,
    A_theta: np.ndarray,
    dA_dtheta: np.ndarray,
) -> np.ndarray:
    """Compute ∇_θ log p^θ(s'|s, b).

    p^θ(s'|s,b) = N(s' | A(θ)s + Bb, σ²_env I)

    log p^θ = -1/(2σ²) ||s' - A(θ)s - Bb||²  + const

    ∂log p / ∂θ_k = (1/σ²) (s' - A(θ)s - Bb)' (∂A(θ)/∂θ_k) s
                  = (1/σ²) residual' dA_k s

    Parameters
    ----------
    s, b, s_next : (n,), (m,), (n,)
    env          : BuildingThermalControlEnv
    A_theta      : (n, n)  current A(θ)
    dA_dtheta    : (n*n, 2n)  Jacobian

    Returns
    -------
    grad : (2n,)
    """
    sigma2 = env.transition_std ** 2
    residual = s_next - A_theta @ s - env.B @ b   # (n,)

    # For each θ_k:  (1/σ²) residual' dA_k s
    # dA_dtheta has shape (n*n, 2n); each column k is vec(dA_k)
    # residual' dA_k s  = (s ⊗ residual)' vec(dA_k)
    #                   = kron_vec' col_k   where kron_vec = s ⊗ residual
    n = env.num_zones
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
    """Compute ∇_θ Q_F^{θ†}(s, b) analytically.

    Q_F(s,b) = s'Ws + b'Ub
              + γ E_ε [(Aθ s + Bb + ε)' P(θ) (Aθ s + Bb + ε)]
              + γ v^{θ}

    where E_ε[(...)'’P(…)] = (Aθ s + Bb)' P (Aθ s + Bb) + σ² Tr(P)
    and   v^{θ} = 1/(1-γ) [−β m/2 log(πβ) + β/2 log det(S) + γσ² Tr(P)]

    ∂Q_F/∂θ_k has seven terms:

      (Aθ s + Bb)' P (Aθ s + Bb) の微分:
        t1 = γ b'B' dP_k B b                   (P 経由, b 二次項)
        t2 = 2γ b'B' dP_k A_theta s            (P 経由, 交差項)
        t3 = γ (A_theta s)' dP_k (A_theta s)   (P 経由, 状態二次項)
        t4 = 2γ b'B' P dA_k s                  (A 経由, 交差項)
        t5 = 2γ (A_theta s)' P dA_k s          (A 経由, 状態二次項)

      ノイズ項 + v^{θ} の微分 (定数項):
        t6 = [γσ²/(1-γ)] Tr(dP_k)
        t7 = [γ²β / (2(1-γ))] Tr(B S⁻¹ B' dP_k)

    Parameters
    ----------
    s, b         : (n,), (m,)
    env_B        : (n, m)  follower action matrix B
    A_theta      : (n, n)  current A(θ)
    sol          : DARE solution dict
    dP_dtheta    : (n*n, 2n)  sensitivity of P w.r.t. θ
    dA_dtheta    : (n*n, 2n)  Jacobian ∂vec(A)/∂θ
    gamma_F      : float
    sigma2       : float  transition noise variance σ²
    beta         : float  inverse temperature β

    Returns
    -------
    grad_QF : (2n,)
    """
    n       = A_theta.shape[0]
    n_theta = dP_dtheta.shape[1]
    B       = env_B
    P       = sol["P"]
    S_inv   = sol["S_inv"]

    As       = A_theta @ s   # (n,)
    BSinvBt  = B @ S_inv @ B.T   # (n, n), 一度計算して共有
    coeff_t6 = gamma_F * sigma2 / (1.0 - gamma_F)
    coeff_t7 = gamma_F**2 * beta / (2.0 * (1.0 - gamma_F))
    grad = np.zeros(n_theta, dtype=np.float64)

    for k in range(n_theta):
        dP_k = dP_dtheta[:, k].reshape(n, n)
        dA_k = dA_dtheta[:, k].reshape(n, n)

        # ---- P 経由 ----
        # t1: γ b'B' dP_k B b
        t1 = gamma_F * float(b @ B.T @ dP_k @ B @ b)
        # t2: 2γ b'B' dP_k A_theta s
        t2 = 2.0 * gamma_F * float(b @ B.T @ dP_k @ As)
        # t3: γ (A_theta s)' dP_k (A_theta s)
        t3 = gamma_F * float(As @ dP_k @ As)

        # ---- A 経由 ----
        dA_k_s = dA_k @ s   # (n,)
        # t4: 2γ b'B' P dA_k s
        t4 = 2.0 * gamma_F * float(b @ B.T @ P @ dA_k_s)
        # t5: 2γ (A_theta s)' P dA_k s
        t5 = 2.0 * gamma_F * float(As @ P @ dA_k_s)

        # ---- ノイズ項 + v^{θ} 定数項 ----
        tr_dP_k        = float(np.trace(dP_k))
        tr_BSinvBt_dPk = float(np.trace(BSinvBt @ dP_k))
        # t6: [γσ²/(1-γ)] Tr(dP_k)
        t6 = coeff_t6 * tr_dP_k
        # t7: [γ²β/(2(1-γ))] Tr(B S⁻¹ B' dP_k)
        t7 = coeff_t7 * tr_BSinvBt_dPk

        grad[k] = t1 + t2 + t3 + t4 + t5 + t6 + t7

    return -grad


# ---------------------------------------------------------------------------
# Leader Q  and  Benefit  (for Term 3)
# ---------------------------------------------------------------------------


def _leader_value(s: np.ndarray, M_s: np.ndarray, V_const: float) -> float:
    """V_L(s) = -s' M_s s + V_const."""
    return -float(s @ M_s @ s) + V_const


# ---------------------------------------------------------------------------
# Full single-step gradient of Theorem 1
# ---------------------------------------------------------------------------

def theorem1_step_gradient(
    s: np.ndarray,
    b: np.ndarray,
    s_next: np.ndarray,
    il: np.ndarray,
    aa: np.ndarray,
    r_L: float,
    env,
    A_theta: np.ndarray,
    sol_grad: dict | None,         # term3 用 DARE 解 (None → term3 スキップ)
    dA_dtheta: np.ndarray,
    dP_dtheta: np.ndarray | None,  # term3 用 ∂P/∂θ  (sol_grad が None なら None)
    M_s: np.ndarray,
    V_const: float,
    gamma_F: float,
    gamma_L: float,
    beta: float,
    include_term3: bool,
) -> np.ndarray:
    """Compute one (s,b,s') sample's contribution to ∇_θ J_L.

    Returns
    -------
    grad : np.ndarray, shape (2*n,)
        Unweighted (no γ_L^t factor) single-step gradient estimate.
    """
    # ---- Term 1 ----
    energy_cost = float(np.sum(b ** 2))
    g1 = _grad_rL_theta(env, il, aa, energy_cost)

    # ---- Term 2 ----
    g_log_p = _grad_log_p_theta(s, b, s_next, env, A_theta, dA_dtheta)
    V_L_s_next = _leader_value(s_next, M_s, V_const)
    g2 = V_L_s_next * g_log_p

    if not include_term3:
        return g1 + g2

    # ---- Term 3 ----
    V_L_s  = _leader_value(s, M_s, V_const)
    Q_L_sb = r_L + gamma_L * V_L_s_next
    B_L    = Q_L_sb - V_L_s

    g_QF   = _grad_QF_theta_full(
        s, b, env.B, A_theta, sol_grad, dP_dtheta, dA_dtheta,
        gamma_F, sigma2=env.transition_std**2, beta=beta,
    )
    g3     = (B_L / beta) * g_QF
    return g1 + g2 + g3


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
    sol_follower: dict,     # フォロワー真の DARE 解 → Lyapunov V_L 計算に使用
    sol_grad: dict | None,  # term3 の ∇Q_F 計算用 DARE 解 (None → term3 スキップ)
    include_term3: bool,
) -> np.ndarray:
    """Compute ∇_θ J_L contribution from one full episode via Theorem 1.

    Parameters
    ----------
    traj         : dict with 's', 'b', 'r_L' lists (time-ordered)
    il, aa       : (n,) insulation & airflow (constant for this episode)
    env          : BuildingThermalControlEnv
    gamma_L, gamma_F, beta : hyperparameters
    sol_follower : DARE solution for true follower policy
                   (Ks/S_inv used for Lyapunov V_L computation)
    sol_grad     : DARE solution used for term3 ∇Q_F
                   softvi       → same as sol_follower
                   irl_known_dyn→ IRL-estimated sol (Q_hat/R_hat)
                   non_irl      → None  (include_term3=False)
    include_term3: bool  whether to compute term3 (Benefit term)

    Returns
    -------
    grad_theta : np.ndarray, shape (2*n,)
    """
    s_list  = [np.asarray(x, dtype=np.float64) for x in traj["s"]]
    b_list  = [np.asarray(x, dtype=np.float64) for x in traj["b"]]
    rL_list = list(traj["r_L"])
    T       = len(rL_list)

    A_theta = env.compute_parameterized_A(il, aa)

    # --- Jacobian ∂A/∂θ ---
    dA_dth = _dA_dtheta(env)  # (n*n, 2n)

    # --- ∂P/∂θ: term3 が必要な場合のみ (sol_grad の P/S_inv/Ks を使用) ---
    if include_term3:
        dP_dth = _dP_dtheta_vec(
            A_theta, env.B, sol_grad["P"], sol_grad["S_inv"], sol_grad["Ks"],
            gamma_F, dA_dth,
        )
    else:
        dP_dth = None

    # --- Leader value matrix M_s: フォロワー真の行動 (sol_follower) に基づく V_L ---
    Sigma_0 = (env.init_temp_std ** 2) * np.eye(env.num_zones)
    lv_info = compute_leader_value_lyapunov(
        env, il, aa, A_theta, sol_follower["Ks"], sol_follower["S_inv"],
        gamma_F, gamma_L, beta, Sigma_0,
    )
    M_s = lv_info["M_s"]
    V_const = lv_info["V_const"]

    n_theta = 2 * env.num_zones
    grad    = np.zeros(n_theta, dtype=np.float64)

    for t in range(T - 1):
        s_t      = s_list[t]
        b_t      = b_list[t]
        s_next_t = s_list[t + 1]
        r_L_t    = float(rL_list[t])

        step_grad = theorem1_step_gradient(
            s_t, b_t, s_next_t, il, aa, r_L_t,
            env, A_theta, sol_grad, dA_dth, dP_dth, M_s, V_const,
            gamma_F, gamma_L, beta, include_term3,
        )
        grad += (gamma_L ** t) * step_grad

    return grad


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
    """Compute ∇_μ J_L from a batch of trajectories using Theorem 1.

    Leader policy: θ ~ N(μ, σ²I) clipped to [0,1]^{2n}
    Each episode has its own θ (il, aa).

    The gradient w.r.t. μ is obtained by the reparameterisation-like chain rule:

        ∇_μ J_L = E_θ [ ∇_θ J_L · ∂θ/∂μ ]

    For θ = clip(ε + μ, 0, 1) with ε ~ N(0, σ²I):
        ∂θ_k/∂μ_k ≈ 1  (ignoring boundary effects at 0, 1)

    So:
        ∇_μ J_L ≈ (1/N) Σ_i ∇_{θ_i} J_L

    Parameters
    ----------
    trajs       : list of trajectory dicts (each has 's', 'b', 'r_L', 'theta')
    env         : BuildingThermalControlEnv
    leader_mu   : (2n,)  current policy mean
    leader_sigma: float  leader policy std
    gamma_L, gamma_F, beta : floats

    Returns
    -------
    grad_mu : np.ndarray, shape (2n,)
    """
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
