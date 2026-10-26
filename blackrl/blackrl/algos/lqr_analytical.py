from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


# ---------------------------------------------------------------------------
# Internal helper: discounted DARE
# ---------------------------------------------------------------------------

def _solve_dare(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    gamma: float,
) -> np.ndarray:
    sq_gamma = np.sqrt(gamma)
    return solve_discrete_are(sq_gamma * A, sq_gamma * B, W, U)


# ---------------------------------------------------------------------------
# conf-mdp (Configurable MDP)
# ---------------------------------------------------------------------------

def solve_confmdp_lqr(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    gamma_F: float,
) -> dict:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    P = _solve_dare(A, B, W, U, gamma_F)
    S = U + gamma_F * (B.T @ P @ B)                       # (m, m)
    S_inv = np.linalg.inv(S)
    Ks = gamma_F * S_inv @ B.T @ P @ A                    # (m, n)
    Pww = gamma_F * P - gamma_F**2 * P @ B @ S_inv @ B.T @ P  # (n, n)

    return {
        "P": P,
        "S": S,
        "S_inv": S_inv,
        "Ks": Ks,
        "Pww": Pww,
        "gamma_F": gamma_F,
    }


def confmdp_q_function(
    s: np.ndarray,
    b: np.ndarray,
    sol: dict,
) -> float:
    s = np.asarray(s, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    S = sol["S"]
    Ks = sol["Ks"]

    # 2 b' S Ks s  (= 2 γ b' B' P A s)
    cross = 2.0 * float(b @ S @ Ks @ s)
    quad_b = float(b @ S @ b)
    return -(quad_b + cross)  # Return as reward (negative)


def confmdp_constant(
    sol: dict,
    beta: float,
    m: int,
    Sigma: np.ndarray | None = None,
) -> float:
    S = sol["S"]
    P = sol["P"]
    gamma = sol["gamma_F"]

    sign, logdet_S = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("S is not positive definite (det(S) ≤ 0).")

    term1 = -beta * m / 2.0 * np.log(np.pi * beta)
    term2 = beta / 2.0 * float(logdet_S)
    term3 = gamma * float(np.trace(P @ Sigma)) if Sigma is not None else 0.0

    return -(term1 + term2 + term3) / (1.0 - gamma)  # Return as reward (negative)


def confmdp_value_function(
    s: np.ndarray,
    sol: dict,
    A: np.ndarray,
    W: np.ndarray,
    beta: float,
    m: int,
    Sigma: np.ndarray | None = None,
) -> float:
    s = np.asarray(s, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    Pww = sol["Pww"]

    As = A @ s
    v_const = confmdp_constant(sol, beta, m, Sigma)
    return -(float(s @ W @ s) + float(As @ Pww @ As)) + v_const  # Return as reward (negative)


def confmdp_optimal_action(s: np.ndarray, sol: dict) -> np.ndarray:
    
    s = np.asarray(s, dtype=np.float64).ravel()
    return -(sol["Ks"] @ s)


def confmdp_sample_action(
    s: np.ndarray,
    sol: dict,
    beta: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from the optimal Gaussian policy for conf-mdp.

    Covariance: Σ_b = (β/2) S^{-1}
    """
    if rng is None:
        rng = np.random.default_rng()
    mean = confmdp_optimal_action(s, sol)
    cov = (beta / 2.0) * sol["S_inv"]
    return rng.multivariate_normal(mean, cov)


# ---------------------------------------------------------------------------
# 2-Player Markov Game (2-PMG)
# ---------------------------------------------------------------------------

def solve_2pmg_lqr(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    gamma_F: float,
    K_theta: np.ndarray,
) -> dict:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    K_theta = np.asarray(K_theta, dtype=np.float64)

    A_tilde = A + C @ K_theta                              # (n, n)  = Ã

    P = _solve_dare(A_tilde, B, W, U, gamma_F)

    S = U + gamma_F * (B.T @ P @ B)                          # (m, m)
    S_inv = np.linalg.inv(S)
    Ks = gamma_F * S_inv @ B.T @ P @ A                       # (m, n)
    Ka = gamma_F * S_inv @ B.T @ P @ C                       # (m, p)
    Pww = gamma_F * P - gamma_F**2 * P @ B @ S_inv @ B.T @ P   # (n, n)

    return {
        "A_tilde": A_tilde,
        "P": P,
        "S": S,
        "S_inv": S_inv,
        "Ks": Ks,
        "Ka": Ka,
        "Pww": Pww,
        "gamma_F": gamma_F,
    }


def q_function_2pmg(
    s: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sol: dict,
) -> float:
    s = np.asarray(s, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    S = sol["S"]
    Ks = sol["Ks"]
    Ka = sol["Ka"]

    # 2 b' S (Ks s + Ka a)
    cross = 2.0 * float(b @ S @ (Ks @ s + Ka @ a))
    quad_b = float(b @ S @ b)
    return -(quad_b + cross)  # Return as reward (negative)


def constant_2pmg(
    sol: dict,
    beta: float,
    m: int,
    C: np.ndarray,
    Sigma_theta: np.ndarray | None = None,
) -> float:
    S = sol["S"]
    Pww = sol["Pww"]
    gamma = sol["gamma_F"]

    sign, logdet_S = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("S is not positive definite (det(S) ≤ 0).")

    term1 = -beta * m / 2.0 * np.log(np.pi * beta)
    term2 = beta / 2.0 * float(logdet_S)
    term3 = 0.0
    if Sigma_theta is not None:
        C = np.asarray(C, dtype=np.float64)
        Sigma_theta = np.asarray(Sigma_theta, dtype=np.float64)
        noise_cov = C @ Sigma_theta @ C.T               # C Σ_θ C',  (n, n)
        term3 = gamma * float(np.trace(Pww @ noise_cov))

    return -(term1 + term2 + term3) / (1.0 - gamma)  # Return as reward (negative)


def value_function_2pmg(
    s: np.ndarray,
    a: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    sol: dict,
    beta: float,
    m: int,
    Sigma_theta: np.ndarray | None = None,
) -> float:
    s = np.asarray(s, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    Pww = sol["Pww"]

    z = A @ s + C @ a                                  # = Ã s + ε̃
    v_const = constant_2pmg(sol, beta, m, C, Sigma_theta)
    return -(float(s @ W @ s) + float(z @ Pww @ z)) + v_const  # Return as reward (negative)


def optimal_action_2pmg(
    s: np.ndarray,
    a: np.ndarray,
    sol: dict,
) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    return -(sol["Ks"] @ s + sol["Ka"] @ a)


def sample_action_2pmg(
    s: np.ndarray,
    a: np.ndarray,
    sol: dict,
    beta: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    
    if rng is None:
        rng = np.random.default_rng()
    mean = optimal_action_2pmg(s, a, sol)
    cov = (beta / 2.0) * sol["S_inv"]
    return rng.multivariate_normal(mean, cov)