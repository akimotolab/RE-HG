"""Analytical LQR solutions for bilevel RL (continuous setting).

Two scenarios are supported:

1. Configurable MDP (conf-mdp)
   Dynamics:  s_{t+1} = A_θ s_t + B_θ b_t + ε_t,  ε_t ~ N(0, Σ)
   Follower reward: r_F(s, b) = -(s' W s + b' U b)
   Optimal follower policy:
       g(b | s) = N(b | -Ks s, (β/2) S^{-1})
   where S = U + γ B' P B,  Ks = γ S^{-1} B' P A,
   and P solves the discounted DARE:
       P = W + γ A' P A - γ² A' P B (U + γ B' P B)^{-1} B' P A

2. Two-Player Markov Game (2-PMG)
   Leader policy: a_t ~ N(K_θ s_t, Σ_θ)
   Equivalent system: Ã = A + C K_θ,  ε̃_t ~ N(0, C Σ_θ C')
   Dynamics:  s_{t+1} = Ã s_t + B b_t + ε̃_t
   Using the identity  Ã s + ε̃ = A s + C a:
   Optimal follower policy:
       g(b | s, a) = N(b | -Ks s - Ka a, (β/2) S^{-1})
   where Ks = γ S^{-1} B' P A,  Ka = γ S^{-1} B' P C.

References
----------
Notation: W, U = state / action cost matrices (= Q, R in standard LQR).
          γ    = follower discount factor.
          β    = inverse temperature (entropy reg).
          m    = follower action dimension.
          Σ    = process noise covariance (conf-mdp).
          Σ_θ  = leader policy covariance (2-PMG).
"""

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
    """Solve the discounted discrete algebraic Riccati equation.

    Equation:
        P = W + γ A' P A - γ² A' P B (U + γ B' P B)^{-1} B' P A

    Reduced to the standard form by substituting Â = √γ A, B̂ = √γ B:
        P = W + Â' P Â - Â' P B̂ (U + B̂' P B̂)^{-1} B̂' P Â

    Parameters
    ----------
    A : (n, n)
    B : (n, m)
    W : (n, n)  state cost, positive semi-definite
    U : (m, m)  action cost, positive definite
    gamma : float in (0, 1)

    Returns
    -------
    P : (n, n)
    """
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
    """Compute the analytical LQR solution for the configurable MDP setting.

    Dynamics: s_{t+1} = A s_t + B b_t + ε_t
    Follower cost: s' W s + b' U b

    Parameters
    ----------
    A : (n, n)  state transition matrix (= A_θ)
    B : (n, m)  follower action matrix (= B_θ)
    W : (n, n)  state cost, positive semi-definite
    U : (m, m)  follower action cost, positive definite
    gamma : float  follower discount factor γ_F

    Returns
    -------
    dict with keys:
        P     : (n, n)  DARE solution
        S     : (m, m)  = U + γ B' P B
        S_inv : (m, m)  inverse of S
        Ks    : (m, n)  feedback gain = γ S^{-1} B' P A
        Pww   : (n, n)  = γ P - γ² P B S^{-1} B' P  (coeff matrix in V)
        gamma : float   stored for downstream use
    """
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
    """Evaluate the b-dependent (quadratic) part of Q for conf-mdp.

    Q(s, b) = b' S b + 2 γ b' B' P A s + ξ(s)

    This function returns only the (b, s)-coupled terms:
        b' S b + 2 b' (S Ks) s

    since S Ks = γ B' P A by definition of Ks = γ S^{-1} B' P A.

    The state-only term ξ(s) requires the full value function; use
    `confmdp_value_function` for the complete expression.
    """
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
    """Compute the state-independent constant v^θ for conf-mdp.

    v^θ = (1 / (1-γ)) * [ -βm/2 log(πβ) + β/2 log det(S) + γ Tr(P Σ) ]

    Parameters
    ----------
    sol   : output of solve_confmdp_lqr
    beta  : inverse temperature β > 0
    m     : follower action dimension
    Sigma : (n, n) process noise covariance; None → zero noise (Tr term = 0)

    Returns
    -------
    float  (the full constant including the 1/(1-γ) factor)
    """
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
    """Evaluate the analytical value function (cost-based) for conf-mdp.

    V(s) = s' W s + (A s)' Pww (A s) + v^θ

    Parameters
    ----------
    s     : state vector
    sol   : output of solve_confmdp_lqr
    A     : (n, n) state transition matrix
    W     : (n, n) state cost matrix
    beta  : inverse temperature β
    m     : follower action dimension
    Sigma : (n, n) process noise covariance (None → no noise)
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    Pww = sol["Pww"]

    As = A @ s
    v_const = confmdp_constant(sol, beta, m, Sigma)
    return -(float(s @ W @ s) + float(As @ Pww @ As)) + v_const  # Return as reward (negative)


def confmdp_optimal_action(s: np.ndarray, sol: dict) -> np.ndarray:
    """Return the mean of the optimal Gaussian follower policy for conf-mdp.

    g(b | s) = N(b | -Ks s, (β/2) S^{-1})  →  mean = -Ks s
    """
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
    """Compute the analytical LQR solution for the 2-PMG setting.

    The leader's linear policy is a_t ~ N(K_θ s_t, Σ_θ), giving:
        Ã = A + C K_θ   (effective transition under K_θ)
        ε̃_t = C (a_t - K_θ s_t) ~ N(0, C Σ_θ C')

    Dynamics: s_{t+1} = Ã s_t + B b_t + ε̃_t

    Using the identity  Ã s + ε̃ = A s + C a, the optimal follower policy
    conditioned on the actual leader action a is:
        g(b | s, a) = N(b | -Ks s - Ka a, (β/2) S^{-1})

    Parameters
    ----------
    A       : (n, n)  base state transition matrix
    B       : (n, m)  follower action matrix
    C       : (n, p)  leader action matrix
    W       : (n, n)  state cost, positive semi-definite
    U       : (m, m)  follower action cost, positive definite
    gamma   : float   follower discount factor γ_F
    K_theta : (p, n)  leader's linear policy gain

    Returns
    -------
    dict with keys:
        A_tilde : (n, n)  = A + C K_θ
        P       : (n, n)  DARE solution for (Ã, B, W, U)
        S       : (m, m)  = U + γ B' P B
        S_inv   : (m, m)  inverse of S
        Ks      : (m, n)  = γ S^{-1} B' P A  (state feedback gain)
        Ka      : (m, p)  = γ S^{-1} B' P C  (leader-action gain)
        Pww     : (n, n)  = γ P - γ² P B S^{-1} B' P
        gamma   : float   stored for downstream use
    """
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
    # Note: Ks uses A (not Ã); Ka uses C.
    # Derivation: minimizing b'Sb + 2γb'B'P(As+Ca) gives b* = -γS^{-1}B'P(As+Ca)
    #             = -ΓS^{-1}B'PA s - γS^{-1}B'PC a  = -Ks s - Ka a.
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
    """Evaluate the b-dependent (quadratic) part of Q for 2-PMG.

    Using the identity  Ã s + ε̃ = A s + C a:

        Q(s, a, b) [quadratic in b] = b' S b + 2γ b' B' P (A s + C a)

    Since Ks = γ S^{-1} B' P A  and  Ka = γ S^{-1} B' P C, we have:
        2γ b' B' P (As + Ca) = 2 b' S (Ks s + Ka a)

    Parameters
    ----------
    s, a, b : state, leader action, follower action
    sol     : output of solve_2pmg_lqr
    """
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
    """Compute the state-independent constant v for 2-PMG.

    v = (1 / (1-γ)) * [ -βm/2 log(πβ) + β/2 log det(S)
                        + γ Tr(Pww C Σ_θ C') ]

    Parameters
    ----------
    sol         : output of solve_2pmg_lqr
    beta        : inverse temperature β
    m           : follower action dimension
    C           : (n, p) leader action matrix
    Sigma_theta : (p, p) leader policy covariance (None → zero noise)

    Returns
    -------
    float  (the full constant including 1/(1-γ))
    """
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
    """Evaluate the value function for 2-PMG given (s, a).

    Using  Ã s + ε̃ = A s + C a:

        V(s, a) = s' W s + (A s + C a)' Pww (A s + C a) + v

    Parameters
    ----------
    s, a        : state and leader action
    A, C, W     : system matrices and state cost
    sol         : output of solve_2pmg_lqr
    beta        : inverse temperature β
    m           : follower action dimension
    Sigma_theta : (p, p) leader policy covariance (None → no noise term)
    """
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
    """Return the mean of the optimal Gaussian follower policy for 2-PMG.

    g(b | s, a) = N(b | -Ks s - Ka a, (β/2) S^{-1})  →  mean = -Ks s - Ka a
    """
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
    """Sample from the optimal Gaussian policy for 2-PMG.

    g(b | s, a) = N(b | -Ks s - Ka a, (β/2) S^{-1})
    """
    if rng is None:
        rng = np.random.default_rng()
    mean = optimal_action_2pmg(s, a, sol)
    cov = (beta / 2.0) * sol["S_inv"]
    return rng.multivariate_normal(mean, cov)