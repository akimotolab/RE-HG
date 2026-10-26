from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401


@dataclass
class Config:
    n_states: int = 2
    n_actions: int = 2
    gamma: float = 0.95
    beta: float = 0.25
    n_iterations: int = 200
    expert_demo_episodes: int = 200
    expert_demo_horizon: int = 30
    theta0: float = 0.9
    theta_lr: float = 0.05
    theta_min: float = 0.0
    theta_max: float = 1.0
    theta_eps: float = 1e-3
    seed: int = 0
    a_bias: float = 10.0
    eig_threshold: float = 1e-6
    reg: float = 1e-6


def transition_prob(theta: float, s: int, b: int) -> np.ndarray:
    """Return P(s'|s,b;theta) for the mixed stochastic/deterministic CMDP.

    Mapping (state/action indices):
      s1 -> 0, s2 -> 1, b1 -> 0, b2 -> 1.

    Transition definition:
      p(s1|s1,b1)=theta,
      p(s1|s2,b2)=theta,
      p(s1|s1,b2)=1,
      p(s1|s2,b1)=0.
    """
    p = np.zeros(2, dtype=np.float64)

    if s == 0 and b == 0:
        p[0] = theta
        p[1] = 1.0 - theta
    elif s == 1 and b == 1:
        p[0] = theta
        p[1] = 1.0 - theta
    elif s == 0 and b == 1:
        p[0] = 1.0
        p[1] = 0.0
    elif s == 1 and b == 0:
        p[0] = 0.0
        p[1] = 1.0
    else:
        raise ValueError(f"Invalid state/action pair: s={s}, b={b}")

    return p


def build_transition_tensor(theta: float) -> np.ndarray:
    P = np.zeros((2, 2, 2), dtype=np.float64)
    for s in range(2):
        for b in range(2):
            P[s, b] = transition_prob(theta, s, b)
    return P


def build_transition_derivative_tensor(theta: float) -> np.ndarray:
    """Return dP(s'|s,b;theta)/dtheta for the synthetic transition."""
    del theta
    dP = np.zeros((2, 2, 2), dtype=np.float64)

    # (s1,b1): p(s1)=theta, p(s2)=1-theta
    dP[0, 0, 0] = 1.0
    dP[0, 0, 1] = -1.0

    # (s2,b2): p(s1)=theta, p(s2)=1-theta
    dP[1, 1, 0] = 1.0
    dP[1, 1, 1] = -1.0

    # (s1,b2) and (s2,b1) are deterministic; derivatives are zero.
    return dP


def soft_value_iteration(theta: float, reward_vec: np.ndarray, gamma: float, beta: float, n_iter: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Solve follower soft Q and policy for the current theta and reward."""
    P = build_transition_tensor(theta)
    r = reward_vec.reshape(2, 2)

    V = np.zeros(2, dtype=np.float64)
    for _ in range(n_iter):
        Q = r + gamma * np.einsum("sbn,n->sb", P, V)
        z = Q / beta
        z_max = z.max(axis=1, keepdims=True)
        z_shifted = z - z_max
        V_new = beta * (np.log(np.exp(z_shifted).sum(axis=1)) + z_max.ravel())
        if np.max(np.abs(V_new - V)) < 1e-10:
            V = V_new
            break
        V = V_new

    Q = r + gamma * np.einsum("sbn,n->sb", P, V)
    logits = Q / beta
    logits = logits - logits.max(axis=1, keepdims=True)
    pi = np.exp(logits)
    pi /= pi.sum(axis=1, keepdims=True)
    return Q, pi


def _policy_state_transition(theta: float, pi: np.ndarray) -> np.ndarray:
    """Build P_pi[s,s'] = sum_b pi(b|s) P(s'|s,b;theta)."""
    P = build_transition_tensor(theta)
    return np.einsum("sb,sbn->sn", pi, P)


def exact_policy_feature_expectation(
    theta: float,
    pi: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Exact infinite-horizon discounted feature expectation.

    mu(s,b) = E[sum_t gamma^t 1{(s_t,b_t)=(s,b)}]
            = d_gamma(s) * pi(b|s),
    where d_gamma = rho0 (I - gamma P_pi)^{-1}.
    """
    rho0 = np.array([0.5, 0.5], dtype=np.float64)

    P_pi = _policy_state_transition(theta, pi)
    eye2 = np.eye(2, dtype=np.float64)
    d_gamma = rho0 @ np.linalg.inv(eye2 - gamma * P_pi)

    mu = np.zeros(4, dtype=np.float64)
    for s in range(2):
        for b in range(2):
            mu[2 * s + b] = d_gamma[s] * pi[s, b]
    return mu


def estimate_expert_feature_expectation_from_demos(
    theta: float,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate discounted feature expectation from expert demonstrations.

    The estimator uses only sampled (s,b,s') trajectories and does not use
    direct access to the expert policy table.
    """
    P = build_transition_tensor(theta)
    mu_hat = np.zeros(4, dtype=np.float64)

    for _ in range(cfg.expert_demo_episodes):
        s = int(rng.integers(0, 2))  # rho0 = uniform over two states
        discount = 1.0
        for _ in range(cfg.expert_demo_horizon):
            b = int(rng.integers(0, 2))  # expert action is observed from demos
            mu_hat[2 * s + b] += discount
            s_next = int(rng.choice(2, p=P[s, b]))
            s = s_next
            discount *= cfg.gamma

    mu_hat /= max(cfg.expert_demo_episodes, 1)
    return mu_hat


def run_local_irl(theta: float, init_w: np.ndarray, mu_expert: np.ndarray, cfg: Config) -> np.ndarray:
    """Convex MDCE-style IRL via occupancy measures.

    We solve a discounted causal-entropy program with soft FEM constraints:
      - variables: discounted state occupancy mu_s and state-action occupancy mu_sa
      - constraints: flow consistency and feature expectation matching
      - objective: minimize causal entropy + slack penalty

    The recovered reward is taken from the dual variables of the FEM
    constraints, centered to remove the constant-shift ambiguity.
    """
    del init_w

    n_s = 2
    n_b = 2
    gamma = float(cfg.gamma)
    rho0 = np.array([0.5, 0.5], dtype=np.float64)
    mu_expert = np.asarray(mu_expert, dtype=np.float64).reshape(n_s * n_b)

    P = build_transition_tensor(theta)

    # Variables: discounted occupancies
    mu_s = cp.Variable(n_s, nonneg=True)
    mu_sa = cp.Variable(n_s * n_b, nonneg=True)

    # Flow matrix: [P^T mu_sa]_s' = sum_{s,b} P(s'|s,b) mu_sa(s,b)
    flow_to = np.zeros((n_s, n_s * n_b), dtype=np.float64)
    for s in range(n_s):
        for b in range(n_b):
            idx = 2 * s + b
            flow_to[:, idx] = P[s, b]

    # Consistency between state occupancy and state-action occupancy.
    mu_s_from_sa = cp.hstack([
        mu_sa[0] + mu_sa[1],
        mu_sa[2] + mu_sa[3],
    ])

    # Causal entropy term for the policy induced by occupancy.
    mu_s_expanded = cp.hstack([
        mu_s[0], mu_s[0],
        mu_s[1], mu_s[1],
    ])
    entropy_obj = cp.sum(cp.rel_entr(mu_sa, mu_s_expanded + 1e-12))

    # Soft FEM constraints with slack; reward is recovered from the duals.
    xi = cp.Variable(n_s * n_b, nonneg=True)
    feat_expr = mu_sa
    c_pos = feat_expr - mu_expert <= xi
    c_neg = mu_expert - feat_expr <= xi

    constraints = [
        mu_s == mu_s_from_sa,
        mu_s == rho0 + gamma * (flow_to @ mu_sa),
        c_pos,
        c_neg,
    ]

    fem_penalty = 1.0
    objective = cp.Minimize(entropy_obj + fem_penalty * cp.sum(xi))
    problem = cp.Problem(objective, constraints)

    solved = False
    last_status = None
    for solver in (cp.SCS, cp.CLARABEL):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception:
            continue
        last_status = problem.status
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            solved = True
            break

    if not solved:
        raise RuntimeError(f"Convex MDCE optimization failed (status={last_status}).")

    # Recover reward parameters from the FEM dual variables.
    dual_pos = np.asarray(c_pos.dual_value, dtype=np.float64)
    dual_neg = np.asarray(c_neg.dual_value, dtype=np.float64)
    w = -(dual_pos - dual_neg)
    w = w - np.mean(w)
    w = np.clip(w, -50.0, 50.0)

    return w


def covariance_from_policy(theta: float, pi: np.ndarray, gamma: float, reg: float) -> np.ndarray:
    """Analytic proxy covariance from discounted state-action occupancy.

    Use one-step feature covariance under the normalized discounted occupancy.
    For one-hot feature phi in R^4:
      Cov[phi] = diag(p) - p p^T.
    """
    mu = exact_policy_feature_expectation(theta, pi, gamma)
    p = (1.0 - gamma) * mu  # normalize discounted occupancy to a distribution
    p = np.clip(p, 0.0, 1.0)
    p = p / max(np.sum(p), 1e-12)
    C = np.diag(p) - np.outer(p, p)
    C = 0.5 * (C + C.T)
    # C += reg * np.eye(4, dtype=np.float64)
    return C


def _soft_state_value_from_q(Q: np.ndarray, beta: float) -> np.ndarray:
    """Compute soft V from Q by log-sum-exp."""
    z = Q / beta
    z_max = z.max(axis=1, keepdims=True)
    return beta * (np.log(np.exp(z - z_max).sum(axis=1)) + z_max.ravel())


def follower_q_grad_exact(theta: float, pi: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """Exact discrete version of ∇_theta Q_F via theorem-style expectation.

    We compute G(s,b) = ∇_theta Q_F(s,b) by solving the recursion

      G(s,b) = gamma * E_{s'|s,b}[ V(s') * ∇_theta log p(s'|s,b)
                + E_{b'~pi(.|s')}[ G(s',b') ] ].

    This is the discrete counterpart of replacing integrals by finite sums.
    """
    P = build_transition_tensor(theta)
    dP = build_transition_derivative_tensor(theta)
    eps = 1e-12

    def idx(s: int, b: int) -> int:
        return 2 * s + b

    A = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)

    for s in range(2):
        for b in range(2):
            i = idx(s, b)
            A[i, i] = 1.0

            for sp in range(2):
                p = float(P[s, b, sp])
                dp = float(dP[s, b, sp])
                grad_log_p = dp / max(p, eps)

                rhs[i] += gamma * p * V[sp] * grad_log_p

                for bp in range(2):
                    j = idx(sp, bp)
                    A[i, j] -= gamma * p * pi[sp, bp]

    g = np.linalg.solve(A, rhs)
    return g.reshape(2, 2)


def kl_to_uniform(pi: np.ndarray) -> float:
    u = np.array([0.5, 0.5], dtype=np.float64)
    eps = 1e-12
    kl_s = []
    for s in range(2):
        p = np.clip(pi[s], eps, 1.0)
        kl_s.append(float(np.sum(p * (np.log(p) - np.log(u)))))
    return float(np.mean(kl_s))


def _set_plot_style() -> None:
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "figure.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
        },
    )


def _plot_series(
    ax: plt.Axes,
    series: list[float],
    label: str,
    *,
    color: str,
    linestyle: str = "-",
    marker: str | None = None,
) -> None:
    xs = np.arange(len(series))
    ax.plot(
        xs,
        np.asarray(series, dtype=np.float64),
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=max(len(series) // 10, 1),
    )


def save_plots(stats_aggregate: dict, stats_single: dict, out_dir: Path, log: Callable[[str], None] | None = None) -> None:
    _set_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_series(axes[0, 0], stats_aggregate["policy_kl_to_uniform"], "aggregate_irl", color="#0072B2", marker="o")
    _plot_series(axes[0, 0], stats_single["policy_kl_to_uniform"], "single_fixed_irl", color="#D55E00", linestyle="--", marker="s")
    axes[0, 0].set_title("Policy KL to Uniform")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("KL")
    axes[0, 0].legend()

    _plot_series(axes[0, 1], stats_aggregate["term3_proxy"], "aggregate_irl", color="#0072B2", marker="o")
    _plot_series(axes[0, 1], stats_single["term3_proxy"], "single_fixed_irl", color="#D55E00", linestyle="--", marker="s")
    axes[0, 1].set_title("Term3 Proxy")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Proxy value")
    axes[0, 1].legend()

    _plot_series(axes[1, 0], stats_aggregate["reward_centered_norm"], "aggregate_irl", color="#0072B2", marker="o")
    _plot_series(axes[1, 0], stats_single["reward_centered_norm"], "single_fixed_irl", color="#D55E00", linestyle="--", marker="s")
    axes[1, 0].set_title("Centered Reward Norm")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("L2 norm")
    axes[1, 0].legend()

    _plot_series(axes[1, 1], stats_aggregate["theta"], "theta", color="#009E73", marker="o")
    axes[1, 1].set_title("Theta Schedule")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("theta")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend()

    fig.tight_layout()
    summary_path = out_dir / "conf_discrete_summary.pdf"
    fig.savefig(str(summary_path), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = ["r(s1,b1)", "r(s1,b2)", "r(s2,b1)", "r(s2,b2)"]
    colors = ["black", "#0072B2", "#D55E00", "#009E73"]

    for idx, ax in enumerate(axes.ravel()):
        _plot_series(
            ax,
            [reward[idx] for reward in stats_aggregate["reward"]],
            "aggregate_irl",
            color=colors[idx],
            linestyle="-",
            marker="o",
        )
        _plot_series(
            ax,
            [reward[idx] for reward in stats_single["reward"]],
            "single_fixed_irl",
            color=colors[idx],
            linestyle="--",
            marker="s",
        )
        ax.set_title(labels[idx])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.legend()

    fig.tight_layout()
    reward_path = out_dir / "conf_discrete_rewards.pdf"
    fig.savefig(str(reward_path), bbox_inches="tight")
    plt.close(fig)

    if log is None:
        print(f"Saved plot: {summary_path}")
        print(f"Saved plot: {reward_path}")
    else:
        log(f"Saved plot: {summary_path}")
        log(f"Saved plot: {reward_path}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    _plot_series(ax, stats_aggregate["theta"], "aggregate_irl theta", color="#0072B2", marker="o")
    _plot_series(ax, stats_single["theta"], "single_fixed_irl theta", color="#D55E00", linestyle="--", marker="s")
    ax.set_title("Theta Schedule Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("theta")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    theta_path = out_dir / "conf_discrete_theta.pdf"
    fig.savefig(str(theta_path), bbox_inches="tight")
    plt.close(fig)

    if log is None:
        print(f"Saved plot: {theta_path}")
    else:
        log(f"Saved plot: {theta_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_series(axes[0, 0], stats_aggregate["policy_kl_to_uniform"], "aggregate_irl", color="#0072B2", marker="o")
    _plot_series(axes[0, 0], stats_single["policy_kl_to_uniform"], "single_fixed_irl", color="#D55E00", linestyle="--", marker="s")
    axes[0, 0].set_title("Policy KL to Uniform")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("KL")
    axes[0, 0].legend()

    _plot_series(axes[0, 1], stats_aggregate["term2_proxy"], "aggregate_irl", color="#0072B2", marker="o")
    _plot_series(axes[0, 1], stats_single["term2_proxy"], "single_fixed_irl", color="#D55E00", linestyle="--", marker="s")
    axes[0, 1].set_title("Term2 Proxy")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Term2")
    axes[0, 1].legend()

    _plot_series(axes[1, 0], stats_aggregate["term3_proxy"], "aggregate_irl", color="#0072B2", marker="o")
    _plot_series(axes[1, 0], stats_single["term3_proxy"], "single_fixed_irl", color="#D55E00", linestyle="--", marker="s")
    axes[1, 0].set_title("Term3 Proxy")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Term3")
    axes[1, 0].legend()

    _plot_series(axes[1, 1], stats_aggregate["theta"], "aggregate_irl theta", color="#0072B2", marker="o")
    _plot_series(axes[1, 1], stats_single["theta"], "single_fixed_irl theta", color="#D55E00", linestyle="--", marker="s")
    axes[1, 1].set_title("Theta Schedule")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("theta")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend()

    fig.tight_layout()
    terms_path = out_dir / "conf_discrete_terms.pdf"
    fig.savefig(str(terms_path), bbox_inches="tight")
    plt.close(fig)

    if log is None:
        print(f"Saved plot: {terms_path}")
    else:
        log(f"Saved plot: {terms_path}")


def _format_run_summary(cfg: Config, agg: dict, single: dict, out_dir: Path) -> str:
    lines = [
        "Synthetic discrete ConfMDP run summary",
        f"output_dir: {out_dir}",
        f"seed: {cfg.seed}",
        f"n_iterations: {cfg.n_iterations}",
        f"theta0: {cfg.theta0}",
        f"a_bias: {cfg.a_bias}",
        "",
        "Final metrics:",
        f"aggregate_irl final policy_kl_to_uniform: {agg['policy_kl_to_uniform'][-1]}",
        f"aggregate_irl final term2_proxy: {agg['term2_proxy'][-1]}",
        f"aggregate_irl final term3_proxy: {agg['term3_proxy'][-1]}",
        f"aggregate_irl final reward_centered_norm: {agg['reward_centered_norm'][-1]}",
        f"single_fixed_irl final policy_kl_to_uniform: {single['policy_kl_to_uniform'][-1]}",
        f"single_fixed_irl final term2_proxy: {single['term2_proxy'][-1]}",
        f"single_fixed_irl final term3_proxy: {single['term3_proxy'][-1]}",
        f"single_fixed_irl final reward_centered_norm: {single['reward_centered_norm'][-1]}",
        f"aggregate_irl theta schedule: {agg['theta']}",
        f"single_fixed_irl theta schedule: {single['theta']}",
        "",
        "Saved files:",
        f"- stats_aggregate_seed{cfg.seed}.pkl",
        f"- stats_single_fixed_seed{cfg.seed}.pkl",
        "- conf_discrete_summary.pdf",
        "- conf_discrete_terms.pdf",
        "- conf_discrete_rewards.pdf",
        "- conf_discrete_theta.pdf",
    ]
    return "\n".join(lines)


def term2_proxy(theta: float, w: np.ndarray, cfg: Config) -> float:
    """Exact scalar diagnostic for the transition-sensitivity term.

    term2 = E[d_gamma(s) pi(b|s) P(s'|s,b) V(s') ∂_theta log P(s'|s,b)].
    """
    P = build_transition_tensor(theta)
    dP = build_transition_derivative_tensor(theta)
    Q, pi = soft_value_iteration(theta, w, cfg.gamma, cfg.beta)
    V = _soft_state_value_from_q(Q, cfg.beta)

    mu = exact_policy_feature_expectation(theta, pi, cfg.gamma)
    p = (1.0 - cfg.gamma) * mu
    p = np.clip(p, 0.0, 1.0)
    p = p / max(np.sum(p), 1e-12)

    term2 = 0.0
    eps = 1e-12
    for s in range(2):
        for b in range(2):
            idx = 2 * s + b
            for sp in range(2):
                prob = float(P[s, b, sp])
                if prob <= 0.0:
                    continue
                dlogp = float(dP[s, b, sp]) / max(prob, eps)
                term2 += p[idx] * prob * float(V[sp]) * dlogp

    return float(term2)


def term3_proxy(theta: float, w: np.ndarray, cfg: Config) -> float:
    """Theorem-style discrete term-3 statistic.

    term3 ~= E_{(s,b)~d_gamma^pi}[ (A_F(s,b)/beta) * ∇_theta Q_F(s,b) ],
    where A_F is approximated by follower soft advantage.
    """
    Q, pi = soft_value_iteration(theta, w, cfg.gamma, cfg.beta)
    V = _soft_state_value_from_q(Q, cfg.beta)
    dQ = follower_q_grad_exact(theta, pi, V, cfg.gamma)

    mu = exact_policy_feature_expectation(theta, pi, cfg.gamma)
    p = (1.0 - cfg.gamma) * mu
    p = np.clip(p, 0.0, 1.0)
    p = p / max(np.sum(p), 1e-12)

    total = 0.0
    for s in range(2):
        baseline = float(np.sum(pi[s] * Q[s]))
        for b in range(2):
            adv = Q[s, b] - baseline
            idx = 2 * s + b
            total += p[idx] * (adv / cfg.beta) * dQ[s, b]
    return float(total)


def run_method(method: str, cfg: Config, w_init: np.ndarray, log: Callable[[str], None] | None = None) -> dict:
    w_init = np.asarray(w_init, dtype=np.float64).copy()
    expert_rng = np.random.default_rng(cfg.seed + 2026)

    sigma_bar = cfg.reg * np.eye(4, dtype=np.float64)
    v_acc = np.zeros(4, dtype=np.float64)
    w_current = w_init.copy()
    theta_current = float(cfg.theta0)

    def emit(message: str) -> None:
        if log is None:
            print(message, flush=True)
        else:
            log(message)

    emit(f"[{method}] init_reward={np.array2string(w_init, precision=4, separator=', ')}")

    theta_lo = max(cfg.theta_min, cfg.theta_eps)
    theta_hi = min(cfg.theta_max, 1.0 - cfg.theta_eps)
    if not theta_lo < theta_hi:
        raise ValueError(
            f"Invalid theta bounds after epsilon clip: lo={theta_lo}, hi={theta_hi}. "
            "Adjust theta_min/theta_max/theta_eps."
        )

    stats = {
        "method": method,
        "theta": [],
        "term2_proxy": [],
        "policy_kl_to_uniform": [],
        "term3_proxy": [],
        "reward": [],
        "reward_centered": [],
        "reward_centered_norm": [],
    }

    # === 両方のメソッドに対して初期バイアス報酬を確信度行列に組み込む（フェアな比較） ===
    # w_init を theta0 環境でのIRL結果とみなし、その環境下での確信度を計算
    _, pi_init = soft_value_iteration(float(cfg.theta0), w_init, cfg.gamma, cfg.beta)
    Sigma_init = covariance_from_policy(float(cfg.theta0), pi_init, cfg.gamma, cfg.reg)

    eigvals_init, eigvecs_init = np.linalg.eigh(Sigma_init)
    mask_init = (eigvals_init >= cfg.eig_threshold).astype(np.float64)
    U_init = eigvecs_init @ np.diag(mask_init) @ eigvecs_init.T
    U_init = 0.5 * (U_init + U_init.T)

    # run_local_irl() の出力は平均0に中心化されるため、初期項も同じゲージに揃える。
    w_init_centered = w_init - np.mean(w_init)

    # 最初のデタラメな結果をアグリゲートの土台にする
    sigma_bar = sigma_bar + U_init
    v_acc = v_acc + U_init @ w_init_centered

    # 初期表示点(iter=0)は両手法で同じ w_init に揃える。
    # 集約の効果はループ内で next_w_current を通して反映する。
    if method == "single_fixed_irl":
        w_current = w_init.copy()
    # ======================================================================

    for it in range(cfg.n_iterations):
        theta = float(np.clip(theta_current, theta_lo, theta_hi))
        mu_expert = estimate_expert_feature_expectation_from_demos(theta, cfg, expert_rng)
        w_eval = w_current.copy()

        if method == "aggregate_irl":
            w_k = run_local_irl(theta, w_current, mu_expert, cfg)
            _, pi_k = soft_value_iteration(theta, w_k, cfg.gamma, cfg.beta)
            Sigma_k = covariance_from_policy(theta, pi_k, cfg.gamma, cfg.reg)

            eigvals, eigvecs = np.linalg.eigh(Sigma_k)
            mask = (eigvals >= cfg.eig_threshold).astype(np.float64)
            U_k = eigvecs @ np.diag(mask) @ eigvecs.T
            U_k = 0.5 * (U_k + U_k.T)

            sigma_bar = sigma_bar + U_k
            v_acc = v_acc + U_k @ w_k
            next_w_current = np.linalg.solve(sigma_bar + cfg.reg * np.eye(4), v_acc)

        elif method == "single_fixed_irl":
            next_w_current = w_init.copy()
        else:
            raise ValueError(f"Unknown method: {method}")

        _, pi = soft_value_iteration(theta, w_eval, cfg.gamma, cfg.beta)
        reward_centered = w_eval - np.mean(w_eval)
        term2_val = term2_proxy(theta, w_eval, cfg)

        stats["theta"].append(theta)
        stats["term2_proxy"].append(term2_val)
        stats["policy_kl_to_uniform"].append(kl_to_uniform(pi))
        stats["term3_proxy"].append(term3_proxy(theta, w_eval, cfg))
        stats["reward"].append(w_eval.tolist())
        stats["reward_centered"].append(reward_centered.tolist())
        stats["reward_centered_norm"].append(float(np.linalg.norm(reward_centered)))

        emit(
            f"[{method}] iter={it:03d} theta={theta:.4f} "
            f"term2={term2_val:.6g} "
            f"kl={stats['policy_kl_to_uniform'][-1]:.6g} "
            f"term3={stats['term3_proxy'][-1]:.6g} "
            f"reward_norm={stats['reward_centered_norm'][-1]:.6g} "
            f"reward={np.array2string(w_eval, precision=4, separator=', ')}"
        )

        theta_grad_proxy = term2_val + stats["term3_proxy"][-1]
        step = cfg.theta_lr * theta_grad_proxy
        theta_current = float(np.clip(theta + step, theta_lo, theta_hi))

        w_current = next_w_current

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic discrete ConfMDP experiment")
    p.add_argument("--output-dir", type=str, default="data/internal/conf_discrete_synth1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iterations", type=int, default=200)
    p.add_argument("--theta0", type=float, default=0.5)
    p.add_argument("--theta-lr", type=float, default=0.05)
    p.add_argument("--theta-min", type=float, default=0.0)
    p.add_argument("--theta-max", type=float, default=1.0)
    p.add_argument("--theta-eps", type=float, default=1e-3)
    p.add_argument("--a-bias", type=float, default=10.0)
    p.add_argument("--expert-demo-episodes", type=int, default=200)
    p.add_argument("--expert-demo-horizon", type=int, default=30)
    args = p.parse_args()

    cfg = Config(
        seed=args.seed,
        n_iterations=args.n_iterations,
        theta0=args.theta0,
        theta_lr=args.theta_lr,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_eps=args.theta_eps,
        a_bias=args.a_bias,
        expert_demo_episodes=args.expert_demo_episodes,
        expert_demo_horizon=args.expert_demo_horizon,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    base_reward = np.array([cfg.a_bias, cfg.a_bias, 0.0, 0.0], dtype=np.float64)
    w_init = rng.multivariate_normal(mean=base_reward, cov=np.eye(4, dtype=np.float64))

    log_path = out / f"run_seed{cfg.seed}.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log("Synthetic discrete ConfMDP run started")
        log(f"output_dir: {out}")
        log(f"seed: {cfg.seed}")
        log(f"n_iterations: {cfg.n_iterations}")
        log(f"theta0: {cfg.theta0}")
        log(f"theta_lr: {cfg.theta_lr}")
        log(f"theta_min: {cfg.theta_min}")
        log(f"theta_max: {cfg.theta_max}")
        log(f"theta_eps: {cfg.theta_eps}")
        log(f"a_bias: {cfg.a_bias}")
        log(f"expert_demo_episodes: {cfg.expert_demo_episodes}")
        log(f"expert_demo_horizon: {cfg.expert_demo_horizon}")
        log(f"init_reward: {np.array2string(w_init, precision=4, separator=', ')}")

        agg = run_method("aggregate_irl", cfg, w_init=w_init, log=log)
        single = run_method("single_fixed_irl", cfg, w_init=w_init, log=log)

        with open(out / f"stats_aggregate_seed{cfg.seed}.pkl", "wb") as f:
            pickle.dump(agg, f)
        with open(out / f"stats_single_fixed_seed{cfg.seed}.pkl", "wb") as f:
            pickle.dump(single, f)

        save_plots(agg, single, out, log=log)

        log(f"Saved: {out / f'stats_aggregate_seed{cfg.seed}.pkl'}")
        log(f"Saved: {out / f'stats_single_fixed_seed{cfg.seed}.pkl'}")

        summary = _format_run_summary(cfg, agg, single, out)
        log_file.write("\n")
        log_file.write(summary)
        log_file.write("\n")
        log_file.flush()

    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
