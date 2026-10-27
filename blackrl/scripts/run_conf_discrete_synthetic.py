from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
import numpy as np
import scienceplots  # noqa: F401


@dataclass
class Config:
    output_dir: str = "data/internal/conf_discrete_synth2"
    n_states: int = 2
    n_actions: int = 2
    gamma: float = 0.95
    beta: float = 0.25
    n_iterations: int = 200
    theta0: float = 0.9
    theta_lr: float = 0.05
    theta_min: float = 0.0
    theta_max: float = 1.0
    theta_eps: float = 1e-3
    seed: int = 0
    a_bias: float = 10.0
    eig_threshold: float = 1e-6


def transition_prob(theta: float, s: int, b: int) -> np.ndarray:
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
    P = build_transition_tensor(theta)
    return np.einsum("sb,sbn->sn", pi, P)


def exact_policy_feature_expectation(
    theta: float,
    pi: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rho0 = np.array([0.5, 0.5], dtype=np.float64)

    P_pi = _policy_state_transition(theta, pi)
    eye2 = np.eye(2, dtype=np.float64)
    d_gamma = rho0 @ np.linalg.inv(eye2 - gamma * P_pi)

    mu = np.zeros(4, dtype=np.float64)
    for s in range(2):
        for b in range(2):
            mu[2 * s + b] = d_gamma[s] * pi[s, b]
    return mu


def exact_expert_feature_expectation_uniform_policy(
    theta: float,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
   
    del rng
    pi_expert = np.full((2, 2), 0.5, dtype=np.float64)
    return exact_policy_feature_expectation(theta, pi_expert, cfg.gamma)


def run_local_irl(theta: float, init_w: np.ndarray, mu_expert: np.ndarray, cfg: Config) -> np.ndarray:
  
    del init_w

    n_s = 2
    n_b = 2
    gamma = float(cfg.gamma)
    rho0 = np.array([0.5, 0.5], dtype=np.float64)
    mu_expert = np.asarray(mu_expert, dtype=np.float64).reshape(n_s * n_b)

    P = build_transition_tensor(theta)

    mu_s = cp.Variable(n_s, nonneg=True)
    mu_sa = cp.Variable(n_s * n_b, nonneg=True)

    flow_to = np.zeros((n_s, n_s * n_b), dtype=np.float64)
    for s in range(n_s):
        for b in range(n_b):
            idx = 2 * s + b
            flow_to[:, idx] = P[s, b]

    mu_s_from_sa = cp.hstack([
        mu_sa[0] + mu_sa[1],
        mu_sa[2] + mu_sa[3],
    ])

    mu_s_expanded = cp.hstack([
        mu_s[0], mu_s[0],
        mu_s[1], mu_s[1],
    ])
    entropy_obj = cp.sum(cp.rel_entr(mu_sa, mu_s_expanded))

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

    for solver in (cp.SCS, cp.CLARABEL):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break


    dual_pos = np.asarray(c_pos.dual_value, dtype=np.float64)
    dual_neg = np.asarray(c_neg.dual_value, dtype=np.float64)
    w = -(dual_pos - dual_neg)

    return w


def covariance_from_policy(theta: float, pi: np.ndarray, gamma: float) -> np.ndarray:
    mu = exact_policy_feature_expectation(theta, pi, gamma)
    p = (1.0 - gamma) * mu  # normalize discounted occupancy to a distribution
    p = p / np.sum(p)
    C = np.diag(p) - np.outer(p, p)
    C = 0.5 * (C + C.T)
    return C


def _soft_state_value_from_q(Q: np.ndarray, beta: float) -> np.ndarray:
    z = Q / beta
    z_max = z.max(axis=1, keepdims=True)
    return beta * (np.log(np.exp(z - z_max).sum(axis=1)) + z_max.ravel())


def follower_q_grad_exact(theta: float, pi: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    P = build_transition_tensor(theta)
    dP = build_transition_derivative_tensor(theta)

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
                rhs[i] += gamma * dp * V[sp]

                for bp in range(2):
                    j = idx(sp, bp)
                    A[i, j] -= gamma * p * pi[sp, bp]

    g = np.linalg.solve(A, rhs)
    return g.reshape(2, 2)


def kl_to_uniform(pi: np.ndarray) -> float:
    u = np.array([0.5, 0.5], dtype=np.float64)
    kl_s = []
    for s in range(2):
        p = pi[s]
        kl_s.append(float(np.sum(p * (np.log(p) - np.log(u)))))
    return float(np.mean(kl_s))


def _set_plot_style() -> None:
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
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
    ax: Axes,
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

    reward_path = out_dir / "conf_discrete_rewards.pdf"
    with PdfPages(reward_path) as pdf:
        fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
        labels = ["r(s1,b1)", "r(s1,b2)", "r(s2,b1)", "r(s2,b2)"]

        for idx, ax in enumerate(axes.ravel()):
            _plot_series(
                ax,
                [reward[idx] for reward in stats_aggregate["reward"]],
                "RE-HG",
                color="#D55E00",
                linestyle="-",
                marker="o",
            )
            _plot_series(
                ax,
                [reward[idx] for reward in stats_single["reward"]],
                "RE-HG (Single)",
                color="#0072B2",
                linestyle="--",
                marker="s",
            )
            ax.set_title(labels[idx])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Reward")
            ax.legend()

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    kl_theta_path = out_dir / "conf_discrete_kl_theta.pdf"
    with PdfPages(kl_theta_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        _plot_series(axes[0], stats_aggregate["policy_kl_to_uniform"], "RE-HG", color="#0072B2", marker="o")
        _plot_series(axes[0], stats_single["policy_kl_to_uniform"], "RE-HG (Single)", color="#D55E00", linestyle="--", marker="s")
        axes[0].set_title("Policy KL to Uniform")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("KL")
        axes[0].legend()

        _plot_series(axes[1], stats_aggregate["theta"], "RE-HG", color="#0072B2", marker="o")
        _plot_series(axes[1], stats_single["theta"], "RE-HG (Single)", color="#D55E00", linestyle="--", marker="s")
        axes[1].set_title("Theta Schedule")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("theta")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    term_path = out_dir / "conf_discrete_terms.pdf"
    with PdfPages(term_path) as pdf:
        fig_terms, axes_terms = plt.subplots(1, 3, figsize=(18, 4.5))

        _plot_series(axes_terms[0], stats_aggregate["term1_exact"], "RE-HG", color="#0072B2", marker="o")
        _plot_series(axes_terms[0], stats_single["term1_exact"], "RE-HG (Single)", color="#D55E00", linestyle="--", marker="s")
        axes_terms[0].set_title("Term1")
        axes_terms[0].set_xlabel("Iteration")
        axes_terms[0].set_ylabel("Term value")
        axes_terms[0].legend()

        _plot_series(axes_terms[1], stats_aggregate["term2_exact"], "RE-HG", color="#0072B2", marker="^")
        _plot_series(axes_terms[1], stats_single["term2_exact"], "RE-HG (Single)", color="#D55E00", linestyle="--", marker="v")
        axes_terms[1].set_title("Term2")
        axes_terms[1].set_xlabel("Iteration")
        axes_terms[1].set_ylabel("Term value")
        axes_terms[1].legend()

        _plot_series(axes_terms[2], stats_aggregate["term3_exact"], "RE-HG", color="#0072B2", marker="d")
        _plot_series(axes_terms[2], stats_single["term3_exact"], "RE-HG (Single)", color="#D55E00", linestyle="--", marker="x")
        axes_terms[2].set_title("Term3")
        axes_terms[2].set_xlabel("Iteration")
        axes_terms[2].set_ylabel("Term value")
        axes_terms[2].legend()

        fig_terms.tight_layout()
        pdf.savefig(fig_terms, bbox_inches="tight")
        plt.close(fig_terms)

    if log is None:
        print(f"Saved plot: {reward_path}")
        print(f"Saved plot: {kl_theta_path}")
        print(f"Saved plot: {term_path}")
    else:
        log(f"Saved plot: {reward_path}")
        log(f"Saved plot: {kl_theta_path}")
        log(f"Saved plot: {term_path}")


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
        f"aggregate_irl final term1_exact: {agg['term1_exact'][-1]}",
        f"aggregate_irl final term2_exact: {agg['term2_exact'][-1]}",
        f"aggregate_irl final term3_exact: {agg['term3_exact'][-1]}",
        f"single_fixed_irl final policy_kl_to_uniform: {single['policy_kl_to_uniform'][-1]}",
        f"single_fixed_irl final term1_exact: {single['term1_exact'][-1]}",
        f"single_fixed_irl final term2_exact: {single['term2_exact'][-1]}",
        f"single_fixed_irl final term3_exact: {single['term3_exact'][-1]}",
        f"aggregate_irl theta schedule: {agg['theta']}",
        f"single_fixed_irl theta schedule: {single['theta']}",
        "",
        "Saved files:",
        f"- stats_aggregate_seed{cfg.seed}.pkl",
        f"- stats_single_fixed_seed{cfg.seed}.pkl",
        "- conf_discrete_report.pdf",
    ]
    return "\n".join(lines)


def term2_exact(theta: float, w: np.ndarray, cfg: Config) -> float:
    dP = build_transition_derivative_tensor(theta)
    Q, pi = soft_value_iteration(theta, w, cfg.gamma, cfg.beta)
    _ = Q

    q_l, v_l, _ = leader_q_and_benefit(theta, pi, cfg.gamma)
    del q_l

    mu = exact_policy_feature_expectation(theta, pi, cfg.gamma)

    term2 = 0.0
    for s in range(2):
        for b in range(2):
            idx = 2 * s + b
            for sp in range(2):
                term2 += mu[idx] * cfg.gamma * float(v_l[sp]) * float(dP[s, b, sp])

    return float(term2)


def term3_exact(theta: float, w: np.ndarray, cfg: Config) -> float:
    Q, pi = soft_value_iteration(theta, w, cfg.gamma, cfg.beta)
    V = _soft_state_value_from_q(Q, cfg.beta)
    dQ = follower_q_grad_exact(theta, pi, V, cfg.gamma)
    _, _, b_l = leader_q_and_benefit(theta, pi, cfg.gamma)

    mu = exact_policy_feature_expectation(theta, pi, cfg.gamma)

    total = 0.0
    for s in range(2):
        for b in range(2):
            idx = 2 * s + b
            total += mu[idx] * (float(b_l[s, b]) / cfg.beta) * dQ[s, b]
    return float(total)


def term1_exact(theta: float, w: np.ndarray, cfg: Config) -> float:
    Q, pi = soft_value_iteration(theta, w, cfg.gamma, cfg.beta)
    V = _soft_state_value_from_q(Q, cfg.beta)
    dQ = follower_q_grad_exact(theta, pi, V, cfg.gamma)
    mu = exact_policy_feature_expectation(theta, pi, cfg.gamma)

    total = 0.0
    for s in range(2):
        baseline = float(np.dot(pi[s], dQ[s]))
        for b in range(2):
            idx = 2 * s + b
            dr_dg = -2.0 * (float(pi[s, b]) - 0.5)
            d_pi = (float(pi[s, b]) / cfg.beta) * (float(dQ[s, b]) - baseline)
            total += mu[idx] * dr_dg * d_pi

    return float(total)


def leader_immediate_reward_from_policy(pi: np.ndarray) -> np.ndarray:
    pi_arr = np.asarray(pi, dtype=np.float64)
    return -((pi_arr - 0.5) ** 2)


def leader_q_and_benefit(theta: float, pi: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = build_transition_tensor(theta)
    r_l = leader_immediate_reward_from_policy(pi)

    q_l = np.zeros((2, 2), dtype=np.float64)
    for _ in range(1000):
        q_old = q_l.copy()
        v_old = np.einsum("sb,sb->s", pi, q_old)
        q_l = r_l + gamma * np.einsum("sbn,n->sb", P, v_old)
        if np.max(np.abs(q_l - q_old)) < 1e-12:
            break

    v_l = np.einsum("sb,sb->s", pi, q_l)
    b_l = q_l - v_l[:, None]
    return q_l, v_l, b_l


def theta_penalty(theta: float) -> float:
    t = float(theta)
    return -((t - 0.5) ** 2)


def discounted_policy_kl_to_uniform(theta: float, pi: np.ndarray, gamma: float) -> float:
    mu = exact_policy_feature_expectation(theta, pi, gamma)
    d_gamma = np.array([mu[0] + mu[1], mu[2] + mu[3]], dtype=np.float64)

    kl = 0.0
    log_u = np.log(0.5)
    for s in range(2):
        p = pi[s]
        kl_s = float(np.sum(p * (np.log(p) - log_u)))
        kl += float(d_gamma[s]) * kl_s
    return float(kl)


def leader_total_objective(theta: float, pi: np.ndarray, gamma: float) -> float:
    mu = exact_policy_feature_expectation(theta, pi, gamma)
    r_l = leader_immediate_reward_from_policy(pi).reshape(-1)
    return float(np.dot(mu, r_l))


def run_method(method: str, cfg: Config, w_init: np.ndarray, log: Callable[[str], None] | None = None) -> dict:
    w_init = np.asarray(w_init, dtype=np.float64).copy()
    expert_rng = np.random.default_rng(cfg.seed + 2026)

    sigma_bar = np.zeros((4, 4), dtype=np.float64)
    v_acc = np.zeros(4, dtype=np.float64)
    w_current = w_init.copy()
    theta_current = float(cfg.theta0)

    def emit(message: str) -> None:
        if log is None:
            print(message, flush=True)
        else:
            log(message)

    emit(f"[{method}] init_reward={np.array2string(w_init, precision=4, separator=', ')}")

    theta_lo = cfg.theta_min
    theta_hi = cfg.theta_max
    if not theta_lo < theta_hi:
        raise ValueError(
            f"Invalid theta bounds after epsilon clip: lo={theta_lo}, hi={theta_hi}. "
            "Adjust theta_min/theta_max/theta_eps."
        )

    stats = {
        "method": method,
        "theta": [],
        "leader_reward": [],
        "term1_exact": [],
        "term2_exact": [],
        "policy_kl_to_uniform": [],
        "term3_exact": [],
        "reward": [],
    }

    # Treat w_init as the IRL result in the theta0 environment and compute the confidence under that environment
    _, pi_init = soft_value_iteration(float(cfg.theta0), w_init, cfg.gamma, cfg.beta)
    Sigma_init = covariance_from_policy(float(cfg.theta0), pi_init, cfg.gamma)

    eigvals_init, eigvecs_init = np.linalg.eigh(Sigma_init)
    mask_init = (eigvals_init >= cfg.eig_threshold).astype(np.float64)
    U_init = eigvecs_init @ np.diag(mask_init) @ eigvecs_init.T
    U_init = 0.5 * (U_init + U_init.T)

    sigma_bar = sigma_bar + U_init
    v_acc = v_acc + U_init @ w_init

    if method == "single_fixed_irl":
        w_current = w_init.copy()

    for it in range(cfg.n_iterations):
        theta = float(np.clip(theta_current, theta_lo, theta_hi))
        mu_expert = exact_expert_feature_expectation_uniform_policy(theta, cfg, expert_rng)
        w_eval = w_current.copy()

        if method == "aggregate_irl":
            w_k = run_local_irl(theta, w_current, mu_expert, cfg)
            _, pi_k = soft_value_iteration(theta, w_k, cfg.gamma, cfg.beta)
            Sigma_k = covariance_from_policy(theta, pi_k, cfg.gamma)

            eigvals, eigvecs = np.linalg.eigh(Sigma_k)
            mask = (eigvals >= cfg.eig_threshold).astype(np.float64)
            U_k = eigvecs @ np.diag(mask) @ eigvecs.T
            U_k = 0.5 * (U_k + U_k.T)

            sigma_bar = sigma_bar + U_k
            v_acc = v_acc + U_k @ w_k
            next_w_current = np.linalg.pinv(sigma_bar) @ v_acc

        elif method == "single_fixed_irl":
            next_w_current = w_init.copy()
        else:
            raise ValueError(f"Unknown method: {method}")

        _, pi = soft_value_iteration(theta, w_eval, cfg.gamma, cfg.beta)
        term1_val = term1_exact(theta, w_eval, cfg)
        term2_val = term2_exact(theta, w_eval, cfg)
        term3_val = term3_exact(theta, w_eval, cfg)
        leader_reward_val = leader_total_objective(theta, pi, cfg.gamma)

        stats["theta"].append(theta)
        stats["leader_reward"].append(leader_reward_val)
        stats["term1_exact"].append(term1_val)
        stats["term2_exact"].append(term2_val)
        stats["policy_kl_to_uniform"].append(kl_to_uniform(pi))
        stats["term3_exact"].append(term3_val)
        stats["reward"].append(w_eval.tolist())

        emit(
            f"[{method}] iter={it:03d} theta={theta:.4f} "
            f"rL={leader_reward_val:.6g} "
            f"term1={term1_val:.6g} "
            f"term2={term2_val:.6g} "
            f"kl={stats['policy_kl_to_uniform'][-1]:.6g} "
            f"term3={term3_val:.6g} "
            f"reward={np.array2string(w_eval, precision=4, separator=', ')}"
        )

        penalty_grad = -2.0 * (theta - 0.5)
        theta_grad = penalty_grad + term1_val + term2_val + term3_val
        step = cfg.theta_lr * theta_grad
        theta_current = float(np.clip(theta + step, theta_lo, theta_hi))

        w_current = next_w_current

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic discrete ConfMDP experiment")
    p.add_argument("--output-dir", type=str, default="data/internal/conf_discrete_synth2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iterations", type=int, default=500)
    p.add_argument("--theta0", type=float, default=0.25)
    p.add_argument("--theta-lr", type=float, default=0.01)
    p.add_argument("--theta-min", type=float, default=0.0)
    p.add_argument("--theta-max", type=float, default=1.0)
    p.add_argument("--theta-eps", type=float, default=1e-3)
    p.add_argument("--a-bias", type=float, default=1.5)
    args = p.parse_args()

    cfg = Config(
        output_dir=args.output_dir,
        seed=args.seed,
        n_iterations=args.n_iterations,
        theta0=args.theta0,
        theta_lr=args.theta_lr,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_eps=args.theta_eps,
        a_bias=args.a_bias,
    )

    out = Path(cfg.output_dir)
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