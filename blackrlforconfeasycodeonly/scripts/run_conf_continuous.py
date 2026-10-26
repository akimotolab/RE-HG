
from __future__ import annotations

import argparse
import importlib
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from blackrl.algos.lqr_analytical import solve_confmdp_lqr
from blackrl.algos.conf_mdp_theorem1_gradient import theorem1_episode_gradient
from scipy.linalg import solve_discrete_lyapunov

# ---------------------------------------------------------------------------
# Environment constants (top-level for worker pickling)
# ---------------------------------------------------------------------------
_ENV_CLASS_NAME  = "BuildingThermalControlEnv"
_ENV_MODULE_PATH = "blackrl.envs.configurable_mdp_continuous_env"

# === 🌟 実行スクリプト側を Source of Truth にする 🌟 ===
_ENV_INIT_KWARGS: dict = {
    "num_zones": 4,
    "num_hvac_units": 2,
    "max_episode_steps": 100,
    "A": np.array([
        [1.0,  0.05,  0.0,  0.05],
        [0.03, 1.0,  0.04,  0.0 ],
        [0.0,  0.04,  1.0,  0.06],
        [0.05, 0.0,  0.03,  1.0 ],
    ], dtype=np.float64),
    "B": np.array([
        [0.1,  0.0 ],
        [0.6,  0.0 ],
        [0.0,  0.55],
        [0.0,  0.3 ],
    ], dtype=np.float64),
    "Q": np.diag([8.0, 1.0, 5.0, 6.0]).astype(np.float64),
    "R": 0.01 * np.eye(2, dtype=np.float64),
    "insulation_effect": np.array([0.04, 0.03, 0.06, 0.05], dtype=np.float64),
    "transition_std": float(np.sqrt(0.02)),
    "init_temp_std": 5.0,
    "temperature_range": (-30.0, 30.0),
    "hvac_range": (-10.0, 10.0),
    "stability_weight": 1.0,
    "energy_weight": 0.5,
    "insulation_cost_weight": 0.1,
    "airflow_cost_weight": 0.1,
    "init_temp_mean": np.zeros(4, dtype=np.float64),
}


def make_env():
    from blackrl.envs.configurable_mdp_continuous_env import BuildingThermalControlEnv
    return BuildingThermalControlEnv(**_ENV_INIT_KWARGS)


# ---------------------------------------------------------------------------
# Top-level picklable worker for ConfMDP continuous
#
# Follower: b = Ks_f @ s + bias_f  (+ Gaussian noise if Sigma_f is not None)
# Leader action theta is constant for the whole episode.
# ---------------------------------------------------------------------------

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
    """軌跡収集のみ行う worker。勾配計算はメインプロセスで theorem1_episode_gradient が担う。"""
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


# ---------------------------------------------------------------------------
# Parallel collection — per-episode (theta, Ks_f, bias_f, Sigma_f)
# ---------------------------------------------------------------------------

def _collect_episodes_parallel(
    per_episode_params: list[tuple],
    discount_leader: float,
    base_seed: int,
    max_workers: int,
) -> tuple[dict, list]:
    """軌跡を並列収集する。per_episode_params の各要素: (theta, Ks_f, bias_f, Sigma_f)。
    勾配計算はしない。メインプロセスで theorem1_episode_gradient を呼ぶこと。
    """
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


# ---------------------------------------------------------------------------
# Leader policy: θ = μ（決定論的）  conf-MDP では θ は固定パラメータ
# ---------------------------------------------------------------------------

class ConstantEpisodeLeaderPolicy:
    """Conf-MDP のリーダーポリシー。

    論文通り θ は確率変数ではなく決定論的パラメータ。
    複数エピソードは θ の多様性のためではなく、
    環境ノイズ・フォロワーノイズを平均化するために使用する。
    """

    def __init__(self, n_theta: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.mu = rng.uniform(0.0, 0.3, size=n_theta).astype(np.float64)

    def get_theta(self) -> np.ndarray:
        """決定論的に θ = μ を返す（クリップして [0, 1]^(2n) に収める）。"""
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
# Feature Expectation Matching (FEM) IRL  ← conf-MDP 用
# • conf-MDP は毎ステップのリーダー行動 a_t が存在しないので
#   φ(s, b) = [s², b²] が正しい特徴量設計（a は入らない）
# • 2PMG の _quad_features(s, a, b) と対比することで両設定の構造的違いが明確になる
# ---------------------------------------------------------------------------

def _conf_quad_features(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """特徴量ベクトル φ(s, b) = [s₁², ..., sₙ², b₁², ..., bₘ²] (conf-MDP用).

    論文の仮定 1: φ: S×B → R^K
    conf-MDPではリーダーが A(θ) をプログラムするため，フォロワーの観測は (s, b) のみ．
    フォロワーコスト: r_F = -(s'Qs + b'Rb) → φ(s,b)·w で表現できる．

    2PMGとの結決的な違い:
      2PMG: _quad_features(s, a, b) ← 嬕数に a を含む (だがコストには不使用)
      conf: _conf_quad_features(s, b) ← a の概念こそ存在しない
    """
    return np.concatenate([s ** 2, b ** 2])


def compute_expert_fev_conf(
    expert_trajs: list[dict],
    gamma_F: float,
) -> np.ndarray:
    """エキスパート軌跡から割引特徴量期待値を計算 (conf-MDP版).

        φ̄_BR^{γ_F} = E^{g*}[ Σ_t γ_F^t φ(s_t, b_t) ]

    conf-MDPでは a が存在しないので φ(s, b) だけで足りる．

    Parameters
    ----------
    expert_trajs : list of dicts with 's', 'b' keys (time-series per episode)
    gamma_F      : float  フォロワー割引率
    """
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
    """割引 Lyapunov 方程式による μ_policy の解析的計算 (conf-MDP版, サンプリング不要).

    2PMG版との対比:
      conf-MDP: a が存在しないので閉ループ行列が単純
        A_cl = A_theta_ref - B @ Ks
        Q_noise = B (β/2) S⁻¹ B' + σ_env² I
      2PMG: K_eff = Ks + Ka K_θ が必要

    割引 Lyapunov 方程式:
        M = γ A_cl M A_cl' + Σ_0 + γ/(1-γ) Q_noise

    FEV:
        μ_s = diag(M)
        μ_b = diag(Ks M Ks') + (1/(1-γ)) diag((β/2) S⁻¹)
    """
    sol = solve_confmdp_lqr(A_theta_ref, env.B, Q, R, gamma_F)


    n_s = A_theta_ref.shape[0]
    m   = env.B.shape[1]
    Ks    = sol["Ks"]      # (m, n)
    S_inv = sol["S_inv"]   # (m, m)

    # ------------------------------------------------------------------
    # Step 1: 閉ループ行列  (conf-MDP は a なしなので 2PMG より単純)
    # b_t = -Ks s_t + ε_b,  ε_b ~ N(0, (β/2) S⁻¹)
    # s_{t+1} = A_theta s_t + B b_t + ε_env
    #         = (A_theta - B Ks) s_t + B ε_b + ε_env
    # ------------------------------------------------------------------
    A_cl = A_theta_ref - env.B @ Ks

    Cov_b = (beta / 2.0) * S_inv          # MaxEntフォロワーノイズ
    Q_noise = (
        env.B @ Cov_b @ env.B.T           # フォロワーノイズの伝播
        + (env.transition_std ** 2) * np.eye(n_s)    # 環境ノイズ
    )

    # ------------------------------------------------------------------
    # Step 2: 割引 Lyapunov 方程式
    #   M = γ A_cl M A_cl' + Σ_0 + γ/(1-γ) Q_noise
    #   solve_discrete_lyapunov(√γ A_cl, Σ_0 + γ/(1-γ) Q_noise)
    # ------------------------------------------------------------------
    Sigma_0 = (env.init_temp_std ** 2) * np.eye(n_s)

    sq_g   = np.sqrt(gamma_F)
    Q_lyap = Sigma_0 + (gamma_F / (1.0 - gamma_F)) * Q_noise
    M = solve_discrete_lyapunov(sq_g * A_cl, Q_lyap)

    # ------------------------------------------------------------------
    # Step 3: FEV
    #   E[Σ_t γ^t s_i^2] = M[i,i]
    #   E[Σ_t γ^t b_j^2] = (Ks M Ks')[j,j] + (1/(1-γ)) Cov_b[j,j]
    # ------------------------------------------------------------------
    mu_s = np.diag(M)
    Mb   = Ks @ M @ Ks.T + (1.0 / (1.0 - gamma_F)) * Cov_b
    mu_b = np.diag(Mb)

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
    """Feature Expectation Matching IRL for conf-MDP (LQR setting).

    conf-MDP版の正しい IRL実装 (割引 Lyapunov 方程式による完全解析的計算):
      - φ(s, b) = [s², b²]   (リーダー行動 a は存在せず，不要)
      - 各イテレーション: DARE (↔Soft VI) → Lyapunov (↔SF) → 割引FEV  (サンプリング不要)
      - OLS / 逆LQR / Nelder-Mead は不要

    2PMG版 (fem_irl_loop) との対比:
      2PMG: a が存在 → K_eff = Ks + Ka K_θ, sigma_leader が必要
      conf: a 概念なし → A_cl = A_theta - B Ks のみ, エントロピーノイズのみ
    """
    n_s = A_theta_ref.shape[0]
    m   = env.B.shape[1]
    log_w = np.zeros(n_s + m, dtype=np.float64)  # 初期値: log_w=0 → Q=R=I

    for i in range(n_irl_iters):
        # 報酬重みパラメトリゼーション: コスト行列 = exp(-log_w)
        #   log_w が増える → コスト減 → フォロワーがその特徴量をより使う
        #   これにより報酬ベース勾配 (μ_expert - μ_policy) が正しく作用する
        Q = np.diag(np.maximum(np.exp(-log_w[:n_s]), 1e-8))
        R = np.diag(np.maximum(np.exp(-log_w[n_s:]), 1e-8))

        mu_policy = compute_policy_fev_conf(
            env, A_theta_ref, Q, R, gamma_F, beta,
        )

        # 報酬ベース IRL 勾配 (MaxEnt IRL 標準形式):
        #   r_F = -(s'Qs + b'Rb) = θ'φ で θ = -[Q_diag; R_diag] を報酬重みと見なす
        #   ∂log P / ∂θ = μ_expert - μ_policy
        #   θ = -exp(-log_w) = -1/exp(log_w) なので chain rule:
        #   ∂L/∂log_wᵢ = (μ_expert - μ_policy)ᵢ * exp(-log_wᵢ)
        grad_reward = expert_mu - mu_policy                     # 報酬勾配
        grad_logw   = grad_reward * np.exp(-log_w)             # chain rule
        # gradient clipping でオーバーフロー防止
        grad_logw   = np.clip(grad_logw, -50.0, 50.0)
        log_w      += lr_irl * grad_logw
        # log_w を [-20, 20] にクランプ (exp overflow 防止)
        log_w       = np.clip(log_w, -20.0, 20.0)

        if verbose and (i % 20 == 0 or i == n_irl_iters - 1):
            diff_norm = float(np.linalg.norm(expert_mu - mu_policy))
            q_diag = np.maximum(np.exp(-log_w[:n_s]), 1e-8)
            r_diag = np.maximum(np.exp(-log_w[n_s:]), 1e-8)
            print(
                f"    [FEM-IRL conf] iter {i:4d}: ||μ_e - μ_π|| = {diff_norm:.6f} | "
                f"Q_diag={q_diag.round(3)}, "
                f"R_diag={r_diag.round(3)}"
            )

    Q_hat = np.diag(np.maximum(np.exp(-log_w[:n_s]), 1e-8))
    R_hat = np.diag(np.maximum(np.exp(-log_w[n_s:]), 1e-8))
    return Q_hat, R_hat


def _omega_from_diag_cost(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Map diagonal costs to ω where diag = exp(-ω), with robust bounds.

    Keep local IRL outputs in the same dynamic range as _diag_from_reward_param
    so a single saturated IRL step does not destabilize recursive aggregation.
    """
    q = np.clip(np.diag(Q), 1e-3, 1e2)
    r = np.clip(np.diag(R), 1e-3, 1e2)
    return np.concatenate([-np.log(q), -np.log(r)]).astype(np.float64)


def _diag_from_reward_param(omega: np.ndarray, n_s: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack unconstrained ω into positive diagonal cost matrices.

    We use the same positive parameterization as FEM-IRL conf mode:
        Q_diag = exp(-ω_q), R_diag = exp(-ω_r)
    """
    omega = np.asarray(omega, dtype=np.float64)
    omega_q = np.clip(omega[:n_s], -8.0, 8.0)
    omega_r = np.clip(omega[n_s:n_s + m], -8.0, 8.0)
    q = np.clip(np.exp(-omega_q), 1e-3, 1e2)
    r = np.clip(np.exp(-omega_r), 1e-3, 1e2)
    return np.diag(q), np.diag(r)


def _conf_info_matrix_from_trajs(
    trajs: list[dict],
    A_theta: np.ndarray,
    B: np.ndarray,
    omega: np.ndarray,
    gamma_F: float,
    n_s: int,
    m: int,
    beta: float,
    reg: float,
) -> np.ndarray:
    """Estimate Sigma_k via analytic gradients of soft-Q w.r.t. omega.

    We compute psi_i(s,b) = dQ_soft(s,b) / domega_i analytically.
    - dQ/domega_i and dR/domega_i are closed-form from
        Q_diag = exp(-omega_q), R_diag = exp(-omega_r), with zero derivative on
        clipped coordinates.
    - dP/domega_i is obtained by solving the discounted discrete Lyapunov eq:
            dP_i = M_i + gamma * A_cl.T * dP_i * A_cl,
        where M_i = dQ_i + K.T * dR_i * K and A_cl = A - B K.

    Then we aggregate
        Sigma_k ~= E[(psi - E_b[psi]) (psi - E_b[psi]).T].
    """
    dim = n_s + m
    A_theta = np.asarray(A_theta, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    if len(trajs) == 0:
        return reg * np.eye(dim, dtype=np.float64)

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
    dq = np.where((q_raw > 1e-3) & (q_raw < 1e2), -q_raw, 0.0)
    dr = np.where((r_raw > 1e-3) & (r_raw < 1e2), -r_raw, 0.0)

    dS = np.zeros((dim, m, m), dtype=np.float64)
    dKs = np.zeros((dim, m, n_s), dtype=np.float64)
    sq_g = np.sqrt(gamma_F)
    for i in range(dim):
        dQ_i = np.zeros((n_s, n_s), dtype=np.float64)
        dR_i = np.zeros((m, m), dtype=np.float64)
        if i < n_s:
            dQ_i[i, i] = dq[i]
        else:
            j = i - n_s
            dR_i[j, j] = dr[j]

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

    if w_sum <= 0.0:
        return reg * np.eye(dim, dtype=np.float64)

    Sigma_k = Sigma_acc / w_sum
    Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)
    Sigma_k += reg * np.eye(dim, dtype=np.float64)

    return Sigma_k

# ---------------------------------------------------------------------------
# Theorem 1 gradient update on μ
# ---------------------------------------------------------------------------
def theorem1_update(
    leader_policy: ConstantEpisodeLeaderPolicy,
    trajs: list[dict],
    learning_rate: float,
    grad_clip_max_norm: float,
) -> float:

    grad_mu = np.zeros_like(leader_policy.mu)
    for traj in trajs:
        grad_mu += np.asarray(traj["grad_theta"], dtype=np.float64)
    grad_mu /= len(trajs)

    # 論文設定: gradient clipping (max norm 1.0)
    grad_norm = float(np.linalg.norm(grad_mu))
    if grad_norm > grad_clip_max_norm:
        grad_mu = grad_mu * (grad_clip_max_norm / grad_norm)

    leader_policy.update(grad_mu, learning_rate)
    returns = [t["total_leader_return"] for t in trajs]
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

PARAMS = {
    "n_leader_iterations":      200,
    "n_episodes_per_iteration":  100,   # 論文設定: 100 trajectories / iteration
    "n_eval_episodes":           50,    # 論文設定: 50 rollouts for evaluation per iteration
    "learning_rate_leader":      1e-1,
    "discount_leader":           0.9,
    "discount_follower":         0.9,
    "beta":                      1e-1,
    "max_workers":               32,
    "grad_clip_max_norm":        1.0,   # 論文設定: estimated gradients clipped to max norm 1.0
    # iterative IRL + RLS mode
    "rls_reg":                   1e-6,
    "rls_prior_precision":       1e-1,
    "rls_eig_threshold":         1e-4,    # 固有値しきい値 ε
    "rls_local_irl_iters":       200,
    "rls_local_irl_lr":          0.01,
    "rls_max_update_iters":      10,     # 何イテレーションまでパラメータ推定を更新するか
    "verbose":                   True,
}

# リーダー報酬の重み係数 (環境デフォルトと合わせる)
_LEADER_REWARD_WEIGHTS = {
    "stability_weight":        1.0,
    "energy_weight":           0.5,
    "insulation_cost_weight":  0.1,
    "airflow_cost_weight":     0.1,
    "beta":                    0.1,
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
    """Evaluate leader with deterministic theta=mu (no leader sampling noise).

    - mu_theta を決定論的リーダー行動として固定 (sigma=0 相当)
    - フォロワーは真のコスト (env.Q, env.R) で DARE → 真の最適応答
    - n_eval_eps エピソードの平均・標準偏差を返す (初期状態・遷移ノイズは確率的)
    """
    n_zones = env.num_zones
    n_hvac  = env.num_hvac_units
    il = mu_theta[:n_zones]
    aa = mu_theta[n_zones:]
    A_theta = env.compute_parameterized_A(il, aa)
    sol     = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
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
    beta    = PARAMS["beta"]
    verbose = PARAMS["verbose"]
    max_w   = PARAMS["max_workers"]

    stats = {
        # training trajectories (gradient用)
        "leader_return":          [], "leader_return_std":          [],
        "stability":              [], "stability_std":              [],
        "energy_cost":            [], "energy_cost_std":            [],
        "insulation_cost":        [], "insulation_cost_std":        [],
        "airflow_cost":           [], "airflow_cost_std":           [],
        # evaluation rollouts (論文の50 rollouts: mu を固定して評価)
        "eval_leader_return":     [], "eval_leader_return_std":     [],
        "eval_stability":         [], "eval_stability_std":         [],
        "eval_energy_cost":       [], "eval_energy_cost_std":       [],
        "eval_insulation_cost":   [], "eval_insulation_cost_std":   [],
        "eval_airflow_cost":      [], "eval_airflow_cost_std":      [],
        "mode": mode
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
        dim = n_zones + n_hvac
        prior_prec = float(PARAMS["rls_prior_precision"])
        rls_sigma_bar = prior_prec * np.eye(dim, dtype=np.float64)
        rls_v_acc = np.zeros(dim, dtype=np.float64)
        omega_bar = np.zeros(dim, dtype=np.float64)

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
            sol     = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
            Ks_f_i, bias_f_i, Sigma_f_i = _sol_to_follower_params_conf(sol, n_hvac, beta)
            for _ in thetas:
                per_ep.append((theta, Ks_f_i, bias_f_i, Sigma_f_i))
            sol_follower  = sol
            sol_grad      = sol
            include_term3 = True

        elif mode == "irl_known_dynamics":
            if not surrogate_fitted:
                # === 正しい FEM-IRL による推定 (conf-MDP 版) ===
                theta_c    = np.full(n_theta, 0.5)
                il_c, aa_c = theta_c[:n_zones], theta_c[n_zones:]
                A_c        = env.compute_parameterized_A(il_c, aa_c)
                sol_c      = solve_confmdp_lqr(A_c, env.B, env.Q, env.R, gamma_F)
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
                    beta        = beta,
                    n_irl_iters = 10000,
                    lr_irl      = 0.01,
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
            Ks_hat, bias_hat, Sigma_hat = _sol_to_follower_params_conf(sol_hat, n_hvac, beta)
            for _ in thetas:
                per_ep.append((theta, Ks_hat, bias_hat, Sigma_hat))
            sol_follower  = sol_hat
            sol_grad      = sol_hat
            include_term3 = True

        elif mode == "irl_recursive_rls":
            # 毎 iteration で IRL を解き、その推定を RLS で累積統合するモード
            il       = theta[:n_zones]
            aa       = theta[n_zones:]
            A_theta  = env.compute_parameterized_A(il, aa)

            if verbose:
                print(f"  [IRL-RLS conf] Updating reward parameters (iter {iteration})...")

            # Expert trajectories in the current environment (true follower reward)
            sol_true = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
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
                beta=beta,
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
                beta=beta,
                reg=PARAMS["rls_reg"],
            )

            # Reward identification accumulation (thresholded covariance)
            assert rls_sigma_bar is not None and rls_v_acc is not None
            assert omega_bar is not None

            eigvals_k, eigvecs_k = np.linalg.eigh(Sigma_k)
            eig_threshold = float(PARAMS.get("rls_eig_threshold", 1e-4))
            mask_k = (eigvals_k >= eig_threshold).astype(np.float64)
            U_k = eigvecs_k @ np.diag(mask_k) @ eigvecs_k.T
            U_k = 0.5 * (U_k + U_k.T)

            rls_update_count += 1
            rls_sigma_bar = rls_sigma_bar + U_k
            rls_v_acc = rls_v_acc + U_k @ omega_k

            # 3. 解析解で reward parameter を更新
            omega_prev = omega_bar.copy()

            ridge = float(PARAMS.get("rls_reg", 1e-6))
            rls_sigma_bar_arr = np.asarray(rls_sigma_bar, dtype=np.float64)
            bar_u = rls_sigma_bar_arr + ridge * np.eye(rls_sigma_bar_arr.shape[0], dtype=np.float64)
            # try:
            omega_bar = np.linalg.solve(bar_u, rls_v_acc)
            # except np.linalg.LinAlgError:
                # omega_bar = np.linalg.pinv(bar_u) @ rls_v_acc

            Q_hat, R_hat = _diag_from_reward_param(omega_bar, n_zones, n_hvac)

            # Use aggregated reward estimate to compute follower response
            sol_hat  = solve_confmdp_lqr(A_theta, env.B, Q_hat, R_hat, gamma_F)
            Ks_hat, bias_hat, Sigma_hat = _sol_to_follower_params_conf(sol_hat, n_hvac, beta)
            for _ in thetas:
                per_ep.append((theta, Ks_hat, bias_hat, Sigma_hat))
            sol_follower  = sol_hat
            sol_grad      = sol_hat
            include_term3 = True

            if verbose:
                diff_omega = np.linalg.norm(omega_k - omega_prev)
                eig_u = np.linalg.eigvalsh(U_k)
                print(f"  [IRL-RLS Diagnostics] iter {iteration}")
                print(f"    Sigma_k eigmin: {eigvals_k.min():.2e}, eigmax: {eigvals_k.max():.2e}, cond: {eigvals_k.max()/(eigvals_k.min()+1e-12):.2e}")
                print(f"    U_k     eigmin: {eig_u.min():.2e}, eigmax: {eig_u.max():.2e}, rank: {np.linalg.matrix_rank(U_k)}")
                print(f"    ||omega_k||: {np.linalg.norm(omega_k):.4f}, ||omega_bar||: {np.linalg.norm(omega_bar):.4f}")
                print(f"    ||omega_k - omega_prev||: {diff_omega:.4f}")
                print(f"    bar_U condition number: {np.linalg.cond(bar_u):.4e}")

        elif mode == "non_irl":
            il      = theta[:n_zones]
            aa      = theta[n_zones:]
            A_theta = env.compute_parameterized_A(il, aa)
            sol     = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
            Ks_f_i, bias_f_i, Sigma_f_i = _sol_to_follower_params_conf(sol, n_hvac, beta)
            for _ in thetas:
                per_ep.append((theta, Ks_f_i, bias_f_i, Sigma_f_i))
            sol_follower  = sol
            sol_grad      = None
            include_term3 = False

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # ---- Parallel episode collection (軌跡収集のみ) ----
        pool, trajs = _collect_episodes_parallel(
            per_ep, gamma_L, base_seed=iteration, max_workers=max_w,
        )

        # ---- 勾配計算: メインプロセスで theorem1_episode_gradient を呼ぶ ----
        # softvi/irl_known_dynamics: term1+2+3 全部計算
        # non_irl:                   term1+2 のみ (include_term3=False)
        il = theta[:n_zones]
        aa = theta[n_zones:]
        for traj in trajs:
            grad = theorem1_episode_gradient(
                traj, il, aa, env,
                gamma_L, gamma_F, beta,
                sol_follower, sol_grad, include_term3,
            )
            traj["grad_theta"] = grad.tolist()

        # ---- Leader update (Theorem 1) ----
        mean_ret = theorem1_update(leader_policy, trajs, lr, PARAMS["grad_clip_max_norm"])
        std_ret  = float(np.std([t["total_leader_return"] for t in trajs]))

        stats["leader_return"].append(mean_ret)
        stats["leader_return_std"].append(std_ret)

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

        # ---- Evaluation rollouts (論文: 50 rollouts でリーダー return を評価) ----
        # mu を決定論的リーダー行動として固定、env.Q/R で真のフォロワー最適応答を使用。
        # seed: iter × 1_000_000 + seed でtraining rolloutと分離
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
    # seed ごとに別ファイル保存 (後で aggregate_seeds で集計)
    path = out / f"stats_{mode}_seed{seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"=== Saved to {path} ===")
    return stats


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    "eval_leader_return",
    "eval_stability", "eval_energy_cost", "eval_insulation_cost", "eval_airflow_cost",
    "leader_return",
    "stability", "energy_cost", "insulation_cost", "airflow_cost",
]


def aggregate_seeds(
    output_dir: str,
    mode: str,
    n_seeds: int = 10,
) -> dict:
    """n_seeds 分の per-seed pkl を読み込み、iteration ごとに mean/std を計算して返す.

    論文設定: 10 seeds, max/min 除外後 8 seeds で mean/std
    → max/min 除外ロジックは実装済みだがコメントアウト中。
    """
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

        # ----- 通常の mean / std (全seed使用) -----
        result[f"{k}_mean"] = arr.mean(axis=0).tolist()
        result[f"{k}_std"]  = arr.std(axis=0).tolist()
    # pkl に保存
    agg_path = out / f"aggregated_{mode}.pkl"
    with open(agg_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[aggregate_seeds] Saved aggregated result -> {agg_path}")
    return result


def merge_and_plot(output_dir: str, n_seeds: int = 10) -> None:
    from scripts.plot_learning_curves import plot_learning_curves, plot_components, plot_leader_performance_only
    out   = Path(output_dir)
    modes = {
        "softvi": "Oracle (SoftVI)",
        "irl_known_dynamics": "Proposed IRL",
        "irl_recursive_rls": "Proposed Aggregate IRL",
        "non_irl": "Baseline No-Influence",
    }

    # まず集計
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
        """aggregated dict → plot_learning_curves が期待する形式に変換."""
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

    main_name  = "Oracle (SoftVI)"
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
    """Fallback: single-seed pkl から直接プロット (旧挙動との互換)."""
    from scripts.plot_learning_curves import plot_learning_curves, plot_components, plot_leader_performance_only
    out   = Path(output_dir)
    modes = {
        "softvi": "Oracle (SoftVI)",
        "irl_known_dynamics": "Proposed IRL",
        "irl_recursive_rls": "Proposed Aggregate IRL",
        "non_irl": "Baseline No-Influence",
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
                        choices=["softvi", "irl_known_dynamics", "irl_recursive_rls", "non_irl", "plot", "aggregate"])
    parser.add_argument("--output-dir", default="data/internal/conf_continuous")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds to aggregate (for plot/aggregate mode)")
    args = parser.parse_args()
    if args.mode == "plot":
        merge_and_plot(args.output_dir, n_seeds=args.n_seeds)
    elif args.mode == "aggregate":
        for m in ["softvi", "irl_known_dynamics", "irl_recursive_rls", "non_irl"]:
            aggregate_seeds(args.output_dir, m, n_seeds=args.n_seeds)
    else:
        run_experiment(args.mode, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
