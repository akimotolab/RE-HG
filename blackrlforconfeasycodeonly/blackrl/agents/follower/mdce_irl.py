from collections.abc import Callable

import numpy as np
import torch


class MDCEIRL:
    def __init__(
        self,
        feature_fn: Callable,
        discount: float = 0.99,
        max_iterations: int = 1000,
        tolerance: float = 0.025,
        n_soft_q_iterations: int = 100,
        learning_rate: float = 1e-3,
        lr_decay: float = 1e-2,
        irl_rollout_episodes: int = 200,
    ):
        self.feature_fn = feature_fn
        self.discount = discount
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_soft_q_iterations = n_soft_q_iterations
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.irl_rollout_episodes = irl_rollout_episodes
        self.w: torch.Tensor | None = None  # requires_grad は使わない（手動更新）

    def compute_expert_fev(
        self,
        trajectories: list[dict],
    ) -> torch.Tensor:
        all_traj_fev = []

        for traj in trajectories:
            obs = traj.get("observations", [])
            leader_acts = traj.get("leader_actions", [])
            follower_acts = traj.get("follower_actions", [])

            if len(obs) == 0:
                continue

            traj_fev = None

            for t in range(len(obs)):
                s = obs[t]
                a = leader_acts[t] if t < len(leader_acts) else None
                b = follower_acts[t] if t < len(follower_acts) else None

                if a is None or b is None:
                    continue

                # Compute feature vector
                phi_t = self.feature_fn(s, a, b)
                if isinstance(phi_t, np.ndarray):
                    phi_t = torch.from_numpy(phi_t).float()
                elif not isinstance(phi_t, torch.Tensor):
                    phi_t = torch.tensor(phi_t, dtype=torch.float32)

                # Discounted feature
                discounted_phi = (self.discount**t) * phi_t

                if traj_fev is None:
                    traj_fev = discounted_phi
                else:
                    traj_fev = traj_fev + discounted_phi

            if traj_fev is not None:
                all_traj_fev.append(traj_fev)

        # (平均をとる処理はそのまま)
        if len(all_traj_fev) > 0:
            fev = torch.mean(torch.stack(all_traj_fev), dim=0)
        else:
            dim = getattr(self.feature_fn, "feature_dim", None)
            if dim is None:
                try:
                    phi0 = self.feature_fn(0, 0, 0)
                    dim = phi0.shape[0] if hasattr(phi0, "shape") else len(phi0)
                except Exception:
                    dim = 18
            fev = torch.zeros(dim)

        return fev

    def _sample_trajectory_fev(
        self,
        policy: Callable,
        leader_policy: Callable,
        env,
    ) -> tuple[torch.Tensor | None, float]:
        """Sample a single trajectory and compute its FEV.

        Args:
            policy: Follower policy g(b|s, a)
            leader_policy: Leader policy f_θ_L(a|s)
            env: Environment instance

        Returns:
            Tuple of (trajectory_fev, weight)

        """
        obs, _ = env.reset()
        traj_fev = None
        weight = 0.0
        t = 0

        while True:
            # リーダーの方策（確率分布 [p(0), p(1)]）を取得
            leader_probs = leader_policy(obs)
            if not isinstance(leader_probs, (list, np.ndarray)):
                # (フォールバック) もし leader_policy が int を返した場合
                leader_act = leader_probs
            else:
                # 確率分布からサンプリングして int (0 or 1) に変換
                n_leader_actions = len(leader_probs)
                leader_act = np.random.choice(n_leader_actions, p=leader_probs)

            # Sample follower action from policy
            # これで follower_act = policy(obs, (int)leader_act) となり安全
            follower_act = policy(obs, leader_act)

            # Compute feature vector
            # これで phi_t = self.feature_fn(obs, (int)leader_act, ...) となり安全
            phi_t = self.feature_fn(obs, leader_act, follower_act)
            if isinstance(phi_t, np.ndarray):
                phi_t = torch.from_numpy(phi_t).float()
            elif not isinstance(phi_t, torch.Tensor):
                phi_t = torch.tensor(phi_t, dtype=torch.float32)

            # Discounted feature
            discounted_phi = (self.discount**t) * phi_t

            if traj_fev is None:
                traj_fev = discounted_phi
            else:
                traj_fev = traj_fev + discounted_phi

            weight += self.discount**t

            # Step environment
            env_step = env.step(leader_act, follower_act)
            obs = env_step.observation

            t += 1

            if env_step.last or t >= env.spec.max_episode_length:
                break

        return traj_fev, weight

    def compute_likelihood(
        self,
        trajectories: list[dict],
        policy_fn: Callable,
    ) -> float:
        total_likelihood = 0.0
        total_weight = 0.0

        for traj in trajectories:
            obs = traj.get("observations", [])
            leader_acts = traj.get("leader_actions", [])
            follower_acts = traj.get("follower_actions", [])

            traj_likelihood = 0.0
            weight = 0.0

            for t in range(len(obs)):
                if t >= len(leader_acts) or t >= len(follower_acts):
                    break

                s = obs[t]
                a = leader_acts[t]
                b = follower_acts[t]

                # Get log probability
                log_prob = policy_fn(s, a, b)

                traj_likelihood += (self.discount**t) * log_prob
                weight += self.discount**t

            total_likelihood += traj_likelihood
            total_weight += weight

        return total_likelihood / total_weight if total_weight > 0 else 0.0

    def get_reward_params(self) -> torch.Tensor:
        return self.w

    def get_reward_fn(self):
        def reward_fn(state, leader_action, follower_action):
            phi = self.feature_fn(state, leader_action, follower_action)
            if isinstance(phi, np.ndarray):
                phi = torch.from_numpy(phi).float()
            elif not isinstance(phi, torch.Tensor):
                phi = torch.tensor(phi, dtype=torch.float32)
            return torch.dot(self.w, phi).item()

        return reward_fn
