"""Follower reward estimation and policy derivation for bilevel RL."""

from collections import defaultdict

import numpy as np
import torch
import cvxpy as cp

from blackrl.agents.follower.follower_policy_model import FollowerPolicyModel
from blackrl.agents.follower.sf_learning import SuccessorFeatureLearning
from blackrl.agents.follower.soft_q_learning import SoftQLearning


def get_follower_action_probs(soft_q_learning, env_spec, state, leader_action):
    """Get follower action probabilities for a given state and leader action."""
    if soft_q_learning is None:
        return np.ones(env_spec.action_space.n) / env_spec.action_space.n
    policy_dict = soft_q_learning.get_policy(state, leader_action)
    probs = np.array(
        [policy_dict.get(b, 0.0) for b in range(env_spec.action_space.n)],
        dtype=np.float32,
    )
    return probs


class FollowerEstimator:
    """Handles follower reward estimation via IRL and follower policy derivation."""

    def __init__(
        self,
        env_spec,
        mdce_irl,
        leader_policy_obj,
        temperature,
        discount_follower,
        learning_rate_follower,
        soft_q_config,
        collector,
        transition_fn=None,
    ):
        self.env_spec = env_spec
        self.mdce_irl = mdce_irl
        self.leader_policy_obj = leader_policy_obj  # shared reference
        self.temperature = temperature
        self.discount_follower = discount_follower
        self.learning_rate_follower = learning_rate_follower
        self.soft_q_config = soft_q_config
        self.collector = collector
        # Optional: callable (s, a, b) -> s'  or  (s, a, b) -> {s': prob}
        # If provided, analytic SF computation is used in the IRL loop.
        self.transition_fn = transition_fn

        self.soft_q_learning = None
        self.sf_learning_model = None
        self.stats = defaultdict(list)

    @property
    def follower_model(self):
        """Model used to produce follower policy/probabilities.

        The concrete type may be SoftQLearning, FollowerPolicyModel, or
        EmpiricalFollowerModel depending on training mode.
        """
        return self.soft_q_learning

    @follower_model.setter
    def follower_model(self, model):
        self.soft_q_learning = model

    def get_action_probs(self, state, leader_action, model=None):
        """Return follower action probabilities for a given (state, leader action).

        Args:
            state: Discrete state.
            leader_action: Discrete leader action.
            model: Optional follower model override. If None, uses current
                ``self.follower_model``.
        """
        target_model = self.follower_model if model is None else model
        return get_follower_action_probs(target_model, self.env_spec, state, leader_action)

    def sample_action(self, state, leader_action, model=None):
        """Sample follower action for a given (state, leader action).

        Args:
            state: Discrete state.
            leader_action: Discrete leader action.
            model: Optional follower model override. If None, uses current
                ``self.follower_model``.
        """
        target_model = self.follower_model if model is None else model

        if target_model is not None and hasattr(target_model, "sample_action"):
            action = target_model.sample_action(state, leader_action)
            return np.asarray(action, dtype=np.int32)

        probs = np.asarray(
            self.get_action_probs(state, leader_action, model=target_model),
            dtype=np.float64,
        )
        probs = probs / max(probs.sum(), 1e-12)
        action = int(np.random.choice(self.env_spec.action_space.n, p=probs))
        return np.array(action, dtype=np.int32)

    def display_learned_rewards(self, reward_fn):
        """Display learned reward function."""
        print("\nLearned Reward Function: r_F(s, a, b)")
        print("-" * 80)

        for s in range(self.env_spec.observation_space.n):
            print(f"\nState {s}:")
            for a in range(self.env_spec.leader_action_space.n):
                rewards = []
                for b in range(self.env_spec.action_space.n):
                    r_val = reward_fn(s, a, b)
                    rewards.append(f"b={b}: {r_val:6.3f}")
                print(f"  Leader action {a}: " + "  ".join(rewards))

    def compute_softvi_q_values(self, env, max_iterations=2000, tolerance=1e-5, verbose=True):
        """Compute Soft Value Iteration Q-values."""
        Q = {}
        for s in range(self.env_spec.observation_space.n):
            for a in range(self.env_spec.leader_action_space.n):
                for b in range(self.env_spec.action_space.n):
                    Q[(s, a, b)] = 0.0

        for iteration in range(max_iterations):
            Q_old = Q.copy()
            max_delta = 0.0

            for s in range(self.env_spec.observation_space.n):
                for a in range(self.env_spec.leader_action_space.n):
                    for b in range(self.env_spec.action_space.n):
                        env.reset(init_state=s)
                        env_step = env.step(a, b)

                        reward = env_step.reward
                        next_state = env_step.observation

                        if env_step.terminal:
                            Q[(s, a, b)] = reward
                        else:
                            next_state_int = int(
                                next_state.item() if isinstance(next_state, np.ndarray) else next_state
                            )

                            v_next = 0.0
                            for a_next in range(self.env_spec.leader_action_space.n):
                                leader_prob = self.leader_policy_obj.policy_table[next_state_int, a_next]
                                q_values = [
                                    Q_old.get((next_state_int, a_next, b_next), 0.0)
                                    for b_next in range(self.env_spec.action_space.n)
                                ]
                                q_tensor = torch.tensor(q_values)
                                soft_v = self.temperature * torch.logsumexp(
                                    q_tensor / self.temperature, dim=0
                                ).item()
                                v_next += leader_prob * soft_v

                            Q[(s, a, b)] = reward + self.discount_follower * v_next

                        delta = abs(Q[(s, a, b)] - Q_old.get((s, a, b), 0.0))
                        max_delta = max(max_delta, delta)

            if verbose:
                print(f"  SoftVI iteration {iteration + 1}: max_delta = {max_delta:.6f}")

            if max_delta < tolerance:
                if verbose:
                    print(f"\n✓ SoftVI converged in {iteration + 1} iterations (max_delta={max_delta:.6f})")
                break

        if iteration == max_iterations - 1:
            if verbose:
                if max_delta < 1e-4:
                    print(
                        f"\n✓ SoftVI nearly converged after {max_iterations} iterations "
                        f"(max_delta={max_delta:.6f} < 1e-4)",
                    )
                else:
                    print(
                        f"\n⚠ WARNING: SoftVI did NOT converge after {max_iterations} iterations "
                        f"(max_delta={max_delta:.6f})",
                    )

        return Q

    def compute_softvi_q_values_known_dynamics(
        self,
        reward_fn,
        max_iterations: int = 2000,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> dict:
        """Compute Soft VI Q-values using known transition (no env sampling).

        Uses transition_fn and reward_fn; no episode collection. For use in
        IRL with known dynamics so the follower policy is derived by value
        iteration instead of Soft Q-learning.
        """
        if self.transition_fn is None:
            raise RuntimeError(
                "compute_softvi_q_values_known_dynamics requires transition_fn."
            )

        n_s = self.env_spec.observation_space.n
        n_a = self.env_spec.leader_action_space.n
        n_b = self.env_spec.action_space.n

        Q = {}
        for s in range(n_s):
            for a in range(n_a):
                for b in range(n_b):
                    Q[(s, a, b)] = 0.0

        for iteration in range(max_iterations):
            Q_old = Q.copy()
            max_delta = 0.0

            for s in range(n_s):
                for a in range(n_a):
                    for b in range(n_b):
                        reward = reward_fn(s, a, b)

                        result = self.transition_fn(s, a, b)
                        if isinstance(result, dict):
                            next_dist = result
                        else:
                            next_dist = {int(result): 1.0}

                        v_next = 0.0
                        for s_next, p_trans in next_dist.items():
                            for a_next in range(n_a):
                                leader_prob = self.leader_policy_obj.policy_table[
                                    s_next, a_next
                                ]
                                q_vals = [
                                    Q_old.get((s_next, a_next, b_next), 0.0)
                                    for b_next in range(n_b)
                                ]
                                q_tensor = torch.tensor(q_vals)
                                soft_v = self.temperature * torch.logsumexp(
                                    q_tensor / self.temperature, dim=0
                                ).item()
                                v_next += p_trans * leader_prob * soft_v

                        Q[(s, a, b)] = reward + self.discount_follower * v_next
                        delta = abs(Q[(s, a, b)] - Q_old.get((s, a, b), 0.0))
                        max_delta = max(max_delta, delta)

            if verbose:
                print(f"  SoftVI (known-dyn) iter {iteration + 1}: max_delta={max_delta:.6f}")
            if max_delta < tolerance:
                break

        return Q

    def estimate_follower_reward(self, trajectories, env, verbose=True):
        """Estimate follower reward using MDCE IRL with hybrid SoftQ+SF approach."""
        # 1. Expert FEV
        expert_fev = self.mdce_irl.compute_expert_fev(trajectories)
        feature_dim = expert_fev.shape[0]

        if self.mdce_irl.w is None:
            self.mdce_irl.w = torch.randn(feature_dim) * 0.01

        sf_lr = 0.05
        q_lr = self.learning_rate_follower
        epsilon = self.soft_q_config.get("epsilon", 0.0)

        # === Learner 1: Soft Q-Learning (Warm Start) ===
        if self.soft_q_learning is None or not isinstance(self.soft_q_learning, SoftQLearning):
            self.soft_q_learning = SoftQLearning(
                env_spec=self.env_spec,
                reward_fn=self.mdce_irl.get_reward_fn(),
                leader_policy=lambda state: self.leader_policy_obj.get_probabilities(state),
                discount=self.discount_follower,
                learning_rate=q_lr,
                temperature=self.temperature,
                epsilon=epsilon,
            )
        else:
            self.soft_q_learning.reward_fn = self.mdce_irl.get_reward_fn()
            self.soft_q_learning.leader_policy = lambda state: self.leader_policy_obj.get_probabilities(state)
            self.soft_q_learning.learning_rate = q_lr
            self.soft_q_learning.epsilon = epsilon

        soft_q_learner = self.soft_q_learning

        # === Learner 2: Successor Features (Warm Start) ===
        if self.sf_learning_model is None:
            self.sf_learning_model = SuccessorFeatureLearning(
                env_spec=self.env_spec,
                feature_dim=feature_dim,
                feature_fn=self.mdce_irl.feature_fn,
                discount=self.discount_follower,
                learning_rate=sf_lr,
                temperature=self.temperature,
            )

        sf_learner = self.sf_learning_model

        # Parallel Setup
        leader_policy_table = self.leader_policy_obj.policy_table.copy()
        n_fa = self.env_spec.action_space.n

        # === IRL Loop ===
        for irl_iteration in range(self.mdce_irl.max_iterations):
            # 1. Parallel Data Collection (On-Policy)
            irl_eps = int(getattr(self.mdce_irl, "irl_rollout_episodes", 200))
            data = self.collector.collect_episodes(
                env=env,
                leader_policy_table=leader_policy_table,
                follower_model=soft_q_learner,
                n_episodes=irl_eps,
            )

            # 2. Update Learners
            n_transitions = len(data["observation"])
            for i in range(n_transitions):
                obs = data["observation"][i]
                la = data["leader_action"][i]
                fa = data["action"][i]
                next_obs = data["next_observation"][i]
                done = data["last"][i]

                phi = self.mdce_irl.feature_fn(obs, la, fa)
                phi_tensor = torch.from_numpy(phi).float()
                current_reward = torch.dot(self.mdce_irl.w, phi_tensor).item()

                # A. Soft Q Update
                soft_q_learner.update(obs, la, fa, current_reward, next_obs, done)

                # B. SF Update (Target: Discounted Sum)
                if not done:
                    next_obs_int = int(next_obs.item() if hasattr(next_obs, "item") else next_obs)
                    leader_probs_next = self.leader_policy_obj.get_probabilities(next_obs_int)
                    expected_next_sf = np.zeros(feature_dim, dtype=np.float32)
                    for la_next, la_prob in enumerate(leader_probs_next):
                        q_vals = [soft_q_learner.get_q_value(next_obs, la_next, b) for b in range(n_fa)]
                        q_vals_array = np.array(q_vals) / self.temperature
                        fa_probs = np.exp(q_vals_array - np.max(q_vals_array))
                        fa_probs = fa_probs / fa_probs.sum()
                        for fa_next, fa_prob in enumerate(fa_probs):
                            psi_next = sf_learner.sf_table[int(next_obs), la_next, fa_next]
                            expected_next_sf += la_prob * fa_prob * psi_next
                    target_sf = phi + self.discount_follower * expected_next_sf
                else:
                    target_sf = phi

                current_sf = sf_learner.sf_table[int(obs), int(la), int(fa)]
                sf_learner.sf_table[int(obs), int(la), int(fa)] += sf_lr * (target_sf - current_sf)

            # 3. Update Reward Parameters w
            policy_fev = torch.zeros(feature_dim)
            p0 = np.zeros(self.env_spec.observation_space.n, dtype=np.float64)
            for traj in trajectories:
                obs = traj.get("observations", [])
                if len(obs) > 0:
                    s_first = int(obs[0].item() if hasattr(obs[0], "item") else obs[0])
                    p0[s_first] += 1.0
            if p0.sum() <= 0:
                p0[0] = 1.0
            p0 /= p0.sum()

            for s0 in range(self.env_spec.observation_space.n):
                if p0[s0] <= 0:
                    continue
                leader_probs_0 = self.leader_policy_obj.get_probabilities(s0)

                for la_0, la_prob in enumerate(leader_probs_0):
                    q_vals = [soft_q_learner.get_q_value(s0, la_0, b) for b in range(n_fa)]
                    fa_probs = torch.softmax(torch.tensor(q_vals) / self.temperature, dim=0).numpy()
                    for fa_0, fa_prob in enumerate(fa_probs):
                        psi_0 = sf_learner.sf_table[s0, la_0, fa_0]
                        policy_fev += p0[s0] * la_prob * fa_prob * torch.from_numpy(psi_0).float()

            gradient = expert_fev - policy_fev

            lr0 = float(getattr(self.mdce_irl, "learning_rate", 1e-3))
            lr_decay = float(getattr(self.mdce_irl, "lr_decay", 1e-2))
            lr = lr0 / (1.0 + lr_decay * irl_iteration)
            with torch.no_grad():
                self.mdce_irl.w += lr * gradient

            # Logging
            delta_fem = torch.max(torch.abs(gradient)).item()
            gradient_norm = torch.norm(gradient).item()

            all_q_values = []
            for s in range(self.env_spec.observation_space.n):
                for la in range(self.env_spec.leader_action_space.n):
                    for fa in range(self.env_spec.action_space.n):
                        all_q_values.append(soft_q_learner.get_q_value(s, la, fa))
            all_q_values = np.array(all_q_values)

            self.stats["irl_delta_fem"].append(delta_fem)
            self.stats["irl_gradient_norm"].append(gradient_norm)
            self.stats["irl_q_value_mean"].append(float(np.mean(all_q_values)))

            if verbose and (irl_iteration % 10 == 0):
                print(f"\n--- IRL iteration {irl_iteration}: Hybrid (SoftQ+SF) Statistics ---")
                print(f"  Grad Norm: {gradient_norm:.6f}, Delta FEM: {delta_fem:.6f}")
                print(f"  Q Mean: {np.mean(all_q_values):.4f}")
                w_norm = torch.norm(self.mdce_irl.w).item()
                print(f"  w norm: {w_norm:.4f}")

            if delta_fem < self.mdce_irl.tolerance:
                if verbose:
                    print(f"Converged at iter {irl_iteration}")
                    self.display_learned_rewards(self.mdce_irl.get_reward_fn())
                break

        if verbose:
            print("Follower policy updated (Hybrid SoftQ+SF).")
            print("\n" + "=" * 80)
            print("FINAL RECOVERED MODEL DETAILS")
            print("=" * 80)

            print("\n[1] Recovered Reward Function: r(s, a, b)")
            self.display_learned_rewards(self.mdce_irl.get_reward_fn())
            print("\n[2] Recovered Follower Policy Probabilities: π(b | s, a)")
            print("-" * 80)

            for s in range(self.env_spec.observation_space.n):
                print(f"\nState {s}:")
                for la in range(self.env_spec.leader_action_space.n):
                    q_vals = [soft_q_learner.get_q_value(s, la, fa) for fa in range(self.env_spec.action_space.n)]
                    q_tensor = torch.tensor(q_vals)
                    probs = torch.softmax(q_tensor / self.temperature, dim=0).numpy()
                    probs_str = "  ".join([f"P(b={b})={p:.4f}" for b, p in enumerate(probs)])
                    print(f"  Leader Action a={la}:  {probs_str}")

            print("=" * 80)

        soft_q_learner.reward_fn = self.mdce_irl.get_reward_fn()

        return self.mdce_irl.w

    # ------------------------------------------------------------------
    # Known-dynamics IRL  (transition_fn must be provided)
    # ------------------------------------------------------------------

    def compute_analytic_sf(
        self,
        soft_q_learner,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """Compute Successor Features analytically using the known transition model.

        Uses synchronous DP over the full state/action space:

            ψ(s,a,b) = φ(s,a,b) + γ Σ_{s'} P(s'|s,a,b) Σ_{a'} π_L(a'|s')
                                      Σ_{b'} π_F(b'|s',a') ψ(s',a',b')

        where π_F is derived from soft_q_learner and π_L from leader_policy_obj.

        Returns
        -------
        psi : np.ndarray, shape (n_s, n_a, n_b, feature_dim)
        """
        if self.transition_fn is None:
            raise RuntimeError(
                "compute_analytic_sf requires transition_fn to be set in FollowerEstimator."
            )

        n_s = self.env_spec.observation_space.n
        n_a = self.env_spec.leader_action_space.n
        n_b = self.env_spec.action_space.n

        # Infer feature dimension from a sample call
        _phi_sample = self.mdce_irl.feature_fn(0, 0, 0)
        feature_dim = _phi_sample.shape[0]

        psi = np.zeros((n_s, n_a, n_b, feature_dim), dtype=np.float32)

        for _ in range(max_iterations):
            psi_old = psi.copy()
            max_delta = 0.0

            for s in range(n_s):
                for a in range(n_a):
                    for b in range(n_b):
                        phi = self.mdce_irl.feature_fn(s, a, b).astype(np.float32)

                        # Resolve next state distribution
                        result = self.transition_fn(s, a, b)
                        if isinstance(result, dict):
                            # Stochastic: {s': prob}
                            next_dist = result
                        else:
                            # Deterministic: single next state
                            next_dist = {int(result): 1.0}

                        # Expected next ψ under π_L × π_F
                        expected_next_psi = np.zeros(feature_dim, dtype=np.float32)
                        for s_next, p_trans in next_dist.items():
                            leader_probs = self.leader_policy_obj.get_probabilities(s_next)
                            for a_next, la_prob in enumerate(leader_probs):
                                q_vals = np.array(
                                    [soft_q_learner.get_q_value(s_next, a_next, b_next)
                                     for b_next in range(n_b)],
                                    dtype=np.float32,
                                )
                                q_scaled = q_vals / self.temperature
                                q_scaled -= q_scaled.max()
                                fa_probs = np.exp(q_scaled)
                                fa_probs /= fa_probs.sum()

                                for b_next in range(n_b):
                                    expected_next_psi += (
                                        p_trans
                                        * la_prob
                                        * fa_probs[b_next]
                                        * psi_old[s_next, a_next, b_next]
                                    )

                        new_psi = phi + self.discount_follower * expected_next_psi
                        delta = np.max(np.abs(new_psi - psi_old[s, a, b]))
                        max_delta = max(max_delta, delta)
                        psi[s, a, b] = new_psi

            if max_delta < tolerance:
                break

        return psi

    def compute_reward_confidence_matrix_known_dynamics(self, env, model=None) -> np.ndarray:
        """Compute Sigma_k = E[Cov(psi)] in known-dynamics discrete setting.

        This computes
            Sigma_k = sum_s d_gamma(s) sum_a pi_L(a|s) Cov_{b~pi_F(.|s,a)}[psi(s,a,b)],
        where psi is the analytic successor feature tensor under the current
        follower model and known transition function.
        """
        if self.transition_fn is None:
            raise RuntimeError(
                "compute_reward_confidence_matrix_known_dynamics requires transition_fn."
            )

        target_model = self.follower_model if model is None else model
        if target_model is None:
            raise RuntimeError(
                "compute_reward_confidence_matrix_known_dynamics requires a follower model."
            )

        n_s = self.env_spec.observation_space.n
        n_a = self.env_spec.leader_action_space.n
        n_b = self.env_spec.action_space.n
        gamma = float(self.discount_follower)

        psi = self.compute_analytic_sf(target_model)
        feature_dim = int(psi.shape[-1])

        # Build induced state transition matrix under pi_L x pi_F.
        p_state = np.zeros((n_s, n_s), dtype=np.float64)
        for s in range(n_s):
            leader_probs = np.asarray(self.leader_policy_obj.get_probabilities(s), dtype=np.float64)
            for a in range(n_a):
                pa = float(leader_probs[a])
                if pa <= 0.0:
                    continue

                follower_probs = np.asarray(
                    self.get_action_probs(s, a, model=target_model), dtype=np.float64,
                )
                follower_probs = follower_probs / max(float(follower_probs.sum()), 1e-12)

                for b in range(n_b):
                    pb = float(follower_probs[b])
                    if pb <= 0.0:
                        continue
                    result = self.transition_fn(s, a, b)
                    if isinstance(result, dict):
                        for s_next, p_trans in result.items():
                            p_state[s, int(s_next)] += pa * pb * float(p_trans)
                    else:
                        p_state[s, int(result)] += pa * pb

        # Discounted state occupancy d_gamma = (1-gamma) * rho0 * (I - gamma P)^(-1)
        rho0 = np.zeros(n_s, dtype=np.float64)
        if hasattr(env, "random_start") and env.random_start:
            rho0[:] = 1.0 / n_s
        else:
            rho0[0] = 1.0
        mat = np.eye(n_s, dtype=np.float64) - gamma * p_state.T
        rhs = (1.0 - gamma) * rho0
        try:
            d_gamma = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            d_gamma = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        d_gamma = np.maximum(d_gamma, 0.0)
        d_gamma = d_gamma / max(float(d_gamma.sum()), 1e-12)

        sigma = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        for s in range(n_s):
            ds = float(d_gamma[s])
            if ds <= 0.0:
                continue
            leader_probs = np.asarray(self.leader_policy_obj.get_probabilities(s), dtype=np.float64)
            for a in range(n_a):
                pa = float(leader_probs[a])
                if pa <= 0.0:
                    continue

                follower_probs = np.asarray(
                    self.get_action_probs(s, a, model=target_model), dtype=np.float64,
                )
                follower_probs = follower_probs / max(float(follower_probs.sum()), 1e-12)

                mean_psi = np.zeros(feature_dim, dtype=np.float64)
                for b in range(n_b):
                    mean_psi += float(follower_probs[b]) * psi[s, a, b]

                cov_sa = np.zeros((feature_dim, feature_dim), dtype=np.float64)
                for b in range(n_b):
                    diff = psi[s, a, b] - mean_psi
                    cov_sa += float(follower_probs[b]) * np.outer(diff, diff)

                sigma += ds * pa * cov_sa

        # Numerical symmetrization.
        sigma = 0.5 * (sigma + sigma.T)
        return sigma

    def compute_reward_confidence_matrix_empirical(self, expert_trajectories, model=None) -> np.ndarray:
        """Compute empirical Sigma_k = E[Cov(psi)] from expert trajectories.
        
        Sigma_k = 1/|D_k| sum_{tau} sum_t gamma^t (psi_t - E_b[psi])( ... )^T
        """
        target_model = self.follower_model if model is None else model
        if target_model is None:
            raise RuntimeError("compute_reward_confidence_matrix_empirical requires a follower model.")
        if self.sf_learning_model is None or self.sf_learning_model.sf_table is None:
            raise RuntimeError("compute_reward_confidence_matrix_empirical requires sf_table to be populated.")
            
        feature_dim = self.mdce_irl.w.shape[0] if self.mdce_irl.w is not None else self.sf_learning_model.sf_table.shape[-1]
        sigma = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        n_fa = self.env_spec.action_space.n
        
        num_episodes = len(expert_trajectories)
        if num_episodes == 0:
            return sigma
            
        for traj in expert_trajectories:
            obs_seq = traj.get("observations", [])
            la_seq = traj.get("leader_actions", [])
            fa_seq = traj.get("follower_actions", [])
            
            for t, (s, la, fa) in enumerate(zip(obs_seq, la_seq, fa_seq)):
                s_int = int(s.item() if hasattr(s, "item") else s)
                la_int = int(la)
                fa_int = int(fa)
                
                psi_t = self.sf_learning_model.sf_table[s_int, la_int, fa_int].copy()
                
                follower_probs = np.asarray(self.get_action_probs(s_int, la_int, model=target_model), dtype=np.float64)
                follower_probs = follower_probs / max(float(follower_probs.sum()), 1e-12)
                
                mean_psi = np.zeros(feature_dim, dtype=np.float64)
                for b in range(n_fa):
                    mean_psi += float(follower_probs[b]) * self.sf_learning_model.sf_table[s_int, la_int, b]
                
                diff = psi_t - mean_psi
                gamma_t = self.discount_follower ** t
                sigma += gamma_t * np.outer(diff, diff)
                
        sigma = sigma / float(num_episodes)
        sigma = 0.5 * (sigma + sigma.T)
        return sigma

    def estimate_follower_reward_known_dynamics(
        self, trajectories, env, verbose: bool = True
    ) -> "torch.Tensor":
        """Estimate follower reward/policy by MDCE convex optimization.

        Known-dynamics mode solves the occupancy-measure convex program:
          - variables: μ(s), μ(s,a,b)
          - objective: maximize discounted causal entropy
          - constraints: feature expectation matching, Bellman flow, and
            leader-marginal consistency

        The recovered follower policy is π_F(b|s,a)=μ(s,a,b)/μ(s,a), and reward
        weights are taken from the dual variables of feature constraints.
        """
        if self.transition_fn is None:
            raise RuntimeError(
                "estimate_follower_reward_known_dynamics requires transition_fn."
            )

        n_s = self.env_spec.observation_space.n
        n_a = self.env_spec.leader_action_space.n
        n_b = self.env_spec.action_space.n
        gamma = float(self.discount_follower)

        # 1) Expert discounted feature expectation
        expert_fev_t = self.mdce_irl.compute_expert_fev(trajectories)
        expert_fev = (
            expert_fev_t.detach().cpu().numpy()
            if isinstance(expert_fev_t, torch.Tensor)
            else np.asarray(expert_fev_t, dtype=np.float64)
        )
        feature_dim = int(expert_fev.shape[0])

        # 2) Build flattened representation over x=(s,a,b)
        n_x = n_s * n_a * n_b

        def idx(s: int, a: int, b: int) -> int:
            return (s * n_a + a) * n_b + b

        # Feature matrix F[k, x] = φ_k(s,a,b)
        feat_mat = np.zeros((feature_dim, n_x), dtype=np.float64)
        # Transition matrix T[s', x] = P(s'|s,a,b)
        trans_to = np.zeros((n_s, n_x), dtype=np.float64)
        # Indices grouped by (s,a): needed for μ(s,a)=Σ_b μ(s,a,b)
        sa_indices: dict[tuple[int, int], list[int]] = {}

        for s in range(n_s):
            for a in range(n_a):
                idxs = []
                for b in range(n_b):
                    x = idx(s, a, b)
                    idxs.append(x)

                    phi_sab = self.mdce_irl.feature_fn(s, a, b)
                    feat_mat[:, x] = np.asarray(phi_sab, dtype=np.float64)

                    result = self.transition_fn(s, a, b)
                    if isinstance(result, dict):
                        for s_next, p in result.items():
                            trans_to[int(s_next), x] += float(p)
                    else:
                        trans_to[int(result), x] = 1.0

                sa_indices[(s, a)] = idxs

        # 4) Initial state distribution P0 from expert trajectories
        p0 = np.zeros(n_s, dtype=np.float64)
        for traj in trajectories:
            obs = traj.get("observations", [])
            if len(obs) > 0:
                s0 = int(obs[0].item() if hasattr(obs[0], "item") else obs[0])
                p0[s0] += 1.0
        if p0.sum() <= 0:
            p0[0] = 1.0
        p0 /= p0.sum()

        leader_pi = np.asarray(self.leader_policy_obj.policy_table, dtype=np.float64)

        # 5) Convex optimization variables
        mu_s = cp.Variable(n_s, nonneg=True)    # μ(s)
        mu_x = cp.Variable(n_x, nonneg=True)    # μ(s,a,b) flattened

        mu_sa_expr: dict[tuple[int, int], cp.Expression] = {}
        for s in range(n_s):
            for a in range(n_a):
                mu_sa_expr[(s, a)] = cp.sum(mu_x[sa_indices[(s, a)]])

        # Max causal entropy <=> Min Σ_x rel_entr(μ_x, μ_sa)
        # Broadcast denom (mu_sa) to the shape of each (state, action, action) triple (s, a, b)
        # Building a list of variables and hstack-ing them is very slow, so use a matrix form
        
        # Build the mu_sa array (size: n_s * n_a)
        mu_sa_list = []
        for s in range(n_s):
            for a in range(n_a):
                mu_sa_list.append(cp.sum(mu_x[sa_indices[(s, a)]]))
        # mu_sa_vec is a 1-D array of size (n_s * n_a,)
        mu_sa_vec = cp.hstack(mu_sa_list)

        # Build indices that repeat each element n_b times
        # e.g. for n_b=3: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
        repeat_idx = np.repeat(np.arange(n_s * n_a), n_b)

        # Expand mu_sa_vec by indexing (size: n_s * n_a * n_b)
        mu_sa_expanded = mu_sa_vec[repeat_idx]

        # Compute the objective in one go
        entropy_obj = cp.sum(cp.rel_entr(mu_x, mu_sa_expanded + 1e-12))


        constraints = []

        # Soft FEM constraints (robust to sampling noise):
        # |F mu - f_E| <= xi, xi >= 0 with penalty.
        # Soft FEM constraints (robust to sampling noise):
        xi = cp.Variable(feature_dim, nonneg=True)
        fem_penalty = 1
        
        # Vectorized definition
        feat_expr = feat_mat @ mu_x
        c_pos = feat_expr - expert_fev <= xi
        c_neg = expert_fev - feat_expr <= xi
        
        constraints.append(c_pos)
        constraints.append(c_neg)

        # Bellman flow constraint (vectorized)
        constraints.append(mu_s == p0 + gamma * (trans_to @ mu_x))

        # Leader-marginal consistency: Σ_b μ(s,a,b) = π_L(a|s) μ(s)
        for s in range(n_s):
            for a in range(n_a):
                constraints.append(mu_sa_expr[(s, a)] == leader_pi[s, a] * mu_s[s])

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
            raise RuntimeError(
                f"MDCE convex optimization failed (status={last_status})."
            )

        mu_x_val = np.maximum(mu_x.value, 0.0)
        mu_sab_val = np.zeros((n_s, n_a, n_b), dtype=np.float64)
        for s in range(n_s):
            for a in range(n_a):
                for b in range(n_b):
                    mu_sab_val[s, a, b] = mu_x_val[idx(s, a, b)]

        # 6) Recover follower policy π_F(b|s,a) = μ(s,a,b) / μ(s,a)
        pi_f = np.zeros((n_s, n_a, n_b), dtype=np.float64)
        for s in range(n_s):
            for a in range(n_a):
                denom = float(mu_sab_val[s, a].sum())
                if denom > 1e-12:
                    pi_f[s, a] = mu_sab_val[s, a] / denom
                else:
                    pi_f[s, a] = np.ones(n_b, dtype=np.float64) / n_b

        # 7) Approximate reward weights from FEM dual multipliers.
        # With soft FEM constraints, θ is approximated from dual difference.
        dual_pos = np.array(c_pos.dual_value, dtype=np.float64)
        dual_neg = np.array(c_neg.dual_value, dtype=np.float64)
        theta = -(dual_pos - dual_neg)
        self.mdce_irl.w = torch.tensor(theta, dtype=torch.float32)

        # 8) Compute true SoftVI Q-values using the recovered reward function.
        # This replaces pseudo-Q to ensure downstream calculations (e.g. term 2 gradients)
        # receive properly scaled Soft-V values instead of identically 0.0.
        reward_fn = self.mdce_irl.get_reward_fn()
        q_known = self.compute_softvi_q_values_known_dynamics(
            reward_fn=reward_fn, max_iterations=2000, tolerance=1e-6, verbose=False,
        )

        softvi_model = FollowerPolicyModel(self.env_spec, self.temperature)
        softvi_model.set_q_values(q_known)
        self.soft_q_learning = softvi_model

        # Logging
        implied_fev = feat_mat @ mu_x_val
        fem_residual = np.asarray(implied_fev - expert_fev, dtype=np.float64)
        delta_fem = float(np.max(np.abs(fem_residual)))
        gradient_norm = float(np.linalg.norm(fem_residual))
        all_q_values = np.array(list(q_known.values()), dtype=np.float64)

        self.stats["irl_delta_fem"].append(delta_fem)
        self.stats["irl_gradient_norm"].append(gradient_norm)
        self.stats["irl_q_value_mean"].append(float(np.mean(all_q_values)))

        if verbose:
            print("\n--- IRL (known-dynamics, Convex MDCE) ---")
            print(f"  Solver status: {problem.status}")
            print(f"  FEM residual (max abs): {delta_fem:.6e}")
            print(f"  FEM residual norm: {gradient_norm:.6e}")
            print(f"  Q Mean: {np.mean(all_q_values):.4f}")
            theta_np = theta if isinstance(theta, np.ndarray) else np.array(theta)
            print(f"  Reward weights θ: {np.array2string(theta_np, precision=4, suppress_small=True)}")
            self.display_learned_rewards(self.mdce_irl.get_reward_fn())
            print("Follower policy updated (known-dynamics convex MDCE).")

        return self.mdce_irl.w

    def derive_follower_policy(self, env, n_iterations=1000, verbose=True):
        """Derive follower policy using Soft Q-Learning with learned reward."""
        reward_fn = self.mdce_irl.get_reward_fn()

        soft_q_config = {
            k: v
            for k, v in self.soft_q_config.items()
            if k in ("temperature", "epsilon")
        }

        self.soft_q_learning = SoftQLearning(
            env_spec=self.env_spec,
            reward_fn=reward_fn,
            leader_policy=lambda state: self.leader_policy_obj.get_probabilities(state),
            discount=self.discount_follower,
            learning_rate=self.learning_rate_follower,
            **soft_q_config,
        )

        for iteration in range(n_iterations):
            obs, _ = env.reset()
            total_reward = 0.0

            while True:
                leader_act = self.leader_policy_obj.sample_action(obs)
                follower_act = self.soft_q_learning.sample_action(obs, leader_act)
                env_step = env.step(leader_act, follower_act)
                reward = reward_fn(obs, leader_act, follower_act)

                self.soft_q_learning.update(
                    obs, leader_act, follower_act, reward,
                    env_step.observation, env_step.last,
                )

                total_reward += reward
                obs = env_step.observation

                if env_step.last:
                    break

            if verbose and iteration % 100 == 0:
                print(f"Soft Q-Learning iteration {iteration}: reward={total_reward:.4f}")

    def initialize_dummy_learner(self):
        """Initialize follower model with dummy reward (for oracle_mode='none', first iteration)."""
        def dummy_reward_fn(s, a, b):
            return 0.0

        def dummy_leader_probs(state):
            return [1.0 / self.env_spec.leader_action_space.n] * self.env_spec.leader_action_space.n

        soft_q_config = {
            k: v
            for k, v in self.soft_q_config.items()
            if k in ("temperature", "epsilon")
        }

        self.soft_q_learning = SoftQLearning(
            env_spec=self.env_spec,
            reward_fn=dummy_reward_fn,
            leader_policy=dummy_leader_probs,
            discount=self.discount_follower,
            learning_rate=self.learning_rate_follower,
            **soft_q_config,
        )

    def adapt_to_current_leader(self, env, n_adaptation_episodes=200):
        """Adapt follower to current leader policy using learned reward.

        Supports two update styles controlled by ``soft_q_config``:
          - online TD updates per transition (default fallback)
          - full-batch Bellman updates over collected transitions
            (``adapt_softql_full_batch=True``)
        """
        w = self.mdce_irl.get_reward_params()

        def adaptation_reward_fn(s, a, b):
            phi = self.mdce_irl.feature_fn(s, a, b)
            if isinstance(phi, np.ndarray):
                phi = torch.from_numpy(phi).float()
            return torch.dot(w, phi).item()

        self.soft_q_learning.reward_fn = adaptation_reward_fn
        self.soft_q_learning.leader_policy = lambda state: self.leader_policy_obj.get_probabilities(state)

        use_full_batch = bool(self.soft_q_config.get("adapt_softql_full_batch", True))
        n_batch_updates = int(self.soft_q_config.get("adapt_softql_batch_updates", 5))
        sq_lr = float(self.soft_q_config.get("learning_rate", self.learning_rate_follower))

        transitions = []

        for _ in range(n_adaptation_episodes):
            obs, _ = env.reset()
            while True:
                leader_act = self.leader_policy_obj.sample_action(obs)
                follower_act = self.soft_q_learning.sample_action(obs, leader_act)
                env_step = env.step(leader_act, follower_act)
                r_val = adaptation_reward_fn(obs, leader_act, follower_act)

                if use_full_batch:
                    transitions.append(
                        (
                            np.asarray(obs).copy(),
                            int(np.asarray(leader_act).item()),
                            int(np.asarray(follower_act).item()),
                            float(r_val),
                            np.asarray(env_step.observation).copy(),
                            bool(env_step.last),
                        )
                    )
                else:
                    self.soft_q_learning.update(
                        obs,
                        leader_act,
                        follower_act,
                        r_val,
                        env_step.observation,
                        env_step.last,
                    )

                obs = env_step.observation
                if env_step.last:
                    break

        if use_full_batch and transitions:
            q_model = self.soft_q_learning
            for _ in range(n_batch_updates):
                target_sum = {}
                target_count = {}

                for s, a, b, r, s_next, done in transitions:
                    if done:
                        target_q = r
                    else:
                        expected_soft_value = 0.0
                        leader_action_probs = self.leader_policy_obj.get_probabilities(s_next)
                        n_a = self.env_spec.leader_policy_env_spec.action_space.n
                        for a_next in range(n_a):
                            if a_next < len(leader_action_probs):
                                v_soft = q_model.compute_soft_value(s_next, a_next)
                                expected_soft_value += leader_action_probs[a_next] * v_soft
                        target_q = r + self.discount_follower * expected_soft_value

                    key = (
                        q_model._state_to_key(s),
                        q_model._action_to_key(a),
                        q_model._action_to_key(b),
                    )
                    target_sum[key] = target_sum.get(key, 0.0) + float(target_q)
                    target_count[key] = target_count.get(key, 0) + 1

                for key, total in target_sum.items():
                    state_key, leader_key, follower_key = key
                    mean_target = total / max(target_count[key], 1)
                    current_q = q_model.Q[state_key][leader_key][follower_key]
                    new_q = current_q + sq_lr * (mean_target - current_q)
                    q_model.Q[state_key][leader_key][follower_key] = float(new_q)

    def train_oracle_softql(self, env, n_oracle_episodes=100):
        """Train follower with true reward (oracle_mode='softql')."""
        def current_leader_probs(state):
            return self.leader_policy_obj.get_probabilities(state)

        sq_lr = self.soft_q_config.get("learning_rate", self.learning_rate_follower)

        self.soft_q_learning = SoftQLearning(
            env_spec=self.env_spec,
            reward_fn=None,
            leader_policy=current_leader_probs,
            discount=self.discount_follower,
            learning_rate=sq_lr,
            temperature=self.temperature,
        )
        # Oracle SoftQL can be unstable when updated online per transition.
        # By default, run full-batch Bellman updates over a collected rollout set.
        use_full_batch = bool(self.soft_q_config.get("oracle_softql_full_batch", True))
        n_batch_updates = int(self.soft_q_config.get("oracle_softql_batch_updates", 50))

        transitions = []
        for _ in range(n_oracle_episodes):
            obs, _ = env.reset()
            while True:
                la = self.leader_policy_obj.sample_action(obs)
                fa = self.soft_q_learning.sample_action(obs, la)
                env_step = env.step(la, fa)
                true_reward = env_step.reward

                if use_full_batch:
                    transitions.append(
                        (
                            np.asarray(obs).copy(),
                            int(np.asarray(la).item()),
                            int(np.asarray(fa).item()),
                            float(true_reward),
                            np.asarray(env_step.observation).copy(),
                            bool(env_step.last),
                        )
                    )
                else:
                    self.soft_q_learning.update(
                        obs,
                        la,
                        fa,
                        true_reward,
                        env_step.observation,
                        env_step.last,
                    )

                obs = env_step.observation
                if env_step.last:
                    break

        if use_full_batch and transitions:
            q_model = self.soft_q_learning
            for _ in range(n_batch_updates):
                target_sum = {}
                target_count = {}

                for s, a, b, r, s_next, done in transitions:
                    if done:
                        target_q = r
                    else:
                        expected_soft_value = 0.0
                        leader_action_probs = current_leader_probs(s_next)
                        n_a = self.env_spec.leader_policy_env_spec.action_space.n
                        for a_next in range(n_a):
                            if a_next < len(leader_action_probs):
                                v_soft = q_model.compute_soft_value(s_next, a_next)
                                expected_soft_value += leader_action_probs[a_next] * v_soft
                        target_q = r + self.discount_follower * expected_soft_value

                    key = (
                        q_model._state_to_key(s),
                        q_model._action_to_key(a),
                        q_model._action_to_key(b),
                    )
                    target_sum[key] = target_sum.get(key, 0.0) + float(target_q)
                    target_count[key] = target_count.get(key, 0) + 1

                for key, total in target_sum.items():
                    state_key, leader_key, follower_key = key
                    mean_target = total / max(target_count[key], 1)
                    current_q = q_model.Q[state_key][leader_key][follower_key]
                    new_q = current_q + sq_lr * (mean_target - current_q)
                    q_model.Q[state_key][leader_key][follower_key] = float(new_q)

    # ------------------------------------------------------------------
    # Non-IRL: empirical surrogate follower policy
    # ------------------------------------------------------------------

    def initialize_empirical_model(self) -> None:
        """Create EmpiricalFollowerModel with Laplace-smoothed uniform init."""
        self.empirical_model = EmpiricalFollowerModel(
            env_spec=self.env_spec,
            temperature=self.temperature,
            smoothing=1.0,
        )
        self.soft_q_learning = self.empirical_model

    def update_empirical_from_data(
        self,
        observations: list,
        leader_actions: list,
        follower_actions: list,
    ) -> None:
        """Re-fit empirical follower from on-policy (s, a, b) data."""
        if not hasattr(self, "empirical_model") or self.empirical_model is None:
            self.initialize_empirical_model()
        self.empirical_model.update_from_data(
            observations, leader_actions, follower_actions
        )
        self.soft_q_learning = self.empirical_model


# ---------------------------------------------------------------------------
# Empirical follower policy (for non_irl oracle mode)
# ---------------------------------------------------------------------------

class EmpiricalFollowerModel:
    """Tabular follower policy estimated from empirical (s, a, b) counts.

    Interface is compatible with SoftQLearning where needed:
      * get_policy(state, leader_action) -> dict {b: prob}
      * sample_action(state, leader_action) -> np.ndarray
      * get_q_value(s, a, b) -> float  (fake: temp * log π̂(b|s,a))
      * Q  — nested dict used by EpisodeCollector._serialize_q_values

    The fake Q values satisfy:
        softmax(Q(s,a,:) / temperature) == π̂(b|s,a)
    so the existing parallel collection infrastructure is unchanged.

    Parameters
    ----------
    env_spec : GlobalEnvSpec  (must have discrete observation/action spaces)
    temperature : float
    smoothing : float
        Laplace smoothing added to all counts (avoids zero probabilities).
    """

    def __init__(self, env_spec, temperature: float = 1.0, smoothing: float = 1.0):
        from collections import defaultdict as _dd
        self.env_spec   = env_spec
        self.temperature = float(temperature)
        self.smoothing   = float(smoothing)

        n_s = env_spec.observation_space.n
        n_a = env_spec.leader_action_space.n
        n_b = env_spec.action_space.n
        self._n_b = n_b

        # Laplace-smoothed counts: starts as uniform
        self.counts = np.ones((n_s, n_a, n_b), dtype=np.float64) * smoothing
        self.Q = _dd(lambda: _dd(lambda: _dd(float)))
        self._rebuild_Q()

    # ------------------------------------------------------------------

    def update_from_data(
        self,
        observations: list,
        leader_actions: list,
        follower_actions: list,
    ) -> None:
        """Reset counts to Laplace smoothing, then add new (s, a, b) data."""
        self.counts[:] = self.smoothing
        for s, a, b in zip(observations, leader_actions, follower_actions):
            self.counts[int(s), int(a), int(b)] += 1.0
        self._rebuild_Q()

    def _rebuild_Q(self) -> None:
        """Recompute Q(s,a,b) = temperature * log π̂(b|s,a)."""
        from collections import defaultdict as _dd

        totals = self.counts.sum(axis=2, keepdims=True)  # (n_s, n_a, 1)
        probs  = self.counts / (totals + 1e-12)           # (n_s, n_a, n_b)
        q_arr  = self.temperature * np.log(probs + 1e-30)  # (n_s, n_a, n_b)

        self.Q = _dd(lambda: _dd(lambda: _dd(float)))
        n_s, n_a, n_b = q_arr.shape
        for s in range(n_s):
            for a in range(n_a):
                for b in range(n_b):
                    self.Q[(s,)][(a,)][(b,)] = float(q_arr[s, a, b])

    # ------------------------------------------------------------------
    # Interface expected by the leader-side components
    # ------------------------------------------------------------------

    def get_q_value(
        self,
        state,
        leader_action,
        follower_action,
    ) -> float:
        s = int(state.item() if hasattr(state, "item") else state)
        a = int(leader_action.item() if hasattr(leader_action, "item") else leader_action)
        b = int(follower_action.item() if hasattr(follower_action, "item") else follower_action)
        prob = self.counts[s, a, b] / (self.counts[s, a].sum() + 1e-12)
        return self.temperature * float(np.log(max(prob, 1e-30)))

    def get_policy(self, state, leader_action) -> dict:
        """Return dict {b: π̂(b|s,a)} for all follower actions."""
        s = int(state.item() if hasattr(state, "item") else state)
        a = int(leader_action.item() if hasattr(leader_action, "item") else leader_action)
        cnts = self.counts[s, a]
        probs = cnts / (cnts.sum() + 1e-12)
        return {b: float(probs[b]) for b in range(self._n_b)}

    def sample_action(self, state, leader_action) -> np.ndarray:
        """Sample b ~ π̂(b|s,a)."""
        policy = self.get_policy(state, leader_action)
        actions = list(policy.keys())
        probs   = np.array(list(policy.values()), dtype=np.float64)
        probs  /= probs.sum()                              # numerical safety
        idx = np.random.choice(len(actions), p=probs)
        return np.array(actions[idx], dtype=np.int32)

    def compute_soft_value(self, state, leader_action) -> float:
        """Compute soft value V(s,a) = τ log Σ_b exp(Q(s,a,b) / τ).
        
        Compatible with SoftQLearning interface for use in gradient estimation.
        """
        s = int(state.item() if hasattr(state, "item") else state)
        a = int(leader_action.item() if hasattr(leader_action, "item") else leader_action)
        
        # Get Q-values for all follower actions
        q_values = []
        for b in range(self._n_b):
            q_val = self.get_q_value(s, a, b)
            q_values.append(q_val / self.temperature)
        
        if len(q_values) == 0:
            return 0.0
        
        # Softmax: τ * log Σ_b exp(q_b / τ)
        q_array = np.array(q_values, dtype=np.float64)
        # Numerically stable logsumexp
        q_max = np.max(q_array)
        log_sum_exp = q_max + np.log(np.sum(np.exp(q_array - q_max)) + 1e-12)
        soft_value = self.temperature * log_sum_exp
        
        return float(soft_value)
