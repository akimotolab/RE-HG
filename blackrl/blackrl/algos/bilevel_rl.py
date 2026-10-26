"""Bi-level RL algorithm orchestrator."""

from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np
import torch

from blackrl.agents.follower.follower_policy_model import FollowerPolicyModel
from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.leader.leader_policy import LeaderPolicy
from blackrl.iteration_buffer.iteration_buffer import IterationRolloutBuffer

from blackrl.algos.episode_collector import EpisodeCollector
from blackrl.algos.evaluation import LeaderEvaluator
from blackrl.algos.follower_estimator import FollowerEstimator
from blackrl.algos.leader_actor import LeaderActor
from blackrl.algos.leader_critic import LeaderCritic


class BilevelRL:
    def __init__(
        self,
        env_spec,
        leader_policy: Callable | None = None,
        follower_policy: Callable | None = None,
        feature_fn: Callable | None = None,
        discount_leader: float = 0.99,
        discount_follower: float = 0.99,
        learning_rate_leader_critic: float | None = None,
        learning_rate_leader_actor: float | None = None,
        learning_rate_follower: float = 1e-3,
        mdce_irl_config: dict | None = None,
        soft_q_config: dict | None = None,
        transition_fn: Callable | None = None,
    ):
        self.env_spec = env_spec
        self.follower_policy = follower_policy
        self._acting_follower_model = None
        self.discount_leader = discount_leader
        self.discount_follower = discount_follower
        soft_q_config = soft_q_config or {}
        mdce_irl_config = mdce_irl_config or {}
        # beta of the TRUE follower: used only to simulate it (demonstrations,
        # evaluation, Oracle). The leader never uses it: the reward it recovers
        # by IRL is r_F / beta, so its own follower model runs at temperature 1.
        self.temperature = soft_q_config.get("temperature", 1.0)
        self.leader_temperature = 1.0

        # Leader policy (shared reference used by all components)
        self.leader_policy_obj = LeaderPolicy(
            env_spec=env_spec,
            policy=leader_policy,
        )

        # MDCE IRL
        mdce_irl = MDCEIRL(
            feature_fn=feature_fn,
            discount=discount_follower,
            **mdce_irl_config,
        )
        mdce_irl.reward_scale = 1.0 / self.temperature

        # --- Component objects ---
        self.collector = EpisodeCollector(env_spec, self.temperature)

        self.follower_estimator = FollowerEstimator(
            env_spec=env_spec,
            mdce_irl=mdce_irl,
            leader_policy_obj=self.leader_policy_obj,
            temperature=self.leader_temperature,
            discount_follower=discount_follower,
            learning_rate_follower=learning_rate_follower,
            soft_q_config=soft_q_config,
            collector=self.collector,
            transition_fn=transition_fn,
        )

        self.leader_critic = LeaderCritic(
            env_spec=env_spec,
            discount_leader=discount_leader,
            learning_rate=learning_rate_leader_critic,
        )

        self.leader_actor = LeaderActor(
            env_spec=env_spec,
            leader_policy_obj=self.leader_policy_obj,
            learning_rate=learning_rate_leader_actor,
            temperature=self.leader_temperature,
            discount_leader=discount_leader,
            discount_follower=discount_follower,
        )

        self.evaluator = LeaderEvaluator(env_spec)

        # Shared state
        self.true_follower_model = None
        self.stats = defaultdict(list)
        self._reward_info_matrix = None
        self._reward_info_vector = None

        # --- Suboptimal-follower experiment (discrete counterpart of the
        # continuous Exp 1a). Zero keeps the exact best response, i.e. the
        # published behaviour. When positive, the follower's BEHAVIOUR is the
        # T-sweep truncation of soft value iteration -- a myopic follower --
        # while every method's internal belief stays exact, so what is measured
        # is robustness to suboptimal demonstrations rather than a change to
        # the algorithms.
        #
        # The discrete setting is the cleaner test of the optimal-follower
        # assumption: with one-hot features Assumption 1 holds exactly, so any
        # degradation is attributable to follower suboptimality alone and not
        # confounded with reward-model misspecification.
        self.follower_softvi_iters = 0
        self._exact_follower_model = None

        # --- Pooled multi-environment IRL baseline (Xcsj-W2/Q3) ------------
        # Accumulated (pi_hat, mask, T) triples, one per outer iteration, used
        # only when reward_info_update == "joint_cao". The list stays empty on
        # every other code path, so the default algorithm is untouched.
        self._joint_cao_data = []
        self._joint_cao_warm_omega = None
        self._joint_cao_warm_vs = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_joint_action(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        leader_act = self.leader_policy_obj.sample_action(observation)

        # Discrete setting: follower expects integer-typed observation/action.
        obs_for_follower = observation.astype(np.int32, copy=False)
        leader_act_for_follower = np.asarray(leader_act, dtype=np.int32)

        if self._acting_follower_model is not None:
            follower_act = self.follower_estimator.sample_action(
                obs_for_follower,
                leader_act_for_follower,
                model=self._acting_follower_model,
            )
        elif callable(self.follower_policy):
            follower_act = self.follower_policy(obs_for_follower, leader_act_for_follower)
        else:
            follower_act = self.follower_estimator.sample_action(
                obs_for_follower,
                leader_act_for_follower,
                model=self.follower_estimator.follower_model,
            )

        return leader_act, follower_act

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _follower_action_probs(self, state, leader_action):
        """Convenience wrapper for get_follower_action_probs."""
        return self.follower_estimator.get_action_probs(
            state,
            leader_action,
            model=self.follower_estimator.follower_model,
        )

    def _set_acting_follower_model(self, model):
        """Set follower model used when sampling online joint actions."""
        self._acting_follower_model = model

    def _make_eval_follower_fn(self):
        """Create evaluation follower policy function from true_follower_model."""
        return LeaderEvaluator.make_true_follower_eval_fn(self.true_follower_model)

    def _get_true_follower_reward_fn(self, env):
        """Return true follower reward function if environment exposes one."""
        if hasattr(env, "reward_fn"):
            return env.reward_fn
        if hasattr(env, "follower_reward_fn"):
            return env.follower_reward_fn
        return None

    def _compute_true_best_response_q(self, env, n_iters: int | None = None):
        """Compute true follower best-response Q-values.

        Prefer known-dynamics value iteration when both transition and true
        follower reward functions are available; otherwise fall back to
        env.step-based SoftVI.

        ``n_iters`` truncates soft value iteration after that many sweeps,
        yielding a myopic (suboptimal) follower. The convergence tolerance is
        disabled in that case so the recursion really does stop at n_iters
        rather than exiting early; ``None`` reproduces the exact solve.
        """
        truncated = n_iters is not None and int(n_iters) > 0
        max_it = int(n_iters) if truncated else 2000
        # A tolerance would let the sweep exit before n_iters and silently undo
        # the truncation.
        tol_known = 0.0 if truncated else 1e-10
        tol_env = 0.0 if truncated else 1e-5

        reward_fn = self._get_true_follower_reward_fn(env)
        if self.follower_estimator.transition_fn is not None and reward_fn is not None:
            return self.follower_estimator.compute_softvi_q_values_known_dynamics(
                reward_fn=reward_fn,
                max_iterations=max_it,
                tolerance=tol_known,
                verbose=False,
                temperature=self.temperature,
            )

        return self.follower_estimator.compute_softvi_q_values(
            env,
            max_iterations=max_it,
            tolerance=tol_env,
            verbose=False,
            temperature=self.temperature,
        )

    def _build_follower_models(self, env):
        """Return (behaviour_model, exact_model) for the follower.

        They coincide unless the suboptimal-follower experiment is enabled, in
        which case the behaviour model is the truncated soft-VI solution and
        the exact model is kept for the diagnostics and for every method's
        internal belief.
        """
        Q_exact = self._compute_true_best_response_q(env)
        exact_model = FollowerPolicyModel(self.env_spec, self.temperature)
        exact_model.set_q_values(Q_exact)

        T = int(self.follower_softvi_iters or 0)
        if T <= 0:
            return exact_model, exact_model

        Q_beh = self._compute_true_best_response_q(env, n_iters=T)
        beh_model = FollowerPolicyModel(self.env_spec, self.temperature)
        beh_model.set_q_values(Q_beh)
        return beh_model, exact_model

    def _follower_total_variation(self, beh_model, exact_model) -> float:
        """Exact mean total-variation distance between the two follower policies.

        Discrete actions make this a closed-form sum, so unlike the continuous
        case no Monte-Carlo estimate is needed. Averaged uniformly over
        (state, leader action) pairs.
        """
        if beh_model is exact_model:
            return 0.0
        n_s = int(self.env_spec.observation_space.n)
        n_a = int(self.env_spec.leader_action_space.n)
        n_b = int(self.env_spec.action_space.n)
        tvs = []
        for s in range(n_s):
            for a in range(n_a):
                p = np.asarray(
                    self.follower_estimator.get_action_probs(s, a, model=beh_model),
                    dtype=np.float64,
                ).ravel()[:n_b]
                q = np.asarray(
                    self.follower_estimator.get_action_probs(s, a, model=exact_model),
                    dtype=np.float64,
                ).ravel()[:n_b]
                tvs.append(0.5 * float(np.abs(p - q).sum()))
        return float(np.mean(tvs)) if tvs else 0.0

    def _get_initial_state_distribution(self, env):
        """Return the known initial-state distribution used by the model-based branch."""
        n_s = int(self.env_spec.observation_space.n)
        rho0 = np.zeros(n_s, dtype=np.float64)
        if hasattr(env, "random_start") and env.random_start:
            rho0[:] = 1.0 / n_s
        else:
            rho0[0] = 1.0
        return rho0

    def _record_joint_cao_env(self, expert_trajectories) -> None:
        """Accumulate one environment for the pooled baseline (no-op by default).

        Called once per outer iteration when the pooled baseline is active. The
        environment is defined by the leader's CURRENT policy table, since the
        follower sees the kernel p(s'|s,a,b) f_theta(a'|s'); the demonstrations
        are the same expert trajectories the local IRL consumes, so neither
        method is given information the other lacks.
        """
        from blackrl.algos.joint_irl_cao import (
            build_transition_tensor, empirical_follower_policy,
        )

        n_s = int(self.env_spec.observation_space.n)
        n_a = int(self.env_spec.leader_action_space.n)
        n_b = int(self.env_spec.action_space.n)

        pi_hat, counts = empirical_follower_policy(
            expert_trajectories, n_s, n_a, n_b,
        )
        leader_probs = np.asarray(
            self.leader_policy_obj.policy_table, dtype=np.float64,
        ).reshape(n_s, n_a)
        T = build_transition_tensor(
            self.follower_estimator.transition_fn, leader_probs, n_s, n_a, n_b,
        )
        mask = np.repeat((counts > 0).astype(np.float64), n_b)
        self._joint_cao_data.append((pi_hat.reshape(-1), mask, T))

    def _solve_joint_cao(self, kwargs) -> float:
        """Pooled solve over every accumulated environment; sets _global_omega."""
        from blackrl.algos.joint_irl_cao import solve_joint_cao

        if not self._joint_cao_data:
            return 0.0

        n_s = int(self.env_spec.observation_space.n)
        n_a = int(self.env_spec.leader_action_space.n)
        n_b = int(self.env_spec.action_space.n)

        chunk = int(kwargs.get("joint_cao_chunk", 1))
        k = len(self._joint_cao_data)
        # Re-solve every ``chunk`` iterations (always on the first one). Between
        # re-solves the previous estimate is kept, exactly as a practitioner
        # would amortize the cost of a pooled re-fit.
        if chunk > 1 and k > 1 and (k % chunk) != 0:
            return 0.0

        omega, vs, loss = solve_joint_cao(
            self._joint_cao_data, n_s, n_a, n_b,
            gamma=float(self.discount_follower),
            # omega is in the internal weight units; 1 / reward_scale is the
            # temperature those units correspond to.
            tau=1.0 / float(self.follower_estimator.mdce_irl.reward_scale),
            n_iters=int(kwargs.get("joint_cao_iters", 2000)),
            lr=float(kwargs.get("joint_cao_lr", 0.005)),
            ridge=float(kwargs.get("joint_cao_ridge", 0.0)),
            warm_omega=self._joint_cao_warm_omega,
            warm_vs=self._joint_cao_warm_vs,
            verbose=bool(kwargs.get("joint_cao_verbose", False)),
        )
        self._joint_cao_warm_omega = omega.copy()
        self._joint_cao_warm_vs = [v.copy() for v in vs]
        self._global_omega = omega
        return float(loss)

    def _update_global_reward_param(self, kwargs):
        method = str(kwargs.get("reward_info_update", "eigenvalue")).lower()

        # Pooled baseline: does not use the accumulated information matrix at
        # all, so it must be handled before the guard below.
        if method in ("joint_cao", "joint", "pooled"):
            return self._solve_joint_cao(kwargs)

        if self._reward_info_matrix is None or self._reward_info_vector is None:
            return 0.0

        if method in ("eigenvalue", "eigen_filtered", "eigen"):
            # Main algorithm solve: U_bar * omega = v_bar
            # (U_k projection is applied before accumulation in Phase C.)
            u_bar = self._reward_info_matrix
            v_bar = self._reward_info_vector

            # Tikhonov regularization is removed. Using lstsq for robust solving of U_bar * omega = v_bar
            self._global_omega = np.linalg.lstsq(u_bar, v_bar, rcond=None)[0]

            residual = u_bar @ self._global_omega - v_bar
            return float(np.linalg.norm(residual))

        elif method in ("analytic", "analytical"):
            # Legacy: standard analytic solve without Tikhonov regularization
            self._global_omega = np.linalg.lstsq(self._reward_info_matrix, self._reward_info_vector, rcond=None)[0]
            residual = self._reward_info_matrix @ self._global_omega - self._reward_info_vector
            return float(np.linalg.norm(residual))

    def _project_covariance_matrix(self, sigma_k: np.ndarray, kwargs) -> np.ndarray:
        """Project Sigma_k to U_k = E I[Lambda > eps] E^T."""
        sigma_sym = 0.5 * (sigma_k + sigma_k.T)
        eigvals, eigvecs = np.linalg.eigh(sigma_sym)

        epsilon = float(kwargs.get("reward_info_eigen_threshold", 1e-6))
        lambda_max = np.max(eigvals) if eigvals.size > 0 else 1.0
        threshold = epsilon * max(lambda_max, 1e-12)
        filter_diag = np.where(eigvals > threshold, 1.0, 0.0)

        # Keep at least one direction for numerical stability.
        if float(np.sum(filter_diag)) == 0 and eigvals.size > 0:
            filter_diag = np.zeros_like(filter_diag)
            filter_diag[np.argmax(eigvals)] = 1.0

        u_k = eigvecs @ np.diag(filter_diag) @ eigvecs.T
        return 0.5 * (u_k + u_k.T)

    def _sample_next_state_known_dynamics(self, transition_fn, s, a, b):
        """Return next state under deterministic known dynamics."""
        return int(transition_fn(int(s), int(a), int(b)))

    def _collect_episodes_known_dynamics(self, env, n_episodes):
        """Collect transition data by sampling known dynamics (no env.step)."""
        data = {
            "observation": [], "leader_action": [], "action": [],
            "reward": [], "leader_reward": [], "next_observation": [],
            "terminal": [], "last": [], "time_step": [],
        }

        max_t = int(self.env_spec.max_episode_length)
        trans_fn = self.follower_estimator.transition_fn

        for _ in range(n_episodes):
            s_raw = env.reset()
            if hasattr(s_raw, "observation"): s_raw = s_raw.observation
            elif isinstance(s_raw, tuple): s_raw = s_raw[0]
            s = int(np.asarray(s_raw).item())
            for t in range(max_t):
                la = int(np.random.choice(self.env_spec.leader_action_space.n, p=self.leader_policy_obj.policy_table[s]))
                f_probs = np.asarray(self._follower_action_probs(s, la), dtype=np.float64)
                f_probs = f_probs / max(f_probs.sum(), 1e-12)
                fa = int(np.random.choice(self.env_spec.action_space.n, p=f_probs))

                ns = self._sample_next_state_known_dynamics(trans_fn, s, la, fa)
                r_f = float(env.reward_fn(s, la, fa)) if hasattr(env, "reward_fn") else 0.0
                r_l = float(env.leader_reward_fn(s, la, fa)) if hasattr(env, "leader_reward_fn") else r_f
                is_last = (t == max_t - 1)

                data["observation"].append(np.array(s, dtype=np.int32))
                data["leader_action"].append(la)
                data["action"].append(fa)
                data["reward"].append(r_f)
                data["leader_reward"].append(r_l)
                data["next_observation"].append(np.array(ns, dtype=np.int32))
                data["terminal"].append(0)
                data["last"].append(1 if is_last else 0)
                data["time_step"].append(t)
                s = ns

        return data

    def _generate_expert_trajectories_known_dynamics(self, env, n_trajectories):
        """Generate expert trajectories by sampling known dynamics and true follower."""
        trajectories = []
        max_t = int(self.env_spec.max_episode_length)
        trans_fn = self.follower_estimator.transition_fn

        for _ in range(n_trajectories):
            s_raw = env.reset()
            if hasattr(s_raw, "observation"): s_raw = s_raw.observation
            elif isinstance(s_raw, tuple): s_raw = s_raw[0]
            s = int(np.asarray(s_raw).item())
            obs, las, fas = [], [], []
            for _t in range(max_t):
                la = int(np.random.choice(self.env_spec.leader_action_space.n, p=self.leader_policy_obj.policy_table[s]))
                fa = int(self.true_follower_model.sample_action(s, la))
                ns = self._sample_next_state_known_dynamics(trans_fn, s, la, fa)

                obs.append(np.array(s, dtype=np.int32))
                las.append(la)
                fas.append(fa)
                s = ns

            trajectories.append(
                {
                    "observations": obs,
                    "leader_actions": las,
                    "follower_actions": fas,
                }
            )

        return trajectories

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_leader_iterations: int = 1000,
        n_episodes_per_iteration: int = 50,
        n_expert_episodes_per_iteration: int | None = None,
        n_critic_updates: int = 10,
        iteration_buffer_size: int = 100000,
        oracle_mode: str = "none",
        mdce_irl_frequency: int = 10,
        use_second_term: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """Train the bi-level RL algorithm."""
        if n_expert_episodes_per_iteration is None:
            n_expert_episodes_per_iteration = n_episodes_per_iteration

        # irl_recursive_rls compatibility defaults:
        # - warmup updates for first 2 iterations (existing behavior)
        # - no periodic refresh unless explicitly requested
        recursive_rls_warmup_iters = int(kwargs.get("recursive_rls_warmup_iters", 2))
        recursive_rls_refresh_frequency = int(kwargs.get("recursive_rls_refresh_frequency", 0))
        recursive_rls_irl_iterations = kwargs.get("recursive_rls_irl_iterations", None)
        run_seed = kwargs.get("run_seed", None)
        run_seed_int = None if run_seed is None else int(run_seed)
        explicit_recursive_rls_iters: set[int] | None = None
        if recursive_rls_irl_iterations is not None:
            if isinstance(recursive_rls_irl_iterations, str):
                tokens = [t.strip() for t in recursive_rls_irl_iterations.split(",") if t.strip()]
                explicit_recursive_rls_iters = {int(t) for t in tokens}
            else:
                explicit_recursive_rls_iters = {int(t) for t in recursive_rls_irl_iterations}

        # Initialize
        self.leader_critic.initialize()
        iteration_buffer = IterationRolloutBuffer(size=iteration_buffer_size, gamma=self.discount_leader)
        self._reward_info_matrix = None
        self._reward_info_vector = None

        # === Initial Evaluation ===
        print("\n=== Initial Evaluation (Before Training) ===")

        beh_model, exact_model = self._build_follower_models(env)
        self.true_follower_model = beh_model
        self._exact_follower_model = exact_model
        if self.follower_softvi_iters:
            tv0 = self._follower_total_variation(beh_model, exact_model)
            print(
                f"[Subopt Follower] soft-VI truncated at {self.follower_softvi_iters} "
                f"sweeps | TV(g_beh, g_exact) = {tv0:.4f}"
            )

        eval_initial = self.evaluator.evaluate(
            env, self.leader_policy_obj, self._make_eval_follower_fn(),
            n_episodes=10, return_trajectories=True, gamma=self.discount_leader
        )
        self.stats["leader_return"].append(eval_initial["mean"])
        self.stats["leader_return_std"].append(eval_initial["std"])
        self.stats["disc_leader_return"].append(eval_initial["disc_mean"])
        self.stats["disc_leader_return_std"].append(eval_initial["disc_std"])
        print(f"Leader Performance (vs Best Response): {eval_initial['mean']:.4f} ± {eval_initial['std']:.4f}")
        print(f"Discounted Leader Performance: {eval_initial['disc_mean']:.4f} ± {eval_initial['disc_std']:.4f}")

        # === Main Training Loop ===
        for iteration in range(n_leader_iterations):
            print(f"\n=== Leader Iteration {iteration} (Mode: {oracle_mode}) ===")

            rollout_seed_base = None
            expert_seed_base = None
            surrogate_expert_seed_base = None
            if run_seed_int is not None:
                iteration_base = run_seed_int * 1_000_003 + iteration * 10_007
                rollout_seed_base = iteration_base + 11
                expert_seed_base = iteration_base + 4_000_011
                surrogate_expert_seed_base = iteration_base + 8_000_033

            is_known_dyn_mode = oracle_mode in ("irl_known_dynamics", "irl_known_dynamics_every_iter", "expert_surrogate_known_dynamics_no_second_term", "softvi_known_dynamics")
            collection_follower_model = None

            # ==========================================================
            # Phase A: Follower Update
            # ==========================================================
            if verbose:
                print("Computing Best Response (SoftVI) for current leader...")

            # Behaviour model drives the demonstrations and the rollouts; the
            # exact model is retained for diagnostics (no-op unless the
            # suboptimal-follower experiment is on).
            beh_model, exact_model = self._build_follower_models(env)
            self.true_follower_model = beh_model
            self._exact_follower_model = exact_model
            if self.follower_softvi_iters:
                tv_it = self._follower_total_variation(beh_model, exact_model)
                self.stats["follower_tv"].append(tv_it)
                if verbose and iteration % 10 == 0:
                    print(f"    [Subopt Follower] TV(g_beh, g_exact) = {tv_it:.4f}")

            if oracle_mode in ("softvi", "softvi_known_dynamics"):
                if verbose:
                    mode_name = "SoftVI" if oracle_mode == "softvi" else "SoftVI, known dynamics"
                    print(f"Using Oracle ({mode_name}): Using computed optimal policy.")
                self.follower_estimator.follower_model = self.true_follower_model
                self._set_acting_follower_model(self.follower_estimator.follower_model)

            elif oracle_mode == "softql":
                if verbose:
                    print("Using Oracle (SoftQL): Learning Best Response with TRUE reward...")
                self.follower_estimator.train_oracle_softql(
                    env, n_oracle_episodes=100, temperature=self.temperature,
                )
                self._set_acting_follower_model(self.follower_estimator.follower_model)

            elif oracle_mode in ("none", "irl_recursive_rls"):
                if iteration == 0:
                    if verbose:
                        print("Step 0: Initializing Learner Model (collect with true follower for iteration 0)...")
                    self.follower_estimator.initialize_dummy_learner()
                    collection_follower_model = self.true_follower_model
                else:
                    if verbose:
                        print("Step 0: Adapting Learner to current leader...")
                    n_adaptation_episodes = int(
                        self.follower_estimator.soft_q_config.get("adaptation_episodes", 200)
                    )
                    self.follower_estimator.adapt_to_current_leader(
                        env, n_adaptation_episodes=n_adaptation_episodes
                    )
                    collection_follower_model = self.follower_estimator.follower_model

                self._set_acting_follower_model(self.follower_estimator.follower_model)

            elif oracle_mode in ("irl_known_dynamics", "irl_known_dynamics_every_iter"):
                if iteration == 0:
                    if verbose:
                        print("Step 0: Initializing Learner Model...")
                    self.follower_estimator.initialize_dummy_learner()
                else:
                    if verbose:
                        print("Step 0: Recomputing SoftVI with learned reward for current leader...")
                    reward_fn = self.follower_estimator.mdce_irl.get_reward_fn()
                    q_rls = self.follower_estimator.compute_softvi_q_values_known_dynamics(
                        reward_fn=reward_fn, max_iterations=300, tolerance=1e-10, verbose=False
                    )
                    model = FollowerPolicyModel(self.env_spec, self.leader_temperature)
                    model.set_q_values(q_rls)
                    self.follower_estimator.follower_model = model

                self._set_acting_follower_model(self.follower_estimator.follower_model)

            elif oracle_mode in ("expert_surrogate_no_second_term", "expert_surrogate_known_dynamics_no_second_term"):
                if iteration == 0:
                    if verbose:
                        print("Step 0: Initializing BC Surrogate.")
                    self.follower_estimator.initialize_empirical_model()

                if verbose:
                    print("Step 0: Collecting data using true follower (expert data).")
                collection_follower_model = self.true_follower_model
                self._set_acting_follower_model(self.follower_estimator.follower_model)

            if collection_follower_model is None:
                collection_follower_model = self.follower_estimator.follower_model

            # ==========================================================
            # Phase B: Data Collection
            # ==========================================================
            if is_known_dyn_mode and self.follower_estimator.transition_fn is None:
                raise ValueError(
                    "oracle_mode in {'irl_known_dynamics','irl_known_dynamics_every_iter'} requires transition_fn to be set "
                    "when constructing BilevelRL."
                )

            if verbose:
                if is_known_dyn_mode:
                    print("Step 1: Skipping rollout collection (known dynamics mode).")
                else:
                    print(f"Step 1: Collecting {n_episodes_per_iteration} episodes (parallelized)...")

            if is_known_dyn_mode:
                data = {"observation": []}
            else:
                # Priority for rollout sampling model:
                # 1) explicit collection override (e.g., iteration 0 in non-oracle mode),
                # 2) currently acting model,
                # 3) estimator's current follower model.
                collection_model = (
                    collection_follower_model
                    if collection_follower_model is not None
                    else (
                        self._acting_follower_model
                        if self._acting_follower_model is not None
                        else self.follower_estimator.follower_model
                    )
                )
                # # User request: Always use true follower data for critic and gradient
                # # computations in unknown dynamics modes.
                # collection_model = self.true_follower_model
                data = self.collector.collect_episodes(
                    env=env,
                    leader_policy_table=self.leader_policy_obj.policy_table,
                    follower_model=collection_model,
                    n_episodes=n_episodes_per_iteration,
                    base_seed=rollout_seed_base,
                )

            if data["observation"]:
                # Debug: follower probabilities and data distribution
                if verbose:
                    self._log_collection_debug(data)

                iteration_buffer.add_transitions(
                    observation=np.array(data["observation"]),
                    leader_action=np.array(data["leader_action"]),
                    action=np.array(data["action"]),
                    reward=np.array(data["reward"]),
                    leader_reward=np.array(data["leader_reward"]),
                    next_observation=np.array(data["next_observation"]),
                    terminal=np.array(data["terminal"]),
                    last=np.array(data["last"]),
                    time_step=np.array(data["time_step"]),
                )

            # ==========================================================
            # Phase C: IRL (oracle_mode="none" only)
            # ==========================================================
            if oracle_mode == "irl_known_dynamics_every_iter":
                should_run_mdce_irl = True
            elif oracle_mode == "irl_recursive_rls":
                if explicit_recursive_rls_iters is not None:
                    should_run_mdce_irl = iteration in explicit_recursive_rls_iters
                else:
                    in_warmup = iteration < recursive_rls_warmup_iters
                    in_refresh = (
                        recursive_rls_refresh_frequency > 0
                        and iteration % recursive_rls_refresh_frequency == 0
                    )
                    should_run_mdce_irl = in_warmup or in_refresh
            else:
                should_run_mdce_irl = iteration == 0

            if oracle_mode in ("none", "irl_recursive_rls") and should_run_mdce_irl:
                if verbose:
                    print(
                        "Step 2: Generating expert trajectories "
                        f"({n_expert_episodes_per_iteration}) & Running MDCE IRL (iter={iteration})..."
                    )

                current_expert_trajectories = self.collector.generate_expert_trajectories(
                    env, self.leader_policy_obj.policy_table, self.true_follower_model,
                    n_trajectories=n_expert_episodes_per_iteration, verbose=verbose,
                    base_seed=expert_seed_base,
                )
                self.follower_estimator.estimate_follower_reward(
                    current_expert_trajectories, env, verbose=verbose,
                )

                if oracle_mode == "irl_recursive_rls":
                    # Online reward identification using Empirical Covariance from D_k
                    sigma_k = self.follower_estimator.compute_reward_confidence_matrix_empirical(
                        expert_trajectories=current_expert_trajectories,
                        model=self.follower_estimator.follower_model,
                    )
                    omega_k = self.follower_estimator.mdce_irl.w
                    if omega_k is not None:
                        omega_k_np = omega_k.detach().cpu().numpy().astype(np.float64)

                        if self._reward_info_matrix is None:
                            dim = int(omega_k_np.shape[0])
                            self._reward_info_matrix = np.zeros((dim, dim), dtype=np.float64)
                            self._reward_info_vector = np.zeros(dim, dtype=np.float64)
                            self._global_omega = omega_k_np.copy()

                        eigvals = np.linalg.eigvalsh(0.5 * (sigma_k + sigma_k.T))
                        min_eig = float(np.min(eigvals)) if eigvals.size > 0 else 0.0

                        method = str(kwargs.get("reward_info_update", "eigenvalue")).lower()
                        if method in ("eigenvalue", "eigen_filtered", "eigen"):
                            u_k = self._project_covariance_matrix(sigma_k, kwargs)
                            self._reward_info_matrix += u_k
                            self._reward_info_vector += u_k @ omega_k_np
                        else:
                            self._reward_info_matrix += sigma_k
                            self._reward_info_vector += sigma_k @ omega_k_np

                        update_norm = self._update_global_reward_param(kwargs)

                        self.follower_estimator.mdce_irl.w = torch.tensor(self._global_omega, dtype=torch.float32)

                        if verbose:
                            method = str(kwargs.get("reward_info_update", "eigenvalue"))
                            print(
                                "Step 2.1: Updated inner SoftQL reward function "
                                f"from {method} aggregate (norm={update_norm:.3e})."
                            )
                        self.follower_estimator.soft_q_learning.reward_fn = self.follower_estimator.mdce_irl.get_reward_fn()

                        self.stats["irl_sigma_min_eig"].append(min_eig)
                        self.stats["irl_sigma_trace"].append(float(np.trace(sigma_k)))
                        self.stats["irl_reward_info_rank"].append(float(np.linalg.matrix_rank(self._reward_info_matrix)))

                        if verbose:
                            print(
                                "Step 2.1: Empirical online reward aggregation "
                                f"rank={int(np.linalg.matrix_rank(self._reward_info_matrix))})"
                            )

            elif oracle_mode == "irl_known_dynamics_every_iter" and should_run_mdce_irl:
                if verbose:
                    print(
                        "Step 2: Generating expert trajectories & Running MDCE IRL "
                        "(known-dynamics every-iter mode)..."
                    )
                current_expert_trajectories = self._generate_expert_trajectories_known_dynamics(
                    env, n_trajectories=n_expert_episodes_per_iteration,
                )
                self.follower_estimator.estimate_follower_reward_known_dynamics(
                    current_expert_trajectories, env, verbose=verbose,
                )

                # Pooled baseline: record this environment BEFORE the leader
                # update, so the accumulated set is exactly D_1, ..., D_k. No-op
                # unless reward_info_update == "joint_cao".
                if str(kwargs.get("reward_info_update", "eigenvalue")).lower() in (
                    "joint_cao", "joint", "pooled",
                ):
                    self._record_joint_cao_env(current_expert_trajectories)

                # Online reward identification (RLS-style information aggregation)
                sigma_k = self.follower_estimator.compute_reward_confidence_matrix_known_dynamics(
                    env=env,
                    model=self.follower_estimator.follower_model,
                )
                omega_k = self.follower_estimator.mdce_irl.w
                if omega_k is not None:
                    omega_k_np = omega_k.detach().cpu().numpy().astype(np.float64)

                    if self._reward_info_matrix is None:
                        dim = int(omega_k_np.shape[0])
                        self._reward_info_matrix = np.zeros((dim, dim), dtype=np.float64)
                        self._reward_info_vector = np.zeros(dim, dtype=np.float64)
                        self._global_omega = omega_k_np.copy()

                    eigvals = np.linalg.eigvalsh(0.5 * (sigma_k + sigma_k.T))
                    min_eig = float(np.min(eigvals)) if eigvals.size > 0 else 0.0

                    method = str(kwargs.get("reward_info_update", "eigenvalue")).lower()
                    if method in ("eigenvalue", "eigen_filtered", "eigen"):
                        u_k = self._project_covariance_matrix(sigma_k, kwargs)
                        self._reward_info_matrix += u_k
                        self._reward_info_vector += u_k @ omega_k_np
                    elif method in ("joint_cao", "joint", "pooled"):
                        # The pooled baseline ignores Sigma_k entirely; keep
                        # accumulating it only so the rank diagnostics stay
                        # comparable across arms.
                        self._reward_info_matrix += sigma_k
                        self._reward_info_vector += sigma_k @ omega_k_np
                    else:
                        self._reward_info_matrix += sigma_k
                        self._reward_info_vector += sigma_k @ omega_k_np

                    update_norm = self._update_global_reward_param(kwargs)

                    self.follower_estimator.mdce_irl.w = torch.tensor(self._global_omega, dtype=torch.float32)

                    # Fix: Update the follower model using the new aggregated reward!
                    if verbose:
                        method = str(kwargs.get("reward_info_update", "eigenvalue"))
                        print(
                            "Step 2.1: Recomputing SoftVI from "
                            f"{method} aggregated reward (norm={update_norm:.3e})."
                        )

                    reward_fn = self.follower_estimator.mdce_irl.get_reward_fn()
                    q_rls = self.follower_estimator.compute_softvi_q_values_known_dynamics(
                        reward_fn=reward_fn, max_iterations=300, tolerance=1e-10, verbose=False
                    )
                    model = FollowerPolicyModel(self.env_spec, self.leader_temperature)
                    model.set_q_values(q_rls)
                    self.follower_estimator.follower_model = model
                    self._set_acting_follower_model(self.follower_estimator.follower_model)

                    self.stats["irl_sigma_min_eig"].append(min_eig)
                    self.stats["irl_sigma_trace"].append(float(np.trace(sigma_k)))
                    self.stats["irl_reward_info_rank"].append(float(np.linalg.matrix_rank(self._reward_info_matrix)))

                    if verbose:
                        print(
                            "Step 2.1: Online reward aggregation "
                            f"rank={int(np.linalg.matrix_rank(self._reward_info_matrix))})"
                        )

            elif oracle_mode == "irl_known_dynamics" and should_run_mdce_irl:
                if verbose:
                    print(
                        "Step 2: Generating expert trajectories & Running MDCE IRL "
                        "(known-dynamics mode)..."
                    )
                current_expert_trajectories = self._generate_expert_trajectories_known_dynamics(
                    env, n_trajectories=n_expert_episodes_per_iteration,
                )
                self.follower_estimator.estimate_follower_reward_known_dynamics(
                    current_expert_trajectories, env, verbose=verbose,
                )

            elif oracle_mode in ("expert_surrogate_no_second_term", "expert_surrogate_known_dynamics_no_second_term"):
                if verbose:
                    print("Step 2: Generating expert trajectories & Running BC to surrogate follower...")

                if "known_dynamics" in oracle_mode:
                    current_expert_trajectories = self._generate_expert_trajectories_known_dynamics(
                        env, n_trajectories=n_expert_episodes_per_iteration,
                    )
                else:
                    current_expert_trajectories = self.collector.generate_expert_trajectories(
                        env, self.leader_policy_obj.policy_table, self.true_follower_model,
                        n_trajectories=n_expert_episodes_per_iteration, verbose=verbose,
                        base_seed=surrogate_expert_seed_base,
                    )

                obs_list, la_list, fa_list = [], [], []
                for traj in current_expert_trajectories:
                    obs_list.extend(traj["observations"])
                    la_list.extend(traj["leader_actions"])
                    fa_list.extend(traj["follower_actions"])

                self.follower_estimator.update_empirical_from_data(obs_list, la_list, fa_list)
                self._set_acting_follower_model(self.follower_estimator.follower_model)

            elif oracle_mode in ("none", "irl_known_dynamics", "irl_known_dynamics_every_iter", "irl_recursive_rls") and verbose:
                if oracle_mode == "irl_recursive_rls":
                    if explicit_recursive_rls_iters is not None:
                        sorted_iters = sorted(explicit_recursive_rls_iters)
                        print(
                            "Step 2: Skipping MDCE IRL "
                            f"(explicit IRL iterations={sorted_iters})."
                        )
                    elif recursive_rls_refresh_frequency > 0:
                        print(
                            "Step 2: Skipping MDCE IRL "
                            f"(RLS refresh every {recursive_rls_refresh_frequency} iters; "
                            f"warmup<{recursive_rls_warmup_iters})."
                        )
                    else:
                        print(
                            "Step 2: Skipping MDCE IRL "
                            f"(RLS aggregation strictly completed at iter < {recursive_rls_warmup_iters})."
                        )
                else:
                    next_irl_iter = ((iteration // mdce_irl_frequency) + 1) * mdce_irl_frequency
                    print(f"Step 2: Skipping MDCE IRL (Next: iteration {next_irl_iter})")


            # ==========================================================
            # Phase D: Leader Learning
            # ==========================================================

            # Step 3: Update Critic
            if is_known_dyn_mode:
                if verbose:
                    print(f"Step 3: Updating leader's Critic ({n_critic_updates} updates)...")
                if (
                    self.follower_estimator.transition_fn is not None
                    and hasattr(env, "leader_reward_fn")
                ):
                    self.leader_critic.update_model_based(
                        leader_policy_obj=self.leader_policy_obj,
                        get_follower_probs_fn=self._follower_action_probs,
                        transition_fn=self.follower_estimator.transition_fn,
                        leader_reward_fn=env.leader_reward_fn,
                        n_updates=n_critic_updates,
                    )
            elif iteration_buffer._current_size > 0:
                if verbose:
                    print(f"Step 3: Updating leader's Critic ({n_critic_updates} updates)...")
                self.leader_critic.update(
                    iteration_buffer, self.leader_policy_obj,
                    self._follower_action_probs,
                    n_updates=n_critic_updates,
                    batch_size=iteration_buffer._current_size,
                    replace=False,
                )

            # Step 4: Estimate gradient
            gradient_info = None
            if is_known_dyn_mode:
                if verbose:
                    print("Step 4: Estimating leader's gradient...")
                if self.follower_estimator.transition_fn is not None:
                    gradient_info = self.leader_actor.estimate_gradient_known_dynamics(
                        leader_q_table=self.leader_critic.q_table,
                        soft_q_learning=self.follower_estimator.follower_model,
                        get_follower_probs_fn=self._follower_action_probs,
                        transition_fn=self.follower_estimator.transition_fn,
                        initial_state_distribution=self._get_initial_state_distribution(env),
                        use_second_term=use_second_term,
                    )
                    if gradient_info and verbose:
                        self._log_gradient_info(gradient_info)

                    # --- 1. HG Cosine Similarity and Norm Diagnostics ---
                    if hasattr(self, "true_follower_model") and self.true_follower_model is not None:
                        oracle_gradient_info = self.leader_actor.estimate_gradient_known_dynamics(
                            leader_q_table=self.leader_critic.q_table,
                            soft_q_learning=self.true_follower_model,
                            get_follower_probs_fn=lambda s, a: self.follower_estimator.get_action_probs(s, a, model=self.true_follower_model),
                            transition_fn=self.follower_estimator.transition_fn,
                            initial_state_distribution=self._get_initial_state_distribution(env),
                            use_second_term=use_second_term,
                        )

                        hg_est = gradient_info["policy_gradients"].flatten()
                        hg_oracle = oracle_gradient_info["policy_gradients"].flatten()

                        norm_est = np.linalg.norm(hg_est)
                        norm_oracle = np.linalg.norm(hg_oracle)

                        if norm_est > 1e-12 and norm_oracle > 1e-12:
                            cosine_sim = np.dot(hg_est, hg_oracle) / (norm_est * norm_oracle)
                        else:
                            cosine_sim = 0.0

                        self.stats.setdefault("hg_cosine_similarity", []).append(float(cosine_sim))
                        self.stats.setdefault("hg_oracle_norm", []).append(float(norm_oracle))
                        self.stats.setdefault("hg_estimated_norm", []).append(float(norm_est))

                        if verbose:
                            print(f"  [Diagnostics] HG Cosine Sim: {cosine_sim:.4f} | "
                                  f"Norm(Est): {norm_est:.4f} | Norm(Oracle): {norm_oracle:.4f}")


            elif iteration_buffer._current_size > 0:
                if verbose:
                    print("Step 4: Estimating leader's gradient...")
                gradient_info = self.leader_actor.estimate_gradient(
                    iteration_buffer, self.leader_critic.q_table,
                    self.follower_estimator.follower_model,
                    self._follower_action_probs,
                    batch_size=iteration_buffer._current_size,
                    replace=False,
                    use_second_term=use_second_term,
                )

                if gradient_info and verbose:
                    self._log_gradient_info(gradient_info)

                if (
                    hasattr(self, "true_follower_model")
                    and self.true_follower_model is not None
                    and self.follower_estimator.transition_fn is not None
                ):
                    _true_get_probs = lambda s, a: self.follower_estimator.get_action_probs(
                        s, a, model=self.true_follower_model
                    )
                    oracle_gradient_info = self.leader_actor.estimate_gradient_known_dynamics(
                        leader_q_table=self.leader_critic.q_table,
                        soft_q_learning=self.true_follower_model,
                        get_follower_probs_fn=_true_get_probs,
                        transition_fn=self.follower_estimator.transition_fn,
                        initial_state_distribution=self._get_initial_state_distribution(env),
                        use_second_term=use_second_term,
                    )

                    hg_est = gradient_info["policy_gradients"].flatten()
                    hg_oracle = oracle_gradient_info["policy_gradients"].flatten()

                    norm_est = np.linalg.norm(hg_est)
                    norm_oracle = np.linalg.norm(hg_oracle)

                    if norm_est > 1e-12 and norm_oracle > 1e-12:
                        cosine_sim = np.dot(hg_est, hg_oracle) / (norm_est * norm_oracle)
                    else:
                        cosine_sim = 0.0

                    self.stats.setdefault("hg_cosine_similarity", []).append(float(cosine_sim))
                    self.stats.setdefault("hg_oracle_norm", []).append(float(norm_oracle))
                    self.stats.setdefault("hg_estimated_norm", []).append(float(norm_est))

                    if verbose:
                        print(f"  [Diagnostics] HG Cosine Sim: {cosine_sim:.4f} | "
                              f"Norm(Est): {norm_est:.4f} | Norm(Oracle): {norm_oracle:.4f}")


            # --- 2. Reward Landscape Diagnostics (All IRL modes) ---
            if ("irl" in oracle_mode or oracle_mode == "none") and hasattr(env, "reward_fn") and hasattr(env, "observation_space") and getattr(self.follower_estimator.mdce_irl, "w", None) is not None:
                n_s = env.observation_space.n
                n_a = env.leader_action_space.n
                n_b = env.action_space.n

                # 1. Standard (Unweighted) Pearson
                r_true_list = []
                r_est_list = []
                est_reward_fn = self.follower_estimator.mdce_irl.get_reward_fn()
                for s in range(n_s):
                    for a in range(n_a):
                        for b in range(n_b):
                            # Estimated reward is in r_F / beta units.
                            r_true_list.append(env.reward_fn(s, a, b) / self.temperature)
                            r_est_list.append(est_reward_fn(s, a, b))

                r_true_arr = np.array(r_true_list)
                r_est_arr = np.array(r_est_list)

                r_diff = r_est_arr - r_true_arr
                var_diff = np.var(r_diff)

                std_true = np.std(r_true_arr)
                std_est = np.std(r_est_arr)
                if std_true > 1e-8 and std_est > 1e-8:
                    correlation = float(np.corrcoef(r_true_arr, r_est_arr)[0, 1])
                else:
                    correlation = 0.0

                self.stats.setdefault("reward_pearson_corr", []).append(correlation)
                self.stats.setdefault("reward_var_diff", []).append(float(var_diff))

                # 2. Occupancy-Weighted Pearson
                weighted_corr = 0.0
                if iteration_buffer._current_size > 0:
                    from collections import Counter

                    obs_arr = iteration_buffer._buffer["observation"][:iteration_buffer._current_size].flatten().tolist()
                    la_arr = iteration_buffer._buffer["leader_action"][:iteration_buffer._current_size].flatten().tolist()
                    a_arr = iteration_buffer._buffer["action"][:iteration_buffer._current_size].flatten().tolist()

                    # Ensure they are ints so Counter hashes them correctly
                    obs_arr = [int(x) for x in obs_arr]
                    la_arr = [int(x) for x in la_arr]
                    a_arr = [int(x) for x in a_arr]

                    w_dict = Counter(zip(obs_arr, la_arr, a_arr))

                    w_arr = np.zeros(len(r_true_arr), dtype=np.float64)

                    i = 0
                    for s in range(n_s):
                        for a in range(n_a):
                            for b in range(n_b):
                                w_arr[i] = w_dict.get((s, a, b), 0)
                                i += 1

                    total_w = np.sum(w_arr)

                    if total_w == 0 and verbose:
                        print(f"  [DEBUG] w_dict keys (first 10): {list(w_dict.keys())[:10]}")
                        print(f"  [DEBUG] obs_arr[:10]: {obs_arr[:10]}, type: {type(obs_arr[0]) if obs_arr else 'N/A'}")

                    if total_w > 0:
                        w_arr = w_arr / total_w
                        mean_true = np.sum(w_arr * r_true_arr)
                        mean_est  = np.sum(w_arr * r_est_arr)
                        cov_true_est = np.sum(w_arr * (r_true_arr - mean_true) * (r_est_arr - mean_est))
                        var_true = np.sum(w_arr * (r_true_arr - mean_true)**2)
                        var_est  = np.sum(w_arr * (r_est_arr - mean_est)**2)

                        if verbose and (var_true <= 1e-12 or var_est <= 1e-12):
                            print(f"  [DEBUG-VAR] total_w={total_w:.1f}, var_true={var_true:.6f}, var_est={var_est:.6f}")

                        if var_true > 1e-12 and var_est > 1e-12:
                            weighted_corr = float(cov_true_est / np.sqrt(var_true * var_est))

                        if verbose:
                            print(f"  [DEBUG-PEARSON] w_dict len: {len(w_dict)}, total_w: {total_w:.1f}")
                            print(f"  [DEBUG-PEARSON] var_true: {var_true:.6f}, var_est: {var_est:.6f}, cov: {cov_true_est:.6f}")
                            print(f"  [DEBUG-PEARSON] non-zero w_arr count: {np.sum(w_arr > 0)}")
                            print(f"  [DEBUG-PEARSON] mean_true: {mean_true:.4f}, mean_est: {mean_est:.4f}")

                self.stats.setdefault("reward_weighted_pearson_corr", []).append(weighted_corr)

                if verbose:
                    print(f"  [Diagnostics] Reward Pearson Corr: {correlation:.4f} (Weighted: {weighted_corr:.4f}) | Var(Δr): {var_diff:.4f}")

            # Step 5: Update Actor
            if verbose:
                print("Step 5: Updating leader's Actor...")
            self.leader_actor.update(gradient_info, self.stats)

            # ==========================================================
            # Evaluation
            # ==========================================================
            if verbose:
                self.evaluator.display_leader_policy(
                    iteration, self.leader_critic.q_table, self.leader_policy_obj,
                )
                print("Recomputing Best Response (SoftVI) for updated leader policy...")

            Q_optimal_updated = self._compute_true_best_response_q(env)
            self.true_follower_model = FollowerPolicyModel(self.env_spec, self.temperature)
            self.true_follower_model.set_q_values(Q_optimal_updated)

            eval_true = self.evaluator.evaluate(
                env, self.leader_policy_obj, self._make_eval_follower_fn(),
                n_episodes=10, return_trajectories=True, gamma=self.discount_leader
            )

            self.stats["leader_return"].append(eval_true["mean"])
            self.stats["leader_return_std"].append(eval_true["std"])
            self.stats["disc_leader_return"].append(eval_true["disc_mean"])
            self.stats["disc_leader_return_std"].append(eval_true["disc_std"])

            if verbose:
                print(f"Leader Performance (vs Best Response): {eval_true['mean']:.4f} ± {eval_true['std']:.4f}")
                print(f"Discounted Leader Performance: {eval_true['disc_mean']:.4f} ± {eval_true['disc_std']:.4f}")

            iteration_buffer.clear()  # uncomment to make critic updates strictly per-iteration (on-policy-ish)

        return self.stats

    # ------------------------------------------------------------------
    # Logging helpers (keep verbose debug output out of main flow)
    # ------------------------------------------------------------------

    def _log_collection_debug(self, data):
        """Log surrogate follower probabilities and collected data distribution."""
        policy_table = self.leader_policy_obj.policy_table
        n_s = policy_table.shape[0]
        n_a = policy_table.shape[1]
        n_b = self.env_spec.action_space.n

        print("  [Surrogate Follower] π_F(b|s,a):")
        for s in range(n_s):
            print(f"    State {s}:")
            for a in range(n_a):
                probs = self._follower_action_probs(s, a)
                probs_str = "  ".join([f"P(b={b})={probs[b]:.4f}" for b in range(n_b)])
                print(f"      a={a}: {probs_str}")

        obs_arr = np.array(data["observation"])
        la_arr = np.array(data["leader_action"])
        fa_arr = np.array(data["action"])
        sab_counts = Counter(zip(obs_arr.tolist(), la_arr.tolist(), fa_arr.tolist()))
        total = len(data["observation"])
        print("  [Collected Data] (s,a,b) counts (this batch):")
        for (s, a, b), c in sorted(sab_counts.items()):
            print(f"    (s={s},a={a},b={b}): {c} ({100.0*c/total:.1f}%)")
        print(f"    total transitions: {total}")

    def _log_gradient_info(self, gradient_info):
        """Log gradient details."""
        self.stats["leader_gradient_norm"].append(gradient_info.get("gradient_norm", 0.0))

        first_term_norm = gradient_info.get("first_term_norm", 0.0)
        second_term_norm = gradient_info.get("second_term_norm", 0.0)
        total_norm = gradient_info.get("gradient_norm", 0.0)
        mean_q = gradient_info.get("mean_q_value", 0.0)
        mean_adv = gradient_info.get("mean_advantage", 0.0)
        mean_adv_follower_centered = gradient_info.get("mean_follower_centered_advantage", 0.0)
        mean_abs_adv_follower_centered = gradient_info.get("mean_abs_follower_centered_advantage", 0.0)
        mean_influence_grad_norm = gradient_info.get("mean_influence_grad_norm", 0.0)
        mean_term2_sample_scale = gradient_info.get("mean_term2_sample_scale", 0.0)
        nonzero_benefit_ratio = gradient_info.get("nonzero_benefit_ratio", 0.0)
        nonzero_influence_ratio = gradient_info.get("nonzero_influence_ratio", 0.0)


        print(f"  Gradient Norm: {total_norm:.6f}")
        print(
            f"    Term 1 (policy gradient): {first_term_norm:.6f} "
            f"({100 * first_term_norm / total_norm if total_norm > 0 else 0:.1f}%)",
        )
        print(
            f"    Term 2 (follower influence): {second_term_norm:.6f} "
            f"({100 * second_term_norm / total_norm if total_norm > 0 else 0:.1f}%)",
        )
        print(f"  Mean Q-value: {mean_q:.6f}")
        print(f"  Mean Advantage: {mean_adv:.6f}")
        print(f"  Mean Advantage (follower-centered): {mean_adv_follower_centered:.6f}")
        print(f"  Mean |Advantage| (follower-centered): {mean_abs_adv_follower_centered:.6f}")
        print(f"  Mean influence-grad norm: {mean_influence_grad_norm:.6f}")
        print(f"  Mean term2 sample scale: {mean_term2_sample_scale:.6f}")
        print(
            f"  Nonzero ratios: benefit={100.0 * nonzero_benefit_ratio:.1f}% "
            f"influence={100.0 * nonzero_influence_ratio:.1f}%"
        )

        policy_gradients = gradient_info.get("policy_gradients")
        if policy_gradients is not None:
            print("  Policy Gradients (per state-action):")
            num_states = policy_gradients.shape[0]
            num_actions = policy_gradients.shape[1]
            for s in range(num_states):
                print(f"    State {s}:")
                for a in range(num_actions):
                    grad_val = policy_gradients[s, a]
                    current_prob = self.leader_policy_obj.policy_table[s, a]
                    update_amount = self.leader_actor.learning_rate * grad_val
                    new_prob = current_prob + update_amount
                    print(
                        f"      Action {a}: grad={grad_val:8.4f}, current_π={current_prob:.4f}, "
                        f"update={update_amount:+.4f}, new_π≈{new_prob:.4f}",
                    )