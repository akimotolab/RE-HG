"""Leader actor (gradient estimation and policy update) for bilevel RL."""

import numpy as np


class LeaderActor:
    """Handles leader's gradient estimation and softmax policy updates with Adam optimizer."""

    def __init__(self, env_spec, leader_policy_obj, learning_rate, temperature,
                 discount_leader, discount_follower):
        self.env_spec = env_spec
        self.leader_policy_obj = leader_policy_obj  # shared reference
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.discount_leader = discount_leader
        self.discount_follower = discount_follower

        # Logits table for softmax-based policy
        self.logits_table = np.log(self.leader_policy_obj.policy_table)
        for state in range(self.logits_table.shape[0]):
            self.logits_table[state] -= np.mean(self.logits_table[state])

        # Adam optimizer state
        self.adam_m = np.zeros_like(self.leader_policy_obj.policy_table)
        self.adam_v = np.zeros_like(self.leader_policy_obj.policy_table)
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_epsilon = 1e-4

    def estimate_gradient(self, iteration_buffer, leader_q_table, soft_q_learning,
                          get_follower_probs_fn, batch_size=64, replace=False, use_second_term=True):
        """Estimate leader's policy gradient.

        Args:
            iteration_buffer: Replay buffer to sample from.
            leader_q_table: Leader's Q-table (numpy array).
            soft_q_learning: Follower's soft Q-learning model (for second term).
            get_follower_probs_fn: Callable(state, leader_action) -> np.ndarray.
            batch_size: Batch size for gradient estimation.
            replace: Whether to sample with replacement.
            use_second_term: Whether to include follower influence term.
        """
        samples = iteration_buffer.sample_transitions(
            batch_size=batch_size,
            replace=replace,
            discount=False,
            with_subsequence=True,
        )

        # 1/tau must match the model's own temperature: an IRL model carries a
        # recovered reward that already absorbs beta, so it runs at tau = 1.
        tau = float(getattr(soft_q_learning, "temperature", self.temperature))

        obs = samples["observation"]
        leader_acts = samples["leader_action"]
        follower_acts = samples["action"]
        subsequences = samples["subsequence"]

        # Convert to numpy arrays if needed
        if hasattr(obs, "cpu"):
            obs = obs.cpu().numpy()
        if hasattr(leader_acts, "cpu"):
            leader_acts = leader_acts.cpu().numpy()
        if hasattr(follower_acts, "cpu"):
            follower_acts = follower_acts.cpu().numpy()

        obs = obs.flatten().astype(int)
        leader_acts = leader_acts.flatten().astype(int)
        follower_acts = follower_acts.flatten().astype(int)

        policy_table = self.leader_policy_obj.policy_table

        q_values = np.array([
            leader_q_table[obs[i], leader_acts[i], follower_acts[i]]
            for i in range(len(obs))
        ])

        first_term_gradients = np.zeros_like(policy_table)
        second_term_gradients = np.zeros_like(policy_table)

        leader_advantages = np.zeros(len(obs), dtype=np.float64)
        follower_centered_advantages = np.zeros(len(obs), dtype=np.float64)
        influence_grad_norms = np.zeros(len(obs), dtype=np.float64)
        term2_sample_scales = np.zeros(len(obs), dtype=np.float64)

        for i in range(len(obs)):
            state = obs[i]
            action = leader_acts[i]
            q_val = q_values[i]

            # Baseline V_L(s)
            v_l_baseline = 0.0
            num_leader_actions = self.env_spec.leader_action_space.n

            for a_prime in range(num_leader_actions):
                pi_l_prob = policy_table[state, a_prime]
                f_probs = get_follower_probs_fn(state, a_prime)
                q_l_expected_b = np.dot(f_probs, leader_q_table[state, a_prime])
                v_l_baseline += pi_l_prob * q_l_expected_b

            # First term in logit space: A(s,a,b) * grad_z log pi(a|s)
            # where grad_z log pi(a|s) = one_hot(a) - pi(s, :)
            leader_advantage = q_val - v_l_baseline
            leader_advantages[i] = leader_advantage
            score = -policy_table[state].copy()
            score[action] += 1.0

            first_term_gradients[state] += leader_advantage * score

            # Second term: Follower influence
            if use_second_term:
                follower_probs = get_follower_probs_fn(state, action)
                expected_q_follower = np.dot(follower_probs, leader_q_table[state, action])
                follower_centered_advantages[i] = q_val - expected_q_follower
                benefit = follower_centered_advantages[i]

                influence_gradients = np.zeros_like(policy_table)

                if soft_q_learning is not None and subsequences is not None:
                    subseq_obs = subsequences["observation"][i]
                    subseq_leader_acts = subsequences["leader_action"][i]

                    if hasattr(subseq_obs, "cpu"):
                        subseq_obs = subseq_obs.cpu().numpy()
                    if hasattr(subseq_leader_acts, "cpu"):
                        subseq_leader_acts = subseq_leader_acts.cpu().numpy()

                    subseq_obs = subseq_obs.flatten().astype(int)
                    subseq_leader_acts = subseq_leader_acts.flatten().astype(int)

                    cumulative_rho = 1.0

                    for t, (s_dot, a_dot) in enumerate(zip(subseq_obs[1:], subseq_leader_acts[1:])):
                        v_f_soft = soft_q_learning.compute_soft_value(s_dot, a_dot)

                        # Baseline V_F(s_dot)
                        v_f_baseline = 0.0
                        for a_prime in range(self.env_spec.leader_action_space.n):
                            pi_l_prob_dot = policy_table[s_dot, a_prime]
                            val = soft_q_learning.compute_soft_value(s_dot, a_prime)
                            v_f_baseline += pi_l_prob_dot * val

                        discount_factor = self.discount_follower ** (t + 1)
                        score_dot = -policy_table[s_dot].copy()
                        score_dot[a_dot] += 1.0
                        influence_gradients[s_dot] += discount_factor * cumulative_rho * (
                            (v_f_soft - v_f_baseline) * score_dot
                        )

                second_term_gradients += benefit * influence_gradients / (tau * (1.0 - self.discount_follower))
                influence_norm = float(np.linalg.norm(influence_gradients))
                influence_grad_norms[i] = influence_norm
                term2_sample_scales[i] = abs(float(benefit)) * influence_norm / max(tau, 1e-12)

        # Unbiased MC estimate under the trajectory sampling distribution.
        # Do not reweight per-(s,a), which distorts visitation weighting.
        batch_n = max(len(obs), 1)
        first_term_gradients = first_term_gradients / batch_n
        second_term_gradients = second_term_gradients / batch_n

        # Normalize by (1 - γ_L)
        first_term_normalized = first_term_gradients / (1.0 - self.discount_leader)
        second_term_normalized = (
            second_term_gradients / (1.0 - self.discount_leader)
            if use_second_term
            else np.zeros_like(first_term_normalized)
        )

        policy_gradients = first_term_normalized + (second_term_normalized if use_second_term else 0)

        first_term_norm = np.linalg.norm(first_term_normalized)
        second_term_norm = np.linalg.norm(second_term_normalized) if use_second_term else 0.0
        gradient_norm = np.linalg.norm(policy_gradients)

        # Statistics: track both leader-advantage (used in updates) and the
        # follower-centered variant (useful for diagnosing term-2 collapse).
        mean_advantage = float(np.mean(leader_advantages)) if len(leader_advantages) else 0.0
        mean_follower_centered_advantage = (
            float(np.mean(follower_centered_advantages)) if len(follower_centered_advantages) else 0.0
        )
        mean_abs_follower_centered_advantage = (
            float(np.mean(np.abs(follower_centered_advantages))) if len(follower_centered_advantages) else 0.0
        )
        mean_influence_grad_norm = (
            float(np.mean(influence_grad_norms)) if len(influence_grad_norms) else 0.0
        )
        mean_term2_sample_scale = (
            float(np.mean(term2_sample_scales)) if len(term2_sample_scales) else 0.0
        )
        nonzero_benefit_ratio = (
            float(np.mean(np.abs(follower_centered_advantages) > 1e-12)) if len(follower_centered_advantages) else 0.0
        )
        nonzero_influence_ratio = (
            float(np.mean(influence_grad_norms > 1e-12)) if len(influence_grad_norms) else 0.0
        )

        # Debug diagnostics (commented out)
        if False:
            print("\n  [DEBUG gradient≈0] Full batch diagnostic:")
            print(f"    batch_size={len(obs)}, unique(s,a,b)={len(set(zip(obs, leader_acts, follower_acts)))}")
            print(f"    Q: min={np.min(q_values):.6f}, max={np.max(q_values):.6f}, "
                  f"mean={np.mean(q_values):.6f}, n_zero={np.sum(np.abs(q_values) < 1e-10)}")
            print("    Per-sample: i (s,a,b) Q V_L(s) advantage benefit")
            for i in range(len(obs)):
                s, a, b = obs[i], leader_acts[i], follower_acts[i]
                q_v = q_values[i]
                v_l = 0.0
                for a_prime in range(self.env_spec.leader_action_space.n):
                    pi_l = policy_table[s, a_prime]
                    f_probs = get_follower_probs_fn(s, a_prime)
                    v_l += pi_l * np.dot(f_probs, leader_q_table[s, a_prime])
                adv = q_v - v_l
                if use_second_term:
                    f_probs_a = get_follower_probs_fn(s, a)
                    exp_q = np.dot(f_probs_a, leader_q_table[s, a])
                    ben = q_v - exp_q
                else:
                    ben = np.nan
                print(f"      {i} ({s},{a},{b}) Q={q_v:.4f} V_L={v_l:.4f} adv={adv:.4f} benefit={ben:.4f}")
            print("    Leader Q-table (all entries):")
            for s in range(leader_q_table.shape[0]):
                for a in range(leader_q_table.shape[1]):
                    for b in range(leader_q_table.shape[2]):
                        q = leader_q_table[s, a, b]
                        print(f"      (s={s},a={a},b={b}) Q_L={q:.6f}")
            print("  [END DEBUG gradient≈0]\n")

        return {
            "gradient_norm": gradient_norm,
            "first_term_norm": first_term_norm,
            "second_term_norm": second_term_norm,
            "estimated_gradient": mean_advantage,
            "mean_q_value": np.mean(q_values),
            "mean_advantage": mean_advantage,
            "mean_follower_centered_advantage": mean_follower_centered_advantage,
            "mean_abs_follower_centered_advantage": mean_abs_follower_centered_advantage,
            "mean_influence_grad_norm": mean_influence_grad_norm,
            "mean_term2_sample_scale": mean_term2_sample_scale,
            "nonzero_benefit_ratio": nonzero_benefit_ratio,
            "nonzero_influence_ratio": nonzero_influence_ratio,
            "mean_is_ratio": 1.0,
            "max_is_ratio": 1.0,
            "policy_gradients": policy_gradients,
        }

    def estimate_gradient_known_dynamics(
        self,
        leader_q_table,
        soft_q_learning,
        get_follower_probs_fn,
        transition_fn,
        initial_state_distribution=None,
        use_second_term=True,
    ):
        """Estimate policy gradient with known dynamics in infinite-horizon form.

        The gradient is computed from discounted state occupancy and fixed-point
        expectations under ``transition_fn`` instead of truncating the horizon.
        """
        tau = float(getattr(soft_q_learning, "temperature", self.temperature))
        policy_table = self.leader_policy_obj.policy_table
        num_states = self.env_spec.observation_space.n
        num_leader_actions = self.env_spec.leader_action_space.n
        num_follower_actions = self.env_spec.action_space.n

        if initial_state_distribution is None:
            rho0 = np.zeros(num_states, dtype=np.float64)
            rho0[0] = 1.0
        else:
            rho0 = np.asarray(initial_state_distribution, dtype=np.float64).reshape(num_states)
            rho0 = np.maximum(rho0, 0.0)
            rho0 = rho0 / max(float(rho0.sum()), 1e-12)

        def _next_state_distribution(s, a, b):
            result = transition_fn(int(s), int(a), int(b))
            if isinstance(result, dict):
                return {
                    int(next_state): float(prob)
                    for next_state, prob in result.items()
                    if float(prob) > 0.0
                }
            return {int(result): 1.0}

        transition_cache = {}
        follower_prob_cache = {}
        score_cache = {}
        follower_value_cache = {}
        leader_baseline_cache = np.zeros(num_states, dtype=np.float64)
        follower_baseline_cache = np.zeros(num_states, dtype=np.float64)
        transition_matrix = np.zeros((num_states, num_states), dtype=np.float64)

        for state in range(num_states):
            follower_baseline_state = 0.0
            for leader_action in range(num_leader_actions):
                follower_probs = np.asarray(get_follower_probs_fn(state, leader_action), dtype=np.float64)
                follower_probs = follower_probs / max(follower_probs.sum(), 1e-12)
                follower_prob_cache[(state, leader_action)] = follower_probs

                score = -policy_table[state].copy()
                score[leader_action] += 1.0
                score_matrix = np.zeros_like(policy_table)
                score_matrix[state] = score
                score_cache[(state, leader_action)] = score_matrix

                q_soft = soft_q_learning.compute_soft_value(state, leader_action) if soft_q_learning is not None else 0.0
                follower_value_cache[(state, leader_action)] = float(q_soft)

                leader_baseline_cache[state] += policy_table[state, leader_action] * float(
                    np.dot(follower_probs, leader_q_table[state, leader_action])
                )
                follower_baseline_state += policy_table[state, leader_action] * float(q_soft)

                for follower_action in range(num_follower_actions):
                    trans_dist = _next_state_distribution(state, leader_action, follower_action)
                    transition_cache[(state, leader_action, follower_action)] = trans_dist
                    for next_state, trans_prob in trans_dist.items():
                        transition_matrix[state, next_state] += (
                            policy_table[state, leader_action]
                            * float(follower_probs[follower_action])
                            * trans_prob
                        )

            follower_baseline_cache[state] = follower_baseline_state

        eye = np.eye(num_states, dtype=np.float64)
        try:
            discounted_state_occupancy = np.linalg.solve(
                (eye - self.discount_leader * transition_matrix).T,
                rho0,
            )
        except np.linalg.LinAlgError:
            discounted_state_occupancy = rho0 @ np.linalg.pinv(eye - self.discount_leader * transition_matrix)

        # Infinite-horizon fixed point for the follower influence term.
        # H[s] stores the discounted sum of future influence matrices starting from state s.
        influence_state_cache = np.zeros((num_states, *policy_table.shape), dtype=np.float64)
        max_fixed_point_iters = 2000
        tolerance = 1e-10
        for _ in range(max_fixed_point_iters):
            influence_next = np.zeros_like(influence_state_cache)
            max_delta = 0.0
            for state in range(num_states):
                state_expectation = np.zeros_like(policy_table)
                for leader_action in range(num_leader_actions):
                    leader_prob = policy_table[state, leader_action]
                    if leader_prob <= 0.0:
                        continue

                    score_matrix = score_cache[(state, leader_action)]
                    follower_value = follower_value_cache[(state, leader_action)]
                    local_term = score_matrix * (follower_value - follower_baseline_cache[state])

                    expected_future = np.zeros_like(policy_table)
                    follower_probs = follower_prob_cache[(state, leader_action)]
                    for follower_action in range(num_follower_actions):
                        action_prob = float(follower_probs[follower_action])
                        if action_prob <= 0.0:
                            continue
                        for next_state, trans_prob in transition_cache[(state, leader_action, follower_action)].items():
                            expected_future += action_prob * trans_prob * influence_state_cache[next_state]

                    state_expectation += leader_prob * (local_term + self.discount_follower * expected_future)

                influence_next[state] = state_expectation
                max_delta = max(max_delta, float(np.max(np.abs(state_expectation - influence_state_cache[state]))))
            influence_state_cache = influence_next
            if max_delta < tolerance:
                break

        first_term_gradients = np.zeros_like(policy_table)
        second_term_gradients = np.zeros_like(policy_table)
        q_weight_sum = 0.0
        adv_weight_sum = 0.0
        benefit_weight_sum = 0.0
        abs_benefit_weight_sum = 0.0
        influence_grad_norm_weight_sum = 0.0
        term2_sample_scale_weight_sum = 0.0
        total_weight = 0.0

        for state in range(num_states):
            state_mass = float(discounted_state_occupancy[state])
            if state_mass <= 0.0:
                continue

            v_l_baseline = leader_baseline_cache[state]
            influence_from_state = influence_state_cache[state]

            for leader_action in range(num_leader_actions):
                leader_prob = policy_table[state, leader_action]
                if leader_prob <= 0.0:
                    continue

                score_matrix = score_cache[(state, leader_action)]
                follower_probs = follower_prob_cache[(state, leader_action)]
                q_soft = follower_value_cache[(state, leader_action)]

                for follower_action in range(num_follower_actions):
                    follower_prob = float(follower_probs[follower_action])
                    if follower_prob <= 0.0:
                        continue

                    q_val = float(leader_q_table[state, leader_action, follower_action])
                    weight = state_mass * leader_prob * follower_prob
                    total_weight += weight
                    q_weight_sum += weight * q_val

                    leader_advantage = q_val - v_l_baseline
                    adv_weight_sum += weight * leader_advantage
                    first_term_gradients += weight * leader_advantage * score_matrix

                    if use_second_term:
                        expected_q_follower = float(np.dot(follower_probs, leader_q_table[state, leader_action]))
                        benefit = q_val - expected_q_follower
                        benefit_weight_sum += weight * benefit
                        abs_benefit_weight_sum += weight * abs(benefit)

                        transition_expectation = np.zeros_like(policy_table)
                        for next_state, trans_prob in transition_cache[(state, leader_action, follower_action)].items():
                            transition_expectation += trans_prob * influence_state_cache[next_state]

                        influence_gradients = self.discount_follower * transition_expectation
                        second_term_gradients += (
                            weight
                            * benefit
                            * influence_gradients
                            / (max(tau, 1e-8) * (1.0 - self.discount_follower))
                        )
                        influence_grad_norm_weight_sum += weight * float(np.linalg.norm(influence_gradients))
                        term2_sample_scale_weight_sum += weight * abs(benefit) * float(np.linalg.norm(influence_gradients)) / max(
                            tau,
                            1e-12,
                        )

        first_term_normalized = first_term_gradients / max(1.0 - self.discount_leader, 1e-12)
        second_term_normalized = (
            second_term_gradients / max(1.0 - self.discount_leader, 1e-12)
            if use_second_term
            else np.zeros_like(first_term_normalized)
        )
        policy_gradients = first_term_normalized + (second_term_normalized if use_second_term else 0)

        mean_q_value = q_weight_sum / max(total_weight, 1e-12)
        mean_advantage = adv_weight_sum / max(total_weight, 1e-12)
        mean_follower_centered_advantage = benefit_weight_sum / max(total_weight, 1e-12)
        mean_abs_follower_centered_advantage = abs_benefit_weight_sum / max(total_weight, 1e-12)
        mean_influence_grad_norm = influence_grad_norm_weight_sum / max(total_weight, 1e-12)
        mean_term2_sample_scale = term2_sample_scale_weight_sum / max(total_weight, 1e-12)
        nonzero_benefit_ratio = 1.0 if abs_benefit_weight_sum > 0.0 else 0.0
        nonzero_influence_ratio = 1.0 if influence_grad_norm_weight_sum > 0.0 else 0.0

        return {
            "gradient_norm": float(np.linalg.norm(policy_gradients)),
            "first_term_norm": float(np.linalg.norm(first_term_normalized)),
            "second_term_norm": float(np.linalg.norm(second_term_normalized)) if use_second_term else 0.0,
            "estimated_gradient": float(mean_advantage),
            "mean_q_value": float(mean_q_value),
            "mean_advantage": float(mean_advantage),
            "mean_follower_centered_advantage": mean_follower_centered_advantage,
            "mean_abs_follower_centered_advantage": mean_abs_follower_centered_advantage,
            "mean_influence_grad_norm": mean_influence_grad_norm,
            "mean_term2_sample_scale": mean_term2_sample_scale,
            "nonzero_benefit_ratio": nonzero_benefit_ratio,
            "nonzero_influence_ratio": nonzero_influence_ratio,
            "policy_gradients": policy_gradients,
        }

    def update(self, gradient_info, stats):
        """Update leader's policy using estimated gradient with Adam optimizer.

        Args:
            gradient_info: Dict from estimate_gradient, or None.
            stats: Shared stats dict to log into.
        """
        if gradient_info is None:
            return

        # Log gradient information
        stats.setdefault("gradient_norm", []).append(gradient_info.get("gradient_norm", 0.0))
        stats.setdefault("first_term_norm", []).append(gradient_info.get("first_term_norm", 0.0))
        stats.setdefault("second_term_norm", []).append(gradient_info.get("second_term_norm", 0.0))
        stats.setdefault("mean_q_value", []).append(gradient_info.get("mean_q_value", 0.0))
        stats.setdefault("mean_advantage", []).append(gradient_info.get("mean_advantage", 0.0))

        # Adam update on logits
        policy_gradients = gradient_info.get("policy_gradients")
        self.adam_t += 1
        self.adam_m = self.beta1 * self.adam_m + (1 - self.beta1) * policy_gradients
        self.adam_v = self.beta2 * self.adam_v + (1 - self.beta2) * (policy_gradients ** 2)
        m_hat = self.adam_m / (1 - self.beta1 ** self.adam_t)
        v_hat = self.adam_v / (1 - self.beta2 ** self.adam_t)
        update_step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)

        # Debug: Adam state for State 1 (commented out)
        if False:
            s1 = 1
            g1 = policy_gradients[s1]
            m1 = m_hat[s1]
            v1 = v_hat[s1]
            sqrt_v1 = np.sqrt(v1)
            u1 = update_step[s1]
            print(
                f"  [Adam State 1] g=({g1[0]:.2f}, {g1[1]:.2f}) "
                f"m_hat=({m1[0]:.4f}, {m1[1]:.4f}) "
                f"sqrt(v_hat)=({sqrt_v1[0]:.2f}, {sqrt_v1[1]:.2f}) "
                f"update=({u1[0]:.6f}, {u1[1]:.6f})"
            )

        # Update logits and convert to probabilities
        self.logits_table = self.logits_table + update_step

        policy_table = self.leader_policy_obj.policy_table
        for state in range(self.logits_table.shape[0]):
            logits = self.logits_table[state]
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            policy_table[state] = exp_logits / np.sum(exp_logits)
