"""Experiment plan template for discrete ConfMDP validation.

Goal
----
Validate the identifiability / bias-cancellation claim in a synthetic
discrete ConfMDP without using an oracle baseline.

Compared methods
----------------
1) aggregate_irl
   - Re-estimate reward repeatedly and aggregate information across theta shifts.
2) single_fixed_irl
   - Run IRL once at the beginning, then keep that reward fixed forever.

Key requirement from the discussion
-----------------------------------
- No oracle curve.
- The first IRL estimate is the initial surrogate for single_fixed_irl.
- Track whether "term3" approaches zero (theoretical expectation).
- Track reward parameter trajectory as well.

This file is intentionally a planning-and-protocol script. It does not execute
the full training pipeline by itself; it defines exactly what to implement and
what to log so the experiment can be wired into the existing scripts cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class SyntheticTaskConfig:
	"""Synthetic discrete ConfMDP setup.

	- State space: 2 states
	- Follower action space: 2 actions
	- True follower reward: all zeros
	- Expert follower policy: uniform (0.5, 0.5)
	"""

	n_states: int = 2
	n_follower_actions: int = 2
	true_reward: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
	expert_policy_uniform: float = 0.5


@dataclass
class BiasTrapConfig:
	"""Initial surrogate bias and environment shift settings."""

	init_reward_bias: tuple[float, float, float, float] = (10.0, 10.0, 0.0, 0.0)
	theta0: float = 0.9
	theta1: float = 0.1


@dataclass
class TrainingConfig:
	"""Training and evaluation loop parameters."""

	n_iterations: int = 100
	n_episodes_per_iter: int = 100
	n_eval_episodes: int = 50
	n_seeds: int = 10
	gamma_leader: float = 0.9
	gamma_follower: float = 0.9


@dataclass
class MethodConfig:
	"""Method switches for this theory validation."""

	use_oracle: bool = False
	compare_methods: tuple[str, str] = ("aggregate_irl", "single_fixed_irl")
	# single_fixed_irl: IRL only at iteration 0, then fixed surrogate.
	single_irl_only_first_iter: bool = True


@dataclass
class LoggingConfig:
	"""Metrics required to support the theoretical claim."""

	log_policy_kl_to_uniform: bool = True
	log_leader_return: bool = True
	log_term1_norm: bool = True
	log_term2_norm: bool = True
	log_term3_norm: bool = True
	log_term3_ratio: bool = True
	log_reward_vector: bool = True
	log_reward_centered_vector: bool = True
	log_reward_centered_norm: bool = True


def build_experiment_plan() -> dict[str, Any]:
	"""Return an implementation-ready plan dictionary.

	Expected behavior:
	- aggregate_irl should reduce reward bias and push term3 toward zero.
	- single_fixed_irl should keep bias after theta shift and show larger term3.
	"""

	task = SyntheticTaskConfig()
	trap = BiasTrapConfig()
	train = TrainingConfig()
	methods = MethodConfig()
	logs = LoggingConfig()

	protocol = [
		"Build a 2-state/2-action discrete configurable MDP.",
		"Set true follower reward to zero for all (s,b).",
		"Set expert follower policy to uniform 0.5.",
		"Initialize surrogate reward around [a,a,0,0] with large a.",
		"Start from theta0, shift once to theta1 at shift_iteration, then update theta by gradient at each leader iteration.",
		"Compare only aggregate_irl vs single_fixed_irl.",
		"For single_fixed_irl, perform IRL only at iteration 0.",
		"Log term1/term2/term3 norms and term3 ratio every iteration.",
		"Log reward vector and centered reward vector every iteration.",
		"Evaluate follower KL to uniform and leader return across seeds.",
	]

	expected_outcome = {
		"aggregate_irl": [
			"policy KL to uniform decreases after shift",
			"centered reward norm moves toward zero",
			"term3 norm and term3 ratio approach near-zero",
		],
		"single_fixed_irl": [
			"policy KL degrades or oscillates after shift",
			"reward bias remains",
			"term3 remains larger than aggregate_irl",
		],
	}

	return {
		"task": asdict(task),
		"bias_trap": asdict(trap),
		"training": asdict(train),
		"methods": asdict(methods),
		"logging": asdict(logs),
		"protocol_steps": protocol,
		"expected_outcome": expected_outcome,
	}


def main() -> None:
	plan = build_experiment_plan()
	print("=== Discrete ConfMDP Experiment Plan (No Oracle) ===")
	for k, v in plan.items():
		print(f"\n[{k}]")
		print(v)


if __name__ == "__main__":
	main()

