import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from blackrl.envs.base import (
    Environment,
    EnvStep,
    GlobalEnvSpec,
    StepType,
)


class BuildingThermalControlEnv(Environment):

    def __init__(
        self,
        num_zones: int,
        num_hvac_units: int,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        insulation_effect: np.ndarray,
        transition_std: float,
        temperature_range: tuple,
        hvac_range: tuple,
        stability_weight: float,
        energy_weight: float,
        insulation_cost_weight: float,
        airflow_cost_weight: float,
        max_episode_steps: int,
        init_temp_mean: np.ndarray,
        init_temp_std: float,
        adjacency: list | None = None,
    ):
        # ---- store parameters ----
        self.num_zones = num_zones
        self.num_hvac_units = num_hvac_units
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.insulation_effect = np.asarray(insulation_effect, dtype=np.float64)
        self.transition_std = float(transition_std)
        self.temperature_range = tuple(temperature_range)
        self.hvac_range = tuple(hvac_range)
        self.stability_weight = float(stability_weight)
        self.energy_weight = float(energy_weight)
        self.insulation_cost_weight = float(insulation_cost_weight)
        self.airflow_cost_weight = float(airflow_cost_weight)
        self._max_episode_steps = int(max_episode_steps)
        self.init_temp_mean = np.asarray(init_temp_mean, dtype=np.float64)
        self.init_temp_std = float(init_temp_std)
        # adjacency: list of (i, j, aa_idx) tuples for hub topology.
        # None → legacy ring topology (backward-compatible).
        self.adjacency: list[tuple[int, int, int]] | None = (
            [tuple(e) for e in adjacency] if adjacency is not None else None
        )

        # ---- spaces ----
        # Follower: HVAC control vector
        self._action_space = spaces.Box(
            low=float(hvac_range[0]),
            high=float(hvac_range[1]),
            shape=(num_hvac_units,),
            dtype=np.float32,
        )
        # Leader: [insulation_level (n), airflow_adjustment (n)], each in [0, 1]
        self._leader_action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 * num_zones,),
            dtype=np.float32,
        )
        # Observation: temperature deviation vector
        self._observation_space = spaces.Box(
            low=float(temperature_range[0]),
            high=float(temperature_range[1]),
            shape=(num_zones,),
            dtype=np.float32,
        )

        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.state: np.ndarray | None = None
        self.steps_n: int = 0
        self.np_random = None
        self.seed()

    # ------------------------------------------------------------------
    # Environment base-class properties
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):
        return self._action_space

    @property
    def leader_action_space(self):
        return self._leader_action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spec(self) -> GlobalEnvSpec:
        return self._spec

    @property
    def render_modes(self):
        return ["human"]

    # ------------------------------------------------------------------
    # Public helpers (also useful for external analysis)
    # ------------------------------------------------------------------

    def parse_leader_action(self, leader_action) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(leader_action, dtype=np.float64).flatten()
        a = np.clip(a, 0.0, 1.0)
        il = a[: self.num_zones]
        aa = a[self.num_zones : 2 * self.num_zones]
        return il, aa

    def compute_parameterized_A(
        self, insulation_level: np.ndarray, airflow_adjustment: np.ndarray
    ) -> np.ndarray:
        """Compute A(theta) with insulation and airflow control.

        Supports two topology modes:

        **Hub mode** (``self.adjacency is not None``):
            A[i,j] stores raw inter-zone conductances h_{i←j} (diagonal=0).
            insulation_effect[i] = maximum external heat-loss rate k_i.

            A_theta[i,j] = h[i,j] * aa[aa_idx]   for each edge (i,j,aa_idx)
            A_theta[i,i] = 1 - k_i*(1-il[i]) - sum_{j} A_theta[i,j]

        **Legacy ring mode** (``self.adjacency is None``):
            The original n=4 ring-topology formula (backward-compatible).
        """
        n   = self.num_zones
        ie  = self.insulation_effect
        il  = np.asarray(insulation_level,  dtype=np.float64)
        aa  = np.asarray(airflow_adjustment, dtype=np.float64)

        if self.adjacency is not None:
            # ---- Hub (adjacency-list) topology ----
            h       = self.A                          # conductance matrix; A[i,i]=0
            param_A = np.zeros((n, n), dtype=np.float64)

            # Off-diagonal: each directed edge scaled by the assigned aa parameter
            for ci, cj, k in self.adjacency:
                scale            = float(aa[k])
                param_A[ci, cj] = h[ci, cj] * scale
                param_A[cj, ci] = h[cj, ci] * scale

            # Diagonal: 1 - external_loss(il) - total_outflow_from_row_i
            # NOTE: il is clamped to at most _IL_MAX_HUB < 1 so that
            #   A_theta[i,i] = 1 - ie[i]*(1-il[i]) - outflow < 1  always.
            # Without this, il→1 makes row-sum→1 (conservative system) and
            # solve_discrete_are hangs.  The practical difference is negligible
            # since γ_F = 0.9 already discounts future heavily.
            _IL_MAX = 1.0 - 1e-3   # 0.999: effectively full insulation
            il_safe = np.clip(il, 0.0, _IL_MAX)
            for i in range(n):
                outflow       = float(np.sum(param_A[i, :]))
                param_A[i, i] = 1.0 - ie[i] * (1.0 - il_safe[i]) - outflow

            return param_A

        else:
            # ---- Legacy ring topology (n=4, backward-compatible) ----
            A       = self.A
            param_A = np.zeros((n, n), dtype=np.float64)

            for i in range(n):
                j_next = (i + 1) % n
                j_prev = (i - 1 + n) % n

                param_A[i, i] = (
                    A[i, i]
                    + ie[i] * (1.0 - il[i])
                    - A[i, j_next] * aa[i]
                    - A[i, j_prev] * aa[j_prev]
                )
                param_A[i, j_next] = A[i, j_next] * aa[i]
                param_A[j_next, i] = A[j_next, i] * aa[i]

            return param_A

    def follower_reward(
        self, state: np.ndarray, hvac_action: np.ndarray
    ) -> float:
        s = np.asarray(state, dtype=np.float64).flatten()
        u = np.asarray(hvac_action, dtype=np.float64).flatten()
        return -(float(s @ self.Q @ s) + float(u @ self.R @ u))

    def leader_reward(
        self,
        state: np.ndarray,
        hvac_action: np.ndarray,
        insulation_level: np.ndarray,
        airflow_adjustment: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        s  = np.asarray(state,             dtype=np.float64).flatten()
        u  = np.asarray(hvac_action,       dtype=np.float64).flatten()
        il = np.asarray(insulation_level,  dtype=np.float64).flatten()
        aa = np.asarray(airflow_adjustment, dtype=np.float64).flatten()

        stability        = -float(np.var(s))
        energy_cost      =  float(np.sum(u  ** 2))
        insulation_cost  =  float(np.sum(il ** 2))
        airflow_cost     =  float(np.sum(aa ** 2))

        total = (
            self.stability_weight      * stability
            - self.energy_weight       * energy_cost
            - self.insulation_cost_weight * insulation_cost
            - self.airflow_cost_weight * airflow_cost
        )
        return total, stability, energy_cost, insulation_cost, airflow_cost

    def reset(self, init_state=None):
        if init_state is not None:
            s = np.asarray(init_state, dtype=np.float64).flatten()
        else:
            noise = (
                self.np_random.standard_normal(self.num_zones) * self.init_temp_std
            )
            s = self.init_temp_mean + noise
            s = np.clip(s, self.temperature_range[0], self.temperature_range[1])

        self.state = s.astype(np.float32)
        self.steps_n = 0
        return np.asarray(self.state, dtype=np.float32), {}

    def step(self, leader_action, action):
        il, aa = self.parse_leader_action(leader_action)
        u = np.clip(
            np.asarray(action, dtype=np.float64).flatten(),
            self.hvac_range[0],
            self.hvac_range[1],
        )
        s = np.asarray(self.state, dtype=np.float64)

        # ---- rewards (based on current state before transition) ----
        f_reward = self.follower_reward(s, u)
        l_reward, stability, energy_cost, insulation_cost, airflow_cost = \
            self.leader_reward(s, u, il, aa)

        # ---- transition ----
        param_A = self.compute_parameterized_A(il, aa)
        s_next = param_A @ s + self.B @ u
        s_next += self.np_random.standard_normal(self.num_zones) * self.transition_std
        s_next = np.clip(s_next, self.temperature_range[0], self.temperature_range[1])

        self.state = s_next.astype(np.float32)
        self.steps_n += 1

        step_type = (
            StepType.TIMEOUT
            if self.steps_n >= self._max_episode_steps
            else StepType.MID
        )

        env_info = {
            "leader_action":    np.asarray(leader_action, dtype=np.float32),
            "leader_reward":    float(l_reward),
            "stability":        float(stability),
            "energy_cost":      float(energy_cost),
            "insulation_cost":  float(insulation_cost),
            "airflow_cost":     float(airflow_cost),
            "follower_reward":  float(f_reward),
            "insulation_level": il.astype(np.float32),
            "airflow_adjustment": aa.astype(np.float32),
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.asarray(u, dtype=np.float32),
            reward=float(f_reward),
            observation=np.asarray(self.state, dtype=np.float32),
            env_info=env_info,
            step_type=step_type,
        )

    def render(self, mode="human"):
        return f"BuildingThermalControl state={self.state} step={self.steps_n}"

    def close(self):
        pass
