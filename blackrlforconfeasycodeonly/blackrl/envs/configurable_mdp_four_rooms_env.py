"""Configurable MDP discrete environment: Four Rooms with configurable rewards.

Port / extension of ptia/HPGD/src/environments/ConfigurableFourRooms.py
(originally JAX/gymnax/distrax) → numpy/gymnasium for blackrl.

The key "Configurable MDP" feature is that the **leader** can adjust
the follower's incentive structure at each step, in addition to selecting
which goal cell the follower should try to reach.

Leader action
-------------
  Discrete integer index into ``available_goals``.
  The leader sets (or updates) the goal on every step.

Follower action
---------------
  Discrete: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.

Follower reward
---------------
  base_reward(state, action)   [default: +1 at goal, 0 otherwise]
  + incentive_fn(state, action, leader_action)   [default: 0]

Leader reward
-------------
  leader_reward_fn(state, action, leader_action)
  [default: same as follower reward, i.e. +1 when follower reaches goal]

Corridor doors
--------------
  Each corridor cell can have an independent probability ``door_closed_prob``
  of being blocked on any given step.  When blocked, the follower stays put.
  This is *fixed* at construction time (not controlled by the leader).

Example usage
-------------
  # Standard setup: leader chooses among three goal cells
  env = ConfigurableFourRoomsEnv(
      available_goals=[(5, 5), (8, 9), (3, 10)],
  )

  # Leader designs follower incentives (custom reward shaping)
  def my_incentive(state_idx, follower_action, leader_goal_idx):
      # bonus of 0.1 for every step taken towards the second goal
      ...
      return bonus

  env = ConfigurableFourRoomsEnv(
      available_goals=[(5, 5), (8, 9)],
      incentive_fn=my_incentive,
  )
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from blackrl.envs.base import (
    Environment,
    EnvStep,
    GlobalEnvSpec,
    StepType,
)

# ---------------------------------------------------------------------------
# Default map  (13 rows × 13 cols) — identical to four_rooms_env.py
# '0' = free cell, '1' = wall
# ---------------------------------------------------------------------------
_DEFAULT_MAP_STR = """\
1111111111111
1000000100001
1000000100001
1000000000001
1000000100001
1000000100001
1100010100001
1000000000001
1000000100001
1000000100001
1000001100001
1000000100001
1111111111111"""

# Corridor coordinates (row, col) — openings between rooms
_DEFAULT_CORRIDOR_COORDS: Tuple[Tuple[int, int], ...] = (
    (3, 6),   # Top/Left room boundary
    (6, 2),   # Left corridor
    (7, 9),   # Right corridor
    (10, 6),  # Bottom corridor
)

_DIRECTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def _parse_map(map_str: str):
    """Return (grid, free_coords) where grid[r][c]==1 means wall."""
    rows = [list(line) for line in map_str.strip().split("\n")]
    grid = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    free_coords = np.argwhere(grid == 0)  # shape (n_cells, 2)
    return grid, free_coords


# ---------------------------------------------------------------------------
# Default reward / incentive callables
# ---------------------------------------------------------------------------

def _default_follower_reward(
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    follower_action: int,
    leader_action: int,
) -> float:
    """Default: +1 when the follower reaches the goal, else 0."""
    return 1.0 if np.array_equal(agent_pos, goal_pos) else 0.0


def _default_leader_reward(
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    follower_action: int,
    leader_action: int,
) -> float:
    """Default: same as the follower reward."""
    return 1.0 if np.array_equal(agent_pos, goal_pos) else 0.0


def _zero_incentive(
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    follower_action: int,
    leader_action: int,
) -> float:
    """Default incentive: zero (no extra shaping)."""
    return 0.0


class ConfigurableFourRoomsEnv(Environment):
    """Four Rooms gridworld with configurable goal selection and reward shaping.

    This is the blackrl port of ConfigurableFourRooms (HPGD / ptia project).
    It extends the basic FourRoomsEnv with:
      * Configurable follower reward function (callable).
      * Configurable incentive add-on (callable, set by the leader).
      * Per-corridor door-closing probability (stochastic passages).

    Parameters
    ----------
    available_goals : list of (row, col)
        Cells the leader can target. Default: [(8, 9)] (classic single goal).
    fail_prob : float
        Probability of the follower taking a uniformly random action instead
        of the chosen one. Default 0.0.
    door_closed_prob : sequence of float, length = num corridors
        Per-corridor probability that the door is closed on a given step.
        Default: all zeros (always open).
    max_episode_steps : int
        Episode length limit. Default 500.
    random_init : bool
        If True, sample the agent start uniformly from free cells.
        If False, always start at the first free cell.
    follower_reward_fn : callable or None
        Signature: (agent_pos, goal_pos, follower_action, leader_action) → float.
        If None, uses the default (+1 at goal).
    leader_reward_fn : callable or None
        Signature: (agent_pos, goal_pos, follower_action, leader_action) → float.
        If None, mirrors the follower reward.
    incentive_fn : callable or None
        Extra reward added to the follower's base reward.
        Signature: (agent_pos, goal_pos, follower_action, leader_action) → float.
        If None, returns 0.
    corridor_coords : sequence of (row, col), optional
        Override the default corridor cell coordinates.
    """

    def __init__(
        self,
        available_goals: Optional[List[Tuple[int, int]]] = None,
        fail_prob: float = 0.0,
        door_closed_prob: Sequence[float] = (),
        max_episode_steps: int = 500,
        random_init: bool = True,
        follower_reward_fn: Optional[Callable] = None,
        leader_reward_fn: Optional[Callable] = None,
        incentive_fn: Optional[Callable] = None,
        corridor_coords: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self._grid, self._free_coords = _parse_map(_DEFAULT_MAP_STR)
        self._n_cells = len(self._free_coords)
        self._coord_to_idx = {
            (int(r), int(c)): i for i, (r, c) in enumerate(self._free_coords)
        }

        # Available goals
        if available_goals is None:
            available_goals = [(8, 9)]
        self._available_goals = [tuple(g) for g in available_goals]
        self._n_goals = len(self._available_goals)

        self.fail_prob = fail_prob
        self._max_episode_steps = max_episode_steps
        self.random_init = random_init

        # Corridor door probabilities
        if corridor_coords is None:
            corridor_coords = _DEFAULT_CORRIDOR_COORDS
        self._corridor_coords = list(corridor_coords)
        n_corridors = len(self._corridor_coords)
        if len(door_closed_prob) == 0:
            door_closed_prob = [0.0] * n_corridors
        if len(door_closed_prob) != n_corridors:
            raise ValueError(
                f"door_closed_prob length ({len(door_closed_prob)}) must match "
                f"number of corridors ({n_corridors})."
            )
        self._door_closed_prob = list(door_closed_prob)

        # Reward / incentive callables
        self._follower_reward_fn = (
            follower_reward_fn
            if follower_reward_fn is not None
            else _default_follower_reward
        )
        self._leader_reward_fn = (
            leader_reward_fn
            if leader_reward_fn is not None
            else _default_leader_reward
        )
        self._incentive_fn = (
            incentive_fn if incentive_fn is not None else _zero_incentive
        )

        # Spaces
        self._action_space = spaces.Discrete(4)
        self._leader_action_space = spaces.Discrete(self._n_goals)
        self._observation_space = spaces.Discrete(self._n_cells)

        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        # Mutable state
        self._agent_pos: Optional[np.ndarray] = None
        self._goal_pos: Optional[np.ndarray] = None
        self._goal_idx: int = 0
        self.steps_n: int = 0
        self.np_random = None
        self.seed()

    # ------------------------------------------------------------------
    # Properties
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
        return ["human", "ansi"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pos_to_obs(self, pos) -> np.ndarray:
        idx = self._coord_to_idx[(int(pos[0]), int(pos[1]))]
        return np.array(idx, dtype=np.int64)

    def _try_move(self, pos: np.ndarray, direction_idx: int) -> np.ndarray:
        """Move in direction; stay if hitting a wall."""
        new_pos = pos + _DIRECTIONS[direction_idx]
        r, c = int(new_pos[0]), int(new_pos[1])
        if 0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1]:
            if self._grid[r, c] == 0:
                return new_pos
        return pos.copy()

    def _is_corridor_blocked(self, new_pos: np.ndarray) -> bool:
        """Return True if new_pos is a corridor cell and its door is closed."""
        for (cr, cc), p_closed in zip(
            self._corridor_coords, self._door_closed_prob
        ):
            if int(new_pos[0]) == cr and int(new_pos[1]) == cc:
                if p_closed > 0.0 and self.np_random.random() < p_closed:
                    return True
        return False

    def _sample_free_cell(self) -> np.ndarray:
        idx = self.np_random.integers(0, self._n_cells)
        return self._free_coords[idx].copy()

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def reset(self, init_state=None):
        """Reset the environment.

        Parameters
        ----------
        init_state : int or None
            Cell index for the agent's start position.  If None, sample
            randomly (if random_init=True) or use the first free cell.

        Returns
        -------
        obs : np.ndarray (cell index)
        episode_info : dict
        """
        if init_state is not None:
            self._agent_pos = self._free_coords[int(init_state)].copy()
        elif self.random_init:
            self._agent_pos = self._sample_free_cell()
        else:
            self._agent_pos = self._free_coords[0].copy()

        self._goal_idx = 0
        self._goal_pos = np.array(self._available_goals[0], dtype=np.int64)
        self.steps_n = 0

        obs = self._pos_to_obs(self._agent_pos)
        episode_info = {
            "goal": self._goal_pos.copy(),
            "goal_idx": self._goal_idx,
        }
        return obs, episode_info

    def step(self, leader_action, action):
        """Step the environment.

        Parameters
        ----------
        leader_action : int
            Index into ``available_goals``. Sets the current goal.
        action : int
            Follower direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).

        Returns
        -------
        EnvStep
        """
        # Leader sets the goal
        goal_idx = int(leader_action) % self._n_goals
        self._goal_idx = goal_idx
        self._goal_pos = np.array(
            self._available_goals[goal_idx], dtype=np.int64
        )

        # Action noise
        if self.fail_prob > 0.0 and self.np_random.random() < self.fail_prob:
            action = int(self.np_random.integers(0, 4))
        else:
            action = int(action)

        # Attempt move
        new_pos = self._try_move(self._agent_pos, action)

        # Corridor door blocking
        if not np.array_equal(new_pos, self._agent_pos):
            if self._is_corridor_blocked(new_pos):
                new_pos = self._agent_pos.copy()

        self._agent_pos = new_pos
        self.steps_n += 1

        obs = self._pos_to_obs(self._agent_pos)

        # ---- Rewards ----
        base_f_reward = self._follower_reward_fn(
            self._agent_pos, self._goal_pos, action, goal_idx
        )
        incentive = self._incentive_fn(
            self._agent_pos, self._goal_pos, action, goal_idx
        )
        follower_reward = base_f_reward + incentive
        leader_reward = self._leader_reward_fn(
            self._agent_pos, self._goal_pos, action, goal_idx
        )

        # ---- Step type ----
        at_goal = np.array_equal(self._agent_pos, self._goal_pos)
        if at_goal:
            step_type = StepType.TERMINAL
        elif self.steps_n >= self._max_episode_steps:
            step_type = StepType.TIMEOUT
        else:
            step_type = StepType.MID

        env_info = {
            "leader_action": leader_action,
            "leader_reward": float(leader_reward),
            "follower_reward": float(follower_reward),
            "base_follower_reward": float(base_f_reward),
            "incentive": float(incentive),
            "agent_pos": self._agent_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
            "at_goal": at_goal,
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.array(action, dtype=np.int64),
            reward=float(follower_reward),
            observation=obs,
            env_info=env_info,
            step_type=step_type,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode="human"):
        grid_str = self._render_ansi()
        if mode in ("human", "ansi"):
            print(grid_str)
        return grid_str

    def _render_ansi(self) -> str:
        rows = []
        for r in range(self._grid.shape[0]):
            row_chars = []
            for c in range(self._grid.shape[1]):
                if self._grid[r, c] == 1:
                    row_chars.append("█")
                elif self._agent_pos is not None and (
                    int(self._agent_pos[0]) == r and int(self._agent_pos[1]) == c
                ):
                    row_chars.append("A")
                elif self._goal_pos is not None and (
                    int(self._goal_pos[0]) == r and int(self._goal_pos[1]) == c
                ):
                    row_chars.append("G")
                else:
                    row_chars.append("·")
            rows.append("".join(row_chars))
        header = (
            f"Step: {self.steps_n}  "
            f"Goal: {tuple(self._goal_pos) if self._goal_pos is not None else '-'}"
        )
        return header + "\n" + "\n".join(rows)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of free cells (= size of observation space)."""
        return self._n_cells

    @property
    def n_goals(self) -> int:
        """Number of available goal cells."""
        return self._n_goals

    @property
    def available_goals(self) -> list:
        """List of (row, col) goal positions."""
        return list(self._available_goals)

    def get_transition_matrix(self, fail_prob: Optional[float] = None) -> np.ndarray:
        """Compute the deterministic (or noisy) transition probability matrix.

        Returns
        -------
        P : np.ndarray, shape (n_cells, 4, n_cells)
            P[s, a, s'] = P(s' | s, a).
        Note
        ----
        Door-closing stochasticity is NOT included in this matrix (it is a
        per-step random event); set door_closed_prob to 0 for exact analysis.
        """
        if fail_prob is None:
            fail_prob = self.fail_prob

        n = self._n_cells
        P = np.zeros((n, 4, n), dtype=np.float32)
        for s_idx, pos in enumerate(self._free_coords):
            for a in range(4):
                new_pos = self._try_move(pos.copy(), a)
                s_next = self._coord_to_idx[(int(new_pos[0]), int(new_pos[1]))]
                if fail_prob > 0.0:
                    for a_rand in range(4):
                        new_pos_rand = self._try_move(pos.copy(), a_rand)
                        s_rand = self._coord_to_idx[
                            (int(new_pos_rand[0]), int(new_pos_rand[1]))
                        ]
                        rand_share = fail_prob / 4.0
                        det_share = (1.0 - fail_prob) if a_rand == a else 0.0
                        P[s_idx, a, s_rand] += det_share + rand_share
                else:
                    P[s_idx, a, s_next] = 1.0
        return P
