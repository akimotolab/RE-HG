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
# Default map  (13 rows × 13 cols)
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

# ---------------------------------------------------------------------------
# Helper: parse the map
# ---------------------------------------------------------------------------
def _parse_map(map_str: str):
    """Return (grid, free_coords) where grid[r][c]==1 means wall."""
    rows = [list(line) for line in map_str.strip().split("\n")]
    grid = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    free_coords = np.argwhere(grid == 0)  # shape (n_cells, 2)  [(row, col), ...]
    return grid, free_coords


# ---------------------------------------------------------------------------
# Directions: UP, RIGHT, DOWN, LEFT
# ---------------------------------------------------------------------------
_DIRECTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


class FourRoomsEnv(Environment):
    """Four Rooms gridworld for bilevel RL.

    Parameters
    ----------
    available_goals : list of (row, col) tuples
        Cells that the leader can set as goals.
        Defaults to [(8, 9)] (the classic single goal).
    fail_prob : float
        Probability that the agent moves in a uniformly random direction
        instead of the chosen one (action noise).  Default 0.0 (deterministic).
    max_episode_steps : int
        Episode length limit.  Default 500.
    random_init : bool
        If True, sample the agent's start position uniformly from free cells.
        If False, always start at the top-left free cell.
    """

    def __init__(
        self,
        available_goals=None,
        fail_prob: float = 0.0,
        max_episode_steps: int = 500,
        random_init: bool = True,
    ):
        self._grid, self._free_coords = _parse_map(_DEFAULT_MAP_STR)
        self._n_cells = len(self._free_coords)

        # Build coord → index lookup
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

        # ---------- Spaces ----------
        # Follower: 4 directions
        self._action_space = spaces.Discrete(4)
        # Leader: choose goal
        self._leader_action_space = spaces.Discrete(self._n_goals)
        # Observation: cell index
        self._observation_space = spaces.Discrete(self._n_cells)

        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        # Mutable state
        self._agent_pos: np.ndarray | None = None  # (row, col)
        self._goal_pos: np.ndarray | None = None   # (row, col)
        self._goal_idx: int = 0
        self.steps_n: int = 0
        self.np_random = None
        self.seed()

    # ------------------------------------------------------------------
    # Properties required by Environment base class
    # ------------------------------------------------------------------

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
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pos_to_obs(self, pos) -> np.ndarray:
        """Convert (row, col) to cell index."""
        idx = self._coord_to_idx[(int(pos[0]), int(pos[1]))]
        return np.array(idx, dtype=np.int64)

    def _try_move(self, pos, direction_idx: int) -> np.ndarray:
        """Apply action; stay if hitting a wall."""
        new_pos = pos + _DIRECTIONS[direction_idx]
        r, c = int(new_pos[0]), int(new_pos[1])
        if 0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1]:
            if self._grid[r, c] == 0:
                return new_pos
        return pos.copy()

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
            If given, use this cell index as the initial agent position.

        Returns
        -------
        obs : np.ndarray  (cell index)
        episode_info : dict
        """
        if init_state is not None:
            self._agent_pos = self._free_coords[int(init_state)].copy()
        elif self.random_init:
            self._agent_pos = self._sample_free_cell()
        else:
            self._agent_pos = self._free_coords[0].copy()

        # Default goal (leader hasn't acted yet; use goal 0)
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
            Index into ``available_goals``.  Sets (or updates) the current goal.
        action : int
            Follower direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).

        Returns
        -------
        EnvStep
        """
        # Leader sets goal
        goal_idx = int(leader_action)
        self._goal_idx = goal_idx
        self._goal_pos = np.array(
            self._available_goals[goal_idx % self._n_goals], dtype=np.int64
        )

        # Action noise
        if self.fail_prob > 0.0 and self.np_random.random() < self.fail_prob:
            action = int(self.np_random.integers(0, 4))
        else:
            action = int(action)

        # Move
        self._agent_pos = self._try_move(self._agent_pos, action)
        self.steps_n += 1

        obs = self._pos_to_obs(self._agent_pos)

        # ---- Rewards ----
        at_goal = np.array_equal(self._agent_pos, self._goal_pos)
        follower_reward = 1.0 if at_goal else 0.0
        leader_reward = 1.0 if at_goal else 0.0

        # ---- Step type ----
        if at_goal:
            step_type = StepType.TERMINAL
        elif self.steps_n >= self._max_episode_steps:
            step_type = StepType.TIMEOUT
        else:
            step_type = StepType.MID

        env_info = {
            "leader_action": leader_action,
            "leader_reward": leader_reward,
            "follower_reward": follower_reward,
            "agent_pos": self._agent_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
            "at_goal": at_goal,
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.array(action, dtype=np.int64),
            reward=follower_reward,
            observation=obs,
            env_info=env_info,
            step_type=step_type,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode="human"):
        """Render the grid to stdout."""
        grid_str = self._render_ansi()
        if mode in ("human", "ansi"):
            print(grid_str)
        return grid_str

    def _render_ansi(self) -> str:
        rows = []
        for r in range(self._grid.shape[0]):
            row_chars = []
            for c in range(self._grid.shape[1]):
                pos = (r, c)
                if self._grid[r, c] == 1:
                    row_chars.append("█")
                elif self._agent_pos is not None and pos == (
                    int(self._agent_pos[0]),
                    int(self._agent_pos[1]),
                ):
                    row_chars.append("A")
                elif self._goal_pos is not None and pos == (
                    int(self._goal_pos[0]),
                    int(self._goal_pos[1]),
                ):
                    row_chars.append("G")
                else:
                    row_chars.append("·")
            rows.append("".join(row_chars))
        header = f"Step: {self.steps_n}  Goal: {tuple(self._goal_pos) if self._goal_pos is not None else '-'}"
        return header + "\n" + "\n".join(rows)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of free cells (= size of discrete observation space)."""
        return self._n_cells

    @property
    def n_goals(self) -> int:
        """Number of available goal cells."""
        return self._n_goals

    @property
    def available_goals(self):
        """List of (row, col) goal positions."""
        return list(self._available_goals)

    def get_transition_matrix(self, fail_prob: float | None = None) -> np.ndarray:
        """Compute the deterministic (or noisy) transition matrix.

        Returns
        -------
        P : np.ndarray, shape (n_cells, 4, n_cells)
            P[s, a, s'] = probability of transitioning to s' from s under action a.
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
