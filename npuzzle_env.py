from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Action semantics: move the blank in the given direction.
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class StepInfo:
    legal: bool
    solved: bool
    manhattan: int
    moved_tile: Optional[int]


def goal_state(size: int) -> np.ndarray:
    total = size * size
    board = np.arange(1, total + 1, dtype=np.int64)
    board[-1] = 0
    return board


def board_to_tuple(board: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in board)


def is_solved(board: Sequence[int], size: int) -> bool:
    target = size * size
    for idx, value in enumerate(board[:-1]):
        if value != idx + 1:
            return False
    return int(board[-1]) == 0 and len(board) == target


def blank_index(board: Sequence[int]) -> int:
    for i, value in enumerate(board):
        if int(value) == 0:
            return i
    raise ValueError("Board has no blank tile (0).")


def legal_actions_mask_from_board(board: Sequence[int], size: int) -> np.ndarray:
    idx = blank_index(board)
    row, col = divmod(idx, size)
    mask = np.zeros(4, dtype=np.bool_)
    mask[0] = row > 0
    mask[1] = row < size - 1
    mask[2] = col > 0
    mask[3] = col < size - 1
    return mask


def swap_index_for_action(board: Sequence[int], size: int, action: int) -> Optional[int]:
    if action not in ACTION_DELTAS:
        return None

    idx = blank_index(board)
    row, col = divmod(idx, size)
    dr, dc = ACTION_DELTAS[action]
    new_row = row + dr
    new_col = col + dc
    if not (0 <= new_row < size and 0 <= new_col < size):
        return None
    return new_row * size + new_col


def tile_for_action(board: Sequence[int], size: int, action: int) -> Optional[int]:
    swap_idx = swap_index_for_action(board, size, action)
    if swap_idx is None:
        return None
    return int(board[swap_idx])


def apply_action_to_board(board: Sequence[int], size: int, action: int) -> Optional[np.ndarray]:
    swap_idx = swap_index_for_action(board, size, action)
    if swap_idx is None:
        return None
    new_board = np.array(board, dtype=np.int64, copy=True)
    empty_idx = blank_index(new_board)
    new_board[empty_idx], new_board[swap_idx] = new_board[swap_idx], new_board[empty_idx]
    return new_board


def manhattan_distance(board: Sequence[int], size: int) -> int:
    distance = 0
    for idx, value in enumerate(board):
        tile = int(value)
        if tile == 0:
            continue
        goal_idx = tile - 1
        row, col = divmod(idx, size)
        goal_row, goal_col = divmod(goal_idx, size)
        distance += abs(row - goal_row) + abs(col - goal_col)
    return int(distance)


def render_ascii(board: Sequence[int], size: int) -> str:
    width = len(str(size * size - 1))
    rows: List[str] = []
    for row in range(size):
        cells: List[str] = []
        for col in range(size):
            value = int(board[row * size + col])
            if value == 0:
                cells.append(" " * width)
            else:
                cells.append(str(value).rjust(width))
        rows.append(" ".join(cells))
    return "\n".join(rows)


class NPuzzleEnv:
    """A lightweight N-puzzle environment with a Gym-like API.

    The action space is fixed to 4 actions:
      0 -> blank UP
      1 -> blank DOWN
      2 -> blank LEFT
      3 -> blank RIGHT
    """

    def __init__(
        self,
        size: int = 3,
        max_steps: int = 128,
        step_penalty: float = 0.05,
        illegal_penalty: float = 1.0,
        solved_bonus: float = 10.0,
        manhattan_scale: float = 0.2,
        revisit_penalty: float = 0.1,
        default_scramble_depth: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")
        self.size = int(size)
        self.n_tiles = self.size * self.size
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.illegal_penalty = float(illegal_penalty)
        self.solved_bonus = float(solved_bonus)
        self.manhattan_scale = float(manhattan_scale)
        self.revisit_penalty = float(revisit_penalty)
        self.default_scramble_depth = (
            int(default_scramble_depth)
            if default_scramble_depth is not None
            else max(10, min(64, self.size * self.size * 2))
        )
        self.rng = np.random.default_rng(seed)

        self.board = goal_state(self.size)
        self.steps_taken = 0
        self._visited_states = {board_to_tuple(self.board)}

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def copy_board(self) -> np.ndarray:
        return np.array(self.board, copy=True)

    def legal_actions_mask(self) -> np.ndarray:
        return legal_actions_mask_from_board(self.board, self.size)

    def _shuffle(self, depth: int) -> np.ndarray:
        for _ in range(32):
            board = goal_state(self.size)
            for _ in range(depth):
                legal_actions = np.flatnonzero(legal_actions_mask_from_board(board, self.size))
                action = int(self.rng.choice(legal_actions))
                next_board = apply_action_to_board(board, self.size, action)
                if next_board is None:
                    raise RuntimeError("Internal shuffle failure.")
                board = next_board
            if depth <= 0 or not is_solved(board, self.size):
                return board
        return board

    def reset(
        self,
        scramble_depth: Optional[int] = None,
        board: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        self.steps_taken = 0
        if board is not None:
            board_np = np.array(board, dtype=np.int64)
            if board_np.shape != (self.n_tiles,):
                raise ValueError(
                    f"Board shape must be ({self.n_tiles},), got {board_np.shape}."
                )
            self.board = board_np
        else:
            depth = int(scramble_depth or self.default_scramble_depth)
            self.board = self._shuffle(depth)
        self._visited_states = {board_to_tuple(self.board)}
        return self.copy_board(), {"manhattan": manhattan_distance(self.board, self.size)}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        action = int(action)
        previous_manhattan = manhattan_distance(self.board, self.size)
        next_board = apply_action_to_board(self.board, self.size, action)
        self.steps_taken += 1

        if next_board is None:
            truncated = self.steps_taken >= self.max_steps
            info = StepInfo(
                legal=False,
                solved=is_solved(self.board, self.size),
                manhattan=previous_manhattan,
                moved_tile=None,
            )
            return self.copy_board(), -self.illegal_penalty, False, truncated, {
                "legal": info.legal,
                "solved": info.solved,
                "manhattan": info.manhattan,
                "moved_tile": info.moved_tile,
            }

        moved_tile = tile_for_action(self.board, self.size, action)
        self.board = next_board
        current_manhattan = manhattan_distance(self.board, self.size)
        reward = -self.step_penalty + self.manhattan_scale * (
            previous_manhattan - current_manhattan
        )

        state_key = board_to_tuple(self.board)
        if state_key in self._visited_states:
            reward -= self.revisit_penalty
        else:
            self._visited_states.add(state_key)

        solved = is_solved(self.board, self.size)
        if solved:
            reward += self.solved_bonus

        terminated = solved
        truncated = self.steps_taken >= self.max_steps and not terminated
        info = StepInfo(
            legal=True,
            solved=solved,
            manhattan=current_manhattan,
            moved_tile=moved_tile,
        )
        return self.copy_board(), reward, terminated, truncated, {
            "legal": info.legal,
            "solved": info.solved,
            "manhattan": info.manhattan,
            "moved_tile": info.moved_tile,
        }

    def solved(self) -> bool:
        return is_solved(self.board, self.size)

    def render(self) -> str:
        return render_ascii(self.board, self.size)
