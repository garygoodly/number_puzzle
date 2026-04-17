from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from npuzzle_env import NPuzzleEnv


Board = Tuple[int, ...]
Move = int


@dataclass
class SearchStats:
    solved: bool
    expanded_nodes: int
    solution_length: int
    elapsed_sec: float
    threshold: int


class IDAStarNPuzzleSolver:
    """IDA* solver for the N-puzzle.

    Uses Manhattan distance + linear conflict as the admissible heuristic.
    Actions use the same convention as NPuzzleEnv:
        0 = up, 1 = down, 2 = left, 3 = right
    These are interpreted as moving the blank.
    """

    ACTION_DELTAS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.n_tiles = self.size * self.size
        self.goal: Board = tuple(list(range(1, self.n_tiles)) + [0])
        self.goal_positions = {
            tile: ((tile - 1) // self.size, (tile - 1) % self.size)
            for tile in range(1, self.n_tiles)
        }
        self.expanded_nodes = 0

    def is_solved(self, board: Board) -> bool:
        return board == self.goal

    def legal_actions(self, blank_idx: int) -> List[Move]:
        row, col = divmod(blank_idx, self.size)
        actions: List[Move] = []
        if row > 0:
            actions.append(0)
        if row < self.size - 1:
            actions.append(1)
        if col > 0:
            actions.append(2)
        if col < self.size - 1:
            actions.append(3)
        return actions

    def apply_action(self, board: Board, blank_idx: int, action: Move) -> Tuple[Board, int]:
        row, col = divmod(blank_idx, self.size)
        d_row, d_col = self.ACTION_DELTAS[action]
        n_row = row + d_row
        n_col = col + d_col
        swap_idx = n_row * self.size + n_col
        board_list = list(board)
        board_list[blank_idx], board_list[swap_idx] = board_list[swap_idx], board_list[blank_idx]
        return tuple(board_list), swap_idx

    def manhattan(self, board: Board) -> int:
        total = 0
        for idx, tile in enumerate(board):
            if tile == 0:
                continue
            row, col = divmod(idx, self.size)
            goal_row, goal_col = self.goal_positions[tile]
            total += abs(row - goal_row) + abs(col - goal_col)
        return total

    def linear_conflict(self, board: Board) -> int:
        conflicts = 0

        # Row conflicts.
        for row in range(self.size):
            row_tiles: List[Tuple[int, int]] = []
            for col in range(self.size):
                tile = board[row * self.size + col]
                if tile == 0:
                    continue
                goal_row, goal_col = self.goal_positions[tile]
                if goal_row == row:
                    row_tiles.append((col, goal_col))
            for i in range(len(row_tiles)):
                _, goal_col_i = row_tiles[i]
                for j in range(i + 1, len(row_tiles)):
                    _, goal_col_j = row_tiles[j]
                    if goal_col_i > goal_col_j:
                        conflicts += 1

        # Column conflicts.
        for col in range(self.size):
            col_tiles: List[Tuple[int, int]] = []
            for row in range(self.size):
                tile = board[row * self.size + col]
                if tile == 0:
                    continue
                goal_row, goal_col = self.goal_positions[tile]
                if goal_col == col:
                    col_tiles.append((row, goal_row))
            for i in range(len(col_tiles)):
                _, goal_row_i = col_tiles[i]
                for j in range(i + 1, len(col_tiles)):
                    _, goal_row_j = col_tiles[j]
                    if goal_row_i > goal_row_j:
                        conflicts += 1

        return 2 * conflicts

    def heuristic(self, board: Board) -> int:
        return self.manhattan(board) + self.linear_conflict(board)

    def _search(
        self,
        board: Board,
        blank_idx: int,
        g: int,
        bound: int,
        path: List[Move],
        last_action: Optional[Move],
        visited: set[Board],
    ) -> Tuple[int, Optional[List[Move]]]:
        h = self.heuristic(board)
        f = g + h
        if f > bound:
            return f, None
        if h == 0:
            return f, list(path)

        self.expanded_nodes += 1
        min_bound = float("inf")

        for action in self.legal_actions(blank_idx):
            if last_action is not None and action == self.OPPOSITE[last_action]:
                continue

            next_board, next_blank = self.apply_action(board, blank_idx, action)
            if next_board in visited:
                continue

            visited.add(next_board)
            path.append(action)
            t, solution = self._search(
                next_board,
                next_blank,
                g + 1,
                bound,
                path,
                action,
                visited,
            )
            if solution is not None:
                return t, solution
            if t < min_bound:
                min_bound = t
            path.pop()
            visited.remove(next_board)

        return int(min_bound), None

    def solve(
        self,
        board: Sequence[int],
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Tuple[Optional[List[Move]], SearchStats]:
        start_board = tuple(int(x) for x in board)
        if len(start_board) != self.n_tiles:
            raise ValueError(f"Expected board of length {self.n_tiles}, got {len(start_board)}")

        if not self.is_solvable(start_board):
            stats = SearchStats(
                solved=False,
                expanded_nodes=0,
                solution_length=0,
                elapsed_sec=0.0,
                threshold=-1,
            )
            return None, stats

        start_time = time.perf_counter()
        self.expanded_nodes = 0
        bound = self.heuristic(start_board)
        blank_idx = start_board.index(0)
        visited: set[Board] = {start_board}

        while True:
            if timeout_sec is not None and (time.perf_counter() - start_time) > timeout_sec:
                elapsed = time.perf_counter() - start_time
                stats = SearchStats(
                    solved=False,
                    expanded_nodes=self.expanded_nodes,
                    solution_length=0,
                    elapsed_sec=elapsed,
                    threshold=bound,
                )
                return None, stats

            threshold, solution = self._search(
                board=start_board,
                blank_idx=blank_idx,
                g=0,
                bound=bound,
                path=[],
                last_action=None,
                visited=visited,
            )
            if solution is not None:
                elapsed = time.perf_counter() - start_time
                stats = SearchStats(
                    solved=True,
                    expanded_nodes=self.expanded_nodes,
                    solution_length=len(solution),
                    elapsed_sec=elapsed,
                    threshold=bound,
                )
                return solution, stats
            if threshold == float("inf"):
                elapsed = time.perf_counter() - start_time
                stats = SearchStats(
                    solved=False,
                    expanded_nodes=self.expanded_nodes,
                    solution_length=0,
                    elapsed_sec=elapsed,
                    threshold=bound,
                )
                return None, stats

            if max_depth is not None and threshold > max_depth:
                elapsed = time.perf_counter() - start_time
                stats = SearchStats(
                    solved=False,
                    expanded_nodes=self.expanded_nodes,
                    solution_length=0,
                    elapsed_sec=elapsed,
                    threshold=int(threshold),
                )
                return None, stats
            bound = int(threshold)

    def is_solvable(self, board: Sequence[int]) -> bool:
        vals = [x for x in board if x != 0]
        inversions = 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] > vals[j]:
                    inversions += 1

        if self.size % 2 == 1:
            return inversions % 2 == 0

        blank_row_from_bottom = self.size - (board.index(0) // self.size)
        return (inversions + blank_row_from_bottom) % 2 == 1


def replay_solution(env: NPuzzleEnv, solution: Iterable[Move]) -> bool:
    for action in solution:
        _, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return bool(info["solved"])
    return env.is_solved()


def benchmark(size: int, scramble_depth: int, episodes: int, seed: int) -> None:
    env = NPuzzleEnv(size=size, max_steps=10000, seed=seed)
    solver = IDAStarNPuzzleSolver(size=size)
    times: List[float] = []
    lengths: List[int] = []
    expanded: List[int] = []
    solved_count = 0

    for episode in range(1, episodes + 1):
        state, _ = env.reset(scramble_depth=scramble_depth)
        solution, stats = solver.solve(state)
        solved_count += int(stats.solved)
        times.append(stats.elapsed_sec)
        lengths.append(stats.solution_length)
        expanded.append(stats.expanded_nodes)
        print(
            f"episode={episode:3d} solved={stats.solved} len={stats.solution_length:3d} "
            f"expanded={stats.expanded_nodes:8d} time={stats.elapsed_sec:7.4f}s"
        )

    print(
        f"benchmark solved={solved_count}/{episodes} "
        f"avg_len={np.mean(lengths):.2f} avg_expanded={np.mean(expanded):.1f} "
        f"avg_time={np.mean(times):.4f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve N-puzzle with IDA* + Manhattan + linear conflict.")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument(
        "--board",
        type=int,
        nargs="+",
        default=None,
        help="Explicit board values row-major with 0 as blank, e.g. --board 1 2 3 4 5 6 7 0 8",
    )
    parser.add_argument("--scramble-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.benchmark:
        benchmark(
            size=args.size,
            scramble_depth=args.scramble_depth,
            episodes=args.episodes,
            seed=args.seed,
        )
        return

    solver = IDAStarNPuzzleSolver(size=args.size)

    if args.board is not None:
        board = tuple(args.board)
    else:
        env = NPuzzleEnv(size=args.size, max_steps=10000, seed=args.seed)
        board, _ = env.reset(scramble_depth=args.scramble_depth)
        board = tuple(int(x) for x in board)

    print("board:", board)
    print("heuristic:", solver.heuristic(board))
    print("solvable:", solver.is_solvable(board))

    solution, stats = solver.solve(
        board=board,
        max_depth=args.max_depth,
        timeout_sec=args.timeout_sec,
    )
    print("stats:", stats)
    print("solution:", solution)

    if args.verify and solution is not None:
        env = NPuzzleEnv(size=args.size, max_steps=10000, seed=args.seed)
        env.board = np.array(board, dtype=np.int64)
        env.steps_taken = 0
        ok = replay_solution(env, solution)
        print("verified:", ok)


if __name__ == "__main__":
    main()
