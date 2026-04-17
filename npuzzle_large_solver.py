from __future__ import annotations

import argparse
import heapq
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from npuzzle_env import NPuzzleEnv
except Exception:
    NPuzzleEnv = None  # optional for standalone usage

Board = Tuple[int, ...]
Pos = Tuple[int, int]
Move = int


@dataclass
class PlanStats:
    solved: bool
    moves: int
    elapsed_sec: float
    expanded_nodes: int
    phases: int


ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


class StrategicNPuzzleSolver:
    """Scalable N-puzzle solver for large boards.

    This solver is designed for large N (including 12x12) where exact optimal
    search is not practical. It uses a constructive reduction strategy and a
    weighted A* micro-planner for each subgoal.

    It does NOT guarantee optimal move count. It prioritizes scalability and
    robustness on large boards.
    """

    def __init__(
        self,
        size: int,
        astar_weight: float = 1.25,
        subgoal_max_expansions: int = 250_000,
        final_max_expansions: int = 1_000_000,
    ) -> None:
        self.size = int(size)
        self.n_tiles = self.size * self.size
        self.goal: Board = tuple(list(range(1, self.n_tiles)) + [0])
        self.goal_positions: Dict[int, Pos] = {
            tile: ((tile - 1) // self.size, (tile - 1) % self.size)
            for tile in range(1, self.n_tiles)
        }
        self.goal_positions[0] = (self.size - 1, self.size - 1)
        self.astar_weight = float(astar_weight)
        self.subgoal_max_expansions = int(subgoal_max_expansions)
        self.final_max_expansions = int(final_max_expansions)
        self.expanded_nodes = 0

    def pos_to_idx(self, pos: Pos) -> int:
        return pos[0] * self.size + pos[1]

    def idx_to_pos(self, idx: int) -> Pos:
        return divmod(idx, self.size)

    def is_solved(self, board: Board) -> bool:
        return board == self.goal

    def is_solvable(self, board: Sequence[int]) -> bool:
        vals = [x for x in board if x != 0]
        inversions = 0
        for i in range(len(vals)):
            ai = vals[i]
            for j in range(i + 1, len(vals)):
                if ai > vals[j]:
                    inversions += 1

        if self.size % 2 == 1:
            return inversions % 2 == 0

        blank_row_from_bottom = self.size - (tuple(board).index(0) // self.size)
        return (inversions + blank_row_from_bottom) % 2 == 1

    def legal_actions(self, blank_idx: int, locked: Set[int]) -> List[Move]:
        row, col = self.idx_to_pos(blank_idx)
        actions: List[Move] = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                swap_idx = self.pos_to_idx((nr, nc))
                if swap_idx not in locked:
                    actions.append(action)
        return actions

    def apply_action(self, board: Board, blank_idx: int, action: Move) -> Tuple[Board, int]:
        row, col = self.idx_to_pos(blank_idx)
        dr, dc = ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc
        swap_idx = self.pos_to_idx((nr, nc))
        arr = list(board)
        arr[blank_idx], arr[swap_idx] = arr[swap_idx], arr[blank_idx]
        return tuple(arr), swap_idx

    def heuristic_targets(self, board: Board, tile_targets: Dict[int, int]) -> int:
        total = 0
        for tile, target_idx in tile_targets.items():
            cur_idx = board.index(tile)
            cr, cc = self.idx_to_pos(cur_idx)
            tr, tc = self.idx_to_pos(target_idx)
            total += abs(cr - tr) + abs(cc - tc)
        return total

    def solve_subgoal(
        self,
        board: Board,
        tile_targets: Dict[int, int],
        locked: Set[int],
        max_expansions: Optional[int] = None,
        weight: Optional[float] = None,
    ) -> Tuple[Optional[List[Move]], Board, int]:
        if all(board[idx] == tile for tile, idx in tile_targets.items()):
            return [], board, 0

        max_exp = self.subgoal_max_expansions if max_expansions is None else int(max_expansions)
        w = self.astar_weight if weight is None else float(weight)

        start = board
        start_blank = start.index(0)
        open_heap: List[Tuple[float, int, int, Board, int, Optional[Move]]] = []
        came_from: Dict[Board, Tuple[Board, Move]] = {}
        g_score: Dict[Board, int] = {start: 0}
        closed: Set[Board] = set()
        push_id = 0
        h0 = self.heuristic_targets(start, tile_targets)
        heapq.heappush(open_heap, (h0 * w, push_id, 0, start, start_blank, None))
        local_expanded = 0

        while open_heap and local_expanded < max_exp:
            _, _, g, cur, blank_idx, last_action = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            local_expanded += 1
            self.expanded_nodes += 1

            if all(cur[idx] == tile for tile, idx in tile_targets.items()):
                path: List[Move] = []
                node = cur
                while node != start:
                    prev, move = came_from[node]
                    path.append(move)
                    node = prev
                path.reverse()
                return path, cur, local_expanded

            for action in self.legal_actions(blank_idx, locked):
                if last_action is not None and action == OPPOSITE[last_action]:
                    continue
                nxt, nxt_blank = self.apply_action(cur, blank_idx, action)
                if nxt in closed:
                    continue
                ng = g + 1
                old_g = g_score.get(nxt)
                if old_g is not None and ng >= old_g:
                    continue
                g_score[nxt] = ng
                came_from[nxt] = (cur, action)
                h = self.heuristic_targets(nxt, tile_targets)
                push_id += 1
                heapq.heappush(open_heap, (ng + w * h, push_id, ng, nxt, nxt_blank, action))

        return None, board, local_expanded

    def solve(self, board: Sequence[int], timeout_sec: Optional[float] = None) -> Tuple[Optional[List[Move]], PlanStats]:
        start_time = time.perf_counter()
        self.expanded_nodes = 0
        cur = tuple(int(x) for x in board)

        if len(cur) != self.n_tiles:
            raise ValueError(f"Expected board length {self.n_tiles}, got {len(cur)}")
        if not self.is_solvable(cur):
            return None, PlanStats(False, 0, 0.0, 0, 0)
        if self.is_solved(cur):
            return [], PlanStats(True, 0, 0.0, 0, 0)

        locked: Set[int] = set()
        moves: List[Move] = []
        phases = 0
        top, left = 0, 0
        bottom, right = self.size - 1, self.size - 1

        def check_timeout() -> None:
            if timeout_sec is not None and (time.perf_counter() - start_time) > timeout_sec:
                raise TimeoutError("Strategic solver timed out.")

        try:
            while (bottom - top + 1) > 3 and (right - left + 1) > 3:
                # Solve current top row except the last pair.
                for col in range(left, right - 1):
                    check_timeout()
                    target_idx = self.pos_to_idx((top, col))
                    target_tile = self.goal[target_idx]
                    path, cur, _ = self.solve_subgoal(cur, {target_tile: target_idx}, locked)
                    if path is None:
                        raise RuntimeError(f"Failed to place tile {target_tile} at row phase ({top}, {col}).")
                    moves.extend(path)
                    locked.add(target_idx)
                    phases += 1

                # Solve last pair of the row together.
                check_timeout()
                pair_targets = {
                    self.goal[self.pos_to_idx((top, right - 1))]: self.pos_to_idx((top, right - 1)),
                    self.goal[self.pos_to_idx((top, right))]: self.pos_to_idx((top, right)),
                }
                path, cur, _ = self.solve_subgoal(cur, pair_targets, locked)
                if path is None:
                    raise RuntimeError(f"Failed to place top-row pair at row {top}.")
                moves.extend(path)
                locked.add(self.pos_to_idx((top, right - 1)))
                locked.add(self.pos_to_idx((top, right)))
                phases += 1
                top += 1

                # Solve current left column except the last pair.
                for row in range(top, bottom - 1):
                    check_timeout()
                    target_idx = self.pos_to_idx((row, left))
                    target_tile = self.goal[target_idx]
                    path, cur, _ = self.solve_subgoal(cur, {target_tile: target_idx}, locked)
                    if path is None:
                        raise RuntimeError(f"Failed to place tile {target_tile} at column phase ({row}, {left}).")
                    moves.extend(path)
                    locked.add(target_idx)
                    phases += 1

                # Solve bottom pair of the left column together.
                check_timeout()
                pair_targets = {
                    self.goal[self.pos_to_idx((bottom - 1, left))]: self.pos_to_idx((bottom - 1, left)),
                    self.goal[self.pos_to_idx((bottom, left))]: self.pos_to_idx((bottom, left)),
                }
                path, cur, _ = self.solve_subgoal(cur, pair_targets, locked)
                if path is None:
                    raise RuntimeError(f"Failed to place left-column pair at column {left}.")
                moves.extend(path)
                locked.add(self.pos_to_idx((bottom - 1, left)))
                locked.add(self.pos_to_idx((bottom, left)))
                phases += 1
                left += 1

            # Final exact search on remaining small subproblem.
            check_timeout()
            remaining_targets: Dict[int, int] = {}
            for idx, tile in enumerate(self.goal):
                if idx not in locked:
                    remaining_targets[tile] = idx
            # For the final phase we match all remaining non-blank tiles and let the blank end naturally.
            remaining_targets.pop(0, None)
            path, cur, _ = self.solve_subgoal(
                cur,
                remaining_targets,
                locked,
                max_expansions=self.final_max_expansions,
                weight=1.0,
            )
            if path is None or cur != self.goal:
                # Fallback: one last weighted search on the full remaining state, including blank.
                final_targets = {tile: idx for idx, tile in enumerate(self.goal) if idx not in locked}
                path2, cur2, _ = self.solve_subgoal(
                    cur,
                    final_targets,
                    locked,
                    max_expansions=self.final_max_expansions,
                    weight=1.0,
                )
                if path2 is None or cur2 != self.goal:
                    raise RuntimeError("Failed in final reduced-board solve.")
                path = path2
                cur = cur2
            moves.extend(path)
            phases += 1
        except TimeoutError:
            elapsed = time.perf_counter() - start_time
            return None, PlanStats(False, len(moves), elapsed, self.expanded_nodes, phases)

        elapsed = time.perf_counter() - start_time
        return moves, PlanStats(cur == self.goal, len(moves), elapsed, self.expanded_nodes, phases)


def replay_solution(env: NPuzzleEnv, solution: Iterable[Move]) -> bool:
    for action in solution:
        _, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return bool(info["solved"])
    return env.is_solved()


def benchmark(size: int, scramble_depth: int, episodes: int, seed: int, timeout_sec: float) -> None:
    if NPuzzleEnv is None:
        raise RuntimeError("npuzzle_env.py not available for benchmark mode.")
    env = NPuzzleEnv(size=size, max_steps=1_000_000, seed=seed)
    solver = StrategicNPuzzleSolver(size=size)
    solved = 0
    move_counts: List[int] = []
    times: List[float] = []
    expanded: List[int] = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(scramble_depth=scramble_depth)
        solution, stats = solver.solve(state, timeout_sec=timeout_sec)
        solved += int(solution is not None and stats.solved)
        move_counts.append(stats.moves)
        times.append(stats.elapsed_sec)
        expanded.append(stats.expanded_nodes)
        print(
            f"episode={episode:3d} solved={stats.solved} moves={stats.moves:5d} "
            f"expanded={stats.expanded_nodes:9d} phases={stats.phases:3d} "
            f"time={stats.elapsed_sec:8.3f}s"
        )

    print(
        f"benchmark solved={solved}/{episodes} avg_moves={np.mean(move_counts):.2f} "
        f"avg_expanded={np.mean(expanded):.1f} avg_time={np.mean(times):.3f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalable strategic N-puzzle solver for large boards.")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--board", type=int, nargs="+", default=None)
    parser.add_argument("--scramble-depth", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--astar-weight", type=float, default=1.25)
    parser.add_argument("--subgoal-max-expansions", type=int, default=250000)
    parser.add_argument("--final-max-expansions", type=int, default=1000000)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = StrategicNPuzzleSolver(
        size=args.size,
        astar_weight=args.astar_weight,
        subgoal_max_expansions=args.subgoal_max_expansions,
        final_max_expansions=args.final_max_expansions,
    )

    if args.benchmark:
        benchmark(args.size, args.scramble_depth, args.episodes, args.seed, args.timeout_sec)
        return

    if args.board is None:
        if NPuzzleEnv is None:
            raise RuntimeError("Provide --board or place npuzzle_env.py next to this script.")
        env = NPuzzleEnv(size=args.size, max_steps=1_000_000, seed=args.seed)
        board, _ = env.reset(scramble_depth=args.scramble_depth)
        board = tuple(int(x) for x in board)
    else:
        board = tuple(args.board)

    print("size:", args.size)
    print("board:", board)
    print("solvable:", solver.is_solvable(board))

    solution, stats = solver.solve(board=board, timeout_sec=args.timeout_sec)
    print("stats:", stats)
    if solution is None:
        print("solution: None")
        return
    print("solution length:", len(solution))
    print("solution prefix:", solution[:60], "..." if len(solution) > 60 else "")

    if args.verify:
        if NPuzzleEnv is None:
            raise RuntimeError("npuzzle_env.py not available for verify mode.")
        env = NPuzzleEnv(size=args.size, max_steps=max(1_000_000, len(solution) + 10), seed=args.seed)
        env.board = np.array(board, dtype=np.int64)
        env.steps_taken = 0
        ok = replay_solution(env, solution)
        print("verified:", ok)


if __name__ == "__main__":
    main()
