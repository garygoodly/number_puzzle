from __future__ import annotations

import argparse
import heapq
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

Board = Tuple[int, ...]
ClickTarget = int

ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # move blank up
    1: (1, 0),   # move blank down
    2: (0, -1),  # move blank left
    3: (0, 1),   # move blank right
}

OPPOSITE_ACTIONS: Dict[int, int] = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}


@dataclass
class SolveStats:
    solved: bool
    primitive_moves: int
    click_moves: int
    elapsed_seconds: float
    expanded_nodes: int
    phases_completed: int
    status: str


class ClickSpaceNPuzzleSolver:
    """
    N-puzzle solver optimized for low planning latency on large boards.

    Strategy:
    - solve the board layer by layer
    - use weighted A* on partial subgoals
    - search in click-space instead of primitive blank moves

    The solver is intentionally not move-optimal. It is designed to find a
    practical solution quickly for browser-based number puzzle games where one
    click can shift multiple tiles in a row or column.
    """

    def __init__(
        self,
        size: int,
        astar_weight: float = 2.2,
        subgoal_max_expansions: int = 1_000_000,
        final_max_expansions: int = 5_000_000,
    ) -> None:
        self.size = int(size)
        self.tile_count = self.size * self.size
        self.goal_state: Board = tuple(list(range(1, self.tile_count)) + [0])

        self.astar_weight = float(astar_weight)
        self.subgoal_max_expansions = int(subgoal_max_expansions)
        self.final_max_expansions = int(final_max_expansions)
        self.expanded_nodes = 0

    def pos_to_index(self, row: int, col: int) -> int:
        return row * self.size + col

    def index_to_pos(self, index: int) -> Tuple[int, int]:
        return divmod(index, self.size)

    def goal_tile_at(self, row: int, col: int) -> int:
        value = row * self.size + col + 1
        return 0 if value == self.tile_count else value

    def is_solved(self, board: Sequence[int]) -> bool:
        return tuple(board) == self.goal_state

    def is_solvable(self, board: Sequence[int]) -> bool:
        values = [value for value in board if value != 0]
        inversions = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i] > values[j]:
                    inversions += 1

        if self.size % 2 == 1:
            return inversions % 2 == 0

        blank_row_from_bottom = self.size - (list(board).index(0) // self.size)
        return (inversions + blank_row_from_bottom) % 2 == 1

    def manhattan_distance(self, index_a: int, index_b: int) -> int:
        row_a, col_a = divmod(index_a, self.size)
        row_b, col_b = divmod(index_b, self.size)
        return abs(row_a - row_b) + abs(col_a - col_b)

    def _available_click_targets(self, blank_index: int, blocked: Set[int]) -> List[int]:
        targets: List[int] = []
        blank_row, blank_col = divmod(blank_index, self.size)

        row = blank_row - 1
        while row >= 0:
            index = row * self.size + blank_col
            if index in blocked:
                break
            targets.append(index)
            row -= 1

        row = blank_row + 1
        while row < self.size:
            index = row * self.size + blank_col
            if index in blocked:
                break
            targets.append(index)
            row += 1

        col = blank_col - 1
        while col >= 0:
            index = blank_row * self.size + col
            if index in blocked:
                break
            targets.append(index)
            col -= 1

        col = blank_col + 1
        while col < self.size:
            index = blank_row * self.size + col
            if index in blocked:
                break
            targets.append(index)
            col += 1

        return targets

    def _click_to_primitive_actions(self, blank_index: int, target_index: int) -> List[int]:
        blank_row, blank_col = divmod(blank_index, self.size)
        target_row, target_col = divmod(target_index, self.size)

        if blank_row == target_row:
            if target_col > blank_col:
                return [3] * (target_col - blank_col)
            return [2] * (blank_col - target_col)

        if blank_col == target_col:
            if target_row > blank_row:
                return [1] * (target_row - blank_row)
            return [0] * (blank_row - target_row)

        raise ValueError("Target must share the same row or column as the blank.")

    def _apply_click(self, board: List[int], blank_index: int, target_index: int) -> int:
        if blank_index == target_index:
            return blank_index

        blank_row, blank_col = divmod(blank_index, self.size)
        target_row, target_col = divmod(target_index, self.size)

        if blank_row == target_row:
            if target_col > blank_col:
                for index in range(blank_index, target_index):
                    board[index] = board[index + 1]
            else:
                for index in range(blank_index, target_index, -1):
                    board[index] = board[index - 1]
            board[target_index] = 0
            return target_index

        if blank_col == target_col:
            step = self.size
            if target_row > blank_row:
                for index in range(blank_index, target_index, step):
                    board[index] = board[index + step]
            else:
                for index in range(blank_index, target_index, -step):
                    board[index] = board[index - step]
            board[target_index] = 0
            return target_index

        raise ValueError("Target must share the same row or column as the blank.")

    def _apply_click_sequence(
        self,
        board: List[int],
        blank_index: int,
        click_targets: Sequence[int],
    ) -> int:
        for target_index in click_targets:
            blank_index = self._apply_click(board, blank_index, target_index)
        return blank_index

    def click_targets_to_primitive_moves(
        self,
        board: Sequence[int],
        click_targets: Sequence[int],
    ) -> List[int]:
        blank_index = list(board).index(0)
        primitive_moves: List[int] = []
        for target_index in click_targets:
            primitive_moves.extend(self._click_to_primitive_actions(blank_index, target_index))
            blank_index = target_index
        return primitive_moves

    def _update_tracked_positions_for_click(
        self,
        blank_index: int,
        target_index: int,
        tracked_positions: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        next_positions = list(tracked_positions)

        blank_row, blank_col = divmod(blank_index, self.size)
        target_row, target_col = divmod(target_index, self.size)

        if blank_row == target_row:
            if target_col > blank_col:
                for i, pos in enumerate(next_positions):
                    if pos // self.size == blank_row and blank_index < pos <= target_index:
                        next_positions[i] = pos - 1
            else:
                for i, pos in enumerate(next_positions):
                    if pos // self.size == blank_row and target_index <= pos < blank_index:
                        next_positions[i] = pos + 1
        else:
            step = self.size
            if target_row > blank_row:
                for i, pos in enumerate(next_positions):
                    if pos % self.size == blank_col and blank_index < pos <= target_index:
                        next_positions[i] = pos - step
            else:
                for i, pos in enumerate(next_positions):
                    if pos % self.size == blank_col and target_index <= pos < blank_index:
                        next_positions[i] = pos + step

        return tuple(next_positions)

    def _search_partial_solution(
        self,
        board: Sequence[int],
        tracked_tiles: Sequence[int],
        goal_indices: Sequence[int],
        blocked_indices: Set[int],
        max_expansions: int,
        deadline: float,
    ) -> Optional[List[int]]:
        tracked_tiles = list(tracked_tiles)
        positions = {tile: index for index, tile in enumerate(board)}

        start_blank = positions[0]
        start_tokens = tuple(positions[tile] for tile in tracked_tiles)
        goal_tokens = tuple(goal_indices)

        if start_tokens == goal_tokens:
            return []

        usable_indices = {index for index in range(self.tile_count) if index not in blocked_indices}
        if start_blank not in usable_indices:
            return None
        if any(pos not in usable_indices for pos in start_tokens):
            return None
        if any(pos not in usable_indices for pos in goal_tokens):
            return None

        def heuristic(state: Tuple[int, ...]) -> int:
            token_positions = state[1:]
            return sum(
                self.manhattan_distance(position, goal)
                for position, goal in zip(token_positions, goal_tokens)
            )

        start_state = (start_blank,) + start_tokens
        best_cost: Dict[Tuple[int, ...], int] = {start_state: 0}
        parent: Dict[Tuple[int, ...], Tuple[Optional[Tuple[int, ...]], Optional[int]]] = {
            start_state: (None, None)
        }

        priority_queue: List[Tuple[float, int, int, Tuple[int, ...]]] = []
        push_id = 0
        start_heuristic = heuristic(start_state)
        heapq.heappush(
            priority_queue,
            (self.astar_weight * start_heuristic, 0, push_id, start_state),
        )

        local_expanded = 0

        while priority_queue:
            if time.perf_counter() > deadline:
                self.expanded_nodes += local_expanded
                return None

            _, cost_so_far, _, state = heapq.heappop(priority_queue)
            if cost_so_far != best_cost.get(state):
                continue

            blank_index = state[0]
            tracked_positions = state[1:]

            if tracked_positions == goal_tokens:
                self.expanded_nodes += local_expanded
                click_targets: List[int] = []
                current_state = state
                while parent[current_state][0] is not None:
                    previous_state, click_index = parent[current_state]
                    assert previous_state is not None
                    assert click_index is not None
                    click_targets.append(click_index)
                    current_state = previous_state
                click_targets.reverse()
                return click_targets

            local_expanded += 1
            if local_expanded > max_expansions:
                self.expanded_nodes += local_expanded
                return None

            for click_target in self._available_click_targets(blank_index, blocked_indices):
                next_blank = click_target
                next_tokens = self._update_tracked_positions_for_click(
                    blank_index=blank_index,
                    target_index=click_target,
                    tracked_positions=tracked_positions,
                )
                next_state = (next_blank,) + next_tokens
                next_cost = cost_so_far + 1

                if next_cost < best_cost.get(next_state, 10**18):
                    best_cost[next_state] = next_cost
                    parent[next_state] = (state, click_target)
                    push_id += 1
                    next_priority = next_cost + self.astar_weight * heuristic(next_state)
                    heapq.heappush(
                        priority_queue,
                        (next_priority, next_cost, push_id, next_state),
                    )

        self.expanded_nodes += local_expanded
        return None

    def _solve_final_subgrid(
        self,
        board: Sequence[int],
        start_layer: int,
        deadline: float,
    ) -> Optional[List[int]]:
        tracked_tiles: List[int] = []
        goal_indices: List[int] = []
        blocked = set(range(self.tile_count))

        for row in range(start_layer, self.size):
            for col in range(start_layer, self.size):
                index = row * self.size + col
                blocked.discard(index)
                tile = self.goal_tile_at(row, col)
                if tile != 0:
                    tracked_tiles.append(tile)
                    goal_indices.append(index)

        return self._search_partial_solution(
            board=board,
            tracked_tiles=tracked_tiles,
            goal_indices=goal_indices,
            blocked_indices=blocked,
            max_expansions=self.final_max_expansions,
            deadline=deadline,
        )

    def solve_clicks(
        self,
        board: Sequence[int],
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[Optional[List[int]], SolveStats]:
        start_board = tuple(int(value) for value in board)

        if len(start_board) != self.tile_count:
            raise ValueError(
                f"Expected board length {self.tile_count}, got {len(start_board)}."
            )

        if not self.is_solvable(start_board):
            stats = SolveStats(
                solved=False,
                primitive_moves=0,
                click_moves=0,
                elapsed_seconds=0.0,
                expanded_nodes=0,
                phases_completed=0,
                status="unsolvable",
            )
            return None, stats

        self.expanded_nodes = 0
        start_time = time.perf_counter()
        deadline = float("inf") if timeout_seconds is None else start_time + float(timeout_seconds)

        current_board = list(start_board)
        blank_index = current_board.index(0)
        frozen_indices: Set[int] = set()
        click_plan: List[int] = []
        phases_completed = 0

        for layer in range(0, self.size - 3):
            top = layer
            left = layer

            for col in range(left, self.size - 2):
                goal_index = top * self.size + col
                target_tile = self.goal_tile_at(top, col)

                if current_board[goal_index] == target_tile:
                    frozen_indices.add(goal_index)
                    continue

                local_clicks = self._search_partial_solution(
                    board=current_board,
                    tracked_tiles=[target_tile],
                    goal_indices=[goal_index],
                    blocked_indices=frozen_indices,
                    max_expansions=self.subgoal_max_expansions,
                    deadline=deadline,
                )
                if local_clicks is None:
                    elapsed = time.perf_counter() - start_time
                    stats = SolveStats(
                        solved=False,
                        primitive_moves=0,
                        click_moves=len(click_plan),
                        elapsed_seconds=elapsed,
                        expanded_nodes=self.expanded_nodes,
                        phases_completed=phases_completed,
                        status="top_row_single_failed",
                    )
                    return None, stats

                blank_index = self._apply_click_sequence(current_board, blank_index, local_clicks)
                click_plan.extend(local_clicks)
                frozen_indices.add(goal_index)
                phases_completed += 1

            goal_index_a = top * self.size + (self.size - 2)
            goal_index_b = top * self.size + (self.size - 1)
            tile_a = self.goal_tile_at(top, self.size - 2)
            tile_b = self.goal_tile_at(top, self.size - 1)

            if not (
                current_board[goal_index_a] == tile_a and current_board[goal_index_b] == tile_b
            ):
                local_clicks = self._search_partial_solution(
                    board=current_board,
                    tracked_tiles=[tile_a, tile_b],
                    goal_indices=[goal_index_a, goal_index_b],
                    blocked_indices=frozen_indices,
                    max_expansions=self.subgoal_max_expansions,
                    deadline=deadline,
                )
                if local_clicks is None:
                    elapsed = time.perf_counter() - start_time
                    stats = SolveStats(
                        solved=False,
                        primitive_moves=0,
                        click_moves=len(click_plan),
                        elapsed_seconds=elapsed,
                        expanded_nodes=self.expanded_nodes,
                        phases_completed=phases_completed,
                        status="top_row_pair_failed",
                    )
                    return None, stats

                blank_index = self._apply_click_sequence(current_board, blank_index, local_clicks)
                click_plan.extend(local_clicks)
                phases_completed += 1

            frozen_indices.add(goal_index_a)
            frozen_indices.add(goal_index_b)

            for row in range(top + 1, self.size - 2):
                goal_index = row * self.size + left
                target_tile = self.goal_tile_at(row, left)

                if current_board[goal_index] == target_tile:
                    frozen_indices.add(goal_index)
                    continue

                local_clicks = self._search_partial_solution(
                    board=current_board,
                    tracked_tiles=[target_tile],
                    goal_indices=[goal_index],
                    blocked_indices=frozen_indices,
                    max_expansions=self.subgoal_max_expansions,
                    deadline=deadline,
                )
                if local_clicks is None:
                    elapsed = time.perf_counter() - start_time
                    stats = SolveStats(
                        solved=False,
                        primitive_moves=0,
                        click_moves=len(click_plan),
                        elapsed_seconds=elapsed,
                        expanded_nodes=self.expanded_nodes,
                        phases_completed=phases_completed,
                        status="left_column_single_failed",
                    )
                    return None, stats

                blank_index = self._apply_click_sequence(current_board, blank_index, local_clicks)
                click_plan.extend(local_clicks)
                frozen_indices.add(goal_index)
                phases_completed += 1

            goal_index_a = (self.size - 2) * self.size + left
            goal_index_b = (self.size - 1) * self.size + left
            tile_a = self.goal_tile_at(self.size - 2, left)
            tile_b = self.goal_tile_at(self.size - 1, left)

            if not (
                current_board[goal_index_a] == tile_a and current_board[goal_index_b] == tile_b
            ):
                local_clicks = self._search_partial_solution(
                    board=current_board,
                    tracked_tiles=[tile_a, tile_b],
                    goal_indices=[goal_index_a, goal_index_b],
                    blocked_indices=frozen_indices,
                    max_expansions=self.subgoal_max_expansions,
                    deadline=deadline,
                )
                if local_clicks is None:
                    elapsed = time.perf_counter() - start_time
                    stats = SolveStats(
                        solved=False,
                        primitive_moves=0,
                        click_moves=len(click_plan),
                        elapsed_seconds=elapsed,
                        expanded_nodes=self.expanded_nodes,
                        phases_completed=phases_completed,
                        status="left_column_pair_failed",
                    )
                    return None, stats

                blank_index = self._apply_click_sequence(current_board, blank_index, local_clicks)
                click_plan.extend(local_clicks)
                phases_completed += 1

            frozen_indices.add(goal_index_a)
            frozen_indices.add(goal_index_b)

        if tuple(current_board) != self.goal_state:
            final_clicks = self._solve_final_subgrid(
                board=current_board,
                start_layer=self.size - 3,
                deadline=deadline,
            )
            if final_clicks is None:
                elapsed = time.perf_counter() - start_time
                stats = SolveStats(
                    solved=False,
                    primitive_moves=0,
                    click_moves=len(click_plan),
                    elapsed_seconds=elapsed,
                    expanded_nodes=self.expanded_nodes,
                    phases_completed=phases_completed,
                    status="final_subgrid_failed",
                )
                return None, stats

            blank_index = self._apply_click_sequence(current_board, blank_index, final_clicks)
            click_plan.extend(final_clicks)
            phases_completed += 1

        solved = tuple(current_board) == self.goal_state
        primitive_moves = self.click_targets_to_primitive_moves(start_board, click_plan)
        elapsed = time.perf_counter() - start_time

        stats = SolveStats(
            solved=solved,
            primitive_moves=len(primitive_moves),
            click_moves=len(click_plan),
            elapsed_seconds=elapsed,
            expanded_nodes=self.expanded_nodes,
            phases_completed=phases_completed,
            status="solved" if solved else "ended_unsolved",
        )
        return click_plan if solved else None, stats

    def solve(
        self,
        board: Sequence[int],
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[Optional[List[int]], SolveStats]:
        click_targets, stats = self.solve_clicks(board=board, timeout_seconds=timeout_seconds)
        if click_targets is None:
            return None, stats
        primitive_moves = self.click_targets_to_primitive_moves(board, click_targets)
        return primitive_moves, stats


def scramble_board(size: int, depth: int, seed: int) -> Board:
    random.seed(seed)
    board = list(range(1, size * size)) + [0]
    previous_action: Optional[int] = None

    for _ in range(depth):
        blank_index = board.index(0)
        blank_row, blank_col = divmod(blank_index, size)
        candidate_actions: List[int] = []

        for action, (delta_row, delta_col) in ACTION_DELTAS.items():
            next_row = blank_row + delta_row
            next_col = blank_col + delta_col
            if 0 <= next_row < size and 0 <= next_col < size:
                if previous_action is not None and action == OPPOSITE_ACTIONS[previous_action]:
                    continue
                candidate_actions.append(action)

        action = random.choice(candidate_actions)
        delta_row, delta_col = ACTION_DELTAS[action]
        swap_index = (blank_row + delta_row) * size + (blank_col + delta_col)
        board[blank_index], board[swap_index] = board[swap_index], board[blank_index]
        previous_action = action

    return tuple(board)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone N-puzzle solver.")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--scramble-depth", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--astar-weight", type=float, default=2.2)
    parser.add_argument("--subgoal-max-expansions", type=int, default=1_000_000)
    parser.add_argument("--final-max-expansions", type=int, default=5_000_000)
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = ClickSpaceNPuzzleSolver(
        size=args.size,
        astar_weight=args.astar_weight,
        subgoal_max_expansions=args.subgoal_max_expansions,
        final_max_expansions=args.final_max_expansions,
    )

    for episode in range(args.episodes):
        board = scramble_board(args.size, args.scramble_depth, args.seed + episode)
        click_targets, stats = solver.solve_clicks(board, timeout_seconds=args.timeout_seconds)
        print("board:", board)
        print("stats:", stats)
        print("click plan length:", len(click_targets) if click_targets is not None else None)


if __name__ == "__main__":
    main()
