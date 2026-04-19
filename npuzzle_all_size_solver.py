from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import heapq

@dataclass
class PlanStats:
    solved: bool
    moves: int
    elapsed_sec: float
    expanded_nodes: int
    phases: int

# 空格移動方向 (0:上, 1:下, 2:左, 3:右)
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
# 互斥方向對照表 (用來消除廢步)
OPPOSITE_ACTIONS = {0: 1, 1: 0, 2: 3, 3: 2}

class StrategicNPuzzleSolver:
    def __init__(self, size: int, **kwargs) -> None:
        self.size = int(size)
        self.n_tiles = self.size * self.size
        self.goal = tuple(list(range(1, self.n_tiles)) + [0])
        self.expanded_nodes = 0

    def pos_to_idx(self, pos): return pos[0] * self.size + pos[1]
    def idx_to_pos(self, idx): return divmod(idx, self.size)
    def is_solvable(self, board): return True

    def _apply_path(self, board, moves, path):
        for act in path:
            b_i = board.index(0)
            br, bc = self.idx_to_pos(b_i)
            dr, dc = ACTION_DELTAS[act]
            s_i = self.pos_to_idx((br+dr, bc+dc))
            board[b_i], board[s_i] = board[s_i], board[b_i]
            moves.append(act)

    def _route_blank(self, board, target_idx, locked):
        start = board.index(0)
        if start == target_idx: return []
        q = deque([(start, [])])
        visited = {start}
        while q:
            curr, path = q.popleft()
            self.expanded_nodes += 1
            if curr == target_idx: return path
            r, c = self.idx_to_pos(curr)
            for act, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    nxt = self.pos_to_idx((nr, nc))
                    if nxt not in locked and nxt not in visited:
                        visited.add(nxt)
                        q.append((nxt, path + [act]))
        return None



    def _push_tile_joint(self, board, moves, tile_val, target_idx, locked):
        start_tile = board.index(tile_val)
        start_blank = board.index(0)
        if start_tile == target_idx:
            return

        def manhattan(idx1, idx2):
            r1, c1 = divmod(idx1, self.size)
            r2, c2 = divmod(idx2, self.size)
            return abs(r1 - r2) + abs(c1 - c2)

        # heuristic：tile 到目標距離 + (小權重) blank 靠近 tile
        def h(tile, blank):
            return manhattan(tile, target_idx) + 0.3 * manhattan(blank, tile)

        # A*
        heap = []
        # (f, g, tile_pos, blank_pos)
        start_h = h(start_tile, start_blank)
        heapq.heappush(heap, (start_h, 0, start_tile, start_blank))

        visited = {(start_tile, start_blank): (None, None)}  # parent, action

        found_state = None

        while heap:
            f, g, tile, blank = heapq.heappop(heap)
            self.expanded_nodes += 1

            if tile == target_idx:
                found_state = (tile, blank)
                break

            br, bc = divmod(blank, self.size)

            for act, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = br + dr, bc + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    nxt_blank = nr * self.size + nc
                    if nxt_blank in locked:
                        continue

                    # tile 會被 blank 推動
                    nxt_tile = blank if nxt_blank == tile else tile
                    state = (nxt_tile, nxt_blank)

                    if state in visited:
                        continue

                    visited[state] = ((tile, blank), act)
                    new_g = g + 1
                    new_f = new_g + h(nxt_tile, nxt_blank)
                    heapq.heappush(heap, (new_f, new_g, nxt_tile, nxt_blank))

        if found_state is None:
            raise RuntimeError(f"發生死結：無法將方塊 {tile_val} 推至目標位置。")

        # reconstruct path
        path = []
        curr = found_state
        while visited[curr][0] is not None:
            prev, act = visited[curr]
            path.append(act)
            curr = prev
        path.reverse()

        self._apply_path(board, moves, path)

    def _optimize_path(self, moves: List[int]) -> List[int]:
        """🌟 廢步消除器：用 Stack 抵銷相鄰的反方向移動 (例如：上+下、左+右)"""
        optimized = []
        for act in moves:
            if optimized and optimized[-1] == OPPOSITE_ACTIONS[act]:
                optimized.pop() # 如果跟上一步相反，互相抵銷！
            else:
                optimized.append(act)
        return optimized

    def solve(self, board: Sequence[int], timeout_sec: Optional[float] = None) -> Tuple[Optional[List[Move]], PlanStats]:
        start_time = time.perf_counter()
        self.expanded_nodes = 0
        board = list(board)
        locked = set()
        moves = []
        phases = 0

        try:
            # ==========================================
            # 階段 1: 橫向排列
            # ==========================================
            for r in range(self.size - 2):
                for c in range(self.size - 2):
                    val = self.goal[self.pos_to_idx((r, c))]
                    target_idx = self.pos_to_idx((r, c))
                    self._push_tile_joint(board, moves, val, target_idx, locked)
                    locked.add(target_idx)

                val_A = self.goal[self.pos_to_idx((r, self.size - 2))] 
                val_B = self.goal[self.pos_to_idx((r, self.size - 1))] 
                
                corner       = self.pos_to_idx((r, self.size - 1))       
                below_corner = self.pos_to_idx((r + 1, self.size - 1))   
                launch_pad   = self.pos_to_idx((r, self.size - 2))       
                
                quarantine_pos = self.pos_to_idx((r + 2, self.size - 1)) 
                self._push_tile_joint(board, moves, val_B, quarantine_pos, locked)

                self._push_tile_joint(board, moves, val_A, corner, locked)
                locked.add(corner) 
                
                self._push_tile_joint(board, moves, val_B, below_corner, locked)
                locked.add(below_corner) 

                temp_locked = locked.copy()
                blank_path = self._route_blank(board, launch_pad, temp_locked)
                self._apply_path(board, moves, blank_path)

                locked.remove(corner)
                locked.remove(below_corner)
                self._apply_path(board, moves, [3, 1])

                locked.add(self.pos_to_idx((r, self.size - 2)))
                locked.add(corner)
                phases += 1

            # ==========================================
            # 階段 2: 直向排列 (最後兩排)
            # ==========================================
            for c in range(self.size - 2):
                val_A = self.goal[self.pos_to_idx((self.size - 2, c))] 
                val_B = self.goal[self.pos_to_idx((self.size - 1, c))] 
                
                corner       = self.pos_to_idx((self.size - 1, c))       
                right_corner = self.pos_to_idx((self.size - 1, c + 1))   
                launch_pad   = self.pos_to_idx((self.size - 2, c))       
                
                quarantine_pos = self.pos_to_idx((self.size - 1, c + 2)) 
                self._push_tile_joint(board, moves, val_B, quarantine_pos, locked)

                self._push_tile_joint(board, moves, val_A, corner, locked)
                locked.add(corner) 
                
                self._push_tile_joint(board, moves, val_B, right_corner, locked)
                locked.add(right_corner) 

                temp_locked = locked.copy()
                blank_path = self._route_blank(board, launch_pad, temp_locked)
                self._apply_path(board, moves, blank_path)

                locked.remove(corner)
                locked.remove(right_corner)
                self._apply_path(board, moves, [1, 3])

                locked.add(self.pos_to_idx((self.size - 2, c)))
                locked.add(corner)
                phases += 1

            # ==========================================
            # 階段 3: 最後 2x2 決戰區
            # ==========================================
            target_dict = {self.goal[i]: i for i in range(self.n_tiles) if i not in locked and self.goal[i] != 0}
            queue = deque([(tuple(board), [])])
            visited = {tuple(board)}
            final_moves = []
            
            while queue:
                curr_b, p = queue.popleft()
                if all(curr_b[idx] == val for val, idx in target_dict.items()):
                    final_moves = p
                    break
                
                b_idx = curr_b.index(0)
                br, bc = self.idx_to_pos(b_idx)
                for act, (dr, dc) in ACTION_DELTAS.items():
                    nr, nc = br + dr, bc + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        s_idx = self.pos_to_idx((nr, nc))
                        if s_idx not in locked:
                            new_b = list(curr_b)
                            new_b[b_idx], new_b[s_idx] = new_b[s_idx], new_b[b_idx]
                            new_tup = tuple(new_b)
                            if new_tup not in visited:
                                visited.add(new_tup)
                                queue.append((new_tup, p + [act]))

            self._apply_path(board, moves, final_moves)

            # 🌟 回傳前，套用廢步消除器
            optimized_moves = self._optimize_path(moves)
            elapsed = time.perf_counter() - start_time
            return optimized_moves, PlanStats(True, len(optimized_moves), elapsed, self.expanded_nodes, phases)

        except Exception as e:
            print(f"\n🚨 [觸發觀察模式] 演算法在途中卡關了！")
            print(f"🚨 錯誤訊息: {e}")
            import traceback
            traceback.print_exc()
            optimized_moves = self._optimize_path(moves)
            elapsed = time.perf_counter() - start_time
            return optimized_moves, PlanStats(False, len(optimized_moves), elapsed, self.expanded_nodes, phases)

    