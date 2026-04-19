# pdb.py
from collections import deque

class PatternDatabase:
    def __init__(self, size, pattern):
        self.size = size
        self.pattern = set(pattern)
        self.goal = tuple(list(range(1, size*size)) + [0])
        self.db = {}

    def compress(self, board):
        return tuple(v if v in self.pattern else -1 for v in board)

    def build(self):
        start = self.compress(self.goal)
        queue = deque([(start, 0)])
        self.db[start] = 0

        while queue:
            state, dist = queue.popleft()

            blank = state.index(0)
            r, c = divmod(blank, self.size)

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.size and 0<=nc<self.size:
                    nxt = nr*self.size+nc

                    new_state = list(state)
                    new_state[blank], new_state[nxt] = new_state[nxt], new_state[blank]
                    new_state = tuple(new_state)

                    if new_state not in self.db:
                        self.db[new_state] = dist+1
                        queue.append((new_state, dist+1))

    def h(self, board):
        key = self.compress(board)
        return self.db.get(key, 0)