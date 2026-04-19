# fast_ida_pdb.py
ACTION = [(-1,0),(1,0),(0,-1),(0,1)]
OPP = [1,0,3,2]

class FastIDA_PDB:
    def __init__(self, size, pdbs):
        self.size = size
        self.pdbs = pdbs

    def h(self, board):
        return sum(pdb.h(board) for pdb in self.pdbs)

    def solve(self, start):
        board = list(start)
        blank = board.index(0)
        path = []

        bound = self.h(board)

        def dfs(g, bound, blank, prev):
            f = g + self.h(board)
            if f > bound:
                return f

            if f == g:
                return True

            min_next = 1e9
            r, c = divmod(blank, self.size)

            for act,(dr,dc) in enumerate(ACTION):
                if prev != -1 and act == OPP[prev]:
                    continue

                nr,nc = r+dr,c+dc
                if 0<=nr<self.size and 0<=nc<self.size:
                    nxt = nr*self.size+nc

                    board[blank], board[nxt] = board[nxt], board[blank]
                    path.append(act)

                    res = dfs(g+1, bound, nxt, act)

                    if res is True:
                        return True

                    if res < min_next:
                        min_next = res

                    path.pop()
                    board[blank], board[nxt] = board[nxt], board[blank]

            return min_next

        while True:
            res = dfs(0, bound, blank, -1)
            if res is True:
                return path
            bound = res