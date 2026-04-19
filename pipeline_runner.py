# pipeline_runner.py
import threading, queue

move_queue = queue.Queue(maxsize=50)

def producer(board, solver):
    moves = solver.solve(board)
    for m in moves:
        move_queue.put(m)
    move_queue.put(None)


def consumer(page, rect, size, board, delay=15):
    ACTION = [(-1,0),(1,0),(0,-1),(0,1)]

    board = list(board)
    w = rect["width"]/size
    h = rect["height"]/size

    coords = [
        (rect["x"]+(c+0.5)*w, rect["y"]+(r+0.5)*h)
        for r in range(size) for c in range(size)
    ]

    while True:
        act = move_queue.get()
        if act is None:
            break

        blank = board.index(0)
        r,c = divmod(blank,size)
        dr,dc = ACTION[act]
        nr,nc = r+dr,c+dc

        idx = nr*size+nc
        x,y = coords[idx]

        page.evaluate("(x,y)=>document.elementFromPoint(x,y).click()", x, y)

        board[blank], board[idx] = board[idx], board[blank]

        if delay>0:
            page.wait_for_timeout(delay)


def run(page, rect, size, board, solver):
    t = threading.Thread(target=producer, args=(board, solver))
    t.start()

    consumer(page, rect, size, board)

    t.join()