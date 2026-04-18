from collections import deque
import time

# 總步數計數器
step_counter = 0

def print_board(board, step_name=""):
    print(f"=== {step_name} ===")
    for r in range(4):
        row_str = [str(board[r*4 + c]).rjust(2, ' ') for c in range(4)]
        print("[" + ", ".join(row_str) + "]")
    print()
    time.sleep(0.1) # 稍微暫停一下，讓印出有動畫感

def apply_swap(board, idx1, idx2, reason=""):
    """所有方塊的移動都必須經過這個函數，絕不偷偷來"""
    global step_counter
    step_counter += 1
    board[idx1], board[idx2] = board[idx2], board[idx1]
    print_board(board, f"第 {step_counter} 步: {reason}")

def get_neighbors(idx):
    """取得上下左右的鄰居索引"""
    r, c = idx // 4, idx % 4
    ns = []
    if r > 0: ns.append(idx - 4) # 上
    if r < 3: ns.append(idx + 4) # 下
    if c > 0: ns.append(idx - 1) # 左
    if c < 3: ns.append(idx + 1) # 右
    return ns

def route_blank(board, target_idx, locked):
    """讓空格找出避開 locked 障礙物的最短路徑"""
    start = board.index(0)
    if start == target_idx: return []
    
    q = deque([(start, [])])
    visited = {start}
    while q:
        curr, path = q.popleft()
        if curr == target_idx:
            return path
        for nxt in get_neighbors(curr):
            if nxt not in locked and nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
    raise RuntimeError(f"空格被困住了！無法到達 {target_idx}")

def push_tile(board, tile_val, target_idx, locked):
    """無腦推車：把方塊一步步推到目標"""
    while True:
        tile_idx = board.index(tile_val)
        if tile_idx == target_idx:
            break
        
        # 1. 找出方塊前往目標的下一步
        q = deque([(tile_idx, [])])
        visited = {tile_idx}
        next_step = None
        while q:
            curr, path = q.popleft()
            if curr == target_idx:
                next_step = path[0]
                break
            for nxt in get_neighbors(curr):
                if nxt not in locked and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))
                    
        # 2. 把空格叫到方塊的「下一步」位置 (空格不能穿過方塊本身)
        locked_for_blank = locked.copy()
        locked_for_blank.add(tile_idx)
        blank_path = route_blank(board, next_step, locked_for_blank)
        
        # 執行空格移動 (步步印出)
        for step in blank_path:
            b_idx = board.index(0)
            moved_val = board[step]
            apply_swap(board, b_idx, step, f"空格為了推 {tile_val}，先移動到數字 {moved_val} 的位置")
            
        # 3. 空格到位後，與方塊交換位置 (也就是正式推動方塊，步步印出)
        b_idx = board.index(0)
        apply_swap(board, b_idx, tile_idx, f"🔥 正式推動！空格與目標方塊 {tile_val} 交換")

def solve_first_row(board):
    locked = set()
    print_board(board, "初始盤面 (Start)")
    
    # 1. 解決 1
    print("\n>>> 開始處理方塊 1 <<<")
    push_tile(board, 1, 0, locked)
    locked.add(0)

    # 2. 解決 2
    print("\n>>> 開始處理方塊 2 <<<")
    push_tile(board, 2, 1, locked)
    locked.add(1)

    # 3. 處理 3 和 4 (WikiHow 倒序佈局法)
    print("\n>>> 開始佈局方塊 4 (停在 idx=2) <<<")
    push_tile(board, 4, 2, locked)
    locked.add(2) # 暫時鎖定 4

    print("\n>>> 開始佈局方塊 3 (停在 idx=6, 也就是 4 的正下方) <<<")
    push_tile(board, 3, 6, locked)
    locked.add(6) # 暫時鎖定 3

    # 4. 把空格叫到角落 (idx=3)
    print("\n>>> 將空格叫到死角發射區 (idx=3) <<<")
    blank_path = route_blank(board, 3, locked)
    for step in blank_path:
        b_idx = board.index(0)
        moved_val = board[step]
        apply_swap(board, b_idx, step, f"佈局收尾：空格走向死角，路過數字 {moved_val}")

    # 5. 執行旋轉公式 (空格往左、再往下)
    print("\n>>> 執行最終旋轉公式 <<<")
    locked.remove(2)
    locked.remove(6)
    
    # 空格往左 (idx=2)
    b_idx = board.index(0)
    apply_swap(board, b_idx, 2, "旋轉公式第一步：空格往左，將 4 擠入右上角")
    
    # 空格往下 (idx=6)
    b_idx = board.index(0)
    apply_swap(board, b_idx, 6, "旋轉公式第二步：空格往下，將 3 擠入歸位")
    
    locked.add(2)
    locked.add(3)
    print("\n✨ 第一排完美歸位，請檢查以上所有步驟是否有作弊！")

# --- 測試區 ---
if __name__ == "__main__":
    # 使用你提供的盤面
    test_board = [11, 12, 15, 8, 4, 2, 7, 14, 0, 1, 6, 10, 5, 13, 9, 3]
    solve_first_row(test_board)