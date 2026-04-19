from __future__ import annotations

import argparse
import math
import time
from typing import Sequence, Tuple

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from npuzzle_all_size_solver import StrategicNPuzzleSolver

def click_first_visible(page, selectors: Sequence[str], timeout_ms: int = 3000) -> bool:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout_ms)
            locator.click(timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def select_size(page, size: int) -> bool:
    labels = [f"{size} x {size}", f"{size}x{size}", f"{size} X {size}"]
    selectors = []
    for label in labels:
        selectors.extend(
            [
                f"button:has-text('{label}')",
                f".btn:has-text('{label}')",
                f"text={label}",
            ]
        )
    return click_first_visible(page, selectors, timeout_ms=4000)


def click_start_after_size(page) -> bool:
    selectors = [
        "#modal-action-btn",
        "button#modal-action-btn",
        "button:has-text('開始挑戰')",
        "button:has-text('开始挑战')",
        "text=開始挑戰",
        "text=开始挑战",
    ]
    return click_first_visible(page, selectors, timeout_ms=5000)


def wait_for_board_ready(page, timeout_ms: int = 10000) -> dict:
    board = page.locator("#board")
    board.wait_for(state="visible", timeout=timeout_ms)
    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_box = None

    while time.perf_counter() < deadline:
        box = board.bounding_box()
        if box and box["width"] > 40 and box["height"] > 40:
            return box
        last_box = box
        page.wait_for_timeout(100)

    raise RuntimeError(f"Could not get usable #board bounding box. Last box: {last_box}")


def read_board(page, size: int) -> Tuple[Tuple[int, ...], dict]:
    rect = wait_for_board_ready(page)
    items = page.locator("#board *").evaluate_all(
        """
        (elements) => elements.map((el) => {
            const r = el.getBoundingClientRect();
            const text = (el.innerText || el.textContent || '').trim();
            return {text, x: r.x, y: r.y, width: r.width, height: r.height};
        })
        """
    )

    cell_w = rect["width"] / size
    cell_h = rect["height"] / size
    board = [0] * (size * size)

    for item in items:
        text = item["text"]
        if not text.isdigit():
            continue
        value = int(text)
        if value <= 0 or value >= size * size:
            continue

        center_x = item["x"] + item["width"] / 2.0
        center_y = item["y"] + item["height"] / 2.0
        col = int((center_x - rect["x"]) / cell_w)
        row = int((center_y - rect["y"]) / cell_h)

        if 0 <= row < size and 0 <= col < size:
            board[row * size + col] = value

    return tuple(board), rect


def click_cell(page, rect: dict, size: int, row: int, col: int) -> None:
    cell_w = rect["width"] / size
    cell_h = rect["height"] / size
    x = rect["x"] + (col + 0.5) * cell_w
    y = rect["y"] + (row + 0.5) * cell_h
    page.mouse.click(x, y)


def apply_solution_in_browser(
    page,
    size: int,
    initial_board: Sequence[int],
    solution: Sequence[int],
    move_delay_ms: int,
) -> None:
    board = list(initial_board)
    rect = wait_for_board_ready(page)

    action_to_delta = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    for step_idx, action in enumerate(solution, start=1):
        blank_idx = board.index(0)
        blank_row, blank_col = divmod(blank_idx, size)
        dr, dc = action_to_delta[int(action)]
        tile_row = blank_row + dr
        tile_col = blank_col + dc

        if not (0 <= tile_row < size and 0 <= tile_col < size):
            raise RuntimeError(
                f"Illegal browser move at step {step_idx}: "
                f"action={action}, blank=({blank_row},{blank_col})"
            )

        tile_idx = tile_row * size + tile_col
        click_cell(page, rect, size, tile_row, tile_col)
        board[blank_idx], board[tile_idx] = board[tile_idx], board[blank_idx]

        if move_delay_ms > 0:
            page.wait_for_timeout(move_delay_ms)

def submit_name(page, name: str) -> None:
    print(f"\n🏆 準備霸榜！等待 HTML 通關畫面跳出，準備刻上大名：【{name}】")
    try:
        # 1. 稍微等一下動畫播完
        page.wait_for_timeout(2000) 
        
        # 2. 抓取畫面上最後一個可見的輸入框
        input_locator = page.locator("input:visible").last
        input_locator.wait_for(state="visible", timeout=5000)
        
        # 3. 點擊獲取焦點
        input_locator.click()
        page.wait_for_timeout(200)
        
        # 4. 全選並刪除 (強制清掉預設的 "匿名玩家")
        input_locator.press("Control+A")
        input_locator.press("Backspace")
        page.wait_for_timeout(200)
        
        # 5. 模擬真人逐字打字！這樣絕對能觸發網頁框架的更新事件
        input_locator.press_sequentially(name, delay=100)
        page.wait_for_timeout(500)
        
        # 6. 按下 Enter
        input_locator.press("Enter")
        page.wait_for_timeout(500)
        
        # 7. 防呆：如果有確認按鈕就點擊
        button_texts = ["確認", "確定", "送出", "Submit", "OK", "保存"]
        selectors = [f"button:has-text('{txt}'):visible" for txt in button_texts] + [f".btn:has-text('{txt}'):visible" for txt in button_texts]
        for sel in selectors:
            try:
                btn = page.locator(sel).first
                if btn.is_visible():
                    btn.click()
                    break
            except:
                pass
                
        print("✅ HTML 大名輸入流程執行完畢！")
        page.wait_for_timeout(2000) # 讓你有時間肉眼確認
        
    except Exception as e:
        # 如果找不到 HTML 輸入框，有可能是被下面的原生攔截器處理掉了，不報錯
        pass

def auto_timeout_sec(size: int, scale: float) -> float:
    if size <= 3:
        base = 20.0
    elif size == 4:
        base = 60.0
    elif size == 5:
        base = 120.0
    elif size == 6:
        base = 240.0
    else:
        base = 240.0 + ((size - 6) ** 2) * 90.0
    return max(5.0, base * scale)


def auto_subgoal_expansions(size: int) -> int:
    if size <= 3: return 200_000
    if size == 4: return 500_000
    if size == 5: return 1_000_000
    if size == 6: return 2_000_000
    return int(2_000_000 + (size - 6) * 1_000_000)


def auto_final_expansions(size: int) -> int:
    if size <= 3: return 1_000_000
    if size == 4: return 2_000_000
    if size == 5: return 5_000_000
    if size == 6: return 8_000_000
    return int(8_000_000 + (size - 6) * 2_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the live number puzzle webpage with a scalable solver."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://superglutenman0312.github.io/number_puzzle/",
    )
        # 🌟 新增：名字參數
    parser.add_argument(
        "--name", 
        type=str, 
        default="Browniebro", 
        help="自動在通關後填入排行榜的大名 (例如: OrthoLoc_Master)"
    )
    
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--move-delay-ms", type=int, default=15)
    parser.add_argument("--channel", type=str, default="msedge", choices=["msedge", "chrome"])
    parser.add_argument("--astar-weight", type=float, default=1.8)
    parser.add_argument("--subgoal-max-expansions", type=int, default=None)
    parser.add_argument("--final-max-expansions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timeout_sec = float(args.timeout_sec) if args.timeout_sec is not None else auto_timeout_sec(args.size, args.timeout_scale)
    subgoal_max_expansions = int(args.subgoal_max_expansions) if args.subgoal_max_expansions is not None else auto_subgoal_expansions(args.size)
    final_max_expansions = int(args.final_max_expansions) if args.final_max_expansions is not None else auto_final_expansions(args.size)

    solver = StrategicNPuzzleSolver(
        size=args.size,
        astar_weight=args.astar_weight,
        subgoal_max_expansions=subgoal_max_expansions,
        final_max_expansions=final_max_expansions,
    )

    print(
        f"solver_config size={args.size} "
        f"timeout_sec={timeout_sec:.1f} "
        f"astar_weight={args.astar_weight:.3f} "
        f"subgoal_max_expansions={subgoal_max_expansions} "
        f"final_max_expansions={final_max_expansions}"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel=args.channel,
            headless=args.headless,
            args=["--start-maximized"] if not args.headless else [],
        )

        if args.headless:
            context = browser.new_context(viewport={"width": 1600, "height": 1200})
        else:
            context = browser.new_context(no_viewport=True)

        page = context.new_page()
        # ==========================================
        # 🌟 終極防禦：原生彈窗 (window.prompt) 攔截器
        # ==========================================
        def handle_dialog(dialog):
            if dialog.type == "prompt" and args.name:
                print(f"\n💬 [系統攔截] 偵測到原生輸入框！自動霸氣填入：【{args.name}】")
                dialog.accept(args.name)
            else:
                dialog.accept()
        page.on("dialog", handle_dialog)
        # ==========================================
        
        page.goto(args.url, wait_until="domcontentloaded")
        page.locator("body").wait_for(state="visible", timeout=10000)

        if args.fullscreen and not args.headless:
            try:
                page.keyboard.press("F11")
            except PlaywrightError:
                pass

        selected = select_size(page, args.size)
        if not selected:
            print(f"Warning: explicit size click for {args.size}x{args.size} did not succeed.")

        click_start_after_size(page)

        if args.manual:
            print("Manual mode enabled. Set up the board in the browser window, then press Enter here.")
            input()

        board, _ = read_board(page, args.size)
        print("board:", board)
        print("solvable:", solver.is_solvable(board))

        start = time.perf_counter()
        solution, stats = solver.solve(board=board, timeout_sec=timeout_sec)
        elapsed = time.perf_counter() - start

        print("search stats:", stats)
        print(f"search wall time: {elapsed:.4f}s")

        if solution is None:
            raise RuntimeError("Solver 完全沒有回傳路徑！")
        elif not stats.solved:
            print(f"\n⚠️ 警告：只找到「部分解法」({len(solution)} 步)！將執行到卡關處暫停供人類觀察...")
        else:
            print(f"\n✅ 成功找到完整解法！共 {len(solution)} 步。")
            
        print("solution length:", len(solution))

        apply_solution_in_browser(
            page=page,
            size=args.size,
            initial_board=board,
            solution=solution,
            move_delay_ms=args.move_delay_ms,
        )

        # 🌟 觸發命名機制
        if args.name:
            # 如果終端機有下 --name 參數，就自動暴力填表
            submit_name(page, args.name)
        else:
            # 💡 如果沒給 --name，代表你要看心情。把網頁留在這，讓你親手在瀏覽器上操作！
            print("\n🏆 通關啦！如果畫面上出現了輸入框，請直接用滑鼠點擊網頁，輸入你當下想叫的名字吧！")

        print("Done. Close the browser window or press Ctrl+C to exit.")
        try:
            page.wait_for_event("close", timeout=0)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()