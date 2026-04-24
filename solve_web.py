from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Sequence, Tuple

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from solver import ClickSpaceNPuzzleSolver

DEFAULT_URL = "https://superglutenman0312.github.io/number_puzzle/"
SUPPORTED_BROWSER_CHANNELS = ["chromium", "chrome", "msedge"]


def click_first_visible(page: Page, selectors: Sequence[str], timeout_ms: int = 3000) -> bool:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout_ms)
            locator.click(timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def select_board_size(page: Page, size: int) -> bool:
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


def start_game(page: Page) -> bool:
    selectors = [
        "#modal-action-btn",
        "button#modal-action-btn",
        "button:has-text('開始挑戰')",
        "button:has-text('开始挑战')",
        "text=開始挑戰",
        "text=开始挑战",
        "button:has-text('Start')",
        "text=Start",
    ]
    return click_first_visible(page, selectors, timeout_ms=5000)


def wait_for_board(page: Page, timeout_ms: int = 10000) -> Dict[str, float]:
    board = page.locator("#board")
    board.wait_for(state="visible", timeout=timeout_ms)

    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_box = None

    while time.perf_counter() < deadline:
        box = board.bounding_box()
        if box and box["width"] > 40 and box["height"] > 40:
            return box
        last_box = box
        page.wait_for_timeout(50)

    raise RuntimeError(f"Could not read a usable #board bounding box. Last box: {last_box}")


def read_board_state(page: Page, size: int) -> Tuple[Tuple[int, ...], Dict[str, float]]:
    rect = wait_for_board(page)
    items = page.locator("#board *").evaluate_all(
        """
        (elements) => elements.map((el) => {
            const r = el.getBoundingClientRect();
            const text = (el.innerText || el.textContent || '').trim();
            return {text, x: r.x, y: r.y, width: r.width, height: r.height};
        })
        """
    )

    cell_width = rect["width"] / size
    cell_height = rect["height"] / size
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
        col = int((center_x - rect["x"]) / cell_width)
        row = int((center_y - rect["y"]) / cell_height)

        if 0 <= row < size and 0 <= col < size:
            board[row * size + col] = value

    return tuple(board), rect


def click_cell(page: Page, rect: Dict[str, float], size: int, row: int, col: int) -> None:
    cell_width = rect["width"] / size
    cell_height = rect["height"] / size
    x = rect["x"] + (col + 0.5) * cell_width
    y = rect["y"] + (row + 0.5) * cell_height
    page.mouse.click(x, y)


def apply_click_plan(
    page: Page,
    rect: Dict[str, float],
    size: int,
    click_targets: Sequence[int],
    move_delay_ms: int,
) -> Dict[str, float]:
    click_time_seconds = 0.0
    wait_time_seconds = 0.0
    start = time.perf_counter()

    for target_index in click_targets:
        row, col = divmod(target_index, size)

        click_start = time.perf_counter()
        click_cell(page, rect, size, row, col)
        click_time_seconds += time.perf_counter() - click_start

        if move_delay_ms > 0:
            wait_start = time.perf_counter()
            page.wait_for_timeout(move_delay_ms)
            wait_time_seconds += time.perf_counter() - wait_start

    total_seconds = time.perf_counter() - start
    steps = max(len(click_targets), 1)

    return {
        "count": len(click_targets),
        "total_seconds": total_seconds,
        "click_seconds": click_time_seconds,
        "wait_seconds": wait_time_seconds,
        "avg_total_ms": total_seconds * 1000.0 / steps,
        "avg_click_ms": click_time_seconds * 1000.0 / steps,
        "avg_wait_ms": wait_time_seconds * 1000.0 / steps,
        "clicks_per_second": len(click_targets) / total_seconds if total_seconds > 0 else 0.0,
        "overhead_ms": max(0.0, (total_seconds - wait_time_seconds) * 1000.0 / steps),
    }


def submit_player_name(page: Page, name: str) -> None:
    try:
        page.wait_for_timeout(1500)

        input_locator = page.locator("input:visible").last
        input_locator.wait_for(state="visible", timeout=5000)
        input_locator.click()
        page.wait_for_timeout(100)
        input_locator.press("Control+A")
        input_locator.press("Backspace")
        page.wait_for_timeout(100)
        input_locator.press_sequentially(name, delay=60)
        page.wait_for_timeout(200)
        input_locator.press("Enter")
        page.wait_for_timeout(300)

        button_texts = ["確認", "確定", "送出", "Submit", "OK", "保存"]
        selectors = [f"button:has-text('{text}'):visible" for text in button_texts]
        selectors += [f".btn:has-text('{text}'):visible" for text in button_texts]

        for selector in selectors:
            try:
                button = page.locator(selector).first
                if button.is_visible():
                    button.click()
                    break
            except Exception:
                pass

        page.wait_for_timeout(1000)
    except Exception:
        pass


def estimate_timeout_seconds(size: int, scale: float) -> float:
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


def estimate_subgoal_expansions(size: int) -> int:
    if size <= 3:
        return 200_000
    if size == 4:
        return 500_000
    if size == 5:
        return 1_000_000
    if size == 6:
        return 2_000_000
    return int(2_000_000 + (size - 6) * 1_000_000)


def estimate_final_expansions(size: int) -> int:
    if size <= 3:
        return 1_000_000
    if size == 4:
        return 2_000_000
    if size == 5:
        return 5_000_000
    if size == 6:
        return 8_000_000
    return int(8_000_000 + (size - 6) * 2_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the live number puzzle webpage with Playwright."
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--name", type=str, default="Browniebro", help="Leaderboard name")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--move-delay-ms", type=int, default=15)
    parser.add_argument(
        "--browser",
        type=str,
        default="msedge",
        choices=SUPPORTED_BROWSER_CHANNELS,
        help="Browser channel for Playwright",
    )
    parser.add_argument("--astar-weight", type=float, default=2.2)
    parser.add_argument("--subgoal-max-expansions", type=int, default=None)
    parser.add_argument("--final-max-expansions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_start = time.perf_counter()

    timeout_seconds = (
        float(args.timeout_seconds)
        if args.timeout_seconds is not None
        else estimate_timeout_seconds(args.size, args.timeout_scale)
    )
    subgoal_max_expansions = (
        int(args.subgoal_max_expansions)
        if args.subgoal_max_expansions is not None
        else estimate_subgoal_expansions(args.size)
    )
    final_max_expansions = (
        int(args.final_max_expansions)
        if args.final_max_expansions is not None
        else estimate_final_expansions(args.size)
    )

    solver = ClickSpaceNPuzzleSolver(
        size=args.size,
        astar_weight=args.astar_weight,
        subgoal_max_expansions=subgoal_max_expansions,
        final_max_expansions=final_max_expansions,
    )

    print(
        f"solver_config size={args.size} "
        f"timeout_seconds={timeout_seconds:.1f} "
        f"astar_weight={args.astar_weight:.3f} "
        f"subgoal_max_expansions={subgoal_max_expansions} "
        f"final_max_expansions={final_max_expansions}"
    )

    with sync_playwright() as playwright:
        launch_kwargs: Dict[str, Any] = {
            "headless": args.headless,
            "args": ["--start-maximized"] if not args.headless else [],
        }
        if args.browser != "chromium":
            launch_kwargs["channel"] = args.browser

        browser = playwright.chromium.launch(**launch_kwargs)

        if args.headless:
            context = browser.new_context(viewport={"width": 1600, "height": 1200})
        else:
            context = browser.new_context(no_viewport=True)

        page = context.new_page()

        def handle_dialog(dialog: Any) -> None:
            if dialog.type == "prompt" and args.name:
                dialog.accept(args.name)
            else:
                dialog.accept()

        page.on("dialog", handle_dialog)

        page_load_start = time.perf_counter()
        page.goto(args.url, wait_until="domcontentloaded")
        page.locator("body").wait_for(state="visible", timeout=10000)
        page_load_seconds = time.perf_counter() - page_load_start

        if args.fullscreen and not args.headless:
            try:
                page.keyboard.press("F11")
            except PlaywrightError:
                pass

        ui_start = time.perf_counter()
        selected = select_board_size(page, args.size)
        if not selected:
            print(f"Warning: could not explicitly select {args.size}x{args.size}.")
        start_game(page)
        ui_setup_seconds = time.perf_counter() - ui_start

        if args.manual:
            print("Manual mode enabled. Prepare the board, then press Enter.")
            input()

        scan_start = time.perf_counter()
        board, rect = read_board_state(page, args.size)
        scan_seconds = time.perf_counter() - scan_start

        print("board:", board)
        print("solvable:", solver.is_solvable(board))

        solve_start = time.perf_counter()
        click_targets, stats = solver.solve_clicks(board=board, timeout_seconds=timeout_seconds)
        solve_seconds = time.perf_counter() - solve_start

        print("search stats:", stats)
        print(f"search wall time: {solve_seconds:.4f}s")

        if click_targets is None:
            raise RuntimeError("Solver did not return a click plan.")

        if stats.solved:
            print(f"Solved successfully with {len(click_targets)} click moves.")
        else:
            print(f"Partial plan returned with {len(click_targets)} click moves.")

        execution_stats = apply_click_plan(
            page=page,
            rect=rect,
            size=args.size,
            click_targets=click_targets,
            move_delay_ms=args.move_delay_ms,
        )

        submission_seconds = 0.0
        if args.name:
            submission_start = time.perf_counter()
            submit_player_name(page, args.name)
            submission_seconds = time.perf_counter() - submission_start
        else:
            print("Run completed. Enter your name manually if the game prompts for it.")

        total_seconds = time.perf_counter() - session_start

        print("\n================ PROFILE SUMMARY ================")
        print(f"page_load_seconds : {page_load_seconds:.4f}")
        print(f"ui_setup_seconds  : {ui_setup_seconds:.4f}")
        print(f"scan_seconds      : {scan_seconds:.4f}")
        print(f"solve_seconds     : {solve_seconds:.4f}")
        print(f"exec_total_seconds: {execution_stats['total_seconds']:.4f}")
        print(f"exec_click_seconds: {execution_stats['click_seconds']:.4f}")
        print(f"exec_wait_seconds : {execution_stats['wait_seconds']:.4f}")
        print(f"submit_seconds    : {submission_seconds:.4f}")
        print(f"total_seconds     : {total_seconds:.4f}")
        print("-----------------------------------------------")
        print(f"step_count        : {execution_stats['count']}")
        print(f"clicks_per_second : {execution_stats['clicks_per_second']:.2f}")
        print(f"avg_total_ms      : {execution_stats['avg_total_ms']:.2f}")
        print(f"avg_click_ms      : {execution_stats['avg_click_ms']:.2f}")
        print(f"avg_wait_ms       : {execution_stats['avg_wait_ms']:.2f}")
        print(f"avg_overhead_ms   : {execution_stats['overhead_ms']:.2f}")
        print("===============================================\n")

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
