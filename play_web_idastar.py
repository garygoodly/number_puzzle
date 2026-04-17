from __future__ import annotations

import argparse
import time
from typing import Sequence, Tuple

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from npuzzle_ida_star_solver import IDAStarNPuzzleSolver


def click_if_visible(page, selector: str, timeout_ms: int = 1000) -> bool:
    try:
        locator = page.locator(selector).first
        locator.wait_for(state="visible", timeout=timeout_ms)
        locator.click(timeout=timeout_ms)
        return True
    except Exception:
        return False


def confirm_instruction(page) -> bool:
    # Strongest selector first: user-provided exact button id.
    selectors = [
        "#modal-action-btn",
        "button#modal-action-btn",
        "button:has-text('開始挑戰')",
        "button:has-text('开始挑战')",
        "text=開始挑戰",
        "text=开始挑战",
        "text=開始",
        "text=Start",
    ]

    for selector in selectors:
        if click_if_visible(page, selector, timeout_ms=1200):
            page.wait_for_timeout(800)
            return True
    return False



def try_select_size(page, size: int) -> bool:
    labels = [
        f"{size} x {size}",
        f"{size}x{size}",
        f"{size} X {size}",
    ]
    selectors = [f"text={label}" for label in labels]

    # Also try buttons or generic clickable elements containing the size text.
    for label in labels:
        selectors.extend(
            [
                f"button:has-text('{label}')",
                f".btn:has-text('{label}')",
                f"*:has-text('{label}')",
            ]
        )

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=1200)
                page.wait_for_timeout(800)
                return True
        except Exception:
            continue
    return False


def ensure_game_ready(page, size: int, settle_ms: int) -> None:
    # First, choose puzzle size.
    selected = try_select_size(page, size)
    if not selected:
        print(f"Warning: could not explicitly click size {size}x{size}; continuing anyway.")

    page.wait_for_timeout(settle_ms)

    # Only after size selection, confirm the instruction modal if it appears.
    confirm_instruction(page)
    page.wait_for_timeout(settle_ms)


def get_board_rect(page) -> dict:
    board = page.locator("#board")
    board.wait_for(state="visible", timeout=15000)

    for _ in range(60):
        box = board.bounding_box()
        if box and box["width"] > 40 and box["height"] > 40:
            return box
        page.wait_for_timeout(250)

    raise RuntimeError("Could not read #board bounding box after waiting for board to render.")


def read_board(page, size: int) -> Tuple[Tuple[int, ...], dict]:
    rect = get_board_rect(page)

    items = page.locator("#board *").evaluate_all(
        """
        (elements) => elements.map((el) => {
            const r = el.getBoundingClientRect();
            const text = (el.innerText || el.textContent || '').trim();
            return {
                text,
                x: r.x,
                y: r.y,
                width: r.width,
                height: r.height,
                display: window.getComputedStyle(el).display,
                visibility: window.getComputedStyle(el).visibility,
                opacity: window.getComputedStyle(el).opacity,
            };
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
    action_to_delta = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    board = list(initial_board)
    rect = get_board_rect(page)

    for step_idx, action in enumerate(solution, start=1):
        blank_idx = board.index(0)
        blank_row, blank_col = divmod(blank_idx, size)

        d_row, d_col = action_to_delta[int(action)]
        tile_row = blank_row + d_row
        tile_col = blank_col + d_col

        if not (0 <= tile_row < size and 0 <= tile_col < size):
            raise RuntimeError(
                f"Illegal browser move at step {step_idx}: "
                f"action={action}, blank=({blank_row},{blank_col})"
            )

        tile_idx = tile_row * size + tile_col

        click_cell(page, rect, size, tile_row, tile_col)

        # Update local board model instead of re-reading the browser board every step.
        board[blank_idx], board[tile_idx] = board[tile_idx], board[blank_idx]

        if move_delay_ms > 0:
            page.wait_for_timeout(move_delay_ms)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the live number puzzle webpage with IDA*."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://superglutenman0312.github.io/number_puzzle/",
    )
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--move-delay-ms", type=int, default=120)
    parser.add_argument("--scramble-wait-ms", type=int, default=1200)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Wait for you to set up the board manually before reading it.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="msedge",
        choices=["msedge", "chrome"],
        help="Use installed Edge or Chrome instead of Playwright-downloaded Chromium.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = IDAStarNPuzzleSolver(size=args.size)

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

        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)

        if args.fullscreen and not args.headless:
            try:
                page.keyboard.press("F11")
                page.wait_for_timeout(1000)
            except PlaywrightError:
                pass

        ensure_game_ready(page, args.size, args.scramble_wait_ms)

        if args.manual:
            print("Manual mode enabled. Set up the board in the browser window, then press Enter here.")
            input()
        else:
            page.wait_for_timeout(args.scramble_wait_ms)

        board, _ = read_board(page, args.size)
        print("board:", board)
        print("heuristic:", solver.heuristic(board))
        print("solvable:", solver.is_solvable(board))

        start = time.perf_counter()
        solution, stats = solver.solve(board=board, timeout_sec=args.timeout_sec)
        elapsed = time.perf_counter() - start

        print("search stats:", stats)
        print(f"search wall time: {elapsed:.4f}s")
        print("solution:", solution)

        if solution is None:
            raise RuntimeError("Solver did not find a solution within the given limits.")

        apply_solution_in_browser(
            page=page,
            size=args.size,
            initial_board=board,
            solution=solution,
            move_delay_ms=args.move_delay_ms,
        )

        print("Done. Close the browser window when you want to exit.")

        try:
            page.wait_for_event("close", timeout=0)
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