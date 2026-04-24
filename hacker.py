from __future__ import annotations

import argparse
import time

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


DEFAULT_URL = "https://superglutenman0312.github.io/number_puzzle/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Edge, open the number puzzle, and force a solved board."
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--size", type=int, default=3, help="Puzzle size, e.g. 3, 4, 5")
    parser.add_argument(
        "--name",
        type=str,
        default="BrowniebroProMax",
        help="Name to submit if the game prompts for one",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Edge in headless mode",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=10.0,
        help="How long to keep the browser open before quitting",
    )
    return parser.parse_args()


def build_solved_board(size: int) -> list[int]:
    return list(range(1, size * size)) + [0]


def click_size_button(wait: WebDriverWait, size: int) -> None:
    xpaths = [
        f"//button[contains(normalize-space(.), '{size} x {size}')]",
        f"//button[contains(normalize-space(.), '{size}x{size}')]",
        f"//button[contains(., '{size} x {size}')]",
    ]

    last_error = None
    for xpath in xpaths:
        try:
            button = wait.until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            button.click()
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not find/click the {size}x{size} size button.") from last_error


def click_start_button(wait: WebDriverWait) -> None:
    selectors = [
        (By.ID, "modal-action-btn"),
        (By.XPATH, "//button[contains(normalize-space(.), '開始挑戰')]"),
        (By.XPATH, "//button[contains(normalize-space(.), '开始挑战')]"),
        (By.XPATH, "//button[contains(normalize-space(.), 'Start')]"),
    ]

    last_error = None
    for by, selector in selectors:
        try:
            button = wait.until(
                EC.element_to_be_clickable((by, selector))
            )
            button.click()
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Could not find/click the start button.") from last_error


def submit_name_if_prompted(driver: webdriver.Edge, wait: WebDriverWait, name: str) -> None:
    try:
        alert = driver.switch_to.alert
        if alert.text is not None:
            alert.send_keys(name)
            alert.accept()
            return
    except Exception:
        pass

    try:
        input_box = WebDriverWait(driver, 3).until(
            EC.visibility_of_element_located((By.XPATH, "//input"))
        )
        input_box.clear()
        input_box.send_keys(name)

        confirm_xpaths = [
            "//button[contains(normalize-space(.), '確認')]",
            "//button[contains(normalize-space(.), '確定')]",
            "//button[contains(normalize-space(.), 'Submit')]",
            "//button[contains(normalize-space(.), 'OK')]",
        ]

        for xpath in confirm_xpaths:
            try:
                button = driver.find_element(By.XPATH, xpath)
                button.click()
                break
            except Exception:
                continue
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    if args.size < 3:
        raise ValueError("Size must be >= 3.")

    solved_board = build_solved_board(args.size)

    opts = Options()
    opts.use_chromium = True
    if args.headless:
        opts.add_argument("--headless=new")

    driver = webdriver.Edge(options=opts)
    wait = WebDriverWait(driver, 15)

    try:
        driver.get(args.url)

        click_size_button(wait, args.size)
        click_start_button(wait)

        wait.until(EC.visibility_of_element_located((By.ID, "board")))
        time.sleep(1)

        result = driver.execute_script(
            """
            const forcedSize = arguments[0];
            const solvedBoard = arguments[1];

            const state = {
                hasCurrentSize: typeof currentSize !== 'undefined',
                hasBoardArr: typeof boardArr !== 'undefined',
                hasInitBoardDOM: typeof initBoardDOM,
                hasCheckWin: typeof checkWin,
            };

            if (!state.hasCurrentSize || !state.hasBoardArr) {
                return {
                    ok: false,
                    reason: 'Game globals are not reachable.',
                    state
                };
            }

            currentSize = forcedSize;
            boardArr = solvedBoard.slice();
            moveCount = 1;
            seconds = 1;
            isGameWon = false;
            isTimerStarted = true;

            if (typeof initBoardDOM === 'function') {
                initBoardDOM();
            }

            if (typeof checkWin === 'function') {
                checkWin();
            }

            return {
                ok: true,
                state,
                boardArr: boardArr
            };
            """,
            args.size,
            solved_board,
        )

        print("inject result:", result)

        time.sleep(2)
        submit_name_if_prompted(driver, wait, args.name)

        print(
            f"Done. size={args.size}, name={args.name}, hold_seconds={args.hold_seconds}"
        )
        time.sleep(args.hold_seconds)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()