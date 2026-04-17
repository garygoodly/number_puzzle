from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import numpy as np
import torch
from playwright.sync_api import Browser, Page, sync_playwright

from dqn_agent import DQNAgent
from npuzzle_env import is_solved, legal_actions_mask_from_board, tile_for_action

DEFAULT_URL = "https://superglutenman0312.github.io/number_puzzle/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a trained DQN agent to play the live Number Puzzle webpage.")
    parser.add_argument("checkpoint", type=str, help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--size", type=int, default=3, help="Puzzle size to launch on the webpage.")
    parser.add_argument("--max-agent-steps", type=int, default=128)
    parser.add_argument("--move-delay", type=float, default=0.12)
    parser.add_argument("--headless", action="store_true", help="Run Chromium headless.")
    return parser.parse_args()


def board_state_from_dom(page: Page) -> Dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const boardEl = document.getElementById('board');
          const wrappers = Array.from(boardEl.querySelectorAll('.tile-wrapper'));
          const size = Math.round(Math.sqrt(wrappers.length + 1));
          const board = Array(size * size).fill(0);

          for (const wrapper of wrappers) {
            const tile = wrapper.querySelector('.tile');
            const num = parseInt(tile.textContent.trim(), 10);
            const top = parseFloat(wrapper.style.top);
            const left = parseFloat(wrapper.style.left);
            const row = Math.round(top / (100 / size));
            const col = Math.round(left / (100 / size));
            board[row * size + col] = num;
          }

          const movesText = document.getElementById('moves')?.textContent ?? '0';
          const timerText = document.getElementById('timer')?.textContent ?? '00:00';
          return {
            board,
            size,
            moves: parseInt(movesText, 10),
            timer: timerText,
            inGame: getComputedStyle(document.getElementById('game-container')).display !== 'none'
          };
        }
        """
    )


def dispatch_tile_click(page: Page, tile_num: int) -> None:
    ok = page.evaluate(
        """
        (num) => {
          const tiles = Array.from(document.querySelectorAll('#board .tile'));
          const target = tiles.find(el => el.textContent.trim() === String(num));
          if (!target) return false;
          target.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
          return true;
        }
        """,
        tile_num,
    )
    if not ok:
        raise RuntimeError(f"Could not find tile {tile_num} on the page.")


def patch_page(page: Page, size: int) -> None:
    # Skip tutorial, suppress blocking dialogs, and avoid posting leaderboard entries.
    page.evaluate(
        """
        (size) => {
          localStorage.setItem('klotski_skip_tutorial', 'true');
          localStorage.setItem(`bestTime_${size}`, '0');
          window.alert = () => {};
          window.prompt = () => 'RL Agent';
          if (typeof saveToCloud === 'function') {
            saveToCloud = () => { if (typeof exitGame === 'function') exitGame(); };
          }
        }
        """,
        size,
    )


def start_game(page: Page, url: str, size: int) -> None:
    page.goto(url, wait_until="domcontentloaded")
    patch_page(page, size)
    page.wait_for_timeout(300)

    # The menu buttons are created on window.onload and contain text like "3 x 3 拼圖".
    page.locator("#menu button.btn").filter(has_text=f"{size} x {size} 拼圖").first.click()
    page.wait_for_timeout(300)

    # Defensive: if the tutorial modal still appears, confirm it.
    modal_button = page.locator("#modal-action-btn")
    if modal_button.is_visible():
        modal_button.click()
        page.wait_for_timeout(300)

    state = board_state_from_dom(page)
    if not state["inGame"]:
        raise RuntimeError("Game did not start correctly.")


def run_agent(
    page: Page,
    agent: DQNAgent,
    size: int,
    max_agent_steps: int,
    move_delay: float,
) -> Dict[str, Any]:
    for step in range(1, max_agent_steps + 1):
        state = board_state_from_dom(page)
        board = state["board"]
        if state["size"] != size:
            raise RuntimeError(f"Unexpected board size on page: {state['size']} != {size}")
        if is_solved(board, size):
            state["agent_steps"] = step - 1
            state["solved"] = True
            return state

        board_np = np.array(board, dtype=np.int64)
        legal_mask = legal_actions_mask_from_board(board_np, size)
        action = agent.select_action(state=board_np, legal_mask=legal_mask, epsilon=0.0)
        tile_num = tile_for_action(board_np, size, action)
        if tile_num is None:
            raise RuntimeError(f"Agent picked illegal action {action}.")

        dispatch_tile_click(page, tile_num)
        time.sleep(move_delay)

    state = board_state_from_dom(page)
    state["agent_steps"] = max_agent_steps
    state["solved"] = is_solved(state["board"], size)
    return state


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, metadata = DQNAgent.load(args.checkpoint, device=device)
    model_size = int(metadata.get("size", int(round(agent.policy_net.n_tiles ** 0.5))))
    if model_size != args.size:
        raise ValueError(
            f"Checkpoint was trained for size {model_size}x{model_size}, but --size={args.size}."
        )

    with sync_playwright() as pw:
        browser: Browser = pw.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1200, "height": 1000})
        start_game(page, url=args.url, size=args.size)
        result = run_agent(
            page=page,
            agent=agent,
            size=args.size,
            max_agent_steps=args.max_agent_steps,
            move_delay=args.move_delay,
        )
        browser.close()

    print(
        f"solved={result['solved']} moves={result['moves']} "
        f"timer={result['timer']} agent_steps={result['agent_steps']}"
    )


if __name__ == "__main__":
    main()
