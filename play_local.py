from __future__ import annotations

import argparse
from typing import List

import torch

from dqn_agent import DQNAgent
from npuzzle_env import NPuzzleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DQN agent on the local N-puzzle environment.")
    parser.add_argument("checkpoint", type=str, help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--scramble-depth", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--show-boards", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, metadata = DQNAgent.load(args.checkpoint, device=device)
    size = int(metadata.get("size", int(round(agent.policy_net.n_tiles ** 0.5))))
    max_steps = args.max_steps or int(metadata.get("max_steps", 96))
    scramble_depth = args.scramble_depth or int(metadata.get("max_depth", max(12, size * size * 4)))

    env = NPuzzleEnv(size=size, max_steps=max_steps)
    solved = 0
    step_counts: List[int] = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(scramble_depth=scramble_depth)
        if args.show_boards:
            print(f"\nEpisode {episode} initial board:\n{env.render()}\n")

        done = False
        while not done:
            mask = env.legal_actions_mask()
            action = agent.select_action(state, mask, epsilon=0.0)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        solved += int(bool(info["solved"]))
        step_counts.append(env.steps_taken)
        status = "SOLVED" if info["solved"] else "FAILED"
        print(
            f"episode={episode:3d} status={status:6s} "
            f"steps={env.steps_taken:3d} manhattan={info['manhattan']:3d}"
        )

        if args.show_boards:
            print(f"Final board:\n{env.render()}\n")

    print(
        f"\nSolve rate: {solved}/{args.episodes} = {solved / args.episodes:.3f}; "
        f"average steps: {sum(step_counts) / len(step_counts):.2f}"
    )


if __name__ == "__main__":
    main()
