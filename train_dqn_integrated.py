from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from npuzzle_env import NPuzzleEnv


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    frac = min(max(step / total_steps, 0.0), 1.0)
    return start + frac * (end - start)


def curriculum_depth(
    episode: int,
    min_depth: int,
    max_depth: int,
    ramp_episodes: int,
) -> int:
    if ramp_episodes <= 0:
        return max_depth
    frac = min(max(episode / ramp_episodes, 0.0), 1.0)
    return int(round(min_depth + frac * (max_depth - min_depth)))


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
class TileEmbeddingDuelingDQN(nn.Module):
    def __init__(
        self,
        n_tiles: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        n_actions: int = 4,
    ) -> None:
        super().__init__()
        self.n_tiles = int(n_tiles)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)

        self.embedding = nn.Embedding(self.n_tiles, self.embedding_dim)
        self.feature = nn.Sequential(
            nn.Linear(self.n_tiles * self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_actions),
        )

    def forward(self, board: Tensor) -> Tensor:
        embedded = self.embedding(board.long())
        flat = embedded.reshape(board.shape[0], -1)
        features = self.feature(flat)
        value = self.value_head(features)
        advantage = self.adv_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# -----------------------------------------------------------------------------
# Prioritized Replay
# -----------------------------------------------------------------------------
@dataclass
class Batch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor
    next_masks: Tensor
    weights: Tensor
    indices: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        alpha: float = 0.6,
        eps: float = 1e-5,
    ) -> None:
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.states = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_masks = np.zeros((capacity, 4), dtype=np.bool_)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.index = 0
        self.full = False
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.capacity if self.full else self.index

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        idx = self.index
        self.states[idx] = state
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.next_masks[idx] = next_mask
        self.priorities[idx] = self.max_priority

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size: int, beta: float, device: torch.device) -> Batch:
        size = len(self)
        if size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        priorities = self.priorities[:size]
        scaled = np.power(np.maximum(priorities, self.eps), self.alpha)
        probs = scaled / scaled.sum()
        indices = np.random.choice(size, size=batch_size, replace=size < batch_size, p=probs)

        weights = np.power(size * probs[indices], -beta)
        weights /= weights.max()

        return Batch(
            states=torch.as_tensor(self.states[indices], device=device, dtype=torch.long),
            actions=torch.as_tensor(self.actions[indices], device=device, dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[indices], device=device, dtype=torch.float32),
            next_states=torch.as_tensor(
                self.next_states[indices], device=device, dtype=torch.long
            ),
            dones=torch.as_tensor(self.dones[indices], device=device, dtype=torch.float32),
            next_masks=torch.as_tensor(
                self.next_masks[indices], device=device, dtype=torch.bool
            ),
            weights=torch.as_tensor(weights, device=device, dtype=torch.float32),
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        updated = np.abs(td_errors).astype(np.float32) + self.eps
        self.priorities[indices] = updated
        self.max_priority = max(self.max_priority, float(updated.max(initial=self.max_priority)))


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        n_tiles: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
        tau: float = 1.0,
        dueling: bool = True,
    ) -> None:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.dueling = bool(dueling)

        network_cls = TileEmbeddingDuelingDQN
        self.policy_net = network_cls(
            n_tiles=n_tiles,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net = network_cls(
            n_tiles=n_tiles,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    @staticmethod
    def mask_q_values(q_values: Tensor, legal_mask: Tensor) -> Tensor:
        masked = q_values.clone()
        masked[~legal_mask] = -1e9
        return masked

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float) -> int:
        legal_indices = np.flatnonzero(legal_mask)
        if legal_indices.size == 0:
            raise RuntimeError("No legal actions available.")

        if np.random.random() < epsilon:
            return int(np.random.choice(legal_indices))

        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.long).unsqueeze(0)
        mask_tensor = torch.as_tensor(legal_mask, device=self.device, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            masked = self.mask_q_values(q_values, mask_tensor)
            action = int(masked.argmax(dim=1).item())
        return action

    def update_target(self) -> None:
        if self.tau >= 1.0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return

        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * policy_param.data)

    def train_step(
        self,
        replay: PrioritizedReplayBuffer,
        batch_size: int,
        beta: float,
    ) -> Dict[str, float]:
        batch = replay.sample(batch_size=batch_size, beta=beta, device=self.device)

        q_values = self.policy_net(batch.states)
        q_selected = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online = self.policy_net(batch.next_states)
            next_online_masked = self.mask_q_values(next_online, batch.next_masks)
            next_actions = next_online_masked.argmax(dim=1, keepdim=True)

            next_target = self.target_net(batch.next_states)
            next_q = next_target.gather(1, next_actions).squeeze(1)
            targets = batch.rewards + (1.0 - batch.dones) * self.gamma * next_q

        td_errors = targets - q_selected
        loss_per_sample = self.loss_fn(q_selected, targets)
        loss = (batch.weights * loss_per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0))
        self.optimizer.step()

        replay.update_priorities(batch.indices, td_errors.detach().cpu().numpy())
        return {
            "loss": float(loss.item()),
            "td_error": float(td_errors.abs().mean().item()),
            "grad_norm": grad_norm,
        }

    def save(self, path: str | Path, metadata: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "model_state": self.policy_net.state_dict(),
            "n_tiles": self.policy_net.n_tiles,
            "embedding_dim": self.policy_net.embedding_dim,
            "hidden_dim": self.policy_net.hidden_dim,
            "metadata": metadata or {},
            "dueling": self.dueling,
            "tau": self.tau,
        }
        torch.save(payload, Path(path))


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    agent: DQNAgent,
    env: NPuzzleEnv,
    episodes: int,
    scramble_depth: int,
) -> Dict[str, float]:
    solved = 0
    steps_used: List[int] = []
    final_manhattans: List[int] = []
    rewards: List[float] = []

    for _ in range(episodes):
        state, info = env.reset(scramble_depth=scramble_depth)
        done = False
        total_reward = 0.0
        while not done:
            mask = env.legal_actions_mask()
            action = agent.select_action(state, mask, epsilon=0.0)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            state = next_state
            info = step_info
            done = terminated or truncated

        solved += int(bool(info["solved"]))
        steps_used.append(env.steps_taken)
        final_manhattans.append(int(info["manhattan"]))
        rewards.append(total_reward)

    return {
        "solve_rate": solved / episodes,
        "avg_steps": float(mean(steps_used)),
        "avg_final_manhattan": float(mean(final_manhattans)),
        "avg_reward": float(mean(rewards)),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Dueling Double DQN agent with prioritized replay for the N-puzzle."
    )
    parser.add_argument("--size", type=int, default=3, help="Board size N for an N x N puzzle.")
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use. 'auto' selects CUDA when available, else CPU.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Final scramble depth. Defaults to max(12, size^2 * 4).",
    )
    parser.add_argument(
        "--curriculum-ramp",
        type=int,
        default=12000,
        help="Slower ramp than the original script to avoid collapsing at mid-depth.",
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=200000,
        help="Longer exploration schedule to keep learning at higher depths.",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Used when tau >= 1.0. Ignored for soft target updates.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="Soft target update rate. Set to 1.0 to use hard periodic copies.",
    )
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="runs/number_puzzle_dueling_ddqn_per")

    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument(
        "--per-beta-steps",
        type=int,
        default=200000,
        help="Anneal importance-sampling correction gradually.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    size = int(args.size)
    max_depth = args.max_depth or max(12, size * size * 4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = NPuzzleEnv(size=size, max_steps=args.max_steps, seed=args.seed)
    eval_env = NPuzzleEnv(size=size, max_steps=args.max_steps, seed=args.seed + 1)

    agent = DQNAgent(
        n_tiles=size * size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=device,
        tau=args.tau,
        dueling=True,
    )
    replay = PrioritizedReplayBuffer(
        capacity=args.buffer_size,
        state_shape=(size * size,),
        alpha=args.per_alpha,
    )

    global_step = 0
    best_solve_rate = -math.inf
    metrics: List[Dict[str, float]] = []

    training_start_time = time.perf_counter()

    print(
        f"Training size={size}x{size} for {args.episodes} episodes on {agent.device}. "
        f"Curriculum depth: {args.min_depth} -> {max_depth} | "
        f"dueling=True double_dqn=True prioritized_replay=True"
    )

    for episode in range(1, args.episodes + 1):
        episode_start_time = time.perf_counter()

        depth = curriculum_depth(
            episode=episode,
            min_depth=args.min_depth,
            max_depth=max_depth,
            ramp_episodes=args.curriculum_ramp,
        )
        state, _ = env.reset(scramble_depth=depth)
        done = False
        losses: List[float] = []
        td_errors: List[float] = []
        episode_reward = 0.0
        epsilon = args.epsilon_start
        beta = args.per_beta_start

        while not done:
            legal_mask = env.legal_actions_mask()
            epsilon = linear_schedule(
                args.epsilon_start,
                args.epsilon_end,
                global_step,
                args.epsilon_decay_steps,
            )
            beta = linear_schedule(
                args.per_beta_start,
                args.per_beta_end,
                global_step,
                args.per_beta_steps,
            )
            action = agent.select_action(state, legal_mask, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_mask = env.legal_actions_mask()
            replay.add(state, action, reward, next_state, terminated or truncated, next_mask)

            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay) >= args.batch_size and global_step >= args.warmup_steps:
                train_stats = agent.train_step(replay, args.batch_size, beta=beta)
                losses.append(train_stats["loss"])
                td_errors.append(train_stats["td_error"])

                if args.tau < 1.0:
                    agent.update_target()
                elif global_step % args.target_update == 0:
                    agent.update_target()

            done = terminated or truncated

        episode_time = time.perf_counter() - episode_start_time

        if episode % 50 == 0 or episode == 1:
            avg_loss = mean(losses) if losses else float("nan")
            avg_td = mean(td_errors) if td_errors else float("nan")
            elapsed = time.perf_counter() - training_start_time
            avg_episode_time = elapsed / episode
            print(
                f"episode={episode:5d} depth={depth:3d} steps={env.steps_taken:3d} "
                f"reward={episode_reward:7.3f} epsilon={epsilon:5.3f} beta={beta:5.3f} "
                f"loss={avg_loss:7.4f} td={avg_td:7.4f} "
                f"ep_time={episode_time:6.2f}s elapsed={format_seconds(elapsed)} "
                f"avg_ep={avg_episode_time:6.2f}s"
            )

        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_start_time = time.perf_counter()

            summary = evaluate(
                agent=agent,
                env=eval_env,
                episodes=args.eval_episodes,
                scramble_depth=max_depth,
            )

            eval_time = time.perf_counter() - eval_start_time
            summary["episode"] = float(episode)
            summary["eval_time_sec"] = float(eval_time)
            metrics.append(summary)

            print(
                "eval "
                f"episode={episode:5d} solve_rate={summary['solve_rate']:.3f} "
                f"avg_steps={summary['avg_steps']:.2f} "
                f"avg_final_manhattan={summary['avg_final_manhattan']:.2f} "
                f"avg_reward={summary['avg_reward']:.3f} "
                f"eval_time={eval_time:.2f}s"
            )

            checkpoint_path = save_dir / "latest.pt"
            metadata = {
                "size": size,
                "max_depth": max_depth,
                "max_steps": args.max_steps,
                "eval_summary": summary,
                "dueling": True,
                "double_dqn": True,
                "prioritized_replay": True,
                "tau": args.tau,
                "per_alpha": args.per_alpha,
            }
            agent.save(checkpoint_path, metadata=metadata)

            if summary["solve_rate"] > best_solve_rate:
                best_solve_rate = summary["solve_rate"]
                agent.save(save_dir / "best.pt", metadata=metadata)

    total_training_time = time.perf_counter() - training_start_time
    avg_episode_time = total_training_time / max(args.episodes, 1)

    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Finished. Best solve rate: {best_solve_rate:.3f}")
    print(
        f"Total training time: {format_seconds(total_training_time)} "
        f"({total_training_time:.2f} sec)"
    )
    print(f"Average episode time: {avg_episode_time:.2f} sec")
    print(f"Artifacts saved under: {save_dir}")


if __name__ == "__main__":
    main()