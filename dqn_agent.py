from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class TileEmbeddingDQN(nn.Module):
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
        self.backbone = nn.Sequential(
            nn.Linear(self.n_tiles * self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions),
        )

    def forward(self, board: Tensor) -> Tensor:
        # board: [batch, n_tiles] with integer tile ids in [0, n_tiles - 1]
        embedded = self.embedding(board.long())
        flat = embedded.reshape(board.shape[0], -1)
        return self.backbone(flat)


@dataclass
class Batch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor
    next_masks: Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]) -> None:
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_masks = np.zeros((capacity, 4), dtype=np.bool_)
        self.index = 0
        self.full = False

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

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        size = len(self)
        indices = np.random.randint(0, size, size=batch_size)
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
        )


class DQNAgent:
    def __init__(
        self,
        n_tiles: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
    ) -> None:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = float(gamma)
        self.policy_net = TileEmbeddingDQN(
            n_tiles=n_tiles,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net = TileEmbeddingDQN(
            n_tiles=n_tiles,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    @staticmethod
    def mask_q_values(q_values: Tensor, legal_mask: Tensor) -> Tensor:
        masked = q_values.clone()
        masked[~legal_mask] = -1e9
        return masked

    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        epsilon: float,
    ) -> int:
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
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> float:
        batch = replay.sample(batch_size=batch_size, device=self.device)

        q_values = self.policy_net(batch.states)
        q_selected = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online = self.policy_net(batch.next_states)
            next_online_masked = self.mask_q_values(next_online, batch.next_masks)
            next_actions = next_online_masked.argmax(dim=1, keepdim=True)

            next_target = self.target_net(batch.next_states)
            next_q = next_target.gather(1, next_actions).squeeze(1)
            targets = batch.rewards + (1.0 - batch.dones) * self.gamma * next_q

        loss = self.loss_fn(q_selected, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path, metadata: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "model_state": self.policy_net.state_dict(),
            "n_tiles": self.policy_net.n_tiles,
            "embedding_dim": self.policy_net.embedding_dim,
            "hidden_dim": self.policy_net.hidden_dim,
            "metadata": metadata or {},
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        lr: float = 1e-3,
    ) -> Tuple["DQNAgent", Dict[str, object]]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            n_tiles=int(payload["n_tiles"]),
            embedding_dim=int(payload["embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            lr=lr,
            gamma=gamma,
            device=device,
        )
        agent.policy_net.load_state_dict(payload["model_state"])
        agent.target_net.load_state_dict(payload["model_state"])
        agent.policy_net.eval()
        agent.target_net.eval()
        return agent, payload.get("metadata", {})
