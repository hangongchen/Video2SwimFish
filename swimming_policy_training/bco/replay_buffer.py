"""Fixed-size transition buffer for the BCO random-exploration rollout (Step 1).

Stores (obs, action, next_obs) triples collected from a random policy in the Isaac Lab
environment; used ONLY to train the inverse dynamics model (IDM, Step 2) and to warm-start
the BC policy's input normalizer (bc_trainer.py) from real interaction statistics. Not a
generic online-RL replay buffer: it fills once, to a fixed target size, and never evicts.
"""

from __future__ import annotations

from pathlib import Path

import torch


class TransitionReplayBuffer:
    """Preallocated (obs_t, action_t, next_obs_t) buffer, filled by vectorized `add()` calls
    from an Isaac Lab vectorized env (num_envs > 1 per call)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str | torch.device):
        assert capacity > 0 and obs_dim > 0 and action_dim > 0
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.action = torch.zeros(self.capacity, self.action_dim, device=self.device)
        self.next_obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._ptr

    @property
    def is_full(self) -> bool:
        return self._full

    def add(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> None:
        """Append a batch of B transitions (obs/next_obs: (B, obs_dim), action: (B, action_dim)).
        Extra transitions past `capacity` are silently dropped (the caller stops collecting once
        `is_full`, so in normal use this only trims the LAST, partial batch)."""
        assert obs.shape == next_obs.shape and obs.shape[-1] == self.obs_dim
        assert action.shape[-1] == self.action_dim and action.shape[0] == obs.shape[0]
        b = obs.shape[0]
        n = min(b, self.capacity - self._ptr)
        if n <= 0:
            self._full = True
            return
        sl = slice(self._ptr, self._ptr + n)
        self.obs[sl] = obs[:n].detach().to(self.device)
        self.action[sl] = action[:n].detach().to(self.device)
        self.next_obs[sl] = next_obs[:n].detach().to(self.device)
        self._ptr += n
        if self._ptr >= self.capacity:
            self._full = True

    def as_dict(self) -> dict[str, torch.Tensor]:
        """The FILLED portion only, as a dict of (N, dim) tensors."""
        n = len(self)
        return {"obs": self.obs[:n], "action": self.action[:n], "next_obs": self.next_obs[:n]}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"obs_dim": self.obs_dim, "action_dim": self.action_dim, **self.as_dict()},
            path,
        )
        print(f"[replay_buffer] saved {len(self)} transitions -> {path}", flush=True)

    @classmethod
    def load(cls, path: str, device: str | torch.device) -> "TransitionReplayBuffer":
        blob = torch.load(path, map_location=device, weights_only=False)
        n = blob["obs"].shape[0]
        buf = cls(capacity=n, obs_dim=blob["obs_dim"], action_dim=blob["action_dim"], device=device)
        buf.add(blob["obs"], blob["action"], blob["next_obs"])
        print(f"[replay_buffer] loaded {len(buf)} transitions <- {path}", flush=True)
        return buf
