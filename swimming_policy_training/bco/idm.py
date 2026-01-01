"""Inverse Dynamics Model (IDM): predicts the action a_t that produced a transition
(s_t -> s_{t+1}), trained on the random-exploration replay buffer (Step 2). Used later
(pseudo_labeler.py) to infer pseudo-action labels for action-free expert observations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .bco_config import BCOConfig

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class InverseDynamicsModel(nn.Module):
    """MLP: concat(s_t, s_{t+1}) -> predicted a_t.

    Args:
        obs_dim: dimensionality of a single observation vector s_t.
        action_dim: dimensionality of the action vector a_t (num_joints for this baseline).
        hidden_layers: hidden-layer widths, e.g. [512, 512, 256].
        activation: key into _ACTIVATIONS (default "elu").
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: list[int],
                 activation: str = "elu"):
        super().__init__()
        assert obs_dim > 0 and action_dim > 0 and len(hidden_layers) > 0
        act_cls = _ACTIVATIONS[activation]
        dims = [2 * obs_dim, *hidden_layers]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), act_cls()]
        layers.append(nn.Linear(dims[-1], action_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs_t: torch.Tensor, obs_tp1: torch.Tensor) -> torch.Tensor:
        assert obs_t.shape[-1] == self.obs_dim and obs_tp1.shape[-1] == self.obs_dim
        x = torch.cat([obs_t, obs_tp1], dim=-1)
        return self.net(x)


def train_idm(
    buffer: dict[str, torch.Tensor],
    cfg: BCOConfig,
    device: str | torch.device,
) -> InverseDynamicsModel:
    """Supervised training loop: MSE(IDM(s_t, s_{t+1}), a_t) over the replay buffer.

    Args:
        buffer: dict with "obs" (N, obs_dim), "action" (N, action_dim), "next_obs" (N, obs_dim)
            -- as produced by replay_buffer.TransitionReplayBuffer.as_dict().
        cfg: BCOConfig (reads idm_hidden_layers/idm_activation/idm_lr/idm_epochs/idm_batch_size).
        device: torch device to train on.

    Returns:
        the trained InverseDynamicsModel (in eval() mode).
    """
    obs, action, next_obs = buffer["obs"], buffer["action"], buffer["next_obs"]
    n = obs.shape[0]
    assert n > 0, "empty replay buffer"
    assert obs.shape == next_obs.shape
    assert action.shape[0] == n
    obs_dim, action_dim = obs.shape[-1], action.shape[-1]

    model = InverseDynamicsModel(obs_dim, action_dim, cfg.idm_hidden_layers, cfg.idm_activation)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.idm_lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(obs, action, next_obs),
        batch_size=min(cfg.idm_batch_size, n),
        shuffle=True,
        drop_last=False,
    )

    print(f"[idm] training on {n} transitions, obs_dim={obs_dim} action_dim={action_dim} "
          f"hidden={cfg.idm_hidden_layers} epochs={cfg.idm_epochs} lr={cfg.idm_lr} "
          f"batch_size={loader.batch_size}", flush=True)
    model.train()
    t0 = time.time()
    for epoch in range(1, cfg.idm_epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for s_t, a_t, s_tp1 in loader:
            s_t, a_t, s_tp1 = s_t.to(device), a_t.to(device), s_tp1.to(device)
            pred = model(s_t, s_tp1)
            loss = loss_fn(pred, a_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        elapsed = time.time() - t0
        eta = elapsed / epoch * (cfg.idm_epochs - epoch)
        print(f"[idm] epoch {epoch}/{cfg.idm_epochs}  mse={epoch_loss:.6f}  "
              f"elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)
    model.eval()
    return model


def save_idm(model: InverseDynamicsModel, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "hidden_layers": [m.out_features for m in model.net if isinstance(m, nn.Linear)][:-1],
    }, path)
    print(f"[idm] saved checkpoint -> {path}", flush=True)


def load_idm(path: str, device: str | torch.device, activation: str = "elu") -> InverseDynamicsModel:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = InverseDynamicsModel(blob["obs_dim"], blob["action_dim"], blob["hidden_layers"], activation)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    print(f"[idm] loaded checkpoint <- {path}", flush=True)
    return model
