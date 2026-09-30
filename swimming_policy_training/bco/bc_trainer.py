"""Behavioral Cloning (BC) pre-training on the IDM's pseudo-labeled expert data (Step 4).

The BC policy is built with rl_games' OWN model_builder, using the EXACT SAME agent yaml
(network architecture, normalize_input/value, sigma init, ...) that scripts/rl_games/
train_ppo.py will use for the PPO fine-tune stage. This is not a "look-alike" MLP: it is the
literal `nn.Module` rl_games' A2CAgent would construct for this task, so the BC-trained
weights can be handed to PPO via its normal `--checkpoint` flag with no format conversion
and no risk of an architecture mismatch (per the task's "do not re-implement PPO" and
"same architecture as PPO policy, do not hardcode" requirements).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .bco_config import BCOConfig


def build_rl_games_model(agent_yaml_path: str, obs_dim: int, action_dim: int) -> nn.Module:
    """Construct rl_games' policy network from its OWN agent yaml -- the identical object
    `rl_games.algos_torch.a2c_continuous.A2CAgent.__init__` builds for this task, obtained via
    the same public API (`model_builder.ModelBuilder`), not a re-implementation."""
    from rl_games.algos_torch import model_builder

    with open(agent_yaml_path) as f:
        agent_cfg = yaml.safe_load(f)
    params = agent_cfg["params"]
    model_wrapper = model_builder.ModelBuilder().load(params)
    build_config = {
        "actions_num": action_dim,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": params["config"].get("value_size", 1),
        "normalize_value": bool(params["config"].get("normalize_value", False)),
        "normalize_input": bool(params["config"].get("normalize_input", False)),
    }
    model = model_wrapper.build(build_config)
    print(f"[bc_trainer] built rl_games model from {agent_yaml_path} "
          f"(obs_dim={obs_dim}, action_dim={action_dim}, "
          f"normalize_input={build_config['normalize_input']})", flush=True)
    return model


@torch.no_grad()
def warm_start_running_mean_std(model: nn.Module, real_obs: torch.Tensor, chunk_size: int = 8192) -> None:
    """Fit the policy's input normalizer on REAL interaction statistics (the Step-1 random
    rollout), then freeze it, BEFORE behavioral cloning ever touches it.

    WHY: the pseudo-labeled expert observations (Step 3) are mostly zero-padded (root
    velocity, joint velocity, target info are all unobservable from a curvature-only video
    sequence -- see pseudo_labeler.py). Letting the normalizer adapt to that data would bias
    it toward "most dims are exactly zero", which is badly wrong for the fully-populated
    observations the policy will actually see once PPO fine-tuning starts stepping the real
    env. The random rollout's observations ARE fully populated/representative, so the
    normalizer is fit on those instead and then frozen (see train_bc: model.train() is called
    every epoch for the MLP weights, but running_mean_std is explicitly re-frozen right after).
    """
    if not hasattr(model, "running_mean_std"):
        return
    rms = model.running_mean_std
    rms.train()
    for i in range(0, real_obs.shape[0], chunk_size):
        rms(real_obs[i:i + chunk_size].to(next(model.parameters()).device))
    rms.eval()
    print(f"[bc_trainer] warm-started running_mean_std from {real_obs.shape[0]} real "
          f"interaction observations, then froze it", flush=True)


def train_bc(
    model: nn.Module,
    pseudo_obs: torch.Tensor,
    pseudo_action: torch.Tensor,
    cfg: BCOConfig,
    device: str | torch.device,
) -> nn.Module:
    """Supervised BC loop: MSE(model(obs)['mus'], pseudo_action), with a held-out validation
    split (default 10%) logged alongside train loss every epoch to catch overfitting."""
    n = pseudo_obs.shape[0]
    assert n == pseudo_action.shape[0] > 0
    assert 0.0 <= cfg.bc_val_split < 1.0
    model.to(device)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
    n_val = int(round(n * cfg.bc_val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = TensorDataset(pseudo_obs[train_idx], pseudo_action[train_idx])
    val_set = TensorDataset(pseudo_obs[val_idx], pseudo_action[val_idx]) if n_val > 0 else None

    loader = DataLoader(train_set, batch_size=min(cfg.bc_batch_size, len(train_set)),
                         shuffle=True, drop_last=False)
    val_loader = (DataLoader(val_set, batch_size=min(cfg.bc_batch_size, len(val_set)), shuffle=False)
                  if val_set is not None else None)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.bc_lr)
    loss_fn = nn.MSELoss()

    print(f"[bc_trainer] BC training on {len(train_set)} train / {n_val} val samples, "
          f"epochs={cfg.bc_epochs} lr={cfg.bc_lr} batch_size={loader.batch_size}", flush=True)
    t0 = time.time()
    for epoch in range(1, cfg.bc_epochs + 1):
        model.train()
        if hasattr(model, "running_mean_std"):
            model.running_mean_std.eval()   # keep it frozen at its Step-1 warm-start stats
        train_loss, n_batches = 0.0, 0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            out = model({"is_train": False, "prev_actions": None, "obs": obs_b})
            pred = out["mus"]
            loss = loss_fn(pred, act_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach())
            n_batches += 1
        train_loss /= max(n_batches, 1)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_loss, n_val_batches = 0.0, 0
            with torch.no_grad():
                for obs_b, act_b in val_loader:
                    obs_b, act_b = obs_b.to(device), act_b.to(device)
                    out = model({"is_train": False, "prev_actions": None, "obs": obs_b})
                    val_loss += float(loss_fn(out["mus"], act_b))
                    n_val_batches += 1
            val_loss /= max(n_val_batches, 1)

        elapsed = time.time() - t0
        eta = elapsed / epoch * (cfg.bc_epochs - epoch)
        print(f"[bc_trainer] epoch {epoch}/{cfg.bc_epochs}  train_mse={train_loss:.6f}  "
              f"val_mse={val_loss:.6f}  elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)
    model.eval()
    return model


def save_rl_games_checkpoint(model: nn.Module, path: str) -> None:
    """Save `model` in the exact dict format rl_games' A2CAgent.restore()/set_full_state_weights
    expects, so `scripts/rl_games/train_ppo.py --checkpoint <path>` (the EXISTING, unmodified
    PPO trainer) loads it directly. `epoch`/`frame` start at 0 (this is a pre-training init, not
    a resumed run); `optimizer` is a freshly-constructed Adam over model.parameters() (rl_games'
    restore() unconditionally reads weights['optimizer'] -- the REAL PPO run builds its own
    optimizer at its own configured lr and overwrites this state on its first step, so only the
    state_dict's SHAPE/param-group structure needs to match, which it does since it is built
    over the identical model instance)."""
    placeholder_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = {
        "model": model.state_dict(),
        "epoch": 0,
        "frame": 0,
        "optimizer": placeholder_opt.state_dict(),
        "last_mean_rewards": -1.0e9,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[bc_trainer] saved rl_games-format BC checkpoint -> {path}", flush=True)
