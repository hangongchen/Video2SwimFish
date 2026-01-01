# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark ROV catch task: SalmonROVChaseEnv with the prey driven by a per-fish BLM+BC
policy (checkpoints/benchmark/blm_il/<tag>/bc_policy.pt, trained from video only) instead of
the frozen Misty reach10 RL policy.

The BC policy was trained on the FreeSwim-VideoState observation ([x, y, vx, vy, psi]
re-rooted at reset + a/a_max), so the fish observation is rebuilt with
SalmonSwimPCAVideoStateEnv._get_observations; the ROV agent, reward, success logic and
basket geometry are the parent's, untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .salmon_rov_chase_env import SalmonROVChaseCfg, SalmonROVChaseEnv
from .salmon_swim_pca_videostate_env import SalmonSwimPCAVideoStateEnv

_REPO = Path(__file__).resolve().parents[6]


@configclass
class BenchROVCatchCfg(SalmonROVChaseCfg):
    # per-fish BLM+BC prey policy (rl_games-format state dict written by bco.bc_trainer)
    fish_ckpt = ""
    fish_policy_yaml = str(_REPO / "source/FISH/FISH/tasks/direct/fish/agents/rl_games_ppo_pca_videostate_cfg.yaml")
    pca_num_modes = 20          # must match SalmonSwimPCAVideoStateCfg (the BC policy's action dim)
    # prey drift/launch caps as in the video-state free-swim eval regime
    body_speed_cap_bl = 1.5
    body_yaw_cap_rad = 8.0


class _BCFishPlayer:
    """Duck-types the rl_games player the parent steps: .model(dict)->{'mus'}, ._preproc_obs, .states.
    _preproc_obs IGNORES the reach10 obs the parent computes and rebuilds the video-state obs."""

    def __init__(self, env, model):
        self.env, self._model, self.states = env, model, None

    def _preproc_obs(self, _obs):
        return SalmonSwimPCAVideoStateEnv._get_observations(self.env)["policy"]

    def model(self, d):
        return self._model(d)


class BenchROVCatchEnv(SalmonROVChaseEnv):
    cfg: BenchROVCatchCfg

    # ------------------------------------------------ prey policy = per-fish BLM+BC
    def _load_fish_player(self):
        import sys
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from bco.bc_trainer import build_rl_games_model  # noqa: E402
        ck = Path(self.cfg.fish_ckpt)
        if not ck.exists():
            raise FileNotFoundError(f"[BenchROVCatch] fish BC policy missing: {ck}")
        obs_dim = 5 + self._num_pca
        model = build_rl_games_model(self.cfg.fish_policy_yaml, obs_dim, int(self._num_pca))
        sd = torch.load(str(ck), map_location=self.device)
        model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
        model.to(self.device).eval()
        dev = self.device
        self._ep_p0_xy = torch.zeros(self.num_envs, 2, device=dev)
        self._ep_psi0 = torch.zeros(self.num_envs, device=dev)
        print(f"[BenchROVCatch] prey = BLM+BC policy {ck} (obs {obs_dim}, act {self._num_pca})", flush=True)
        return _BCFishPlayer(self, model)

    # ------------------------------------------------ video-state re-rooting at reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_ep_p0_xy"):
            return
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        root = self.robot.data.root_state_w[ids]
        self._ep_p0_xy[ids] = root[:, 0:2]
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(len(ids), 1)
        fwd_w = math_utils.quat_apply(root[:, 3:7], fwd_b)
        self._ep_psi0[ids] = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
