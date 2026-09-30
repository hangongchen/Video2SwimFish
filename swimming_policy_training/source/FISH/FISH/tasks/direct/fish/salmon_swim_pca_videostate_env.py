# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-video BLM imitation environment (see salmon_swim_pca_videostate_cfg.py). Identical
to SalmonSwimPCAEnv (same action space, same straight-swim reward/dones) except for
_get_observations, which is REWRITTEN to emit only the quantities a top-view video can
supply: [x, y, vx, vy, psi, a/a_max] -- world position/heading-derived, RE-ROOTED at each
episode's own reset (position=0, heading=0 at reset), matching the convention
build_video_blm_bc_dataset.py uses to re-root each real-ZeF run at its own first frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from .salmon_swim_pca_env import SalmonSwimPCAEnv
from .salmon_swim_pca_videostate_cfg import SalmonSwimPCAVideoStateCfg


class SalmonSwimPCAVideoStateEnv(SalmonSwimPCAEnv):
    cfg: SalmonSwimPCAVideoStateCfg

    # ------------------------------------------------------------------ spaces
    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        # [x, y, vx, vy, psi] (5) + a/a_max (K) -- REPLACES the parent's 46-dim obs entirely
        self._obs_dim = 5 + self._num_pca
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        print(f"[PCAVideoState] obs REPLACED with video-plausible state -> {self._obs_dim} dims "
              f"([x,y,vx,vy,psi] + {self._num_pca} PCA coeffs)", flush=True)

    def __init__(self, cfg: SalmonSwimPCAVideoStateCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        self._ep_p0_xy = torch.zeros(self.num_envs, 2, device=dev)
        self._ep_psi0 = torch.zeros(self.num_envs, device=dev)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        root = self.robot.data.root_state_w[ids]
        self._ep_p0_xy[ids] = root[:, 0:2]
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(len(ids), 1)
        fwd_w = math_utils.quat_apply(root[:, 3:7], fwd_b)
        self._ep_psi0[ids] = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        d = self.robot.data
        root = d.root_state_w
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        fwd_w = math_utils.quat_apply(root[:, 3:7], fwd_b)
        psi = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
        psi_rel = torch.atan2(torch.sin(psi - self._ep_psi0), torch.cos(psi - self._ep_psi0))

        c0, s0 = torch.cos(-self._ep_psi0), torch.sin(-self._ep_psi0)
        d_xy = root[:, 0:2] - self._ep_p0_xy
        x = (c0 * d_xy[:, 0] - s0 * d_xy[:, 1]) / self._body_length
        y = (s0 * d_xy[:, 0] + c0 * d_xy[:, 1]) / self._body_length

        v_w = root[:, 7:9]                                   # world-frame linear velocity, xy
        vx = (c0 * v_w[:, 0] - s0 * v_w[:, 1]) / self._body_length
        vy = (s0 * v_w[:, 0] + c0 * v_w[:, 1]) / self._body_length

        obs = torch.cat([
            x.unsqueeze(1), y.unsqueeze(1), vx.unsqueeze(1), vy.unsqueeze(1), psi_rel.unsqueeze(1),
            self._a / self._a_max,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}
