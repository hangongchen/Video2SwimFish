# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Swim-to-target (the OLD SalmonSwimEnv task) + an AMP style-imitation reward term.

Per the requested design: REUSE the old target task's WHOLE observation and WHOLE reward
(SalmonSwimEnv) unchanged, and ADD ONLY the env-owned AMP discriminator's style reward:

    reward = old_swim_reward  +  w_amp * r_style

The AMP machinery (3D bend feature, discriminator net, reference, disc update) is BORROWED
VERBATIM from the current AMP env (SalmonAMPTankEnv) by method reference -- that file is NOT
modified and NOT structurally depended on beyond these four leaf methods. Trains on the
auto-skeleton fish at dt=1/120 via the cfg (SalmonSwimAMPCfg inherits SalmonAMPAutoskelCfg).

The discriminator INPUT is a transition [Phi(s), Phi(s')] of the 46-dim feature: 40 bend
[L/R + U/D at K=20 body points, body frame, / true 3D length, head at index 0] + 6 motion
[speed_bl, dir_nose, dir_left, dir_up, yaw_body, pitch_body] -- the velocity written in the body's
OWN basis, so head-first / broadside / tail-first are DIFFERENT features. See AMP_MOTION_NAMES in
salmon_amp_tank_env.py for the full rationale (the old 4-channel world-frame block made those three
indistinguishable, which is what the `backward`/`offaxis` reward patches were compensating for).
It judges GAIT STYLE only; the policy's (old swim) observation carries the target for navigation.
The two do not overlap.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

from .salmon_amp_tank_env import AMP_N_MOTION, SalmonAMPTankEnv
from .salmon_swim_amp_cfg import SalmonSwimAMPCfg
from .salmon_swim_env import SalmonSwimEnv


class SalmonSwimAMPEnv(SalmonSwimEnv):
    cfg: SalmonSwimAMPCfg
    # reward-term printout: suppress the base-class print so the AMP style columns join the same line
    _defer_reward_print = True

    # --- borrow the AMP discriminator machinery from the (unmodified) AMP env, verbatim ---
    # These are leaf methods that use only generic self attributes (self.robot, self._soft_view,
    # self.cfg, self.num_envs, self.device, self._K, self._prev_*). SalmonSwimEnv provides all of
    # them (it creates _soft_view when with_deformable=True), so the borrow is safe.
    _amp_feature = SalmonAMPTankEnv._amp_feature
    _bend3d_np = SalmonAMPTankEnv._bend3d_np
    _init_disc = SalmonAMPTankEnv._init_disc
    # NOTE the borrow is by EXPLICIT NAME, so any helper a borrowed method calls must be listed here
    # too -- otherwise it resolves against SalmonSwimEnv and raises AttributeError at runtime.
    _amp_reference_contract_check = SalmonAMPTankEnv._amp_reference_contract_check
    _update_disc = SalmonAMPTankEnv._update_disc
    _save_disc = SalmonAMPTankEnv._save_disc
    _load_disc = SalmonAMPTankEnv._load_disc

    def __init__(self, cfg: SalmonSwimAMPCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        # AMP-style state: the exact subset of the AMP env's _configure_gym_env_spaces that the
        # borrowed feature/disc methods need. _init_disc (reference load, disc net, replay buffer,
        # feat-stats) is LAZY -- called on the first reward step, matching SalmonAMPTankEnv.
        self._K = int(self.cfg.profile_len)
        self._nbend = 2 * self._K
        self._prev_feat = torch.zeros(self.num_envs, self._nbend + AMP_N_MOTION, device=dev)
        self._prev_hd = torch.zeros(self.num_envs, device=dev)                       # prev nose azimuth
        self._prev_pit = torch.zeros(self.num_envs, device=dev)                      # prev nose elevation
        self._prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        # obs_midline: latest 2K bend profile appended to the policy obs (zeros until the first
        # reward step of an episode computes a feature -- and right after each reset).
        self._last_bend = torch.zeros(self.num_envs, self._nbend, device=dev)

    def _configure_gym_env_spaces(self):
        """Widen the policy obs by the 2K-dim AMP bend profile when cfg.obs_midline is set, so the
        policy sees the same midline shape the discriminator judges (base obs is joints-only; the
        bend comes from the FEM mesh, which lags/wobbles around the skeleton)."""
        super()._configure_gym_env_spaces()
        if getattr(self.cfg, "obs_midline", False):
            _nb = 2 * int(self.cfg.profile_len)
            obs_dim = int(self.single_observation_space["policy"].shape[0]) + _nb
            self.single_observation_space = gym.spaces.Dict(
                {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,))})
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs)
            print(f"[SalmonSwimAMP] obs_midline ON: policy obs {obs_dim - _nb} -> {obs_dim} "
                  f"(+{_nb} FEM bend dims)", flush=True)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if getattr(self.cfg, "obs_midline", False):
            # _get_rewards (which refreshes _last_bend) runs BEFORE _get_observations in the step
            # order, so this bend is current for stepping envs and zeroed for just-reset ones.
            obs["policy"] = torch.cat([obs["policy"], self._last_bend], dim=-1)
        return obs

    # --- AMP style reward (LSGAN) + replay-buffer push + periodic disc update + prev-state advance ---
    def _amp_style_reward(self) -> torch.Tensor:
        if not hasattr(self, "_disc"):
            self._init_disc()
        dev = self.device
        c = self.cfg
        feat = self._amp_feature()                                              # (E, 2K+6); advances _prev_hd/pit
        self._last_bend = feat[:, :self._nbend].detach()                        # obs_midline cache (raw, unnormalized)
        featn = (feat - self._feat_mean) / self._feat_std
        trans = torch.cat([(self._prev_feat - self._feat_mean) / self._feat_std, featn], dim=1)  # (E, 2*feat_dim)
        with torch.no_grad():
            score = self._disc(trans).squeeze(-1)
            rs = (1.0 - 0.25 * (score - 1.0) ** 2).clamp(0.0, 1.0)              # LSGAN style reward, in [0,1]
        r_style = torch.nan_to_num(torch.where(self._prev_valid, rs, torch.zeros_like(rs)))
        # push valid policy transitions to the replay buffer (the disc's "fake" distribution)
        valid = self._prev_valid
        if valid.any():
            add = trans[valid].detach()
            m = add.shape[0]
            idx = (torch.arange(m, device=dev) + self._buf_ptr) % self._buf.shape[0]
            self._buf[idx] = add
            self._buf_ptr = int((self._buf_ptr + m) % self._buf.shape[0])
            self._buf_n = min(self._buf_n + m, self._buf.shape[0])
        self._amp_step += 1
        # skip disc training under inference_mode (play/eval): the R1 grad-penalty needs autograd
        if (self._buf_n >= int(c.disc_batch) and self._amp_step % int(c.disc_update_every) == 0
                and not torch.is_inference_mode_enabled()):
            self._update_disc()
        # advance prev-state
        self._prev_feat = feat.detach()
        self._prev_valid = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        return r_style

    def _get_rewards(self) -> torch.Tensor:
        task_r = super()._get_rewards()                                         # OLD swim reward (whole, unchanged)
        r_style = self._amp_style_reward()
        w = float(getattr(self.cfg, "w_amp", 0.5))
        reward = task_r + w * r_style
        self.extras.setdefault("log", {}).update({
            "r_style": r_style.mean().detach(),
            "reward_task": task_r.mean().detach(),
            "reward_amp": (w * r_style).mean().detach(),
            "disc_updates": torch.tensor(float(getattr(self, "_disc_updates", 0))),
        })
        if getattr(self.cfg, "print_reward_terms", False):
            self._print_reward_terms(extra={"r_style": r_style, "amp(w*)": w * r_style, "TOTAL": reward})
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)                                            # OLD swim reset (target, FEM, root)
        idx = slice(None) if env_ids is None else env_ids
        # AMP prev-state: force the first post-reset step to be a non-transition (r_style=0, no buffer push)
        self._prev_valid[idx] = False
        self._prev_hd[idx] = 0.0
        self._prev_pit[idx] = 0.0
        # obs_midline: the cached bend belongs to the PRE-reset pose; zero it so the first obs of the
        # new episode doesn't carry the old episode's body shape.
        if hasattr(self, "_last_bend"):
            self._last_bend[idx] = 0.0
