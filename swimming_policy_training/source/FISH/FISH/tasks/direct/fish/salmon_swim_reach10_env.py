# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""10-target benchmark environments: all-PC PCA vs ZeF-calibrated CPG (see the cfg file for
the protocol). TenTargetMixin owns the attempt state machine; the two env classes differ only
in the controller."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform

from .salmon_amp_tank_env import SalmonAMPTankEnv
from .salmon_swim_amp_env import SalmonSwimAMPEnv
from .salmon_swim_env import SalmonSwimEnv
from .salmon_swim_pca_reach_env import SalmonSwimPCAReachEnv
from .salmon_swim_reach10_cfg import (
    SalmonSwimCPGReach10Cfg,
    SalmonSwimJointAMPReach10Cfg,
    SalmonSwimJointReach10Cfg,
    SalmonSwimPCAReach10Cfg,
)


class TenTargetMixin:
    """Attempt state machine: success / 20 s timeout / early abort -> record 0/1, next target,
    NO reset; episode truncates after exactly n_targets_per_episode attempts."""

    def _init_ten_target(self):
        E, dev = self.num_envs, self.device
        c = self.cfg
        self._n_att = int(c.n_targets_per_episode)
        self._att_timeout_steps = int(round(float(c.attempt_timeout_s)
                                            / (self.cfg.sim.dt * self.cfg.decimation)))
        self._att_idx = torch.zeros(E, dtype=torch.long, device=dev)
        self._att_timer = torch.zeros(E, dtype=torch.long, device=dev)
        self._att_results = torch.full((E, self._n_att), -1, dtype=torch.long, device=dev)
        self._ep_outcomes = deque(maxlen=200)          # per-episode result vectors (ended eps)
        # optional deterministic RELATIVE target sequences (evaluation): (E, n_att, 2)
        self._tgt_seq = None
        sp = str(getattr(c, "target_seq_path", "") or "")
        if sp:
            z = np.load(sp)
            off, dist = z["bearing_off"], z["dist"]
            assert off.shape[0] >= E and off.shape[1] >= self._n_att, \
                f"target seq {off.shape} too small for {E} envs x {self._n_att} attempts"
            self._tgt_seq = (torch.tensor(off[:E, :self._n_att], device=dev, dtype=torch.float32),
                             torch.tensor(dist[:E, :self._n_att], device=dev, dtype=torch.float32))
            print(f"[Reach10] DETERMINISTIC target sequences loaded from {sp}", flush=True)
        print(f"[Reach10] {self._n_att} attempts/episode, timeout {self._att_timeout_steps} steps, "
              f"abort dist {c.attempt_abort_dist} m", flush=True)

    # ---- target sampling: sequence-driven when a sequence is loaded ----
    def _sample_targets(self, env_ids, base_pos, heading_az=None):
        if self._tgt_seq is None:
            return super()._sample_targets(env_ids, base_pos, heading_az)
        ids = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=self.device)
        n = int(len(ids))
        if heading_az is None:
            rq = self.robot.data.root_state_w[ids, 3:7]
            fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
            fwd_b = torch.tensor([fs, 0.0, 0.0], dtype=rq.dtype, device=rq.device).repeat(n, 1)
            fwd_w = math_utils.quat_apply(rq, fwd_b)
            heading_az = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
        att = self._att_idx[ids].clamp(max=self._n_att - 1)
        bearing = heading_az.to(self.device) + self._tgt_seq[0][ids, att]
        dist = self._tgt_seq[1][ids, att]
        tgt = base_pos.clone().to(self.device)
        tgt[:, 0] = base_pos[:, 0] + dist * torch.cos(bearing)
        tgt[:, 1] = base_pos[:, 1] + dist * torch.sin(bearing)
        tgt[:, 2] = self._target_height
        self.target_positions_w[ids] = tgt

    # ---- the attempt state machine replaces the fixed-horizon dones ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        root_state = self.robot.data.root_state_w
        local_pos = root_state[:, 0:3] - self.scene.env_origins
        out_of_bounds = torch.linalg.norm(local_pos, dim=1) > self.cfg.root_position_limit
        joint_blowup = self.joint_vel.abs().max(dim=1).values > self.cfg.joint_velocity_limit
        nonfinite = (~torch.isfinite(root_state).all(dim=1)) \
            | (~torch.isfinite(self.joint_pos).all(dim=1)) | (~torch.isfinite(self.joint_vel).all(dim=1))
        self._fem_blowup = torch.zeros_like(joint_blowup)
        _fvl = getattr(self.cfg, "fem_velocity_limit", None)
        if _fvl is not None and getattr(self, "_soft_view", None) is not None:
            try:
                _nvel = self._soft_view.get_simulation_mesh_nodal_velocities()
                _nvmax = torch.nan_to_num(_nvel, nan=float("inf")).abs().flatten(1).amax(dim=1)
                self._fem_blowup = _nvmax > float(_fvl)
            except Exception:  # noqa: BLE001
                pass
        nonfinite = nonfinite | self._fem_blowup
        _blew = joint_blowup | nonfinite
        self._blew_now = int(_blew.sum())
        _jvm = self.joint_vel.abs().max()
        self._jvel_max = float(_jvm) if torch.isfinite(_jvm) else float("inf")
        terminated = out_of_bounds | joint_blowup | nonfinite

        dist = torch.linalg.norm(self.target_positions_w[:, 0:3] - root_state[:, 0:3], dim=1)
        self._att_timer += 1
        success = dist < self._cur_radius
        timeout = self._att_timer >= self._att_timeout_steps
        abort = dist > float(self.cfg.attempt_abort_dist)
        advance = (success | timeout | abort) & ~terminated
        self._reached_now = success & advance                      # reward's success bonus
        self._targets_reached += self._reached_now.float()

        if advance.any():
            ids = advance.nonzero(as_tuple=False).flatten()
            self._att_results[ids, self._att_idx[ids].clamp(max=self._n_att - 1)] = \
                success[ids].long()
            self._att_idx[ids] += 1
            cont = ids[self._att_idx[ids] < self._n_att]
            if len(cont) > 0:
                self._sample_targets(cont, root_state[cont, 0:3])
                self._att_timer[cont] = 0
                self._prev_distance[cont] = torch.linalg.norm(
                    self.target_positions_w[cont, 0:3] - root_state[cont, 0:3], dim=1)
                if hasattr(self, "_hcos_stale"):
                    self._hcos_stale[cont] = True

        episode_done = self._att_idx >= self._n_att
        # backstop only (protocol ends first: 10 x timeout <= episode_length_s)
        episode_done = episode_done | (self.episode_length_buf >= self.max_episode_length - 1)
        self.extras.setdefault("log", {}).update({
            "reach10/att_idx": self._att_idx.float().mean().detach(),
            "reach10/succ_frac_running": (self._att_results.clamp(min=0).sum(1).float()
                                          / self._att_idx.clamp(min=1).float()).mean().detach(),
            # unbiased: mean over EVERY episode completed since the last read (no capacity cap,
            # no completion-order selection) -- env-count invariant by construction
            "reach10/targets_per_ep": torch.tensor(
                float(getattr(self, "_ep_score_ema", 0.0)) if getattr(self, "_ep_score_n", 0)
                else (float(np.mean([o.sum() for o in self._ep_outcomes])) if self._ep_outcomes else 0.0)),
            "reach10/targets_per_ep_deque_LEGACY": torch.tensor(
                float(np.mean([o.sum() for o in self._ep_outcomes])) if self._ep_outcomes else 0.0),
            "reach10/blowups": _blew.float().sum().detach(),
        })
        self.extras["success"] = self._reached_now.detach().clone()
        return terminated, episode_done

    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        lst = ids.tolist() if torch.is_tensor(ids) else list(ids)
        if hasattr(self, "_att_results"):
            for e in lst:
                res = self._att_results[e]
                if (res >= 0).any():                      # completed(ish) episode -> record
                    _sc = float(res.clamp(min=0).sum())
                    # UNBIASED accumulator: every episode counted exactly once. The
                    # _ep_outcomes deque below is capacity-capped (200) and completion-ordered,
                    # so at >200 envs it silently drops episodes -- and it drops the ones that
                    # ended FIRST (blow-ups / early failures), inflating the logged score.
                    _n = getattr(self, "_ep_score_n", 0) + 1
                    self._ep_score_n = _n
                    _a = 1.0 / min(_n, 200)               # equal weight per EPISODE, 200-ep window
                    self._ep_score_ema = (1.0 - _a) * getattr(self, "_ep_score_ema", _sc) + _a * _sc
                    self._ep_outcomes.append(res.clamp(min=0).cpu().numpy())
            self._att_idx[lst] = 0
            self._att_timer[lst] = 0
            self._att_results[lst] = -1
        super()._reset_idx(env_ids)


class SalmonSwimPCAReach10Env(TenTargetMixin, SalmonSwimPCAReachEnv):
    """All-PC PCA controller on the 10-target protocol (action = 20 coefficient rates)."""

    cfg: SalmonSwimPCAReach10Cfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_ten_target()


class SalmonSwimCPGReach10Env(SalmonSwimPCAReach10Env):
    """ZeF-calibrated CPG on the 10-target protocol.

    kappa(s,t) = kappa_mean(s) + A*E(s)*sin(theta + phi(s)) + b   ->   q = Mdec (kappa - k_rest)
    which collapses to  q(t) = q0 + A*(sin(theta) u1 + cos(theta) u2) + b * ub  with
    u1 = M(E*cos phi), u2 = M(E*sin phi), ub = M 1.  Action = [dA, df, db], all bounds and
    rates measured from the ZeF data (cpg_params.npz). Same decoder, same everything else."""

    cfg: SalmonSwimCPGReach10Cfg

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        nj = int(self._control_joint_ids.shape[0])
        self._num_cpg = 3
        # base proprioception + CPG state (3 params + sin/cos phase) + target dir_b + dist
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + (self._num_cpg + 2) + 4
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_cpg,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_cpg), device=self.device)
        print(f"[CPGReach10] action = [dA, df, db]; obs = {self._obs_dim}", flush=True)

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        pth = str(cfg.cpg_params_path or "") or str(
            Path(__file__).resolve().parents[6] / "outputs/zef_manifold/cpg_params.npz")
        p = np.load(pth)
        E_s = torch.tensor(p["envelope"], device=dev, dtype=torch.float32)      # (20,)
        phi = torch.tensor(p["phase"], device=dev, dtype=torch.float32)         # (20,)
        M = self._Mdec                                                          # (7, 20) shared decoder
        self._cpg_u1 = M @ (E_s * torch.cos(phi))                               # (7,)
        self._cpg_u2 = M @ (E_s * torch.sin(phi))
        self._cpg_ub = M @ torch.ones_like(E_s)
        self._cpg_f0 = float(p["f0"])
        self._cp_lo = torch.tensor([0.0, float(p["f_lo"]), -float(p["b_max"])], device=dev)
        self._cp_hi = torch.tensor([float(p["A_max"]), float(p["f_hi"]), float(p["b_max"])], device=dev)
        self._cdp = torch.tensor([float(p["dA_max"]), float(p["df_max"]), float(p["db_max"])], device=dev)
        self._cpg_p = torch.zeros(self.num_envs, 3, device=dev)
        self._cpg_p[:, 1] = self._cpg_f0
        self._cpg_theta = torch.zeros(self.num_envs, device=dev)
        self._ctrl_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        print(f"[CPGReach10] ZeF-calibrated: A<= {float(p['A_max']):.2f} f in "
              f"[{float(p['f_lo']):.2f},{float(p['f_hi']):.2f}] b<= {float(p['b_max']):.2f} | "
              f"rates {self._cdp.cpu().numpy().round(3).tolist()} | f0={self._cpg_f0:.2f} Hz", flush=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_cpg)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        self._cpg_p = torch.clamp(self._cpg_p + actions * self._cdp, self._cp_lo, self._cp_hi)
        A, f, b = self._cpg_p[:, 0], self._cpg_p[:, 1], self._cpg_p[:, 2]
        self._cpg_theta = torch.remainder(
            self._cpg_theta + 2.0 * math.pi * f * self._ctrl_dt, 2.0 * math.pi)
        q = (self._q0.unsqueeze(0)
             + A.unsqueeze(1) * (torch.sin(self._cpg_theta).unsqueeze(1) * self._cpg_u1.unsqueeze(0)
                                 + torch.cos(self._cpg_theta).unsqueeze(1) * self._cpg_u2.unsqueeze(0))
             + b.unsqueeze(1) * self._cpg_ub.unsqueeze(0))
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
        hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        self._pos_targets = torch.clamp(q, lo, hi)

    def _get_observations(self) -> dict:
        if self._success_markers is not None:
            self._success_markers.visualize(
                translations=self.target_positions_w,
                scales=torch.full((self.num_envs, 3), float(self._cur_radius), device=self.device))
        d = self.robot.data
        root = d.root_state_w
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        heading_dir = math_utils.quat_apply(root[:, 3:7], fwd_b)
        heading_dir = heading_dir / heading_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up_b = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        up_dir = math_utils.quat_apply(root[:, 3:7], up_b)
        jp = d.joint_pos[:, self._control_joint_ids] * self.cfg.obs_scales.joint_pos
        jv = d.joint_vel[:, self._control_joint_ids] * self.cfg.obs_scales.joint_vel
        p_norm = (self._cpg_p - self._cp_lo) / (self._cp_hi - self._cp_lo).clamp(min=1e-6) * 2.0 - 1.0
        delta_w = self.target_positions_w[:, 0:3] - root[:, 0:3]
        delta_b = math_utils.quat_apply_inverse(root[:, 3:7], delta_w)
        dist = delta_b.norm(dim=1, keepdim=True)
        dir_b = delta_b / dist.clamp(min=1e-6)
        obs = torch.cat((
            d.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel,
            d.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel,
            jp, jv, heading_dir, up_dir,
            p_norm, torch.sin(self._cpg_theta).unsqueeze(1), torch.cos(self._cpg_theta).unsqueeze(1),
            dir_b, dist,
        ), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()                   # the EXACT same task reward
        self.extras["log"].update({
            "cpg/A": self._cpg_p[:, 0].mean().detach(),
            "cpg/f_hz": self._cpg_p[:, 1].mean().detach(),
            "cpg/b": self._cpg_p[:, 2].mean().detach(),
            "cpg/p_sat_frac": (((self._cpg_p - self._cp_lo).abs() < 1e-6)
                               | ((self._cpg_p - self._cp_hi).abs() < 1e-6)).float().mean().detach(),
        })
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        if hasattr(self, "_cpg_p"):
            self._cpg_p[idx] = 0.0
            self._cpg_p[idx, 1] = self._cpg_f0
            self._cpg_theta[idx] = 0.0


class SalmonSwimJointReach10Env(TenTargetMixin, SalmonSwimPCAReachEnv):
    """Raw PER-JOINT position-target controller on the 10-target protocol (the IL/BCO
    baseline's action space: no curvature manifold, no CPG parameterization -- RL outputs
    one target per controlled joint directly).

    Built as a SIBLING of SalmonSwimPCAReach10Env (same TenTargetMixin + SalmonSwimPCAReachEnv
    bases, not a further subclass of it) so `_get_rewards` resolves via MRO straight to
    SalmonSwimPCAReachEnv._get_rewards -- the task reward is inherited VERBATIM, not
    re-derived, guaranteeing it is bit-for-bit identical to the PCA/CPG baselines'. That
    reward's logging references self._a/self._a_max (the PCA coefficient state) purely for
    its a1/a2 diagnostics; _build_pca_mapping() still runs in the __init__ chain below (see
    _configure_gym_env_spaces) so those tensors exist and stay harmlessly unused/constant --
    the SAME trick SalmonSwimCPGReach10Env relies on for its own, unrelated CPG state.
    """

    cfg: SalmonSwimJointReach10Cfg

    def _configure_gym_env_spaces(self):
        # Go straight to the BASE env's space config (7-dim [-1,1] action = one raw target per
        # controlled joint, sized off cfg.control_dof_suffix) -- SKIP SalmonSwimPCAEnv's version,
        # which would instead size the action space to pca_num_modes coefficients.
        SalmonSwimEnv._configure_gym_env_spaces(self)
        nj = int(self._control_joint_ids.shape[0])
        # _num_pca is never set by the call above (we bypassed the method that sets it); keep
        # it at the cfg default so _build_pca_mapping() below still succeeds (see docstring).
        self._num_pca = int(self.cfg.pca_num_modes)
        # obs: body lin vel (3) + body ang vel (3) + controlled joint pos (nj) + vel (nj)
        #      + heading dir (3) + up dir (3) + target dir_b (3) + target dist (1)
        # (EXACTLY SalmonSwimPCAReachEnv's obs layout minus the PCA coefficient-state slice --
        # this controller has no such state; the current joint pose/velocity, already present
        # as jp/jv, plays that role instead)
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + 4
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        print(f"[JointReach10] action = {nj} raw joint position targets; obs = {self._obs_dim} dims",
              flush=True)

    def __init__(self, cfg: SalmonSwimJointReach10Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_ten_target()

    # ------------------------------------------------------------------ control
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """action in [-1,1]^nj -> each controlled joint's OWN target, zero-centered onto its
        _soft_joint_limits (a>=0 -> a*hi, a<0 -> a*(-lo), matching this codebase's established
        zero-centered joint-action convention e.g. the ROV-arm task) so action=0 is the
        joint's rest pose and |action|=1 reaches its physical limit exactly -- the SAME
        reachable range the PCA/CPG controllers clamp their own decoded targets to. Do NOT
        reuse cfg.pos_action_scale here: it is a ~25 deg constant left stranded by an earlier
        joint_limit_deg default and is dead code for PCA/CPG (they clamp to _soft_joint_limits
        directly); using it here would silently shrink this baseline's action range below the
        other controllers' and break the fairness of the comparison."""
        actions = actions.view(self.num_envs, self._num_actions)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
        hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        tgt = torch.where(actions >= 0, actions * hi, actions * (-lo))
        self._pos_targets = torch.clamp(tgt, lo, hi)
        # base _apply_action (inherited, unmodified) does the rest: set_joint_position_target +
        # panel hydro wrench + FEM nodal-velocity guard.

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        if self._success_markers is not None:
            self._success_markers.visualize(
                translations=self.target_positions_w,
                scales=torch.full((self.num_envs, 3), float(self._cur_radius), device=self.device))
        d = self.robot.data
        root = d.root_state_w
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        heading_dir = math_utils.quat_apply(root[:, 3:7], fwd_b)
        heading_dir = heading_dir / heading_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up_b = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        up_dir = math_utils.quat_apply(root[:, 3:7], up_b)
        jp = d.joint_pos[:, self._control_joint_ids] * self.cfg.obs_scales.joint_pos
        jv = d.joint_vel[:, self._control_joint_ids] * self.cfg.obs_scales.joint_vel
        delta_w = self.target_positions_w[:, 0:3] - root[:, 0:3]
        delta_b = math_utils.quat_apply_inverse(root[:, 3:7], delta_w)
        dist = delta_b.norm(dim=1, keepdim=True)
        dir_b = delta_b / dist.clamp(min=1e-6)
        obs = torch.cat((
            d.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel,
            d.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel,
            jp, jv, heading_dir, up_dir, dir_b, dist,
        ), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------ reward (BCO ablation)
    def _get_rewards(self) -> torch.Tensor:
        # ALWAYS run the parent's full reward computation first, unconditionally -- it owns
        # _prev_distance/_cum_reaches updates, the reach-history deque, and every diagnostic in
        # extras["log"] (reach10/success bookkeeping itself lives in _get_dones, not here, so
        # success is tracked identically either way). Only the RETURNED value PPO actually
        # optimizes is zeroed when cfg.disable_task_reward is set (the "BCO pure" ablation).
        reward = super()._get_rewards()
        if bool(getattr(self.cfg, "disable_task_reward", False)):
            return torch.zeros_like(reward)
        return reward


class SalmonSwimJointAMPReach10Env(SalmonSwimJointReach10Env):
    """"Joint-space RL + AMP" baseline (see SalmonSwimJointAMPReach10Cfg's docstring for the
    full compatibility rationale). IDENTICAL to SalmonSwimJointReach10Env in every respect
    (action space, observations, Reach10 protocol) except _get_rewards, which adds an AMP
    style-imitation term on top of the unchanged task reward -- the SAME pattern
    SalmonSwimAMPEnv already uses for the (differently-configured) single-target task, and the
    SAME borrow-by-explicit-name mechanism (methods listed here are the ONLY ones any borrowed
    method may call; anything else would silently resolve against this class's own MRO instead
    and raise AttributeError, per SalmonSwimAMPEnv's own comment on this exact risk)."""

    cfg: SalmonSwimJointAMPReach10Cfg

    _amp_feature = SalmonAMPTankEnv._amp_feature
    _bend3d_np = SalmonAMPTankEnv._bend3d_np
    _init_disc = SalmonAMPTankEnv._init_disc
    _amp_reference_contract_check = SalmonAMPTankEnv._amp_reference_contract_check
    _update_disc = SalmonAMPTankEnv._update_disc
    _save_disc = SalmonAMPTankEnv._save_disc
    _load_disc = SalmonAMPTankEnv._load_disc
    _amp_style_reward = SalmonSwimAMPEnv._amp_style_reward

    def __init__(self, cfg: SalmonSwimJointAMPReach10Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        # AMP-style state (verbatim from SalmonSwimAMPEnv.__init__): _init_disc itself (reference
        # load, disc net, replay buffer, feat-stats) is LAZY, called on the first reward step.
        self._K = int(self.cfg.profile_len)
        self._nbend = 2 * self._K
        self._prev_feat = torch.zeros(self.num_envs, self._nbend + 6, device=dev)  # 6 = AMP_N_MOTION
        self._prev_hd = torch.zeros(self.num_envs, device=dev)
        self._prev_pit = torch.zeros(self.num_envs, device=dev)
        self._prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._last_bend = torch.zeros(self.num_envs, self._nbend, device=dev)

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        if bool(getattr(self.cfg, "obs_midline", False)):
            _nb = 2 * int(self.cfg.profile_len)
            obs_dim = int(self.single_observation_space["policy"].shape[0]) + _nb
            self.single_observation_space = gym.spaces.Dict(
                {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,))})
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs)
            print(f"[JointAMPReach10] obs_midline ON: policy obs -> {obs_dim} (+{_nb} FEM bend dims)",
                  flush=True)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if bool(getattr(self.cfg, "obs_midline", False)):
            obs["policy"] = torch.cat([obs["policy"], self._last_bend], dim=-1)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        task_r = super()._get_rewards()                     # unchanged Reach10 joint-task reward
        r_style = self._amp_style_reward()
        w = float(getattr(self.cfg, "w_amp", 0.2))
        reward = task_r + w * r_style
        self.extras.setdefault("log", {}).update({
            "reach/r_style": r_style.mean().detach(),
            "reach/reward_task": task_r.mean().detach(),
            "reach/reward_amp": (w * r_style).mean().detach(),
        })
        return reward
