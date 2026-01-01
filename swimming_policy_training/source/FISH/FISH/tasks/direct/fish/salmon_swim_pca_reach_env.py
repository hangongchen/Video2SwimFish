# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PCA-manifold RANDOM-TARGET REACHING environment (see salmon_swim_pca_reach_cfg.py).

Layers the reaching task on top of SalmonSwimPCAEnv while REUSING the base SalmonSwimEnv
machinery wherever it already exists:
  - target sampling (front cone + normal distance): SalmonSwimEnv._sample_targets, unchanged
  - reach -> count + resample WITHOUT reset:        SalmonSwimEnv._get_dones (multi_target),
                                                    re-aliased verbatim below
  - target marker:                                  the base dynamic success-sphere system
  - action space:                                   the PCA bounded-rate coefficients (parent)

New here: target-relative observations (body frame), the progress+success reward with
per-component logging, reach-statistics logging, and the env-1 point-cloud frame recorder.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils

from .salmon_swim_env import SalmonSwimEnv
from .salmon_swim_pca_env import SalmonSwimPCAEnv
from .salmon_swim_pca_reach_cfg import SalmonSwimPCAReachCfg


class SalmonSwimPCAReachEnv(SalmonSwimPCAEnv):
    cfg: SalmonSwimPCAReachCfg

    # reach -> count + resample (multi_target) + blow-up/timeout terminations: the ORIGINAL
    # base-class implementation does exactly what this task needs -- reuse it verbatim
    # (the parent PCA env had replaced it with a target-free version).
    _get_dones = SalmonSwimEnv._get_dones

    # ------------------------------------------------------------------ spaces
    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        # + body-frame unit direction to target (3) + distance (1)
        self._obs_dim += 4
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        print(f"[SalmonSwimPCAReach] obs widened to {self._obs_dim} (+3 target dir_b, +1 dist)",
              flush=True)

    def __init__(self, cfg: SalmonSwimPCAReachCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # rolling reach statistics (window of recently ENDED episodes)
        self._reach_hist = deque(maxlen=200)          # reaches per ended episode
        self._cum_reaches = 0
        # env-1 frame recorder state
        self._viz_e = int(cfg.reach_frame_env)
        self._viz_ep_count = 0
        self._viz_active = False
        self._viz_trace: list = []
        self._viz_targets: list = []
        self._viz_reach_pts: list = []
        d = cfg.reach_frame_dir or str(
            Path(__file__).resolve().parents[6] / "outputs" / "pca_reach_frames"
            / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self._viz_dir = Path(d)
        self._viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SalmonSwimPCAReach] env-{self._viz_e} frames every "
              f"{cfg.reach_frame_every_n_episodes} episodes -> {self._viz_dir}", flush=True)

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # keep the success spheres aimed at the live targets (base-class behavior the PCA
        # parent dropped; visualize() is ~free headless)
        if self._success_markers is not None:
            self._success_markers.visualize(
                translations=self.target_positions_w,
                scales=torch.full((self.num_envs, 3), float(self._cur_radius), device=self.device))
        root = self.robot.data.root_state_w
        delta_w = self.target_positions_w[:, 0:3] - root[:, 0:3]
        delta_b = math_utils.quat_apply_inverse(root[:, 3:7], delta_w)   # target in BODY frame
        dist = delta_b.norm(dim=1, keepdim=True)
        dir_b = delta_b / dist.clamp(min=1e-6)
        obs["policy"] = torch.cat([obs["policy"], dir_b, dist], dim=-1)
        return {"policy": torch.nan_to_num(obs["policy"], nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------ reward
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d = self.robot.data
        BL = self._body_length
        delta_w = self.target_positions_w[:, 0:3] - d.root_state_w[:, 0:3]
        dist = torch.linalg.norm(delta_w, dim=1)
        tgt_dir_w = delta_w / dist.clamp(min=1e-6).unsqueeze(-1)
        # nose direction + heading alignment + target-directed velocity (always computed --
        # logged as the anti-gliding diagnostics; only PAID in the head-first variant)
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        nose_w = math_utils.quat_apply(d.root_state_w[:, 3:7], fwd_b)
        nose_w = nose_w / nose_w.norm(dim=1, keepdim=True).clamp(min=1e-6)
        heading_cos = (nose_w * tgt_dir_w).sum(-1)
        v_target_bl = (d.root_state_w[:, 7:10] * tgt_dir_w).sum(-1) / BL
        # progress = previous_distance - current_distance (in BL so scales match the swim task).
        # NOTE _get_dones ran FIRST this step: on a reach it already resampled the target AND
        # reset _prev_distance to the NEW target's distance, so the target jump contributes
        # exactly zero progress here; the reach itself pays through the success bonus below.
        progress_bl = (self._prev_distance - dist) / BL
        if getattr(c, "heading_gate", False):
            # HEAD-FIRST: positive progress pays in proportion to nose alignment (broadside ~ 0);
            # negative progress stays UNGATED so drifting away always costs full.
            gate = heading_cos.clamp(0.0, 1.0)
            r_prog = c.w_progress * (progress_bl.clamp(min=0.0) * gate + progress_bl.clamp(max=0.0))
        else:
            r_prog = c.w_progress * progress_bl
        # alignment pays ONLY while actually closing (aligned-but-frozen and broadside-but-closing
        # both earn ~0); bounded by the 0.5 BL/s clamp.
        r_align = float(getattr(c, "w_align", 0.0)) * heading_cos.clamp(min=0.0) \
            * v_target_bl.clamp(0.0, 0.5)
        r_succ = c.w_success * self._reached_now.float()                 # latched by _get_dones
        if getattr(c, "success_align_gate", False):
            # ESCALATION (measured necessity): with only progress gated, the flat +25 success
            # bonus still fully finances sideways bump-reaches (v2 @150 epochs: reaches rose
            # 1.25->1.62 while final-approach heading error sat at ~105 deg). Scale the bonus by
            # nose alignment AT THE REACH MOMENT: head-first reach pays 25, broadside pays ~0.
            r_succ = r_succ * heading_cos.clamp(0.0, 1.0)
        jv = d.joint_vel[:, self._control_joint_ids]
        energy = (jv ** 2).mean(dim=1)
        r_energy = -c.w_energy * energy
        # OPT-IN anti-spin shaping (both default 0.0 -> reward unchanged): the deterministic
        # rear-target failure mode is a coefficient-saturated held C-curl that spins the body
        # (|yaw| 9-29 rad/s vs <~3 in healthy turns). Charge (a) yaw rate beyond the biological
        # range and (b) sitting at the coefficient bounds, so the policy prefers sequenced
        # in-manifold turns over the bound-locked spin.
        # POTENTIAL-BASED HEADING SHAPING (Ng et al. 1999): r = w * [cos H(t) - cos H(t-1)].
        # Dense payment for ROTATING TOWARD the target -- the missing gradient on behind-targets,
        # where the heading gates silence every other positive term until |H| < 90 deg. Any
        # closed loop nets zero, so spinning earns nothing and the optimal policy is unchanged.
        r_head = torch.zeros_like(energy)
        w_hs = float(getattr(c, "w_heading_shape", 0.0))
        if w_hs > 0.0:
            if not hasattr(self, "_prev_hcos"):
                self._prev_hcos = heading_cos.detach().clone()
                self._hcos_stale = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            fresh = ~self._hcos_stale
            r_head[fresh] = w_hs * (heading_cos - self._prev_hcos)[fresh]
            self._prev_hcos = heading_cos.detach().clone()
            self._hcos_stale[:] = False
        r_yaw = torch.zeros_like(energy)
        w_yaw = float(getattr(c, "w_yaw_penalty", 0.0))
        if w_yaw > 0.0:
            yaw_abs = d.root_ang_vel_b[:, 2].abs()
            r_yaw = -w_yaw * (yaw_abs - float(getattr(c, "yaw_penalty_thresh", 3.0))).clamp(min=0.0)
        r_sat = torch.zeros_like(energy)
        w_sat = float(getattr(c, "w_sat_penalty", 0.0))
        if w_sat > 0.0 and hasattr(self, "_a"):
            sat = ((self._a.abs() / self._a_max) - 0.9).clamp(min=0.0).mean(dim=1) / 0.1
            r_sat = -w_sat * sat
        reward = torch.nan_to_num(r_prog + r_align + r_succ + r_energy + r_yaw + r_sat + r_head)
        self._prev_distance = dist.detach()
        self._episode_rewards += reward
        self._cum_reaches += int(self._reached_now.sum())

        v_b = d.root_lin_vel_b
        self.extras.setdefault("log", {}).update({
            # reward components (separately, so reward hacking is visible)
            "reach/rew_progress": r_prog.mean().detach(),
            "reach/rew_align": r_align.mean().detach() if torch.is_tensor(r_align)
            else torch.tensor(0.0),
            "reach/rew_success": r_succ.mean().detach(),
            "reach/rew_energy": r_energy.mean().detach(),
            "reach/rew_total": reward.mean().detach(),
            # anti-gliding diagnostics
            "reach/heading_cos": heading_cos.mean().detach(),
            "reach/v_target_bl": v_target_bl.mean().detach(),
            "reach/final_heading_err_deg": torch.rad2deg(torch.arccos(
                heading_cos[dist < 0.5].clamp(-1.0, 1.0))).mean().detach()
            if (dist < 0.5).any() else torch.tensor(float("nan")),
            "reach/final_heading_cos": heading_cos[dist < 0.5].mean().detach()
            if (dist < 0.5).any() else torch.tensor(float("nan")),
            # target-reaching metrics
            "reach/target_distance": dist.mean().detach(),
            "reach/reach_events_now": self._reached_now.float().sum().detach(),
            "reach/cum_reaches": torch.tensor(float(self._cum_reaches)),
            "reach/reaches_per_episode": torch.tensor(
                float(np.mean(self._reach_hist)) if self._reach_hist else 0.0),
            # locomotion state
            "reach/v_fwd_bl": (self._fwd_sign * v_b[:, 0] / BL).mean().detach(),
            "reach/v_lat_bl": (v_b[:, 1:3].norm(dim=1) / BL).mean().detach(),
            "reach/yaw_rate": d.root_ang_vel_b[:, 1].abs().mean().detach(),
            # PCA coefficient usage
            "reach/a1_mean": self._a[:, 0].mean().detach(),
            "reach/a2_mean": self._a[:, 1].mean().detach(),
            "reach/a1_abs": self._a[:, 0].abs().mean().detach(),
            "reach/a2_abs": self._a[:, 1].abs().mean().detach(),
            "reach/a_sat_frac": (self._a.abs() >= 0.999 * self._a_max).float().mean().detach(),
            "reach/energy_jv2": energy.mean().detach(),
        })
        self._maybe_record_viz(dist, r_prog, r_succ, reward)
        return reward

    # ------------------------------------------------------------------ reset bookkeeping
    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        # capture reaches-per-episode BEFORE the base reset zeroes _targets_reached
        if hasattr(self, "_reach_hist"):
            for e in (ids.tolist() if torch.is_tensor(ids) else list(ids)):
                self._reach_hist.append(float(self._targets_reached[e]))
            if self._viz_e in (ids.tolist() if torch.is_tensor(ids) else list(ids)):
                self._on_viz_episode_end()
        super()._reset_idx(env_ids)

    # ------------------------------------------------------------------ env-1 frame recorder
    def _on_viz_episode_end(self):
        if self._viz_active and self._viz_trace:
            self._save_viz_frame(final=True)
        self._viz_ep_count += 1
        self._viz_active = (self._viz_ep_count % int(self.cfg.reach_frame_every_n_episodes) == 0)
        self._viz_trace, self._viz_targets, self._viz_reach_pts = [], [], []

    def _maybe_record_viz(self, dist, r_prog, r_succ, reward):
        """Cheap per-step bookkeeping for env-1; a PNG only every reach_frame_every_steps
        steps of every Nth episode (matplotlib Agg, ~50 ms -- negligible vs training)."""
        if not self._viz_active:
            return
        e = self._viz_e
        p = self.robot.data.root_state_w[e, 0:3].cpu().numpy()
        tgt = self.target_positions_w[e].cpu().numpy()
        self._viz_trace.append(p.copy())
        if not self._viz_targets or np.linalg.norm(tgt - self._viz_targets[-1]) > 1e-6:
            self._viz_targets.append(tgt.copy())
        if bool(self._reached_now[e]):
            self._viz_reach_pts.append(p.copy())
        step = len(self._viz_trace)
        if step % int(self.cfg.reach_frame_every_steps) == 0:
            self._viz_last_info = {
                "rew_total": float(reward[e]), "rew_prog": float(r_prog[e]),
                "rew_succ": float(r_succ[e]), "dist": float(dist[e]),
                "reaches": int(self._targets_reached[e]),
                "a1": float(self._a[e, 0]), "a2": float(self._a[e, 1]),
            }
            self._save_viz_frame(final=False)

    def _save_viz_frame(self, final: bool):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            e = self._viz_e
            info = getattr(self, "_viz_last_info", {})
            fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
            # FEM point cloud (the actual soft body, world frame)
            if getattr(self, "_soft_view", None) is not None:
                pts = self._soft_view.get_simulation_mesh_nodal_positions()[e].cpu().numpy()
                ax.scatter(pts[:, 0], pts[:, 1], s=3, c="#4a90c4", alpha=0.7, label="FEM point cloud")
            tr = np.array(self._viz_trace)
            ax.plot(tr[:, 0], tr[:, 1], "-", color="gray", lw=1, label="root path")
            for i, t in enumerate(self._viz_targets):
                is_cur = i == len(self._viz_targets) - 1
                ax.plot(t[0], t[1], "*", ms=18 if is_cur else 10,
                        color="red" if is_cur else "#c08080",
                        label="TARGET (current)" if is_cur else ("past targets" if i == 0 else None))
                if is_cur:
                    ax.add_patch(plt.Circle((t[0], t[1]), float(self._cur_radius),
                                            fill=False, color="red", ls="--", lw=1))
            for i, rp in enumerate(self._viz_reach_pts):
                ax.plot(rp[0], rp[1], "o", ms=9, mfc="none", mec="green", mew=2,
                        label="reach event" if i == 0 else None)
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper right")
            txt = (f"ep {self._viz_ep_count} step {len(self._viz_trace)}\n"
                   f"rew_total={info.get('rew_total', 0):+.3f}  prog={info.get('rew_prog', 0):+.3f}  "
                   f"succ={info.get('rew_succ', 0):+.1f}\n"
                   f"dist={info.get('dist', 0):.3f} m  reaches={info.get('reaches', 0)}\n"
                   f"a1={info.get('a1', 0):+.2f}  a2={info.get('a2', 0):+.2f}")
            ax.set_title(txt, fontsize=8)
            tag = "final" if final else f"s{len(self._viz_trace):05d}"
            fig.tight_layout()
            fig.savefig(self._viz_dir / f"ep{self._viz_ep_count:05d}_{tag}.png")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 -- recording must never kill training
            print(f"[SalmonSwimPCAReach] viz frame failed: {exc}", flush=True)
