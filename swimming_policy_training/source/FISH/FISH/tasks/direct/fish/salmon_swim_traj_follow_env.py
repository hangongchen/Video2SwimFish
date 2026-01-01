# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-following environments: given a real ZeF ground-truth path Gamma(s) (arc-length
parameterized, s in BL), the agent must track it via PHASE-BASED lookahead -- at each step the
fish's own position is projected onto its nearest point on Gamma, and the reward/obs target is
the point delta (BL) further ahead in arc length, NOT a fixed-time index. This means a fish that
falls behind is NOT penalized for "running out of time" against a clock -- its own phase stays
wherever it actually is, and the episode ends when its OWN projected phase reaches the end of
the path (plus a safety time_out backstop).

TrajFollowMixin owns: loading the real ZeF reference, sampling+planting a per-env GT path (with
its own arc-length/tangent/local-speed parameterization) at each reset, the phase projection +
arc-length lookahead interpolation, the 4-component reward (track/heading/speed/energy), the
discrete Fréchet distance computed at episode end (raw temporal order, NOT arc-length
resampled), and the shared trajectory-observation component. The two concrete env classes below
differ only in the controller (PCA coefficients vs raw joints).
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import collections
import math
from pathlib import Path
import numpy as np
import torch

import isaaclab.utils.math as math_utils

from .salmon_swim_env import SalmonSwimEnv
from .salmon_swim_pca_env import SalmonSwimPCAEnv
from .salmon_swim_traj_follow_cfg import (SalmonSwimCPGTrajFollowCfg, SalmonSwimJointAMPTrajFollowCfg, SalmonSwimJointTrajFollowCfg,
                                          SalmonSwimPCATrajFollowCfg)
from .salmon_amp_tank_env import SalmonAMPTankEnv
from .salmon_swim_amp_env import SalmonSwimAMPEnv


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Frechet distance (Eiter & Mannila 1994), O(mn) DP, iterative (no recursion
    depth risk). P (m,d), Q (n,d) -- raw temporal order, NOT arc-length resampled (the caller
    is responsible for that; this function only ever sees whatever it's given)."""
    m, n = len(P), len(Q)
    if m == 0 or n == 0:
        return float("nan")
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    ca = np.zeros((m, n), dtype=np.float64)
    ca[0, 0] = D[0, 0]
    for i in range(1, m):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
    for j in range(1, n):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    for i in range(1, m):
        row_prev, row_cur = ca[i - 1], ca[i]
        d_row = D[i]
        for j in range(1, n):
            row_cur[j] = max(min(row_prev[j], row_prev[j - 1], row_cur[j - 1]), d_row[j])
    return float(ca[m - 1, n - 1])


from . import biofidelity_metrics as bm  # noqa: E402


def bm_kappa(bp_e, bq_e, geo):
    """one env, one step -> kappa_bl (20,) or None (curvature_utils estimator, same as the video reference)"""
    pr = bm.cu.kappa_from_bones(bp_e.cpu().numpy(), bq_e.cpu().numpy(), geo, smooth_tol_m=2e-3)
    return None if pr is None else np.asarray(pr["kappa_bl"], dtype=np.float32)


class TrajFollowMixin:
    """Real-ZeF phase-based trajectory tracking, shared by the PCA and Joint controllers."""

    def _init_traj_follow(self):
        dev = self.device
        Z = np.load(self.cfg.zef_traj_path)
        p_bl = torch.tensor(Z["p_bl"], dtype=torch.float64, device=dev)          # (T_zef, 2)
        psi_raw = torch.tensor(Z["psi"], dtype=torch.float64, device=dev)        # (T_zef,)
        runs = Z["runs"]
        fps = float(Z["fps"])
        self._zef_runs = [(int(a), int(b)) for a, b in runs]
        # ---- HEAD/TAIL DISAMBIGUATION OF THE REFERENCE HEADING (2026-09-24) ----
        # `psi` is the body heading fitted to the video midline, but the midline extractor picks
        # which END of the midline is the head ONCE PER RUN and gets it backwards about half the
        # time (measured on catfish_fish001: 11 of 21 runs offset by ~180 deg from the run's own
        # travel direction, and TIGHTLY so -- one run sits at -175.1 +/- 7.6 deg, which is a sign
        # convention, not a fish swimming backwards).  psi is temporally smooth inside a run
        # (0 adjacent-frame jumps > 90 deg), so the error is a single global flip per run.
        #
        # This matters for initialisation: _sample_trajectory plants the reference path by
        # de-rotating it with psi[window start] and re-rotating into the sim fish's heading, which
        # is exactly what reproduces the REAL fish's initial pose relative to its own path.  With a
        # flipped run that construction spawns the fish facing 180 deg AWAY from the path.
        #
        # Fix: per run, compare psi against the run's own direction of travel (over the faster
        # frames, where the travel direction is meaningful) and flip the whole run by pi when the
        # circular mean disagrees by more than 90 deg.  Runs that barely move carry no reliable
        # travel direction and are left untouched.
        psi_raw = self._disambiguate_reference_heading(p_bl, psi_raw, self._zef_runs)
        self._zef_p_bl = p_bl
        self._zef_psi = psi_raw
        self._sim_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        self._zef_stride = max(1, int(round(fps * self._sim_dt)))
        win_frames = int(round(self.cfg.traj_window_s * fps))
        self._traj_win_frames = win_frames
        self._traj_T_max = win_frames // self._zef_stride + 1
        T = self._traj_T_max
        self._traj_path = torch.zeros(self.num_envs, T, 3, device=dev)
        self._traj_s_bl = torch.zeros(self.num_envs, T, device=dev)          # cumulative arc length (BL)
        self._traj_tangent = torch.zeros(self.num_envs, T, 2, device=dev)     # unit tangent (world XY)
        self._traj_speed_bl = torch.zeros(self.num_envs, T, device=dev)      # local ref speed (BL/s)
        self._traj_len = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._fish_hist = torch.zeros(self.num_envs, T, 3, device=dev)       # actual fish path, this episode
        self._ep_p0 = torch.zeros(self.num_envs, 3, device=dev)
        self._ep_psi0 = torch.zeros(self.num_envs, device=dev)
        # cached per-step phase-projection results (computed once in _get_dones, reused by
        # _get_rewards/_get_observations -- Isaac Lab calls _get_dones first each step)
        self._cur_p_star = torch.zeros(self.num_envs, 3, device=dev)
        self._cur_tangent_star = torch.zeros(self.num_envs, 2, device=dev)
        self._cur_v_star = torch.zeros(self.num_envs, device=dev)
        self._cur_traj_reached_end = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._last_frechet = torch.full((self.num_envs,), float("nan"), device=dev)
        self._last_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        # ---- online biofidelity metrics (benchmark): fish's OWN video reference, subset of envs ----
        self._bio = None
        self._blowup_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._ep_blowups = collections.deque(maxlen=256)     # 1 = episode ended in a blowup
        ref_curv = getattr(self.cfg, "ref_curvature_path", None)
        if ref_curv and Path(ref_curv).exists():
            self._bio_ref = bm.RefStats(ref_curv, self.cfg.zef_traj_path)
            n_eval = int(min(getattr(self.cfg, "eval_metric_envs", 16), self.num_envs))
            self._bio_ids = np.arange(n_eval)
            self._bio = bm.EpisodeMetrics(self._bio_ids, T, self._sim_dt, self._bio_ref)
            calib = np.load(self.cfg.pca_calib_path, allow_pickle=True)
            head_sign = int(calib["head_sign"]) if "head_sign" in calib.files else 1
            self._bio_geo = bm.cu.rest_geometry(self.cfg.panel_hydro_path, head_sign=head_sign)
            print(f"[TrajFollow] biofidelity metrics ON: {n_eval} envs, ref clean frames="
                  f"{len(self._bio_ref.kappa_clean)}, ref f={self._bio_ref.dominant_freq_hz:.2f} Hz, "
                  f"ref speed={self._bio_ref.speed_bl_s:.2f} BL/s", flush=True)
        print(f"[TrajFollow] real ZeF reference loaded: {len(self._zef_runs)} runs, "
              f"window={self.cfg.traj_window_s:.1f}s ({win_frames} ZeF frames @ {fps:.0f}fps, "
              f"stride {self._zef_stride} -> up to {T} sim steps), "
              f"phase-based delta={self.cfg.traj_delta_bl:.2f} BL, "
              f"lookahead K={int(self.cfg.traj_lookahead_k)}", flush=True)

    @staticmethod
    def _disambiguate_reference_heading(p_bl, psi, runs, min_net_bl: float = 0.25,
                                        min_moving_frames: int = 10):
        """Flip whole runs of `psi` by pi so the reference heading points HEAD-forward.

        See the call site for why this is needed. Returns a corrected copy of `psi`; logs the
        per-run decision so a bad reference file is visible in the training log instead of
        silently spawning half the episodes backwards.
        """
        psi = psi.clone()
        flipped, skipped = [], []
        for a, b in runs:
            if b - a < 3:
                continue
            seg = p_bl[a:b]
            dp = seg[1:] - seg[:-1]
            step = torch.linalg.norm(dp, dim=1)
            if step.numel() < min_moving_frames:
                continue
            thr = torch.quantile(step, 0.6)
            m = step > thr
            if int(m.sum()) < min_moving_frames:
                continue
            net = float(torch.linalg.norm(seg[-1] - seg[0]))
            if net < min_net_bl:                      # hovering: travel direction is noise
                skipped.append((a, b, net))
                continue
            trav = torch.atan2(dp[m, 1], dp[m, 0])
            d = psi[a:b - 1][m] - trav
            mu = float(torch.atan2(torch.sin(d).mean(), torch.cos(d).mean()))
            if abs(mu) > math.pi / 2:
                psi[a:b] = psi[a:b] + math.pi
                flipped.append((a, b, math.degrees(mu)))
        print(f"[TrajFollow] reference heading head/tail check: {len(flipped)}/{len(runs)} runs "
              f"flipped by 180 deg, {len(skipped)} low-displacement runs left as-is", flush=True)
        for a, b, mu in flipped[:8]:
            print(f"    flipped run [{a},{b}) (psi was {mu:+.1f} deg from its travel direction)",
                  flush=True)
        return psi

    def _sample_trajectory(self, env_ids):
        """Draw a random real-ZeF window per env, convert to a body-frame path, re-plant it at
        that env's CURRENT (just-spawned) pose/body-length, and precompute the arc-length (BL),
        unit tangent, and local reference speed (BL/s) needed for phase-based lookahead."""
        ids = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=self.device)
        n = int(len(ids))
        if n == 0:
            return
        dev = self.device
        BL = self._body_length
        root = self.robot.data.root_state_w[ids]
        p0 = root[:, 0:3]
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=dev).repeat(n, 1)
        fwd_w = math_utils.quat_apply(root[:, 3:7], fwd_b)
        psi_a0 = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])

        if not hasattr(self, "_traj_ti"):
            self._traj_ti = torch.full((self.num_envs,), -1, dtype=torch.long, device=dev)   # window start frame per env (for replay / fixed-window videos)
        fixed_ti = int(getattr(self.cfg, "fixed_traj_ti", -1))
        for k, e in enumerate(ids.tolist()):
            run = self._zef_runs[np.random.randint(len(self._zef_runs))]
            a, b = run
            span = b - a
            w = min(self._traj_win_frames, span - 1)
            ti = a + (np.random.randint(0, span - w) if span - w > 0 else 0)
            if fixed_ti >= 0:                       # video/eval helper: every episode uses the SAME reference window
                for (a2, b2) in self._zef_runs:
                    if a2 <= fixed_ti < b2: a, b = a2, b2; break
                w = min(self._traj_win_frames, b - 1 - fixed_ti); ti = fixed_ti
            self._traj_ti[e] = ti
            tj = ti + w
            seg = self._zef_p_bl[ti:tj + 1:self._zef_stride]
            psi_np = self._zef_psi[ti:tj + 1:self._zef_stride].cpu().numpy()
            psi_seg = torch.tensor(np.unwrap(psi_np), dtype=torch.float64, device=dev)
            T_e = seg.shape[0]
            d = seg - seg[0]
            c0, s0 = torch.cos(-psi_seg[0]), torch.sin(-psi_seg[0])
            db = torch.stack([c0 * d[:, 0] - s0 * d[:, 1], s0 * d[:, 0] + c0 * d[:, 1]], 1)

            ca, sa = torch.cos(psi_a0[k]), torch.sin(psi_a0[k])
            wx = p0[k, 0] + (ca * db[:, 0] - sa * db[:, 1]) * BL
            wy = p0[k, 1] + (sa * db[:, 0] + ca * db[:, 1]) * BL
            wz = torch.full((T_e,), float(self._target_height), device=dev, dtype=torch.float64)
            path = torch.stack([wx, wy, wz], 1).to(torch.float32)

            seg_vec = path[1:] - path[:-1]
            seg_len_bl = seg_vec.norm(dim=1) / BL                                  # (T_e-1,)
            s_bl = torch.cat([torch.zeros(1, device=dev), torch.cumsum(seg_len_bl, 0)])
            tangent = torch.zeros(T_e, 2, device=dev)
            tangent[:-1] = seg_vec[:, 0:2] / seg_vec[:, 0:2].norm(dim=1, keepdim=True).clamp(min=1e-6)
            tangent[-1] = tangent[-2] if T_e > 1 else tangent[-1]
            speed_bl = torch.zeros(T_e, device=dev)
            speed_bl[:-1] = seg_len_bl / self._sim_dt
            speed_bl[-1] = speed_bl[-2] if T_e > 1 else 0.0

            self._traj_path[e, :T_e] = path
            self._traj_path[e, T_e:] = path[-1]
            self._traj_s_bl[e, :T_e] = s_bl
            self._traj_s_bl[e, T_e:] = s_bl[-1] + 1.0e6                            # never-searched pad
            self._traj_tangent[e, :T_e] = tangent
            self._traj_tangent[e, T_e:] = tangent[-1]
            self._traj_speed_bl[e, :T_e] = speed_bl
            self._traj_speed_bl[e, T_e:] = 0.0
            self._traj_len[e] = T_e
            self._ep_p0[e] = p0[k]
            self._ep_psi0[e] = psi_a0[k]
        self._fish_hist[ids] = 0.0

    def _interp_path_at_arclength(self, target_s: torch.Tensor):
        """target_s: (E,) query arc-length (BL), per env. Returns (p_star (E,3) world,
        tangent (E,2) unit world-XY, v_star (E,) BL/s) via linear interpolation along that
        env's own path, clamped to [0, that path's own max arc length]."""
        E = self.num_envs
        ar = torch.arange(E, device=self.device)
        max_s = self._traj_s_bl[ar, (self._traj_len - 1).clamp(min=0)]
        s_c = target_s.clamp(min=0.0).minimum(max_s)
        idx_hi = torch.searchsorted(self._traj_s_bl, s_c.unsqueeze(1)).squeeze(1).clamp(1, self._traj_T_max - 1)
        idx_lo = (idx_hi - 1).clamp(min=0)
        s_lo, s_hi = self._traj_s_bl[ar, idx_lo], self._traj_s_bl[ar, idx_hi]
        frac = ((s_c - s_lo) / (s_hi - s_lo).clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(1)
        p_star = self._traj_path[ar, idx_lo] + frac * (self._traj_path[ar, idx_hi] - self._traj_path[ar, idx_lo])
        tan = self._traj_tangent[ar, idx_lo] + frac * (self._traj_tangent[ar, idx_hi] - self._traj_tangent[ar, idx_lo])
        tan = tan / tan.norm(dim=1, keepdim=True).clamp(min=1e-6)
        v_star = self._traj_speed_bl[ar, idx_lo] + frac.squeeze(1) * (
            self._traj_speed_bl[ar, idx_hi] - self._traj_speed_bl[ar, idx_lo])
        return p_star, tan, v_star

    def _project_phase(self):
        """s_t = argmin_s ||Gamma(s) - p_fish||: nearest-point projection of the fish's CURRENT
        world position onto its own planted path. Returns (s_t (E,) BL, phase_idx (E,))."""
        E = self.num_envs
        p = self.robot.data.root_state_w[:, 0:3]
        diff = self._traj_path - p.unsqueeze(1)
        d2 = (diff ** 2).sum(-1)
        mask = torch.arange(self._traj_T_max, device=self.device).unsqueeze(0) >= self._traj_len.unsqueeze(1)
        d2 = d2.masked_fill(mask, float("inf"))
        idx = d2.argmin(dim=1)
        s_t = self._traj_s_bl[torch.arange(E, device=self.device), idx]
        return s_t, idx

    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        ids_t = ids if torch.is_tensor(ids) else torch.tensor(ids, device=self.device)
        self._compute_frechet_for(ids_t)
        super()._reset_idx(env_ids)
        self._sample_trajectory(ids_t)

    def _compute_frechet_for(self, ids: torch.Tensor):
        """Discrete Frechet distance between the fish's ACTUAL recorded path this episode and
        the ZeF reference path it was given, both in body-frame/BL coordinates (re-rooted at
        THAT episode's own start pose), RAW temporal order (no arc-length resampling) -- called
        right before an env's trajectory gets overwritten by a fresh one."""
        if ids.numel() == 0:
            return
        BL = self._body_length
        for e in ids.tolist():
            n_steps_e = int(self.episode_length_buf[e].item())
            if n_steps_e > 0:
                self._ep_blowups.append(1.0 if bool(self._blowup_flag[e].item()) else 0.0)
            if self._bio is not None and e in self._bio.pos:
                self._bio.finish(e, n_steps_e)
        for e in ids.tolist():
            T_e = int(self._traj_len[e].item())
            n_steps = int(self.episode_length_buf[e].item())
            # m must be bounded by ALL THREE: how many steps the episode actually ran
            # (n_steps -- an early blowup/OOB termination means indices beyond this were never
            # written and are stale/zero), the reference path's own valid length (T_e -- past
            # this the fish is no longer "tracking its assigned path", it's just doing whatever
            # it does after the path ends), and the buffer capacity (T_max, always >= T_e).
            m = max(1, min(n_steps, T_e, self._traj_T_max))
            if T_e < 2 or m < 2:
                self._last_frechet[e] = float("nan")
                self._last_completed[e] = False
                continue
            p0 = self._ep_p0[e].cpu().numpy()
            psi0 = float(self._ep_psi0[e].item())
            c0, s0 = np.cos(-psi0), np.sin(-psi0)

            def to_body(pts_world):
                d = pts_world - p0
                x = c0 * d[:, 0] - s0 * d[:, 1]
                y = s0 * d[:, 0] + c0 * d[:, 1]
                return np.stack([x, y], 1) / BL

            fish_w = self._fish_hist[e, :m].cpu().numpy()
            gt_w = self._traj_path[e, :T_e].cpu().numpy()
            fish_b = to_body(fish_w)
            gt_b = to_body(gt_w)
            # a blown-up episode is NEVER a completion: the exploded root position projects onto
            # an arbitrary path point (often the end), which used to count as "reached end" --
            # sturgeon020 joint_rl showed completion 0.2-0.46 with 20-step episodes and 90% blow-ups.
            # Its Frechet is likewise meaningless (positions ~1e5 BL) -> nan (excluded from mean;
            # train/blowup_rate reports these episodes separately).
            if bool(self._blowup_flag[e].item()):
                self._last_frechet[e] = float("nan")
                self._last_completed[e] = False
                continue
            self._last_frechet[e] = discrete_frechet(fish_b, gt_b)
            # the REAL completion signal: did this env's phase projection actually reach the
            # end of ITS path this step (cached from _get_dones), NOT "did enough wall-clock
            # steps elapse" -- every episode elapses >= T_e steps whenever it runs to the
            # episode_length_s time_out backstop (~1800 steps, T_e ~151), which would make a
            # step-count check TRUE unconditionally and always report 100% completion.
            self._last_completed[e] = bool(self._cur_traj_reached_end[e].item())

    # ------------------------------------------------------------------ dones
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
        terminated = out_of_bounds | joint_blowup | nonfinite

        # record this step's actual position for the end-of-episode Frechet comparison.
        # index = episode_length_buf - 1: the base DirectRLEnv.step() does
        # `self.episode_length_buf += 1` BEFORE calling _get_dones() (direct_rl_env.py), so on
        # the FIRST step of a fresh episode episode_length_buf is already 1, not 0 -- indexing
        # by episode_length_buf directly (the original bug) never writes slot 0 at all, leaving
        # it at its zero-init value (world origin) for the entire episode. Since a real fish's
        # spawn point is essentially never at the world origin, that single always-wrong first
        # point dominates the discrete Frechet distance (a worst-case/leash-length metric),
        # inflating it by 1-2 orders of magnitude regardless of how well the fish actually
        # tracks. STOP writing (rather than clamp-and-overwrite) once past the buffer's own
        # length: episodes routinely run far longer (up to the 1800-step time_out backstop)
        # than the sampled path (~151 steps), and clamping would keep overwriting the last slot
        # with wherever the fish ends up hundreds of steps later.
        idx = self.episode_length_buf - 1
        in_buf = (idx >= 0) & (idx < self._traj_T_max)
        if in_buf.any():
            ar = torch.arange(self.num_envs, device=self.device)[in_buf]
            self._fish_hist[ar, idx[in_buf]] = root_state[in_buf, 0:3]

        s_t, phase_idx = self._project_phase()
        delta_bl = float(self.cfg.traj_delta_bl)
        p_star, tan_star, v_star = self._interp_path_at_arclength(s_t + delta_bl)
        self._cur_p_star = p_star
        self._cur_tangent_star = tan_star
        self._cur_v_star = v_star

        self._blowup_flag = joint_blowup | nonfinite
        if self._bio is not None:
            step_idx, krows, srows, prows = {}, {}, {}, {}
            bp = self.robot.data.body_link_pos_w; bq = self.robot.data.body_link_quat_w
            v_bl = torch.linalg.norm(root_state[:, 7:9], dim=1) / self._body_length
            for e in self._bio_ids.tolist():
                step_idx[e] = int(idx[e].item())
                try:
                    pr = bm_kappa(bp[e], bq[e], self._bio_geo)
                except Exception:  # noqa: BLE001
                    pr = None
                krows[e] = pr; srows[e] = float(v_bl[e].item()); prows[e] = int(phase_idx[e].item())
            self._bio.record(step_idx, krows, srows, prows)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # BUG FIX 2026-09-22 (completion was "projection reached the path end"): a fish passing far
        # from the path still projected onto the last point and counted as complete (white_bass008
        # joint_rl "converged" at completion 0.94 with Frechet 2.1 BL). Require the fish to actually
        # be within completion_tol_bl of the path END when the phase gets there.
        _end_idx = (self._traj_len - 1).clamp(min=0)
        _p_end = self._traj_path[torch.arange(self.num_envs, device=self.device), _end_idx]
        _d_end_bl = torch.linalg.norm(root_state[:, 0:3] - _p_end, dim=1) / self._body_length
        traj_reached_end = (phase_idx >= (self._traj_len - 1)) & (_d_end_bl <= float(getattr(self.cfg, "completion_tol_bl", 0.5)))
        self._cur_traj_reached_end = traj_reached_end
        episode_done = (traj_reached_end | time_out) & ~terminated

        dist_bl = torch.linalg.norm(root_state[:, 0:3] - p_star, dim=1) / self._body_length
        log = self.extras.setdefault("log", {})
        log.update({
            "traj/dist_bl": dist_bl.mean().detach(),
            "traj/phase_progress": (phase_idx.float() / (self._traj_len - 1).clamp(min=1).float()).mean().detach(),
            "traj/completion_rate": self._last_completed.float().mean().detach(),
        })
        # only log frechet_dist once at least one episode has actually finished (else every
        # entry is still the init NaN, which corrupts the tensorboard/wandb running mean).
        _finite = torch.isfinite(self._last_frechet)
        if _finite.any():
            log["traj/frechet_dist"] = self._last_frechet[_finite].mean().detach()
            log["eval/frechet_dist_bl"] = log["traj/frechet_dist"]
        log["eval/completion_rate"] = log["traj/completion_rate"]
        if self._ep_blowups:
            log["train/blowup_rate"] = torch.tensor(float(np.mean(self._ep_blowups)), device=self.device)
        if self._bio is not None:
            for k, v in self._bio.log_dict().items():
                log[k] = torch.tensor(float(v), device=self.device)
        return terminated, episode_done

    # ------------------------------------------------------------------ reward
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d = self.robot.data
        root_state = d.root_state_w
        p = root_state[:, 0:3]

        dist_bl = torch.linalg.norm(p - self._cur_p_star, dim=1) / self._body_length
        r_track = torch.exp(-float(c.traj_alpha) * dist_bl)

        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        fwd_w = math_utils.quat_apply(root_state[:, 3:7], fwd_b)
        theta_fish = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
        theta_traj = torch.atan2(self._cur_tangent_star[:, 1], self._cur_tangent_star[:, 0])
        r_heading = torch.cos(theta_fish - theta_traj)

        v_fish_bl = torch.linalg.norm(root_state[:, 7:9], dim=1) / self._body_length
        r_speed = -float(c.traj_beta) * torch.abs(v_fish_bl - self._cur_v_star)

        # POST-MAPPING joint-space command (self._pos_targets, radians), NOT the raw action --
        # for PCA, self.actions is the 20-dim coefficient-RATE command, pre-W-mapping, so its
        # units/scale depend on each mode's (arbitrary) gain in _W and don't correspond to
        # physical joint effort. self._pos_targets is the actual commanded joint angle AFTER the
        # PCA->joint mapping (q0 + a@W, clamped to soft limits) -- same (E, nj) shape/units the
        # Joint controller's own self._pos_targets already uses, so this term is now comparable
        # between the PCA and Joint variants.
        u = self._pos_targets
        r_energy = -float(c.traj_gamma) * (u ** 2).sum(dim=1)

        reward = torch.nan_to_num(
            float(c.traj_w1) * r_track + float(c.traj_w2) * r_heading
            + float(c.traj_w3) * r_speed + float(c.traj_w4) * r_energy)
        tol_bl = float(getattr(c, "traj_tolerance_bl", 1.0))
        frac_within_tol = (dist_bl < tol_bl).float().mean()
        self.extras.setdefault("log", {}).update({
            "traj/rew_track": r_track.mean().detach(), "traj/rew_heading": r_heading.mean().detach(),
            "traj/rew_speed": r_speed.mean().detach(), "traj/rew_energy": r_energy.mean().detach(),
            "traj/rew_total": reward.mean().detach(), "traj/frac_within_tol": frac_within_tol.detach(),
            "traj/speed_error_bl": torch.abs(v_fish_bl - self._cur_v_star).mean().detach(),
        })
        return reward

    # ------------------------------------------------------------------ shared obs component
    def _traj_obs_component(self) -> torch.Tensor:
        """[dir_b(3), dist(1)] to p_star (s_t+delta), then (K-1) further lookahead points each
        an additional delta ahead in arc length -- all in the fish's CURRENT body frame, /BL."""
        root = self.robot.data.root_state_w
        p = root[:, 0:3]
        delta_bl = float(self.cfg.traj_delta_bl)
        K = int(self.cfg.traj_lookahead_k)

        delta_w = self._cur_p_star - p
        dist = delta_w.norm(dim=1, keepdim=True)
        dir_b = math_utils.quat_apply_inverse(root[:, 3:7], delta_w) / dist.clamp(min=1e-6)

        look = []
        if K > 1:
            s_t, _ = self._project_phase()
            for i in range(2, K + 1):
                pt_w, _, _ = self._interp_path_at_arclength(s_t + i * delta_bl)
                pt_b = math_utils.quat_apply_inverse(root[:, 3:7], pt_w - p) / self._body_length
                look.append(pt_b)
        look = torch.cat(look, dim=1) if look else torch.zeros(self.num_envs, 0, device=p.device)
        return torch.nan_to_num(torch.cat([dir_b, dist / self._body_length, look], dim=1))


class SalmonSwimPCATrajFollowEnv(TrajFollowMixin, SalmonSwimPCAEnv):
    """BLM+RL (PCA) trajectory-following baseline."""

    cfg: SalmonSwimPCATrajFollowCfg

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        traj_dim = 4 + 3 * (int(self.cfg.traj_lookahead_k) - 1)
        self._obs_dim = int(self.single_observation_space["policy"].shape[0]) + traj_dim
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        print(f"[PCATrajFollow] obs widened by {traj_dim} traj dims -> {self._obs_dim}", flush=True)

    def __init__(self, cfg: SalmonSwimPCATrajFollowCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_traj_follow()

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        obs["policy"] = torch.cat([obs["policy"], self._traj_obs_component()], dim=-1)
        return obs


class SalmonSwimJointTrajFollowEnv(TrajFollowMixin, SalmonSwimPCAEnv):
    """Raw per-joint trajectory-following baseline (the IL/BCO family's action space): RL
    directly outputs one target per controlled joint. Sibling of SalmonSwimPCATrajFollowEnv
    (not a subclass of it), identical to SalmonSwimJointReach10Env's own pattern."""

    cfg: SalmonSwimJointTrajFollowCfg

    def _configure_gym_env_spaces(self):
        SalmonSwimEnv._configure_gym_env_spaces(self)
        nj = int(self._control_joint_ids.shape[0])
        self._num_pca = int(self.cfg.pca_num_modes)     # harmless compat, see class docstring above
        traj_dim = 4 + 3 * (int(self.cfg.traj_lookahead_k) - 1)
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + traj_dim
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        print(f"[JointTrajFollow] action = {nj} raw joint position targets; obs = {self._obs_dim} dims",
              flush=True)

    def __init__(self, cfg: SalmonSwimJointTrajFollowCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_traj_follow()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_actions)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
        hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        tgt = torch.where(actions >= 0, actions * hi, actions * (-lo))
        self._pos_targets = torch.clamp(tgt, lo, hi)

    def _get_observations(self) -> dict:
        d = self.robot.data
        root = d.root_state_w
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        heading_dir = math_utils.quat_apply(root[:, 3:7], fwd_b)
        heading_dir = heading_dir / heading_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up_b = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        up_dir = math_utils.quat_apply(root[:, 3:7], up_b)
        jp = d.joint_pos[:, self._control_joint_ids] * self.cfg.obs_scales.joint_pos
        jv = d.joint_vel[:, self._control_joint_ids] * self.cfg.obs_scales.joint_vel
        obs = torch.cat((
            d.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel,
            d.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel,
            jp, jv, heading_dir, up_dir, self._traj_obs_component(),
        ), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}


class SalmonSwimCPGTrajFollowEnv(TrajFollowMixin, SalmonSwimPCAEnv):
    """CPG+RL trajectory following (per-fish calibrated traveling wave; same maths as
    salmon_swim_reach10_env.SalmonSwimCPGReach10Env, task part from TrajFollowMixin)."""

    cfg: SalmonSwimCPGTrajFollowCfg

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        nj = int(self._control_joint_ids.shape[0])
        self._num_cpg = 3
        traj_dim = 4 + 3 * (int(self.cfg.traj_lookahead_k) - 1)
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + (self._num_cpg + 2) + traj_dim
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_cpg,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_cpg), device=self.device)
        print(f"[CPGTrajFollow] action = [dA, df, db]; obs = {self._obs_dim}", flush=True)

    def __init__(self, cfg: SalmonSwimCPGTrajFollowCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        p = np.load(str(cfg.cpg_params_path))
        E_s = torch.tensor(p["envelope"], device=dev, dtype=torch.float32)
        phi = torch.tensor(p["phase"], device=dev, dtype=torch.float32)
        M = self._Mdec
        self._cpg_u1 = M @ (E_s * torch.cos(phi)); self._cpg_u2 = M @ (E_s * torch.sin(phi)); self._cpg_ub = M @ torch.ones_like(E_s)
        self._cpg_f0 = float(p["f0"])
        self._cp_lo = torch.tensor([0.0, float(p["f_lo"]), -float(p["b_max"])], device=dev)
        self._cp_hi = torch.tensor([float(p["A_max"]), float(p["f_hi"]), float(p["b_max"])], device=dev)
        self._cdp = torch.tensor([float(p["dA_max"]), float(p["df_max"]), float(p["db_max"])], device=dev)
        self._cpg_p = torch.zeros(self.num_envs, 3, device=dev); self._cpg_p[:, 1] = self._cpg_f0
        self._cpg_theta = torch.zeros(self.num_envs, device=dev)
        self._ctrl_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        self._init_traj_follow()
        print(f"[CPGTrajFollow] fish-calibrated CPG: A<={float(p['A_max']):.2f} f in [{float(p['f_lo']):.2f},{float(p['f_hi']):.2f}] "
              f"b<={float(p['b_max']):.2f} f0={self._cpg_f0:.2f} Hz", flush=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions.view(self.num_envs, self._num_cpg), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        self._cpg_p = torch.clamp(self._cpg_p + actions * self._cdp, self._cp_lo, self._cp_hi)
        A, f, b = self._cpg_p[:, 0], self._cpg_p[:, 1], self._cpg_p[:, 2]
        self._cpg_theta = torch.remainder(self._cpg_theta + 2.0 * math.pi * f * self._ctrl_dt, 2.0 * math.pi)
        q = (self._q0.unsqueeze(0) + A.unsqueeze(1) * (torch.sin(self._cpg_theta).unsqueeze(1) * self._cpg_u1.unsqueeze(0)
             + torch.cos(self._cpg_theta).unsqueeze(1) * self._cpg_u2.unsqueeze(0)) + b.unsqueeze(1) * self._cpg_ub.unsqueeze(0))
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]; hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        self._pos_targets = torch.clamp(q, lo, hi)

    def _get_observations(self) -> dict:
        d = self.robot.data; root = d.root_state_w
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        heading_dir = math_utils.quat_apply(root[:, 3:7], fwd_b); heading_dir = heading_dir / heading_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up_dir = math_utils.quat_apply(root[:, 3:7], torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        jp = d.joint_pos[:, self._control_joint_ids] * self.cfg.obs_scales.joint_pos
        jv = d.joint_vel[:, self._control_joint_ids] * self.cfg.obs_scales.joint_vel
        p_norm = (self._cpg_p - self._cp_lo) / (self._cp_hi - self._cp_lo).clamp(min=1e-6) * 2.0 - 1.0
        obs = torch.cat((d.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel, d.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel,
                         jp, jv, heading_dir, up_dir, p_norm, torch.sin(self._cpg_theta).unsqueeze(1), torch.cos(self._cpg_theta).unsqueeze(1),
                         self._traj_obs_component()), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    def _get_rewards(self) -> torch.Tensor:
        reward = TrajFollowMixin._get_rewards(self)
        self.extras["log"].update({"cpg/A": self._cpg_p[:, 0].mean().detach(), "cpg/f_hz": self._cpg_p[:, 1].mean().detach(),
                                   "cpg/b": self._cpg_p[:, 2].mean().detach()})
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        if hasattr(self, "_cpg_p"):
            self._cpg_p[idx] = 0.0; self._cpg_p[idx, 1] = self._cpg_f0; self._cpg_theta[idx] = 0.0


class SalmonSwimJointAMPTrajFollowEnv(SalmonSwimJointTrajFollowEnv):
    """Joint RL + AMP trajectory following (same borrow-by-name pattern as
    salmon_swim_reach10_env.SalmonSwimJointAMPReach10Env; reward = task + w_amp * r_style)."""

    cfg: SalmonSwimJointAMPTrajFollowCfg
    _amp_feature = SalmonAMPTankEnv._amp_feature
    _bend3d_np = SalmonAMPTankEnv._bend3d_np
    _init_disc = SalmonAMPTankEnv._init_disc
    _amp_reference_contract_check = SalmonAMPTankEnv._amp_reference_contract_check
    _update_disc = SalmonAMPTankEnv._update_disc
    _save_disc = SalmonAMPTankEnv._save_disc
    _load_disc = SalmonAMPTankEnv._load_disc
    _amp_style_reward = SalmonSwimAMPEnv._amp_style_reward

    def __init__(self, cfg: SalmonSwimJointAMPTrajFollowCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        self._K = int(self.cfg.profile_len); self._nbend = 2 * self._K
        self._prev_feat = torch.zeros(self.num_envs, self._nbend + 6, device=dev)
        self._prev_hd = torch.zeros(self.num_envs, device=dev); self._prev_pit = torch.zeros(self.num_envs, device=dev)
        self._prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._last_bend = torch.zeros(self.num_envs, self._nbend, device=dev)

    def _get_rewards(self) -> torch.Tensor:
        task_r = super()._get_rewards()
        r_style = self._amp_style_reward()
        w = float(getattr(self.cfg, "w_amp", 0.2))
        self.extras.setdefault("log", {}).update({"amp/r_style": r_style.mean().detach(), "amp/reward_task": task_r.mean().detach()})
        return task_r + w * r_style
