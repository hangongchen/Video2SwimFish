# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reactive natural-swimming AMP tank env (step 3 of AMP_TANK_SWIM_PIPELINE.md).

Cooked-FEM zebrafish in a square tank with four collidable walls (inherited from the tank env),
trained by rl_games PPO to swim + turn like a real zebrafish while reacting to the walls, forever,
with an energy penalty. No goal/target.

Style imitation: the installed rl_games has no AMP, so the discriminator lives INSIDE this env. The
motion feature is the SCALE-INVARIANT spine-bend profile (same `spine_bend_profile` used to build the
ZeF reference), so real (3 cm fish) and sim (0.5 m fish) are comparable; the disc matches the
(bend_t, bend_t+1) TRANSITION distribution -> undulation + turn-posture dynamics. Speed/turning are
driven by the task reward + walls, not the style feature (avoids the real/sim kinematic scale gap).

Reward = w_style*AMP + w_keepmove*cruise + wall_penalty + ENERGY_PENALTY(-w*sum|tau*omega|)
         + alive + stability. Reference-free reactive observation (proprio + wall-distance rays).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import isaaclab.utils.math as math_utils

from .salmon_tank_swim_env import SalmonTankSwimEnv

# the EXACT feature extractor used on the real reference clouds (consistency is critical)
sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "scripts"))
from extract_amp_features import anterior_aligned_profile  # noqa: E402

# ---- AMP motion-feature contract (MUST match scripts/zef3d_build_ref.py exactly) -------------
# Phi = [ bend(2K) | motion(AMP_N_MOTION) ]. The bend block stays FIRST so every existing
# `feat[:, :self._nbend]` / `feat[:, :self._K]` slice (track reward, bend EMA, envelope) is unaffected.
#
# ORIENTATION (2026-08-06): the motion block used to be [|v_horizontal|, v_z_world, d/dt(heading of
# the VELOCITY), d/dt(pitch of the VELOCITY)] -- all world-frame or bare magnitudes. The 2K bend
# channels are expressed in the body's OWN frame and are therefore EXACTLY invariant to yawing the
# whole fish (verified numerically: max|delta bend| = 9e-19 for a 90 deg yaw). So NOTHING in the old
# 44-dim Phi was a function of the angle between the body axis and the direction of travel, and a
# policy sliding BROADSIDE produced a bit-identical feature to one swimming head-first at the same
# speed. The discriminator was structurally blind to the exact failure the policy had learned
# (measured: median 70 deg between travel and body line, sideways in 98% of frames) -- which is why
# the `rew_scales.backward` / `rew_scales.offaxis` reward patches had to exist at all.
#
# The motion block is now the velocity expressed in the body's OWN (nose, left, up) basis -- the same
# triad the bend profile is already measured in -- SPLIT into direction and magnitude on purpose:
# the real fish cruises at ~3.6 BL/s while this sim fish tops out near 0.7 BL/s, so any channel
# carrying absolute speed hands D a free separating axis. Splitting confines that gap to ONE channel
# (speed_bl) and leaves THREE scale-free channels that say only "which way is the fish pointing
# relative to where it is going".
AMP_MOTION_NAMES = ("speed_bl", "dir_nose", "dir_left", "dir_up", "yaw_body", "pitch_body")
AMP_N_MOTION = len(AMP_MOTION_NAMES)
# below this speed (BL/s) the travel DIRECTION is numerical noise -> report the zero vector
# ("no opinion") instead of a random unit vector. Same constant as the reference builder.
AMP_TRAVEL_MIN_BL = 0.05


class _Disc(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SalmonAMPTankEnv(SalmonTankSwimEnv):

    # ---- spaces / obs-time state (runs during super().__init__, before rewards) ----
    def _configure_gym_env_spaces(self):
        self._control_joint_ids = torch.arange(self.robot.num_joints, device=self.device)
        self._num_actions = self._control_joint_ids.shape[0]
        self._K = int(self.cfg.profile_len)
        self._R = int(self.cfg.n_wall_rays)
        # reactive obs: grav(3)+lin(3)+ang(3)+joint_pos(nj)+joint_vel(nj)+wall_rays(R)
        #               + up_b(3) + z_err(1) + phase(2 = sin,cos of the kinematic-imitation clock)
        obs_dim = 9 + 2 * self._num_actions + self._R + 4 + (2 if getattr(self.cfg, "use_phase_clock", True) else 0)
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_actions), device=self.device)

        dev = self.device
        self._half = float(self.cfg.tank_size) / 2.0
        ang = torch.linspace(0, 2 * np.pi, self._R + 1, device=dev)[:-1]
        self._ray_body_ang = ang                                    # (R,) body-frame ray angles
        self._nbend = 2 * self._K                                                  # bend dims (left/right + up/down)
        self._prev_feat = torch.zeros(self.num_envs, self._nbend + AMP_N_MOTION, device=dev)
        # prev NOSE azimuth / elevation (for the BODY turn rates). These used to hold the heading and
        # pitch of the VELOCITY vector; they now track the body axis -- see AMP_MOTION_NAMES.
        self._prev_hd = torch.zeros(self.num_envs, device=dev)
        self._prev_pit = torch.zeros(self.num_envs, device=dev)
        self._prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._prev_actions = torch.zeros(self.num_envs, self._num_actions, device=dev)
        # previous world root linear velocity, for the finite-difference acceleration penalty.
        # Reset per-env in _reset_idx; the accel penalty is gated by _prev_valid so the first
        # post-reset step (where this is stale) is never penalised.
        self._prev_root_vel = torch.zeros(self.num_envs, 3, device=dev)
        # NET-DISPLACEMENT ring buffer: position W steps ago, for r_displacement (un-gameable by noise
        # -- exploration noise averages to ~0 net displacement, so only a COHERENT MEAN gait earns it).
        self._disp_W = int(getattr(self.cfg, "displacement_window", 60))
        self._pos_ring = torch.zeros(self.num_envs, self._disp_W, 3, device=dev)
        self._ring_ptr = 0
        self._ep_step = torch.zeros(self.num_envs, dtype=torch.long, device=dev)   # steps since reset
        # EARLY-TERMINATION (ET) freeze counter + the LIVE lateral DOF mask (%3==2; the other 2 DOF/bone
        # are locked and never move, so averaging |joint_vel| over all 27 would dilute the freeze signal).
        self._freeze_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._live_dof = (torch.arange(self._num_actions, device=dev) % 3 == 2)
        # KINEMATIC-IMITATION phase clock: per-env phase offset (randomized on reset for gait-start
        # diversity); the instantaneous phase = _phase0 + omega * ep_step * dt advances at track_freq.
        self._phase0 = torch.zeros(self.num_envs, device=dev)
        self._track_env = None                                # reference envelope A(s), lazy-loaded
        # EMA of the spine-bend profile, for the STATIC-BEND penalty. A real undulation averages to
        # ~0 over a beat cycle; a parked C-curve does not, so |EMA| isolates the posture from the gait.
        self._bend_ema = torch.zeros(self.num_envs, 2 * self._K, device=dev)
        # L2 blow-up cooldown: per-env countdown; while >0 the action is zeroed and the reward + disc
        # push are masked, so a just-reset (post-blow-up) FEM settles instead of re-diverging.
        self._cooldown = torch.zeros(self.num_envs, dtype=torch.long, device=dev)

    # ---- lazy discriminator / reference init (first reward step, after full env init) ----
    def _init_disc(self):
        dev = self.device
        c = self.cfg
        d = np.load(c.amp_reference_path)
        feat = torch.tensor(d["feat"], dtype=torch.float32, device=dev)        # (T, K+2) bend + [v_fwd,yaw]
        self._feat_dim = feat.shape[1]
        seg = torch.tensor(d["segment_id"], dtype=torch.long, device=dev)
        # per-channel z-score with a STD FLOOR (Problem 2): a channel with ~0 reference std would blow
        # up the z-score and let D win on that numerical artifact; clamp the std to 1e-2.
        self._feat_mean = feat.mean(0)
        self._feat_std = feat.std(0).clamp_min(1e-2)
        fn = (feat - self._feat_mean) / self._feat_std
        pair = seg[1:] == seg[:-1]
        self._ref_trans = torch.cat([fn[:-1][pair], fn[1:][pair]], dim=1).contiguous()  # (P, 2*feat_dim)
        self._disc = _Disc(2 * self._feat_dim, tuple(c.disc_hidden)).to(dev)
        self._disc_opt = torch.optim.Adam(self._disc.parameters(), lr=float(c.disc_lr))
        self._buf = torch.zeros(int(c.disc_buffer_size), 2 * self._feat_dim, device=dev)
        self._buf_n = 0
        self._buf_ptr = 0
        self._amp_step = 0
        self._disc_updates = 0
        # optional warm-start / eval-time load of the trained discriminator + optimizer state
        if getattr(c, "disc_load_on_init", False):
            self._load_disc(getattr(c, "disc_ckpt_path", None))
        print(f"[AMP] reference {self._ref_trans.shape[0]} transitions; feat_dim {self._feat_dim} "
              f"({2*self._K} bend [L/R+U/D] + {self._feat_dim - 2*self._K} motion); disc in {2*self._feat_dim} "
              f"hidden {tuple(c.disc_hidden)} lr {float(c.disc_lr):.1e} R1 {float(getattr(c,'disc_r1',0)):.1f} "
              f"noise {float(getattr(c,'disc_noise_std',0.0)):.2f}; num_envs {self.num_envs}", flush=True)
        self._amp_reference_contract_check(d)

    def _amp_reference_contract_check(self, ref):
        """Prove, at startup, that sim and reference speak the SAME feature language.

        AMP is only valid if the observation map Phi is identical on both sides, and every way that
        can break here is SILENT: a stale reference file, an asset whose head points the other way, a
        flipped nose axis. This repo has been bitten by exactly that class of bug before (the FEM
        material that was never bound), so the rule is the same: print the contract, and make a
        violation impossible to miss in the log.

        Three checks:
          1. the reference's own `vel_names` match AMP_MOTION_NAMES (hard error -- a width match with
             a semantics mismatch is the worst case, it trains happily on garbage);
          2. station 0 of the SIM bend profile really is the head (the tail is a thin blade, the head
             is round, so the head end is much wider across its THINNEST axis);
          3. the reference really is head-first (dir_nose > 0), so the sign convention agrees.
        """
        names = [str(s) for s in ref["vel_names"]] if "vel_names" in ref else None
        if names is not None and tuple(names) != tuple(AMP_MOTION_NAMES):
            raise RuntimeError(
                f"[AMP] REFERENCE CONTRACT MISMATCH: {self.cfg.amp_reference_path} has motion channels "
                f"{tuple(names)} but this env produces {tuple(AMP_MOTION_NAMES)}. Rebuild the reference "
                f"with `python scripts/zef3d_build_ref.py` (or revert _amp_feature). Widths can match "
                f"while the meanings do not -- that trains silently on nonsense.")
        _fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
        try:
            pos = self._soft_view.get_simulation_mesh_nodal_positions()
            nodes = torch.as_tensor(pos, dtype=torch.float32).detach().cpu().numpy()[0]
            root = self.robot.data.root_state_w[0].detach().cpu().numpy()
            w, x, y, z = root[3:7]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            fwd3 = R @ np.array([_fs, 0.0, 0.0]); fwd3 /= np.linalg.norm(fwd3) + 1e-9
            rel = nodes - root[0:3]
            along = rel @ fwd3
            e1 = np.cross(fwd3, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1) + 1e-9
            e2 = np.cross(fwd3, e1)
            lo, hi = np.percentile(along, 2), np.percentile(along, 98)
            # Cross-body spans at each end. A caudal fin is a TALL THIN BLADE, so its LATERAL span
            # collapses while the head keeps a round cross-section -- lateral is the discriminating
            # axis (the vertical one goes the WRONG way: the fin is taller than the head).
            def _spans(sel):
                if sel.sum() < 4:
                    return float("nan"), float("nan"), 0
                return float(np.ptp(rel[sel] @ e1)), float(np.ptp(rel[sel] @ e2)), int(sel.sum())
            n_lat, n_ver, n_n = _spans(along >= hi)     # high `along` == index 0 after the pts[::-1]
            t_lat, t_ver, t_n = _spans(along <= lo)
            ok = n_lat > t_lat
            # ADVISORY, not authoritative: on a COARSE FEM tetrahedral mesh a thin caudal blade is
            # fattened by the tessellation, which shrinks this margin toward 1. Treat a ratio near 1
            # as "no signal" rather than as a pass. The authoritative checks are the asset's own
            # pipeline metadata (head_at_plus_x) and scripts/-level replay of this binning.
            ratio = n_lat / max(t_lat, 1e-9)
            print(f"[AMP] orientation (advisory): body_forward_sign={_fs:+.0f} -> station 0 looks like "
                  f"the {'HEAD' if ok else 'TAIL'}; lateral span {n_lat:.4f} m ({n_n} nodes) vs station "
                  f"{self._K-1} {t_lat:.4f} m ({t_n} nodes), ratio {ratio:.2f}"
                  f"{'  [WEAK: <1.5, coarse sim mesh -- do not rely on this]' if ratio < 1.5 else ''} "
                  f"| vertical spans {n_ver:.4f} vs {t_ver:.4f}", flush=True)
            if not ok:
                print("[AMP] *** WARNING: the sim bend profile looks TAIL-anchored while the reference "
                      "is HEAD-anchored. Set cfg.body_forward_sign to match the asset (+1 = head at "
                      "body +X). The discriminator can separate sim from real on station ORDER alone, "
                      "which no gait can fix. ***", flush=True)
        except Exception as exc:                                      # never let a diagnostic kill a run
            print(f"[AMP] orientation self-check skipped ({exc})", flush=True)
        if names is not None and "dir_nose" in names:
            j = 2 * self._K + names.index("dir_nose")
            rf = torch.tensor(ref["feat"], dtype=torch.float32)[:, j]
            print(f"[AMP] reference is head-first in {float((rf > 0).float().mean()):.1%} of frames "
                  f"(mean dir_nose {float(rf.mean()):+.3f} = median travel-vs-nose angle "
                  f"{float(torch.rad2deg(torch.arccos(rf.median().clamp(-1, 1)))):.1f} deg)", flush=True)

    def _save_disc(self, path):
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"disc": self._disc.state_dict(), "opt": self._disc_opt.state_dict(),
                        "feat_mean": self._feat_mean, "feat_std": self._feat_std,
                        "K": self._K, "disc_updates": self._disc_updates}, path)
        except Exception as e:                                        # never let a save crash training
            print(f"[AMP] disc save failed: {e}", flush=True)

    def _load_disc(self, path):
        if not path or not Path(path).exists():
            print(f"[AMP] no disc checkpoint at {path}; using a fresh discriminator", flush=True)
            return
        ck = torch.load(path, map_location=self.device)
        self._disc.load_state_dict(ck["disc"])
        try:
            self._disc_opt.load_state_dict(ck["opt"])
        except Exception:
            pass
        print(f"[AMP] loaded disc from {path} (trained {ck.get('disc_updates','?')} updates)", flush=True)

    # ---- AMP motion feature: [3D bend profile (2K)] + [body-frame motion (AMP_N_MOTION)] ----
    def _amp_feature(self) -> torch.Tensor:
        """Phi = 2K bend + AMP_N_MOTION motion. Built IDENTICALLY to the 3D reference
        (scripts/zef3d_build_ref.py) -- AMP requires the SAME observation map on real and sim, so any
        change here must be mirrored there and vice versa.

        BEND (2K = 40): left/right + up/down at K=20 body points, in the body's own frame, / TRUE 3D
        arc length, head at point 0.
        MOTION (6): the world velocity expressed in the body's OWN (nose, left, up) basis, split into
        magnitude (speed_bl) + unit direction (dir_nose/left/up), plus the turn rates of the BODY's
        nose axis. See the AMP_MOTION_NAMES block at the top of this file for why.
        """
        pos = self._soft_view.get_simulation_mesh_nodal_positions()
        pos_np = torch.as_tensor(pos, dtype=torch.float32, device=self.device).detach().cpu().numpy()
        root = self.robot.data.root_state_w
        rp = root[:, 0:3].detach().cpu().numpy()
        rq = root[:, 3:7].detach().cpu().numpy()
        # +1 = head at body +X (SimFishLib pipeline fish: auto-skeleton, Misty), -1 = head at -X (the
        # hand-built salmon). Without this the station order is REVERSED for +X-head assets.
        _fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
        out = [self._bend3d_np(pos_np[e], rp[e], rq[e], _fs) for e in range(self.num_envs)]
        bend = np.stack([o[0] for o in out])                                   # (E,2K)
        basis = np.stack([o[1] for o in out])                                  # (E,3,3) rows nose/left/up
        bend_t = torch.as_tensor(bend, dtype=torch.float32, device=self.device)
        basis_t = torch.as_tensor(basis, dtype=torch.float32, device=self.device)
        bl = max(float(getattr(self.cfg, "body_length", 0.5)), 1e-6)
        vw = self.robot.data.root_lin_vel_w                                    # (E,3) world linear vel
        # velocity in the BODY's own basis (BL/s). THIS is what makes head-first, broadside and
        # tail-first three different points in feature space instead of one.
        v_body = torch.einsum("eij,ej->ei", basis_t, vw) / bl                  # (E,3) [nose,left,up]
        speed_bl = (torch.linalg.norm(vw, dim=-1) / bl).clamp(0.0, 10.0).unsqueeze(-1)
        n = torch.linalg.norm(v_body, dim=-1, keepdim=True)
        vdir = torch.where(n > AMP_TRAVEL_MIN_BL, v_body / n.clamp(min=1e-9),
                           torch.zeros_like(v_body)).clamp(-1.0, 1.0)
        # turn rates of the BODY's nose axis (NOT of the velocity direction -- that one is pure noise
        # when the fish is slow, and is decoupled from the body whenever it slides).
        nose_w = basis_t[:, 0, :]
        az = torch.atan2(nose_w[:, 1], nose_w[:, 0])                           # nose azimuth
        el = torch.asin(nose_w[:, 2].clamp(-1.0, 1.0))                         # nose elevation
        dt = max(float(self.cfg.sim.dt) * float(self.cfg.decimation), 1e-6)
        wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))              # to [-pi, pi]
        yaw = torch.where(self._prev_valid, wrap(az - self._prev_hd) / dt, torch.zeros_like(az)).clamp(-10, 10)
        pitch = torch.where(self._prev_valid, wrap(el - self._prev_pit) / dt, torch.zeros_like(el)).clamp(-8, 8)
        self._prev_hd = az.detach()
        self._prev_pit = el.detach()
        return torch.cat([bend_t, speed_bl, vdir,
                          yaw.unsqueeze(-1), pitch.unsqueeze(-1)], dim=1)      # (E, 2K + AMP_N_MOTION)

    def _bend3d_np(self, nodes, root_pos, quat, fwd_sign=-1.0):
        """Per-env 3D bend + the body basis it was measured in.

        Bin the FEM nodes along the NOSE axis into K stations (head at index 0), take the 3D centroid
        per station, express in the body frame (anterior tangent, world-up fixes roll), return
        ([left/right(K), up/down(K)] / true 3D arc length, basis) where basis rows are (nose, left, up)
        as unit world vectors. Mirrors scripts/zef3d_build_ref.py:bend3d exactly.

        `fwd_sign` is cfg.body_forward_sign: +1 when the asset's head is at body +X, -1 when at -X.
        This USED TO BE HARDCODED to -1 (correct for the hand-built salmon), which silently REVERSED
        the profile for every SimFishLib pipeline fish -- their heads are at +X. On the Misty fish the
        feature was tail-anchored while the reference is head-anchored, so the bend envelope grew
        toward index 0 instead of index K-1 and `left` was mirrored: a real-vs-fake tell no gait could
        ever close, independent of how well the fish swims.

        `tan` is the anterior tangent and runs HEAD -> TAIL, so the nose points the other way
        (nose = -tan). Keep measuring lat/ver against `tan` so the bend block is unchanged.
        """
        K = self._K
        w, x, y, z = quat
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        # points at the NOSE, so `along` grows toward the head and the pts[::-1] below (which keeps the
        # highest-along station first) yields head-at-index-0 for BOTH head conventions.
        fwd3 = R @ np.array([fwd_sign, 0.0, 0.0]); fwd3 /= np.linalg.norm(fwd3) + 1e-9
        along = (nodes - root_pos) @ fwd3
        lo, hi = np.percentile(along, 1), np.percentile(along, 99)
        edges = np.linspace(lo, hi, K + 1)
        pts = np.full((K, 3), np.nan)
        for i in range(K):
            sel = (along >= edges[i]) & (along <= edges[i + 1])
            if sel.sum() > 2:
                pts[i] = nodes[sel].mean(0)
        pts = pts[::-1]                                                        # head (nose, high along) first
        good = ~np.isnan(pts[:, 0])
        _eye = np.eye(3)
        if good.sum() < 6:
            return np.zeros(2 * K, np.float32), _eye
        if good.sum() < K:
            idx = np.where(good)[0]
            for k in range(3):
                pts[:, k] = np.interp(np.arange(K), idx, pts[good, k])
        head = pts[0]; rel = pts - head
        tan = rel[max(1, K // 4)] - rel[0]; tan /= np.linalg.norm(tan) + 1e-9
        up = np.array([0.0, 0.0, 1.0]); up = up - up.dot(tan) * tan; up /= np.linalg.norm(up) + 1e-9
        left = np.cross(up, tan)
        bl = max(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum(), 1e-6)
        bend = np.concatenate([(rel @ left) / bl, (rel @ up) / bl]).astype(np.float32)   # (2K,)
        return bend, np.stack([-tan, left, up])                                # nose = -anterior tangent

    # ---- kinematic-imitation traveling-wave target (phase clock) ----
    def _load_track(self):
        env = np.load(self.cfg.bend_envelope_path)                            # (K,) reference envelope
        self._track_env = torch.tensor(env, dtype=torch.float32, device=self.device)
        self._track_s = torch.linspace(0.0, 1.0, self._K, device=self.device)
        self._track_omega = 2.0 * np.pi * float(self.cfg.track_freq)
        self._track_k = float(self.cfg.track_wavenumber)
        self._track_env_ms = float((self._track_env ** 2).mean())             # for error normalization

    def _track_phase(self) -> torch.Tensor:
        dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        return self._phase0 + self._track_omega * self._ep_step.float() * dt   # (E,)

    def _track_target(self, phase: torch.Tensor) -> torch.Tensor:
        # traveling wave: A(s) * sin(phase + k*s)  -> head->tail wave matching the reference
        return self._track_env[None, :] * torch.sin(phase[:, None] + self._track_k * self._track_s[None, :])

    # ---- Reference State Initialization (RSI): full-state swimming seed for the reset ----
    def _load_rsi(self):
        """Lazy-load the recorded scripted-gait full-state reference (root, joints, FEM nodal pos+vel).
        env0-LOCAL (world - env0 origin), exactly as build_scripted_gait_rsi.py / play.py --record_demo
        store it; _apply_rsi re-adds each env's origin. Sets _rsi_ok False if the file is missing/short."""
        self._rsi_ok = False
        p = getattr(self.cfg, "rsi_reference_path", None)
        if not p or not Path(p).exists():
            print(f"[AMP][RSI] reference not found at {p}; RSI DISABLED (all resets from the still pose)",
                  flush=True)
            return
        d = np.load(p)
        dev = self.device
        self._rsi_root = torch.tensor(d["root"], dtype=torch.float32, device=dev)          # (T,13)
        self._rsi_jpos = torch.tensor(d["joint_pos"], dtype=torch.float32, device=dev)      # (T,J)
        self._rsi_jvel = torch.tensor(d["joint_vel"], dtype=torch.float32, device=dev)      # (T,J)
        self._rsi_pc = torch.tensor(d["pc"], dtype=torch.float32, device=dev)               # (T,N,3)
        self._rsi_nvel = torch.tensor(d["nodal_vel"], dtype=torch.float32, device=dev)      # (T,N,3)
        self._rsi_T = int(self._rsi_root.shape[0])
        self._rsi_freq = float(getattr(self.cfg, "rsi_gait_freq", 1.5))
        self._rsi_dt = float(d["step_dt"]) if "step_dt" in d else (
            float(self.cfg.sim.dt) * float(self.cfg.decimation))
        ok_j = self._rsi_jpos.shape[1] == self.robot.num_joints
        self._rsi_ok = ok_j and self._rsi_T > 1
        print(f"[AMP][RSI] loaded {self._rsi_T} full-state frames (J={self._rsi_jpos.shape[1]} "
              f"N={self._rsi_pc.shape[1]}) from {p}; frac={self.cfg.rsi_frac} freq={self._rsi_freq}Hz "
              f"-> RSI {'ON' if self._rsi_ok else 'OFF (joint count mismatch)'}", flush=True)

    def _apply_rsi(self, env_ids: torch.Tensor, k: torch.Tensor):
        """Place envs `env_ids` at recorded gait frames `k` (full state -> FEM-consistent, no attachment
        strain). Also set _phase0 so the kinematic-imitation clock CONTINUES the seed gait's phase, so
        r_track starts near-in-phase (removing the frozen local optimum, where a still fish scored higher
        than any not-yet-phase-locked motion)."""
        origins = self.scene.env_origins[env_ids]                                   # (m,3)
        root = self._rsi_root[k].clone()                                            # (m,13) env0-local
        root[:, 0:3] = root[:, 0:3] + origins                                       # -> world per env
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:13], env_ids)
        jpos = self._rsi_jpos[k]
        jvel = self._rsi_jvel[k]
        self.robot.write_joint_state_to_sim(jpos, jvel, None, env_ids)
        if getattr(self, "_soft_view", None) is not None:
            try:
                pos = self._soft_view.get_simulation_mesh_nodal_positions()
                vel = self._soft_view.get_simulation_mesh_nodal_velocities()
                pos[env_ids] = (self._rsi_pc[k] + origins[:, None, :]).to(pos.dtype)
                vel[env_ids] = self._rsi_nvel[k].to(vel.dtype)
                self._soft_view.set_simulation_mesh_nodal_positions(pos)
                self._soft_view.set_simulation_mesh_nodal_velocities(vel)
            except Exception as e:  # noqa: BLE001  -- fall back to joint-only seed if the FEM set is rejected
                print(f"[AMP][RSI] FEM state-set failed: {e}", flush=True)
        # align the tracking clock to the seed gait's temporal phase (2*pi*f*k*dt); the spatial-wavenumber
        # mismatch between the scripted gait and the ZeF reference envelope is a small residual.
        self._phase0[env_ids] = (2.0 * np.pi * self._rsi_freq * k.float() * self._rsi_dt) % (2.0 * np.pi)

    # ---- wall-distance rays (analytic, axis-aligned square of half-size self._half) ----
    def _wall_rays(self) -> torch.Tensor:
        if not getattr(self.cfg, "spawn_tank_walls", True):
            # open water: no walls anywhere -> max distance on every ray (also zeroes the wall penalty)
            return torch.ones(self.num_envs, self._R, device=self.device)
        root = self.robot.data.root_state_w
        p = (root[:, 0:2] - self.scene.env_origins[:, 0:2])                     # (E,2) env-local xy
        fwd = math_utils.quat_apply(root[:, 3:7], torch.tensor([1.0, 0.0, 0.0], device=self.device)
                                    .expand(self.num_envs, 3))
        yaw = torch.atan2(fwd[:, 1], fwd[:, 0])                                 # (E,)
        ang = yaw[:, None] + self._ray_body_ang[None, :]                        # (E,R) world ray angle
        dx = torch.cos(ang); dy = torch.sin(ang)
        eps = 1e-6
        H = self._half
        tx = (torch.sign(dx) * H - p[:, None, 0]) / torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        ty = (torch.sign(dy) * H - p[:, None, 1]) / torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)
        t = torch.minimum(tx, ty).clamp(min=0.0, max=2 * H)                     # (E,R) distance to wall
        return t / (2 * H)                                                      # normalized [0,1]

    def _get_observations(self) -> dict:
        grav = self.robot.data.projected_gravity_b
        linv = self.robot.data.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel
        angv = self.robot.data.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel
        jpos = (self.robot.data.joint_pos - self._default_joint_pos) * self.cfg.obs_scales.joint_pos
        jvel = self.robot.data.joint_vel * self.cfg.obs_scales.joint_vel
        rays = self._wall_rays()
        # vertical containment: attitude (body-frame world-up; grav is zero in zero-g) + depth error
        root = self.robot.data.root_state_w
        up_w = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        up_b = math_utils.quat_apply_inverse(root[:, 3:7], up_w)
        z_err = ((root[:, 2] - self.scene.env_origins[:, 2]) - float(self.cfg.target_root_height))
        cols = [grav, linv, angv, jpos, jvel, rays, up_b, z_err.unsqueeze(-1)]     # 75 dims (pre-phase)
        # kinematic-imitation phase clock (sin, cos) so the policy knows the target-wave phase to track.
        # Optional (use_phase_clock): OFF -> obs=75, matching the pre-kinematic July-18 swimmer checkpoints
        # so they load/deploy in this env (the phase dims are appended LAST, so obs[:75] is unchanged).
        if getattr(self.cfg, "use_phase_clock", True):
            if self._track_env is None:
                self._load_track()
            phase = self._track_phase()
            cols += [torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)]
        obs = torch.nan_to_num(torch.cat(cols, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    # ---- reward: style + cruise + wall + energy + alive + stability ----
    def _get_rewards(self) -> torch.Tensor:
        if not hasattr(self, "_disc"):
            self._init_disc()
        c = self.cfg
        dev = self.device
        feat = self._amp_feature()                                             # (E, 2K + AMP_N_MOTION)
        featn = (feat - self._feat_mean) / self._feat_std

        # AMP style reward from the env-owned discriminator (LSGAN style reward), only for valid pairs
        r_style = torch.zeros(self.num_envs, device=dev)
        trans = torch.cat([(self._prev_feat - self._feat_mean) / self._feat_std, featn], dim=1)  # (E,2K)
        with torch.no_grad():
            score = self._disc(trans).squeeze(-1)
            rs = (1.0 - 0.25 * (score - 1.0) ** 2).clamp(0.0, 1.0)
        r_style = torch.where(self._prev_valid, rs, torch.zeros_like(rs))
        r_style = torch.nan_to_num(r_style)

        # ---- movement: heading-aligned FORWARD-velocity tracking (replaces speed-magnitude) ----
        root_state = self.robot.data.root_state_w
        root_vel = self.robot.data.root_lin_vel_w                               # (E,3) world lin vel
        # heading = robot forward (-X body axis) in world, normalized (empirically the nose direction)
        fwd_b = torch.tensor([-1.0, 0.0, 0.0], device=dev, dtype=root_vel.dtype).expand(self.num_envs, 3)
        heading = math_utils.quat_apply(root_state[:, 3:7], fwd_b)              # (E,3)
        heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp(min=1e-6)
        v_forward = torch.sum(root_vel * heading, dim=-1)                       # (E,) signed forward speed
        # forward-progress reward: MONOTONIC from standstill (constant gradient ~1/target, so PPO is
        # actually pulled toward moving forward -- the original narrow Gaussian was ~flat near v=0 and
        # the policy converged to hovering), peaking at the target speed and falling off past it (tent).
        tgt = max(float(c.keepmove_target_speed), 1e-6)
        ratio = (v_forward / tgt).clamp(min=0.0)
        r_move = torch.minimum(ratio, 2.0 - ratio).clamp(0.0, 1.0)   # logged only (w_keepmove=0)

        # PROGRESS reward (the only task term): forward speed = body -X component (head is at -X --
        # verified from the bone layout: bone1 head at x=-0.138, thin peduncle + caudal fin at +x), in
        # BL/s, through a SATURATING tanh. SIGNED (no clamp): the spec's clamp(min=0) makes r_progress a
        # FLAT ZERO with ZERO GRADIENT for any backward velocity -> a dead-zone trap. Across 4 runs the
        # policy settled into a tail-first (+x) backward drift and the clamped term gave NO gradient to
        # escape it. tanh(v/sat) is still saturating and still "small" (weight 0.3), but now gently
        # penalizes backward with a real gradient everywhere, breaking the fwd/back symmetry the disc
        # (body-shape-only, direction-agnostic) cannot. AMP still owns the gait shape.
        v_forward_b = -self.robot.data.root_lin_vel_b[:, 0]                     # body -X = nose forward
        v_forward_bl = v_forward_b / max(float(c.body_length), 1e-6)
        r_progress = torch.tanh(v_forward_bl / max(float(c.progress_saturation_speed), 1e-6))

        # KINEMATIC-IMITATION tracking reward (PRIMARY, remedy b). Match the fish's bend profile to a
        # phase-clocked traveling-wave target built from the reference envelope. ANTI-noise: exploration
        # noise degrades the match, so the deterministic MEAN must produce the undulation (fixes freeze).
        if self._track_env is None:
            self._load_track()
        phase = self._track_phase()                                          # same phase the obs saw
        bend_target = self._track_target(phase)                              # (E,K)
        fish_bend = feat[:, :self._K]
        track_err = ((fish_bend - bend_target) ** 2).mean(dim=1) / max(self._track_env_ms, 1e-9)
        r_track = torch.where(self._prev_valid, torch.exp(-float(c.track_sharpness) * track_err),
                              torch.zeros_like(track_err))

        # NET-FORWARD-DISPLACEMENT reward (the real anti-freeze). Net forward travel over a W-step
        # window (~2 s), in BL/s. Noise averages to ~0 net displacement, so ONLY a coherent MEAN gait
        # earns it -- unlike step-to-step activity that exploration noise satisfies. MODEST weight:
        # r_style still owns the gait SHAPE; this only makes "move coherently" outrank "hold still".
        pos_now = root_state[:, 0:3]
        pos_old = self._pos_ring[:, self._ring_ptr]                            # position W steps ago
        disp_vec = pos_now - pos_old
        dt_win = max(self._disp_W * float(self.cfg.sim.dt) * float(self.cfg.decimation), 1e-6)
        # ANTI-FREEZE = net HORIZONTAL displacement MAGNITUDE over the window (any direction), NOT the
        # signed forward component. Un-gameable by exploration noise (a jittering-but-stationary MEAN
        # random-walks to ~0 net displacement) AND does not demand forward-straight swimming ("real fish
        # don't necessarily swim straight and forward") NOR penalize the backward-drifting RSI seed gait
        # (a signed-forward reward made r_displacement NEGATIVE on the seed -> the policy abandoned it).
        # Horizontal (xy) only, so it rewards cruising in the swim plane, not a vertical lift-launch.
        net_disp = torch.linalg.norm(disp_vec[:, :2], dim=-1)                 # m, horizontal magnitude
        net_disp_bl = (net_disp / max(float(c.body_length), 1e-6)) / dt_win   # BL/s (window-averaged speed)
        net_fwd = torch.sum(disp_vec * heading, dim=-1)                       # signed fwd, kept for logging
        net_fwd_bl = (net_fwd / max(float(c.body_length), 1e-6)) / dt_win
        disp_valid = self._ep_step >= self._disp_W                            # ring filled with clean history
        r_displacement = torch.where(disp_valid,
                                     torch.tanh(net_disp_bl / max(float(c.displacement_sat), 1e-6)),
                                     torch.zeros_like(net_disp_bl))
        self._pos_ring[:, self._ring_ptr] = pos_now.detach()                  # write current pos
        self._ring_ptr = (self._ring_ptr + 1) % self._disp_W
        self._ep_step = self._ep_step + 1

        # ANTI-FREEZE activity reward. Without it the policy COLLAPSES TO A NEAR-STATIC POSE (0.001 BL/s,
        # tail osc 0.01 BL): forward motion is hydro-limited so r_progress is unreachable, the disc
        # saturates to a constant, and the energy/action penalties then reward stillness -> the fish
        # freezes (the exact velocity-feature-absent collapse the AMP paper documents). This rewards
        # UNDULATION VIGOR directly -- the mean frame-to-frame change of the bend profile, saturating at
        # the reference's activity level -- so a real, visible tail beat is rewarded regardless of the
        # (saturated) disc. w_activity is modest; AMP still shapes WHICH undulation.
        bend_activity = (feat[:, :self._nbend] - self._prev_feat[:, :self._nbend]).abs().mean(dim=1)  # 40 bend dims
        r_activity = torch.where(self._prev_valid,
                                 torch.tanh(bend_activity / max(float(c.activity_target), 1e-6)),
                                 torch.zeros_like(bend_activity))

        # lateral slip: velocity component orthogonal to heading -> small squared-speed penalty.
        # Clip the speed before squaring (like the accel penalty) so this downside term stays bounded
        # even when root velocity is large-but-finite near a blow-up (nan_to_num only catches NaN/Inf).
        v_lateral = root_vel - v_forward.unsqueeze(-1) * heading               # (E,3)
        lateral_speed = torch.linalg.norm(v_lateral, dim=-1)                   # (E,)
        pen_lateral = lateral_speed.clamp(max=float(c.lateral_clip)) ** 2

        # SLIP-RATIO penalty. The absolute-speed penalty above is quadratic in a quantity that sits
        # around 0.13 m/s, so it evaluates to ~0.016 -- it vanishes exactly in the regime where it is
        # needed, and the policy learned to CRAB (62 deg angle of attack, lateral speed > forward).
        # The ratio |v_lat|/|v| is scale-free and near-maximal for that gait, so it actually bites.
        # Gated on genuinely moving: at ~0 speed the ratio is numerical noise, and punishing it would
        # penalize a stationary fish for float dust.
        speed3d = torch.linalg.norm(root_vel, dim=-1)
        slip_ratio = lateral_speed / speed3d.clamp(min=1e-6)
        moving = speed3d > float(c.slip_min_speed)
        pen_slip = torch.where(moving, slip_ratio.clamp(0.0, 1.0) ** 2, torch.zeros_like(slip_ratio))

        # excessive root acceleration: finite diff of world velocity, tolerance + clip, gated on reset.
        dt = max(float(self.cfg.sim.dt) * float(self.cfg.decimation), 1e-6)
        accel = (root_vel - self._prev_root_vel) / dt                          # (E,3)
        accel_mag = torch.linalg.norm(accel, dim=-1)                           # (E,)
        accel_excess = (accel_mag - float(c.accel_threshold)).clamp(min=0.0, max=float(c.accel_clip))
        pen_accel = torch.where(self._prev_valid, accel_excess ** 2, torch.zeros_like(accel_excess))
        # keep the old horizontal-speed gauge for logging comparability
        speed = torch.linalg.norm(root_vel[:, 0:2], dim=1)

        # wall proximity/contact penalty (nearest ray clearance in metres)
        clr = self._wall_rays().min(dim=1).values * (2 * self._half)
        pen_wall = ((float(c.wall_margin) - clr) / float(c.wall_margin)).clamp(0.0, 1.0)

        # vertical containment: squared z-error beyond the free band around the swim height
        z_local = root_state[:, 2] - self.scene.env_origins[:, 2]
        z_dev = (z_local - float(c.target_root_height)).abs()
        pen_z = ((z_dev - float(c.z_band)).clamp(min=0.0, max=2.0)) ** 2

        # STATIC-BEND penalty. The policy parked its joints at up to 11.7 deg of a +-15 deg range and
        # wobbled only +-2.9 deg around that: a permanently bent body, which acts as a rudder (it
        # circled -265 deg in 7 s) and as a cambered foil (sideways lift). A traveling wave averages
        # to ~0 over a beat, so an EMA of the bend profile penalizes the parked curve WITHOUT taxing
        # the oscillation that actually produces thrust.
        a = float(c.bend_ema_alpha)
        bend_only = feat[:, :self._nbend].detach()                                # 40 bend dims (exclude motion)
        self._bend_ema = torch.where(
            self._prev_valid.unsqueeze(-1),
            (1.0 - a) * self._bend_ema + a * bend_only,
            bend_only,
        )
        pen_bend = torch.where(self._prev_valid, (self._bend_ema ** 2).mean(dim=1),
                               torch.zeros(self.num_envs, device=dev))

        # ENERGY penalty: mechanical power P = sum_j |tau_j * omega_j|
        tau = getattr(self.robot.data, "applied_torque", None)
        if tau is None:
            tau = torch.zeros_like(self.robot.data.joint_vel)
        power = (tau.abs() * self.robot.data.joint_vel.abs()).sum(dim=1)

        # stability shaping. jvel penalty is THRESHOLDED: zero for a healthy gait (rms ~2-4 rad/s),
        # biting only as jvel_rms approaches the blow-up regime -- so it does not suppress the 2.88 Hz
        # gait (verified in scripts/smoke_amp_reward.py). act_rate stays a small smoothness term.
        jvel_rms = torch.sqrt((self.robot.data.joint_vel ** 2).mean(dim=1) + 1e-12)
        jvel_pen = (jvel_rms - float(getattr(c, "jvel_soft_threshold", 12.0))).clamp(min=0.0) ** 2
        act_rate = ((self.actions - self._prev_actions) ** 2).mean(dim=1)

        # AMP-PRIMARY reward: r_style (weight 1) is the objective; r_progress (0.1) breaks fwd/back
        # symmetry; the rest are minimal numerical safety (depth band, energy, jvel, action-rate, alive).
        # keepmove/lateral/slip/accel/bend are weight-0 (computed above for logging only).
        reward = (float(c.w_style) * r_style
                  + float(getattr(c, "w_track", 0.0)) * r_track
                  + float(c.w_progress) * r_progress
                  + float(getattr(c, "w_displacement", 0.0)) * r_displacement
                  + float(getattr(c, "w_activity", 0.0)) * r_activity
                  + float(c.w_keepmove) * r_move
                  - float(c.w_lateral) * pen_lateral
                  - float(c.w_slip) * pen_slip
                  - float(c.w_bend) * pen_bend
                  - float(c.w_accel) * pen_accel
                  - float(c.w_wall) * pen_wall
                  - float(c.w_zband) * pen_z
                  - float(c.w_energy) * power
                  + float(c.w_alive)
                  - float(c.w_jvel) * jvel_pen
                  - float(c.w_action_rate) * act_rate)

        # blow-up = a rare numerical event, NOT the policy's fault. The OLD handling zeroed the reward
        # in-place, which made a vigorous fish (which occasionally risks a zeroed step) STRICTLY worse
        # than a frozen one (which never does) -- a penalty-side stillness basin. NEW handling: the
        # blow-up step gets a NEUTRAL reward (the alive bonus, so no punishment for the divergence), and
        # the episode is TRUNCATED in _get_dones (value bootstrap, not value-0) so vigor is not taxed.
        blew = ((self.robot.data.joint_vel.abs().amax(dim=1) > float(c.joint_velocity_limit))
                | (~torch.isfinite(root_state).all(dim=1))
                | (~torch.isfinite(self.robot.data.joint_vel).all(dim=1))
                | (torch.linalg.norm(root_vel, dim=-1) > float(c.root_speed_limit)))
        _fvl = getattr(c, "fem_velocity_limit", None)
        if _fvl is not None and getattr(self, "_soft_view", None) is not None:
            _nvel = torch.as_tensor(self._soft_view.get_simulation_mesh_nodal_velocities(),
                                    device=dev, dtype=torch.float32)
            _nvmax = torch.nan_to_num(_nvel, nan=float("inf")).abs().flatten(1).amax(dim=1)
            blew = blew | (_nvmax > float(_fvl))
        pen_blowup = blew.float()
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        # neutral (not zeroed, not the garbage-penalty computed on the diverged state)
        reward = torch.where(blew, torch.full_like(reward, float(c.w_alive)), reward)

        # ---- push valid transitions to the replay buffer + periodic discriminator update ----
        # exclude envs blowing up THIS step: their bend feature is computed from a diverged mesh and
        # must not enter the discriminator's "policy motion" distribution.
        valid = self._prev_valid & (~blew)
        if valid.any():
            add = trans[valid].detach()
            m = add.shape[0]
            idx = (torch.arange(m, device=dev) + self._buf_ptr) % self._buf.shape[0]
            self._buf[idx] = add
            self._buf_ptr = int((self._buf_ptr + m) % self._buf.shape[0])
            self._buf_n = min(self._buf_n + m, self._buf.shape[0])
        self._amp_step += 1
        # Skip discriminator training during play/eval (rollouts run under torch.inference_mode,
        # where the grad-penalty autograd call raises "does not require grad").
        if (self._buf_n >= int(c.disc_batch) and self._amp_step % int(c.disc_update_every) == 0
                and not torch.is_inference_mode_enabled()):
            self._update_disc()

        # advance prev-state
        self._prev_feat = feat.detach()
        self._prev_valid = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        self._prev_actions = self.actions.detach().clone()
        self._prev_root_vel = root_vel.detach().clone()

        # L2 cooldown: zero the reward AND mask the disc-buffer push (via _prev_valid) for envs still
        # settling after a blow-up reset, then count the window down. The blow-up penalty on the
        # diverged step itself was already charged above (cooldown arms afterwards in _reset_idx).
        if hasattr(self, "_cooldown"):
            cd = self._cooldown > 0
            reward = torch.where(cd, torch.zeros_like(reward), reward)
            self._prev_valid = self._prev_valid & (~cd)
            self._cooldown = (self._cooldown - 1).clamp_min(0)

        self.extras.setdefault("log", {})
        self.extras["log"]["fem_clamp_hits"] = torch.tensor(float(getattr(self, "_fem_clamp_hits", 0)))
        self.extras["log"]["cooldown_active"] = (self._cooldown > 0).float().mean().detach()
        self.extras["log"]["r_style"] = r_style.mean().detach()
        self.extras["log"]["r_progress"] = r_progress.mean().detach()
        self.extras["log"]["r_track"] = r_track.mean().detach()
        self.extras["log"]["track_err"] = track_err.mean().detach()
        self.extras["log"]["r_displacement"] = r_displacement.mean().detach()
        self.extras["log"]["net_fwd_bl"] = net_fwd_bl.mean().detach()
        self.extras["log"]["net_disp_bl"] = net_disp_bl.mean().detach()      # any-direction cruise speed
        self.extras["log"]["r_activity"] = r_activity.mean().detach()
        # sanity gauges: the displacement term must out-weigh the smoothness/jvel penalties at low speed
        self.extras["log"]["pen_actrate_w"] = (float(c.w_action_rate) * act_rate).mean().detach()
        self.extras["log"]["r_disp_w"] = (float(getattr(c, "w_displacement", 0.0)) * r_displacement).mean().detach()
        self.extras["log"]["bend_activity"] = bend_activity.mean().detach()
        self.extras["log"]["v_forward_bl"] = v_forward_bl.mean().detach()
        self.extras["log"]["jvel_pen"] = jvel_pen.mean().detach()
        self.extras["log"]["r_move"] = r_move.mean().detach()
        self.extras["log"]["wall_pen"] = pen_wall.mean().detach()
        self.extras["log"]["energy"] = power.mean().detach()
        self.extras["log"]["cruise_speed"] = speed.mean().detach()
        # new movement-reward diagnostics
        self.extras["log"]["v_forward"] = v_forward.mean().detach()
        self.extras["log"]["lateral_speed"] = lateral_speed.mean().detach()
        self.extras["log"]["accel_mag"] = accel_mag.mean().detach()
        self.extras["log"]["pen_lateral"] = pen_lateral.mean().detach()
        self.extras["log"]["pen_slip"] = pen_slip.mean().detach()
        self.extras["log"]["slip_ratio"] = slip_ratio.clamp(0.0, 1.0).mean().detach()
        self.extras["log"]["pen_bend"] = pen_bend.mean().detach()
        self.extras["log"]["bend_ema_abs"] = self._bend_ema.abs().mean().detach()
        # DISC HEALTH: r_style was flat at ~0.485 for 100+ epochs (implied score pinned at -0.44 =
        # "fake"), i.e. the discriminator had saturated and the style term was a CONSTANT -> zero
        # policy gradient. Log the score's spread across envs, not just its mean, so a saturated
        # disc (std -> 0) is visible instead of being mistaken for a healthy mid-range reward.
        self.extras["log"]["disc_score"] = score.mean().detach()
        self.extras["log"]["disc_score_std"] = score.std().detach()
        self.extras["log"]["pen_accel"] = pen_accel.mean().detach()
        self.extras["log"]["blowup_frac"] = pen_blowup.mean().detach()
        self.extras["log"]["pen_z"] = pen_z.mean().detach()
        self.extras["log"]["z_local"] = z_local.mean().detach()
        return reward

    def _update_disc(self):
        c = self.cfg
        dev = self.device
        b = int(c.disc_batch)
        # instance noise DECAYED to 0 over disc_noise_decay updates -> strong early (breaks a premature
        # separator), gone late (so it doesn't cap the final gait quality).
        ns0 = float(getattr(c, "disc_noise_std", 0.0))
        decay = float(getattr(c, "disc_noise_decay", 2000.0))
        ns = ns0 * max(0.0, 1.0 - self._disc_updates / max(decay, 1.0))
        r1 = float(getattr(c, "disc_r1", getattr(c, "disc_grad_penalty", 5.0)))
        real_lbl = float(getattr(c, "disc_real_label", 0.9))                   # SOFT label (0.9 not 1.0)
        d_real = d_fake = None
        with torch.enable_grad():
            for _ in range(int(c.disc_updates_per_burst)):
                fi = torch.randint(0, self._buf_n, (b,), device=dev)
                fake = self._buf[fi]
                ri = torch.randint(0, self._ref_trans.shape[0], (b,), device=dev)
                real = self._ref_trans[ri].clone()
                if ns > 0:
                    real = real + ns * torch.randn_like(real)
                    fake = fake + ns * torch.randn_like(fake)
                real = real.requires_grad_(True)
                d_real = self._disc(real)
                d_fake = self._disc(fake)
                # LSGAN with soft real label; fake target -real_lbl (symmetric, less overconfident)
                loss = 0.5 * ((d_real - real_lbl) ** 2).mean() + 0.5 * ((d_fake + real_lbl) ** 2).mean()
                # R1: zero-centered gradient penalty on REAL inputs (Mescheder 2018) -- the main
                # saturation control; R1/2 * E[||grad_D(real)||^2].
                grad = torch.autograd.grad(d_real.sum(), real, create_graph=True)[0]
                loss = loss + 0.5 * r1 * (grad.norm(dim=1) ** 2).mean()
                self._disc_opt.zero_grad(set_to_none=True)
                loss.backward()
                self._disc_opt.step()
        self._disc_updates += 1
        self.extras.setdefault("log", {})
        with torch.no_grad():
            self.extras["log"]["disc_loss"] = loss.detach()
            self.extras["log"]["disc_acc"] = 0.5 * ((d_real > 0).float().mean() + (d_fake < 0).float().mean())
            # SUCCESS-CRITERIA metrics: D-spread over the fake batch (must stay > 0 = not saturated),
            # and the mean D(real)/D(fake) gap (should be FINITE, e.g. ~+0.5 / ~-0.5, not pinned).
            self.extras["log"]["D_real_mean"] = d_real.mean().detach()
            self.extras["log"]["D_fake_mean"] = d_fake.mean().detach()
            self.extras["log"]["D_fake_spread"] = d_fake.std().detach()
            self.extras["log"]["D_gap"] = (d_real.mean() - d_fake.mean()).detach()
            self.extras["log"]["disc_noise"] = torch.tensor(ns)
        if int(getattr(c, "disc_ckpt_every", 0)) > 0 and self._disc_updates % int(c.disc_ckpt_every) == 0:
            self._save_disc(getattr(c, "disc_ckpt_path", None))

    # ---- failure-only termination (no target success); reuse base throughput/records ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _term, time_out = super()._get_dones()
        root = self.robot.data.root_state_w
        local = root[:, 0:3] - self.scene.env_origins
        oob = torch.linalg.norm(local, dim=1) > self.cfg.root_position_limit
        # vertical escape: the tank walls span z in [z_min, z_max] with open top/bottom -- leaving
        # that band is out-of-bounds (the containment penalty + z obs discourage it; this backstops it)
        zoob = (local[:, 2] < float(self.cfg.z_min)) | (local[:, 2] > float(self.cfg.z_max))
        blow = self.robot.data.joint_vel.abs().max(dim=1).values > self.cfg.joint_velocity_limit
        nonf = (~torch.isfinite(root).all(dim=1)) | (~torch.isfinite(self.robot.data.joint_vel).all(dim=1))
        # non-physical root speed -> treated like a blow-up (dismissed, cooldown-settled)
        fast = torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=-1) > float(self.cfg.root_speed_limit)
        # also terminate on a pure-FEM blow-up (nodal-velocity guard, computed by the base _get_dones)
        fem = getattr(self, "_fem_blowup", torch.zeros_like(oob))
        # stash the BLOW-UP-only mask (not oob/timeout) so _reset_idx arms the cooldown only on blow-ups
        self._blowup_mask = (blow | nonf | fem | fast)
        # PLAY-FOREVER viewing mode (cfg.never_reset): never end an episode on timeout / out-of-bounds /
        # speed -- only a genuine numerical blow-up (NaN / FEM / joint explosion) resets, to avoid NaN
        # propagation. Lets the fish swim continuously for watching. NOT for training (no learning signal).
        if getattr(self.cfg, "never_reset", False):
            hard = blow | nonf | fem
            return hard, torch.zeros_like(hard)
        # FAILURE end conditions are TRUNCATIONS (value bootstrap), NOT value-0 failures: blow-ups are
        # numerical (not the policy's fault) and the position/z limits are just the arena boundary.
        # Value-0-terminating on THOSE would make stillness strictly safer than motion (a moving fish
        # risks a limit/blow-up), the penalty-side stillness basin -- so they bootstrap.
        truncate = (time_out | _term | oob | zoob | blow | nonf | fem | fast)
        # EARLY TERMINATION on freeze is the OPPOSITE bias, and intentional: value-0-terminate ONLY when
        # the fish STOPS undulating (live-DOF stall past the grace window), so STILLNESS -- not motion --
        # forfeits future reward. This is what dissolves the "frozen policy is optimal" maximum.
        terminate = torch.zeros_like(oob)
        if getattr(self.cfg, "et_enable", False) and hasattr(self, "_freeze_ctr"):
            act_live = self.robot.data.joint_vel[:, self._live_dof].abs().mean(dim=1)   # rad/s undulation
            frozen_now = act_live < float(self.cfg.et_freeze_floor)
            self._freeze_ctr = torch.where(frozen_now, self._freeze_ctr + 1,
                                           torch.zeros_like(self._freeze_ctr))
            terminate = ((self._freeze_ctr >= int(self.cfg.et_freeze_patience))
                         & (self._ep_step >= int(self.cfg.et_freeze_grace)))
            self.extras.setdefault("log", {})
            self.extras["log"]["et_freeze_frac"] = terminate.float().mean().detach()
            self.extras["log"]["act_live_mean"] = act_live.mean().detach()
        # a freeze-terminated env must not ALSO be flagged truncated (mutually exclusive for the wrapper)
        return terminate, (truncate & ~terminate)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # RESET-LOOP SAFEGUARD: zero the FIRST post-reset action so the FEM settles one step before
        # the policy actuates (a just-reset env has _prev_valid=False). Without this, an extreme first
        # action can re-diverge the freshly-reset FEM at dt=1/60, looping blow-up->reset->blow-up.
        if getattr(self, "_prev_valid", None) is not None:
            actions = actions.view(self.num_envs, self._num_actions).clone()
            # zero the action on the first post-reset step AND across a blow-up cooldown window
            actions[(~self._prev_valid) | (self._cooldown > 0)] = 0.0
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_prev_valid") and env_ids is not None:
            self._prev_valid[env_ids] = False
            self._prev_actions[env_ids] = 0.0
            if hasattr(self, "_ep_step"):
                self._ep_step[env_ids] = 0            # displacement window re-fills from clean history
            if hasattr(self, "_freeze_ctr"):
                self._freeze_ctr[env_ids] = 0         # ET freeze counter re-arms per episode
            if hasattr(self, "_prev_hd"):
                self._prev_hd[env_ids] = 0.0          # heading/pitch history for the yaw/pitch rate channels
                self._prev_pit[env_ids] = 0.0
            if hasattr(self, "_phase0"):              # random phase-clock offset per episode (diversity)
                n = env_ids.shape[0] if torch.is_tensor(env_ids) else len(env_ids)
                self._phase0[env_ids] = 2.0 * np.pi * torch.rand(n, device=self.device)
            # zero the accel finite-difference reference for reset envs; combined with the
            # _prev_valid gate this prevents a spurious acceleration spike on the reset step.
            self._prev_root_vel[env_ids] = 0.0
            # L2: arm the cooldown ONLY for envs whose reset was caused by a BLOW-UP (not timeout/oob),
            # so the freshly-reset FEM gets a few zero-action, reward/disc-masked steps to settle.
            bm = getattr(self, "_blowup_mask", None)
            if bm is not None and int(self.cfg.blowup_cooldown_steps) > 0:
                hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                hit[env_ids] = True
                self._cooldown[hit & bm] = int(self.cfg.blowup_cooldown_steps)

            # RSI: seed a fraction of the reset envs mid-gait (full state) so the policy CONTINUES a gait
            # rather than discovering it from frozen. Applied AFTER the base reset (which set the still
            # default pose + reset the FEM), overriding the selected envs' root/joint/FEM state + _phase0.
            if getattr(self.cfg, "rsi_enable", False):
                if not hasattr(self, "_rsi_ok"):
                    self._load_rsi()
                if self._rsi_ok:
                    ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(
                        list(env_ids), device=self.device, dtype=torch.long)
                    frac = float(getattr(self.cfg, "rsi_frac", 0.8))
                    pick = torch.rand(ids.shape[0], device=self.device) < frac        # per-env Bernoulli
                    sel = ids[pick]
                    if sel.numel() > 0:
                        k = torch.randint(0, self._rsi_T, (sel.shape[0],), device=self.device)
                        self._apply_rsi(sel, k)                                        # sets state + _phase0
                        # a mid-gait seed is NOT a blow-up start: clear any cooldown so the policy actuates
                        # immediately (the FEM state is already consistent with the joint pose).
                        self._cooldown[sel] = 0
