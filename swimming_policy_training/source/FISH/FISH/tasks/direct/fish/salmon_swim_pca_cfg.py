# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the 2-PC PCA-manifold PPO swim experiment (SalmonSwimPCAEnv).

DECISIVE EXPERIMENT: can PPO learn sustained straight forward swimming when the action space
is the 2-dim real-fish PCA locomotion manifold (a1, a2) instead of the 21 joint DOFs?

Inherits SalmonSwimAMPMistyPanelsCfg for the ASSET + WATER + FEM setup only (Misty Minnow,
strip-theory panel hydro with the corrected cd_t=0.01, FEM soft body active, GPU buffer
sizing). The AMP/target machinery it also carries is NEVER exercised: SalmonSwimPCAEnv
overrides actions/observations/rewards/dones completely and the AMP discriminator only
exists in SalmonSwimAMPEnv (a different class). Registered as a SEPARATE task id so the
existing AMP experiment stays reproducible.

Action space (bounded-rate coefficient dynamics, all bounds DATA-DRIVEN at env startup from
outputs/zef_manifold/pca_basis.npz -- see cfg fields pca_*):
    a(t+1) = clamp(a(t) + action * da_max,  -a_max, +a_max)      action in [-1,1]^2
    kappa_bl(s,t) = mean(s) + a1 v1(s) + a2 v2(s)
    q(t) = clamp( ridge_pinv(Phi) (kappa - kappa_rest),  joint limits )   [precomputed affine]
Phi / kappa_rest come from the playback calibration (outputs/zef_playback/recording.npz)
-- the exact in-sim calibrated curvature->joint mapping, NOT re-derived.

Task: swim FORWARD in a straight line. No target, no navigation. Reward =
forward body-frame velocity - sustained-lateral-drift penalty - sustained-yaw-rate penalty
- small energy penalty. The drift/yaw penalties are EMA-filtered (tau ~1 s) on purpose:
instantaneous lateral velocity and yaw rate OSCILLATE as part of any undulatory gait, and
penalizing them per-step would punish swimming itself.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from .salmon_swim_amp_misty_panels_cfg import SalmonSwimAMPMistyPanelsCfg

_REPO = str(Path(__file__).resolve().parents[6])          # .../FISH repo root


@configclass
class SalmonSwimPCACfg(SalmonSwimAMPMistyPanelsCfg):
    # ---- action space: only the lateral D6 family is driven (measured: ':1' alone reaches
    # 0.53 BL/s scripted; the other two families make roll/pitch, not thrust) ----
    control_dof_suffix = ":1"

    # "loosen the joint dof to whatever you need": the PC1 tail joints ask for up to ~80 deg at
    # the |a| p99 extremes; +/-45 deg keeps most of the demanded range while staying in the PD
    # drives' verified stable regime (playback held 100% tracking at 25 deg static; the clamp
    # below 45 is applied per-step to the mapped targets).
    joint_limit_deg = 45.0

    # ---- PCA manifold inputs (existing artifacts; loaded, never re-derived) ----
    pca_basis_path = f"{_REPO}/outputs/zef_manifold/pca_basis.npz"
    pca_calib_path = f"{_REPO}/outputs/zef_playback/recording.npz"
    pca_num_modes = 2
    # a_max = this percentile of |coeff| in the ZeF data (heavy-tailed: p99 ~ 3.5 sigma)
    pca_amp_percentile = 99.0
    # da_max per control step = this percentile of the ZeF per-(1/30s) coefficient change.
    # p99 = [8.9, 7.2] allows a FULL-amplitude 3 Hz oscillation (needs 7.3) -- i.e. exactly up
    # to the gait band the PD (corner 3.2 Hz) can track -- while forbidding pose teleports.
    pca_rate_percentile = 99.0
    # ridge factor for the curvature->joint least squares (same value the playback used)
    pca_ridge = 1e-3
    pca_phi_mode = "calib"      # "analytic" = geometric turning-angle Phi (scripts/benchmark/analytic_phi.py) instead of the measured one

    # ---- reward (locomotion only; NO target anywhere) ----
    w_forward = 2.0          # * v_fwd in BL/s (signed: backward swimming is negative reward)
    w_lateral = 1.0          # * |EMA of body-frame lateral+vertical velocity| in BL/s
    w_yawrate = 0.2          # * |EMA of body-frame yaw rate| in rad/s
    w_energy = 0.001         # * mean(controlled joint_vel^2) in (rad/s)^2
    ema_tau_s = 1.0          # EMA time constant for the drift/yaw penalties

    # ---- task setup: open water, no target, straight-swim episodes ----
    episode_length_s = 20.0          # 600 control steps at 30 Hz
    draw_target_marker = False       # no target in this task (kills the marker spawn too)
    marker_success_sphere = False
    random_target = False
    curriculum = False
    dist_curriculum = False
    multi_target = False
    # small forward glide at spawn kills the cold-start tax and makes the forward-velocity
    # signal non-zero from step 0 (the policy must SUSTAIN it against drag to keep earning)
    spawn_glide_speed = 0.05         # m/s ~ 0.1 BL/s
    initial_joint_pos_range = (-0.02, 0.02)
    initial_joint_vel_range = (0.0, 0.0)

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # LOCK the physics rate to the live-run-certified dt=1/120 (decimation 4 -> 30 Hz
        # control) regardless of the FISH_TEST_DT_HZ env var the parent cfg chain reads --
        # this experiment must be reproducible from the cfg alone.
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
