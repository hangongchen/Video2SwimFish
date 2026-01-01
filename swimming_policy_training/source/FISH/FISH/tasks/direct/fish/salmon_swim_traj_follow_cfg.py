# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for the TRAJECTORY-FOLLOWING task: given a real ZeF ground-truth path (planted in
the fish's own body frame at episode start, same convention as eval_traj_fidelity.py's target
planting), the agent must track it point-by-point as it advances in time -- NOT reach a single
static point. The GT path (current point + a short lookahead) is IN the observation. The
episode ends the moment the sampled GT path is exhausted, and resets immediately (both in
training and in play) -- see TrajFollowMixin in salmon_swim_traj_follow_env.py.

Real data: outputs/trajectory_fidelity/zef_reference.npz's two long contiguous real runs
(the SAME file eval_traj_fidelity.py/eval_biofidelity_suite.py already use for Chamfer
comparisons) -- a random window of traj_window_s seconds is drawn from one of them each
episode, converted to a body-frame path, then re-planted at the fish's OWN current pose/BL,
exactly mirroring the single-point planting logic already proven correct in this project.

The two cfgs below differ ONLY in the controller block (locomotion representation), matching
the exact same PCA-vs-Joint split used throughout the rest of this project's Reach10 family.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from .salmon_swim_pca_cfg import SalmonSwimPCACfg


@configclass
class _TrajFollowProtocolCfg(SalmonSwimPCACfg):
    """Shared trajectory-following protocol fields."""

    # real ZeF reference (the SAME file already used for eval_traj_fidelity.py/
    # eval_biofidelity_suite.py's Chamfer comparisons -- reusing it, not re-deriving)
    zef_traj_path = str(Path(__file__).resolve().parents[6] / "outputs/trajectory_fidelity/zef_reference.npz")
    traj_window_s = 5.0            # duration of the sampled GT path per episode (real seconds)
    fixed_traj_ti = -1             # >=0: every episode replays the reference window starting at this frame (videos/side-by-side); -1 = random
    traj_lookahead_k = 5           # number of future phase-ahead points included in the obs
                                    # (point 1 is p*_t itself, at +delta; points 2..K step +delta further each)

    # ---- phase-based tracking (replaces the old fixed-time-advancing reference) ----
    # PROBLEM this replaces: advancing the reference 1:1 with real time meant real ZeF's ~3.4
    # BL/s swim speed (far faster than any sim policy here manages, ~0.4-1.2 BL/s) made the
    # reference outrun the fish for the WHOLE episode, so a distance-based reward almost never
    # saw a small distance. Fix: the reference point is now defined by the fish's OWN nearest-
    # point projection onto the path (s_t = argmin_s ||Gamma(s) - p_fish||) plus a small FIXED
    # arc-length lookahead -- it can never be more than delta+(local speed) ahead of the fish by
    # construction, regardless of how slow the policy swims.
    completion_tol_bl = 0.5          # completion requires the fish within this many BL of the path END (not just the projection)
    traj_delta_bl = 0.3            # arc-length (BL) the lookahead point p*_t sits ahead of s_t

    # reward: r_total = w1*exp(-alpha*dist_bl) + w2*cos(theta_fish-theta_traj)
    #                 + w3*(-beta*|v_fish-v_zef|) + w4*(-gamma*||a_t||^2)
    traj_w1 = 1.0                   # track
    traj_w2 = 0.3                   # heading
    traj_w3 = 0.1                   # speed
    traj_w4 = 0.01                  # energy
    traj_alpha = 2.0                # track falloff rate (per BL)
    traj_beta = 1.0                 # speed-error scale (per BL/s)
    traj_gamma = 0.01                # energy penalty scale (per action-norm^2)

    # BL: "close enough" tolerance for the traj/frac_within_tol diagnostic (NOT used in the
    # reward itself -- purely a logged tracking-quality metric, see TrajFollowMixin._get_rewards).
    traj_tolerance_bl = 1.0

    # episode_length_s is a SAFETY BACKSTOP only -- the real termination is "GT path exhausted"
    # (see TrajFollowMixin._get_dones), which fires well before this at traj_window_s scale.
    episode_length_s = 60.0
    curriculum = False
    dist_curriculum = False
    multi_target = False           # unused by this task; kept False so nothing else assumes it


@configclass
class SalmonSwimPCATrajFollowCfg(_TrajFollowProtocolCfg):
    """BLM+RL (PCA) controller: action = bounded-rate deltas on ALL 20 PCA coefficients."""

    pca_num_modes = 20


@configclass
class SalmonSwimJointTrajFollowCfg(_TrajFollowProtocolCfg):
    """Direct per-joint position-target controller (see
    salmon_swim_traj_follow_env.SalmonSwimJointTrajFollowEnv) -- the raw-joint baseline's
    action space. No curvature manifold: RL directly outputs one target per controlled joint
    (nj=7, control_dof_suffix=':1'), zero-centered onto each joint's OWN _soft_joint_limits,
    identical convention to SalmonSwimJointReach10Cfg."""


@configclass
class SalmonSwimCPGTrajFollowCfg(_TrajFollowProtocolCfg):
    """CPG+RL trajectory following: RL controls [dA, df, db] of a traveling-wave CPG defined in
    curvature space and decoded through the SAME ridge kappa->joint map as the PCA controller
    (see salmon_swim_traj_follow_env.SalmonSwimCPGTrajFollowEnv). cpg_params_path = per-fish
    scripts/benchmark/calibrate_cpg_from_fish.py output."""

    pca_num_modes = 20
    cpg_params_path = ""


@configclass
class SalmonSwimJointAMPTrajFollowCfg(SalmonSwimJointTrajFollowCfg):
    """Joint RL + AMP trajectory following: identical to SalmonSwimJointTrajFollowCfg plus an AMP
    style term (reward = task + w_amp * r_style) against the fish's OWN video reference
    (scripts/benchmark/build_amp_ref_from_fish.py -> dataset/<tag>/amp_reference.npz, 46-dim contract).
    Discriminator settings copied from SalmonSwimJointAMPReach10Cfg."""

    amp_reference_path = ""
    profile_len = 20
    w_amp = 0.2
    disc_hidden = (128, 128)
    disc_lr = 1.5e-5
    disc_update_every = 128
    disc_batch = 512
    disc_r1 = 5.0
    disc_noise_std = 0.3
    disc_buffer_size = 40000
    disc_load_on_init = False
    disc_ckpt_path = ""
    obs_midline = False
