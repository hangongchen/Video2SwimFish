# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for the 10-TARGET controlled benchmark: ZeF-calibrated CPG+RL vs all-PC PCA+RL.

PROTOCOL (identical for both methods; implemented in salmon_swim_reach10_env.TenTargetMixin):
  - each episode = EXACTLY 10 target attempts; the attempt index is NOT in the observation
  - per attempt: reach (dist < 0.2 m) -> success=1; 20 s timeout -> 0; early abort
    (dist > attempt_abort_dist, fixed+documented, same for both methods) -> 0
  - every outcome IMMEDIATELY samples the next target from the fish's current pose
    (no reset, no episode termination); episode ends only after 10 attempts (max 200 s)
  - primary metric: targets_reached / 10
  - targets: same rules as before (uniform +/-15 deg cone about current heading, distance
    ~ N(1.0, 0.25) clamped [0.4, 1.8], radius 0.2); for EVALUATION a seeded per-(env,attempt)
    RELATIVE sequence file makes both methods face identical (bearing-offset, distance) draws
  - reward: the head-first reach reward, unchanged, same weights, no method-specific terms

The two cfgs below differ ONLY in the controller block (locomotion representation).
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from .salmon_swim_pca_reach_cfg import SalmonSwimPCAReachHeadCfg


@configclass
class _Reach10ProtocolCfg(SalmonSwimPCAReachHeadCfg):
    """Shared 10-target protocol fields (both methods inherit these unchanged)."""

    n_targets_per_episode = 10
    attempt_timeout_s = 20.0
    # EARLY ABORT: target is unrecoverable if the fish gets this far from it. Targets spawn at
    # <= 1.8 m; drifting past 3.0 m cannot be recovered within the 20 s budget at these swim
    # speeds. FIXED and identical for both methods -- never tuned per controller.
    attempt_abort_dist = 3.0
    # episode backstop; the protocol itself ends episodes at 10 attempts (<= 200 s)
    episode_length_s = 205.0
    # ROAM ROOM: chaining 10 targets ~1 m apart legitimately walks the fish 10+ m from spawn.
    # The inherited 6 m tank fence TERMINATED successful deterministic chains around target 5-6
    # (31/35 "blow-ups" in the 2026-08-13 eval were this fence; stochastic training fish curl
    # from action noise and stay inside -- a survivorship artifact that faked a train/play gap).
    root_position_limit = 30.0
    # CFL circuit-breaker armed for train AND eval alike (a strong coherent gait can diverge the
    # FEM-articulation coupling; the clamp bounds it instead of letting the guard reset-and-zero).
    fem_nodal_vel_clamp = 4.0
    # deterministic RELATIVE target sequences for evaluation ("" = sample live, training)
    target_seq_path = ""
    # OPTIONAL empirical bearing distribution ("" = original uniform +/-15 deg cone): path to an
    # npz with raw theta_deg samples (e.g. zef_target_angles.npz, computed from real ZeF turns;
    # positive = CCW from heading). Bearings are bootstrap-sampled from it.
    target_angle_dist_path = ""
    # OPTIONAL empirical ELEVATION distribution ("" = flat, fixed target_height, original
    # behavior): path to an npz with raw phi_deg samples (zef_target_elevation.npz, the real
    # 3D-triangulated ZeF-05 body-frame vertical travel-direction angle, dir_up -> arcsin).
    # Bootstrap-sampled independently of the bearing; decomposes the sampled distance into
    # horizontal + vertical components so targets sit above/below the fish, not just around it.
    target_elevation_dist_path = ""
    # OPT-IN body velocity governor layer (None = the hydro/physics is untouched): root
    # linear/yaw velocity ceilings vs the spin-and-sail numerical runaway. See
    # SalmonSwimEnv._body_velocity_governor.
    body_speed_cap_bl = 0.0   # 0.0 = OFF (float, not None: hydra float overrides must type-match)
    body_yaw_cap_rad = 0.0    # 0.0 = OFF
    # Per-bone drag impulse limiter INSIDE the hydro wrench (drag may stop a bone within one
    # substep, never reverse it). Explicit here rather than a hidden default, so the base hydro
    # model stays original for every other task; 0.5 is the value every Reach10 policy trained
    # with since 2026-08-13. Set 0.0 to recover the untouched quadratic-drag model.
    hydro_impulse_beta = 0.5
    # OPT-IN anti-spin reward shaping (0.0 = reward untouched); see the reach reward.
    w_yaw_penalty = 0.0
    yaw_penalty_thresh = 3.0
    w_sat_penalty = 0.0
    w_heading_shape = 0.0    # potential-based turn shaping (U-turn fix); 0 = off
    # the old fixed-length multi-target machinery must not interfere: the mixin owns dones
    multi_target = True                      # (kept True: reward/marker helpers expect it)


@configclass
class SalmonSwimPCAReach10Cfg(_Reach10ProtocolCfg):
    """All-PC PCA controller: action = bounded-rate deltas on ALL 20 PCA coefficients."""

    pca_num_modes = 20                       # THE change vs the old experiment: full basis


@configclass
class SalmonSwimCPGReach10Cfg(_Reach10ProtocolCfg):
    """ZeF-calibrated CPG controller (see salmon_swim_reach10_env.SalmonSwimCPGReach10Env).

    kappa_cpg(s,t) = kappa_mean(s) + A(t) * E(s) * sin(theta(t) + phi(s)) + b(t)
    with E(s), phi(s), f0/f-range, A/b bounds AND rate bounds all measured from the SAME
    ZeF-05 curvature data as the PCA basis (scripts/zef_pca_rl/calibrate_cpg_from_zef.py ->
    outputs/zef_manifold/cpg_params.npz). Joints via the IDENTICAL ridge decoder as PCA.
    RL controls [dA, df, db] (3-D)."""

    cpg_params_path = ""                     # "" -> <repo>/outputs/zef_manifold/cpg_params.npz


@configclass
class SalmonSwimJointReach10Cfg(_Reach10ProtocolCfg):
    """Direct per-joint position-target controller (see
    salmon_swim_reach10_env.SalmonSwimJointReach10Env) -- the IL/BCO baseline's action space.

    No curvature manifold, no CPG parameterization: RL directly outputs one target per
    controlled joint (nj=7 with the inherited control_dof_suffix=':1'), zero-centered onto
    each joint's OWN _soft_joint_limits (action=+1 -> that joint's upper limit, action=-1 ->
    its lower limit), so the REACHABLE joint range is identical to the PCA/CPG controllers'
    (both clamp against the same _soft_joint_limits) -- only the controller parameterization
    differs, which is what makes this a fair drop-in baseline. Every other field (protocol,
    reward, episode length, targets, asset) is inherited UNCHANGED from _Reach10ProtocolCfg."""

    # BCO ablation switch (OFF by default -> reward bit-identical to PCA/CPG, per the class
    # docstring). ON: SalmonSwimJointReach10Env._get_rewards still computes the full task
    # reward (so _prev_distance/_cum_reaches/success bookkeeping and diagnostics are all
    # UNCHANGED) but returns zero to PPO -- i.e. "BCO pure": fine-tuning proceeds with no task
    # reward signal, so success/reward can still be MEASURED for comparison even though the
    # policy isn't optimizing them.
    disable_task_reward = False


@configclass
class SalmonSwimJointAMPReach10Cfg(SalmonSwimJointReach10Cfg):
    """"Joint-space RL + AMP" baseline: IDENTICAL to SalmonSwimJointReach10Cfg (raw 7-joint
    action space, same Reach10 protocol/reward/physics) with ONE addition -- an AMP style-
    imitation term borrowed VERBATIM from SalmonSwimAMPEnv/SalmonAMPTankEnv (see
    salmon_swim_reach10_env.SalmonSwimJointAMPReach10Env):

        reward = <the unchanged Reach10 task reward>  +  w_amp * r_style

    r_style rewards the FEM midline's bend+velocity signature for matching a REAL zebrafish
    reference (outputs/zef05_amp_ref/amp_reference_3d_v2.npz, 46-dim Phi: 40 bend [L/R+U/D at
    K=20 body points] + 6 motion [speed_bl, dir_nose, dir_left, dir_up, yaw_body, pitch_body]),
    via an LSGAN discriminator trained jointly with the policy (self-contained inside the env;
    plain PPO via the existing rl_games/train_ppo.py pipeline needs no changes).

    COMPATIBILITY, verified before this cfg was written (not assumed):
      - _amp_feature/_bend3d_np read raw FEM MESH NODE positions (self._soft_view), not a fixed
        bone count -- they are asset-agnostic and were already proven correct on this exact
        Misty asset by SalmonSwimAMPMistyCfg's own body_forward_sign=+1.0 comment (head at +X).
      - Reach10's Misty tasks already run with_deformable=True (verified via a live run's saved
        env.yaml), so self._soft_view exists at _get_rewards time -- no cfg change needed there.
      - The v2 reference file's feat is (N,46), matching AMP_N_MOTION=6 (2*20+6=46) exactly.

    obs_midline stays OFF (unlike some historical AMP tasks) so the ONLY difference from the
    plain "Joint-space RL" baseline is the added reward term, not a wider observation -- the
    fair, single-variable comparison the paper's Table 1/2 rows are meant to isolate.
    """

    body_forward_sign = 1.0            # Misty: head at +X (see class docstring)
    amp_reference_path = str(
        Path(__file__).resolve().parents[6] / "outputs" / "zef05_amp_ref" / "amp_reference_3d_v2.npz")
    profile_len = 20
    w_amp = 0.2                        # calibrated value from SalmonSwimAMPCfg's own history
                                        # (0.5 swamped the task reward; 0.2 makes AMP co-equal)
    disc_hidden = (128, 128)
    disc_lr = 1.5e-5
    disc_update_every = 128
    disc_batch = 512
    disc_r1 = 5.0
    disc_noise_std = 0.3
    disc_buffer_size = 40000
    disc_load_on_init = False
    disc_ckpt_path = str(Path(__file__).resolve().parents[6] / "outputs" / "zef05_amp_ref" / "disc_state_v2.pth")
    obs_midline = False                 # see class docstring: keep obs identical to baseline #1
