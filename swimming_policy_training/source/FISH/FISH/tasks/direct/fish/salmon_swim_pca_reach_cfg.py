# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the PCA-manifold RANDOM-TARGET REACHING task (SalmonSwimPCAReachEnv).

Extends the 2-PC straight-swim experiment (SalmonSwimPCACfg): same PCA action space
(bounded-rate a1/a2 -> curvature -> calibrated joint map -> PD/FEM), but the task becomes
"repeatedly reach randomly generated targets":

- TARGETS (all machinery REUSED from SalmonSwimEnv._sample_targets -- not reimplemented):
  bearing uniform in +/-15 deg about the fish's CURRENT heading (target_front_cone_deg=30),
  distance ~ Normal(mean=1.0 m, std=0.25 m) -- std chosen so ~95% of targets fall in
  [0.5, 1.5] m: far enough to need sustained swimming, near enough for several reaches per
  episode -- hard-clamped to [success_radius+0.2, 1.8] m so a target can never spawn
  behind/inside the fish (the bearing cone handles "in front", the min clamp "not inside").
- REACH (multi_target=True, the base env's existing no-reset path): entering the 0.2 m
  success radius counts one reach, logs it, and IMMEDIATELY resamples a new target from the
  fish's current pose. The episode does NOT reset; it ends only at the 40 s horizon (1200
  control steps -> several reaches per episode) or on a physics blow-up.
- REWARD: progress (previous_distance - current_distance) + success bonus + tiny energy
  term. No world-direction prior, no yaw/lateral penalties (turning is the task now).
- The base env's translucent success-sphere marker (marker_success_sphere) visualizes the
  live target at radius=success_radius in the GUI; it follows resampled targets automatically.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .salmon_swim_pca_cfg import SalmonSwimPCACfg


@configclass
class SalmonSwimPCAReachCfg(SalmonSwimPCACfg):
    # ---- target generation (consumed by the EXISTING SalmonSwimEnv._sample_targets) ----
    random_target = True              # sample a fresh target at every episode reset
    multi_target = True               # reach -> count + resample, NO env reset
    target_front_cone_deg = 15.0      # HALF-angle: bearing ~ uniform(-15, +15) deg about heading
    target_dist_mean = 1.0            # m
    target_dist_std = 0.25            # m (documented choice: 95% of samples in [0.5, 1.5] m)
    target_dist_clamp_max = 1.8       # m hard ceiling; min clamp = success_radius+0.2 (base env)
    success_radius = 0.2
    curriculum = False                # radius stays 0.2 (no shrink curriculum)
    dist_curriculum = False           # distance stays Normal(1.0, 0.25)

    # ---- target marker: reuse the existing dynamic success-sphere implementation ----
    draw_target_marker = True
    marker_success_sphere = True
    marker_success_opacity = 0.3

    # ---- episode: long horizon so one episode holds several consecutive reaches ----
    episode_length_s = 40.0           # 1200 control steps at 30 Hz; ~1 m per reach -> ~4-6 reaches

    # ---- reward ----
    # progress: w * (prev_dist - cur_dist)/BL per step. Closing at 0.25 m/s earns ~0.5/step;
    # one full 1.0 m reach integrates to ~60. Success bonus 25 per reach = a meaningful kicker
    # (~40% of a reach's progress integral) without letting bonus-farming dominate shaping.
    w_progress = 30.0
    w_success = 25.0
    w_energy = 0.001                  # kept from the swim task (mean joint_vel^2)
    # HEAD-FIRST variant knobs (defaults = OFF, preserving this cfg's original reward exactly;
    # SalmonSwimPCAReachHeadCfg turns them on -- see that class for the loophole rationale):
    heading_gate = False              # gate POSITIVE progress by clamp(heading_cos, 0, 1)
    w_align = 0.0                     # + w_align * max(heading_cos,0) * clamp(v_target_bl, 0, 0.5)
    # straight-swim shaping terms are DISABLED for reaching (turning is now required):
    w_forward = 0.0
    w_lateral = 0.0
    w_yawrate = 0.0

    # ---- env-1 visualization recording (point cloud + target + reward overlay) ----
    reach_frame_env = 1               # the single env recorded (of 256)
    reach_frame_every_n_episodes = 5  # record during every 5th episode of that env
    reach_frame_every_steps = 75      # one frame each 2.5 s inside a recorded episode
    reach_frame_dir = ""              # "" -> outputs/pca_reach_frames/run_<timestamp>/


@configclass
class SalmonSwimPCAReachHeadCfg(SalmonSwimPCAReachCfg):
    """HEAD-FIRST reaching: closes the broadside-gliding loophole of the plain reach reward.

    LOOPHOLE (measured on the v1 policy): progress = prev_dist - cur_dist is INVARIANT to body
    orientation, and the PCA gait produces large lateral forces almost for free, so sliding in
    sideways pays exactly like swimming in nose-first -- and needs no turning maneuver. Fix
    (smallest principled change, no new penalties that could freeze the fish):
      - POSITIVE progress is multiplied by clamp(heading_cos, 0, 1): broadside closing earns ~0,
        head-first closing earns full. NEGATIVE progress stays ungated (drifting away tail-first
        still costs full, so the gate cannot be exploited to dodge losses).
      - + w_align * max(heading_cos, 0) * clamp(v_target_bl, 0, 0.5): pointing at the target pays
        ONLY while actually closing on it (aligned-but-frozen earns ~0; broadside-but-closing
        earns ~0). At heading_cos=1 and 0.3 BL/s closing this is ~0.3/step, the same order as the
        gated progress term -- a bootstrap signal for learning the turn, not a dominant term.
    Success bonus, energy term, targets, multi-reach logic: unchanged from the v1 task.
    """

    heading_gate = True
    w_align = 1.0
    # success bonus scaled by clamp(heading_cos,0,1) at the reach moment (v2.1 escalation: the
    # flat bonus alone financed broadside bump-reaches; see _get_rewards comment)
    success_align_gate = True


@configclass
class SalmonSwimCPGReachCfg(SalmonSwimPCAReachHeadCfg):
    """CPG+RL BASELINE cfg: identical task/reward/sim to the head-first PCA reach experiment;
    only the CONTROLLER differs (see salmon_swim_cpg_reach_env.py). All fields below are the
    CPG's own parameter box -- everything else is inherited unchanged.

    Ranges mirror the validated scripted swimmer: amplitude up to the 45-deg joint limit,
    frequency up to 3.5 Hz (PD corner 3.2 Hz -- higher is unreachable anyway), turn bias as a
    uniform curvature offset up to 25 deg. Rate limits allow a full-range sweep in ~0.5-1 s,
    comparable smoothness to the PCA task's data-driven da bounds (which allow a full-amplitude
    3 Hz oscillation); the CPG carries the oscillation internally, so its params only need to
    move at MANEUVER timescales, not gait timescales."""

    cpg_amp_max = 0.785398        # 45 deg (the widened joint limit)
    cpg_freq_range = (0.5, 3.5)   # Hz
    cpg_freq_init = 2.0           # Hz at reset (A starts at 0 -> no motion until commanded)
    cpg_bias_max = 0.436332       # 25 deg uniform offset = constant-radius turn
    cpg_amp_rate = 0.0523599      # 3 deg / control step
    cpg_freq_rate = 0.1           # Hz / control step
    cpg_bias_rate = 0.0290888     # 1.667 deg / control step
    cpg_wavelengths = 1.0         # full waves along the body (validated scripted value)
