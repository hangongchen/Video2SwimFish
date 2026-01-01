# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the swim-to-target + AMP task (SalmonSwimAMPEnv).

Inherits SalmonAMPAutoskelCfg, which brings:
  - the auto-skeleton fish + dt=1/120 (decimation 4 -> 30 Hz control) + FEM buffer sizing,
  - the per-slice hydro (autoskel per_bone_hydro.npz) + coarse-dt hydro clips,
  - the AMP discriminator params (amp_reference_path, profile_len=20, disc_hidden/lr/batch/...),
  - AND -- via its SalmonSwimEnvCfg base -- the OLD swim reward scales (distance=100, success=200,
    heading=0, time=1e-3, effort=0) and the target machinery.

This cfg ADDS ONLY `w_amp` (weight of the added AMP style term) and restores the OLD swim-task
target (the tank cfg base had narrowed it to (0.8,0)/0.2). The OLD reward + observation are reused
UNCHANGED; SalmonSwimAMPEnv adds `+ w_amp * r_style`.
"""

from __future__ import annotations

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

from .salmon_amp_autoskel_cfg import SalmonAMPAutoskelCfg


@configclass
class SalmonSwimAMPCfg(SalmonAMPAutoskelCfg):
    # reward = old_swim_reward + w_amp * r_style.  r_style is ~0.6/step and accrues EVERY step, so over
    # a ~3000-step episode it totals ~1800*w_amp -- at w_amp=0.5 that (~900) SWAMPED the bounded task
    # rewards (progress ~140 + success 200 = ~340) and the policy just undulated in place for AMP without
    # reaching the target (run 1: success ~2%, distance flat at 1.36 m). Lowered to 0.2 so AMP episodic
    # (~360) ~ task (~340): the target is the PRIMARY driver, AMP a co-equal style shaper. Main knob.
    w_amp = 0.2

    # the AMP bend feature reads the FEM soft body -> the deformable MUST be active (base default False).
    with_deformable = True

    # append the 2K-dim AMP bend profile (the FEM midline the discriminator judges) to the POLICY obs.
    # The joints already determine the midline up to FEM lag/wobble, so this is mostly-redundant
    # information -- it exists to TEST whether seeing the exact discriminator input helps the policy
    # earn the style reward. Widens obs by 2*profile_len (40 at K=20).
    obs_midline = False

    # restore the OLD swim-task target (SalmonTankSwimEnvCfg base had narrowed it to (0.8,0)/0.2). The
    # auto-skeleton fish swims weakly, so keep the forgiving 0.6 m success radius from the old task.
    target_offset_xy = (1.0, 1.0)
    target_height = 1.0
    target_root_height = 1.0
    success_radius = 0.6

    # STATIC WORLD camera -- attached to NOTHING (follow OFF, not locked to the fish, not env-relative).
    # origin_type="world" fixes eye/lookat in absolute world coordinates, so the fish is NOT re-centred:
    # you see BOTH its pose change AND its TRANSLATION across the scene (it swims from ~origin toward the
    # target at (1,1,1)). 3/4 elevated view framing the origin->target swim region; orbit/zoom freely in
    # the GUI (the camera stays put, so panning still shows the fish moving through the world).
    viewer: ViewerCfg = ViewerCfg(
        origin_type="world",
        eye=(3.0, -1.5, 2.5), lookat=(0.5, 0.5, 1.0), resolution=(1280, 720),
    )
