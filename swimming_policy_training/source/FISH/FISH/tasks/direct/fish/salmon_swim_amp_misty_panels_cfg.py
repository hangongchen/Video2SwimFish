# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Training 2: swim-to-target + AMP on the Misty fish, but the water force comes DIRECTLY from the
.obj surface geometry (strip-theory panels) instead of per-bone ellipsoids.

Identical to SalmonSwimAMPMistyCfg (same fish, upright spawn, head/tail fix, FEM AMP feature) except
`hydro_mode="panels"`: the env loads `panel_hydro.npz` (the Meshy .obj decimated to ~1454 flat panels
pinned to bones) and each step computes flat-plate drag per panel, summed per bone. The fins/taper of
the .obj literally shape the thrust. Same per-bone force/torque clips -> same stability envelope.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

from .salmon_swim_amp_misty_cfg import SalmonSwimAMPMistyCfg

_GEN = Path(__file__).resolve().parents[6] / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"


import copy as _copy
import math as _math

# NOTE: import the MODULE-LEVEL articulation object, not SalmonSwimAMPMistyCfg.robot_cfg --
# @configclass turns annotated fields into dataclass fields, so they are not class attributes.
from .salmon_swim_amp_misty_cfg import _ART as _MISTY_ART

# Actuator + joint-range settings measured on this fish with an open-loop scripted gait
# (scripts/scripted_swim_test.py, isolated 40 s runs, 0 blow-ups at every level):
#
#   joint limit  +/-15 -> +/-25 deg : 0.136 -> 0.236 BL/s. +/-35 and +/-50 are SLOWER (the body
#                                     over-curls into a C instead of running a travelling wave).
#   stiffness    60 -> 240          : the drive is a first-order lag with corner = stiffness/damping,
#                                     so at a 3 Hz beat the joints reached only 47% of the commanded
#                                     angle. 240 moves the corner 1.6 -> 6.4 Hz and tracking to 91%.
#                                     Measured tracking matched the lag prediction at EVERY frequency.
#   -> best measured: +/-25 deg, k=240, ~4 Hz beat = 0.691 BL/s (5.6x the original 0.124).
#
# Pushing harder makes it WORSE, and not because of instability (0 blow-ups at k=480 and +/-40 deg,
# jvel 8.1 vs the 200 guard): +/-40 deg gives 5x the tail-tip speed but LESS forward speed, because
# the limit is now conversion efficiency (body recoil), not actuator authority.
# STIFFNESS 240 -> 120 (2026-08-04, after a live training run). The scripted sweep saw 0 blow-ups at
# 240 and even 480 -- but it drove a SMOOTH sine wave. RL exploration issues sharp, discontinuous
# commands and a stiffer drive amplifies exactly those: at 240 the run hit 2 blow-ups in 28 resets
# (~1 in 14) by epoch 4, against a historical ~1 in 1811. Typical jvel also rose from 4 (k=60) to 15
# with p95 25, so normal operation sat much closer to the 200 guard.
# 120 is nearly free: tracking 73% vs 91%, scripted speed 0.461 vs 0.483 BL/s -- only 5% slower,
# because the speed gain had already saturated between 120 and 240. Nearly all of it came from 60->120.
# LESSON: a scripted-gait stability sweep CANNOT certify a setting for RL -- the stimulus is too smooth.
_MISTY_STIFFNESS = 120.0
_MISTY_JOINT_LIMIT_DEG = 25.0

_ART_PANELS = _copy.deepcopy(_MISTY_ART)
_ART_PANELS.actuators["all_joints"].stiffness = _MISTY_STIFFNESS


@configclass
class SalmonSwimAMPMistyPanelsCfg(SalmonSwimAMPMistyCfg):
    robot_cfg = _ART_PANELS
    joint_limit_deg = _MISTY_JOINT_LIMIT_DEG
    # must match the joint limit in radians, or the action scale caps the command below the limit
    pos_action_scale = _math.radians(_MISTY_JOINT_LIMIT_DEG)
    # FREE camera (origin_type="world"): eye/lookat are absolute world coords and the viewport is
    # yours to drag -- nothing re-aims it each step. The fish spawns near (0,0,1).
    # To make the camera CHASE the fish instead (it cannot then swim out of frame), swap to:
    #   origin_type="asset_root", asset_name="robot", eye=(1.2, 1.2, 0.6)  <- eye becomes the zoom
    # NOTE hydra rejects env.viewer.* overrides on the command line, so this has to be edited here.
    viewer: ViewerCfg = ViewerCfg(eye=(2.5, 2.5, 2.0), lookat=(0.0, 0.0, 1.0),
                                  origin_type="world")
    hydro_mode = "panels"
    panel_hydro_path = str(_GEN / "misty_minnow_fixed" / "panel_hydro.npz")
    panel_cd_normal = 1.0            # normal form drag (flat plate) -- also the TAIL's thrust source
    # Tangential SKIN drag over the whole wetted area (0.116 m^2, which matches a smooth 0.5 m fish
    # -- it is NOT inflated). Was 0.2, copied from the ellipsoid model's `cd_slender`, where that
    # number multiplies (Amax - Aproj) as a crossflow correction, NOT a wetted-area friction
    # coefficient. At 0.2 it produced 90% of all forward drag -- Cd on frontal area 2.55, worse than
    # a flat board held broadside -- and the fish stopped dead in 0.25 body lengths, unable to coast.
    # Physical value: laminar Cf = 1.328/sqrt(Re) = 0.0027 at Re 2.5e5 (0.496 m fish at 1 BL/s),
    # x1.19 form factor, x2-4 swimming augmentation -> 0.006-0.013.
    # Measured glide (coast test, zero joint drive), body length 0.496 m: 0.2 -> 0.25 BL,
    # 0.02 -> 1.29 BL, 0.01 -> 1.68 BL. Real fish is 2.8-4.7 BL, and even cd_t=0 only reaches
    # 2.41 BL -- the rest of the gap is the FORM term (0.0025 m^2, Cd_form 0.27 vs a real 0.01-0.05),
    # which is structural: the panel model has no pressure recovery, so leeward panels add drag
    # instead of returning it. Do not tune cd_tangent to hide that; keep it physical.
    panel_cd_tangent = 0.01
