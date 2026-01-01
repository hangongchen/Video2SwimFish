# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP reactive-swimming on the VLM/auto-skeleton-generated fish at Isaac's DEFAULT dt (1/60).

Same env-owned discriminator / spine-bend motion imitation / reactive obs / energy penalty as the
AMP tank task, but the AGENT is the auto_skeleton_fish USD (not the ZeF asset) and the physics runs
at dt=1/60 (decimation 2 -> 30 Hz control) instead of the FEM-stable 1/960. The stress test showed
this asset stays stable at 1/60, so this trains the natural-swimming policy at the faster dt.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .salmon_amp_tank_cfg import SalmonAMPTankEnvCfg
from .salmon_swim_cfg import SALMON_SWIM_ARTICULATION_CFG

_NEW_USD = (
    Path(__file__).resolve().parents[6]
    / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"
    / "auto_skeleton_fish" / "fish_articulated.usd"
)

_ART: ArticulationCfg = copy.deepcopy(SALMON_SWIM_ARTICULATION_CFG)
_ART.spawn.usd_path = str(_NEW_USD)
# ORIENTATION FIX: SimFishLib exports the fish as X=length, Y=dorsoventral(tall), Z=lateral(thin),
# so with the identity quat in a Z-up world the fish spawns LYING ON ITS SIDE (tall axis horizontal).
# Rotate +90 deg about the length (X) axis -> dorsoventral Y maps to world Z (vertical) = upright fish.
# (Verified offline: post-rot vertical extent = 0.179 tall, lateral = 0.079.) See [[fish-spawn-vertical]].
_ART.init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(0.70710678, 0.70710678, 0.0, 0.0))

_HZ = float(os.environ.get("FISH_TEST_DT_HZ", "60"))      # default 60 Hz = Isaac default dt
_DT = 1.0 / _HZ
_DECIM = max(1, round(_HZ / 30.0))                        # ~30 Hz control


@configclass
class SalmonAMPAutoskelCfg(SalmonAMPTankEnvCfg):
    # AGENT = the auto-skeleton generated fish (fixups + FEM + reward inherited from the AMP tank cfg)
    robot_cfg: ArticulationCfg = _ART

    # DEFAULT dt (1/60), not the FEM-stable 1/960. episode_length_s stays 100 s (inherited) ->
    # 100 / (decimation * dt) = 3000 control steps at 30 Hz.
    decimation = _DECIM
    sim: SimulationCfg = SimulationCfg(
        dt=_DT, render_interval=_DECIM, gravity=(0.0, 0.0, 0.0),
        physx=PhysxCfg(
            solver_type=1, enable_external_forces_every_iteration=True,
            gpu_collision_stack_size=2 ** 30, gpu_heap_capacity=2 ** 30,
            gpu_temp_buffer_capacity=2 ** 28, gpu_max_soft_body_contacts=2 ** 22,
            gpu_max_particle_contacts=2 ** 28,
        ),
    )

    # keep the FEM soft-body blow-up guard on (m/s)
    fem_velocity_limit = 50.0

    # PER-SLICE hydro for THIS fish (10 bones, fitted from its own FEM body by
    # scripts/precompute_autoskel_hydro.py): real body-girth drag surfaces (~9x4x1 cm half-extents)
    # instead of the thin-bone inertia fallback ("weak thrust") -> the undulation produces real thrust.
    per_bone_hydro_path = str(
        Path(__file__).resolve().parents[6]
        / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"
        / "auto_skeleton_fish" / "per_bone_hydro.npz")
    # Hydro stability knobs at coarse dt (env-var tunable for probing). The TORQUE clip is the
    # critical one: at dt=1/120 a torque at the old 20 N.m clip imparts thousands of rad/s on a
    # small bone's inertia in one substep -> instant explosion (jvel 31k, all envs, step 1).
    hydro_torque_clip = float(os.environ.get("HYDRO_TORQUE_CLIP", "0.05"))
    hydro_force_clip = float(os.environ.get("HYDRO_FORCE_CLIP", "5.0"))   # stability-certified default
    hydro_semi_scale = float(os.environ.get("HYDRO_SEMI_SCALE", "1.0"))
    # KUTTA LIFT ck = default 1.0 (the known-good hydro; a successful swimmer exists in this env). The
    # earlier ck=15 override was added under the mistaken conclusion that the hydro could not propel;
    # reverted per the confirmation that the physics is known-good and the failure was training-side.
    ck = float(os.environ.get("HYDRO_CK", "1.0"))
    # FISH_OPEN_WATER=1 -> remove the tank (no walls spawned, rays read max-distance, wall penalty
    # auto-zeroes) and drop the now-pointless bone colliders. Same task id; retrains from scratch.
    spawn_tank_walls = os.environ.get("FISH_OPEN_WATER", "0") != "1"
    keep_bone_colliders = os.environ.get("FISH_OPEN_WATER", "0") != "1"

    # STATIC world camera (follow OFF): origin_type="env" fixes the camera in the env frame -- it does
    # NOT track the fish, so the fish's travel across the scene is visible (it may swim out of frame in
    # open water; zoom out in the GUI if needed). 3/4 elevated view of the swim region around env0.
    # Harmless for headless training (which ignores the viewer). Hydra rejects env.viewer.* overrides.
    viewer: ViewerCfg = ViewerCfg(
        origin_type="env", env_index=0,
        eye=(1.6, 1.6, 2.3), lookat=(-0.1, -0.1, 1.0), resolution=(1280, 720),
    )
