# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the square-tank swim task (step 1 of the AMP reactive-swimming pipeline).

Reuses the analytic-water + cooked-FEM zebrafish physics and the swim task's target reward, but adds
a square tank of four COLLIDABLE walls and re-enables the bone colliders so the fish physically
collides with them. Small env count -- this is a stability probe (FEM + wall contact), not a full run.
"""

from __future__ import annotations

import copy
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .salmon_swim_cfg import SalmonSwimEnvCfg, SALMON_SWIM_ARTICULATION_CFG

_ZEF_DIR = (
    Path(__file__).resolve().parents[6]
    / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"
    / "vlm_rigged_single_fish" / "zef_f0001"
)
_ZEF_USD = _ZEF_DIR / "fish_articulated_cooked.usd"
if not _ZEF_USD.exists():
    _ZEF_USD = _ZEF_DIR / "fish_articulated.usd"

_TANK_ART: ArticulationCfg = copy.deepcopy(SALMON_SWIM_ARTICULATION_CFG)
_TANK_ART.spawn.usd_path = str(_ZEF_USD)
_TANK_ART.init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0))


@configclass
class SalmonTankSwimEnvCfg(SalmonSwimEnvCfg):
    robot_cfg: ArticulationCfg = _TANK_ART

    # cooked zebrafish asset fixups (see salmon_IL_water_zef_cfg)
    deformable_prim_token = "final_mesh"
    fix_multi_articulation_root = True
    fix_reflected_deformable = True          # cooked asset is already right-handed -> no-op, safe
    per_bone_hydro_path = ""

    # FEM active (we are stress-testing FEM + wall contact) + re-enable bone colliders for the walls
    with_deformable = True
    reset_deformable = True
    keep_bone_colliders = True

    # ---- square tank of four collidable walls (env-local, centred on the env origin) ----
    tank_size = 2.0          # inner side length (m); fish ~0.5 m -> ~4 body lengths across
    wall_thickness = 0.1
    wall_height = 2.0
    wall_center_z = 1.0      # fish swims at z=1.0

    # keep the swim target so the untrained fish drives toward a wall (step-1 contact probe);
    # place it near the +x wall (wall at +1.0) so contact actually happens.
    draw_target_marker = True
    target_offset_xy = (0.8, 0.0)
    success_radius = 0.2
    # keep the fish inside the tank; the blow-up guard stays well outside the 1 m walls
    root_position_limit = 6.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=8.0, replicate_physics=False,
    )

    record_video = False
    record_trajectory = False
    debug_print = True
