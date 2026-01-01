# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tank-swim env: the fish swims inside a square water tank with four COLLIDABLE walls.

Step 1 of the reactive natural-swimming (AMP) pipeline (see AMP_TANK_SWIM_PIPELINE.md): this only
adds the tank + wall collision on top of the proven analytic-water + cooked-FEM physics, so we can
verify wall contact works AND the fragile FEM body stays stable under impulsive wall contact (bone
colliders are re-enabled here; they are disabled in the swim/IL env). The AMP reward / perception /
reference data are layered on later -- for now it reuses the swim task's target reward so the
(untrained) fish actually moves toward a wall and we can watch the contact.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from .salmon_swim_env import SalmonSwimEnv


class SalmonTankSwimEnv(SalmonSwimEnv):

    def _spawn_env0_extras(self, env0_ns: str):
        """Spawn the four collidable tank walls under env0 (before clone -> replicated per env)."""
        if not getattr(self.cfg, "spawn_tank_walls", True):
            print(f"[SalmonTank] OPEN-WATER mode: no tank walls spawned under {env0_ns}", flush=True)
            return
        S = float(self.cfg.tank_size)
        t = float(self.cfg.wall_thickness)
        h = float(self.cfg.wall_height)
        z = float(self.cfg.wall_center_z)
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.9), opacity=0.12)
        coll = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        # static colliders (no rigid body) at the four sides of a square centred on the env origin
        walls = [
            ("wall_px", (t, S + t, h), (S / 2.0, 0.0, z)),
            ("wall_nx", (t, S + t, h), (-S / 2.0, 0.0, z)),
            ("wall_py", (S + t, t, h), (0.0, S / 2.0, z)),
            ("wall_ny", (S + t, t, h), (0.0, -S / 2.0, z)),
        ]
        for name, size, pos in walls:
            wcfg = sim_utils.CuboidCfg(size=size, collision_props=coll, visual_material=mat)
            wcfg.func(f"{env0_ns}/{name}", wcfg, translation=pos)
        print(f"[SalmonTank] spawned 4 collidable walls (tank {S} m, wall h {h} m) under {env0_ns}",
              flush=True)
