# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Salmon "swim to a target in water" DirectRLEnv.

Mirrors the Template-Fish-Direct-v0 target-reaching scaffold, adds an analytic
MuJoCo ellipsoid hydrodynamic wrench (the "water"), and drives the D6 joints by PD
position targets (authoring the missing angular DriveAPI so they actuate).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, sample_uniform

from .salmon_swim_cfg import SalmonSwimEnvCfg


# ============================================================================
# Analytic MuJoCo ellipsoid fluid model (ported from scripts/demo_hydro_swim.py).
# ============================================================================
def extract_ellipsoids(robot, device):
    """Per-bone equivalent-ellipsoid semi-axes + principal-frame rotation, derived
    from each link's inertia tensor + mass (asset-agnostic). Returns semi (B,3),
    R_pl (B,3,3) principal->link, mass (B,). Bones are identical across envs, so this
    is computed once from env 0."""
    mass = robot.data.default_mass[0].to(device).float()
    inertia = robot.data.default_inertia[0].to(device).float().reshape(-1, 3, 3)
    inertia = 0.5 * (inertia + inertia.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(inertia)
    Ix, Iy, Iz = evals[:, 0], evals[:, 1], evals[:, 2]
    coef = 5.0 / (2.0 * mass.clamp_min(1e-9))
    rx2 = (coef * (Iy + Iz - Ix)).clamp_min(1e-10)
    ry2 = (coef * (Ix + Iz - Iy)).clamp_min(1e-10)
    rz2 = (coef * (Ix + Iy - Iz)).clamp_min(1e-10)
    semi = torch.sqrt(torch.stack([rx2, ry2, rz2], dim=-1))
    return semi, evecs, mass


def mujoco_fluid_wrench(V_w, W_w, Q_wl, semi, R_pl, *, rho, visc,
                        cd_blunt, cd_slender, cd_angular, ck, cm):
    """Faithful MuJoCo ellipsoid fluid model, vectorized over the first dim (N = E*B).
    Velocity-only (no acceleration) -> stable as an explicit external wrench, and
    passive (drag/viscous dissipate, lift does no net work). Inputs world-frame; the
    wrench is computed in each body's principal frame and rotated back to world."""
    eps = 1e-9
    pi = math.pi
    v_l = quat_rotate_inverse(Q_wl, V_w)
    w_l = quat_rotate_inverse(Q_wl, W_w)
    v = torch.einsum("nji,nj->ni", R_pl, v_l)
    w = torch.einsum("nji,nj->ni", R_pl, w_l)

    rx, ry, rz = semi[:, 0], semi[:, 1], semi[:, 2]
    rmax = semi.max(-1).values
    rmin = semi.min(-1).values
    rmid = semi.sum(-1) - rmax - rmin
    Vol = (4.0 / 3.0) * pi * rx * ry * rz

    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    speed = v.norm(dim=-1)
    den = torch.sqrt((ry**2 * rz**2 * vx**2 + rz**2 * rx**2 * vy**2 + rx**2 * ry**2 * vz**2).clamp_min(0.0))
    num = pi * torch.sqrt((ry**4 * rz**4 * vx**2 + rz**4 * rx**4 * vy**2 + rx**4 * ry**4 * vz**2).clamp_min(0.0))
    Aproj = torch.where(den > eps, num / den.clamp_min(eps), torch.zeros_like(den))
    Amax = pi * rmax * rmid

    drag_coef = (cd_blunt * Aproj + cd_slender * (Amax - Aproj)).unsqueeze(-1)
    f_D = -rho * drag_coef * (speed.unsqueeze(-1) * v)

    rj4 = torch.stack([
        torch.maximum(ry, rz) ** 4,
        torch.maximum(rx, rz) ** 4,
        torch.maximum(rx, ry) ** 4], dim=-1)
    ID = (8.0 * pi / 15.0) * semi * rj4
    Imax = ID.max(-1, keepdim=True).values
    wspeed = w.norm(dim=-1, keepdim=True)
    g_D = -rho * wspeed * ((cd_angular * ID + cd_slender * (Imax - ID)) * w)

    f_V = -6.0 * pi * (semi.mean(-1) * visc).unsqueeze(-1) * v
    g_V = -8.0 * pi * (semi.mean(-1) ** 3 * visc).unsqueeze(-1) * w

    f_M = cm * rho * Vol.unsqueeze(-1) * torch.linalg.cross(w, v)

    ns = torch.stack([ry * rz * vx / rx, rz * rx * vy / ry, rx * ry * vz / rz], dim=-1)
    n_hat = ns / (ns.norm(dim=-1, keepdim=True) + eps)
    v_hat = v / (speed.unsqueeze(-1) + eps)
    dotp = (v_hat * n_hat).sum(-1, keepdim=True)
    f_K = ck * rho * Aproj.unsqueeze(-1) * dotp * torch.linalg.cross(torch.linalg.cross(n_hat, v), v)

    f_p = torch.nan_to_num(f_D + f_V + f_M + f_K)
    g_p = torch.nan_to_num(g_D + g_V)
    f_l = torch.einsum("nij,nj->ni", R_pl, f_p)
    g_l = torch.einsum("nij,nj->ni", R_pl, g_p)
    F_w = quat_rotate(Q_wl, f_l)
    T_w = quat_rotate(Q_wl, g_l)
    return F_w, T_w


def _prepare_env_assets(stage, env_root, control_mode, keep_deformable=False, joint_limit_deg=None,
                        fem_youngs_modulus=None, fem_elasticity_damping=None, fem_damping_scale=None,
                        deformable_token="deformable_salmon", fix_multi_articulation_root=False,
                        fix_reflected_deformable=False, keep_bone_colliders=False):
    """Per spawned env: handle the deformable subtree (deactivate for skeleton-only RL,
    or keep it active with fixed collision offsets if keep_deformable), disable the
    internal rigid-bone colliders, optionally widen the D6 rot limits (joint_limit_deg,
    in degrees -- the asset ships +/-15), and -- for position control -- author a force-mode
    angular DriveAPI on every D6 joint's rotX/Y/Z so set_joint_position_target actuates
    them (the USD ships only PhysicsLimitAPI). Run BEFORE sim.reset()/play."""
    from pxr import UsdPhysics

    n_soft = n_drive = n_coll = 0

    prefix = str(env_root)

    # ---- single-articulation-root fixup ----
    # Some fish USDs (the SimFishLib video->USD pipeline) stamp PhysicsArticulationRootAPI on EVERY
    # bone, so Isaac finds N roots under the spawn prim and refuses ("Failed to find a single
    # articulation"). Strip the API off every bone and re-apply it ONCE on the spawn root prim
    # (env_root), giving a single floating-base articulation. Salmon ships a correct single root, so
    # this is gated off by default.
    if fix_multi_articulation_root:
        n_removed = 0
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix):
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                n_removed += 1
        root_prim = stage.GetPrimAtPath(prefix)
        if root_prim and root_prim.IsValid():
            UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        print(f"[SalmonSwim] articulation-root fixup: removed {n_removed} bone roots, "
              f"applied 1 root on {prefix}", flush=True)

    # ---- FEM deformable-body MATERIAL (CREATE + BIND) ----
    # This asset ships the deformable mesh (PhysxDeformableBodyAPI) WITHOUT any deformable-body
    # material prim (the only Material prims are rigid: backboneMaterial/branchMaterial/default).
    # So PhysX silently used its DEFAULT deformable material (youngsModulus 5e7, stiff) -- which is
    # why every prior youngsModulus override was a no-op and the FEM stayed in the CFL-unstable
    # regime. We CREATE a PhysxDeformableBodyMaterialAPI material and BIND it to the deformable mesh,
    # under env_0/skeleton BEFORE clone (the cloner copies the subtree + remaps the binding per env).
    # youngsModulus softens (headroom); elasticityDamping is the CURE (stiffness-proportional damping
    # whose modal ratio grows with frequency -> flips the unstable high-omega mode's growth-rate
    # sign); dampingScale (in [0,1]) gives that term full authority.
    if keep_deformable and fem_youngs_modulus is not None:
        try:
            from omni.physx.scripts import deformableUtils, physicsUtils
        except Exception as e:  # pragma: no cover
            deformableUtils = physicsUtils = None
            print(f"[SalmonSwim] WARNING: deformable material utils import failed: {e}", flush=True)
        _n_mat = 0
        if deformableUtils is not None:
            for prim in stage.Traverse():
                pp = str(prim.GetPath())
                if not pp.startswith(prefix) or deformable_token not in pp:
                    continue
                if "PhysxDeformableBodyAPI" not in prim.GetAppliedSchemas():
                    continue
                mat_path = pp + "/deformableBodyMaterial"
                ok = deformableUtils.add_deformable_body_material(
                    stage, mat_path,
                    youngs_modulus=float(fem_youngs_modulus),
                    poissons_ratio=0.45,
                    elasticity_damping=(None if fem_elasticity_damping is None
                                        else float(fem_elasticity_damping)),
                    damping_scale=(None if fem_damping_scale is None
                                   else float(fem_damping_scale)),
                )
                physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
                _n_mat += 1
                print(f"[SalmonSwim] FEM material CREATED+BOUND at {mat_path} | "
                      f"youngs={fem_youngs_modulus} elasticityDamping={fem_elasticity_damping} "
                      f"dampingScale={fem_damping_scale} (add_material ok={ok})", flush=True)
        if _n_mat == 0:
            print("[SalmonSwim] WARNING: no deformable body prim (PhysxDeformableBodyAPI) found "
                  "to bind a FEM material to -- youngsModulus/elasticityDamping NOT applied!", flush=True)

    # ---- reflected-deformable-xform fixup ----
    # The SimFishLib Blender->USD export gives the deformable mesh a NEGATIVE uniform scale
    # (det < 0, a reflection). Isaac's DeformablePrim.initialize rejects a left-handed frame
    # ("Non-positive determinant ... in rotation matrix"), so the FEM `_soft_view` cannot be
    # built and the IL reward has no sim cloud. Neutralise the reflection WITHOUT moving the fish:
    # flip the scale sign to positive AND negate the mesh points (same world geometry, det > 0),
    # and reverse face winding so surface normals stay outward. Salmon ships a proper frame -> off.
    if keep_deformable and fix_reflected_deformable:
        from pxr import UsdGeom, Vt, Gf
        _n_fix = 0
        for prim in stage.Traverse():
            pp = str(prim.GetPath())
            if not pp.startswith(prefix) or deformable_token not in pp:
                continue
            if "PhysxDeformableBodyAPI" not in prim.GetAppliedSchemas():
                continue
            # find the ancestor (or self) carrying a negative-determinant scale
            node = prim
            fixed_scale = False
            while node and str(node.GetPath()).startswith(prefix):
                sa = node.GetAttribute("xformOp:scale")
                if sa and sa.IsValid() and sa.Get() is not None:
                    s = sa.Get()
                    if s[0] * s[1] * s[2] < 0:
                        sa.Set(Gf.Vec3f(abs(s[0]), abs(s[1]), abs(s[2])))
                        fixed_scale = True
                        break
                node = node.GetParent()
            if not fixed_scale:
                continue
            # negate points to keep world geometry identical under the now-positive scale
            pattr = prim.GetAttribute("points")
            if pattr and pattr.IsValid() and pattr.Get() is not None:
                pts = pattr.Get()
                pattr.Set(Vt.Vec3fArray([Gf.Vec3f(-q[0], -q[1], -q[2]) for q in pts]))
            # reverse winding per face so normals point outward again
            fvi_a = prim.GetAttribute("faceVertexIndices")
            fvc_a = prim.GetAttribute("faceVertexCounts")
            if all(a and a.IsValid() and a.Get() is not None for a in (fvi_a, fvc_a)):
                fvi = list(fvi_a.Get()); fvc = list(fvc_a.Get()); out = []; k = 0
                for c in fvc:
                    out.extend(fvi[k:k + c][::-1]); k += c
                fvi_a.Set(Vt.IntArray(out))
            _n_fix += 1
            print(f"[SalmonSwim] reflected-deformable fixup: flipped scale +, negated points, "
                  f"reversed winding on {pp}", flush=True)
        if _n_fix == 0:
            print("[SalmonSwim] reflected-deformable fixup: no negative-scale deformable found "
                  "(already right-handed?)", flush=True)

    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(prefix):
            continue
        if deformable_token in p:
            if keep_deformable:
                # keep ACTIVE but replace the degenerate -inf collision offsets so the GPU
                # contact pipeline doesn't choke on the permanent deformable-bone overlap
                if prim.GetAttribute("physxDeformable:simulationPoints").IsValid():
                    _dissip = 0
                    for attr, val in [("physxCollision:contactOffset", 0.01),
                                      ("physxCollision:restOffset", 0.0),
                                      ("physxDeformable:maxDepenetrationVelocity", 1.0),
                                      ("physxDeformable:selfCollisionFilterDistance", 0.01),
                                      ("physxDeformable:enableCCD", False),
                                      # OPTION 2: dissipate the energy the undulation pumps into the
                                      # FEM body (it was accumulating -> diverging ~step 1000).
                                      ("physxDeformable:vertexVelocityDamping", 4.0),   # (cure) broadband velocity sink (1.0->4.0)
                                      ("physxDeformable:settlingThreshold", 0.1),
                                      ("physxDeformable:sleepDamping", 10.0),           # (cure) schema default, quenches residual ring (1.0->10.0)
                                      ("physxDeformable:solverPositionIterationCount", 96)]:  # (cure) tighter implicit solve -> less residual energy (64->96)
                        a = prim.GetAttribute(attr)
                        if a and a.IsValid():
                            a.Set(val)
                            if any(k in attr for k in ("Damping", "IterationCount")):
                                _dissip += 1
                    n_soft += 1
                    print(f"[SalmonSwim] FEM dissipation: applied {_dissip} damping/iter attrs "
                          f"(vertexVelocityDamping etc.) on the deformable", flush=True)
            elif prim.IsActive():
                prim.SetActive(False)
                n_soft += 1
            continue
        # disable rigid-bone colliders (no ground contact needed in zero-g water). KEEP them when
        # keep_bone_colliders (e.g. the tank task, where the bones must collide with the walls).
        # BUG FIX 2026-09-22: on the Video2SwimFish assets this branch never fired ("disabled 0
        # colliders") and the 15-16 bone collision meshes stayed ENABLED inside the FEM skin. The
        # skin is auto-attached to those same bones, so attachment + contact fought each other and
        # injected momentum: a zero-action fish pitched 40-90 deg and drifted 1-2.4 BL in 3 s
        # (disabling the colliders at runtime removed it completely). Match on the attribute as
        # well as the API, and only within the skeleton subtree.
        _coll_attr = prim.GetAttribute("physics:collisionEnabled")
        if prim.HasAPI(UsdPhysics.CollisionAPI) or (_coll_attr and _coll_attr.IsValid()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(bool(keep_bone_colliders))
            else:
                _coll_attr.Set(bool(keep_bone_colliders))
            if not keep_bone_colliders:
                n_coll += 1
        is_d6 = "D6Joint" in prim.GetName() and str(prim.GetTypeName()) == "PhysicsJoint"
        # widen the rot limits (any control mode): the asset ships +/-15 deg; bigger range
        # -> bigger undulation amplitude -> more thrust. Applied to rotX/Y/Z (deg).
        if is_d6 and joint_limit_deg is not None:
            for axis in ("rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(prim, axis)
                lim.CreateLowAttr().Set(-float(joint_limit_deg))
                lim.CreateHighAttr().Set(float(joint_limit_deg))
        if is_d6 and control_mode == "position":
            for axis in ("rotX", "rotY", "rotZ"):
                d = UsdPhysics.DriveAPI.Apply(prim, axis)
                d.CreateTypeAttr().Set("force")
                d.CreateStiffnessAttr().Set(100.0)   # overwritten by the ImplicitActuator at init
                d.CreateDampingAttr().Set(10.0)
                d.CreateMaxForceAttr().Set(1.0e6)
                d.CreateTargetPositionAttr().Set(0.0)
            n_drive += 1
    return n_soft, n_drive, n_coll


class SalmonSwimEnv(DirectRLEnv):
    cfg: SalmonSwimEnvCfg

    def __init__(self, cfg: SalmonSwimEnvCfg, render_mode: str | None = None, **kwargs):
        # CHASE CAM -- must be patched BEFORE super().__init__, which is where DirectRLEnv reads
        # cfg.viewer and builds the viewport camera. Hydra refuses env.viewer.* overrides on the
        # CLI, so cfg.camera_follow is the only way to switch this from a launch command.
        if getattr(cfg, "camera_follow", False):
            _eye = tuple(float(v) for v in getattr(cfg, "camera_follow_eye", (1.2, 1.2, 0.6)))
            _track = bool(getattr(cfg, "camera_follow_track", True))
            cfg.viewer.asset_name = "robot"
            cfg.viewer.env_index = 0
            cfg.viewer.eye = _eye
            if _track:
                # glued to the fish -> great for the gait, but the fish never appears to move
                cfg.viewer.origin_type = "asset_root"
                cfg.viewer.lookat = (0.0, 0.0, 0.0)
            else:
                # pinned to env_0's origin -> stationary frame, the fish visibly swims across it.
                # lookat is aimed at the spawn height (~1 m above the env origin).
                cfg.viewer.origin_type = "env"
                cfg.viewer.lookat = (0.0, 0.0, 1.0)
            print(f"[SalmonSwim] camera {'FOLLOWS env_0 robot root' if _track else 'PINNED to env_0 origin'},"
                  f" eye offset {_eye}", flush=True)
        super().__init__(cfg, render_mode, **kwargs)
        # BUG FIX 2026-09-22: the pre-clone prep never reached the bone collision meshes of the
        # Video2SwimFish assets (referenced/instanced subtree -> stage.Traverse() skipped them; the
        # log always said "disabled 0 colliders"). Enabled bone colliders inside the auto-attached
        # FEM skin injected momentum: zero-action fish pitched 40-90 deg and drifted 1-2 BL in 3 s.
        # Disabling them here, on the built scene (instance proxies included), removes it entirely.
        if not getattr(self.cfg, "keep_bone_colliders", False):
            self._disable_bone_colliders_post_scene()

        # SUCCESS-REGION spheres (flag set in _prepare_env_assets when cfg.marker_success_sphere):
        # one translucent red sphere per env, re-aimed every control step in _get_observations.
        # Prototype radius 1.0 so the per-call `scales` IS the radius -> tracks the live curriculum.
        self._success_markers = None
        if getattr(self, "_want_success_markers", False):
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            _op = float(getattr(self.cfg, "marker_success_opacity", 0.3))
            # RTX reads UsdPreviewSurface `opacity` as CUTOUT opacity: without fractionalCutoutOpacity
            # the 0.3 is rounded to a binary keep/discard and the sphere renders SOLID, hiding the
            # fish the moment it swims inside the success region. These two settings make it a real
            # blend. (Harmless when the renderer isn't running, e.g. headless training.)
            try:
                import carb
                _s = carb.settings.get_settings()
                _s.set("/rtx/raytracing/fractionalCutoutOpacity", True)
                _s.set("/rtx/pathtracing/fractionalCutoutOpacity", True)
            except Exception as exc:  # noqa: BLE001
                print(f"[SalmonSwim] could not enable fractional cutout opacity: {exc}", flush=True)
            def _mk(mat):
                return VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/Visuals/success_region",
                    markers={"sphere": sim_utils.SphereCfg(radius=1.0, visual_material=mat)},
                ))

            _preview = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.05, 0.05), opacity=_op,
                # a little self-glow so the shell still reads as a boundary once it is
                # see-through -- a purely diffuse 30% sphere nearly vanishes on a dark bg.
                emissive_color=(0.25, 0.0, 0.0), roughness=0.6)
            if str(getattr(self.cfg, "marker_success_material", "glass")).lower() == "glass":
                try:
                    # ior 1.0 = no light bending, so the fish behind/inside the sphere stays
                    # undistorted; thin_walled treats it as a shell, not a solid ball of glass.
                    self._success_markers = _mk(sim_utils.GlassMdlCfg(
                        glass_color=(1.0, 0.12, 0.12), frosting_roughness=0.15,
                        thin_walled=True, glass_ior=1.0))
                except Exception as exc:  # noqa: BLE001
                    print(f"[SalmonSwim] glass marker material failed ({exc}); using preview surface",
                          flush=True)
                    self._success_markers = _mk(_preview)
            else:
                self._success_markers = _mk(_preview)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        # WIDEN the D6 rotation limits past what the asset ships, if asked. Measured on the Misty
        # fish: +/-15 -> +/-25 deg nearly doubles the scripted-gait speed (0.136 -> 0.236 BL/s), and
        # +/-35..50 gets SLOWER (the body over-curls into a C instead of a travelling wave).
        # NOTE cfg.pos_action_scale must be raised to match (limit in radians) -- BOTH gate the
        # command, so raising only one changes nothing.
        _jl = float(getattr(self.cfg, "joint_limit_deg", 0.0) or 0.0)
        if _jl > 0.0:
            _rad = math.radians(_jl)
            _lim = torch.zeros(self.num_envs, self.robot.num_joints, 2, device=self.device)
            _lim[..., 0], _lim[..., 1] = -_rad, _rad
            self.robot.write_joint_position_limit_to_sim(_lim)
            print(f"[SalmonSwim] joint limits widened to +/-{_jl:.1f} deg", flush=True)
        self._soft_joint_limits = self.robot.data.soft_joint_pos_limits
        print(f"[SalmonSwim] D6 pos-limit half-range ~{float(self._soft_joint_limits[..., 1].abs().max()) * 57.2958:.1f} deg"
              f" | pos_action_scale={self.cfg.pos_action_scale} rad"
              f" | stiffness={self.cfg.robot_cfg.actuators['all_joints'].stiffness}", flush=True)
        self._default_joint_pos = self.robot.data.default_joint_pos
        # sized by the number of CONTROLLED joints -- set_joint_*_target is called with
        # joint_ids=self._control_joint_ids, so these must match that width, not num_joints.
        self._torques = torch.zeros((self.num_envs, self._num_actions), device=self.device)
        self._pos_targets = torch.zeros((self.num_envs, self._num_actions), device=self.device)

        # target
        self._target_offset_xy = torch.tensor(self.cfg.target_offset_xy, dtype=torch.float32, device=self.device)
        self._target_height = torch.tensor(self.cfg.target_height, dtype=torch.float32, device=self.device)
        self.target_positions_w = self._compute_target_positions_world()

        # per-bone ellipsoid geometry for the hydro model (identical across envs).
        # PREFER the precomputed PER-SLICE ellipsoids (body girth around each bone -> real
        # thrust); fall back to the thin inertia-derived bone ellipsoids if absent.
        import os
        path = self.cfg.per_bone_hydro_path
        if path and os.path.exists(path):
            data = np.load(path)
            semi = torch.tensor(data["semi"], device=self.device, dtype=torch.float32)
            _sscale = float(getattr(self.cfg, "hydro_semi_scale", 1.0))
            semi = semi * _sscale
            R_pl = torch.tensor(data["R_pl"], device=self.device, dtype=torch.float32)
            _src = f"PER-SLICE ({os.path.basename(path)}, scale={_sscale})"
        else:
            semi, R_pl, _ = extract_ellipsoids(self.robot, self.device)
            _src = "thin-bone inertia (per-slice file MISSING -> weak thrust)"
        self._nb = self.robot.num_bodies
        # pre-expand to (E*B, ...) for the batched wrench
        self._semi = semi.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3).contiguous()
        self._R_pl = R_pl.unsqueeze(0).expand(self.num_envs, -1, -1, -1).reshape(-1, 3, 3).contiguous()
        print(f"[SalmonSwim] hydro={_src}; mean 2*semi={[round(float(2*x),3) for x in semi.mean(0)]} m; "
              f"control={self.cfg.control_mode}", flush=True)

        # STRIP-THEORY PANEL hydro (Training 2): the .obj surface, decimated to flat panels pinned to
        # bones, IS the force source. Loaded here; used by _panel_hydro_wrench when cfg.hydro_mode=="panels".
        self._hydro_mode = str(getattr(self.cfg, "hydro_mode", "ellipsoid"))
        if self._hydro_mode == "panels":
            ppath = getattr(self.cfg, "panel_hydro_path", None)
            assert ppath and os.path.exists(ppath), f"[SalmonSwim] hydro_mode=panels but panel file missing: {ppath}"
            pd = np.load(ppath)
            self._panel_bone = torch.tensor(pd["panel_bone"], device=self.device, dtype=torch.long)      # (Pn,)
            self._panel_r = torch.tensor(pd["panel_r_local"], device=self.device, dtype=torch.float32)   # (Pn,3)
            self._panel_n = torch.tensor(pd["panel_n_local"], device=self.device, dtype=torch.float32)   # (Pn,3)
            self._panel_a = torch.tensor(pd["panel_area"], device=self.device, dtype=torch.float32)      # (Pn,)
            self._panel_cd_n = float(getattr(self.cfg, "panel_cd_normal", self.cfg.cd_blunt))
            self._panel_cd_t = float(getattr(self.cfg, "panel_cd_tangent", self.cfg.cd_slender))
            print(f"[SalmonSwim] hydro=PANELS ({os.path.basename(ppath)}): {self._panel_bone.shape[0]} panels, "
                  f"area={float(self._panel_a.sum()):.4f} m^2, cd_n={self._panel_cd_n}, cd_t={self._panel_cd_t}",
                  flush=True)
            # Body length = extent of the ASSEMBLED panel cloud (panels placed on their rest-pose
            # bones). Do NOT use pd["vert_local"] -- in the misty npz that array is a stale,
            # length-squashed copy (0.228 m vs the real 0.496 m) that the hydro never reads, and
            # using it silently halves every BL/s number.
            _q = pd["bone_rest_quat"][pd["panel_bone"]].astype(np.float64)
            _w, _x, _y, _z = _q[:, 0], _q[:, 1], _q[:, 2], _q[:, 3]
            _R = np.stack([1-2*(_y*_y+_z*_z), 2*(_x*_y-_w*_z), 2*(_x*_z+_w*_y),
                           2*(_x*_y+_w*_z), 1-2*(_x*_x+_z*_z), 2*(_y*_z-_w*_x),
                           2*(_x*_z-_w*_y), 2*(_y*_z+_w*_x), 1-2*(_x*_x+_y*_y)], -1).reshape(-1, 3, 3)
            _pw = pd["bone_rest_pos"][pd["panel_bone"]] + np.einsum(
                "nij,nj->ni", _R, pd["panel_r_local"].astype(np.float64))
            self._body_length = float(np.ptp(_pw, axis=0).max())

        # BODY LENGTH (m) -- swim speed is logged in BODY LENGTHS/s, the unit fish biology uses
        # (real fish cruise 1-4 BL/s), so the number is comparable across assets and to real fish.
        # Measured from the asset mesh when available; cfg.body_length overrides.
        _bl_cfg = float(getattr(self.cfg, "body_length", 0.0) or 0.0)
        if _bl_cfg > 0.0:
            self._body_length = _bl_cfg
        elif not hasattr(self, "_body_length"):
            self._body_length = 0.26        # salmon asset default; set cfg.body_length for others
        print(f"[SalmonSwim] body_length={self._body_length:.4f} m -> speed logged in BL/s", flush=True)

        # episode stats
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._best_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)
        self._targets_reached = torch.zeros(self.num_envs, device=self.device)      # per-episode count (multi_target)
        self._reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # latched per episode: True once this env has reached its FIRST target. Gates the
        # pre-reach time penalty (rew_scales.prereach_time), which switches itself off for good
        # the moment the fish scores -- see _get_rewards.
        self._first_reach_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # adaptive-curriculum state: the LIVE success radius (shrinks from start->min as reaches accumulate)
        self._reach_total = 0
        self._cur_radius = (float(self.cfg.curriculum_radius_start) if getattr(self.cfg, "curriculum", False)
                            else float(self.cfg.success_radius))
        # DISTANCE curriculum state: the LIVE target-spawn distance mean (GROWS start->max as reaches
        # accumulate). Close targets make reaching the COMMON case (dense success signal, chaining
        # practice) before the distance is pushed out -- the inverse lever of the radius curriculum.
        self._cur_dist_mean = (float(getattr(self.cfg, "dist_curriculum_start", 0.6))
                               if getattr(self.cfg, "dist_curriculum", False)
                               else float(self.cfg.target_dist_mean))
        # rate-gate state (dist_curriculum_gated): promotion needs a GOOD RECENT RATE, not a cumulative
        # count (a failing policy also scrapes together 300 reaches eventually -> difficulty ratchets up
        # regardless of competence -- the observed 0.26->0.03 decay). Demotion re-grooves eroding skill.
        self._gate_reaches = 0
        self._gate_steps = 0
        self._last_gate_rate = float("nan")
        self._episode_reward_queue = deque(maxlen=100)
        # rolling window of recently-ended-episode outcomes (1=reached target, 0=not)
        self._success_hist = deque(maxlen=500)
        self._return_hist = deque(maxlen=500)    # rolling episode RETURNS -> extras["log"]["episode_return"]
        self._reach_hist = deque(maxlen=500)     # rolling FINAL per-episode reach counts -> reaches_per_episode
        self._nan_reported = False

        # ---- REWARD MANIFEST -------------------------------------------------------------------
        # Print exactly which reward terms are live. Every scale defaults to 0.0 and is enabled only
        # by a hydra override, so "is term X on?" is otherwise invisible until you dig through
        # params/env.yaml -- and a term that is silently off (or silently on) has burned this project
        # before. One line in the log removes the doubt, the same way the FEM-material print does.
        _rs = self.cfg.rew_scales
        _all = [f for f in ("distance", "success", "heading", "time", "effort", "spin", "upright",
                            "z_band", "launch", "approach", "backward", "offaxis")
                if hasattr(_rs, f)]
        _on = [f"{f}={float(getattr(_rs, f)):g}" for f in _all if float(getattr(_rs, f)) != 0.0]
        _off = [f for f in _all if float(getattr(_rs, f)) == 0.0]
        _gate = float(getattr(self.cfg, "align_gate_floor", 1.0))
        print(f"[SalmonSwim] REWARD ACTIVE: {', '.join(_on) if _on else '(none)'}"
              f" | align_gate_floor={_gate:g} ({'OFF' if _gate >= 1.0 else 'ON'})"
              f" | OFF: {', '.join(_off) if _off else '(none)'}", flush=True)
        # Call the two orientation patches out by name. They exist ONLY to punish tail-first /
        # sideways travel, which is now supposed to be the AMP discriminator's job (its motion feature
        # is written in the body frame, so it can finally SEE travel direction). Keeping them at 0
        # while that is evaluated is deliberate -- if they are non-zero, the two mechanisms are
        # fighting for the same behaviour and neither result is interpretable.
        _b, _o = float(getattr(_rs, "backward", 0.0)), float(getattr(_rs, "offaxis", 0.0))
        print(f"[SalmonSwim] orientation reward patches: backward={_b:g} offaxis={_o:g}"
              f" -> {'DISABLED (discriminator owns swim direction)' if (_b == 0.0 and _o == 0.0) else 'ENABLED'}",
              flush=True)

        # top-down rolling video recorder (env 0)
        self._rec = bool(self.cfg.record_video) and getattr(self, "_vcam", None) is not None
        if self._rec:
            import os
            os.makedirs(self.cfg.video_dir, exist_ok=True)
            self._vframes = []
            self._vclip = 0       # epoch counter
            self._vstep = 0
            self._vcam_posed = False
            print(f"[SalmonSwim] recording top-down env0 video -> {self.cfg.video_dir} "
                  f"(1 clip / {self.cfg.video_clip_steps} steps, keep last {self.cfg.video_keep})", flush=True)

        # 3D-trajectory recorder (env 0; no camera needed). Each RUN -> its OWN timestamped subfolder;
        # save every Nth completed episode, KEEP ALL (no rolling). Plus a live fixed-name file.
        self._rec_traj = bool(self.cfg.record_trajectory)
        if self._rec_traj:
            import os, time as _time
            self._traj_run_dir = os.path.join(self.cfg.traj_dir, _time.strftime("run_%Y%m%d_%H%M%S"))
            os.makedirs(self._traj_run_dir, exist_ok=True)
            self._traj_xyz = []   # env-local root positions for the current env0 episode
            self._traj_ep = 0     # completed-episode counter (env0)
            self._traj_saved = 0  # number of trajectories actually written
            self._traj_gstep = 0  # global control-step counter (epoch ~= gstep // horizon)
            _n = max(1, int(getattr(self.cfg, "traj_save_every_n_episodes", 5)))
            print(f"[SalmonSwim] recording env0 3D trajectory -> {self._traj_run_dir} "
                  f"(1 PNG+OBJ every {_n} env0 episodes, keep ALL; live file every "
                  f"{int(getattr(self.cfg, 'traj_live_every', 0))} steps)", flush=True)

        # throughput meter: env-frames/s = num_envs x control-step rate; printed ~every 5 s
        import time
        self._tp_t = time.perf_counter()
        self._tp_n = 0

        # FEM-reset support (inference): view of the soft body + its rest (root-local) shape, so
        # _reset_idx can teleport the soft body WITH the bones instead of leaving it to snap back.
        self._soft_view = None
        if self.cfg.reset_deformable and self.cfg.with_deformable:
            self._setup_deformable_reset()

    def _record_step(self):
        """Grab one top-down frame of env 0; flush a clip every video_clip_steps."""
        if not self._rec:
            return
        cam = self._vcam
        if not self._vcam_posed:
            # center on the swim area: midpoint between the spawn (env origin) and the
            # target offset, at the fish's swim height
            look = self.scene.env_origins[0].clone()
            look[0] += 0.5 * float(self.cfg.target_offset_xy[0])
            look[1] += 0.5 * float(self.cfg.target_offset_xy[1])
            look[2] += self.cfg.target_root_height
            eye = look.clone()
            eye[1] -= 1e-3   # tiny offset so straight-down isn't degenerate
            eye[2] += self.cfg.video_cam_height
            cam.set_world_poses_from_view(eye.unsqueeze(0), look.unsqueeze(0))
            self._vcam_posed = True
        cam.update(float(self.cfg.sim.dt * self.cfg.decimation))
        try:
            rgb = cam.data.output["rgb"]
        except (KeyError, TypeError):
            return
        if rgb is None or rgb.numel() == 0:
            return
        frame = rgb[0, ..., :3].detach().to("cpu").numpy()
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        self._vframes.append(frame)
        self._vstep += 1
        if self._vstep >= self.cfg.video_clip_steps:
            self._save_clip()

    def _save_clip(self):
        import glob
        import os
        if self._vframes:
            path = os.path.join(self.cfg.video_dir, f"epoch_{self._vclip:06d}.mp4")
            try:
                import imageio.v2 as imageio
                with imageio.get_writer(path, fps=int(self.cfg.video_fps)) as w:
                    for f in self._vframes:
                        w.append_data(f)
                print(f"[SalmonSwim] saved {path} ({len(self._vframes)} frames)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[SalmonSwim] video save failed: {e}", flush=True)
            # rolling: keep only the most recent N clips
            clips = sorted(glob.glob(os.path.join(self.cfg.video_dir, "epoch_*.mp4")))
            for old in clips[:-int(self.cfg.video_keep)]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        self._vframes = []
        self._vstep = 0
        self._vclip += 1

    # ---- env0 3D-trajectory recorder (PNG + OBJ, target marked) ----
    def _record_traj_step(self, done0: bool):
        """Append env0's (env-local) root position each step; on episode end, save PNG+OBJ."""
        if not self._rec_traj:
            return
        self._traj_gstep += 1                # one control step per env.step()
        p = self.robot.data.root_state_w[0, 0:3] - self.scene.env_origins[0]
        if torch.isfinite(p).all():          # skip the NaN sample on a blow-up step
            self._traj_xyz.append(p.detach().cpu().numpy().copy())
        # LIVE: overwrite the fixed-name partial trajectory every traj_live_every steps mid-episode
        live_every = int(getattr(self.cfg, "traj_live_every", 0) or 0)
        if live_every > 0 and len(self._traj_xyz) >= 2 and (self._traj_gstep % live_every) == 0:
            self._save_traj_live()
        if done0:                            # env0 episode ended
            self._traj_ep += 1
            n = max(1, int(getattr(self.cfg, "traj_save_every_n_episodes", 5)))
            if (self._traj_ep % n) == 1 or n == 1:   # save episodes 1, 1+n, 1+2n, ... (early one + every n)
                self._save_traj_clip()
            self._traj_xyz = []

    def _save_traj_clip(self):
        """Write env0's just-finished episode trajectory (PNG+OBJ) into this run's subfolder and
        KEEP it (no rolling deletion). Filename carries the epoch proxy + the episode index."""
        import os
        if len(self._traj_xyz) < 2:
            return                           # too short to be meaningful
        traj = np.asarray(self._traj_xyz, dtype=np.float32)                      # (T,3) env-local
        tgt = (self.target_positions_w[0] - self.scene.env_origins[0]).detach().cpu().numpy()
        # epoch-equivalent = total control steps / rl_games horizon_length (== video_clip_steps,
        # documented as one epoch). env can't see the true rl_games epoch, so this is the proxy.
        horizon = max(1, int(self.cfg.video_clip_steps))
        epoch = self._traj_gstep // horizon
        base = os.path.join(self._traj_run_dir, f"traj_ep{epoch:06d}_episode{self._traj_ep:04d}")
        try:
            self._write_traj_obj(base + ".obj", traj, tgt)
        except Exception as e:               # noqa: BLE001
            print(f"[SalmonSwim] traj OBJ save failed: {e}", flush=True)
        try:
            self._plot_traj(base + ".png", traj, tgt)
        except Exception as e:               # noqa: BLE001
            print(f"[SalmonSwim] traj PNG save failed: {e}", flush=True)
        self._traj_saved += 1
        print(f"[SalmonSwim] saved env0 trajectory {os.path.basename(base)} ({len(traj)} pts, "
              f"~epoch {epoch}, episode {self._traj_ep}, total kept {self._traj_saved}, "
              f"target=({tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f}))", flush=True)

    def _save_traj_live(self):
        """Overwrite a FIXED-name traj_live_env0.{png,obj} with env0's CURRENT (partial) episode
        path, so it can be watched building up mid-episode. No rolling, no episode/idx in the name;
        excluded from the per-episode rolling cleanup."""
        import os
        if len(self._traj_xyz) < 2:
            return
        traj = np.asarray(self._traj_xyz, dtype=np.float32)
        tgt = (self.target_positions_w[0] - self.scene.env_origins[0]).detach().cpu().numpy()
        base = os.path.join(self._traj_run_dir, "traj_live_env0")
        try:
            self._write_traj_obj(base + ".obj", traj, tgt)
        except Exception as e:               # noqa: BLE001
            print(f"[SalmonSwim] live traj OBJ save failed: {e}", flush=True)
        try:
            self._plot_traj(base + ".png", traj, tgt)
        except Exception as e:               # noqa: BLE001
            print(f"[SalmonSwim] live traj PNG save failed: {e}", flush=True)

    @staticmethod
    def _write_traj_obj(path, traj, tgt):
        """Wavefront OBJ: 'o trajectory' polyline + 'o target' octahedron marker (meters)."""
        L = ["# env0 3D swim trajectory (env-local frame), units = meters\n",
             f"# target_position {tgt[0]:.6f} {tgt[1]:.6f} {tgt[2]:.6f}\n",
             "o trajectory\n"]
        for p in traj:
            L.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        L.append("l " + " ".join(str(i + 1) for i in range(len(traj))) + "\n")
        n = len(traj)
        r = 0.06                             # target marker half-size (m)
        L.append("o target\n")
        for dx, dy, dz in [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]:
            L.append(f"v {tgt[0]+dx:.6f} {tgt[1]+dy:.6f} {tgt[2]+dz:.6f}\n")
        for a, b, c in [(1, 3, 5), (3, 2, 5), (2, 4, 5), (4, 1, 5),
                        (3, 1, 6), (2, 3, 6), (4, 2, 6), (1, 4, 6)]:
            L.append(f"f {n+a} {n+b} {n+c}\n")
        with open(path, "w") as fh:
            fh.writelines(L)

    @staticmethod
    def _plot_traj(path, traj, tgt):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="tab:blue", lw=1.5, label="env0 path")
        ax.scatter(*traj[0], c="green", s=70, label="start", depthshade=False)
        ax.scatter(*traj[-1], c="orange", s=45, label="end", depthshade=False)
        ax.scatter(*tgt, c="red", marker="*", s=260, label="TARGET", depthshade=False)
        ax.text(float(tgt[0]), float(tgt[1]), float(tgt[2]),
                f"  target ({tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f})", color="red", fontsize=8)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"env0 swim trajectory  ({len(traj)} steps)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _print_throughput(self):
        """EVERY control step: throughput + blow-up status merged into one line. The blow-up fields
        (now/cum/distinct_envs/resets/jvel_max) are stashed on self by _get_dones each step."""
        import time
        now = time.perf_counter()
        if not hasattr(self, "_tp_start"):
            self._tp_start = now
        self._tp_n += 1
        step_dt = now - self._tp_t
        self._tp_t = now
        rate = self._tp_n / max(1e-6, now - self._tp_start)     # cumulative avg control-steps/s
        blew_now = getattr(self, "_blew_now", 0)
        cum = getattr(self, "_blowup_total", 0)
        resets = getattr(self, "_reset_total", 0)
        jvmax = getattr(self, "_jvel_max", float("nan"))
        nblown = len(getattr(self, "_blown_env_ids", ()))
        step0 = int(self.episode_length_buf[0])
        epoch = getattr(self, "_bstep_total", 0) // max(1, int(self.cfg.video_clip_steps))  # ~rl_games epoch
        tag = "   <<< BLOWUP THIS STEP" if blew_now else ""
        print(f"[epoch {epoch} global_step {self._tp_n}] {rate:.2f} ctrl-steps/s ({step_dt:.2f}s/step, "
              f"{rate * self.num_envs:,.0f} env-frames/s) | env0_episode_step={step0} "
              f"jvel_max={jvmax:.1f}/{self.cfg.joint_velocity_limit:.0f} | "
              f"blowups now={blew_now} cum={cum} distinct_envs={nblown} resets={resets}{tag}", flush=True)

    def _spawn_env0_extras(self, env0_ns: str):
        """Hook: spawn extra per-env prims under env0 BEFORE clone_environments. No-op by default;
        the tank task overrides this to spawn the four collidable walls."""
        return

    def _disable_bone_colliders_post_scene(self):
        from pxr import Usd
        n_seen = n_off = 0
        for prim in Usd.PrimRange(self.scene.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not path.startswith("/World/envs/") or "/skeleton/" not in path:
                continue
            attr = prim.GetAttribute("physics:collisionEnabled")
            if not (attr and attr.IsValid()):
                continue
            n_seen += 1
            if attr.Get():
                attr.Set(False); n_off += 1
        print(f"[SalmonSwim] post-scene bone colliders: {n_seen} found under */skeleton/*, disabled {n_off}", flush=True)

    def _setup_scene(self):
        from pxr import UsdGeom

        UsdGeom.SetStageMetersPerUnit(self.scene.stage, 1)
        self.robot = Articulation(self.cfg.robot_cfg)

        # post-spawn USD edits on env_0 BEFORE cloning (propagate to clones):
        # deactivate the deformable, disable bone colliders, author the D6 DriveAPI.
        env0 = f"{self.scene.env_ns}/env_0/skeleton"
        n_soft, n_drive, n_coll = _prepare_env_assets(
            self.scene.stage, env0, self.cfg.control_mode, keep_deformable=self.cfg.with_deformable,
            joint_limit_deg=self.cfg.joint_limit_deg, fem_youngs_modulus=self.cfg.fem_youngs_modulus,
            fem_elasticity_damping=self.cfg.fem_elasticity_damping,
            fem_damping_scale=self.cfg.fem_damping_scale,
            deformable_token=getattr(self.cfg, "deformable_prim_token", "deformable_salmon"),
            fix_multi_articulation_root=getattr(self.cfg, "fix_multi_articulation_root", False),
            fix_reflected_deformable=getattr(self.cfg, "fix_reflected_deformable", False),
            keep_bone_colliders=getattr(self.cfg, "keep_bone_colliders", False))
        _soft_word = "kept-active" if self.cfg.with_deformable else "deactivated"
        print(f"[SalmonSwim] env_0 prep: {_soft_word} {n_soft} soft, disabled {n_coll} colliders, "
              f"authored DriveAPI on {n_drive} D6 joints", flush=True)

        if self.cfg.record_video:
            # tint env_0's skeleton (no MDL material) so the fish is visible in the
            # top-down video against the bright dome background
            from pxr import Gf, UsdGeom, Vt
            col = Vt.Vec3fArray([Gf.Vec3f(0.90, 0.30, 0.15)])  # salmon-orange
            for prim in self.scene.stage.Traverse():
                p = str(prim.GetPath())
                if not p.startswith(env0) or getattr(self.cfg, "deformable_prim_token", "deformable_salmon") in p:
                    continue
                g = UsdGeom.Gprim(prim)
                if g:
                    try:
                        g.CreateDisplayColorAttr(col)
                    except Exception:  # noqa: BLE001
                        pass

        # target marker (no ground plane -- the fish swims free in zero-g water). For inference a
        # translucent red sphere of radius=success_radius shows the actual success region;
        # otherwise the small red cuboid.
        # NOTE the AMP cfg family ships draw_target_marker=False (salmon_amp_tank_cfg.py), which used
        # to silently swallow marker_success_sphere=true too. An explicit request for the success
        # sphere now overrides that gate -- you asked to see it, so you see it.
        if getattr(self.cfg, "draw_target_marker", True) or self.cfg.marker_success_sphere:
            if self.cfg.marker_success_sphere:
                # DYNAMIC success-region spheres: the old static sphere was spawned ONCE at the fixed
                # target_offset_xy corner, so with random_target/multi_target it sat at the wrong spot
                # forever. Instead we create VisualizationMarkers AFTER the scene is built (see
                # __init__) and re-aim them every control step at target_positions_w, scaled to the
                # LIVE curriculum radius (_cur_radius), one sphere per env. Nothing to spawn here.
                self._want_success_markers = True
                print("[SalmonSwim] target marker = DYNAMIC translucent success spheres "
                      "(follow target_positions_w, radius = live _cur_radius)", flush=True)
            else:
                marker_cfg = self.cfg.target_marker_cfg
                marker_z = float(self.cfg.target_marker_height)
                target_translation = (float(self.cfg.target_offset_xy[0]),
                                      float(self.cfg.target_offset_xy[1]), marker_z)
                marker_cfg.func(prim_path=f"{self.scene.env_ns}/env_0/target_marker",
                                cfg=marker_cfg, translation=target_translation)

        # subclass hook: spawn extra per-env prims (e.g. tank walls) under env_0 BEFORE cloning,
        # so the cloner replicates them into every env.
        self._spawn_env0_extras(f"{self.scene.env_ns}/env_0")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" or not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        # dim dome when recording (intensity 2000 overexposes -> the tinted fish washes
        # out white); a dim dome + a key light gives the top-down video good contrast.
        if self.cfg.record_video:
            dome = sim_utils.DomeLightCfg(intensity=150.0, color=(0.30, 0.40, 0.55))
            dome.func("/World/Light", dome)
            key = sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.98, 0.92))
            key.func("/World/KeyLight", key, orientation=(0.92, 0.0, 0.38, 0.0))
        else:
            # UNDERWATER backdrop (light-blue dome = viewport background, warm key light,
            # sandy visual-only seabed). Lives in a STANDALONE, user-editable USD; edit
            # agents/underwater_background.usd and every later launch picks it up.
            _bg = Path(__file__).parent / "agents/underwater_background.usd"
            if _bg.exists():
                bg_cfg = sim_utils.UsdFileCfg(usd_path=str(_bg))
                bg_cfg.func("/World/Background", bg_cfg)
            else:
                light_cfg = sim_utils.DomeLightCfg(intensity=1600.0, color=(0.45, 0.68, 0.92))
                light_cfg.func("/World/Light", light_cfg)

        # top-down recording camera over env 0 (single camera, NOT under /World/envs so
        # it is not cloned). Needs --enable_cameras at launch.
        self._vcam = None
        if self.cfg.record_video:
            from isaaclab.sensors import Camera, CameraCfg
            cam_cfg = CameraCfg(
                prim_path="/World/top_cam", update_period=0.0,
                height=self.cfg.video_cam_res, width=self.cfg.video_cam_res, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 1.0e4)))
            self._vcam = Camera(cam_cfg)

    def _configure_gym_env_spaces(self):
        # ACTION-SPACE RESTRICTION: optionally drive only the joints whose name ends with
        # cfg.control_dof_suffix. The D6 joints are named "D6Joint_NN:D" where D in {0,1,2} selects
        # one of the 3 rotation axes. MEASURED: family ":1" alone reaches +/-0.53 BL/s with a scripted
        # wave -- it is provably sufficient for propulsion -- while the trained policy spreads its
        # motion 33/35/32% across all three families, so ~2/3 of its joint power makes roll and pitch
        # (drag + recoil, no thrust). Restricting to ":1" removes that waste and cuts the exploration
        # space from 21 dims to 7. None = drive every joint (original behaviour).
        _dof_suffix = getattr(self.cfg, "control_dof_suffix", None)
        if _dof_suffix:
            _names = self.robot.data.joint_names
            _keep = [i for i, n in enumerate(_names) if n.endswith(str(_dof_suffix))]
            assert _keep, f"control_dof_suffix={_dof_suffix!r} matched no joints in {_names}"
            self._control_joint_ids = torch.tensor(_keep, device=self.device, dtype=torch.long)
            print(f"[SalmonSwim] action space restricted to {len(_keep)}/{len(_names)} joints "
                  f"ending {_dof_suffix!r}: {[_names[i] for i in _keep]}", flush=True)
        else:
            self._control_joint_ids = torch.arange(self.robot.num_joints, device=self.device)
        self._num_actions = self._control_joint_ids.shape[0]
        # obs holds ALL joints (the policy still SEES every joint even when control_dof_suffix
        # restricts which ones it can DRIVE), so this must use num_joints, not _num_actions.
        obs_dim = 18 + 2 * self.robot.num_joints   # grav+linv+angv+heading+target_delta+up_dir = 18
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,))}
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_actions), device=self.device)
        self.state_space = None

    # ---- control ----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_actions)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        if self.cfg.control_mode == "position":
            # action -> joint target angle, clamped to the soft joint limits
            tgt = actions * self.cfg.pos_action_scale
            # index the limits by the CONTROLLED joints -- with control_dof_suffix the action
            # vector is narrower than the joint count, so the full-width limits would not broadcast.
            lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
            hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
            self._pos_targets = torch.clamp(tgt, lo, hi)
        else:
            self._torques = actions * self.cfg.action_scale

    def _panel_hydro_wrench(self):
        """STRIP-THEORY panel hydro: the .obj surface panels (pinned to bones) ARE the force source.
        Each panel's world pose+velocity come from its bone (v = v_bone + w x r_offset); each panel is a
        flat plate -> normal FORM drag (-0.5 rho Cd_n A |v.n| (v.n) n) + tangential SKIN drag. Summed per
        bone. Passive/dissipative (drag opposes motion) -> stable, same per-bone clips as the ellipsoid path."""
        E, B = self.num_envs, self._nb
        bid = self._panel_bone                                        # (Pn,)
        Pn = bid.shape[0]
        bp = self.robot.data.body_link_pos_w                          # (E,B,3)
        bq = self.robot.data.body_link_quat_w                         # (E,B,4)
        blv = self.robot.data.body_link_lin_vel_w                     # (E,B,3)
        bav = self.robot.data.body_link_ang_vel_w                     # (E,B,3)
        gbq = bq[:, bid].reshape(-1, 4)                               # (E*Pn,4)
        r_w = math_utils.quat_apply(gbq, self._panel_r.unsqueeze(0).expand(E, -1, -1).reshape(-1, 3)).reshape(E, Pn, 3)
        n_w = math_utils.quat_apply(gbq, self._panel_n.unsqueeze(0).expand(E, -1, -1).reshape(-1, 3)).reshape(E, Pn, 3)
        v_p = blv[:, bid] + torch.linalg.cross(bav[:, bid], r_w)      # (E,Pn,3) panel world velocity
        vn = (v_p * n_w).sum(-1, keepdim=True)                        # (E,Pn,1) normal component
        v_t = v_p - vn * n_w                                          # tangential component
        area = self._panel_a.view(1, Pn, 1)
        Fn = -0.5 * self.cfg.rho * self._panel_cd_n * area * vn.abs() * vn * n_w
        Ft = -0.5 * self.cfg.rho * self._panel_cd_t * area * v_t.norm(dim=-1, keepdim=True) * v_t
        Fp = torch.nan_to_num(Fn + Ft)                               # (E,Pn,3)
        Tp = torch.linalg.cross(r_w, Fp)                             # torque about the bone origin
        idx = bid.view(1, Pn, 1).expand(E, Pn, 3)
        F = torch.zeros(E, B, 3, device=self.device, dtype=Fp.dtype).scatter_add_(1, idx, Fp)
        T = torch.zeros(E, B, 3, device=self.device, dtype=Fp.dtype).scatter_add_(1, idx, Tp)
        c = self.cfg.hydro_force_clip
        ct = getattr(self.cfg, "hydro_torque_clip", None)
        ct = float(c) if ct is None else float(ct)
        F = torch.nan_to_num(F).clamp(-c, c)
        T = torch.nan_to_num(T).clamp(-ct, ct)
        # IMPULSE LIMITER: explicit quadratic drag goes numerically UNSTABLE once the per-substep
        # impulse can reverse the bone's velocity (F*dt/m > ~v): each substep then AMPLIFIES the
        # relative velocity instead of damping it -> runaway speed (measured: median 5.9, up to
        # 28.7 BL/s right before FEM blow-ups on the 42-bone fish; a fixed 20 N clip lets a 12 g
        # bone gain ~14 m/s per 1/120 substep). Physical drag can at most STOP a body within a
        # step, never reverse it, so cap |F| at beta*m*|v|/dt per bone (beta<1 safety). Inactive
        # at normal swim speeds -- it only removes the unphysical high-speed regime.
        # OPT-IN (default OFF): the base hydro model must stay bit-for-bit the original. Tasks
        # that want the limiter set cfg.hydro_impulse_beta explicitly (the Reach10 protocol does,
        # =0.5, matching the physics its policies were trained in since 2026-08-13).
        beta = getattr(self.cfg, "hydro_impulse_beta", 0.0)
        if beta:
            if not hasattr(self, "_bone_mass_v"):
                self._bone_mass_v = self.robot.data.default_mass[0].to(self.device).float().view(1, -1)
            # per-bone speed scale = the FASTEST panel on that bone (a beating tail bone can spin
            # fast while its origin barely moves -- origin speed would wrongly zero its drag)
            vp_mag = v_p.norm(dim=-1)                                 # (E,Pn)
            vmag = torch.zeros(E, B, device=self.device, dtype=vp_mag.dtype).scatter_reduce_(
                1, bid.view(1, Pn).expand(E, Pn), vp_mag, reduce="amax", include_self=False)
            f_max = float(beta) * self._bone_mass_v * vmag / float(self.cfg.sim.dt)
            fmag = F.norm(dim=-1).clamp_min(1e-9)                     # (E,B)
            s = (f_max / fmag).clamp(max=1.0).unsqueeze(-1)
            F = F * s
            T = T * s                                                 # same scale keeps the wrench direction
        return F, T

    def _hydro_wrench(self):
        """Per-bone MuJoCo fluid wrench (E,B,3) world-frame from current body velocities."""
        if getattr(self, "_hydro_mode", "ellipsoid") == "panels":
            return self._panel_hydro_wrench()
        V = self.robot.data.body_link_lin_vel_w.reshape(-1, 3)
        W = self.robot.data.body_link_ang_vel_w.reshape(-1, 3)
        Q = self.robot.data.body_link_quat_w.reshape(-1, 4)
        F, T = mujoco_fluid_wrench(
            V, W, Q, self._semi, self._R_pl,
            rho=self.cfg.rho, visc=self.cfg.visc, cd_blunt=self.cfg.cd_blunt,
            cd_slender=self.cfg.cd_slender, cd_angular=self.cfg.cd_angular,
            ck=self.cfg.ck, cm=self.cfg.cm)
        c = self.cfg.hydro_force_clip
        ct = getattr(self.cfg, "hydro_torque_clip", None)
        ct = float(c) if ct is None else float(ct)
        F = torch.nan_to_num(F).clamp(-c, c).reshape(self.num_envs, self._nb, 3)
        T = torch.nan_to_num(T).clamp(-ct, ct).reshape(self.num_envs, self._nb, 3)
        return F, T

    def _apply_action(self) -> None:
        if self.cfg.control_mode == "position":
            self.robot.set_joint_position_target(self._pos_targets, joint_ids=self._control_joint_ids)
        else:
            self.robot.set_joint_effort_target(self._torques, joint_ids=self._control_joint_ids)
        F, T = self._hydro_wrench()
        F, T = self._angular_impulse_layer(F, T)
        self.robot.set_external_force_and_torque(F, T, is_global=True)
        self._body_velocity_governor()       # anti-runaway circuit-breaker (per physics substep)
        self._clamp_fem_nodal_velocities()   # CFL circuit-breaker (per physics substep)

    def _body_velocity_governor(self):
        """OPT-IN body-level circuit-breaker ON TOP of the (untouched) hydro model. The
        spin-and-sail runaway (rear-target C-curl lock -> ~5 BL/s sideways drift on the Misty)
        is a numerical artifact of explicit drag at high slip/spin; rather than reshaping any
        force (the angular-impulse cap variant broke normal turning -- articulated bones carry
        whole-chain inertia, so per-bone torque caps engage in the healthy regime), cap the ROOT
        linear/angular velocity at ceilings ~3x the healthiest observed swimming. Below the
        ceilings NOTHING is altered -- forces, gait and turning are bit-for-bit the trained
        physics; above them (a regime that should not exist) velocity is rescaled in place.
        Host-sync check gates the write so clean steps cost one bool. Off unless
        cfg.body_speed_cap_bl / cfg.body_yaw_cap_rad are set."""
        vcap_bl = getattr(self.cfg, "body_speed_cap_bl", None)
        wcap = getattr(self.cfg, "body_yaw_cap_rad", None)
        if not vcap_bl and not wcap:
            return
        rv = self.robot.data.root_com_vel_w.clone()                       # (E,6) lin+ang
        over = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if vcap_bl:
            vmax = float(vcap_bl) * self._body_length
            sp = rv[:, 0:3].norm(dim=1, keepdim=True)
            over |= (sp > vmax).squeeze(-1)
            rv[:, 0:3] *= (vmax / sp.clamp_min(1e-9)).clamp(max=1.0)
        if wcap:
            wm = rv[:, 3:6].norm(dim=1, keepdim=True)
            over |= (wm > float(wcap)).squeeze(-1)
            rv[:, 3:6] *= (float(wcap) / wm.clamp_min(1e-9)).clamp(max=1.0)
        # PER-ENV write (see the FEM guard): only the envs actually over cap are written, so the
        # guard can never perturb a compliant env just because another env tripped it.
        if bool(over.any()):
            ids = over.nonzero(as_tuple=False).flatten()
            self.robot.write_root_com_velocity_to_sim(rv[ids], env_ids=ids)

    def _angular_impulse_layer(self, F, T):
        """OPT-IN post-filter ON TOP of the (untouched) hydro model — the ANGULAR twin of the
        linear impulse limiter. During a fast spin, the panel torque OPPOSING the rotation can
        exceed the angular momentum a bone can shed in one substep; the explicit integrator then
        REVERSES and re-amplifies the spin each step, and the spinning curled body 'sails'
        sideways at unphysical speed (measured: ~5 BL/s drift on the 8-joint Misty after a
        rear-target C-curl lock). Cap ONLY the component of T anti-parallel to each bone's
        angular velocity at beta*I*|w|/dt (drag can stop a spin within a step, never reverse
        it); spin-UP torques and all torque on non-rotating bones pass through UNCHANGED, so a
        gliding fish still turns normally. Inactive unless cfg.hydro_angular_impulse_beta is
        set — the default model is bit-for-bit unchanged."""
        beta = getattr(self.cfg, "hydro_angular_impulse_beta", None)
        if not beta:
            return F, T
        if not hasattr(self, "_bone_inertia_s"):
            inert = self.robot.data.default_inertia[0].to(self.device).float()   # (B,9)
            self._bone_inertia_s = inert.view(-1, 3, 3).diagonal(dim1=1, dim2=2) \
                .mean(-1).clamp_min(1e-12).view(1, -1)                           # (1,B) scalar I
        w = self.robot.data.body_link_ang_vel_w                                  # (E,B,3)
        wmag = w.norm(dim=-1)                                                    # (E,B)
        w_hat = w / wmag.clamp_min(1e-9).unsqueeze(-1)
        along = (T * w_hat).sum(-1, keepdim=True)                                # signed, along spin
        cap = (float(beta) * self._bone_inertia_s * wmag / float(self.cfg.sim.dt)).unsqueeze(-1)
        capped = torch.maximum(along, -cap)                                      # limit opposing part only
        return F, T + (capped - along) * w_hat

    def _clamp_fem_nodal_velocities(self):
        """CFL circuit-breaker (L1): cap each FEM node's velocity MAGNITUDE every physics substep so a
        stiff-mode divergence can't grow geometrically (exp blow-up -> bounded drift). No-op below the
        cap (a host-side peak>cap check gates the write), so normal-swim (<~1 m/s nodal) motion is never
        touched; only a would-be blow-up is caught. Runs in _apply_action (between scene update and the
        next sim.step) so the integrator steps from the capped state. Preventing the divergence also
        prevents the corrupted internal solver state that caused the deterministic reset-loop."""
        vmax = getattr(self.cfg, "fem_nodal_vel_clamp", None)
        view = getattr(self, "_soft_view", None)
        if vmax is None or view is None:
            return
        vmax = float(vmax)
        try:
            vel = view.get_simulation_mesh_nodal_velocities()            # (E,N,3) device tensor
            speed = torch.linalg.norm(vel, dim=-1, keepdim=True)         # (E,N,1)
            # PER-ENV gate + PER-ENV write. A global `amax > vmax` gate followed by a full-tensor
            # write makes the guard's firing rate depend on the ENV COUNT (with 256 envs some env
            # is nearly always over cap), so a well-behaved env would get its nodal velocities
            # rewritten because a DIFFERENT env misbehaved -- an env-count-dependent perturbation.
            # Writing only the offending rows keeps every env's trajectory identical at any N.
            bad = (~torch.isfinite(speed).all(dim=(1, 2))) | (speed.amax(dim=(1, 2)) > vmax)  # (E,)
            if bool(bad.any()):
                ids = bad.nonzero(as_tuple=False).flatten()
                v = torch.nan_to_num(vel[ids], nan=0.0, posinf=vmax, neginf=-vmax)
                s = torch.linalg.norm(v, dim=-1, keepdim=True)
                v = v * (vmax / s.clamp_min(1e-6)).clamp(max=1.0)        # rescale ONLY over-cap nodes
                view.set_simulation_mesh_nodal_velocities(v, indices=ids)
                self._fem_clamp_hits = getattr(self, "_fem_clamp_hits", 0) + int(ids.numel())
        except Exception:  # noqa: BLE001  -- fall back to guard+cooldown if the mid-loop write is rejected
            pass

    # ---- observations ----
    def _get_observations(self) -> dict:
        # re-aim the success-region spheres at the CURRENT targets, scaled to the LIVE curriculum
        # radius. visualize() early-returns when the instancer isn't visible, so this is ~free
        # headless; with the GUI it makes the spheres jump the instant a target is resampled.
        if self._success_markers is not None:
            _r = float(self._cur_radius)
            self._success_markers.visualize(
                translations=self.target_positions_w,
                scales=torch.full((self.num_envs, 3), _r, device=self.device))

        root_state = self.robot.data.root_state_w
        projected_gravity = self.robot.data.projected_gravity_b
        root_lin_vel = root_state[:, 7:10] * self.cfg.obs_scales.root_lin_vel
        root_ang_vel = root_state[:, 10:13] * self.cfg.obs_scales.root_ang_vel
        joint_pos_error = (self.joint_pos - self._default_joint_pos) * self.cfg.obs_scales.joint_pos
        joint_vel = self.joint_vel * self.cfg.obs_scales.joint_vel
        target_delta = self.target_positions_w[:, 0:3] - root_state[:, 0:3]   # 3D

        root_quat = root_state[:, 3:7]
        # body-forward axis. Default -X (the historical convention). For assets whose HEAD is at +X
        # (auto-skeleton + Meshy fish both are), set cfg.body_forward_sign=+1 so the policy's heading_dir
        # points at the real nose instead of the tail. ONLY affects this policy obs -- the AMP feature
        # (head-anchored via its own reversal) and the direction-agnostic reward are untouched.
        _fwd_sign = float(getattr(self.cfg, "body_forward_sign", -1.0))
        forward_b = torch.tensor([_fwd_sign, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
        forward_b = forward_b.repeat(self.num_envs, 1)
        heading_vec = math_utils.quat_apply(root_quat, forward_b)              # 3D body-forward in world
        heading_dir = heading_vec / torch.linalg.norm(heading_vec, dim=1, keepdim=True).clamp(min=1e-6)

        # UP-VECTOR: the fish's own up axis (local +Y) in world coords. REQUIRED because the water has
        # zero gravity, so `projected_gravity` is the ZERO vector -- without this the fish has no sense
        # of which way is up at all and cannot learn to hold its depth. Measured symptom: it pitches up
        # and floats 3-5 m above the target plane (horizontal travel ~0.01 m) and never returns, since
        # nothing restores it (no gravity, and the thin water damps very little).
        up_local = torch.tensor([0.0, 1.0, 0.0], dtype=root_state.dtype,
                                device=root_state.device).repeat(self.num_envs, 1)
        up_dir = math_utils.quat_apply(root_quat, up_local)
        obs = torch.cat(
            (projected_gravity, root_lin_vel, root_ang_vel,
             joint_pos_error, joint_vel, heading_dir, target_delta, up_dir), dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    # ---- rewards ----
    def _get_rewards(self) -> torch.Tensor:
        root_state = self.robot.data.root_state_w
        target_delta = self.target_positions_w[:, 0:3] - root_state[:, 0:3]   # 3D (x,y,z)
        distance = torch.linalg.norm(target_delta, dim=1)
        distance = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
        success = distance < self._cur_radius

        # progress toward target (3D)
        progress = (self._prev_distance - distance) * self.cfg.rew_scales.distance
        # heading: face the target (3D). Use body_forward_sign so it rewards pointing the HEAD (+X for
        # the misty/auto-skeleton fish) at the target, not the tail.
        target_dir = target_delta / distance.unsqueeze(-1).clamp(min=1e-6)
        _hfwd = float(getattr(self.cfg, "body_forward_sign", -1.0))
        forward_b = torch.tensor([_hfwd, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device).repeat(self.num_envs, 1)
        heading_vec = math_utils.quat_apply(root_state[:, 3:7], forward_b)
        heading_dir = heading_vec / torch.linalg.norm(heading_vec, dim=1, keepdim=True).clamp(min=1e-6)
        heading = (heading_dir * target_dir).sum(-1) * self.cfg.rew_scales.heading
        # bonuses / penalties. For multi_target, the reach bonus uses the reach flag set in _get_dones
        # (which then resampled the target), so the bonus survives the target moving away.
        _reached = getattr(self, "_reached_now", None)
        _succ = _reached if (_reached is not None and getattr(self.cfg, "multi_target", False)) else success
        success_rew = _succ.float() * self.cfg.rew_scales.success
        time_pen = -torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scales.time
        effort_pen = -(self.actions ** 2).mean(-1) * self.cfg.rew_scales.effort

        # anti-spin / keep-level (natural swimming): discourage moving by rotating the whole body
        _w_spin = float(getattr(self.cfg.rew_scales, "spin", 0.0))
        _w_up = float(getattr(self.cfg.rew_scales, "upright", 0.0))
        spin_pen = torch.zeros(self.num_envs, device=self.device)
        if _w_spin:
            spin_pen = spin_pen - _w_spin * (self.robot.data.root_ang_vel_w ** 2).sum(-1)
        if _w_up:
            up_local = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=root_state.dtype).repeat(self.num_envs, 1)
            up_world = math_utils.quat_apply(root_state[:, 3:7], up_local)
            spin_pen = spin_pen - _w_up * (1.0 - up_world[:, 2].clamp(-1.0, 1.0))

        # DEPTH BAND: free inside +/- z_band_halfwidth of the TARGET's height, then linear outside. The
        # water has zero gravity and (since the drag fix) very little damping, so a slight nose-up pitch
        # makes the fish climb metres out of the target plane and never come back -- measured: 78% of
        # episodes ended with the fish 3-5 m high having travelled ~0.01 m horizontally. The 3D distance
        # term alone punishes this far too weakly. A BAND (not a point target) so ordinary swimming
        # bob is free and only real departures are charged.
        _w_z = float(getattr(self.cfg.rew_scales, "z_band", 0.0))
        z_pen = torch.zeros(self.num_envs, device=self.device)
        if _w_z:
            _zhw = float(getattr(self.cfg, "z_band_halfwidth", 0.3))
            _zerr = (root_state[:, 2] - self.target_positions_w[:, 2]).abs()
            z_pen = -_w_z * (_zerr - _zhw).clamp(min=0.0)

        # ANTI-LAUNCH: penalize linear speed^2 weighted UP as the fish nears the target -> it must slow
        # down and STOP on the goal instead of flinging itself away (the "launch" that blocks the reach).
        _w_launch = float(getattr(self.cfg.rew_scales, "launch", 0.0))
        launch_pen = torch.zeros(self.num_envs, device=self.device)
        if _w_launch:
            _d0 = float(getattr(self.cfg, "launch_near_dist", 0.6))
            prox = (1.0 - distance / _d0).clamp(0.0, 1.0)               # 1 at target -> 0 at/beyond _d0
            lin_v2 = (self.robot.data.root_lin_vel_w ** 2).sum(-1)      # |v|^2 (world frame)
            launch_pen = -_w_launch * prox * lin_v2

        # POTENTIAL-BASED approach shaping: Phi = exp(-dist/scale) (higher the closer it is). The reward is
        # the CHANGE in potential, so it is STEEP near the target (fills the "comfortable valley" the orbit
        # sits in and pays hard for the final metre) yet telescopes to ~0 over any closed loop -- unlike a
        # LEVEL proximity bonus it does NOT create a "hover at the edge" trap, and reaching is not punished
        # (on a reach _prev_distance is reset to the new far target in _get_dones, so this term is ~0 there).
        _w_app = float(getattr(self.cfg.rew_scales, "approach", 0.0))
        approach = torch.zeros(self.num_envs, device=self.device)
        if _w_app:
            _asc = float(getattr(self.cfg, "approach_scale", 0.4))
            approach = _w_app * (torch.exp(-distance / _asc) - torch.exp(-self._prev_distance / _asc))

        # ---- SWIM FORWARD, DON'T REVERSE ----------------------------------------------------
        # `progress` above is direction-blind: it only asks "is the gap smaller?", never "did you
        # swim forward?". With rew_scales.heading disabled (it paid a motionless fish to turn in
        # place), NOTHING preferred forward over backward, so the cheapest way to reach a target
        # behind the fish was to reverse into it. Two terms fix that, and BOTH are multiplied by
        # actual motion, so neither can be farmed by a frozen fish the way `heading` could:
        #   1. align gate -- closing the gap pays full only when the head points at the target;
        #   2. backward penalty -- charged only while the fish travels tail-first.
        _v_fwd = (self.robot.data.root_lin_vel_w * heading_dir).sum(-1)   # >0 head-first, <0 reversing

        _floor = float(getattr(self.cfg, "align_gate_floor", 1.0))
        align_gate = torch.ones(self.num_envs, device=self.device)
        if _floor < 1.0:
            _cos = (heading_dir * target_dir).sum(-1).clamp(min=0.0)      # 1 facing target, 0 facing away
            align_gate = _floor + (1.0 - _floor) * _cos
            # gate the GAINS only -- a discounted loss would make facing away a cheap way to soften
            # the punishment for drifting off, which is the opposite of what we want.
            progress = torch.where(progress > 0, progress * align_gate, progress)
            approach = torch.where(approach > 0, approach * align_gate, approach)

        _speed = torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=-1)
        _w_back = float(getattr(self.cfg.rew_scales, "backward", 0.0))
        back_pen = torch.zeros(self.num_envs, device=self.device)
        if _w_back:
            back_pen = -_w_back * (-_v_fwd).clamp(min=0.0)
        # OFF-AXIS: charges every bit of speed that is not head-first, so it also catches the
        # SIDEWAYS slide that `backward` is blind to (sideways => v_fwd = 0 => back_pen = 0).
        # |v| - max(0, v_fwd): 0 for pure head-first, |v| for pure sideways or pure tail-first.
        _w_off = float(getattr(self.cfg.rew_scales, "offaxis", 0.0))
        if _w_off:
            back_pen = back_pen - _w_off * (_speed - _v_fwd.clamp(min=0.0))

        # ---- PRE-REACH TIME PENALTY -----------------------------------------------------------
        # A per-step charge that runs from the episode start and STOPS FOREVER once this env
        # reaches its first target. It makes dawdling expensive exactly where the fish is currently
        # stuck (it has never scored), and costs nothing afterwards, so it cannot tax a fish that
        # is already working. Latch first, so the scoring step itself is free.
        self._first_reach_done = self._first_reach_done | _succ
        _w_pre = float(getattr(self.cfg.rew_scales, "prereach_time", 0.0))
        prereach_pen = torch.zeros(self.num_envs, device=self.device)
        if _w_pre:
            prereach_pen = -_w_pre * (~self._first_reach_done).float()

        reward = (progress + heading + approach + success_rew + time_pen + effort_pen
                  + spin_pen + launch_pen + z_pen + back_pen + prereach_pen)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # per-step reward-term printout (play/debug; cfg.print_reward_terms). Stash the tensors and
        # print env 0's values -- the AMP subclass defers the print so its style term lands on the
        # SAME line (it calls _print_reward_terms itself with the extra columns).
        if getattr(self.cfg, "print_reward_terms", False):
            self._reward_terms = {
                "progress": progress, "heading": heading, "approach": approach,
                "success": success_rew, "time": time_pen, "prereach": prereach_pen,
                "effort": effort_pen, "spin": spin_pen, "launch": launch_pen,
                "z_band": z_pen, "back/offax": back_pen, "TASK": reward,
            }
            if not getattr(self, "_defer_reward_print", False):
                self._print_reward_terms()

        self._prev_distance = distance
        self._episode_rewards += reward
        self._best_distance = torch.minimum(self._best_distance, distance)
        # merge (don't overwrite) so the success_rate set in _get_dones survives
        # regardless of reward/dones call order
        # SWIM SPEED in body-lengths/s (real fish cruise 1-4 BL/s). Three views: total speed, the
        # component along the fish's own heading (forward swimming; negative = drifting backwards),
        # and the component along the target direction (closing speed -- what the task actually pays).
        _bl = float(getattr(self, "_body_length", 0.26))
        _v_w = self.robot.data.root_lin_vel_w
        _speed_bl = torch.linalg.norm(_v_w, dim=-1) / _bl
        self.extras.setdefault("log", {}).update({
            "speed_bl": _speed_bl.mean(),
            "speed_fwd_bl": ((_v_w * heading_dir).sum(-1) / _bl).mean(),
            "speed_to_target_bl": ((_v_w * target_dir).sum(-1) / _bl).mean(),
            "speed_bl_p90": torch.quantile(_speed_bl, 0.9),
            # fraction of envs currently travelling TAIL-FIRST. This is the number that says whether
            # the "reverse into the target" habit is gone: it should fall toward ~0.
            "frac_backward": (_v_fwd < 0).float().mean(),
            "align_cos": (heading_dir * target_dir).sum(-1).mean(),   # 1 = head aimed at target
            "reward_backward": back_pen.mean(),
            # HOW the fish travels, independent of WHERE. cos between the velocity and the head
            # direction: +1 = head-first (swimming), 0 = SIDEWAYS slide, -1 = tail-first.
            # Measured 0.34 (=70 deg, sideways) on the frozen policy -- this is the number that
            # tells you a "faster" fish is really just a better-aimed broadside drift.
            "travel_cos": (_v_fwd / _speed.clamp(min=1e-4)).mean(),
            "frac_sideways": ((_v_fwd / _speed.clamp(min=1e-4)).abs() < 0.5).float().mean(),
            # share of envs that have NOT yet scored this episode, i.e. still paying the pre-reach
            # time penalty. Should fall toward 0 as the fish learns to score early.
            "frac_prereach": (~self._first_reach_done).float().mean(),
            "reward_prereach": prereach_pen.mean(),
            "distance_to_target": distance.mean(),
            "best_distance": self._best_distance.mean(),
            "reward_approach": approach.mean(),
            "reward_z_band": z_pen.mean(),
            "z_err_to_target": (self.robot.data.root_state_w[:, 2]
                                - self.target_positions_w[:, 2]).abs().mean(),
            # NOTE: instantaneous in-radius fraction (NOT the success rate -- that is the
            # EPISODIC metric logged in _get_dones). Kept as a gauge only.
            "in_radius_frac": success.float().mean(),
            "reward_mean": reward.mean(),
            "targets_reached": getattr(self, "_targets_reached",
                                       torch.zeros(1, device=self.device)).float().mean(),
            "success_radius_cur": torch.tensor(float(self._cur_radius), device=self.device),
            "target_dist_cur": torch.tensor(float(getattr(self, "_cur_dist_mean",
                                                          self.cfg.target_dist_mean)), device=self.device),
            "reach_rate_gate": torch.tensor(float(getattr(self, "_last_gate_rate", float("nan"))),
                                            device=self.device),
        })
        return reward

    def _print_reward_terms(self, extra=None):
        """One compact line of env 0's reward terms this control step (cfg.print_reward_terms).
        Throttled by cfg.print_reward_every. `extra` lets subclasses append columns (AMP style)."""
        _every = max(1, int(getattr(self.cfg, "print_reward_every", 1)))
        if int(self.common_step_counter) % _every:
            return
        terms = dict(self._reward_terms)
        if extra:
            terms.update(extra)
        line = "  ".join(f"{k}={float(v[0]):+.3f}" for k, v in terms.items())
        print(f"[rew env0 ep_step {int(self.episode_length_buf[0])}] {line}", flush=True)

    def _sample_targets(self, env_ids, base_pos, heading_az=None):
        """Sample a new target for each env in env_ids: EVEN (uniform) bearing + NORMAL distance
        (mean/std from cfg), placed around base_pos (the fish's current position). Clamped positive
        (> success_radius) and kept inside the arena. Writes into self.target_positions_w.

        target_front_cone_deg > 0 narrows the bearing from all-around to +/- that many degrees
        about the fish's CURRENT nose azimuth, so every target spawns IN FRONT of the fish (both at
        reset and at each multi_target resample). heading_az (rad, world) can be passed by callers
        holding a root pose newer than self.robot.data (the reset path writes the pose to sim in the
        same call); when None it is read from the live root quat via body_forward_sign."""
        n = int(len(env_ids))
        # OPTIONAL empirical bearing distribution (e.g. zef_target_angles.npz): bootstrap-sample
        # the raw ZeF theta_deg samples (positive = CCW from heading, matching `bearing` below)
        # instead of the uniform front cone. ""/unset keeps the original behavior bit-for-bit.
        _adist = str(getattr(self.cfg, "target_angle_dist_path", "") or "")
        if _adist:
            if not hasattr(self, "_tgt_angle_samples"):
                _th = np.load(_adist)["theta_deg"].astype(np.float64)
                self._tgt_angle_samples = torch.tensor(np.deg2rad(_th), device=self.device,
                                                       dtype=torch.float32)
                print(f"[SalmonSwim] target bearings from EMPIRICAL distribution {_adist} "
                      f"({len(_th)} samples, [{_th.min():.0f},{_th.max():.0f}] deg)", flush=True)
            if heading_az is None:
                rq = self.robot.data.root_state_w[env_ids, 3:7]
                _fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
                fwd_b = torch.tensor([_fs, 0.0, 0.0], dtype=rq.dtype, device=rq.device).repeat(n, 1)
                fwd_w = math_utils.quat_apply(rq, fwd_b)
                heading_az = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
            idx = torch.randint(0, self._tgt_angle_samples.shape[0], (n,), device=self.device)
            bearing = heading_az.to(self.device) + self._tgt_angle_samples[idx]
            _cone = -1.0                                             # skip the cone branch below
        else:
            _cone = math.radians(float(getattr(self.cfg, "target_front_cone_deg", 0.0)))
        if _cone > 0.0:
            if heading_az is None:
                rq = self.robot.data.root_state_w[env_ids, 3:7]
                _fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
                fwd_b = torch.tensor([_fs, 0.0, 0.0], dtype=rq.dtype, device=rq.device).repeat(n, 1)
                fwd_w = math_utils.quat_apply(rq, fwd_b)
                heading_az = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
            bearing = heading_az.to(self.device) + sample_uniform(-_cone, _cone, (n,), self.device)
        elif not _adist:
            bearing = sample_uniform(0.0, 2.0 * math.pi, (n,), self.device)
        _dmean = float(getattr(self, "_cur_dist_mean", self.cfg.target_dist_mean))   # live (distance curriculum)
        dist = torch.randn(n, device=self.device) * float(self.cfg.target_dist_std) + _dmean
        # HARD ceiling on the sampled distance. This used to be a literal 2.0, which silently capped
        # the distance curriculum no matter how high dist_curriculum_max was set.
        _dmax = float(getattr(self.cfg, "target_dist_clamp_max", 2.0))
        # FLOOR on the sampled distance, so a target never spawns already-inside the success
        # region. The pad used to be a literal 0.2 m -- an ABSOLUTE constant on a task whose
        # other distances are body-length-scaled. On a 6.7 cm fish (scaled radius 0.027 m,
        # scaled mean 0.134 m) that floor is 0.227 m, above the whole sampled distribution:
        # every target collapses onto the floor at 3.4 BL instead of the intended 2.0 BL,
        # while a 31 cm fish is barely affected -- silently making the two incomparable.
        # Now configurable; the 0.2 default keeps every pre-existing task bit-for-bit.
        _dpad = float(getattr(self.cfg, "target_dist_clamp_pad", 0.2))
        dist = dist.clamp(min=float(self._cur_radius) + _dpad, max=_dmax)
        # OPTIONAL empirical ELEVATION distribution (3D heatmap): bootstrap-sample real
        # elevation angles (extracted from the 3D-triangulated ZeF reconstruction's body-frame
        # vertical travel-direction component, dir_up -> arcsin -> degrees; see
        # zef_target_elevation.npz) and decompose `dist` into horizontal + vertical components,
        # so targets sit above/below the fish's current height like a real fish's swim targets
        # do. ""/unset keeps the original 2D (fixed target_height) behavior bit-for-bit.
        _edist = str(getattr(self.cfg, "target_elevation_dist_path", "") or "")
        if _edist:
            if not hasattr(self, "_tgt_elev_samples"):
                _ph = np.load(_edist)["phi_deg"].astype(np.float64)
                self._tgt_elev_samples = torch.tensor(np.deg2rad(_ph), device=self.device,
                                                      dtype=torch.float32)
                print(f"[SalmonSwim] target ELEVATION from EMPIRICAL distribution {_edist} "
                      f"({len(_ph)} samples, [{_ph.min():.0f},{_ph.max():.0f}] deg)", flush=True)
            eidx = torch.randint(0, self._tgt_elev_samples.shape[0], (n,), device=self.device)
            phi = self._tgt_elev_samples[eidx]
            horiz = dist * torch.cos(phi)
            vert = dist * torch.sin(phi)
        else:
            horiz = dist
            vert = torch.zeros_like(dist)
        tgt = base_pos.clone().to(self.device)
        tgt[:, 0] = base_pos[:, 0] + horiz * torch.cos(bearing)
        tgt[:, 1] = base_pos[:, 1] + horiz * torch.sin(bearing)
        tgt[:, 2] = self._target_height + vert
        # keep inside the arena (relative to each env origin)
        lim = float(self.cfg.root_position_limit) * 0.7
        local = tgt[:, 0:2] - self.scene.env_origins[env_ids, 0:2]
        tgt[:, 0:2] = self.scene.env_origins[env_ids, 0:2] + local.clamp(-lim, lim)
        self.target_positions_w[env_ids] = tgt

    # ---- dones ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        root_state = self.robot.data.root_state_w

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        local_pos = root_state[:, 0:3] - self.scene.env_origins
        out_of_bounds = torch.linalg.norm(local_pos, dim=1) > self.cfg.root_position_limit
        joint_blowup = self.joint_vel.abs().max(dim=1).values > self.cfg.joint_velocity_limit
        nonfinite = (~torch.isfinite(root_state).all(dim=1)) | (~torch.isfinite(self.joint_pos).all(dim=1)) \
            | (~torch.isfinite(self.joint_vel).all(dim=1))
        # OPTIONAL FEM soft-body guard: flag a blow-up on nodal-velocity divergence too, so a pure-FEM
        # blow-up that hasn't yet reached the joints is still caught. Folded into `nonfinite` so all
        # downstream uses (terminated / counters / logging) pick it up. Off unless cfg.fem_velocity_limit.
        self._fem_blowup = torch.zeros_like(joint_blowup)
        _fvl = getattr(self.cfg, "fem_velocity_limit", None)
        if _fvl is not None and getattr(self, "_soft_view", None) is not None:
            try:
                _nvel = self._soft_view.get_simulation_mesh_nodal_velocities()      # (num_envs, N, 3)
                _nvmax = torch.nan_to_num(_nvel, nan=float("inf")).abs().flatten(1).amax(dim=1)
                self._fem_blowup = _nvmax > float(_fvl)
                self._nvel_max = float(torch.nan_to_num(_nvmax.max(), posinf=float("inf")))
            except Exception:  # noqa: BLE001
                pass
        nonfinite = nonfinite | self._fem_blowup
        # stash per-step blow-up state for the (every-step) throughput line + distinct-env tracking
        _blew_mask = joint_blowup | nonfinite
        self._blew_now = int(_blew_mask.sum())
        _jvm = self.joint_vel.abs().max()
        self._jvel_max = float(_jvm) if torch.isfinite(_jvm) else float("inf")
        if self._blew_now:
            if not hasattr(self, "_blown_env_ids"):
                self._blown_env_ids = set()
            self._blown_env_ids.update(_blew_mask.nonzero(as_tuple=False).flatten().tolist())
        distance = torch.linalg.norm(self.target_positions_w[:, 0:3] - root_state[:, 0:3], dim=1)   # 3D
        success = distance < self._cur_radius
        self._reached_now = success.clone()   # reach flag for the reward (survives the target moving)

        # ADAPTIVE CURRICULA: accumulate reaches (shared milestone counter for both curricula).
        _radius_cur = getattr(self.cfg, "curriculum", False)
        _dist_cur = getattr(self.cfg, "dist_curriculum", False)
        if _radius_cur or _dist_cur:
            self._reach_total += int(success.sum().item())
        # (a) radius curriculum: SHRINK the success radius one step per milestone (harder to hit).
        if _radius_cur:
            _milestones = self._reach_total // max(1, int(self.cfg.curriculum_reaches_per_shrink))
            self._cur_radius = max(float(self.cfg.curriculum_radius_min),
                                   float(self.cfg.curriculum_radius_start)
                                   - float(self.cfg.curriculum_radius_shrink) * _milestones)
        # (b) DISTANCE curriculum: GROW the target-spawn distance one step per milestone (farther to
        # swim). Start close so reaching is the COMMON case -> the policy grooves the final approach
        # + chaining, then the distance is pushed out only as it keeps earning reaches.
        if _dist_cur:
            if getattr(self.cfg, "dist_curriculum_gated", False):
                # RATE-GATED with DEMOTION: every dist_gate_window control steps, measure the recent
                # reach rate (reaches per episode-equivalent). Promote only above the promote bar;
                # step BACK DOWN below the demote floor so eroding skill re-grooves at an easier level
                # instead of decaying in place (the cumulative-count rule promoted a failing policy).
                self._gate_reaches += int(success.sum().item())
                self._gate_steps += 1
                if self._gate_steps >= max(1, int(self.cfg.dist_gate_window)):
                    rate = (self._gate_reaches * float(self.max_episode_length)
                            / (self._gate_steps * self.num_envs))
                    _g = float(self.cfg.dist_curriculum_grow)
                    if rate >= float(self.cfg.dist_gate_promote):
                        self._cur_dist_mean = min(float(self.cfg.dist_curriculum_max),
                                                  self._cur_dist_mean + _g)
                    elif rate <= float(self.cfg.dist_gate_demote):
                        self._cur_dist_mean = max(float(self.cfg.dist_curriculum_min),
                                                  self._cur_dist_mean - _g)
                    self._last_gate_rate = float(rate)
                    self._gate_reaches = 0
                    self._gate_steps = 0
            else:
                _dm = self._reach_total // max(1, int(self.cfg.dist_curriculum_reaches_per_grow))
                self._cur_dist_mean = min(float(self.cfg.dist_curriculum_max),
                                          float(self.cfg.dist_curriculum_start)
                                          + float(self.cfg.dist_curriculum_grow) * _dm)

        if getattr(self.cfg, "multi_target", False):
            # reaching does NOT end the episode: count it, spawn a NEW target around the fish, and reset
            # that env's prev-distance so the progress reward doesn't spike when the target jumps away.
            if success.any():
                rids = success.nonzero(as_tuple=False).flatten()
                self._targets_reached[rids] += 1.0
                self._sample_targets(rids, root_state[rids, 0:3])
                self._prev_distance[rids] = torch.linalg.norm(
                    self.target_positions_w[rids, 0:3] - root_state[rids, 0:3], dim=1)
            terminated = out_of_bounds | joint_blowup | nonfinite
        else:
            terminated = out_of_bounds | joint_blowup | nonfinite | success
        if self.cfg.step_debug_extras:
            target_delta = self.target_positions_w[:, 0:3] - root_state[:, 0:3]
            self.extras["debug_step"] = {
                "episode_step": self.episode_length_buf.detach().clone(),
                "root_state_w": root_state.detach().clone(),
                "joint_pos": self.joint_pos.detach().clone(),
                "joint_vel": self.joint_vel.detach().clone(),
                "projected_gravity_b": self.robot.data.projected_gravity_b.detach().clone(),
                "target_position_w": self.target_positions_w.detach().clone(),
                "target_delta": target_delta.detach().clone(),
                "distance_to_target": distance.detach().clone(),
                "success": success.detach().clone(),
                "terminated": terminated.detach().clone(),
                "time_out": time_out.detach().clone(),
                "out_of_bounds": out_of_bounds.detach().clone(),
                "joint_blowup": joint_blowup.detach().clone(),
                "nonfinite": nonfinite.detach().clone(),
                "actions": self.actions.detach().clone(),
                "pos_targets": self._pos_targets.detach().clone(),
                "torques": self._torques.detach().clone(),
            }
        # EPISODIC success rate: of the episodes that END this step (terminated or timed
        # out), the fraction that ended by REACHING the target. (The old instantaneous
        # `success.mean()` was ~0 because success also terminates the episode, so an env
        # is in-radius for only one step.) Rolling window over recently-ended episodes.
        done = terminated | time_out
        if done.any():
            self._success_hist.extend(success[done].float().tolist())
        sr = (sum(self._success_hist) / len(self._success_hist)) if self._success_hist else 0.0
        self.extras["success"] = success.detach().clone()
        if not isinstance(self.extras.get("log"), dict):
            self.extras["log"] = {}
        self.extras["log"]["success_rate"] = torch.tensor(float(sr), device=self.device)
        self.extras["log"]["episodes_ended"] = done.float().sum()
        # OVERALL REWARD: the total return of each episode that ENDS this step (sum of per-step
        # rewards over the whole episode), averaged over a rolling window of recent episodes. This
        # is the "is the policy actually getting better" curve -- unlike the per-step reward_mean it
        # is not diluted by episode length, and unlike rl_games' own reward it is env-side, so it
        # stays comparable across runs with different horizon/minibatch settings.
        # REACHES PER EPISODE -- the honest headline number. Read the per-episode counter of the
        # episodes that END this step (this runs BEFORE _reset_idx clears it) and keep a rolling mean.
        # Prefer this over `targets_reached`, which is the TIME-AVERAGE of a counter still filling up
        # (so it reads ~half the true per-episode total), and over `reach_rate_gate`, which is correct
        # but only recomputes every cfg.dist_gate_window steps so it plots as a step function.
        if done.any():
            self._reach_hist.extend(self._targets_reached[done].tolist())
        if self._reach_hist:
            self.extras["log"]["reaches_per_episode"] = torch.tensor(
                sum(self._reach_hist) / len(self._reach_hist), device=self.device)
        if done.any():
            self._return_hist.extend(self._episode_rewards[done].tolist())
        if self._return_hist:
            self.extras["log"]["episode_return"] = torch.tensor(
                sum(self._return_hist) / len(self._return_hist), device=self.device)

        # ALL-ENV blow-up rate (not just env0) -- so we can compare train vs play blow-up frequency
        self._blowup_total = getattr(self, "_blowup_total", 0) + int((joint_blowup | nonfinite).sum())
        self._reset_total = getattr(self, "_reset_total", 0) + int(done.sum())
        self._bstep_total = getattr(self, "_bstep_total", 0) + 1
        if self._bstep_total % 200 == 0:
            envsteps = self._bstep_total * self.num_envs
            print(f"[blowup-rate] {self._blowup_total} all-env blow-ups / {self._reset_total} resets "
                  f"in {self._bstep_total} steps x {self.num_envs} envs "
                  f"({100.0*self._blowup_total/max(1,envsteps):.4f}% of env-steps, "
                  f"{100.0*self._blowup_total/max(1,self._reset_total):.1f}% of resets are blow-ups)", flush=True)

        if self.cfg.debug_print:
            # scan ALL envs; print ONLY when an env blows up (joint vel over limit, or non-finite)
            blew = joint_blowup | nonfinite
            if blew.any():
                epoch = self._bstep_total // max(1, int(self.cfg.video_clip_steps))  # ~rl_games epoch
                for i in blew.nonzero(as_tuple=False).flatten().tolist():
                    if bool(joint_blowup[i]):
                        why = "JOINT_BLOWUP"
                    elif bool(self._fem_blowup[i]):
                        why = "FEM_BLOWUP"
                    else:
                        why = "NONFINITE"
                    jv = float(self.joint_vel[i].abs().max())
                    print(f"ep{epoch:06d} ENV{i:04d} BLOWUP ({why}, jvel={jv:.1f}, "
                          f"step={int(self.episode_length_buf[i])})", flush=True)

        self._record_step()
        self._record_traj_step(bool(done[0]))
        self._print_throughput()
        return terminated, time_out

    # ---- reset ----
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos += sample_uniform(self.cfg.initial_joint_pos_range[0], self.cfg.initial_joint_pos_range[1],
                                    joint_pos.shape, joint_pos.device)
        joint_vel += sample_uniform(self.cfg.initial_joint_vel_range[0], self.cfg.initial_joint_vel_range[1],
                                    joint_vel.shape, joint_vel.device)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.scene.env_origins[env_ids]
        pos_noise = sample_uniform(self.cfg.initial_root_pos_range[0], self.cfg.initial_root_pos_range[1],
                                   (len(env_ids), 2), joint_pos.device)
        root_state[:, 0:2] += pos_noise
        root_state[:, 2] = self.cfg.target_root_height
        root_state[:, 7:13] = 0.0

        yaw_noise = sample_uniform(self.cfg.initial_root_rot_range[0], self.cfg.initial_root_rot_range[1],
                                   (len(env_ids), 1), joint_pos.device).squeeze(-1)
        yaw_quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw_noise), torch.zeros_like(yaw_noise), yaw_noise)
        root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, root_state[:, 3:7])

        # SPAWN-GLIDE: start each episode with a small forward velocity along the fish's heading instead of
        # from REST -- kills the cold-start tax (the fish used to waste much of each episode just spinning
        # up its gait). Applied AFTER the yaw so the glide points where the nose points.
        _glide = float(getattr(self.cfg, "spawn_glide_speed", 0.0))
        if _glide:
            _fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
            _fwd_b = torch.tensor([_fs, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device).repeat(len(env_ids), 1)
            _fwd_w = math_utils.quat_apply(root_state[:, 3:7], _fwd_b)
            root_state[:, 7:10] = _glide * _fwd_w

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # teleport the soft body WITH the bones so it doesn't snap back from where the fish was
        if getattr(self, "_soft_view", None) is not None:
            self._reset_deformable(env_ids, root_state[:, 0:3], root_state[:, 3:7])

        if getattr(self.cfg, "random_target", False):
            # sample a fresh target around the reset spawn (only for these envs; keep others' targets).
            # heading_az from the LOCAL root_state: the pose was just written to sim, so
            # self.robot.data still holds the pre-reset quat and the cone would aim at the OLD nose.
            _fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
            _fwd_b = torch.tensor([_fs, 0.0, 0.0], dtype=root_state.dtype,
                                  device=root_state.device).repeat(len(env_ids), 1)
            _fwd_w = math_utils.quat_apply(root_state[:, 3:7], _fwd_b)
            self._sample_targets(env_ids, root_state[:, 0:3],
                                 heading_az=torch.atan2(_fwd_w[:, 1], _fwd_w[:, 0]))
        else:
            # CROSS-ENV CLOBBER FIX: only the RESETTING envs get their target rewritten. This used to
            # assign the whole (num_envs,3) tensor, ignoring env_ids -- and Isaac Lab calls
            # _reset_idx(reset_env_ids) on ANY step where ANY env is done (direct_rl_env.py:396-398).
            # So with 256 envs, one env timing out snapped EVERY env's target back to the fixed
            # corner. Under multi_target that wiped the curriculum-sampled waypoints after only
            # ~25-70 control steps (1-2 s) against a ~10 s reach interval, so the target the policy
            # actually chased was a reset-rate artefact, not the curriculum.
            self.target_positions_w[env_ids] = self._compute_target_positions_world()[env_ids]
        # ALWAYS clear the PER-EPISODE counters. These used to be reset only inside the random_target
        # branch, so whenever random_target=False `_targets_reached` was never cleared and the logged
        # "targets_reached" was a LIFETIME total that only ever grew -- it read 17.5 after 8 episodes
        # when the true rate was ~2/episode, making training look ~8x better than it was.
        # (NOTE: random_target=False is the cfg DEFAULT but is NOT what every run used -- the runs
        # from 2026-07-30..2026-08-03 passed env.random_target=true on the command line.)
        self._targets_reached[env_ids] = 0.0
        self._reached_now[env_ids] = False
        self._first_reach_done[env_ids] = False   # re-arm the pre-reach time penalty
        self._episode_rewards[env_ids] = 0.0
        self._best_distance[env_ids] = float("inf")
        distance = torch.linalg.norm(
            root_state[:, 0:3] - self.target_positions_w[env_ids, 0:3], dim=1)   # 3D, from the reset spawn
        self._prev_distance[env_ids] = torch.nan_to_num(distance)

    def _setup_deformable_reset(self):
        """Create a DeformablePrim view over all envs + capture the soft body's rest shape in the
        root-local frame, so _reset_idx can rigidly re-place it at each reset env's new pose."""
        soft0 = None
        for prim in self.scene.stage.Traverse():
            p = str(prim.GetPath())
            if "/env_0/" in p and getattr(self.cfg, "deformable_prim_token", "deformable_salmon") in p and \
                    prim.GetAttribute("physxDeformable:simulationPoints").IsValid():
                soft0 = p
                break
        if soft0 is None:
            print("[SalmonSwim] FEM reset: deformable prim not found -> disabled", flush=True)
            return
        expr = soft0.replace("/env_0/", "/env_*/")
        try:
            from isaacsim.core.prims import DeformablePrim
            view = DeformablePrim(prim_paths_expr=expr, reset_xform_properties=False)
            view.initialize()
            nodal = view.get_simulation_mesh_nodal_positions()             # (num_envs, N, 3) world
        except Exception as e:  # noqa: BLE001
            print(f"[SalmonSwim] FEM reset: view init failed ({e}) -> disabled", flush=True)
            return
        # rest shape in root-local frame from env 0 (all envs identical at the default pose)
        root = self.robot.data.root_state_w[0]
        nodal0 = nodal[0].to(self.device).float()                          # (N,3) world
        n = nodal0.shape[0]
        self._soft_rest_local = quat_rotate_inverse(
            root[3:7].unsqueeze(0).expand(n, 4), nodal0 - root[0:3].unsqueeze(0))   # (N,3) local
        self._n_soft_nodes = n
        self._soft_view = view
        print(f"[SalmonSwim] FEM reset: ON ({n} nodes)", flush=True)

    def _reset_deformable(self, env_ids, new_pos, new_quat):
        """Rigidly place the soft body's rest shape at the reset envs' new root pose + zero its
        nodal velocities. new_pos (M,3), new_quat (M,4) aligned with env_ids."""
        try:
            m, n = new_pos.shape[0], self._n_soft_nodes
            q = new_quat.unsqueeze(1).expand(m, n, 4).reshape(m * n, 4)
            rl = self._soft_rest_local.unsqueeze(0).expand(m, n, 3).reshape(m * n, 3)
            nodal_new = quat_rotate(q, rl).reshape(m, n, 3) + new_pos.unsqueeze(1)   # (M,N,3) world
            pos = self._soft_view.get_simulation_mesh_nodal_positions()
            vel = self._soft_view.get_simulation_mesh_nodal_velocities()
            pos[env_ids] = nodal_new.to(pos.dtype)
            vel[env_ids] = 0.0
            self._soft_view.set_simulation_mesh_nodal_positions(pos)
            self._soft_view.set_simulation_mesh_nodal_velocities(vel)
        except Exception as e:  # noqa: BLE001
            print(f"[SalmonSwim] FEM reset step failed ({e}); disabling", flush=True)
            self._soft_view = None

    def _compute_target_positions_world(self) -> torch.Tensor:
        target_positions = self.scene.env_origins.clone()
        target_positions[:, 0:2] += self._target_offset_xy
        target_positions[:, 2] = self._target_height
        return target_positions
