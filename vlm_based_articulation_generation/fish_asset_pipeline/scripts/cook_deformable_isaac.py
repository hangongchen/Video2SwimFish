#!/usr/bin/env python
"""Post-process a generated fish USD so its FEM deformable body is SIMULATION-READY in Isaac Sim.

WHY THIS STEP EXISTS
--------------------
`add_isaac_physics_to_usd.py` runs in a plain-`pxr` (Blender / usd-core) environment that has no
`PhysxSchema`, so it authors the deformable's PhysX parameters as bare `custom` attributes and fakes
the applied-schema token. Isaac Sim then does NOT honour `simulationHexahedralResolution`, and the
FEM body cooks to a degenerate ~8-node bounding box (a single hex cell) instead of a real volumetric
mesh -- so the "deformable" fish deforms like a rigid block.

This step re-opens the USD inside Isaac Sim (which HAS `omni.physx` / `PhysxSchema`) and re-authors
the deformable the canonical way via `deformableUtils.add_physx_deformable_body(...,
simulation_hexahedral_resolution=N)`, so PhysX cooks a proper ~150-node tetra/hex mesh at load (the
same way the reference salmon asset does). It also:
  * creates + binds a real `PhysxDeformableBodyMaterialAPI` (so youngsModulus / elasticityDamping are
    honoured instead of being silent no-ops),
  * right-hands a reflected (negative-scale) deformable xform that Isaac's DeformablePrim rejects,
so the shipped asset needs no per-consumer fixups.

Run under the Isaac Lab python (NOT Blender's):
  <env_isaaclab>/bin/python cook_deformable_isaac.py --in fish_articulated.usd --out fish_articulated.usd
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--in", dest="in_usd", required=True, help="input fish USD (from add_isaac_physics_to_usd)")
_ap.add_argument("--out", dest="out_usd", default=None, help="output USD (default: overwrite input)")
_ap.add_argument("--mesh-token", default="final_mesh", help="substring identifying the deformable mesh prim")
_ap.add_argument("--hex-resolution", type=int, default=10,
                 help="simulation voxel resolution along the longest axis (10 ~ salmon's ~150 nodes)")
# FEM material + solver params: default to the analytic-water salmon CURE (CFL-stable at dt=1/960)
_ap.add_argument("--youngs-modulus", type=float, default=1.0e5)
_ap.add_argument("--poissons-ratio", type=float, default=0.45)
_ap.add_argument("--elasticity-damping", type=float, default=0.05)
_ap.add_argument("--damping-scale", type=float, default=1.0)
_ap.add_argument("--solver-iters", type=int, default=96)
_ap.add_argument("--vertex-velocity-damping", type=float, default=4.0)
_ap.add_argument("--sleep-damping", type=float, default=10.0)
_ap.add_argument("--settling-threshold", type=float, default=0.1)
AppLauncher.add_app_launcher_args(_ap)
args = _ap.parse_args()

# headless Kit boot (loads omni.physx / PhysxSchema)
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402

from pxr import Gf, Usd, UsdShade, Vt  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402


def find_deformable_mesh(stage, token):
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and token in prim.GetName():
            if "PhysxDeformableBodyAPI" in prim.GetAppliedSchemas() \
                    or prim.GetAttribute("physxDeformable:simulationHexahedralResolution").IsValid():
                return prim
    for prim in stage.Traverse():  # fallback: any mesh with the token
        if prim.GetTypeName() == "Mesh" and token in prim.GetName():
            return prim
    # NAME-AGNOSTIC fallback: any mesh carrying the deformable schema/attrs, regardless of name.
    # Meshy exports name the body 'mesh_001' (not 'final_mesh'), and the pipeline authors the bare
    # deformable API on it -- without this the cook silently no-ops and the FEM stays a degenerate
    # ~8-node box (which then breaks per-slice hydro + the AMP bend feature downstream).
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and (
            "PhysxDeformableBodyAPI" in prim.GetAppliedSchemas()
            or prim.GetAttribute("physxDeformable:simulationHexahedralResolution").IsValid()
            or prim.GetAttribute("physxDeformable:simulationRestPoints").IsValid()
        ):
            print(f"[cook] token '{token}' not matched; found deformable mesh by schema: {prim.GetName()}")
            return prim
    return None


def right_hand_reflected_xform(prim):
    """Flip a negative-scale (reflected, det<0) deformable xform to a right-handed frame with the
    SAME world geometry (positive scale + negated points + reversed winding). Isaac's DeformablePrim
    rejects a left-handed frame ("Non-positive determinant")."""
    node = prim
    fixed = False
    while node and node.GetPath() != node.GetStage().GetPseudoRoot().GetPath():
        sa = node.GetAttribute("xformOp:scale")
        if sa and sa.IsValid() and sa.Get() is not None:
            s = sa.Get()
            if s[0] * s[1] * s[2] < 0:
                sa.Set(Gf.Vec3f(abs(s[0]), abs(s[1]), abs(s[2])))
                fixed = True
                break
        node = node.GetParent()
    if not fixed:
        return False
    pa = prim.GetAttribute("points")
    if pa and pa.IsValid() and pa.Get() is not None:
        pa.Set(Vt.Vec3fArray([Gf.Vec3f(-q[0], -q[1], -q[2]) for q in pa.Get()]))
    fvi_a = prim.GetAttribute("faceVertexIndices")
    fvc_a = prim.GetAttribute("faceVertexCounts")
    if all(a and a.IsValid() and a.Get() is not None for a in (fvi_a, fvc_a)):
        fvi = list(fvi_a.Get()); fvc = list(fvc_a.Get()); out = []; k = 0
        for c in fvc:
            out.extend(fvi[k:k + c][::-1]); k += c
        from pxr import Vt as _Vt
        fvi_a.Set(_Vt.IntArray(out))
    return True


def main():
    out_usd = args.out_usd or args.in_usd
    stage = Usd.Stage.Open(args.in_usd)
    mesh = find_deformable_mesh(stage, args.mesh_token)
    if mesh is None:
        raise SystemExit(f"[cook] no deformable mesh matching '{args.mesh_token}' in {args.in_usd}")
    mesh_path = mesh.GetPath()
    print(f"[cook] deformable mesh: {mesh_path}")

    # 1) right-hand a reflected xform (else the FEM view can't initialise)
    if right_hand_reflected_xform(mesh):
        print("[cook] right-handed a reflected (negative-scale) deformable xform")

    # 2) strip the pipeline's bare `custom` physxDeformable attrs so the real schema wins cleanly
    removed = 0
    for a in list(mesh.GetAttributes()):
        n = a.GetName()
        if n.startswith("physxDeformable:") or n.startswith("physxCollision:"):
            mesh.RemoveProperty(n); removed += 1
    print(f"[cook] stripped {removed} bare custom physxDeformable/physxCollision attrs")

    # 3) re-author the deformable via the proper PhysX schema so the cook honours the resolution
    ok = deformableUtils.add_physx_deformable_body(
        stage, mesh_path,
        collision_simplification=True,
        simulation_hexahedral_resolution=int(args.hex_resolution),
        solver_position_iteration_count=int(args.solver_iters),
        vertex_velocity_damping=float(args.vertex_velocity_damping),
        sleep_damping=float(args.sleep_damping),
        settling_threshold=float(args.settling_threshold),
        self_collision=False,
    )
    print(f"[cook] add_physx_deformable_body(hex_res={args.hex_resolution}) ok={ok}")

    # 4) create + bind a real FEM material (youngsModulus / elasticityDamping honoured, not no-op'd)
    mat_path = mesh_path.AppendChild("deformableBodyMaterial")
    deformableUtils.add_deformable_body_material(
        stage, mat_path,
        youngs_modulus=float(args.youngs_modulus),
        poissons_ratio=float(args.poissons_ratio),
        elasticity_damping=float(args.elasticity_damping),
        damping_scale=float(args.damping_scale),
    )
    physicsUtils.add_physics_material_to_prim(stage, mesh.GetPrim(), mat_path)
    print(f"[cook] created + bound FEM material at {mat_path} (youngs={args.youngs_modulus})")

    stage.GetRootLayer().Export(out_usd)
    print(f"[cook] wrote sim-ready USD -> {out_usd}")


_cook_ok = False
try:
    main()
    _cook_ok = True
finally:
    # flush BEFORE the hard exit or the buffered [cook] logs are lost. simulation_app.close()
    # hangs on this box (Blackwell + FEM); the USD is already exported, so force-exit instead
    # of waiting on Kit teardown. Exit NON-ZERO if the cook failed (else a mesh-not-found /
    # SystemExit is masked as success and the caller ships a degenerate 8-node FEM).
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if _cook_ok else 1)
