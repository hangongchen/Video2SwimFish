"""Precompute a STRIP-THEORY PANEL hydro proxy for the misty fish DIRECTLY from the high-res .obj mesh.

Unlike precompute_misty_hydro.py (which fits ONE ellipsoid per bone to the FEM slice), this uses the
actual Meshy surface geometry as the water-force source:
  * read the fish's render mesh (mesh_001 = the .obj aligned+scaled into the fish frame, 71,354 verts),
  * aggregate its ~140k triangles into ~N_PANELS flat "super-panels" by a voxel grid (area-weighted
    centroid + normal per voxel) -- dependency-free decimation that preserves total area and shape,
  * pin each panel RIGIDLY to its nearest bone, storing the panel centroid-offset + outward normal +
    area in that bone's LOCAL frame.
At runtime the env recovers each panel's world pose/velocity from the bone (v = v_bone + w x r) and
applies flat-plate drag, summed per bone -> the .obj fins/taper literally shape the thrust.

Also saves per-vertex skinning (vert_local + vert_bone) + faces so the training can render the deformed
proxy overlaid on the live fish.

Output: Video2SwimFish/outputs/catfish_fish001/panel_hydro.npz
(catfish_fish001 copy of precompute_misty_panels.py: only GEN/USD/TOKEN defaults differ)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--voxel", type=float, default=0.010, help="voxel size (m) for panel aggregation")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
# re-parse to honor --voxel if passed
import sys as _sys
_extra = [a for a in _sys.argv[1:] if a.startswith("--voxel")]
if _extra:
    args, _ = parser.parse_known_args(["--headless"] + _extra)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
from pathlib import Path
_os_root = Path(__file__).resolve().parents[1]

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

import omni.usd

sys.path.insert(0, str(_os_root / "source" / "FISH"))
from FISH.tasks.direct.fish.salmon_swim_env import _prepare_env_assets  # noqa: E402

DEVICE = "cuda:0"
DT = 1.0 / 240.0
import os as _os
GEN = _os.environ.get("FISH_GEN_DIR",
    str(_os_root / "data" / "fish_assets" / "catfish_fish001")) + "/"
USD = GEN + "K_final.usd"
OUT = GEN + "panel_hydro.npz"
TOKEN = _os.environ.get("FISH_MESH_TOKEN", "Mesh_0")   # substring of the SKIN mesh prim path
VOXEL = float(getattr(args, "voxel", 0.010))


def quat_to_R(q):
    """q wxyz (N,4) -> R (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, gravity=(0.0, 0.0, 0.0), device=DEVICE))
cfg = ArticulationCfg(
    prim_path="/World/Fish/skeleton",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16,
            solver_velocity_iteration_count=2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=25.0, damping=20.0)})
robot = Articulation(cfg)

stage = omni.usd.get_context().get_stage()
_prepare_env_assets(stage, "/World/Fish/skeleton", control_mode="position", keep_deformable=True,
                    deformable_token=TOKEN, fix_multi_articulation_root=True,
                    fix_reflected_deformable=True, keep_bone_colliders=False,
                    fem_youngs_modulus=1.0e5, fem_elasticity_damping=0.05, fem_damping_scale=1.0)

mesh_path = None
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.startswith("/World/Fish") and TOKEN in p and prim.GetTypeName() == "Mesh":
        mesh_path = p
        break
assert mesh_path is not None, f"skin mesh with token {TOKEN!r} not found"

sim.reset()
for _ in range(3):
    robot.write_data_to_sim(); sim.step(render=False); robot.update(DT)

# ---- bone rest poses (world) ----
P = robot.data.body_link_pos_w[0].cpu().numpy().astype(np.float64)     # (B,3)
Q = robot.data.body_link_quat_w[0].cpu().numpy().astype(np.float64)    # (B,4) wxyz
B = P.shape[0]
Rb = quat_to_R(Q)                                                      # (B,3,3) bone rest rotations
RbT = np.transpose(Rb, (0, 2, 1))                                      # world->bone-local

# ---- render mesh_001 world vertices + triangles ----
from pxr import UsdGeom, Usd  # noqa: E402
mesh_prim = stage.GetPrimAtPath(mesh_path)
mg = UsdGeom.Mesh(mesh_prim)
pts = np.array(mg.GetPointsAttr().Get(), dtype=np.float64)             # (M,3) local
fvi = np.array(mg.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
fvc = np.array(mg.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
M2W = np.array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh_prim), dtype=np.float64)
Vh = np.c_[pts, np.ones(len(pts))] @ M2W                              # USD row-vector convention
V = (Vh[:, :3] / Vh[:, 3:4])                                          # (M,3) world verts
M = V.shape[0]

# triangulate (fan)
tris = []
i = 0
for c in fvc:
    idx = fvi[i:i + c]
    for k in range(1, c - 1):
        tris.append((idx[0], idx[k], idx[k + 1]))
    i += c
tris = np.asarray(tris, dtype=np.int64)                               # (T,3)

# ---- per-triangle centroid / area / outward normal ----
v0, v1, v2 = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
cross = np.cross(v1 - v0, v2 - v0)
area2 = np.linalg.norm(cross, axis=1)
tri_area = 0.5 * area2
tri_cent = (v0 + v1 + v2) / 3.0
tri_norm = cross / np.clip(area2[:, None], 1e-12, None)               # unit outward (CCW winding)
valid = tri_area > 1e-10
tri_area, tri_cent, tri_norm = tri_area[valid], tri_cent[valid], tri_norm[valid]

# ---- voxel-grid aggregation into super-panels (area-weighted) ----
key = np.floor(tri_cent / VOXEL).astype(np.int64)
uniq, inv = np.unique(key, axis=0, return_inverse=True)
Pn = uniq.shape[0]
w = tri_area
panel_area = np.zeros(Pn)
panel_cent = np.zeros((Pn, 3))
panel_norm = np.zeros((Pn, 3))
np.add.at(panel_area, inv, w)
np.add.at(panel_cent, inv, w[:, None] * tri_cent)
np.add.at(panel_norm, inv, w[:, None] * tri_norm)
panel_cent /= np.clip(panel_area[:, None], 1e-12, None)
panel_norm /= np.clip(np.linalg.norm(panel_norm, axis=1, keepdims=True), 1e-12, None)

# ---- assign each panel to nearest bone; store centroid-offset + normal in bone-local frame ----
d = np.linalg.norm(panel_cent[:, None, :] - P[None, :, :], axis=2)    # (Pn,B)
panel_bone = d.argmin(axis=1)
r_local = np.einsum("pij,pj->pi", RbT[panel_bone], panel_cent - P[panel_bone])
n_local = np.einsum("pij,pj->pi", RbT[panel_bone], panel_norm)

# ---- per-vertex skinning (nearest bone) for the overlay render ----
dv = np.linalg.norm(V[:, None, :] - P[None, :, :], axis=2)            # (M,B)
vert_bone = dv.argmin(axis=1)
vert_local = np.einsum("pij,pj->pi", RbT[vert_bone], V - P[vert_bone])

per_bone_counts = np.bincount(panel_bone, minlength=B)
print(f"[panels] {B} bones, {M} verts, {tris.shape[0]} tris -> {Pn} panels (voxel={VOXEL} m)", flush=True)
print(f"[panels] total surface area = {panel_area.sum():.4f} m^2", flush=True)
print(f"[panels] panels per bone: {per_bone_counts.tolist()}", flush=True)
print(f"[panels] area per bone (m^2): {[round(float(panel_area[panel_bone==b].sum()),4) for b in range(B)]}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(
    OUT,
    panel_bone=panel_bone.astype(np.int64),
    panel_r_local=r_local.astype(np.float32),
    panel_n_local=n_local.astype(np.float32),
    panel_area=panel_area.astype(np.float32),
    vert_local=vert_local.astype(np.float32),
    vert_bone=vert_bone.astype(np.int64),
    faces=tris.astype(np.int64),
    bone_rest_pos=P.astype(np.float32),
    bone_rest_quat=Q.astype(np.float32),
    num_bones=B,
    voxel=VOXEL,
    bone_names=np.array(list(robot.data.body_names)),  # runtime (PhysX) link order, for calibrate_* scripts
)
print(f"[panels] bone_names (runtime order): {list(robot.data.body_names)}", flush=True)
print(f"[panels] saved {OUT}", flush=True)
print("PANELS_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
