"""Extract the fish body's real 3D centerline (head-to-tail midline curve) by slicing the
skin mesh into thin cross-sections along its length axis and taking each slice's centroid.

Why this exists: the ORIGINAL actor-critic system (video2swimfish/articulation/) had the
actor output a full 3D position [x, y, z] for every bone. The fine-tuned actor's biggest
failure mode (confirmed on catfish_fish001/006 with real renders + logged positions) was
NOT bone shape/size -- it was position: the model would place the whole bone chain along the
WRONG axis, or have it collapse geometrically toward a single point instead of spanning the
body. Both are 3D-placement mistakes. Precomputing the real centerline and having the actor
choose only a single scalar `centerline_t` (0=head, 1=tail) removes 2 of the 3 position
degrees of freedom entirely -- the bone's (y, z) are ALWAYS the true body-centroid at that
point along x, looked up from this table, never guessed by the VLM.

Convention (matches the canonical frame every fish mesh is already put into by
generate_auto_skeleton_blend.align_target_mesh_like_supervised): length along X, head at +X.
So t=0 -> x=x_max (head), t=1 -> x=x_min (tail), monotonically increasing toward the tail.

Run under Blender (single-fish quick check):
    blender -b <fish.blend> --python extract_centerline.py -- --out centerline.json [--n 101]
Or call compute_centerline(skin_obj, n_samples) directly from another Blender-side script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2sf_paths as P  # noqa: E402
PIPE_SCRIPTS = P.PIPE_SCRIPTS
sys.path.insert(0, str(PIPE_SCRIPTS))
import generate_auto_skeleton_blend as gask  # noqa: E402


def compute_centerline(skin_obj, n_samples: int = 101) -> list[dict]:
    """Returns n_samples points, evenly spaced in t in [[0, 1]] (t=0 head/+X, t=1 tail/-X),
    each {"t": t, "x": .., "y": .., "z": ..}. y/z are the cross-section centroid at that x
    (widened progressively if a slice is too thin to find >=3 vertices, same pattern as
    verify_stage1.py's midline_tangent_at, generalized into a full sampled table here)."""
    verts = gask.get_world_verts(skin_obj) if hasattr(gask, "get_world_verts") else _world_verts(skin_obj)
    xs = [v.x for v in verts]
    x_min, x_max = min(xs), max(xs)
    length = x_max - x_min
    base_window = max(length * 0.01, 1e-5)

    samples = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        x_query = x_max - t * length
        window = base_window
        pts = [v for v in verts if abs(v.x - x_query) < window]
        widen_iters = 0
        while len(pts) < 3 and widen_iters < 10:
            window *= 1.7
            pts = [v for v in verts if abs(v.x - x_query) < window]
            widen_iters += 1
        cy = sum(p.y for p in pts) / len(pts) if pts else 0.0
        cz = sum(p.z for p in pts) / len(pts) if pts else 0.0
        samples.append({"t": round(t, 6), "x": round(x_query, 6), "y": round(cy, 6), "z": round(cz, 6)})
    return samples


def _world_verts(obj):
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(depsgraph)
    me = eo.to_mesh()
    verts = [obj.matrix_world @ v.co for v in me.vertices]
    eo.to_mesh_clear()
    return verts


def lookup_xyz(centerline: list[dict], t: float) -> tuple[float, float, float]:
    """Linear interpolation of the centerline table at an arbitrary t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    n = len(centerline)
    pos = t * (n - 1)
    i0 = int(pos)
    i1 = min(i0 + 1, n - 1)
    frac = pos - i0
    a, b = centerline[i0], centerline[i1]
    x = a["x"] + frac * (b["x"] - a["x"])
    y = a["y"] + frac * (b["y"] - a["y"])
    z = a["z"] + frac * (b["z"] - a["z"])
    return x, y, z


def x_to_t(centerline: list[dict], x: float) -> float:
    """Inverse of lookup_xyz's x-coordinate: given a real (possibly off-midline) bone's own
    x-position, find the centerline t whose x matches -- i.e. PROJECT that bone onto the
    centerline at the same point along the body length, approximating its true (x,y,z) by
    (x, y_mid(t), z_mid(t)). x is monotonic decreasing in t (t=0 head/x_max, t=1 tail/x_min)
    by construction, so this is a simple monotonic search, no root-finding needed."""
    xs = [c["x"] for c in centerline]  # decreasing
    n = len(centerline)
    if x >= xs[0]:
        return 0.0
    if x <= xs[-1]:
        return 1.0
    for i in range(n - 1):
        if xs[i] >= x >= xs[i + 1]:
            span = xs[i] - xs[i + 1]
            frac = (xs[i] - x) / span if span > 1e-12 else 0.0
            t_i, t_i1 = centerline[i]["t"], centerline[i + 1]["t"]
            return t_i + frac * (t_i1 - t_i)
    return 0.5


def _blender_main():
    import bpy

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else "centerline.json"
    n_samples = int(argv[argv.index("--n") + 1]) if "--n" in argv else 101

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    skin = max(mesh_objs, key=lambda o: gask.world_bbox_for_objects([o])["size"].x
                                        * gask.world_bbox_for_objects([o])["size"].y
                                        * gask.world_bbox_for_objects([o])["size"].z)
    centerline = compute_centerline(skin, n_samples)
    Path(out_path).write_text(json.dumps(centerline, indent=2))
    print(f"[extract_centerline] {len(centerline)} samples -> {out_path}", flush=True)


if __name__ == "__main__":
    _blender_main()
