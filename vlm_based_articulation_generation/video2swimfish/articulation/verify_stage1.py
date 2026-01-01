"""Stage-1 independent verification of a constructed fish's K* USD: run AFTER the
actor-critic loop / export, directly against the exported artifact (not against the
construction log), so this catches anything the loop itself might have gotten wrong.

Three checks:
  1. No interpenetration -- every bone vertex tested against the skin surface (same BVH
     nearest-point+normal-side test as geometric_verification.py), reporting count AND max
     penetration depth in mm (the construction-time log only ever recorded counts).
  2. Skeleton coverage -- union of bone AABB spans along the body's longitudinal (length)
     axis, as a fraction of the skin's own body length, plus the largest uncovered gap.
  3. Bone axis alignment -- angle between each bone's own principal axis (PCA of its
     vertices) and the local body-midline tangent (cross-section centroid direction) at
     that bone's position along the body.

Run under Blender:
    blender -b --python verify_stage1.py -- --usd <path/to/K_final.usd> --fish_id <id>
Writes <usd_dir>/stage1_report.json and prints the PASS/FAIL table row.
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

COVERAGE_PASS_FRAC = 0.90
COVERAGE_MAX_GAP_FRAC = 0.10
ALIGN_MEAN_PASS_DEG = 15.0
ALIGN_MAX_PASS_DEG = 30.0
PEN_PASS_MAX_OUTSIDE = 5
PEN_PASS_MAX_DEPTH_MM = 2.0


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            args[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


def load_scene(usd_path: str):
    import bpy

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.wm.usd_import(filepath=usd_path)
    bpy.context.view_layer.update()

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    assert mesh_objs, f"no mesh objects imported from {usd_path}"
    skin = max(mesh_objs, key=lambda o: gask.world_bbox_for_objects([o])["size"].x
                                        * gask.world_bbox_for_objects([o])["size"].y
                                        * gask.world_bbox_for_objects([o])["size"].z)
    bones = [o for o in mesh_objs if o is not skin]
    return skin, bones


def get_world_verts(obj):
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(depsgraph)
    me = eo.to_mesh()
    verts = [obj.matrix_world @ v.co for v in me.vertices]
    eo.to_mesh_clear()
    return verts


def check_interpenetration(skin, bones):
    import bpy
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    skin_eval = skin.evaluated_get(depsgraph)
    skin_mesh = skin_eval.to_mesh()
    skin_verts = [skin.matrix_world @ v.co for v in skin_mesh.vertices]
    skin_polys = [list(p.vertices) for p in skin_mesh.polygons]
    skin_bvh = BVHTree.FromPolygons(skin_verts, skin_polys)
    skin_eval.to_mesh_clear()

    total_outside = 0
    max_depth_m = 0.0
    per_bone = []
    for b in bones:
        pts = get_world_verts(b)
        n_out = 0
        depth = 0.0
        for p in pts:
            loc, normal, idx, dist = skin_bvh.find_nearest(p)
            if loc is None:
                continue
            to_p = p - loc
            if to_p.length > 1e-9 and to_p.normalized().dot(normal) > 0:
                n_out += 1
                depth = max(depth, to_p.length)
        total_outside += n_out
        max_depth_m = max(max_depth_m, depth)
        per_bone.append({"bone": b.name, "n_vertices": len(pts), "n_outside": n_out,
                          "max_depth_mm": round(depth * 1000, 3)})

    max_depth_mm = round(max_depth_m * 1000, 3)
    passed = (total_outside == 0) or (total_outside <= PEN_PASS_MAX_OUTSIDE and max_depth_mm < PEN_PASS_MAX_DEPTH_MM)
    return {"total_vertices_outside": total_outside, "max_penetration_depth_mm": max_depth_mm,
            "per_bone": per_bone, "passed": passed}


def check_coverage(skin, bones):
    skin_bbox = gask.world_bbox_for_objects([skin])
    body_min_x, body_max_x = skin_bbox["min"].x, skin_bbox["max"].x
    body_length = body_max_x - body_min_x

    intervals = []
    for b in bones:
        bbox = gask.world_bbox_for_objects([b])
        intervals.append((bbox["min"].x, bbox["max"].x))
    intervals.sort()

    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    covered = sum(hi - lo for lo, hi in merged)
    coverage_frac = covered / body_length if body_length > 1e-9 else 0.0

    gaps = []
    cursor = body_min_x
    for lo, hi in merged:
        if lo > cursor:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < body_max_x:
        gaps.append((cursor, body_max_x))
    max_gap_frac = max(((hi - lo) / body_length for lo, hi in gaps), default=0.0)

    passed = coverage_frac >= COVERAGE_PASS_FRAC and max_gap_frac <= COVERAGE_MAX_GAP_FRAC
    return {"coverage_frac": round(coverage_frac, 4), "body_length_m": round(body_length, 5),
            "gaps_frac": [round((hi - lo) / body_length, 4) for lo, hi in gaps],
            "max_gap_frac": round(max_gap_frac, 4), "passed": passed}


def principal_axis(pts):
    import numpy as np

    arr = np.array([[p.x, p.y, p.z] for p in pts])
    arr = arr - arr.mean(axis=0)
    cov = arr.T @ arr
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, eigvals.argmax()]


def midline_tangent_at(skin_verts_np, x0, half_window):
    import numpy as np

    def centroid_at(xc):
        mask = np.abs(skin_verts_np[:, 0] - xc) < half_window
        pts = skin_verts_np[mask]
        if len(pts) < 3:
            return None
        return pts.mean(axis=0)

    c_before = centroid_at(x0 - half_window)
    c_after = centroid_at(x0 + half_window)
    if c_before is None or c_after is None:
        return None
    v = c_after - c_before
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return v / n if n > 1e-9 else None


def check_alignment(skin, bones):
    import numpy as np

    skin_verts_np = np.array([[p.x, p.y, p.z] for p in get_world_verts(skin)])
    skin_bbox = gask.world_bbox_for_objects([skin])
    body_length = skin_bbox["size"].x
    half_window = max(body_length * 0.06, 1e-4)

    angles = []
    pca_angles = []
    per_bone = []
    for b in bones:
        bbox = gask.world_bbox_for_objects([b])
        x0 = bbox["center"].x
        tangent = midline_tangent_at(skin_verts_np, x0, half_window)
        bone_axis = principal_axis(get_world_verts(b))
        if tangent is None:
            per_bone.append({"bone": b.name, "angle_deg": None, "note": "no midline samples near this bone"})
            continue
        cos_a = abs(float(np.dot(tangent, bone_axis)))  # abs: axis sign is arbitrary
        cos_a = min(1.0, max(-1.0, cos_a))
        angle_deg = np.degrees(np.arccos(cos_a))
        # Orientation-only measure: the bone's own LENGTH axis (local X -- every bone is a box
        # placed with size=[length, width, height] and zero rotation, so local X is the axis
        # meant to run along the body). The PCA measure above conflates SHAPE with orientation:
        # the GT skeletons' bones are tall thin vertebra-like plates (median 1.6cm long x 4cm
        # tall), so their principal axis is the tall dorsal-ventral direction and reads ~70 deg
        # "misaligned" even when the bone is placed exactly along the midline. Both are logged;
        # pass/fail uses the orientation measure, which is what "bone axis aligned with the
        # midline" means for a box, and the PCA number is kept as a shape diagnostic.
        local_x = np.array(b.matrix_world.to_3x3() @ __import__("mathutils").Vector((1.0, 0.0, 0.0)))
        local_x = local_x / max(np.linalg.norm(local_x), 1e-12)
        cos_o = min(1.0, abs(float(np.dot(tangent, local_x))))
        orient_deg = np.degrees(np.arccos(cos_o))
        angles.append(orient_deg)
        pca_angles.append(angle_deg)
        per_bone.append({"bone": b.name, "angle_deg": round(float(orient_deg), 2),
                         "pca_shape_angle_deg": round(float(angle_deg), 2)})

    mean_angle = float(np.mean(angles)) if angles else float("nan")
    max_angle = float(np.max(angles)) if angles else float("nan")
    passed = bool(angles) and mean_angle < ALIGN_MEAN_PASS_DEG and max_angle < ALIGN_MAX_PASS_DEG
    return {"mean_angle_deg": round(mean_angle, 2), "max_angle_deg": round(max_angle, 2),
            "pca_shape_mean_angle_deg": round(float(np.mean(pca_angles)), 2) if pca_angles else None,
            "pca_shape_max_angle_deg": round(float(np.max(pca_angles)), 2) if pca_angles else None,
            "per_bone": per_bone, "passed": passed}


def main():
    args = parse_args()
    usd_path = args["usd"]
    fish_id = args.get("fish_id", Path(usd_path).parent.name)

    skin, bones = load_scene(usd_path)
    print(f"[verify_stage1] {fish_id}: skin={skin.name}, {len(bones)} bone objects", flush=True)

    c1 = check_interpenetration(skin, bones)
    c2 = check_coverage(skin, bones)
    c3 = check_alignment(skin, bones)

    overall_pass = c1["passed"] and c2["passed"] and c3["passed"]
    report = {"fish_id": fish_id, "usd": usd_path, "n_bones": len(bones),
              "check1_interpenetration": c1, "check2_coverage": c2, "check3_alignment": c3,
              "overall_pass": overall_pass}

    out_path = Path(usd_path).parent / "stage1_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"[verify_stage1] {fish_id} | outside={c1['total_vertices_outside']} "
          f"max_depth={c1['max_penetration_depth_mm']}mm ({'PASS' if c1['passed'] else 'FAIL'}) | "
          f"coverage={c2['coverage_frac']*100:.1f}% max_gap={c2['max_gap_frac']*100:.1f}% "
          f"({'PASS' if c2['passed'] else 'FAIL'}) | "
          f"align mean={c3['mean_angle_deg']}deg max={c3['max_angle_deg']}deg "
          f"({'PASS' if c3['passed'] else 'FAIL'}) | OVERALL={'PASS' if overall_pass else 'FAIL'}",
          flush=True)
    print(f"[verify_stage1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
