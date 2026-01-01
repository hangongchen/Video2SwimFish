"""Geometric verification (paper Section 3.2: "This is a hard constraint, not optional" --
run after EVERY actor modification, before the critic ever sees the result).

Containment test = generate_auto_skeleton_blend.py's proven nearest-point + normal-side BVH
check. The SHRINK step, however, is implemented here rather than reusing
gask.shrink_bone_to_fit_skin, because that function scales the bone UNIFORMLY on all three
axes (`bone_obj.scale *= 0.9`). Measured consequence on catfish_fish006: a bone that only
poked out of the skin along its tall (dorsal-ventral) axis was shrunk 19 times -> 0.9^19 =
13.5% of its proposed size in EVERY dimension, including its along-body length, which is what
turned the actor's 1.6cm x 4cm bones into sub-pixel dots. The shrink here is ANISOTROPIC:
each step tries scaling one axis at a time and keeps whichever single-axis shrink removes the
most outside vertices, so a too-tall bone gets shorter but keeps its length.

Only importable under Blender (bpy) -- invoked from the Blender-side apply-action scripts,
never from the plain-Python orchestrator.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2sf_paths as P  # noqa: E402
PIPE_SCRIPTS = P.PIPE_SCRIPTS
if str(PIPE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PIPE_SCRIPTS))

import generate_auto_skeleton_blend as gask  # noqa: E402

AXES = (0, 1, 2)  # bone.scale components; bones are placed with rotation_euler=0 so local==world axes


def build_skin_bvh(skin_obj):
    import bpy
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    skin_eval = skin_obj.evaluated_get(depsgraph)
    skin_mesh = skin_eval.to_mesh()
    skin_verts = [skin_obj.matrix_world @ v.co for v in skin_mesh.vertices]
    skin_polys = [list(p.vertices) for p in skin_mesh.polygons]
    bvh = BVHTree.FromPolygons(skin_verts, skin_polys)
    skin_eval.to_mesh_clear()
    return bvh


def point_is_inside(point, skin_bvh) -> bool:
    loc, normal, idx, dist = skin_bvh.find_nearest(point)
    if loc is None:
        return False
    to_p = point - loc
    return not (to_p.length > 1e-9 and to_p.normalized().dot(normal) > 0)


def verify_bone(bone_obj, skin_obj, max_iters: int = 26, shrink_factor: float = 0.9,
                max_outside_fraction: float = 0.0) -> dict:
    """Anisotropic shrink-to-fit against the skin's CURRENT evaluated mesh. Returns vertex
    outside counts before/after plus which axes were shrunk (per the logging requirement)."""
    import bpy

    skin_bvh = build_skin_bvh(skin_obj)

    before = _outside_count(bone_obj, skin_bvh)
    # BUG FIX: the shrink re-centres the bone's bounding box onto `center` after every step.
    # This used to be `bone_obj.location` (the object ORIGIN) -- but template-derived bones
    # have origins that are nowhere near their bbox centre (the "add" step places the bbox
    # centre at the requested position by offsetting the origin), so every shrink step dragged
    # the bone toward its origin: a systematic displacement, e.g. lake_sturgeon_fish009 bones
    # ending 0.4-0.45 half-heights off the midline right after being snapped onto it. The bbox
    # centre is the thing that must stay put.
    center = gask.world_bbox_for_objects([bone_obj])["center"].copy()
    center_inside = point_is_inside(center, skin_bvh)

    n_out, n_total = before["n_out"], before["n_total"]
    iters = 0
    axis_shrinks = [0, 0, 0]
    while n_total and (n_out / n_total) > max_outside_fraction and iters < max_iters:
        base_scale = bone_obj.scale.copy()
        best_axis, best_out = None, n_out
        for ax in AXES:
            trial = base_scale.copy()
            trial[ax] *= shrink_factor
            bone_obj.scale = trial
            bpy.context.view_layer.update()
            _recenter(bone_obj, center)
            out_ax = _outside_count(bone_obj, skin_bvh)["n_out"]
            # strict improvement wins; on ties prefer NOT touching the along-body axis (X)
            if out_ax < best_out or (out_ax == best_out and best_axis is not None and ax != 0 and best_axis == 0):
                best_axis, best_out = ax, out_ax
        if best_axis is None:
            # no single-axis shrink helps (typically: bone center itself is outside the skin, so
            # shrinking toward it can never fix containment) -- fall back to one uniform step so
            # behaviour degrades to the old method instead of spinning
            bone_obj.scale = base_scale * shrink_factor
            for ax in AXES:
                axis_shrinks[ax] += 1
        else:
            final = base_scale.copy()
            final[best_axis] *= shrink_factor
            bone_obj.scale = final
            axis_shrinks[best_axis] += 1
        bpy.context.view_layer.update()
        _recenter(bone_obj, center)
        n_out = _outside_count(bone_obj, skin_bvh)["n_out"]
        iters += 1

    after = _outside_count(bone_obj, skin_bvh)
    return {
        "bone_name": bone_obj.name,
        "vertices_outside_before": before["n_out"],
        "vertices_total_before": before["n_total"],
        "vertices_outside_after": after["n_out"],
        "vertices_total_after": after["n_total"],
        "shrink_iters": iters,
        "axis_shrink_steps": {"x_length": axis_shrinks[0], "y": axis_shrinks[1], "z": axis_shrinks[2]},
        "final_scale": [float(s) for s in bone_obj.scale],
        "center_inside_skin": center_inside,
        "fully_contained": after["n_out"] == 0,
    }


def _recenter(bone_obj, center):
    import bpy

    bbox = gask.world_bbox_for_objects([bone_obj])
    if bbox:
        bone_obj.location = bone_obj.location + (center - bbox["center"])
        bpy.context.view_layer.update()


def _outside_count(bone_obj, skin_bvh) -> dict:
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eo = bone_obj.evaluated_get(depsgraph)
    me = eo.to_mesh()
    pts = [bone_obj.matrix_world @ v.co for v in me.vertices]
    eo.to_mesh_clear()
    n_out = sum(0 if point_is_inside(p, skin_bvh) else 1 for p in pts)
    return {"n_out": n_out, "n_total": len(pts)}


def verify_all_bones(bone_objs: list, skin_obj, **kwargs) -> list[dict]:
    return [verify_bone(b, skin_obj, **kwargs) for b in bone_objs]
