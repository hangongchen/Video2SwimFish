import argparse
import json
import os
from datetime import datetime, timezone

import bpy
from math import radians
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def parse_args():
    parser = argparse.ArgumentParser(description="Append a skeleton template into an unlabeled fish blend and fit it by bounding box.")
    parser.add_argument("--template-blend", help="Supervised blend file that already contains skeleton data.")
    parser.add_argument("--template-skeleton-json", help="Extracted skeleton.json for the supervised template.")
    parser.add_argument("--template-index", help="Template index JSON. Used to select a template from target fish shape.")
    parser.add_argument("--target-length-m", type=float, default=0.5, help="Scale the fish longest axis to this real-world length before skeleton fitting.")
    parser.add_argument("--output", required=True, help="Output directory for generated dataset sample.")
    # VLM-predicted skeleton parameters (Video2SwimFish pipeline): the mesh's OWN bounding-box
    # aspect ratio is a poor proxy for body plan on a raw Meshy reconstruction -- barbels, fin
    # rays, or reconstruction noise can distort it in ways a VLM looking at the source PHOTO
    # does not have. When provided, these REPLACE the bbox-derived numbers used only for
    # template-selection SCORING (see select_template_from_index); actual mesh scale/placement
    # still comes from the real mesh bbox. Per the pipeline's own design rule, the VLM predicts
    # compact parameters only -- it never writes USD.
    parser.add_argument("--vlm-bone-count", type=int, default=None, help="VLM-predicted bone count override (replaces desired_bone_count(bbox ratio)).")
    parser.add_argument("--vlm-length-height-ratio", type=float, default=None, help="VLM-predicted length/height ratio override for template scoring.")
    parser.add_argument("--vlm-length-thickness-ratio", type=float, default=None, help="VLM-predicted length/thickness ratio override for template scoring.")
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def matrix_to_list(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def vec_to_list(vec):
    return [float(vec.x), float(vec.y), float(vec.z)]


def world_bbox_for_objects(objects):
    points = []
    for obj in objects:
        if not hasattr(obj, "bound_box") or not obj.bound_box:
            continue
        points.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    if not points:
        return None
    mins = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxs = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    size = maxs - mins
    return {"min": mins, "max": maxs, "center": (mins + maxs) * 0.5, "size": size}


def bbox_feature_record(bbox):
    size = bbox["size"]
    values = [float(size.x), float(size.y), float(size.z)]
    length = max(values)
    thickness = min(values)
    height = sum(values) - length - thickness
    return {
        "size": values,
        "length_axis": values.index(length),
        "height_axis": values.index(height),
        "thickness_axis": values.index(thickness),
        "length": length,
        "height": height,
        "thickness": thickness,
        "length_height_ratio": length / height if height else 0.0,
        "length_thickness_ratio": length / thickness if thickness else 0.0,
    }


def normalize_fish_name(name):
    base = os.path.splitext(os.path.basename(name))[0].lower()
    for suffix in ("_obj_1", "_obj", "_fish"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base.endswith(" fish"):
        base = base[:-5]
    return "".join(ch for ch in base if ch.isalnum())


def desired_bone_count(length_height_ratio):
    if length_height_ratio >= 3.2:
        return 8
    if length_height_ratio >= 2.6:
        return 7
    if length_height_ratio >= 2.15:
        return 6
    return 5


def select_template_from_index(template_index_path, target_mesh, warnings,
                                vlm_bone_count=None, vlm_lhr=None, vlm_ltr=None):
    if not template_index_path:
        return None
    with open(template_index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    templates = index.get("templates", [])
    target_bbox = world_bbox_for_objects([target_mesh]) if target_mesh else None
    if not target_bbox or not templates:
        warnings.append({"code": "TEMPLATE_INDEX_SELECTION_FAILED", "message": "Cannot select template without target bbox and templates."})
        return None

    target_features = bbox_feature_record(target_bbox)
    target_key = normalize_fish_name(bpy.data.filepath)
    name_matches = [item for item in templates if item.get("key") == target_key]
    if name_matches:
        selected = name_matches[0]
        selected["selection_reason"] = "matched_by_normalized_file_name"
        selected["target_features"] = target_features
        return selected

    # VLM overrides (Video2SwimFish): the raw mesh bbox ratio can be distorted by
    # reconstruction noise/appendages a VLM looking at the source photo would not be fooled by.
    scoring_lhr = vlm_lhr if vlm_lhr is not None else target_features["length_height_ratio"]
    scoring_ltr = vlm_ltr if vlm_ltr is not None else target_features["length_thickness_ratio"]
    target_count = vlm_bone_count if vlm_bone_count is not None else desired_bone_count(scoring_lhr)

    def score(item):
        features = item.get("features", {})
        bone_delta = abs(float(item.get("bone_count", 0)) - target_count)
        aspect_delta = abs(float(features.get("length_height_ratio", 0.0)) - scoring_lhr)
        thickness_delta = abs(float(features.get("length_thickness_ratio", 0.0)) - scoring_ltr) * 0.15
        return bone_delta * 2.0 + aspect_delta + thickness_delta

    selected = sorted(templates, key=score)[0]
    selected["selection_reason"] = ("shape_similarity_vlm_override" if
                                     (vlm_bone_count is not None or vlm_lhr is not None or vlm_ltr is not None)
                                     else "shape_similarity_desired_bone_count")
    selected["target_features"] = target_features
    selected["target_desired_bone_count"] = target_count
    selected["vlm_override"] = {"bone_count": vlm_bone_count, "length_height_ratio": vlm_lhr,
                                 "length_thickness_ratio": vlm_ltr}
    return selected


def visible_meshes():
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.data and len(obj.data.vertices) > 0 and not obj.hide_get() and not obj.hide_viewport
    ]


def select_main_mesh():
    candidates = []
    for obj in visible_meshes():
        bbox = world_bbox_for_objects([obj])
        if bbox:
            volume = abs(bbox["size"].x * bbox["size"].y * bbox["size"].z)
            candidates.append((volume, obj))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def axis_bilateral_symmetry_score(mesh_obj, axis_idx, bbox, n_bins=20):
    """Bilateral-symmetry score (0..1, higher = more symmetric) of mesh_obj's geometry when
    reflected across the center-plane perpendicular to axis_idx, via a coarse voxel-occupancy
    overlap in the OTHER two axes.

    Why this exists: bbox_feature_record's height_axis/thickness_axis assignment is PURE
    bbox-magnitude ranking ("2nd-largest = height, smallest = thickness"), which silently
    breaks for any real fish whose body happens to be WIDER (thickness) than it is TALL
    (height) -- e.g. a thick-bodied reconstruction from a single side-view photo, which has no
    real information about the true cross-section and can easily over-estimate width. Found via
    a real mismatch: a Meshy-reconstructed sturgeon's Y axis (ranked "height" by size) was
    visually the WIDTH, and Z (ranked "thickness") was the true dorsal-ventral height -- the
    bone/spur placement, which orients toward the "height" axis, ended up pointing sideways
    instead of dorsally.

    A fish body is close to bilaterally symmetric LEFT-RIGHT (thickness axis: the two sides of
    a fish look nearly identical) but NOT top-bottom (height axis: dorsal fin / back shape
    differs from the belly). Whichever of the two candidate axes has HIGHER symmetry is the
    true thickness axis; the other is the true height axis -- this disambiguates the two
    without any assumption about which one has the larger bbox extent."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mw = mesh_obj.matrix_world
    pts = [mw @ v.co for v in mesh.vertices]
    eval_obj.to_mesh_clear()
    if not pts:
        return 0.0

    other_axes = [i for i in range(3) if i != axis_idx]
    center, size = bbox["center"], bbox["size"]

    def bin_idx(p, ax):
        lo = center[ax] - size[ax] / 2.0
        frac = (p[ax] - lo) / max(size[ax], 1e-9)
        return min(n_bins - 1, max(0, int(frac * n_bins)))

    pos_hist, neg_hist = {}, {}
    for p in pts:
        key = (bin_idx(p, other_axes[0]), bin_idx(p, other_axes[1]))
        hist = pos_hist if (p[axis_idx] - center[axis_idx]) >= 0 else neg_hist
        hist[key] = hist.get(key, 0) + 1

    keys = set(pos_hist) | set(neg_hist)
    if not keys:
        return 0.0
    num = sum(min(pos_hist.get(k, 0), neg_hist.get(k, 0)) for k in keys)
    den = sum(max(pos_hist.get(k, 0), neg_hist.get(k, 0)) for k in keys)
    return num / den if den > 0 else 0.0


def resolve_height_thickness_axis(mesh_obj, bbox, height_axis, thickness_axis):
    """Returns the (height_axis, thickness_axis) pair, swapped from the bbox-magnitude guess
    if bilateral-symmetry evidence disagrees with it (see axis_bilateral_symmetry_score)."""
    if height_axis == thickness_axis:
        return height_axis, thickness_axis
    score_h = axis_bilateral_symmetry_score(mesh_obj, height_axis, bbox)
    score_t = axis_bilateral_symmetry_score(mesh_obj, thickness_axis, bbox)
    if score_h > score_t:
        # the axis guessed as "height" is actually the MORE symmetric one -> it's really thickness
        return thickness_axis, height_axis
    return height_axis, thickness_axis


def align_target_mesh_like_supervised(target_mesh):
    # Canonicalise the fish into the supervised template frame: LENGTH -> X(0), HEIGHT -> Y(1),
    # THICKNESS -> Z(2), with the HEAD at +X (the template convention). BUGFIX: the old version only
    # handled meshes whose length was ALREADY along X (it just corrected a Y<->Z swap) and bailed with
    # "unsupported_axis_layout_kept_original" for any other layout -- e.g. a Meshy export whose length
    # runs along Y. The downstream placement hard-codes axis 0 as the length, so an un-canonicalised
    # length-along-Y fish had its whole skeleton laid out along the THIN axis (wrong head/tail axis).
    # This generalises to ANY axis layout and also fixes the head/tail DIRECTION.
    bbox = world_bbox_for_objects([target_mesh])
    if not bbox:
        return {"applied": False, "reason": "no_bbox"}
    features_before = bbox_feature_record(bbox)
    La = features_before["length_axis"]
    Ha, Ta = resolve_height_thickness_axis(
        target_mesh, bbox, features_before["height_axis"], features_before["thickness_axis"])
    steps = []
    applied = False
    if (Ha, Ta) != (features_before["height_axis"], features_before["thickness_axis"]):
        steps.append({"height_thickness_swapped_by_symmetry_check": True,
                      "bbox_magnitude_guess": {"height_axis": features_before["height_axis"],
                                                "thickness_axis": features_before["thickness_axis"]},
                      "symmetry_corrected": {"height_axis": Ha, "thickness_axis": Ta}})

    # 1) rotate axis layout -> (length X, height Y, thickness Z), as a PROPER rotation (det +1).
    if (La, Ha, Ta) != (0, 1, 2):
        R3 = Matrix(((0.0, 0.0, 0.0),) * 3)
        for j, tgt in ((La, (1, 0, 0)), (Ha, (0, 1, 0)), (Ta, (0, 0, 1))):
            for i in range(3):
                R3[i][j] = float(tgt[i])
        if R3.determinant() < 0:                       # reflection -> flip lateral (thickness); harmless
            for i in range(3):
                R3[i][Ta] = -R3[i][Ta]
        center = bbox["center"]
        transform = Matrix.Translation(center) @ R3.to_4x4() @ Matrix.Translation(-center)
        target_mesh.matrix_world = transform @ target_mesh.matrix_world
        bpy.context.view_layer.update()
        applied = True
        steps.append({"axis_canonicalized_from": {"length_axis": La, "height_axis": Ha, "thickness_axis": Ta}})

    # 2) head/tail direction. The HEAD is the laterally BULKY end (large thickness/Z); the tail tapers
    #    to a thin peduncle + thin caudal fin (small Z). Template convention = HEAD at +X, so if the
    #    bulky end sits at -X, flip 180 about Y (swaps head<->tail, keeps up/down, mirrors L/R harmlessly).
    b2 = world_bbox_for_objects([target_mesh])
    cx = b2["center"].x
    mw = target_mesh.matrix_world
    z_plus, z_minus = [], []
    for v in target_mesh.data.vertices:
        w = mw @ v.co
        (z_plus if w.x >= cx else z_minus).append(w.z)
    def _span(a):
        return (max(a) - min(a)) if a else 0.0
    head_at_plus_x = _span(z_plus) >= _span(z_minus)
    if not head_at_plus_x:
        center = b2["center"]
        flip = Matrix.Translation(center) @ Matrix.Rotation(radians(180.0), 4, "Y") @ Matrix.Translation(-center)
        target_mesh.matrix_world = flip @ target_mesh.matrix_world
        bpy.context.view_layer.update()
        applied = True
        steps.append({"head_tail_flipped_180_about_y": True, "reason": "bulky_head_end_was_at_minus_x"})

    bbox_after = world_bbox_for_objects([target_mesh])
    return {
        "applied": applied,
        "reason": ("canonicalized_to_length_x_height_y_thickness_z_head_plus_x" if applied
                   else "already_canonical_length_x_height_y_thickness_z_head_plus_x"),
        "steps": steps,
        "features_before": features_before,
        "features_after": bbox_feature_record(bbox_after),
    }


def scale_target_mesh_to_real_length(target_mesh, target_length):
    bbox = world_bbox_for_objects([target_mesh])
    if not bbox:
        return {"applied": False, "reason": "no_bbox"}
    if target_length <= 0:
        return {"applied": False, "reason": "target_length_not_positive", "target_length_m": float(target_length)}
    before = bbox_feature_record(bbox)
    current_length = float(before["length"])
    if current_length <= 1e-8:
        return {"applied": False, "reason": "current_length_too_small", "features_before": before}
    scale = float(target_length) / current_length
    center = bbox["center"]
    transform = Matrix.Translation(center) @ Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)
    target_mesh.matrix_world = transform @ target_mesh.matrix_world
    bpy.context.view_layer.update()
    after_bbox = world_bbox_for_objects([target_mesh])
    return {
        "applied": True,
        "reason": "scaled_target_mesh_longest_axis_to_real_length",
        "target_length_m": float(target_length),
        "scale": float(scale),
        "features_before": before,
        "features_after": bbox_feature_record(after_bbox),
        "transform": matrix_to_list(transform),
    }


def load_template_object_names(template_skeleton_json):
    if not template_skeleton_json or not os.path.exists(template_skeleton_json):
        return None
    with open(template_skeleton_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = []
    names.extend([item["object_name"] for item in data.get("armatures", [])])
    names.extend([item["object_name"] for item in data.get("skeleton_objects", [])])
    return names


def load_template_skeleton_data(template_skeleton_json):
    if not template_skeleton_json or not os.path.exists(template_skeleton_json):
        return None
    with open(template_skeleton_json, "r", encoding="utf-8") as f:
        return json.load(f)


def append_template_skeleton(template_blend, template_skeleton_json=None):
    before = set(bpy.data.objects)
    requested_names = load_template_object_names(template_skeleton_json)
    with bpy.data.libraries.load(template_blend, link=False) as (data_from, data_to):
        if requested_names:
            available = set(data_from.objects)
            data_to.objects = [name for name in requested_names if name in available]
        else:
            data_to.objects = list(data_from.objects)
    appended = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            appended.append(obj)
    new_objects = [obj for obj in appended if obj not in before]

    # If no extracted skeleton.json was provided, the supervised template may have
    # loaded both the fish mesh and skeleton. Keep the target unlabeled fish mesh,
    # so remove the largest appended mesh as the likely template fish body.
    appended_meshes = [obj for obj in new_objects if obj.type == "MESH"]
    template_body = None
    template_body_name = None
    if appended_meshes and not requested_names:
        ranked = []
        for obj in appended_meshes:
            bbox = world_bbox_for_objects([obj])
            if bbox is None:
                continue
            volume = abs(bbox["size"].x * bbox["size"].y * bbox["size"].z)
            ranked.append((volume, obj))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked:
            template_body = ranked[0][1]
            template_body_name = template_body.name
            new_objects.remove(template_body)
            bpy.data.objects.remove(template_body, do_unlink=True)

    armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
    mesh_helpers = [obj for obj in new_objects if obj.type == "MESH"]
    missing_names = []
    if requested_names:
        loaded_names = {obj.name for obj in new_objects}
        missing_names = [name for name in requested_names if name not in loaded_names]
    return armatures, mesh_helpers, new_objects, template_body_name, requested_names, missing_names


def fit_objects_to_target(objects, target_mesh):
    target_bbox = world_bbox_for_objects([target_mesh])
    source_bbox = world_bbox_for_objects(objects)
    if target_bbox is None or source_bbox is None:
        return None

    def safe_scale(target, source):
        if abs(source) < 1e-8:
            return 1.0
        return target / source

    scale_vec = Vector(
        (
            safe_scale(target_bbox["size"].x, source_bbox["size"].x),
            safe_scale(target_bbox["size"].y, source_bbox["size"].y),
            safe_scale(target_bbox["size"].z, source_bbox["size"].z),
        )
    )
    # The supervised files use vertical helper bone meshes. Keep their orientation
    # and scale primarily by fish length/height, not by the very thin depth axis.
    values = [target_bbox["size"].x, target_bbox["size"].y, target_bbox["size"].z]
    thickness_axis = values.index(min(values))
    usable_scales = [abs(scale_vec.x), abs(scale_vec.y), abs(scale_vec.z)]
    usable_scales.pop(thickness_axis)
    uniform_scale = min(usable_scales)
    source_center = source_bbox["center"]
    target_center = target_bbox["center"]

    transform = Matrix.Translation(target_center) @ Matrix.Diagonal((uniform_scale, uniform_scale, uniform_scale, 1.0)) @ Matrix.Translation(-source_center)
    roots = [obj for obj in objects if obj.parent not in objects]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world
    return {
        "source_bbox_size": vec_to_list(source_bbox["size"]),
        "target_bbox_size": vec_to_list(target_bbox["size"]),
        "source_bbox_center": vec_to_list(source_center),
        "target_bbox_center": vec_to_list(target_center),
        "uniform_scale": float(uniform_scale),
        "ignored_thickness_axis_for_scale": thickness_axis,
        "fit_matrix": matrix_to_list(transform),
    }


def axis_center(bbox, axis):
    return (bbox["min"][axis] + bbox["max"][axis]) * 0.5


def get_world_verts(mesh_obj):
    """World-space vertex positions of mesh_obj's evaluated (modifier-applied) geometry."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mw = mesh_obj.matrix_world
    pts = [mw @ v.co for v in mesh.vertices]
    eval_obj.to_mesh_clear()
    return pts


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * pct
    lo_i, hi_i = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - lo_i
    return sorted_values[lo_i] * (1 - frac) + sorted_values[hi_i] * frac


def slice_bbox(world_points, x_min, x_max, pad_frac=0.15, min_points=8, max_widen=6,
                trim_pct=0.08):
    """Y/Z extent of world_points whose X falls in [x_min, x_max] -- the ACTUAL local
    cross-section of the target mesh at one bone's length-axis position, used instead of the
    whole-fish average bbox. Found via a real defect: placing every bone at the SAME global
    center_y/center_z (the fish's overall bbox center) put bones up to ~19mm outside the skin
    wherever the real local cross-section deviates from that global average -- most visibly at
    the head/jaw, which is narrower and differently shaped than the mid-body.

    Uses the [trim_pct, 1-trim_pct] PERCENTILE range, not raw min/max: a bone-width X-slice can
    still catch a handful of vertices from a nearby FIN (which flares much further out in Y/Z
    than the body trunk the bone actually sits inside), and raw min/max lets even a few such
    outliers inflate the "local extent" enough that a bone sized to fit it still pokes through
    the much-thinner actual trunk surface -- this was found directly: bones whose local slice
    picked up fin vertices had ~2x the size_z of neighboring bones and were the worst offenders
    for poking outside the skin even after this local-fit was first added.

    Widens the slice if too few points land inside (e.g. a thin bone segment between sparse
    vertex rings) so the local estimate stays well-conditioned."""
    span = max(x_max - x_min, 1e-6)
    lo, hi = x_min, x_max
    pts = [p for p in world_points if lo <= p.x <= hi]
    tries = 0
    while len(pts) < min_points and tries < max_widen:
        lo -= span * pad_frac
        hi += span * pad_frac
        pts = [p for p in world_points if lo <= p.x <= hi]
        tries += 1
    if not pts:
        return None
    ys = sorted(p.y for p in pts)
    zs = sorted(p.z for p in pts)
    y_lo, y_hi = _percentile(ys, trim_pct), _percentile(ys, 1 - trim_pct)
    z_lo, z_hi = _percentile(zs, trim_pct), _percentile(zs, 1 - trim_pct)
    return {
        "center_y": (y_lo + y_hi) / 2.0,
        "center_z": (z_lo + z_hi) / 2.0,
        "size_y": max(y_hi - y_lo, 1e-6),
        "size_z": max(z_hi - z_lo, 1e-6),
        "n_points": len(pts),
    }


def shrink_bone_to_fit_skin(bone_obj, skin_bvh, max_iters=26, shrink_factor=0.9,
                             max_outside_fraction=0.0):
    """Directly test bone_obj's vertices against the skin's actual surface (nearest point +
    normal-side test via skin_bvh) and progressively scale the bone DOWN toward its own
    center until at most max_outside_fraction of its vertices are outside.

    Why this exists, replacing a purely size/position-based local-cross-section fit: that
    approach (slice_bbox, tried first) computes a bbox from the skin's raw vertex extent in a
    length-axis slice, but a fish skin is a THIN, CURVED SHELL -- its bbox center is not the
    same as being safely inside the shell, and a bone can be well within its "local extent"
    bbox by every size/position measure and still have vertices poking through the actual
    curved surface (confirmed directly: verified by rendering, not just distance numbers, on
    catfish_fish001's dorsal spine bones near the head -- clearly visible poking above the
    dorsal line even after the local-cross-section fit was in place). Testing against the
    REAL surface directly, not an indirect proxy, is the only way to guarantee containment on
    a shape like this."""
    depsgraph = bpy.context.evaluated_depsgraph_get()

    def outside_count_and_pts():
        eo = bone_obj.evaluated_get(depsgraph)
        me = eo.to_mesh()
        pts = [bone_obj.matrix_world @ v.co for v in me.vertices]
        eo.to_mesh_clear()
        n_out = 0
        for p in pts:
            loc, normal, idx, dist = skin_bvh.find_nearest(p)
            if loc is None:
                continue
            to_p = p - loc
            if to_p.length > 1e-9 and to_p.normalized().dot(normal) > 0:
                n_out += 1
        return n_out, len(pts)

    center = bone_obj.location.copy()
    n_out, n_total = outside_count_and_pts()
    iters = 0
    while n_total and (n_out / n_total) > max_outside_fraction and iters < max_iters:
        bone_obj.scale = bone_obj.scale * shrink_factor
        bpy.context.view_layer.update()
        # re-center: uniform scale-about-origin can drift the bbox center if the mesh's own
        # local origin isn't at its bbox center -- pull back to the same target center each time
        bbox = world_bbox_for_objects([bone_obj])
        if bbox:
            bone_obj.location = bone_obj.location + (center - bbox["center"])
            bpy.context.view_layer.update()
        n_out, n_total = outside_count_and_pts()
        iters += 1
    return {"final_outside_count": n_out, "final_total": n_total, "shrink_iters": iters}


def arrange_mesh_bones_with_template_spacing(mesh_helpers, target_mesh, template_skeleton_json):
    template_data = load_template_skeleton_data(template_skeleton_json)
    if not template_data or not mesh_helpers or target_mesh is None:
        return None

    template_fish = template_data.get("fish_mesh") or {}
    template_fish_bbox = template_fish.get("bbox_world")
    template_objects = template_data.get("skeleton_objects", [])
    if not template_fish_bbox or not template_objects:
        return None

    template_by_name = {item["object_name"]: item for item in template_objects}
    target_bbox = world_bbox_for_objects([target_mesh])
    if not target_bbox:
        return None

    records = []
    for obj in mesh_helpers:
        item = template_by_name.get(obj.name)
        if not item or not item.get("bbox_world"):
            continue
        bbox = item["bbox_world"]
        center_x = axis_center(bbox, 0)
        records.append({"object": obj, "template": item, "center_x": center_x})
    records.sort(key=lambda item: item["center_x"])
    if not records:
        return None

    template_fish_size = template_fish_bbox["size"]
    template_fish_center = [axis_center(template_fish_bbox, axis) for axis in range(3)]
    template_skel_min_x = min(item["template"]["bbox_world"]["min"][0] for item in records)
    template_skel_max_x = max(item["template"]["bbox_world"]["max"][0] for item in records)
    template_skel_center_x = (template_skel_min_x + template_skel_max_x) * 0.5
    template_skel_span_x = template_skel_max_x - template_skel_min_x

    target_size = [target_bbox["size"].x, target_bbox["size"].y, target_bbox["size"].z]
    target_center = target_bbox["center"]

    coverage_ratio = template_skel_span_x / template_fish_size[0] if template_fish_size[0] else 0.52
    coverage_ratio = min(max(coverage_ratio, 0.38), 0.72)
    center_offset_ratio = (template_skel_center_x - template_fish_center[0]) / template_fish_size[0] if template_fish_size[0] else 0.0
    center_offset_ratio = min(max(center_offset_ratio, -0.12), 0.12)

    template_widths = [item["template"]["bbox_world"]["size"][0] for item in records]
    template_heights = [item["template"]["bbox_world"]["size"][1] for item in records]
    template_depths = [item["template"]["bbox_world"]["size"][2] for item in records]
    template_gaps = []
    for left, right in zip(records, records[1:]):
        left_bbox = left["template"]["bbox_world"]
        right_bbox = right["template"]["bbox_world"]
        template_gaps.append(right_bbox["min"][0] - left_bbox["max"][0])
    positive_gaps = [gap for gap in template_gaps if gap > 0]
    gap_ratio = (sum(positive_gaps) / len(positive_gaps) / template_fish_size[0]) if positive_gaps and template_fish_size[0] else 0.006
    target_gap = min(max(gap_ratio * target_size[0], target_size[0] * 0.004), target_size[0] * 0.012)

    n = len(records)
    target_span = target_size[0] * coverage_ratio
    total_gap = target_gap * max(0, n - 1)
    available_width = max(target_span - total_gap, target_size[0] * 0.12)
    width_sum = sum(template_widths) or 1.0
    target_widths = [available_width * width / width_sum for width in template_widths]

    target_skel_center_x = target_center.x + center_offset_ratio * target_size[0]
    start_x = target_skel_center_x - (sum(target_widths) + total_gap) * 0.5
    y_scale = target_size[1] / template_fish_size[1] if template_fish_size[1] else 1.0
    z_scale = target_size[2] / template_fish_size[2] if template_fish_size[2] else 1.0

    placements = []
    cursor = start_x
    bone_height_scale = 0.88
    bone_depth_scale = 0.95
    local_fit_margin = 0.82  # keep each bone within 82% of its LOCAL cross-section extent
    target_verts = get_world_verts(target_mesh)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    skin_eval = target_mesh.evaluated_get(depsgraph)
    skin_eval_mesh = skin_eval.to_mesh()
    skin_polys = [list(p.vertices) for p in skin_eval_mesh.polygons]
    skin_bvh = BVHTree.FromPolygons(target_verts, skin_polys)
    skin_eval.to_mesh_clear()
    for idx, item in enumerate(records):
        obj = item["object"]
        template_bbox = item["template"]["bbox_world"]
        width = target_widths[idx]
        height = min(template_heights[idx] * y_scale, target_size[1] * 0.92) * bone_height_scale
        depth = min(max(template_depths[idx] * z_scale, target_size[2] * 0.04), target_size[2] * 0.45) * bone_depth_scale

        center_x = cursor + width * 0.5
        cursor += width + target_gap

        # LOCAL cross-section of the actual mesh at this bone's X range, not the whole-fish
        # average -- see slice_bbox's docstring for the defect this fixes.
        local = slice_bbox(target_verts, center_x - width * 0.5, center_x + width * 0.5)
        if local:
            center_y = local["center_y"]
            center_z = local["center_z"]
            height = min(height, local["size_y"] * local_fit_margin)
            depth = min(depth, local["size_z"] * local_fit_margin)
        else:
            center_y = target_center.y
            center_z = target_center.z

        desired_center = Vector((center_x, center_y, center_z))
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.location = desired_center
        bpy.context.view_layer.update()
        obj.dimensions = (width, height, depth)
        bpy.context.view_layer.update()
        actual_bbox = world_bbox_for_objects([obj])
        if actual_bbox:
            actual_center = actual_bbox["center"]
            obj.location = obj.location + (desired_center - actual_center)
            bpy.context.view_layer.update()

        # FINAL containment enforcement: directly test against the real (curved, thin-shell)
        # skin surface and shrink until contained -- see shrink_bone_to_fit_skin's docstring
        # for why the size/position estimate above is not sufficient by itself.
        shrink_report = shrink_bone_to_fit_skin(obj, skin_bvh)

        placements.append(
            {
                "object_name": obj.name,
                "center": [float(center_x), float(center_y), float(center_z)],
                "dimensions": [float(width), float(height), float(depth)],
                "used_local_cross_section": local is not None,
                "local_cross_section": local,
                "shrink_to_fit": shrink_report,
            }
        )

    return {
        "method": "template_spacing_v4_local_cross_section_fit",
        "coverage_ratio": float(coverage_ratio),
        "center_offset_ratio": float(center_offset_ratio),
        "target_gap": float(target_gap),
        "bone_height_scale": float(bone_height_scale),
        "bone_depth_scale": float(bone_depth_scale),
        "local_fit_margin": float(local_fit_margin),
        "placements": placements,
    }


def bone_gap_report(mesh_helpers):
    rows = []
    for obj in mesh_helpers:
        bbox = world_bbox_for_objects([obj])
        if not bbox:
            continue
        rows.append(
            {
                "object_name": obj.name,
                "min_x": float(bbox["min"].x),
                "max_x": float(bbox["max"].x),
                "center_x": float((bbox["min"].x + bbox["max"].x) * 0.5),
                "size_x": float(bbox["size"].x),
            }
        )
    rows.sort(key=lambda item: item["center_x"])
    gaps = []
    duplicate_like_pairs = []
    overlap_pairs = []
    for left, right in zip(rows, rows[1:]):
        gap = right["min_x"] - left["max_x"]
        pair = [left["object_name"], right["object_name"]]
        gaps.append({"pair": pair, "gap": float(gap)})
        if gap < 0:
            overlap_pairs.append({"pair": pair, "overlap": float(-gap)})
        center_delta = abs(right["center_x"] - left["center_x"])
        avg_width = max((left["size_x"] + right["size_x"]) * 0.5, 1e-8)
        if center_delta / avg_width < 0.05:
            duplicate_like_pairs.append({"pair": pair, "center_delta": float(center_delta)})
    return {
        "ordered_bones": rows,
        "gaps": gaps,
        "min_gap": min((item["gap"] for item in gaps), default=None),
        "overlap_pairs": overlap_pairs,
        "duplicate_like_pairs": duplicate_like_pairs,
        "has_overlap": bool(overlap_pairs),
        "has_duplicate_like_centers": bool(duplicate_like_pairs),
    }


def bbox_violation(objects, target_mesh, margin_ratio=0.08):
    target_bbox = world_bbox_for_objects([target_mesh])
    source_bbox = world_bbox_for_objects(objects)
    if not target_bbox or not source_bbox:
        return None
    margin = max(target_bbox["size"].x, target_bbox["size"].y, target_bbox["size"].z) * margin_ratio
    under = [
        max(0.0, target_bbox["min"].x - margin - source_bbox["min"].x),
        max(0.0, target_bbox["min"].y - margin - source_bbox["min"].y),
        max(0.0, target_bbox["min"].z - margin - source_bbox["min"].z),
    ]
    over = [
        max(0.0, source_bbox["max"].x - (target_bbox["max"].x + margin)),
        max(0.0, source_bbox["max"].y - (target_bbox["max"].y + margin)),
        max(0.0, source_bbox["max"].z - (target_bbox["max"].z + margin)),
    ]
    return {
        "target_bbox_min": vec_to_list(target_bbox["min"]),
        "target_bbox_max": vec_to_list(target_bbox["max"]),
        "skeleton_bbox_min": vec_to_list(source_bbox["min"]),
        "skeleton_bbox_max": vec_to_list(source_bbox["max"]),
        "margin": float(margin),
        "under": under,
        "over": over,
        "has_violation": any(v > 0 for v in under + over),
    }


def recenter_if_outside(objects, target_mesh):
    check = bbox_violation(objects, target_mesh)
    if not check or not check["has_violation"]:
        return {"applied": False, "before": check, "after": check}
    target_bbox = world_bbox_for_objects([target_mesh])
    source_bbox = world_bbox_for_objects(objects)
    values = [target_bbox["size"].x, target_bbox["size"].y, target_bbox["size"].z]
    thickness_axis = values.index(min(values))
    target_center = target_bbox["center"]
    source_center = source_bbox["center"]
    delta = Vector((0.0, 0.0, 0.0))
    for axis in range(3):
        if axis == thickness_axis:
            delta[axis] = target_center[axis] - source_center[axis]
        else:
            delta[axis] = (target_center[axis] - source_center[axis]) * 0.35
    roots = [obj for obj in objects if obj.parent not in objects]
    for obj in roots:
        obj.matrix_world = Matrix.Translation(delta) @ obj.matrix_world
    return {"applied": True, "delta": vec_to_list(delta), "before": check, "after": bbox_violation(objects, target_mesh)}


def save_blend(output_dir):
    path = os.path.join(output_dir, "auto_skeleton.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output)
    ensure_dir(os.path.join(output_dir, "logs"))

    warnings = []
    target_mesh = select_main_mesh()
    if target_mesh is None:
        warnings.append({"code": "NO_TARGET_MESH", "message": "No visible target fish mesh found."})
    target_alignment = align_target_mesh_like_supervised(target_mesh) if target_mesh is not None else None
    target_real_scale = scale_target_mesh_to_real_length(target_mesh, args.target_length_m) if target_mesh is not None else None
    selected_template = select_template_from_index(
        args.template_index, target_mesh, warnings,
        vlm_bone_count=args.vlm_bone_count, vlm_lhr=args.vlm_length_height_ratio,
        vlm_ltr=args.vlm_length_thickness_ratio,
    ) if args.template_index else None
    template_blend = args.template_blend
    template_skeleton_json = args.template_skeleton_json
    if selected_template:
        template_blend = selected_template.get("template_blend")
        template_skeleton_json = selected_template.get("template_skeleton_json")
    if not template_blend:
        raise ValueError("Either --template-blend or --template-index must provide a template blend.")
    armatures, mesh_helpers, new_objects, removed_template_body, requested_names, missing_names = append_template_skeleton(
        template_blend, template_skeleton_json
    )
    if not armatures:
        warnings.append({"code": "TEMPLATE_HAS_NO_ARMATURE", "message": "Template file did not provide any Armature object."})
    if missing_names:
        warnings.append(
            {
                "code": "TEMPLATE_OBJECTS_MISSING",
                "message": "Some objects from template skeleton.json were not found in the template blend.",
                "objects": missing_names,
            }
        )

    arrangement = arrange_mesh_bones_with_template_spacing(mesh_helpers, target_mesh, template_skeleton_json)
    fit = None
    if arrangement is None:
        fit = fit_objects_to_target(new_objects, target_mesh) if target_mesh is not None else None
        if fit is None:
            warnings.append({"code": "FIT_FAILED", "message": "Could not fit template skeleton to target mesh bounding box."})
    containment_adjustment = recenter_if_outside(new_objects, target_mesh) if target_mesh is not None and fit is not None else None
    if containment_adjustment and containment_adjustment.get("after", {}).get("has_violation"):
        warnings.append({"code": "SKELETON_BBOX_OUTSIDE_TARGET_AFTER_ADJUST", "message": "Skeleton bbox still extends beyond target fish bbox margin after adjustment."})
    gaps = bone_gap_report(mesh_helpers)
    if gaps.get("has_overlap"):
        warnings.append({"code": "BONE_OVERLAP_DETECTED", "message": "Some neighboring bone objects overlap along the fish length axis.", "pairs": gaps.get("overlap_pairs")})
    if gaps.get("has_duplicate_like_centers"):
        warnings.append({"code": "DUPLICATE_LIKE_BONE_CENTERS", "message": "Some neighboring bone objects have nearly identical centers.", "pairs": gaps.get("duplicate_like_pairs")})

    blend_path = save_blend(output_dir)
    metadata = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_unlabeled_blend": bpy.data.filepath,
        "template_blend": template_blend,
        "template_skeleton_json": template_skeleton_json,
        "template_selection": selected_template,
        "output_blend": blend_path,
        "target_mesh": target_mesh.name if target_mesh else None,
        "target_alignment": target_alignment,
        "target_real_scale": target_real_scale,
        "requested_template_objects": requested_names,
        "removed_template_body_mesh": removed_template_body,
        "appended_armatures": [obj.name for obj in armatures],
        "appended_mesh_helpers": [obj.name for obj in mesh_helpers],
        "fit": fit,
        "arrangement": arrangement,
        "containment_adjustment": containment_adjustment,
        "bone_gap_report": gaps,
        "warnings": warnings,
    }
    with open(os.path.join(output_dir, "auto_skeleton_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "logs", "auto_skeleton.log"), "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
