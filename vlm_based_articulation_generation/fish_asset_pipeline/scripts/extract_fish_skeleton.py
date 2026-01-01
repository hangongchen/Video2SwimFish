import argparse
import json
import math
import os
from datetime import datetime, timezone

import bpy
from mathutils import Vector


SCHEMA_VERSION = "0.1.0"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract fish mesh and skeleton data from a Blender file.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--render-views", action="store_true", help="Reserved for future rendered view export.")
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


def quat_to_list(quat):
    return [float(quat.w), float(quat.x), float(quat.y), float(quat.z)]


def object_transform_record(obj):
    quat = obj.rotation_euler.to_quaternion()
    return {
        "world_matrix": matrix_to_list(obj.matrix_world),
        "local_matrix": matrix_to_list(obj.matrix_local),
        "location": vec_to_list(obj.location),
        "rotation_euler": [float(v) for v in obj.rotation_euler],
        "rotation_quaternion": quat_to_list(quat),
        "scale": vec_to_list(obj.scale),
    }


def world_bbox(obj):
    if not hasattr(obj, "bound_box") or not obj.bound_box:
        return None
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxs = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    size = maxs - mins
    volume = abs(size.x * size.y * size.z)
    return {
        "min": vec_to_list(mins),
        "max": vec_to_list(maxs),
        "size": vec_to_list(size),
        "volume": float(volume),
    }


def mesh_counts(obj):
    if obj.type != "MESH" or obj.data is None:
        return 0, 0
    return len(obj.data.vertices), len(obj.data.polygons)


def mesh_world_volume(obj):
    if obj.type != "MESH" or obj.data is None:
        return None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        vertices = [eval_obj.matrix_world @ vertex.co for vertex in mesh.vertices]
        signed_volume = 0.0
        for poly in mesh.polygons:
            indices = list(poly.vertices)
            if len(indices) < 3:
                continue
            p0 = vertices[indices[0]]
            for i in range(1, len(indices) - 1):
                p1 = vertices[indices[i]]
                p2 = vertices[indices[i + 1]]
                signed_volume += p0.dot(p1.cross(p2)) / 6.0
        return abs(float(signed_volume))
    finally:
        eval_obj.to_mesh_clear()


def is_visible_candidate(obj):
    if obj.type != "MESH":
        return False
    if obj.hide_get() or obj.hide_viewport:
        return False
    return obj.data is not None and len(obj.data.vertices) > 0


def select_main_fish_mesh(warnings):
    candidates = []
    for obj in bpy.context.scene.objects:
        if not is_visible_candidate(obj):
            continue
        bbox = world_bbox(obj)
        if bbox is None:
            continue
        vertex_count, face_count = mesh_counts(obj)
        candidates.append(
            {
                "object": obj,
                "bbox": bbox,
                "vertex_count": vertex_count,
                "face_count": face_count,
            }
        )

    candidates.sort(key=lambda item: item["bbox"]["volume"], reverse=True)
    if not candidates:
        warnings.append(
            {
                "code": "NO_MAIN_MESH_CANDIDATE",
                "message": "No visible mesh object with vertices was found.",
                "objects": [],
            }
        )
        return None, []

    if len(candidates) > 1:
        top = candidates[0]["bbox"]["volume"]
        second = candidates[1]["bbox"]["volume"]
        if top > 0 and second / top >= 0.85:
            warnings.append(
                {
                    "code": "MULTIPLE_MAIN_MESH_CANDIDATES",
                    "message": "Multiple mesh objects have similar world bounding-box volume. The largest one was selected.",
                    "objects": [candidates[0]["object"].name, candidates[1]["object"].name],
                }
            )

    return candidates[0]["object"], candidates


def export_obj(obj, output_path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True)
    except Exception:
        bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)


def fish_mesh_record(obj, export_path, warnings):
    if obj is None:
        return None
    vertex_count, face_count = mesh_counts(obj)
    record = {
        "object_name": obj.name,
        "object_type": obj.type,
        "selection_reason": "largest_visible_mesh_by_world_bbox_volume",
        "confidence": "medium" if warnings else "high",
        **object_transform_record(obj),
        "bbox_world": world_bbox(obj),
        "mesh_volume_world_m3": mesh_world_volume(obj),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "export_path": os.path.basename(export_path),
    }
    return record


def decompose_matrix(matrix):
    loc, rot, scale = matrix.decompose()
    return {
        "location": vec_to_list(loc),
        "rotation_quaternion": quat_to_list(rot),
        "rotation_euler": [float(v) for v in rot.to_euler()],
        "scale": vec_to_list(scale),
    }


def bone_record(armature_obj, bone):
    head_local = bone.head_local.copy()
    tail_local = bone.tail_local.copy()
    center_local = (head_local + tail_local) * 0.5
    head_world = armature_obj.matrix_world @ head_local
    tail_world = armature_obj.matrix_world @ tail_local
    center_world = armature_obj.matrix_world @ center_local
    orientation_local = tail_local - head_local
    orientation_world = tail_world - head_world
    length = float(orientation_world.length)
    if orientation_local.length > 0:
        orientation_local.normalize()
    if orientation_world.length > 0:
        orientation_world.normalize()

    matrix_world = armature_obj.matrix_world @ bone.matrix_local
    return {
        "bone_name": bone.name,
        "parent_bone": bone.parent.name if bone.parent else None,
        "child_bones": [child.name for child in bone.children],
        "head_local": vec_to_list(head_local),
        "tail_local": vec_to_list(tail_local),
        "head_world": vec_to_list(head_world),
        "tail_world": vec_to_list(tail_world),
        "center_local": vec_to_list(center_local),
        "center_world": vec_to_list(center_world),
        "length": length,
        "orientation_vector_local": vec_to_list(orientation_local),
        "orientation_vector_world": vec_to_list(orientation_world),
        "matrix_local": matrix_to_list(bone.matrix_local),
        "matrix_world": matrix_to_list(matrix_world),
        **decompose_matrix(matrix_world),
    }


def armature_record(obj):
    bones = [bone_record(obj, bone) for bone in obj.data.bones]
    return {
        "object_name": obj.name,
        "object_type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "children": [child.name for child in obj.children],
        **object_transform_record(obj),
        "bone_count": len(bones),
        "bones": bones,
    }


def skeleton_object_record(obj):
    vertex_count, face_count = mesh_counts(obj)
    return {
        "object_name": obj.name,
        "object_type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "children": [child.name for child in obj.children],
        **object_transform_record(obj),
        "bbox_world": world_bbox(obj),
        "vertex_count": vertex_count,
        "face_count": face_count,
    }


def scene_object_summary(obj):
    return {
        "object_name": obj.name,
        "object_type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "children": [child.name for child in obj.children],
    }


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output)
    ensure_dir(os.path.join(output_dir, "logs"))
    warnings = []

    fish_obj, mesh_candidates = select_main_fish_mesh(warnings)
    obj_path = os.path.join(output_dir, "fish_mesh.obj")
    if fish_obj is not None:
        export_obj(fish_obj, obj_path)

    armature_objs = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armature_objs:
        warnings.append(
            {
                "code": "NO_ARMATURE_FOUND",
                "message": "No Armature object was found in this Blender file.",
                "objects": [],
            }
        )

    skeleton_mesh_objs = []
    for item in mesh_candidates:
        obj = item["object"]
        if fish_obj is not None and obj.name == fish_obj.name:
            continue
        skeleton_mesh_objs.append(obj)

    result = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "blend_path": bpy.data.filepath,
            "blend_file_name": os.path.basename(bpy.data.filepath),
            "blender_version": bpy.app.version_string,
            "unit_system": bpy.context.scene.unit_settings.system,
            "unit_scale": float(bpy.context.scene.unit_settings.scale_length),
        },
        "export": {
            "output_dir": output_dir,
            "fish_mesh_obj": "fish_mesh.obj" if fish_obj is not None else None,
            "rendered_views": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "fish_mesh": fish_mesh_record(fish_obj, obj_path, warnings),
        "armatures": [armature_record(obj) for obj in armature_objs],
        "skeleton_objects": [skeleton_object_record(obj) for obj in skeleton_mesh_objs],
        "scene_objects_summary": [scene_object_summary(obj) for obj in bpy.context.scene.objects],
        "warnings": warnings,
    }

    json_path = os.path.join(output_dir, "skeleton.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log_path = os.path.join(output_dir, "logs", "extraction.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Extraction complete\n")
        f.write(f"Source: {bpy.data.filepath}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Main fish mesh: {fish_obj.name if fish_obj else 'None'}\n")
        f.write(f"Armatures: {len(armature_objs)}\n")
        f.write(f"Skeleton mesh objects: {len(skeleton_mesh_objs)}\n")
        for warning in warnings:
            f.write(f"WARNING {warning['code']}: {warning['message']}\n")


if __name__ == "__main__":
    main()
