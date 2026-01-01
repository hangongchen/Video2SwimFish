"""Uniformly scale a GT-style skeleton blend (fish mesh + bone meshes) so the
fish's longest axis matches a real-world target length, then save.

The stock pipeline scales the fish during auto-skeleton fitting. When the bones
are already placed (e.g. produced by the VLM rigger), that step is skipped, so we
apply the same real-length scaling here about the fish bbox center to every
object, preserving the relative bone placement.

    blender -b input.blend --python scale_skeleton_blend_to_real_length.py -- \
        --target-length-m 0.5 --output out.blend
"""

import argparse
import os
import sys

import bpy
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--target-length-m", type=float, default=0.5)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def world_bbox(objs):
    pts = []
    for o in objs:
        pts.extend([o.matrix_world @ Vector(c) for c in o.bound_box])
    mn = Vector((min(p[i] for p in pts) for i in range(3)))
    mx = Vector((max(p[i] for p in pts) for i in range(3)))
    return mn, mx


def main():
    args = parse_args()
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.data and len(o.data.vertices) > 0]

    def vol(o):
        mn, mx = world_bbox([o])
        s = mx - mn
        return abs(s.x * s.y * s.z)

    fish = max(meshes, key=vol)
    fmn, fmx = world_bbox([fish])
    size = fmx - fmn
    length = max(size.x, size.y, size.z)
    scale = float(args.target_length_m) / length if length > 1e-9 else 1.0
    center = (fmn + fmx) * 0.5
    xform = Matrix.Translation(center) @ Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)

    roots = [o for o in bpy.context.scene.objects if o.parent is None]
    for o in roots:
        o.matrix_world = xform @ o.matrix_world
    bpy.context.view_layer.update()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"SCALED fish '{fish.name}' length {length:.4f} -> {args.target_length_m} (scale {scale:.4f}); saved {args.output}")


if __name__ == "__main__":
    main()
