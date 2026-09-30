"""Blender script: render a few solid-shaded views of a mesh (GLB or the body mesh inside a
USD) for the VLM quality-critic step (verify_mesh_vlm.py) to look at.

  blender -b --python render_mesh_views.py -- --mesh mesh.glb --out_prefix /tmp/check
  blender -b --python render_mesh_views.py -- --usd fish_articulated.usda --mesh_name Mesh_0 --out_prefix /tmp/check

Writes <out_prefix>_persp.png, <out_prefix>_top.png, <out_prefix>_side.png.
"""
import math
import sys

import bpy


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            val = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "true"
            args[key] = val
            i += 2 if val != "true" else 1
        else:
            i += 1
    return args


def main():
    args = parse_args()
    out_prefix = args["out_prefix"]

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    if "mesh" in args:
        bpy.ops.import_scene.gltf(filepath=args["mesh"])
    elif "usd" in args:
        bpy.ops.wm.usd_import(filepath=args["usd"])
        mesh_name = args.get("mesh_name")
        if mesh_name:
            for o in list(bpy.context.scene.objects):
                if o.type == "MESH" and o.name != mesh_name:
                    o.hide_render = True
    else:
        raise ValueError("need --mesh or --usd")

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    # FOUND BY: catfish_fish001 got REJECTED (and lake_sturgeon_fish009 got rejected once) for
    # a "dark hole" that turned out to be a normal, minor geometric crease rendering as flat
    # black in "MATERIAL" (no-texture) mode -- confirmed by re-rendering WITH texture: the mark
    # blends into the fish's own naturally dark coloring and does not read as a defect at all.
    # The critic must see what the FINISHED, textured asset actually looks like, not an
    # artificially flat-shaded proxy that exaggerates every minor concavity into looking like a
    # hole. Always use TEXTURE color (with a neutral-strength light so it isn't washed out).
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.studiolight_intensity = 1.4  # a bit brighter so dark-fish textures
    # (e.g. catfish) aren't rendered near-black, which would ITSELF make real defects hard to see

    from mathutils import Vector
    depsgraph = bpy.context.evaluated_depsgraph_get()
    allpts = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        eo = o.evaluated_get(depsgraph)
        me = eo.to_mesh()
        for v in me.vertices:
            allpts.append(o.matrix_world @ v.co)
        eo.to_mesh_clear()
    assert allpts, "no visible mesh geometry found"
    mn = Vector((min(p.x for p in allpts), min(p.y for p in allpts), min(p.z for p in allpts)))
    mx = Vector((max(p.x for p in allpts), max(p.y for p in allpts), max(p.z for p in allpts)))
    center = (mn + mx) * 0.5
    diag = max((mx - mn)[i] for i in range(3)) * 1.9 + 0.02

    for c in list(bpy.data.objects):
        if c.type == "CAMERA":
            bpy.data.objects.remove(c, do_unlink=True)

    bpy.ops.object.camera_add(location=(center.x, center.y - diag, center.z + diag * 0.35),
                               rotation=(math.radians(68), 0, 0))
    cam_persp = bpy.context.active_object
    cam_persp.data.type = "ORTHO"
    cam_persp.data.ortho_scale = diag * 1.05

    bpy.ops.object.camera_add(location=(center.x, center.y, center.z + diag), rotation=(0, 0, 0))
    cam_top = bpy.context.active_object
    cam_top.data.type = "ORTHO"
    cam_top.data.ortho_scale = diag

    bpy.ops.object.camera_add(location=(center.x + diag, center.y, center.z),
                               rotation=(math.radians(90), 0, math.radians(90)))
    cam_side = bpy.context.active_object
    cam_side.data.type = "ORTHO"
    cam_side.data.ortho_scale = diag

    for cam, tag in ((cam_persp, "persp"), (cam_top, "top"), (cam_side, "side")):
        scene.camera = cam
        scene.render.filepath = f"{out_prefix}_{tag}.png"
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath, flush=True)


if __name__ == "__main__":
    main()
