"""Render the current articulation K^(k) (skin + bone mesh objects) from top/front/side
views for the actor and critic VLM prompts (paper Section 3.2 step 1 of both the actor and
the critic: "render current mesh+articulation ... from multiple views").

Dual-mode, like articulation.py:
  - `render_views(...)` is the plain-Python entry point used by run_auto_construction.py --
    it shells out to Blender (bpy can't be imported in a normal python process here).
  - The `__main__` block is what actually runs UNDER Blender and does the rendering.

Adapts the already-validated camera/ortho-projection pattern from
Video2SwimFish/scripts/render_mesh_views.py, but with a DIFFERENT shading goal: that script
needed full texture to avoid mistaking normal dark coloring for a mesh hole (a lesson from
earlier this session); this one needs the OPPOSITE -- bones must read as visually distinct
from the skin, and the skin must be seen through (x-ray) so internal bone placement is
actually visible, which plain texture shading cannot show. Skin is rendered as a translucent
neutral gray shell; every bone object is rendered solid red.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2sf_paths as P  # noqa: E402


def render_views(blend_path: str, out_prefix: str, enlarge: bool | None = None) -> dict:
    """Runs Blender headless on `blend_path` and writes <out_prefix>_top.png,
    _front.png, _side.png. Returns {"views": {tag: path}}.

    `enlarge` (default True; env V2SF_ENLARGE_BONES=0 flips the default) applies the bone-highlight
    dilation to the PNGs. NOTE: the actor's SFT training images were rendered WITHOUT it (it was
    added afterwards), the inference-time views (run_auto_construction) always have it; see README."""
    if enlarge is None:
        enlarge = os.environ.get("V2SF_ENLARGE_BONES", "1") != "0"
    cmd = [P.blender_bin(), "-b", str(blend_path), "--python", str(Path(__file__).resolve()),
           "--", "--out_prefix", str(out_prefix)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"renderer.py failed (exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    views = {tag: f"{out_prefix}_{tag}.png" for tag in ("top", "front", "side")}
    missing = [t for t, p in views.items() if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"renderer.py did not produce views {missing}; stdout:\n{r.stdout[-2000:]}")
    if enlarge:
        for p in views.values():
            _enlarge_bone_highlights(p)
    return {"views": views}


def _enlarge_bone_highlights(png_path: str, dilate_px: int = 6) -> None:
    """BUG FIX: even with a correctly-colored, correctly-lit bone material, a real bone (a few
    mm-1.5cm) rendered at true scale inside an 18cm fish is only a handful of PIXELS across in
    a whole-body view (measured directly: ~200 of 810,000 pixels) -- no amount of color/emission
    tuning fixes that, since the underlying problem is SIZE, not brightness. This is an
    image-space fix instead of a 3D one: find every bone-colored pixel (the render pipeline
    only ever paints bones this orange -- see bone_mat below -- so any pixel with this hue is
    unambiguously a bone, never the skin or background) and morphologically DILATE that region
    by `dilate_px` so each bone becomes a visibly large, unmistakable solid marker, then paint
    the dilated region a clean, fully-saturated highlight color (not just a light tint) so it
    reads clearly even after JPEG compression or at small on-screen size."""
    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
    except ImportError:
        # running inside Blender's python (e.g. build_sft_dataset.py --enlarge): no PIL/scipy there,
        # so hand the PNG to the VLM python
        r = subprocess.run([P.VLM_PYTHON, str(Path(__file__).resolve()), "--enlarge", str(png_path),
                            "--dilate_px", str(dilate_px)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bone-highlight enlargement failed via {P.VLM_PYTHON}:\n{r.stderr[-1500:]}")
        return

    im = Image.open(png_path).convert("RGB")
    arr = np.array(im)
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    bone_mask = (r - b) > 25  # bone_mat is orange (high R, low B); skin/background are neutral gray

    if not bone_mask.any():
        return

    struct = iterate_structure(generate_binary_structure(2, 1), dilate_px)
    dilated = binary_dilation(bone_mask, structure=struct)

    highlight = np.array([255, 130, 20], dtype=np.uint8)
    arr[dilated] = highlight
    Image.fromarray(arr).save(png_path)


def _blender_main():
    import math

    import bpy

    argv = sys.argv[sys.argv.index("--") + 1:]
    out_prefix = argv[argv.index("--out_prefix") + 1]

    scene = bpy.context.scene
    # BUG FIX: BLENDER_WORKBENCH's "X-Ray" (show_xray/xray_alpha) is a GLOBAL viewport dimmer,
    # not real per-object transparency -- it faded the small red bones by the same 35% as the
    # skin, so on a real multi-bone scene the bones collapsed into an indistinguishable smudge
    # (confirmed by inspecting the actual renders this fed to the actor/critic: individual
    # bones were not visually separable at all). Switched to BLENDER_EEVEE with real per-object
    # materials instead: the skin gets genuine alpha-blend transparency (surface_render_method
    # = BLENDED) while the bones stay fully opaque + slightly emissive, so they read as solid
    # shapes seen THROUGH a translucent skin rather than everything fading together.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    # BUG FIX: Blender 5.0's default color-management view transform is AgX, which is designed
    # to compress/desaturate bright values for a filmic look -- direct pixel sampling of an
    # actual rendered image showed the bone material (set at pure saturated orange + emission)
    # coming out as a washed-out pale tan (163,133,117) rather than a vivid orange, because AgX
    # rolled off the bright emission. 'Standard' is a direct linear-to-sRGB mapping with no
    # rolloff, so the bone's actual set color reaches the pixel undistorted.
    scene.view_settings.view_transform = "Standard"

    from mathutils import Vector

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    assert mesh_objs, "no mesh objects in .blend"
    bone_objs = [o for o in mesh_objs if o.name.startswith("bone_")]
    skin_objs = [o for o in mesh_objs if o not in bone_objs]
    # fall back to largest-volume-object-is-skin if naming didn't apply (e.g. a bootstrap
    # scene where the skin hasn't been renamed yet)
    if not skin_objs:
        skin_objs = [max(mesh_objs, key=lambda o: (o.dimensions.x * o.dimensions.y * o.dimensions.z))]
        bone_objs = [o for o in mesh_objs if o not in skin_objs]

    skin_mat = bpy.data.materials.new("skin_xray_mat")
    skin_mat.use_nodes = True
    bsdf = skin_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.8, 1.0)
    bsdf.inputs["Alpha"].default_value = 0.18
    bsdf.inputs["Roughness"].default_value = 0.6
    skin_mat.surface_render_method = "BLENDED"
    skin_mat.show_transparent_back = True
    skin_mat.use_backface_culling = False

    bone_mat = bpy.data.materials.new("bone_solid_mat")
    bone_mat.use_nodes = True
    bsdf_b = bone_mat.node_tree.nodes["Principled BSDF"]
    bsdf_b.inputs["Base Color"].default_value = (1.0, 0.25, 0.0, 1.0)
    bsdf_b.inputs["Alpha"].default_value = 1.0
    # strong emission so bones read as a bright highlight, not just "opaque vs translucent
    # skin" -- 0.5 (the first attempt) still let the ambient/sun lighting wash them toward the
    # skin's own brightness in a crowded scene; boosted well past "lit surface" into "glows"
    if "Emission Color" in bsdf_b.inputs:
        bsdf_b.inputs["Emission Color"].default_value = (1.0, 0.35, 0.0, 1.0)
        bsdf_b.inputs["Emission Strength"].default_value = 1.4

    for o in skin_objs:
        o.data.materials.clear()
        o.data.materials.append(skin_mat)
    for o in bone_objs:
        o.data.materials.clear()
        o.data.materials.append(bone_mat)

    # dark neutral world fill (matches the old renderer's background contrast) + two low-power
    # suns from opposite sides so every one of the 3 ortho views gets even, near-shadowless
    # light -- harsh single-direction shadows would hide bone edges just as badly as no light
    world = bpy.data.worlds.new("flat_fill") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.16, 0.16, 0.18, 1.0)
        bg.inputs[1].default_value = 1.0

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    sun1 = bpy.context.active_object
    sun1.data.energy = 1.2
    sun1.rotation_euler = (math.radians(35), math.radians(20), 0)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, -5))
    sun2 = bpy.context.active_object
    sun2.data.energy = 0.6
    sun2.rotation_euler = (math.radians(-35), math.radians(-20), 0)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    allpts = []
    for o in mesh_objs:
        eo = o.evaluated_get(depsgraph)
        me = eo.to_mesh()
        for v in me.vertices:
            allpts.append(o.matrix_world @ v.co)
        eo.to_mesh_clear()
    mn = Vector((min(p.x for p in allpts), min(p.y for p in allpts), min(p.z for p in allpts)))
    mx = Vector((max(p.x for p in allpts), max(p.y for p in allpts), max(p.z for p in allpts)))
    center = (mn + mx) * 0.5
    diag = max((mx - mn)[i] for i in range(3)) * 1.6 + 0.02

    for c in list(bpy.data.objects):
        if c.type == "CAMERA":
            bpy.data.objects.remove(c, do_unlink=True)

    # fish convention established earlier this session: length along X, head at +X
    # top: looking down -Z (dorsal view)
    bpy.ops.object.camera_add(location=(center.x, center.y, center.z + diag), rotation=(0, 0, 0))
    cam_top = bpy.context.active_object
    cam_top.data.type = "ORTHO"
    cam_top.data.ortho_scale = diag

    # side: looking along +Y -> -Y (lateral profile view)
    bpy.ops.object.camera_add(location=(center.x, center.y - diag, center.z),
                               rotation=(math.radians(90), 0, 0))
    cam_side = bpy.context.active_object
    cam_side.data.type = "ORTHO"
    cam_side.data.ortho_scale = diag

    # front: looking along -X from ahead of the head (anterior view)
    bpy.ops.object.camera_add(location=(center.x + diag, center.y, center.z),
                               rotation=(math.radians(90), 0, math.radians(90)))
    cam_front = bpy.context.active_object
    cam_front.data.type = "ORTHO"
    cam_front.data.ortho_scale = diag

    for cam, tag in ((cam_top, "top"), (cam_front, "front"), (cam_side, "side")):
        scene.camera = cam
        scene.render.filepath = f"{out_prefix}_{tag}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[renderer] wrote {scene.render.filepath}", flush=True)


if __name__ == "__main__":
    if "--enlarge" in sys.argv:            # plain-python helper mode (see _enlarge_bone_highlights)
        _dpx = int(sys.argv[sys.argv.index("--dilate_px") + 1]) if "--dilate_px" in sys.argv else 6
        _enlarge_bone_highlights(sys.argv[sys.argv.index("--enlarge") + 1], _dpx)
    else:
        _blender_main()
