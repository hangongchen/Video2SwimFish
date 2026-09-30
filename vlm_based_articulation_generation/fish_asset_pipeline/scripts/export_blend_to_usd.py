import argparse
import os

import bpy


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Blender scene to USD in background mode.")
    parser.add_argument("--output", required=True, help="Output USD/USDA path.")
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    try:
        bpy.ops.wm.usd_export(
            filepath=args.output,
            selected_objects_only=False,
            export_animation=False,
            export_materials=True,
            export_meshes=True,
            export_lights=True,
            export_cameras=True,
            export_textures=True,           # write packed images to a textures/ dir
            overwrite_textures=True,
            relative_paths=True,
        )
    except TypeError:
        bpy.ops.wm.usd_export(filepath=args.output, selected_objects_only=False, export_animation=False)


if __name__ == "__main__":
    main()
