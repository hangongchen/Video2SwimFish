"""Build a supervised fine-tuning (SFT) dataset for the Actor VLM from the 14 manually
constructed ground-truth fish skeletons in fish_skeleton_dataset/ (paper Section 3.5:
"a Fish Skeleton Dataset... fourteen controllable fish models... serving as fish-specific
fine-tuning examples for the actor").

For each fish, its human-placed bones are added ONE AT A TIME in head-to-tail order (all
14 templates have length along X with the head at +X, confirmed via index.json). At step i
the articulation contains bones [0..i-1]; that PARTIAL state is rendered (top/front/side,
via the same renderer.py code path used at inference time) and paired with the action JSON
that places bone i. This reproduces exactly the (rendered views, current articulation) ->
action supervision the real actor sees during construction, using real human placements as
the labels.

Ground-truth position/size come directly from each bone's bbox_world in skeleton.json
(min/max corners -> center/size) -- these bones have no rotation (rotation_euler is [0,0,0]
in every sampled record), so bbox_world.size matches how the real "add" action interprets
`size` (Blender's Object.dimensions, applied with rotation reset to zero -- see
articulation.py). The actor is not asked for joint parameters at all: every joint is driven
by a fixed physics preset applied later in the pipeline (add_isaac_physics_to_usd.py), so
there is nothing to supervise there.

Run under Blender:
    blender -b --python video2swimfish/finetune/build_sft_dataset.py -- --out_dir outputs_sft/sft_data \
        [--val_fish roach,trevally] [--enlarge]
By default the renders have NO bone-highlight dilation (this is how the shipped actor_lora_v2 training
images were made; the dilation was added to renderer.py later and is always on at inference time).
`--enlarge` opts in to the inference-style views.
Produces rig_sft.jsonl (74 rows = 12 train fish) and rig_sft_val.jsonl (11 rows = roach + trevally);
ground truth comes from data/fish_skeleton_dataset/<key>/skeleton.json + <key>.blend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINETUNE_DIR.parents[1]))      # package root (runs under Blender's python too)
import v2sf_paths as P  # noqa: E402

DATASET_DIR = P.SKELETON_DATASET                      # data/fish_skeleton_dataset (14 hand-built fish)
ARTICULATION_DIR = P.ARTICULATION_DIR
PIPE_SCRIPTS = P.PIPE_SCRIPTS

sys.path.insert(0, str(PIPE_SCRIPTS))
sys.path.insert(0, str(ARTICULATION_DIR))

import generate_auto_skeleton_blend as gask  # noqa: E402
from actor import ACTOR_PROMPT_TEMPLATE  # noqa: E402
from skeleton_template import template_library_prompt_str  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {"out_dir": str(P.ROOT / "outputs_sft" / "sft_data"), "val_fish": "roach,trevally", "enlarge": False}
    i = 0
    while i < len(argv):
        if argv[i] == "--out_dir":
            args["out_dir"] = argv[i + 1]
            i += 2
        elif argv[i] == "--val_fish":
            args["val_fish"] = argv[i + 1]
            i += 2
        elif argv[i] == "--enlarge":       # opt in to the inference-time bone-highlight dilation
            args["enlarge"] = True
            i += 1
        else:
            i += 1
    return args


TARGET_LENGTH_M = 0.25
# The 14 templates are hand-built in arbitrary Blender units (e.g. pumpkinseed length ~14
# units) -- copying their raw position/size numbers into training labels would teach the
# actor to output values on the wrong numeric scale, since at REAL inference time the mesh
# is scaled to its true measured length in meters (~0.1-0.5 m, see articulation.py's
# find_real_length_m). Rescaling every template to one representative real-world length
# (a simplification -- a species-accurate lookup would be better, but is not needed for
# this bootstrap fine-tune) keeps train/test numeric magnitudes comparable.


def normalize_position_size(position, size, skin_center, scale):
    pos = [(position[a] - skin_center[a]) * scale for a in range(3)]
    sz = [size[a] * scale for a in range(3)]
    return pos, sz


ENLARGE = False   # set from --enlarge in main(); default = how the shipped actor_lora_v2 data was rendered


def render_state(skin, all_bones, visible_bones, out_prefix):
    import bpy
    sys.path.insert(0, str(ARTICULATION_DIR))
    import renderer as rmod

    for b in all_bones:
        b.hide_render = b not in visible_bones
    bpy.context.view_layer.update()

    tmp_blend = f"{out_prefix}_scene.blend"
    # renderer.render_views shells out to a fresh Blender process, so hide_render flags
    # must be persisted to disk first, then restored to True (visible) for the caller.
    bpy.ops.wm.save_as_mainfile(filepath=tmp_blend)
    result = rmod.render_views(tmp_blend, out_prefix, enlarge=ENLARGE)
    for b in all_bones:
        b.hide_render = False
    return result["views"]


def build_one_fish(key: str, entry: dict, out_dir: Path) -> list[dict]:
    import bpy

    blend_path = DATASET_DIR / key / f"{key}.blend"
    skel_json = json.loads((DATASET_DIR / key / "skeleton.json").read_text())
    bones_gt = skel_json["skeleton_objects"]
    if not bones_gt:
        print(f"[build_sft_dataset] {key}: no skeleton_objects, skipping", flush=True)
        return []

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    for o in data_to.objects:
        if o is not None:
            bpy.context.collection.objects.link(o)
    bpy.context.view_layer.update()

    scene_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    gt_names = {b["object_name"] for b in bones_gt}
    skin_candidates = [o for o in scene_meshes if o.name not in gt_names]
    if not skin_candidates:
        print(f"[build_sft_dataset] {key}: could not isolate a skin mesh, skipping", flush=True)
        return []
    skin = max(skin_candidates, key=lambda o: gask.world_bbox_for_objects([o])["size"].x
                                              * gask.world_bbox_for_objects([o])["size"].y
                                              * gask.world_bbox_for_objects([o])["size"].z)
    bone_objs = [o for o in scene_meshes if o.name in gt_names]

    def bone_center_x(rec):
        return (rec["bbox_world"]["min"][0] + rec["bbox_world"]["max"][0]) / 2.0

    bones_sorted = sorted(bones_gt, key=bone_center_x, reverse=True)  # head (+X) first
    n = len(bones_sorted)

    skin_bbox = skel_json["fish_mesh"]["bbox_world"]
    skin_center = [(skin_bbox["min"][a] + skin_bbox["max"][a]) / 2.0 for a in range(3)]
    raw_length = skin_bbox["size"][0]
    scale = TARGET_LENGTH_M / raw_length if raw_length > 1e-9 else 1.0

    render_dir = out_dir / "images" / key
    render_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    visible_so_far: list = []
    for i, rec in enumerate(bones_sorted):
        prefix = str(render_dir / f"step{i:02d}")
        views = render_state(skin, bone_objs, list(visible_so_far), prefix)

        current_bones_json = []
        for j in range(i):
            raw_pos = [(bones_sorted[j]["bbox_world"]["min"][a] + bones_sorted[j]["bbox_world"]["max"][a]) / 2.0
                       for a in range(3)]
            raw_size = [bones_sorted[j]["bbox_world"]["max"][a] - bones_sorted[j]["bbox_world"]["min"][a]
                        for a in range(3)]
            pos, sz = normalize_position_size(raw_pos, raw_size, skin_center, scale)
            current_bones_json.append({
                "bone_id": j,
                "template": key,
                "position": [round(v, 5) for v in pos],
                "size": [round(v, 5) for v in sz],
                "object_name": f"bone_{j}",
            })

        bbox = rec["bbox_world"]
        raw_position = [(bbox["min"][a] + bbox["max"][a]) / 2.0 for a in range(3)]
        raw_size = [bbox["max"][a] - bbox["min"][a] for a in range(3)]
        position, size = normalize_position_size(raw_position, raw_size, skin_center, scale)
        action = {
            "action": "add",
            "bone_id": None,
            "template": key,
            "position": [round(v, 5) for v in position],
            "size": [round(v, 5) for v in size],
            "reasoning": (
                f"Adding bone {i + 1}/{n} of the {key}-shaped skeleton, continuing "
                f"{'from the head' if i == 0 else 'toward the tail'} to keep continuous "
                f"coverage along the body's longitudinal axis."
            ),
        }

        prompt = ACTOR_PROMPT_TEMPLATE.format(
            current_articulation_json=json.dumps({"bones": current_bones_json}),
            critic_feedback=(
                "(none, this is the first iteration)" if i == 0
                else "Previous bone accepted (fully inside the body, aligned with the "
                     "midline). Continue extending the skeleton toward the tail."
            ),
            template_library=template_library_prompt_str(),
        )

        rows.append({
            "fish_key": key,
            "step": i,
            "images": [views["top"], views["front"], views["side"]],
            "prompt": prompt,
            "response": json.dumps(action, indent=2),
        })

        visible_so_far.append(bone_objs[[b.name for b in bone_objs].index(rec["object_name"])])

    print(f"[build_sft_dataset] {key}: built {len(rows)} rows ({n} bones)", flush=True)
    return rows


def main():
    global ENLARGE
    args = parse_args()
    ENLARGE = bool(args["enlarge"])
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    val_fish = set(f.strip() for f in args["val_fish"].split(",") if f.strip())

    index = json.loads((DATASET_DIR / "index.json").read_text())
    train_rows, val_rows = [], []
    for entry in index["fish"]:
        key = entry["key"]
        rows = build_one_fish(key, entry, out_dir)
        if key in val_fish:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    train_path = out_dir / "rig_sft.jsonl"
    val_path = out_dir / "rig_sft_val.jsonl"
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")

    print(f"[build_sft_dataset] wrote {len(train_rows)} train rows -> {train_path}", flush=True)
    print(f"[build_sft_dataset] wrote {len(val_rows)} val rows -> {val_path} "
          f"(held-out fish: {sorted(val_fish)})", flush=True)


if __name__ == "__main__":
    main()
