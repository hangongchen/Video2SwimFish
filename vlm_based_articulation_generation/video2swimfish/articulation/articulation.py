"""Articulation data structure (paper Section 3.2: K^(k) is the articulation at iteration k)
and USD export for the final K*.

Two independent halves in this one file, per the requested module layout:

  1. Plain-Python `Bone`/`Articulation` dataclasses + JSON (de)serialization -- used by the
     orchestrator (run_auto_construction.py) and by actor.py/critic.py to render the current
     articulation as structured text for the VLM prompts. No bpy dependency.

  2. `export_articulation_to_usd(...)` -- once the loop converges on K*, its .blend already
     contains the skin mesh + one mesh object per bone (built up by the Blender-side apply-
     action step in this same file's __main__). extract_fish_skeleton.py is ALREADY fully
     generic here (verified by reading it): it picks the largest-volume mesh as the fish body
     and treats every OTHER visible mesh object as a skeleton bone, with no naming assumptions.
     So K*'s .blend can be fed directly into the EXISTING deterministic USD-authoring chain
     (extract_fish_skeleton.py -> export_blend_to_usd.py -> add_isaac_physics_to_usd.py ->
     convert_usda_to_usd.py -> cook_deformable_isaac.py) unchanged -- no new USD-writing code.

The Blender-side "apply one actor action to the current .blend, then run geometric
verification" step lives in this file's __main__ (run under Blender: see ARTICULATION_APPLY
usage at the bottom), since that is where the articulation data structure actually gets
mutated into real geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# This file runs both as a normal module AND under Blender's python (`blender -b --python
# articulation.py -- ...`), so it locates the package root from __file__ (no env var needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2sf_paths as P  # noqa: E402

PIPE_ROOT = P.PIPE_ROOT
PIPE_SCRIPTS = P.PIPE_SCRIPTS
CFG = P.USD_PHYSICS_CFG
SPECIES_MANIFOLD_ROOT = P.SPECIES_MANIFOLD_ROOT
ISAAC_PY = P.ISAAC_PYTHON     # None unless ISAAC_PYTHON is set: only the optional FEM cook needs it


def find_real_length_m(fish_id: str | None):
    """Look up the tank-calibrated real body length for `fish_id` (e.g. "catfish_fish001")
    from <SPECIES_MANIFOLD_ROOT>/<species>/fish_sizes.json, the measurement table that
    scripts/measure_fish_size.py writes. Mirrors the --species/--fish_id lookup in
    scripts/run_pipeline.py. Returns
    (length_m or None, source_str)."""
    if not fish_id:
        return None, "no fish_id given"
    m = re.match(r"^(.*)_fish(\d+)$", fish_id)
    if not m:
        return None, f"fish_id '{fish_id}' doesn't match '<species>_fish<NNN>'"
    species, num = m.group(1), m.group(2)
    sizes_path = SPECIES_MANIFOLD_ROOT / species / "fish_sizes.json"
    if not sizes_path.exists():
        return None, f"no sizes file at {sizes_path}"
    data = json.loads(sizes_path.read_text())
    entry = data.get("per_fish", {}).get(num)
    if not entry:
        return None, f"fish key '{num}' not in {sizes_path}"
    return float(entry["length_m"]), str(sizes_path)


# --------------------------------------------------------------------------- data structure
@dataclass
class Bone:
    # NOTE: there is deliberately NO per-bone joint configuration here. The actor is never asked
    # for joint parameters. Joints are authored later, identically for every fish, by
    # fish_asset_pipeline/scripts/add_isaac_physics_to_usd.py:create_joint from
    # fish_asset_pipeline/configs/usd_physics_defaults.json: 9 D6 joints, +-15 deg limits on
    # rotX/Y/Z, and NO DriveAPI at all (the "drive" block in that json is never read).
    # The wider +-45 deg limits and the PD drive (kp=120, kd=6 in the paper) are applied by the
    # SIMULATION side when it loads the USD (swimming_policy_training: salmon_swim_env.py
    # `_prepare_env_assets` + the task cfgs). See README.md, "Joint preset: USD vs training".
    bone_id: int
    template: str
    position: list[float]      # [x, y, z], world space
    size: list[float]          # [l, w, h]
    object_name: str | None = None  # assigned once realized as a Blender object


@dataclass
class Articulation:
    bones: list[Bone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"bones": [asdict(b) for b in self.bones]}

    @staticmethod
    def from_dict(d: dict) -> "Articulation":
        return Articulation(bones=[Bone(**b) for b in d.get("bones", [])])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "Articulation":
        return Articulation.from_dict(json.loads(Path(path).read_text()))

    def next_bone_id(self) -> int:
        return (max((b.bone_id for b in self.bones), default=-1)) + 1

    def find(self, bone_id: int) -> Bone | None:
        for b in self.bones:
            if b.bone_id == bone_id:
                return b
        return None

    def apply_action(self, action: dict) -> "Articulation":
        """Pure bookkeeping copy-on-write: returns a NEW Articulation with `action` applied.
        The actual Blender geometry sync happens separately (see __main__ below) -- this is
        just K^(k+1) = T(K^(k), a^(k)) at the data-structure level."""
        bones = [Bone(**asdict(b)) for b in self.bones]
        act = action.get("action")
        if act == "add":
            requested_id = action.get("bone_id")
            new_id = requested_id if requested_id is not None else self.next_bone_id()
            bones.append(Bone(
                bone_id=new_id,
                template=action["template"],
                position=list(action["position"]),
                size=list(action["size"]),
                object_name=f"bone_{new_id}",
            ))
        elif act == "remove":
            bones = [b for b in bones if b.bone_id != action.get("bone_id")]
        elif act in ("reposition", "resize"):
            for b in bones:
                if b.bone_id == action.get("bone_id"):
                    if act == "reposition" and action.get("position") is not None:
                        b.position = list(action["position"])
                    if act == "resize" and action.get("size") is not None:
                        b.size = list(action["size"])
        return Articulation(bones=bones)


# --------------------------------------------------------------------------- USD export
def export_articulation_to_usd(final_blend: str, out_dir: str, hex_res: int = 10,
                                 bones: list | None = None, fish_length_m: float | None = None,
                                 fish_height_m: float | None = None,
                                 fish_thickness_m: float | None = None) -> dict:
    """Run the EXISTING deterministic USD-authoring chain on K*'s .blend (skin + bone mesh
    objects already placed and geometrically verified by the actor-critic loop). Mirrors
    Video2SwimFish/scripts/run_pipeline.py's own steps 3-8, reusing the identical scripts.
    No separate real-length scaling step here (unlike run_pipeline.py's step 3): the skin
    was already scaled to the tank-measured real length during this loop's bootstrap.

    If `bones` + the fish's real length/height/thickness are given, a final step overwrites
    add_isaac_physics_to_usd.py's own per-bone mass (that EXISTING, unmodified script splits
    its total bone-mass budget EQUALLY across every bone regardless of size -- confirmed by
    inspecting a real export: all N bones got the identical mass) with a volume-proportional
    share of an ellipsoid estimate of the whole fish's mass, so a bigger bone gets a bigger
    share of the mass instead of an equal slice. See apply_volume_proportional_mass."""
    out_dir = Path(out_dir)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    report = {}

    def run(cmd, log_name, step):
        log_path = work / log_name
        with open(log_path, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"[{step}] failed (exit {r.returncode}) -- see {log_path}\n"
                                 f"{log_path.read_text()[-2000:]}")

    skel_dir = work / "skeleton_json"
    BLENDER, BPY = P.blender_bin(), P.blender_py()
    run([str(BLENDER), "-b", final_blend, "--python", str(PIPE_SCRIPTS / "extract_fish_skeleton.py"),
         "--", "--output", str(skel_dir)], "step1_extract.log", "extract-skeleton")
    skel_json = skel_dir / "skeleton.json"

    base_usda = work / "usd" / "base_export.usda"
    run([str(BLENDER), "-b", final_blend, "--python", str(PIPE_SCRIPTS / "export_blend_to_usd.py"),
         "--", "--output", str(base_usda)], "step2_export.log", "export-usd")
    report["textures_in_export"] = base_usda.read_text().count("UsdUVTexture") if base_usda.exists() else 0

    final_usda = out_dir / "K_final.usda"
    run([str(BPY), str(PIPE_SCRIPTS / "add_isaac_physics_to_usd.py"),
         "--input-usd", str(base_usda), "--skeleton-json", str(skel_json),
         "--config", str(CFG), "--output-usd", str(final_usda)], "step3_physics.log", "physics")

    final_usd = out_dir / "K_final.usd"
    run([str(BPY), str(PIPE_SCRIPTS / "convert_usda_to_usd.py"),
         "--input", str(final_usda), "--output", str(final_usd)], "step4_convert.log", "convert")

    if not ISAAC_PY:
        # optional stage: without Isaac Sim the USD is still a valid articulation (bones, D6
        # joints, attachments, mass) but its deformable skin is not cooked into a FEM mesh
        report["fem_cooked"] = False
        report["fem_cook_error"] = "ISAAC_PYTHON not set: FEM cook skipped (see README, 'Isaac Sim is optional')"
    else:
        try:
            run([str(ISAAC_PY), str(PIPE_SCRIPTS / "cook_deformable_isaac.py"),
                 "--in", str(final_usd), "--out", str(final_usd), "--hex-resolution", str(hex_res)],
                "step5_fem_cook.log", "fem-cook")
            report["fem_cooked"] = True
        except RuntimeError as e:
            report["fem_cooked"] = False
            report["fem_cook_error"] = str(e)

    report["final_usd"] = str(final_usd)
    report["final_usda"] = str(final_usda)

    if bones and fish_length_m and fish_height_m and fish_thickness_m:
        try:
            report["mass_distribution"] = apply_volume_proportional_mass(
                str(final_usd), bones, fish_length_m, fish_height_m, fish_thickness_m)
        except Exception as e:  # noqa: BLE001 -- mass override is an enhancement, never fail the export over it
            report["mass_distribution_error"] = f"{type(e).__name__}: {e}"

    return report


def apply_volume_proportional_mass(usd_path: str, bones: list, fish_length_m: float,
                                     fish_height_m: float, fish_thickness_m: float,
                                     density_kg_per_m3: float = 1000.0) -> dict:
    """Overwrite each bone prim's (/root/skeleton/bone_<id>) uniform mass -- set by the
    EXISTING, unmodified add_isaac_physics_to_usd.py as total_bone_mass / len(bones), i.e.
    every bone gets the identical mass regardless of its own size (confirmed by inspecting a
    real exported USD: all bones had mass=0.1453 kg) -- with a physically more sensible split:
    total fish mass estimated as an ELLIPSOID from the fish's own real measured
    length/height/thickness (semi-axes L/2, H/2, T/2; density 1000 kg/m^3, matching the
    deterministic pipeline's own water_density_kg_per_m3 convention in
    fish_asset_pipeline/configs/usd_physics_defaults.json), distributed to each bone in
    proportion to that bone's own box volume (bone.size is [length,width,height] in meters)
    out of the summed volume of all bones. Runs via BPY (Blender's bundled python has `pxr`;
    the plain simfishlib conda env this orchestrator otherwise runs under does not)."""
    bones_payload = [{"bone_id": b.bone_id, "size": b.size} for b in bones]
    out_report = Path(usd_path).with_suffix(".mass_report.json")
    cmd = [P.blender_py(), str(Path(__file__).resolve()), "--", "--apply_mass",
           "--usd", str(usd_path), "--bones_json", json.dumps(bones_payload),
           "--length_m", str(fish_length_m), "--height_m", str(fish_height_m),
           "--thickness_m", str(fish_thickness_m), "--density", str(density_kg_per_m3),
           "--out_report", str(out_report)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out_report.exists():
        raise RuntimeError(f"apply_volume_proportional_mass failed (exit {r.returncode}):\n"
                             f"--- stdout (tail) ---\n{r.stdout[-3000:]}\n--- stderr (tail) ---\n{r.stderr[-3000:]}")
    return json.loads(out_report.read_text())


def _apply_mass_main():
    """Runs UNDER BPY (Blender's bundled python -- has `pxr`, no bpy/Blender-context needed
    here, this is pure USD-stage editing): see apply_volume_proportional_mass's docstring."""
    import math

    from pxr import UsdPhysics
    from pxr import Usd as _Usd

    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply_mass", action="store_true", help="dispatch flag, consumed here")
    ap.add_argument("--usd", required=True)
    ap.add_argument("--bones_json", required=True)
    ap.add_argument("--length_m", type=float, required=True)
    ap.add_argument("--height_m", type=float, required=True)
    ap.add_argument("--thickness_m", type=float, required=True)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--out_report", required=True)
    args = ap.parse_args(argv)

    bones = json.loads(args.bones_json)
    fish_volume_m3 = (4.0 / 3.0) * math.pi * (args.length_m / 2) * (args.height_m / 2) * (args.thickness_m / 2)
    total_fish_mass_kg = fish_volume_m3 * args.density

    bone_volumes = {b["bone_id"]: max(float(b["size"][0]) * float(b["size"][1]) * float(b["size"][2]), 1e-12)
                    for b in bones}
    total_bone_volume_m3 = sum(bone_volumes.values())

    stage = _Usd.Stage.Open(args.usd)
    report = {
        "fish_length_m": args.length_m, "fish_height_m": args.height_m, "fish_thickness_m": args.thickness_m,
        "fish_volume_m3_ellipsoid": fish_volume_m3, "density_kg_per_m3": args.density,
        "total_fish_mass_kg": total_fish_mass_kg, "total_bone_volume_m3": total_bone_volume_m3,
        "bones": {},
    }
    for bone_id, vol in bone_volumes.items():
        prim = stage.GetPrimAtPath(f"/root/skeleton/bone_{bone_id}")
        if not prim.IsValid():
            report["bones"][bone_id] = {"warning": "prim not found at /root/skeleton/bone_" + str(bone_id)}
            continue
        mass_kg = total_fish_mass_kg * (vol / total_bone_volume_m3)
        mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(float(mass_kg))
        report["bones"][str(bone_id)] = {"volume_m3": vol, "mass_kg": mass_kg}
    stage.GetRootLayer().Save()

    Path(args.out_report).write_text(json.dumps(report, indent=2))
    print(f"[articulation] volume-proportional mass applied -> {args.usd}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


# --------------------------------------------------------------------------- robust body geometry
def _slice_core(pts, n_bins: int = 24):
    """Thickness-weighted body core of one cross-section slice. Fins, scutes and barbels are
    THIN in z (or y) but can carry many vertices, so median/percentile estimators still drift
    toward them (catfish_fish006's dorsal fin pulled the median centerline up ~40% of the
    half-height). Weighting each y-strip by its z-thickness makes them nearly weightless.
    Returns (cy, cz, half_y, half_z) or None."""
    if len(pts) < 5:
        return None
    ys = [p.y for p in pts]
    y_lo, y_hi = min(ys), max(ys)
    if y_hi - y_lo < 1e-9:
        return None
    step = (y_hi - y_lo) / n_bins
    bins = [[] for _ in range(n_bins)]
    for p in pts:
        b = min(n_bins - 1, int((p.y - y_lo) / step))
        bins[b].append(p.z)
    thick = [(max(b) - min(b)) if len(b) >= 2 else 0.0 for b in bins]
    zmid = [((max(b) + min(b)) / 2.0) if len(b) >= 2 else 0.0 for b in bins]
    tmax = max(thick)
    if tmax <= 1e-9:
        return None
    w = [t for t in thick]
    ws = sum(w)
    cy = sum(wi * (y_lo + (i + 0.5) * step) for i, wi in enumerate(w)) / ws
    cz = sum(wi * zmid[i] for i, wi in enumerate(w)) / ws
    core = [i for i, t in enumerate(thick) if t >= 0.3 * tmax]
    half_y = (max(core) - min(core) + 1) * step / 2.0
    half_z = tmax / 2.0
    return cy, cz, half_y, half_z


def _robust_centerline(skin_verts, x_min: float, x_max: float, n: int = 201, smooth_frac: float = 0.06) -> list[dict]:
    """Body centerline that ignores fins/barbels/scutes: per slice the thickness-weighted core
    centre (see _slice_core), then a moving-average smoothing over `smooth_frac` of the body
    length."""
    L = x_max - x_min
    base = max(L * 0.01, 1e-5)
    xs, ys, zs = [], [], []
    for i in range(n):
        t = i / (n - 1)
        xq = x_max - t * L
        w = base
        pts = [v for v in skin_verts if abs(v.x - xq) < w]
        k = 0
        while len(pts) < 12 and k < 10:
            w *= 1.7
            pts = [v for v in skin_verts if abs(v.x - xq) < w]
            k += 1
        core = _slice_core(pts)
        if core:
            ys.append(core[0])
            zs.append(core[1])
        else:
            ys.append(ys[-1] if ys else 0.0)
            zs.append(zs[-1] if zs else 0.0)
        xs.append(xq)
    half = max(1, int(round(smooth_frac * n / 2)))

    def smooth(a):
        out = []
        for i in range(len(a)):
            lo, hi = max(0, i - half), min(len(a), i + half + 1)
            out.append(sum(a[lo:hi]) / (hi - lo))
        return out
    ys, zs = smooth(ys), smooth(zs)
    return [{"t": i / (n - 1), "x": xs[i], "y": ys[i], "z": zs[i]} for i in range(n)]


def _robust_half_extents(skin_verts, x: float, win: float):
    """Local body-core half-height / half-thickness at x (fins excluded, see _slice_core)."""
    loc = [v for v in skin_verts if abs(v.x - x) < win]
    core = _slice_core(loc)
    return (core[2], core[3]) if core else None


# --------------------------------------------------------------------------- Blender-side repair
def _repair_skeleton(skin, skin_bbox, gv, gask, action: dict) -> dict:
    """Deterministic post-pass run ONCE after the actor-critic loop ends (accepted, cycle-stopped
    or capped). It enforces the hard structural requirements the fine-tuned actor cannot be
    relied on for (it was trained on add-only sequences of GT skeletons that themselves cover
    only 38-64% of the body and sit slightly off-midline):
      1. every bone's center is snapped onto the real body centerline (fixes the systematic
         ventral offset seen on catfish_fish006: mean |offset| 0.36 of half-height);
      2. inner gaps wider than `gap_frac` of body length are filled with new box bones;
      3. the chain is extended toward the head and the tail until it reaches the points where
         the body cross-section drops below `head_area_frac`/`tail_area_frac` of the max
         cross-section (i.e. skull to caudal-fin base -- not snout tip / fin membrane / barbels);
      4. every new/moved bone goes through the same anisotropic containment shrink.
    Bone ordering for the joint chain is by x (add_isaac_physics_to_usd.ordered_bones sorts on
    center x), so new bones can take any free id."""
    import bpy
    from mathutils import Vector

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # extract_centerline.py lives next to this file
    from extract_centerline import compute_centerline, lookup_xyz, x_to_t  # noqa: E402

    gap_frac = float(action.get("gap_frac", 0.02))
    head_area_frac = float(action.get("head_area_frac", 0.15))
    tail_area_frac = float(action.get("tail_area_frac", 0.03))
    max_bones = int(action.get("max_bones", 22))

    skin_verts = [skin.matrix_world @ v.co for v in skin.data.vertices]
    L = skin_bbox["size"].x
    x_max, x_min = skin_bbox["max"][0], skin_bbox["min"][0]
    centerline = _robust_centerline(skin_verts, x_min, x_max, 201)
    win = max(L * 0.02, 1e-4)

    def local_half_extents(x):
        return _robust_half_extents(skin_verts, x, win)

    # cross-section "area" profile along the body -> where does the trunk really start/end
    xs_prof = [x_max - i * L / 100.0 for i in range(101)]
    areas = []
    for x in xs_prof:
        he = local_half_extents(x)
        areas.append(he[0] * he[1] if he else 0.0)
    a_max = max(areas) if areas else 1.0
    x_head_target = next((x for x, a in zip(xs_prof, areas) if a >= head_area_frac * a_max), x_max)
    x_tail_target = next((x for x, a in zip(reversed(xs_prof), reversed(areas)) if a >= tail_area_frac * a_max), x_min)

    bones = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name.startswith("bone_")]
    rep = {"snapped": [], "filled_spans": [], "added": [], "x_head_target": x_head_target,
           "x_tail_target": x_tail_target, "body_x_range": [x_min, x_max]}

    # 1. snap to centerline
    for o in bones:
        bb = gask.world_bbox_for_objects([o])
        cx, cy, cz = lookup_xyz(centerline, x_to_t(centerline, bb["center"].x))
        d = Vector((0.0, cy - bb["center"].y, cz - bb["center"].z))
        if d.length > 1e-6:
            o.location = o.location + d
            bpy.context.view_layer.update()
            rep["snapped"].append({"bone": o.name, "dy": d.y, "dz": d.z})
            gv.verify_bone(o, skin)

    def intervals():
        out = []
        for o in [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name.startswith("bone_")]:
            bb = gask.world_bbox_for_objects([o])
            out.append((bb["max"][0], bb["min"][0], o))
        return sorted(out, key=lambda t: -t[0])  # head first

    # drop bones that mostly overlap a neighbour along x (the cycling actor sometimes stacks
    # several bones on the same spot -- lake_sturgeon_fish009); keep the longer one
    removed = []
    changed = True
    while changed:
        changed = False
        iv = intervals()
        for (hi_a, lo_a, oa), (hi_b, lo_b, ob) in zip(iv, iv[1:]):
            overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
            shorter = min(hi_a - lo_a, hi_b - lo_b)
            if shorter > 0 and overlap > 0.5 * shorter:
                victim = oa if (hi_a - lo_a) < (hi_b - lo_b) else ob
                removed.append(victim.name)
                bpy.data.objects.remove(victim, do_unlink=True)
                changed = True
                break
    rep["removed_overlapping"] = removed

    iv = intervals()
    if iv:
        lengths = sorted(hi - lo for hi, lo, _ in iv)
        median_len = lengths[len(lengths) // 2]
        ratios_y, ratios_z = [], []
        for hi, lo, o in iv:
            he = local_half_extents((hi + lo) / 2.0)
            if he:
                bb = gask.world_bbox_for_objects([o])
                ratios_y.append(bb["size"].y / max(2 * he[0], 1e-6))
                ratios_z.append(bb["size"].z / max(2 * he[1], 1e-6))
        ratio_y = sorted(ratios_y)[len(ratios_y) // 2] if ratios_y else 0.4
        ratio_z = sorted(ratios_z)[len(ratios_z) // 2] if ratios_z else 0.15
    else:
        median_len, ratio_y, ratio_z = L * 0.06, 0.4, 0.15
    median_len = max(median_len, L * 0.06)  # actor bones can come out very short; keep segments fish-scaled

    def next_id():
        ids = [int(o.name.split("_")[-1]) for o in bpy.context.scene.objects
               if o.type == "MESH" and o.name.startswith("bone_") and o.name.split("_")[-1].isdigit()]
        return (max(ids) + 1) if ids else 0

    def add_box(cx_target, seg_len):
        cx, cy, cz = lookup_xyz(centerline, x_to_t(centerline, cx_target))
        he = local_half_extents(cx)
        if he is None:
            return None
        size = [max(seg_len * 0.95, 0.002), max(2 * he[0] * ratio_y, 0.002), max(2 * he[1] * ratio_z, 0.002)]
        bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
        o = bpy.context.active_object
        o.name = f"bone_{next_id()}"
        o.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        o.dimensions = size
        bpy.context.view_layer.update()
        v = gv.verify_bone(o, skin)
        if not v["fully_contained"]:
            # typically the snout/mouth region: the centerline point sits in the mouth cavity, so
            # no amount of shrinking helps -- drop the bone rather than export a protruding one
            bpy.data.objects.remove(o, do_unlink=True)
            return {"bone": None, "x": cx, "size_requested": size, "verification": v, "dropped": True}
        return {"bone": o.name, "x": cx, "size_requested": size, "verification": v}

    # 2+3. spans to fill: head margin, inner gaps, tail margin (head-first order)
    spans = []
    if iv:
        if x_head_target - iv[0][0] > gap_frac * L:
            spans.append((x_head_target, iv[0][0], "head"))
        for (hi_a, lo_a, _), (hi_b, lo_b, _) in zip(iv, iv[1:]):
            if lo_a - hi_b > gap_frac * L:
                spans.append((lo_a, hi_b, "inner"))
        if iv[-1][1] - x_tail_target > gap_frac * L:
            spans.append((iv[-1][1], x_tail_target, "tail"))
    else:
        spans.append((x_head_target, x_tail_target, "all"))

    # tail first (thrust comes from there), then inner gaps, head last -- so a bone cap never
    # starves the tail
    spans.sort(key=lambda t: {"tail": 0, "all": 0, "inner": 1, "head": 2}[t[2]])
    n_bones = len(iv)
    for hi, lo, kind in spans:
        span = hi - lo
        n = max(1, int(round(span / median_len)))
        seg = span / n
        added_here = []
        for i in range(n):
            if n_bones >= max_bones:
                break
            r = add_box(hi - seg * (i + 0.5), seg)
            if r and not r.get("dropped"):
                added_here.append(r)
                n_bones += 1
            elif r:
                rep.setdefault("dropped", []).append(r)
        rep["filled_spans"].append({"kind": kind, "x_hi": hi, "x_lo": lo, "span_frac": span / L, "n_added": len(added_here)})
        rep["added"].extend(added_here)

    # coverage after
    iv2 = intervals()
    union = 0.0
    cur = None
    for hi, lo, _ in sorted(iv2, key=lambda t: t[1]):
        if cur is None:
            cur = [lo, hi]
        elif lo <= cur[1]:
            cur[1] = max(cur[1], hi)
        else:
            union += cur[1] - cur[0]
            cur = [lo, hi]
    if cur:
        union += cur[1] - cur[0]
    rep["coverage_frac_after"] = union / L
    rep["n_bones_after"] = len(iv2)
    return rep


# --------------------------------------------------------------------------- Blender-side apply
def _pca_level_skin(skin, min_deg: float = 0.5, max_deg: float = 80.0) -> dict:
    """Rotate `skin` (Blender mesh object, world transform already canonicalized) so the principal
    axis of its vertex cloud is exactly +X. Minimal rotation about the axis perpendicular to
    (e1, +X); the sign of e1 is chosen so the head (+X side) stays on +X. Tilts above `max_deg`
    are left alone (something else is wrong then)."""
    import bpy
    import numpy as np
    from math import acos, degrees, radians
    from mathutils import Matrix, Vector

    V = np.array([tuple(skin.matrix_world @ v.co) for v in skin.data.vertices], dtype=np.float64)
    if len(V) < 100:
        return {"applied": False, "reason": "too_few_vertices"}
    c = V.mean(0)
    _, _, Wt = np.linalg.svd(V - c, full_matrices=False)
    e1 = Wt[0] / np.linalg.norm(Wt[0])
    if e1[0] < 0:
        e1 = -e1
    x = np.array([1.0, 0.0, 0.0])
    ang = degrees(acos(float(np.clip(np.dot(e1, x), -1.0, 1.0))))
    rep = {"tilt_deg_before": round(ang, 2), "principal_axis_before": [round(float(a), 4) for a in e1]}
    if ang < min_deg or ang > max_deg:
        rep.update(applied=False, reason="below_min" if ang < min_deg else "above_max")
        return rep
    axis = np.cross(e1, x)
    axis = axis / np.linalg.norm(axis)
    R = Matrix.Rotation(radians(ang), 4, Vector(axis.tolist()))
    T = Matrix.Translation(Vector(c.tolist())) @ R @ Matrix.Translation(-Vector(c.tolist()))
    skin.matrix_world = T @ skin.matrix_world
    bpy.context.view_layer.update()
    V2 = np.array([tuple(skin.matrix_world @ v.co) for v in skin.data.vertices], dtype=np.float64)
    _, _, Wt2 = np.linalg.svd(V2 - V2.mean(0), full_matrices=False)
    e1b = Wt2[0] / np.linalg.norm(Wt2[0]); e1b = e1b if e1b[0] >= 0 else -e1b
    rep.update(applied=True, rotation_axis=[round(float(a), 4) for a in axis],
               tilt_deg_after=round(degrees(acos(float(np.clip(e1b[0], -1, 1)))), 2))
    return rep


def _recenter_skin(skin) -> dict:
    """Translate `skin` so its world-space bbox centre is at the origin. Returns the shift applied."""
    import bpy
    from mathutils import Matrix, Vector
    xs = [skin.matrix_world @ v.co for v in skin.data.vertices]
    lo = Vector((min(p.x for p in xs), min(p.y for p in xs), min(p.z for p in xs)))
    hi = Vector((max(p.x for p in xs), max(p.y for p in xs), max(p.z for p in xs)))
    centre = (lo + hi) * 0.5
    if centre.length < 1e-6:
        return {"applied": False, "shift_m": [0.0, 0.0, 0.0]}
    skin.matrix_world = Matrix.Translation(-centre) @ skin.matrix_world
    bpy.context.view_layer.update()
    return {"applied": True, "shift_m": [round(-c, 5) for c in centre]}


def _blender_apply_action():
    """Runs UNDER BLENDER: apply one actor action to the current .blend (importing the raw
    GLB skin first if this is iteration 0 / an empty articulation), then run geometric
    verification, save, and write a JSON report. Reuses generate_auto_skeleton_blend.py's
    append_template_skeleton/world_bbox_for_objects and geometric_verification.verify_bone
    directly rather than reimplementing mesh/library handling."""
    import bpy

    sys.path.insert(0, str(PIPE_SCRIPTS))
    import generate_auto_skeleton_blend as gask  # noqa: E402

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import geometric_verification as gv  # noqa: E402

    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_blend", default=None, help="existing .blend, or omit for a fresh scene")
    ap.add_argument("--mesh_glb", default=None, help="raw skin GLB, used only when --in_blend is omitted")
    ap.add_argument("--fish_id", default=None,
                     help="used only when bootstrapping from --mesh_glb, to align + real-size scale the skin")
    ap.add_argument("--action_json", required=True)
    ap.add_argument("--out_blend", required=True)
    ap.add_argument("--out_report", required=True)
    args = ap.parse_args(argv)

    action = json.loads(args.action_json)

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    if args.in_blend and Path(args.in_blend).exists():
        with bpy.data.libraries.load(args.in_blend, link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for o in data_to.objects:
            if o is not None:
                bpy.context.collection.objects.link(o)
        # CRITICAL: right after linking, matrix_world reads back as IDENTITY (a stale
        # default) until the depsgraph is refreshed -- found by direct reproduction: without
        # this update, every world_bbox_for_objects()/BVH call below silently used the RAW,
        # un-canonicalized, un-scaled mesh (identity transform over the local vertex coords),
        # even though the saved .blend's matrix_world was correct on disk.
        bpy.context.view_layer.update()
    else:
        assert args.mesh_glb, "need --mesh_glb to bootstrap a fresh scene (K^(0))"
        bpy.ops.import_scene.gltf(filepath=args.mesh_glb)
        for o in bpy.context.scene.objects:
            if o.type == "MESH":
                o.name = "Mesh_0"
                o.data.name = "Mesh_0"
                break

    skin = max((o for o in bpy.context.scene.objects if o.type == "MESH"),
               key=lambda o: gask.world_bbox_for_objects([o])["size"].x
                             * gask.world_bbox_for_objects([o])["size"].y
                             * gask.world_bbox_for_objects([o])["size"].z)
    skin.name = "fish_skin"

    align_report = None
    scale_report = None
    is_bootstrap = not (args.in_blend and Path(args.in_blend).exists())
    if is_bootstrap:
        # raw Meshy output has NO guaranteed orientation (a real bug hit earlier this session:
        # a length-along-Y export got its skeleton laid along the thin axis) -- canonicalize
        # to length-X/height-Y/thickness-Z/head-+X BEFORE any position ever gets interpreted
        # as "along the fish", then scale to the tank-measured real body length so the joint
        # physical parameters the actor proposes are for a physically real-sized fish.
        align_report = gask.align_target_mesh_like_supervised(skin)
        # Meshy builds the fish in the PHOTO's pose: a fish photographed tilted (head-down catfish_fish005,
        # bluegill_fish006 ...) comes out as a tilted mesh whose bbox is still "length along X", so the
        # axis canonicalization above leaves the body diagonal -> skeleton and Stage-1 axes off by the
        # tilt (catfish_fish005 after the two-fish fix: alignment 38 deg, coverage 58%). Level it: rotate
        # so the vertex-cloud principal axis is exactly +X (minimal rotation, head stays on +X).
        level_report = _pca_level_skin(skin)
        if align_report is None:
            align_report = {}
        align_report["pca_level"] = level_report
        target_length_m, length_source = find_real_length_m(args.fish_id)
        if target_length_m:
            scale_report = gask.scale_target_mesh_to_real_length(skin, target_length_m)
        else:
            scale_report = {"applied": False, "reason": length_source}
        # Recenter: neither the axis canonicalization nor the real-length scaling moves the mesh, and
        # the actor was trained on origin-centred fish. A cleaned two-fish Meshy mesh (catfish_fish005)
        # left the surviving fish 0.32 m off-origin -> every proposed bone landed outside the skin and
        # was shrunk to ~nothing (coverage 17%). Put the skin bbox centre at the origin.
        align_report["recenter"] = _recenter_skin(skin)

    skin_bbox = gask.world_bbox_for_objects([skin])

    report = {"action": action, "bone_touched": None,
              "verification": None, "warning": None,
              "skin_bbox": {"min": list(skin_bbox["min"]), "max": list(skin_bbox["max"])},
              "align_to_canonical_frame": align_report,
              "scale_to_real_length": scale_report,
              "position_projected": None}

    act = action.get("action")
    bone_obj = None

    def resolve_position(requested):
        """Fix C: a bone whose CENTER is outside the skin can never be made to fit by shrinking
        (the shrink pulls vertices toward that outside center -- observed on catfish_fish006:
        523/624 vertices outside -> 26 shrink steps -> 624/624 outside). If the requested center
        is outside, keep its x (position along the body) but snap y,z onto the body's real
        centerline at that x. Inside positions are left exactly as proposed."""
        from mathutils import Vector

        sys.path.insert(0, str(Path(__file__).resolve().parent))  # extract_centerline.py lives next to this file
        from extract_centerline import compute_centerline, lookup_xyz, x_to_t  # noqa: E402

        # the actor is a language model: coerce whatever it emitted into exactly 3 finite floats
        # (catfish_fish018 crashed on a non-3D position: "vectors must have the same dimensions")
        try:
            vals = [float(v) for v in (requested if isinstance(requested, (list, tuple)) else [requested])]
        except (TypeError, ValueError):
            vals = []
        vals = [v if math.isfinite(v) else 0.0 for v in vals][:3]
        while len(vals) < 3:
            vals.append(0.0)
        requested = vals
        p = Vector(requested)
        skin_bvh = gv.build_skin_bvh(skin)
        if gv.point_is_inside(p, skin_bvh):
            return list(requested), None
        centerline = _robust_centerline([skin.matrix_world @ v.co for v in skin.data.vertices],
                                        skin_bbox["min"][0], skin_bbox["max"][0], 201)
        x_clamped = min(max(p.x, skin_bbox["min"][0]), skin_bbox["max"][0])
        cx, cy, cz = lookup_xyz(centerline, x_to_t(centerline, x_clamped))
        projected = [cx, cy, cz]
        return projected, {"requested": list(requested), "projected": projected,
                           "reason": "requested center was outside the skin"}

    if act == "add":
        tmpl = action["template"]
        from skeleton_template import get_template
        t = get_template(tmpl)
        if t is None:
            report["warning"] = f"unknown template '{tmpl}', action skipped"
        else:
            armatures, mesh_helpers, new_objects, _, _, _ = gask.append_template_skeleton(
                t["template_blend"], t.get("template_skeleton_json"))
            if not mesh_helpers:
                report["warning"] = f"template '{tmpl}' has no bone mesh objects"
            else:
                # pick the template bone whose relative length-axis position is CLOSEST to
                # the requested position (normalized by that template's own bone span) --
                # gives a shape appropriate to the requested body region (head/mid/tail)
                target_x = action["position"][0]
                skin_bbox = gask.world_bbox_for_objects([skin])
                target_frac = ((target_x - skin_bbox["min"][0])
                                / max(skin_bbox["size"].x, 1e-9))
                helper_bboxes = [(h, gask.world_bbox_for_objects([h])) for h in mesh_helpers]
                xs = [b["center"].x for _, b in helper_bboxes]
                lo, hi = min(xs), max(xs)
                span = max(hi - lo, 1e-9)

                def frac_of(b):
                    return (b["center"].x - lo) / span

                mesh_helpers.sort(key=lambda h: abs(
                    frac_of(dict(helper_bboxes)[h]) - target_frac))
                bone_obj = mesh_helpers[0]
                for h in mesh_helpers[1:]:
                    bpy.data.objects.remove(h, do_unlink=True)

                new_id = action.get("bone_id")
                bone_obj.name = f"bone_{new_id}" if new_id is not None else bone_obj.name
                bone_obj.rotation_euler = (0.0, 0.0, 0.0)
                target_pos, proj = resolve_position(action["position"])
                report["position_projected"] = proj
                bone_obj.location = target_pos
                bpy.context.view_layer.update()
                bone_obj.dimensions = action["size"]
                bpy.context.view_layer.update()
                bbox = gask.world_bbox_for_objects([bone_obj])
                bone_obj.location = [bone_obj.location[i] + (target_pos[i] - bbox["center"][i])
                                      for i in range(3)]
                bpy.context.view_layer.update()

    elif act == "remove":
        name = None
        for o in bpy.context.scene.objects:
            if o.type == "MESH" and o.name == f"bone_{action.get('bone_id')}":
                name = o.name
                bpy.data.objects.remove(o, do_unlink=True)
                break
        if name is None:
            report["warning"] = f"bone_id {action.get('bone_id')} not found, nothing removed"

    elif act in ("reposition", "resize"):
        for o in bpy.context.scene.objects:
            if o.type == "MESH" and o.name == f"bone_{action.get('bone_id')}":
                bone_obj = o
                break
        if bone_obj is None:
            report["warning"] = f"bone_id {action.get('bone_id')} not found, nothing changed"
        else:
            if act == "reposition" and action.get("position") is not None:
                target_pos, proj = resolve_position(action["position"])
                report["position_projected"] = proj
                bone_obj.location = target_pos
            if act == "resize" and action.get("size") is not None:
                bpy.context.view_layer.update()
                bone_obj.dimensions = action["size"]
            bpy.context.view_layer.update()

    elif act == "repair":
        report["repair"] = _repair_skeleton(skin, skin_bbox, gv, gask, action)

    else:
        report["warning"] = f"unknown action type '{act}'"

    if bone_obj is not None:
        verify_report = gv.verify_bone(bone_obj, skin)
        report["bone_touched"] = bone_obj.name
        report["verification"] = verify_report

    if act == "repair" or action.get("report_bones"):
        report["bones"] = []
        for o in sorted((o for o in bpy.context.scene.objects if o.type == "MESH" and o.name.startswith("bone_")),
                        key=lambda o: -gask.world_bbox_for_objects([o])["center"].x):
            bb = gask.world_bbox_for_objects([o])
            report["bones"].append({"object_name": o.name, "bone_id": int(o.name.split("_")[-1]),
                                    "position": [bb["center"].x, bb["center"].y, bb["center"].z],
                                    "size": [bb["size"].x, bb["size"].y, bb["size"].z]})

    # Measured midline offsets for EVERY bone (fed to the critic as facts, not left to its
    # eyesight): catfish_fish006's accepted skeleton ran along the belly, ~half a body-height
    # below the true centerline, and the critic still called it "well-aligned along the
    # longitudinal midline". Offsets are the bone-center-vs-centerline distance at the bone's x,
    # as a fraction of the local body half-extent on that axis (0 = on the midline, 1 = at the
    # skin), so they mean the same thing for a 0.18 m and a 0.31 m fish.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # extract_centerline.py lives next to this file
        from extract_centerline import compute_centerline, lookup_xyz, x_to_t  # noqa: E402
        skin_verts = [skin.matrix_world @ v.co for v in skin.data.vertices]
        L = skin_bbox["size"].x
        centerline = _robust_centerline(skin_verts, skin_bbox["min"][0], skin_bbox["max"][0], 201)
        offsets = []
        for o in bpy.context.scene.objects:
            if o.type != "MESH" or not o.name.startswith("bone_"):
                continue
            c = gask.world_bbox_for_objects([o])["center"]
            cx, cy, cz = lookup_xyz(centerline, x_to_t(centerline, c.x))
            he = _robust_half_extents(skin_verts, c.x, max(L * 0.02, 1e-4))
            if he is None:
                continue
            half_y, half_z = max(he[0], 1e-6), max(he[1], 1e-6)
            offsets.append({"bone": o.name, "x_m": round(c.x, 4),
                            "offset_y_frac": round((c.y - cy) / half_y, 3),
                            "offset_z_frac": round((c.z - cz) / half_z, 3)})
        report["midline_offsets"] = offsets
    except Exception as e:  # noqa: BLE001 -- diagnostics must never break the apply step
        report["midline_offsets_error"] = f"{type(e).__name__}: {e}"

    Path(args.out_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.out_blend).resolve()))
    Path(args.out_report).write_text(json.dumps(report, indent=2, default=str))
    print(f"[articulation] applied {act} -> {args.out_blend}", flush=True)
    print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    # Only meaningful when run under Blender/BPY (`blender -b --python articulation.py -- ...`
    # or `<blender's bundled python3> articulation.py -- ...`); bpy/pxr are imported lazily
    # inside these two functions so plain-python callers (the orchestrator, actor.py,
    # critic.py) can still import the dataclasses above freely.
    if "--apply_mass" in sys.argv:
        _apply_mass_main()
    else:
        _blender_apply_action()
