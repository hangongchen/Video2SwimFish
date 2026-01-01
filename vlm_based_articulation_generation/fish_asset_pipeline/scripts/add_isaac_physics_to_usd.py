import argparse
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def parse_args():
    parser = argparse.ArgumentParser(description="Add Isaac/PhysX-style physics metadata and joints to an exported fish USD.")
    parser.add_argument("--input-usd", required=True)
    parser.add_argument("--skeleton-json", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "usd_physics_defaults.json"))
    parser.add_argument("--output-usd", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def apply_schema_token(prim, schema):
    schemas = list(prim.GetAppliedSchemas())
    if schema not in schemas:
        prim.GetReferences()
        prim.ApplyAPI(schema)


def set_attr(prim, name, type_name, value):
    attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)
    return attr


def set_explicit_api_schemas(prim, schemas):
    existing = []
    metadata = prim.GetMetadata("apiSchemas")
    if metadata:
        explicit = getattr(metadata, "explicitItems", None)
        if explicit is None and hasattr(metadata, "GetExplicitItems"):
            explicit = metadata.GetExplicitItems()
        existing = [str(item) for item in (explicit or [])]
    merged = []
    for item in existing + list(schemas):
        if item not in merged:
            merged.append(item)
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(merged))


def find_prim_by_name(stage, name):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def find_first_mesh(stage, exclude_names):
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and prim.GetName() not in exclude_names:
            return prim
    return None


def mesh_child_or_self(prim):
    if prim and prim.GetTypeName() == "Mesh":
        return prim
    if prim:
        for child in prim.GetChildren():
            if child.GetTypeName() == "Mesh":
                return child
    return prim


def ensure_reference_form_hierarchy(stage, bone_names):
    root = stage.GetDefaultPrim()
    root_path = str(root.GetPath()) if root else "/root"
    skeleton_path = Sdf.Path(root_path + "/skeleton")
    UsdGeom.Xform.Define(stage, skeleton_path)

    moved = []
    for name in sorted(bone_names):
        old_path = Sdf.Path(root_path + "/" + safe_name(name))
        new_path = Sdf.Path(str(skeleton_path) + "/" + safe_name(name))
        if stage.GetPrimAtPath(new_path):
            moved.append(str(new_path))
            continue
        if stage.GetPrimAtPath(old_path):
            editor = Usd.NamespaceEditor(stage)
            editor.MovePrimAtPath(old_path, new_path)
            if not editor.CanApplyEdits():
                raise RuntimeError(f"Could not move {old_path} to {new_path}.")
            editor.ApplyEdits()
            moved.append(str(new_path))
    return skeleton_path


def world_center_from_bbox(bbox):
    mn = bbox["min"]
    mx = bbox["max"]
    return Gf.Vec3d((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)


def ordered_bones(skeleton_json):
    bones = []
    for item in skeleton_json.get("skeleton_objects", []):
        bbox = item.get("bbox_world")
        if not bbox:
            continue
        bones.append({"name": item["object_name"], "bbox": bbox, "center": world_center_from_bbox(bbox)})
    bones.sort(key=lambda item: item["center"][0])
    return bones


def local_pos(stage, prim, world_point):
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    inv = mat.GetInverse()
    return Gf.Vec3f(inv.Transform(world_point))


def make_rigid_material(stage, path, dynamic_friction=0.0, density=0.0, restitution=0.0, static_friction=0.0):
    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    UsdPhysics.MaterialAPI.Apply(prim)
    set_attr(prim, "physics:dynamicFriction", Sdf.ValueTypeNames.Float, float(dynamic_friction))
    set_attr(prim, "physics:staticFriction", Sdf.ValueTypeNames.Float, float(static_friction))
    set_attr(prim, "physics:restitution", Sdf.ValueTypeNames.Float, float(restitution))
    set_attr(prim, "physics:density", Sdf.ValueTypeNames.Float, float(density))
    return material


def make_deformable_material(stage, path, config):
    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    deform = config["fish"]["deformable_material"]
    set_explicit_api_schemas(prim, ["PhysxDeformableBodyMaterialAPI"])
    set_attr(prim, "physxDeformableBodyMaterial:youngsModulus", Sdf.ValueTypeNames.Float, float(deform["youngs_modulus"]))
    set_attr(prim, "physxDeformableBodyMaterial:poissonsRatio", Sdf.ValueTypeNames.Float, float(deform["poissons_ratio"]))
    set_attr(prim, "physxDeformableBodyMaterial:elasticityDamping", Sdf.ValueTypeNames.Float, float(deform["elasticity_damping"]))
    set_attr(prim, "physxDeformableBodyMaterial:dampingScale", Sdf.ValueTypeNames.Float, float(deform["damping_scale"]))
    return material


def apply_bone_physics(prim, config, bone_mass=None):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    set_attr(prim, "physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, bool(config["bone"]["rigid_body_enabled"]))
    set_attr(prim, "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, bool(config["bone"]["kinematic_enabled"]))
    set_attr(prim, "physics:startsAsleep", Sdf.ValueTypeNames.Bool, bool(config["bone"]["starts_asleep"]))
    set_attr(prim, "physics:velocity", Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0, 0, 0))
    set_attr(prim, "physics:angularVelocity", Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0, 0, 0))
    if bone_mass is not None:
        set_attr(prim, "physics:mass", Sdf.ValueTypeNames.Float, float(bone_mass))
        set_attr(prim, "physics:density", Sdf.ValueTypeNames.Float, 0.0)
    if prim.GetTypeName() == "Mesh":
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        set_attr(prim, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, bool(config["bone"]["collision_enabled"]))
        set_attr(prim, "physics:approximation", Sdf.ValueTypeNames.Token, config["bone"]["collision_approximation"])
    for child in prim.GetChildren():
        if child.GetTypeName() == "Mesh":
            UsdPhysics.CollisionAPI.Apply(child)
            UsdPhysics.MeshCollisionAPI.Apply(child)
            set_attr(child, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, bool(config["bone"]["collision_enabled"]))
            set_attr(child, "physics:approximation", Sdf.ValueTypeNames.Token, config["bone"]["collision_approximation"])


def apply_fish_physics(fish_prim, config, fish_mass=None):
    if not fish_prim:
        return
    fish_prim = mesh_child_or_self(fish_prim)
    set_explicit_api_schemas(fish_prim, ["PhysxDeformableBodyAPI", "PhysxCollisionAPI", "PhysicsMassAPI"])
    UsdPhysics.MassAPI.Apply(fish_prim)
    mass = float(config["fish"]["mass"]) if fish_mass is None else float(fish_mass)
    set_attr(fish_prim, "physics:mass", Sdf.ValueTypeNames.Float, mass)
    set_attr(fish_prim, "physics:density", Sdf.ValueTypeNames.Float, float(config["fish"]["density"]))
    deform = config["fish"]["deformable_material"]
    # These PhysX attributes are authored as custom attrs so Isaac Sim can read
    # or convert them even when the local USD build lacks PhysxSchema.
    set_attr(fish_prim, "physxDeformable:youngsModulus", Sdf.ValueTypeNames.Float, float(deform["youngs_modulus"]))
    set_attr(fish_prim, "physxDeformable:poissonsRatio", Sdf.ValueTypeNames.Float, float(deform["poissons_ratio"]))
    set_attr(fish_prim, "physxDeformable:elasticityDamping", Sdf.ValueTypeNames.Float, float(deform["elasticity_damping"]))
    set_attr(fish_prim, "physxDeformable:dampingScale", Sdf.ValueTypeNames.Float, float(deform["damping_scale"]))
    set_attr(fish_prim, "physxDeformable:enableCCD", Sdf.ValueTypeNames.Bool, True)
    set_attr(fish_prim, "physxDeformable:kinematicEnabled", Sdf.ValueTypeNames.Bool, False)
    set_attr(fish_prim, "physxDeformable:collisionSimplification", Sdf.ValueTypeNames.Bool, True)
    set_attr(fish_prim, "physxDeformable:collisionSimplificationRemeshing", Sdf.ValueTypeNames.Bool, True)
    set_attr(fish_prim, "physxDeformable:simulationHexahedralResolution", Sdf.ValueTypeNames.Int, 10)


def bbox_volume_m3(skeleton_json):
    bbox = (skeleton_json.get("fish_mesh") or {}).get("bbox_world") or {}
    volume = bbox.get("volume")
    if volume is not None:
        return float(volume)
    size = bbox.get("size") or []
    if len(size) == 3:
        return abs(float(size[0]) * float(size[1]) * float(size[2]))
    return None


def mass_distribution_from_mesh(skeleton_json, config):
    fish_cfg = config.get("fish", {})
    deformable_mass = float(fish_cfg.get("deformable_mass", fish_cfg.get("mass", 0.00001)))
    fish_record = skeleton_json.get("fish_mesh") or {}
    fill_fraction = float(fish_cfg.get("volume_fill_fraction", 0.35))
    min_fraction = float(fish_cfg.get("mesh_volume_min_bbox_fraction", 0.15))
    max_fraction = float(fish_cfg.get("mesh_volume_max_bbox_fraction", 0.8))

    bbox_volume = bbox_volume_m3(skeleton_json)
    mesh_volume = fish_record.get("mesh_volume_world_m3")
    mesh_volume = float(mesh_volume) if mesh_volume is not None else None

    # The signed-volume integral in extract_fish_skeleton.py is only valid for
    # watertight meshes; open meshes can report volumes 100-1000x too small.
    # Trust it only when it lands in the plausible fraction of the bbox volume,
    # otherwise estimate volume as bbox volume times a typical fish fill fraction.
    volume = None
    volume_source = None
    if mesh_volume and mesh_volume > 0 and bbox_volume and bbox_volume > 0:
        fraction = mesh_volume / bbox_volume
        if min_fraction <= fraction <= max_fraction:
            volume = mesh_volume
            volume_source = "mesh_volume_world_m3"
    if volume is None and bbox_volume and bbox_volume > 0:
        volume = bbox_volume * fill_fraction
        volume_source = "bbox_volume_times_fill_fraction"
    if volume is None and mesh_volume and mesh_volume > 0:
        volume = mesh_volume
        volume_source = "mesh_volume_world_m3_unchecked"
    if volume is None or float(volume) <= 0:
        return {
            "volume_m3": None,
            "volume_source": "fallback_config_mass",
            "density_kg_per_m3": None,
            "total_mass_kg": float(config["bone"].get("total_mass", 25.0)),
            "bone_total_mass_kg": float(config["bone"].get("total_mass", 25.0)),
            "fish_deformable_mass_kg": deformable_mass,
            "asset_total_mass_kg": float(config["bone"].get("total_mass", 25.0)) + deformable_mass,
        }

    density = float(fish_cfg.get("water_density_kg_per_m3", 1000.0))
    bone_total_mass = float(volume) * density
    return {
        "volume_m3": float(volume),
        "volume_source": volume_source,
        "mesh_volume_world_m3": mesh_volume,
        "bbox_volume_m3": bbox_volume,
        "density_kg_per_m3": density,
        "total_mass_kg": bone_total_mass,
        "bone_total_mass_kg": bone_total_mass,
        "fish_deformable_mass_kg": deformable_mass,
        "asset_total_mass_kg": bone_total_mass + deformable_mass,
    }


def add_translate_delta(prim, delta):
    translate_attr = prim.GetAttribute("xformOp:translate")
    current = translate_attr.Get() if translate_attr else None
    if current is None:
        current = Gf.Vec3d(0.0, 0.0, 0.0)
    new_translate = Gf.Vec3d(
        float(current[0]) + float(delta[0]),
        float(current[1]) + float(delta[1]),
        float(current[2]) + float(delta[2]),
    )
    set_attr(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3, new_translate)
    order_attr = prim.GetAttribute("xformOpOrder")
    order = [str(item) for item in (order_attr.Get() if order_attr and order_attr.Get() else [])]
    if "xformOp:translate" not in order:
        order = ["xformOp:translate"] + order
    set_attr(prim, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Vt.TokenArray(order))
    return new_translate


def move_asset_fish_bottom_to_target(stage, skeleton_json, config, fish_prim, skeleton_path):
    target = config.get("stage", {}).get("target_fish_bottom_z")
    if target is None:
        return None
    bbox = (skeleton_json.get("fish_mesh") or {}).get("bbox_world") or {}
    mn = bbox.get("min")
    if not mn or len(mn) < 3:
        return None
    current_bottom = float(mn[2])
    delta_z = float(target) - current_bottom
    delta = Gf.Vec3d(0.0, 0.0, delta_z)
    moved = {}
    if fish_prim:
        fish_translate = add_translate_delta(fish_prim, delta)
        moved[str(fish_prim.GetPath())] = [float(fish_translate[0]), float(fish_translate[1]), float(fish_translate[2])]
    skeleton_prim = stage.GetPrimAtPath(skeleton_path)
    if skeleton_prim:
        for child in skeleton_prim.GetChildren():
            child_translate = add_translate_delta(child, delta)
            moved[str(child.GetPath())] = [float(child_translate[0]), float(child_translate[1]), float(child_translate[2])]
    return {
        "target_fish_bottom_z": float(target),
        "source_fish_bottom_z": current_bottom,
        "asset_translate_z_delta": delta_z,
        "moved_prims": moved,
    }


def create_joint(stage, path, body0, body1, midpoint, config):
    joint = UsdPhysics.Joint.Define(stage, path)
    prim = joint.GetPrim()
    set_attr(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(midpoint))
    set_attr(prim, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Vt.TokenArray(["xformOp:translate"]))
    limit_apis = {}
    for api in ["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]:
        limit_apis[api] = UsdPhysics.LimitAPI.Apply(prim, api)

    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateLocalPos0Attr(local_pos(stage, body0, midpoint))
    joint.CreateLocalPos1Attr(local_pos(stage, body1, midpoint))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    joint.CreateCollisionEnabledAttr(bool(config["joint"]["collision_enabled"]))
    joint.CreateExcludeFromArticulationAttr(bool(config["joint"]["exclude_from_articulation"]))
    joint.CreateJointEnabledAttr(True)
    joint.CreateBreakForceAttr(math.inf)
    joint.CreateBreakTorqueAttr(math.inf)

    trans = config["joint"]["translation_limits"]
    rot = config["joint"]["rotation_limits"]
    for axis in ["transX", "transY", "transZ"]:
        api = limit_apis[axis]
        api.CreateLowAttr(float(trans["low"]))
        api.CreateHighAttr(float(trans["high"]))
    for axis in ["rotX", "rotY", "rotZ"]:
        api = limit_apis[axis]
        api.CreateLowAttr(float(rot["low"]))
        api.CreateHighAttr(float(rot["high"]))
    return prim


def create_ground_plane(stage, skeleton_json, config):
    if not config.get("ground_plane", {}).get("enabled", True):
        return None
    bbox = skeleton_json.get("fish_mesh", {}).get("bbox_world")
    if not bbox:
        return None
    z = float(config["ground_plane"].get("world_z", bbox["min"][2] - float(config["ground_plane"].get("z_offset_below_fish", 1.0))))
    root = stage.GetDefaultPrim()
    root_path = str(root.GetPath()) if root else "/root"
    plane = UsdGeom.Mesh.Define(stage, root_path + "/GroundPlane")
    size = float(config["ground_plane"]["size"])
    points = [(-size, -size, z), (size, -size, z), (size, size, z), (-size, size, z)]
    plane.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane_prim = plane.GetPrim()
    UsdPhysics.CollisionAPI.Apply(plane_prim)
    UsdPhysics.MeshCollisionAPI.Apply(plane_prim)
    set_attr(plane_prim, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, True)
    set_attr(plane_prim, "physics:approximation", Sdf.ValueTypeNames.Token, "none")
    plane_prim.SetActive(bool(config["ground_plane"].get("active", True)))
    return plane_prim


def create_attachments(stage, fish_prim, bones, body_prims, config):
    fish_prim = mesh_child_or_self(fish_prim)
    attachments = []
    for idx, bone in enumerate(bones):
        body = body_prims.get(bone["name"])
        if not body or not fish_prim:
            continue
        suffix = "" if idx == 0 else f"_{idx:02d}"
        prim = stage.DefinePrim(f"{fish_prim.GetPath()}/attachment{suffix}", "PhysxPhysicsAttachment")
        set_explicit_api_schemas(prim, ["PhysxAutoAttachmentAPI"])
        prim.CreateRelationship("actor0").SetTargets([fish_prim.GetPath()])
        prim.CreateRelationship("actor1").SetTargets([body.GetPath()])
        prim.CreateRelationship("physxAutoAttachment:maskShapes").SetTargets([])
        set_attr(prim, "physxAutoAttachment:enableRigidSurfaceAttachments", Sdf.ValueTypeNames.Bool, bool(config["attachments"]["attach_rigid_surface"]))
        set_attr(prim, "physxAutoAttachment:overlapOffset", Sdf.ValueTypeNames.Float, 0.0)
        set_attr(prim, "physxAttachment:inputCrc", Sdf.ValueTypeNames.UCharArray, Vt.UCharArray([0] * 16))
        attachments.append(prim)
    return attachments


def main():
    args = parse_args()
    config = load_json(args.config)
    skeleton = load_json(args.skeleton_json)
    stage = Usd.Stage.Open(args.input_usd)
    if stage.GetDefaultPrim() is None:
        first = next(stage.Traverse(), None)
        if first:
            stage.SetDefaultPrim(first)

    bones = ordered_bones(skeleton)
    bone_names = {item["name"] for item in bones}
    body_prims = {}
    skeleton_path = ensure_reference_form_hierarchy(stage, bone_names)
    mass_distribution = mass_distribution_from_mesh(skeleton, config)
    bone_mass = float(mass_distribution["bone_total_mass_kg"]) / max(1, len(bones))
    for bone in bones:
        prim = find_prim_by_name(stage, bone["name"])
        if prim:
            body_prims[bone["name"]] = prim
            apply_bone_physics(prim, config, bone_mass)

    fish_name = (skeleton.get("fish_mesh") or {}).get("object_name")
    fish_prim = find_prim_by_name(stage, fish_name) if fish_name else None
    if fish_prim is None:
        fish_prim = find_first_mesh(stage, bone_names)
    fish_asset_prim = fish_prim
    apply_fish_physics(fish_prim, config, fish_mass=mass_distribution["fish_deformable_mass_kg"])
    fish_prim = mesh_child_or_self(fish_prim)

    for idx, (left, right) in enumerate(zip(bones, bones[1:])):
        body0 = body_prims.get(left["name"])
        body1 = body_prims.get(right["name"])
        if not body0 or not body1:
            continue
        midpoint = (left["center"] + right["center"]) * 0.5
        name = "D6Joint" if idx == 0 else f"D6Joint_{idx:02d}"
        create_joint(stage, f"{skeleton_path}/{name}", body0, body1, midpoint, config)

    root = stage.GetDefaultPrim()
    root_path = str(root.GetPath()) if root else "/root"
    make_rigid_material(stage, root_path + "/RigidMaterial", dynamic_friction=0.0)
    make_deformable_material(stage, root_path + "/deformableMaterial", config)
    create_ground_plane(stage, skeleton, config)
    create_attachments(stage, fish_prim, bones, body_prims, config)
    stage_position = move_asset_fish_bottom_to_target(stage, skeleton, config, fish_asset_prim, skeleton_path)

    meta = stage.GetRootLayer()
    meta.customLayerData = {
        "fish_asset_pipeline": "generated_by_add_isaac_physics_to_usd",
        "source_skeleton_json": args.skeleton_json,
        "physics_config": args.config,
        "mass_distribution": json.dumps(mass_distribution, sort_keys=True),
        "stage_position": json.dumps(stage_position, sort_keys=True),
    }
    os.makedirs(os.path.dirname(args.output_usd), exist_ok=True)
    stage.GetRootLayer().Export(args.output_usd)


if __name__ == "__main__":
    main()
