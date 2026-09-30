import argparse
import csv
import json
from pathlib import Path

from pxr import Usd


def parse_args():
    parser = argparse.ArgumentParser(description="Report generated articulated USD contents.")
    parser.add_argument("--usd-root", required=True, help="dir of <sample>/fish_articulated.usda")
    parser.add_argument("--source-root", required=True, help="dir of <sample>/skeleton.json")
    parser.add_argument("--output", required=True, help="output CSV path")
    return parser.parse_args()


def count_skeleton(path):
    data = json.load(open(path, encoding="utf-8"))
    return len(data.get("skeleton_objects", [])) + sum(item.get("bone_count", 0) for item in data.get("armatures", []))


def main():
    args = parse_args()
    usd_root = Path(args.usd_root)
    source_root = Path(args.source_root)
    rows = []
    for sample_dir in sorted([p for p in usd_root.iterdir() if p.is_dir()]):
        usd_path = sample_dir / "fish_articulated.usda"
        source_json = source_root / sample_dir.name / "skeleton.json"
        row = {
            "sample": sample_dir.name,
            "usd_exists": usd_path.exists(),
            "source_skeleton_exists": source_json.exists(),
            "bone_count": count_skeleton(source_json) if source_json.exists() else None,
            "joint_count": None,
            "rigid_body_count": None,
            "articulation_root_count": None,
            "collision_count": None,
            "mesh_collision_count": None,
            "physics_material_count": None,
            "attachment_count": None,
            "joint_count_matches_bones_minus_one": False,
        }
        if usd_path.exists():
            stage = Usd.Stage.Open(str(usd_path))
            apis = {}
            types = {}
            for prim in stage.Traverse():
                types[prim.GetTypeName()] = types.get(prim.GetTypeName(), 0) + 1
                for api in prim.GetAppliedSchemas():
                    apis[api] = apis.get(api, 0) + 1
            row["joint_count"] = types.get("PhysicsJoint", 0)
            row["rigid_body_count"] = apis.get("PhysicsRigidBodyAPI", 0)
            row["articulation_root_count"] = apis.get("PhysicsArticulationRootAPI", 0)
            row["collision_count"] = apis.get("PhysicsCollisionAPI", 0)
            row["mesh_collision_count"] = apis.get("PhysicsMeshCollisionAPI", 0)
            row["physics_material_count"] = apis.get("PhysicsMaterialAPI", 0)
            row["attachment_count"] = types.get("PhysxPhysicsAttachment", 0)
            if row["bone_count"] is not None:
                row["joint_count_matches_bones_minus_one"] = row["joint_count"] == max(0, row["bone_count"] - 1)
        rows.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print("samples", len(rows))
    print("missing_usd", [row["sample"] for row in rows if not row["usd_exists"]])
    print("bad_joint_count", [row["sample"] for row in rows if not row["joint_count_matches_bones_minus_one"]])
    print("output", args.output)


if __name__ == "__main__":
    main()
