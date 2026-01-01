#!/usr/bin/env python
"""Copy every processed K_final.usd and swap its BOX bone collision/visual meshes for the
fish-spine shape, scaled per-axis so each bone's world bounding box is IDENTICAL.

What is changed:  only the child Mesh geometry of a bone (points / faceVertexCounts /
                  faceVertexIndices / normals / extent / st / doubleSided).
What is NOT changed: bone count, bone Xform (translate/rotate/scale), physics:mass,
                  RigidBody/Articulation/Mass APIs, D6 joints, FEM attachments, the skin.
                  -> DOF is unchanged, so trained policies and the per-bone
                     calibration/panel_hydro/amp_reference files stay valid.
Caveat: the collider is physics:approximation="convexHull", so the hull shrinks from the
        full box to the spine's convex hull (same bbox, smaller volume).

  python box_bones_to_spine.py --dataset Video2SwimFish/dataset --out Video2SwimFish/dataset_spine
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path

from pxr import Usd, UsdGeom, Gf, Vt

TEMPLATE_PRIM = "lower_curved_triangular_fish_spine_07_mesh"


def norm_points(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    lo = (min(xs), min(ys), min(zs)); hi = (max(xs), max(ys), max(zs))
    d = [hi[i] - lo[i] for i in range(3)]
    out = []
    for p in pts:
        out.append(Gf.Vec3f(*[2.0 * (p[i] - lo[i]) / d[i] - 1.0 if d[i] > 1e-12 else 0.0
                              for i in range(3)]))
    return out


def load_template(usd_path: Path):
    stage = Usd.Stage.Open(str(usd_path))
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and TEMPLATE_PRIM in prim.GetName():
            m = UsdGeom.Mesh(prim)
            return dict(
                points=norm_points(list(m.GetPointsAttr().Get())),
                counts=list(m.GetFaceVertexCountsAttr().Get()),
                indices=list(m.GetFaceVertexIndicesAttr().Get()),
                normals=list(m.GetNormalsAttr().Get()),
            )
    raise RuntimeError(f"template prim {TEMPLATE_PRIM} not found in {usd_path}")


def is_box(mesh: UsdGeom.Mesh) -> bool:
    p = mesh.GetPointsAttr().Get()
    return p is not None and len(p) == 8


def world_bbox(stage, prim):
    c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = c.ComputeWorldBound(prim).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def convert(src: Path, dst: Path, tmpl, verbose=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    stage = Usd.Stage.Open(str(dst))
    if stage is None:
        raise RuntimeError(f"cannot open {dst}")

    bones, swapped, kept = 0, 0, 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        parent = prim.GetParent()
        if not parent or not parent.HasAPI(Usd.SchemaRegistry().GetAPITypeFromSchemaTypeName("PhysicsRigidBodyAPI")) \
           and "PhysicsRigidBodyAPI" not in (parent.GetAppliedSchemas() or []):
            continue
        bones += 1
        mesh = UsdGeom.Mesh(prim)
        if not is_box(mesh):
            kept += 1
            continue
        before = world_bbox(stage, prim)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(tmpl["points"]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(tmpl["counts"]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(tmpl["indices"]))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray(tmpl["normals"]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)]))
        mesh.CreateDoubleSidedAttr(True)
        st = prim.GetAttribute("primvars:st")
        if st and st.IsValid():
            st.Block()                       # old 24-value UV set no longer matches
        after = world_bbox(stage, prim)
        err = max(abs(before[0][i] - after[0][i]) for i in range(3)) + \
              max(abs(before[1][i] - after[1][i]) for i in range(3))
        if err > 1e-6:
            raise RuntimeError(f"{dst.parent.name}/{prim.GetName()}: bbox moved by {err:.3e} m")
        swapped += 1
        if verbose:
            print(f"    {prim.GetPath()}  box -> spine  bbox err {err:.2e}")
    stage.GetRootLayer().Save()
    return bones, swapped, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template_from", default=None, help="fish tag to take the spine shape from")
    ap.add_argument("--only", default=None, help="convert just this tag")
    ap.add_argument("--link_siblings", action="store_true",
                    help="symlink the sibling .npz/.json next to each converted usd")
    ap.add_argument("--write_usda", action="store_true", help="also export an ASCII copy")
    args = ap.parse_args()

    ds = Path(args.dataset).resolve(); out = Path(args.out).resolve()
    tags = sorted(p.name for p in ds.iterdir() if p.is_dir() and (p / "K_final.usd").exists())
    if args.only:
        tags = [t for t in tags if t == args.only]
    tmpl_tag = args.template_from or "catfish_fish001"
    tmpl = load_template(ds / tmpl_tag / "K_final.usd")
    print(f"template: {tmpl_tag}/{TEMPLATE_PRIM}  "
          f"({len(tmpl['points'])} pts, {len(tmpl['counts'])} faces), normalised to [-1,1]^3")
    print(f"{len(tags)} fish -> {out}")

    tot_b = tot_s = 0; fails = []
    for i, tag in enumerate(tags, 1):
        try:
            b, s, k = convert(ds / tag / "K_final.usd", out / tag / "K_final.usd", tmpl,
                              verbose=(args.only is not None))
            tot_b += b; tot_s += s
            print(f"[{i:3d}/{len(tags)}] {tag:26s} bones={b:2d}  box->spine={s:2d}  spine_kept={k:2d}")
            if args.write_usda:
                Usd.Stage.Open(str(out / tag / "K_final.usd")).GetRootLayer().Export(
                    str(out / tag / "K_final.usda"))
            if args.link_siblings:
                for f in (ds / tag).iterdir():
                    if f.is_file() and f.name != "K_final.usd" and f.suffix in (".npz", ".json"):
                        lnk = out / tag / f.name
                        if not lnk.exists():
                            lnk.symlink_to(f)
        except Exception as e:
            print(f"[{i:3d}/{len(tags)}] {tag:26s} FAILED: {e}")
            fails.append((tag, str(e)))
    print(f"\ndone: {tot_s} box bones -> spine across {len(tags)-len(fails)} fish "
          f"({tot_b} bones seen); {len(fails)} failed")
    for t, e in fails: print("  FAIL", t, e)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
