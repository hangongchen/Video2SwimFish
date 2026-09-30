#!/usr/bin/env python
"""Drop stray SEPARATE bodies from a Meshy mesh (e.g. a second fish that was in the canonical frame).

Why: catfish_fish005's canonical frame had two fish touching; Meshy reconstructed BOTH, the VLM
mesh critic passed it, and the skeleton was laid across both bodies (Stage 1 alignment 49 deg).

Rule (deterministic, conservative): weld vertices, split into connected components. Keep the
largest. Drop any other component whose axis-aligned bbox is DISJOINT from the largest component's
bbox along at least one axis (a spatially separate body), or whose surface share exceeds 15%
(no fin is that large). Small components overlapping the main body (detached fins, eyes, barbels)
are kept. Nothing is dropped when no component
is disjoint -> the file is left untouched.

Usage: clean_mesh_components.py --glb outputs/<tag>/mesh.glb [--dry_run]
Writes: <glb> (cleaned, original kept as <glb>.meshy_original.glb) + <glb dir>/mesh_clean_report.json
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import trimesh

SECOND_BODY_FRAC = 0.15   # a welded component this large that is not the main body is another fish
OUTGROW_FRAC = 0.10       # secondary component may extend the main bbox by at most 10% per axis (fins do less)

def analyse(glb: Path):
    # Meshy glbs are a soup of ~2000 unwelded patches: weld coincident vertices first, otherwise
    # face-adjacency components are meaningless (catfish_fish005: 1938 comps -> 7 after welding,
    # the two fish showing up as 66% / 33% of the surface)
    sc = trimesh.load(str(glb), force="scene", process=True)
    geoms = list(sc.geometry.values())
    m = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    m.merge_vertices(merge_tex=True, merge_norm=True)
    comps = m.split(only_watertight=False)
    comps = sorted(comps, key=lambda c: float(c.area), reverse=True)
    tot = sum(float(c.area) for c in comps)
    main = comps[0]
    # do every bbox test in the MAIN component's principal-axis frame: a tilted fish has a fat
    # axis-aligned bbox that hides fragments of another body which stick out once the fish is
    # leveled (catfish_fish005 again: fragment kept -> body length inflated -> coverage 17%)
    Vm = np.asarray(main.vertices, dtype=np.float64); cm = Vm.mean(0)
    _, _, Rt = np.linalg.svd(Vm - cm, full_matrices=False)
    def obb(c):
        P = (np.asarray(c.vertices, dtype=np.float64) - cm) @ Rt.T
        return np.stack([P.min(0), P.max(0)])
    b0 = obb(main)
    keep, drop = [main], []
    ext0 = b0[1] - b0[0]
    for c in comps[1:]:
        b1 = obb(c)
        gap = np.maximum(b1[0] - b0[1], b0[0] - b1[1])       # >0 on an axis => disjoint on that axis
        big = float(c.area) / tot > SECOND_BODY_FRAC            # a fin is never this large a share
        # a fragment of ANOTHER body can overlap the main bbox yet stick far out of it (catfish_fish005:
        # a sliver of fish #2 was kept and inflated the body length -> Stage-1 coverage 17%). Drop any
        # component that extends the main component's bbox by more than OUTGROW_FRAC of its extent.
        outgrow = np.maximum(b0[0] - b1[0], b1[1] - b0[1]) / np.maximum(ext0, 1e-9)
        sticks_out = bool((outgrow > OUTGROW_FRAC).any())
        (drop if ((gap > 0).any() or big or sticks_out) else keep).append(c)
    rep = {"n_components": len(comps), "largest_area_frac": float(main.area) / tot,
           "dropped_area_frac": sum(float(c.area) for c in drop) / tot, "n_dropped": len(drop),
           "n_kept": len(keep)}
    return m, keep, drop, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--min_drop_frac", type=float, default=0.02,
                    help="only rewrite the file when the dropped area is at least this fraction")
    a = ap.parse_args()
    glb = Path(a.glb)
    m, keep, drop, rep = analyse(glb)
    rep["source"] = str(glb)
    print(json.dumps(rep))
    if a.dry_run or not drop or rep["dropped_area_frac"] < a.min_drop_frac:
        rep["rewritten"] = False
    else:
        backup = glb.with_suffix(".meshy_original.glb")
        if not backup.exists():
            shutil.copy2(glb, backup)
        cleaned = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
        cleaned.export(str(glb))
        rep["rewritten"] = True
        rep["backup"] = str(backup)
        print(f"[clean_mesh] {glb}: dropped {len(drop)} separate component(s) "
              f"({rep['dropped_area_frac']:.1%} of surface); original saved as {backup.name}")
    (glb.parent / "mesh_clean_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
