#!/usr/bin/env python
"""One table for the whole controllable-fish dataset (all species): per fish -> mesh, skeleton
(bones / VLM iterations / critic after repair), Stage 1 (containment / coverage / alignment),
hydro proxy, per-fish PCA (clean frames / basis source), real size (+ how it was obtained).
Writes $ASSET_OUT/DATASET_REPORT.md and DATASET_REPORT.csv.

Usage: python dataset_report.py [--species catfish,lake_sturgeon,bluegill,white_bass]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P

ASSET = P.ASSET_OUT; MESHES = P.MESH_OUT; SPM = P.SPECIES_MANIFOLD_ROOT
# PCA columns come from the sibling swimming_policy_training repo (optional; "-" if absent)
SWIM_SPM = P.POLICY_ROOT / "data" / "species_manifold"
ap = argparse.ArgumentParser(); ap.add_argument("--species", default="catfish,lake_sturgeon,bluegill,white_bass"); a = ap.parse_args()

def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

rows = []
for sp in a.species.split(","):
    sizes = jload(SPM / sp / "fish_sizes.json") or {"per_fish": {}}
    for fid in sorted(sizes["per_fish"]):
        tag = f"{sp}_fish{fid}"; d = ASSET / tag; s = sizes["per_fish"][fid]
        r = {"fish": tag, "length_cm": round(s["length_cm"], 1),
             "size_source": "median/estimate" if s.get("suspect") else "measured",
             "mesh": "yes" if (MESHES / tag / "mesh.glb").exists() else "-",
             "usd": "yes" if (d / "K_final.usd").exists() else "-"}
        log = jload(d / "articulation_construction_log.json")
        if log:
            car = log.get("critic_after_repair") or {}
            r.update(bones=log.get("final_num_bones"), vlm_iters=len(log.get("iterations", [])),
                     critic_after_repair=car.get("score") if isinstance(car, dict) else car)
        s1 = jload(d / "stage1_report.json")
        if s1:
            r.update(contain="PASS" if s1["check1_interpenetration"]["passed"] else f"FAIL({s1['check1_interpenetration'].get('total_vertices_outside')}v)",
                     coverage=f"{100*s1['check2_coverage'].get('coverage_frac',0):.0f}%",
                     align_mean=round(s1["check3_alignment"]["mean_angle_deg"], 1), align_max=round(s1["check3_alignment"]["max_angle_deg"], 1))
        r["hydro"] = "yes" if (d / "panel_hydro.npz").exists() else "-"
        man = SWIM_SPM / f"{sp}_fix" / "top"
        b = man / f"fish{fid}_pca_basis.npz"; c = man / f"fish{fid}_curvature.npz"
        if b.exists():
            bb = np.load(b, allow_pickle=True); src = str(bb["source"]) if "source" in bb else "own"
            r["pca"] = "pooled-fallback" if "pooled" in src else "own"
            if c.exists():
                cc = np.load(c); v = cc["valid"]; r["pca_clean_frames"] = int((v & (np.abs(cc["kappa_bl"]).max(1) < 4)).sum())
        else:
            r["pca"] = "-"
        rows.append(r)

cols = ["fish", "length_cm", "size_source", "mesh", "usd", "bones", "vlm_iters", "critic_after_repair", "contain", "coverage",
        "align_mean", "align_max", "hydro", "pca", "pca_clean_frames"]
with open(ASSET / "DATASET_REPORT.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({k: r.get(k, "") for k in cols}) for r in rows]
def cell(v): return "" if v is None else str(v)
md = ["# Controllable fish dataset report", "", f"{len(rows)} fish. usd = controllable asset exported (skeleton + FEM + physics). "
      "Stage 1: containment / coverage of body length / bone-axis alignment (mean, max deg). pca = per-fish top-view PCA basis "
      "(own video, or species-pooled fallback when the fish's own video was unusable).", "",
      "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(cell(r.get(k)) for k in cols) + " |" for r in rows]
done = sum(1 for r in rows if r["usd"] == "yes"); hyd = sum(1 for r in rows if r["hydro"] == "yes")
own = sum(1 for r in rows if r.get("pca") == "own"); pooled = sum(1 for r in rows if r.get("pca") == "pooled-fallback")
md += ["", f"**Totals:** assets {done}/{len(rows)}, hydro {hyd}/{len(rows)}, PCA own {own} / pooled {pooled}."]
(ASSET / "DATASET_REPORT.md").write_text("\n".join(md))
print("\n".join(md[-1:])); print("wrote", ASSET / "DATASET_REPORT.md", "and .csv")
