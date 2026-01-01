#!/usr/bin/env python
"""Careful silhouette IoU on the 2 cleanest-segmented fish per species (chosen by eye from the U2-Net outlines):
similarity alignment from the head label (as canonical_iou.py) -> then a 6-DOF AFFINE refinement (anisotropic
scale, rotation, shear, shift) that maximises IoU with Powell, coarse-to-fine. Reports similarity IoU, affine IoU,
and a body-only IoU (both masks opened with a kernel of 6% body length, i.e. fins stripped) so the fin-pose
mismatch can be separated from the body-shape mismatch."""
import sys, os, numpy as np, cv2
from scipy.optimize import minimize
sys.argv = ["x", "--tags", "none"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_iou.py")).read().split("tags = sorted(")[0]; exec(src)
OUT = f"{EVAL}/canonical_iou/best8"; os.makedirs(OUT, exist_ok=True)
BEST = ["bluegill_fish013", "bluegill_fish019", "catfish_fish002", "catfish_fish013",
        "lake_sturgeon_fish002", "lake_sturgeon_fish006", "white_bass_fish004", "white_bass_fish008"]

def warp_affine(R, p):
    sx, sy, ang, sh, tx, ty = p; c = S / 2
    A = np.array([[sx * np.cos(ang), -sy * np.sin(ang) + sh, 0], [sx * np.sin(ang), sy * np.cos(ang), 0]], float)
    A[:, 2] = [c - A[0, 0] * c - A[0, 1] * c + tx * S, c - A[1, 0] * c - A[1, 1] * c + ty * S]
    return cv2.warpAffine(R.astype(np.uint8) * 255, A, (S, S), flags=cv2.INTER_LINEAR) > 127

def affine_iou(P, R):
    f = lambda p: -iou(P, warp_affine(R, p))
    p0 = np.zeros(6); p0[:2] = 1
    best = (f(p0), p0)
    for init in (p0, p0 + [0.05, 0, 0, 0, 0, 0], p0 + [-0.05, 0.05, 0, 0, 0, 0]):
        r = minimize(f, init, method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 3000})
        if r.fun < best[0]: best = (r.fun, r.x)
    return -best[0], warp_affine(R, best[1]), best[1]

def strip_fins(M):
    k = max(3, int(0.06 * 0.8 * S)); k += k % 2 == 0
    return cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0

rows = []; tiles = []
for t in BEST:
    sp = t.rsplit("_fish", 1)[0]
    mp, imp = photo_mask(f"{P.RAW_VIDEO_ROOT}/{sp}_canonical_frames_cropped/{t}_canonical.jpg"); mr, _ = render_mask(f"{EVAL}/dataset_sheet/mesh_views/{t}_side.png")
    sim_m, sim_r, P, R = best_iou(mp, mr, side=HEAD_SIDE[t])
    aff, Ra, p = affine_iou(P, R)
    Pb, Rb = strip_fins(P), strip_fins(Ra); body = iou(Pb, Rb)
    rows.append((t, sim_m, sim_r, aff, body, p))
    print(f"{t}: similarity {sim_r:.3f}  affine {aff:.3f}  body-only(affine) {body:.3f}  params sx={p[0]:.2f} sy={p[1]:.2f} rot={np.degrees(p[2]):.1f}deg shear={p[3]:.2f}", flush=True)
    ov = np.zeros((S, S, 3), np.uint8); ov[P & Ra] = (255, 255, 255); ov[P & ~Ra] = (60, 60, 230); ov[~P & Ra] = (230, 160, 40)
    cv2.putText(ov, f"{t}  sim {sim_r:.2f}  affine {aff:.2f}  body {body:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    sc = min(S / imp.shape[1], S / imp.shape[0]); ph = cv2.resize(imp, (int(imp.shape[1] * sc), int(imp.shape[0] * sc)))
    pm = cv2.resize(mp * 255, ph.shape[1::-1], interpolation=cv2.INTER_NEAREST); ph[pm == 0] = (ph[pm == 0] * 0.35).astype(np.uint8)
    pad = np.zeros((S, S, 3), np.uint8); pad[:ph.shape[0], :ph.shape[1]] = ph
    rm = cv2.imread(f"{EVAL}/dataset_sheet/mesh_views/{t}_side.png"); ys, xs = np.nonzero(mr); rm = rm[ys.min() - 5:ys.max() + 5, xs.min() - 5:xs.max() + 5]
    sc = min(S / rm.shape[1], S / rm.shape[0]); rm = cv2.resize(rm, (int(rm.shape[1] * sc), int(rm.shape[0] * sc))); pad2 = np.zeros((S, S, 3), np.uint8); pad2[:rm.shape[0], :rm.shape[1]] = rm
    tiles.append(np.hstack([pad, pad2, ov])); cv2.imwrite(f"{OUT}/{t}_overlay.png", tiles[-1])
cv2.imwrite(f"{OUT}/best8_sheet.png", np.vstack(tiles))
sp_names = sorted({r[0].rsplit("_fish", 1)[0] for r in rows})
with open(f"{OUT}/best8_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Silhouette IoU between the generated mesh (orthographic lateral render) and its canonical input frame for the two most cleanly segmented fish per species. Similarity: 4-DOF alignment; affine: 6-DOF; body-only: both silhouettes with fins morphologically stripped before the affine IoU.}\n\\label{tab:canonical-iou-best8}\n\\begin{center}\\small\n\\begin{tabular}{llccc}\n\\toprule\nSpecies & Fish & IoU (similarity) & IoU (affine) & IoU (body only) \\\\\n\\midrule\n")
    for t, sm, sr, af, bo, p in rows: f.write(f"{t.rsplit('_fish',1)[0].replace('_',' ')} & {t.rsplit('_fish',1)[1]} & {sr:.2f} & {af:.2f} & {bo:.2f} \\\\\n")
    f.write("\\midrule\n\\textbf{mean} & & %.2f & %.2f & %.2f \\\\\n\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}\n" % (np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows])))
print("mean similarity %.3f  affine %.3f  body-only %.3f" % (np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows]))); print("wrote", OUT)
