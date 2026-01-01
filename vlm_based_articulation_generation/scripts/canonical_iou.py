#!/usr/bin/env python
"""Silhouette IoU between each generated fish mesh (Blender orthographic LATERAL render,
$EVAL_OUT/dataset_sheet/mesh_views/<tag>_side.png -- make it with scripts/render_mesh_views.py
--mesh <glb> --out_prefix $EVAL_OUT/dataset_sheet/mesh_views/<tag>) and the canonical input frame it was
generated from ($RAW_VIDEO_ROOT/<species>_canonical_frames_cropped/<tag>_canonical.jpg).

Photo mask: U2-Net (rembg) alpha (fallback: GrabCut). Render mask: colour
distance from the uniform background. The two silhouettes are put in the same frame by a SIMILARITY transform
from image moments (centroid, principal-axis angle, extent along the axis); head ends matched from a by-eye head-side label per photo (HEAD_SIDE; renders are all head-left), dorsal flip by best IoU ("moment-aligned IoU"). A small local search (scale +-12%, angle +-8 deg, shift +-6%) then gives the
"refined IoU" (upper bound under rigid 2-D alignment). Controls: the same photo vs OTHER fish's renders of the
same species (specificity) and vs an ellipse with the photo's own moments (shape-vs-blob).

Usage: canonical_iou.py [--tags a,b] [--out $EVAL_OUT/canonical_iou] [--debug]
"""
import argparse, csv, glob, os, json
import sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v2sf_paths as P
EVAL = str(P.EVAL_OUT)
ap = argparse.ArgumentParser(); ap.add_argument("--tags", default=None); ap.add_argument("--out", default=f"{EVAL}/canonical_iou")
ap.add_argument("--debug", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
S = 512   # common frame

def photo_mask(path):
    im = cv2.imread(path); h, w = im.shape[:2]
    # preferred: U2-Net (rembg) alpha computed once by /tmp/rembg_venv into $EVAL_OUT/canonical_iou/photo_masks/
    # (GrabCut grabbed the tank floor for the catfish / sturgeon frames and over-segmented the bluegill)
    alpha = f"{EVAL}/canonical_iou/photo_masks/{os.path.basename(path).replace('_canonical.jpg', '')}_alpha.png"
    if os.path.exists(alpha):
        # alpha > 30 (not 128): the translucent caudal / pectoral fins sit at alpha 30-120 and were cut off
        fg = (cv2.imread(alpha, cv2.IMREAD_GRAYSCALE) > 30).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n > 1: fg = (lab == (1 + np.argmax(st[1:, 4]))).astype(np.uint8)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fg = np.zeros_like(fg); cv2.drawContours(fg, cnts, -1, 1, -1)
        return fg, im
    m = np.zeros((h, w), np.uint8); bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    rect = (int(0.03 * w), int(0.06 * h), int(0.94 * w), int(0.88 * h))
    cv2.grabCut(im, m, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1: fg = (lab == (1 + np.argmax(st[1:, 4]))).astype(np.uint8)   # largest blob = the fish
    return fg, im

def render_mask(path):
    im = cv2.imread(path); bg = im[2, 2].astype(int)
    fg = (np.abs(im.astype(int) - bg).sum(-1) > 40).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1: fg = (lab == (1 + np.argmax(st[1:, 4]))).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # fill interior holes where the
    fg = np.zeros_like(fg); cv2.drawContours(fg, cnts, -1, 1, -1)                 # dark texture matches the bg
    return fg, im

def moments(m):
    ys, xs = np.nonzero(m); c = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.stack([xs - c[0], ys - c[1]])); w, v = np.linalg.eigh(cov); ax = v[:, np.argmax(w)]
    proj = (xs - c[0]) * ax[0] + (ys - c[1]) * ax[1]; L = proj.max() - proj.min()
    return c, np.arctan2(ax[1], ax[0]), L

# Head side of the fish in each canonical PHOTO, labelled by eye (2026-09-25) from the crops: L/R/U/D = the head is
# left/right/up/down of the fish centroid in the crop; '?' = could not tell (falls back to the max-IoU flip). All 80
# mesh renders have the head on the LEFT (checked on a sheet of all renders), so only the photo needs a label.
# Automatic width rules ("head 30% wider", "peduncle = narrowest neck") disagreed with the true side on 16-18 fish.
HEAD_SIDE = {
 "bluegill_fish001":"R","bluegill_fish002":"R","bluegill_fish003":"R","bluegill_fish004":"L","bluegill_fish005":"L",
 "bluegill_fish006":"D","bluegill_fish007":"R","bluegill_fish008":"R","bluegill_fish009":"R","bluegill_fish010":"R",
 "bluegill_fish011":"L","bluegill_fish012":"L","bluegill_fish013":"L","bluegill_fish014":"R","bluegill_fish015":"R",
 "bluegill_fish016":"R","bluegill_fish017":"L","bluegill_fish018":"R","bluegill_fish019":"L","bluegill_fish020":"R",
 "catfish_fish001":"L","catfish_fish002":"L","catfish_fish003":"R","catfish_fish004":"L","catfish_fish005":"?",
 "catfish_fish006":"R","catfish_fish007":"L","catfish_fish008":"R","catfish_fish009":"R","catfish_fish010":"R",
 "catfish_fish011":"D","catfish_fish012":"R","catfish_fish013":"L","catfish_fish014":"?","catfish_fish015":"R",
 "catfish_fish016":"R","catfish_fish017":"L","catfish_fish018":"L","catfish_fish019":"R","catfish_fish020":"R",
 "lake_sturgeon_fish001":"L","lake_sturgeon_fish002":"L","lake_sturgeon_fish003":"L","lake_sturgeon_fish004":"L","lake_sturgeon_fish005":"L",
 "lake_sturgeon_fish006":"L","lake_sturgeon_fish007":"R","lake_sturgeon_fish008":"L","lake_sturgeon_fish009":"L","lake_sturgeon_fish010":"L",
 "lake_sturgeon_fish011":"L","lake_sturgeon_fish012":"R","lake_sturgeon_fish013":"R","lake_sturgeon_fish014":"L","lake_sturgeon_fish015":"L",
 "lake_sturgeon_fish016":"L","lake_sturgeon_fish017":"L","lake_sturgeon_fish018":"R","lake_sturgeon_fish019":"L","lake_sturgeon_fish020":"L",
 "white_bass_fish001":"L","white_bass_fish002":"R","white_bass_fish003":"L","white_bass_fish004":"L","white_bass_fish005":"L",
 "white_bass_fish006":"R","white_bass_fish007":"R","white_bass_fish008":"R","white_bass_fish009":"L","white_bass_fish010":"R",
 "white_bass_fish011":"R","white_bass_fish012":"L","white_bass_fish013":"R","white_bass_fish014":"R","white_bass_fish015":"L",
 "white_bass_fish016":"R","white_bass_fish017":"R","white_bass_fish018":"L","white_bass_fish019":"L","white_bass_fish020":"R"}

def canon_M(m, dscale=1.0, dang=0.0, dx=0.0, dy=0.0):
    c, ang, L = moments(m); s = 0.8 * S / L * dscale
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), np.degrees(ang) + dang, s)
    M[:, 2] += np.array([S / 2, S / 2]) - c + np.array([dx, dy]) * S
    return M, c

def photo_head_left(mp, side):
    """True if the labelled head lands on the LEFT of the canonical frame; None if unlabelled."""
    if side not in "LRUD" or side == "": return None
    M, c = canon_M(mp); off = {"L": (-1, 0), "R": (1, 0), "U": (0, -1), "D": (0, 1)}[side]
    hp = M @ np.array([c[0] + 50 * off[0], c[1] + 50 * off[1], 1.0]); return bool(hp[0] < S / 2)

def render_head_left(mr):
    """All renders have the head on the LEFT in the image, but the moment rotation has a 180-degree sign
    ambiguity, so map the leftmost mask pixel through the same transform and check where it lands."""
    M, c = canon_M(mr); ys, xs = np.nonzero(mr); x = xs.min(); y = ys[xs == x].mean()
    return bool((M @ np.array([x, y, 1.0]))[0] < S / 2)

def canon(m, flipx=False, flipy=False, dscale=1.0, dang=0.0, dx=0.0, dy=0.0):
    """Warp mask into the common frame: principal axis -> +x, length -> 0.8*S, centroid -> centre."""
    c, ang, L = moments(m); s = 0.8 * S / L * dscale
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), np.degrees(ang) + dang, s)
    M[:, 2] += np.array([S / 2, S / 2]) - c + np.array([dx, dy]) * S
    out = cv2.warpAffine(m * 255, M, (S, S), flags=cv2.INTER_NEAREST) > 127
    if flipx: out = out[:, ::-1]
    if flipy: out = out[::-1, :]
    return out

def iou(x, y): return (x & y).sum() / max(1, (x | y).sum())

def head_on_left(C):
    """Head-end rule on a canonical-frame mask (axis along x): the caudal PEDUNCLE is the narrowest interior
    point of the width profile (the caudal fin fans out again behind it), and the snout has no such neck. So
    the end nearer the interior width minimum (10-90 %% of the length, profile smoothed) is the TAIL. A
    'head 30 %% is wider' rule failed on catfish / sturgeon (their caudal fan is wider than the flat head).
    Applied identically to photo and render; disagreements with the max-IoU flip are reported."""
    cols = np.nonzero(C.any(0))[0]; x0, x1 = cols.min(), cols.max(); L = x1 - x0 + 1
    w = C.sum(0)[x0:x1 + 1].astype(float); k = max(3, int(0.04 * L)); w = np.convolve(w, np.ones(k) / k, mode="same")
    lo, hi = int(0.10 * L), int(0.90 * L); xm = lo + int(np.argmin(w[lo:hi]))
    return xm > L / 2            # neck in the right half -> tail on the right -> head on the left

def best_iou(mp, mr, refine=True, side=None):
    P = canon(mp); hl = photo_head_left(mp, side)
    if hl is False: P = P[:, ::-1]                  # photo: labelled head to the left
    fxs = (not render_head_left(mr),) if hl is not None else (False, True)   # render: flip so its head is left; unlabelled photo -> try both
    best = (0, None)
    for fx in fxs:
        for fy in (False, True):                    # dorsal side: keep the better of up/down (renders are z-up, photos upright)
            R = canon(mr, fx, fy); v = iou(P, R)
            if v > best[0]: best = (v, (fx, fy, 1.0, 0.0, 0.0, 0.0), R)
    moment_iou = best[0]
    if refine:
        fx, fy = best[1][:2]; cur = best[1][2:]; b = best[0]
        for _ in range(3):
            for i, step in enumerate((0.04, 2.0, 0.02, 0.02)):
                for sgn in (-1, 1):
                    p = list(cur); p[i] += sgn * step
                    R = canon(mr, fx, fy, *p); v = iou(P, R)
                    if v > b: b, cur, best = v, p, (v, (fx, fy, *p), R)
        best = (b, best[1], best[2])
    return moment_iou, best[0], P, best[2]

def ellipse_like(m):
    c, ang, L = moments(m); ys, xs = np.nonzero(m); ax = np.array([np.cos(ang), np.sin(ang)]); pe = np.array([-ax[1], ax[0]])
    W = ((xs - c[0]) * pe[0] + (ys - c[1]) * pe[1]); W = W.max() - W.min()
    e = np.zeros_like(m); cv2.ellipse(e, (int(c[0]), int(c[1])), (int(L / 2), int(W / 2)), float(np.degrees(ang)), 0, 360, 1, -1); return e

tags = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{P.ASSET_OUT}/*_fish*/"))
if a.tags: tags = a.tags.split(",")
data = {}
for t in tags:
    sp = t.rsplit("_fish", 1)[0]
    fp = f"{P.RAW_VIDEO_ROOT}/{sp}_canonical_frames_cropped/{t}_canonical.jpg"; fr = f"{EVAL}/dataset_sheet/mesh_views/{t}_side.png"
    if not (os.path.exists(fp) and os.path.exists(fr)): print("skip", t); continue
    mp, imp = photo_mask(fp); mr, imr = render_mask(fr)
    if mp.sum() < 200 or mr.sum() < 200: print("empty mask", t); continue
    data[t] = (sp, mp, mr, imp)
rows = []
for t, (sp, mp, mr, imp) in data.items():
    m_iou, r_iou, P, R = best_iou(mp, mr, side=HEAD_SIDE.get(t, '?'))
    e_iou = best_iou(mp, ellipse_like(mp), refine=False, side=HEAD_SIDE.get(t, '?'))[1]
    others = [best_iou(mp, data[o][2], refine=False, side=HEAD_SIDE.get(t, '?'))[0] for o in data if o != t and data[o][0] == sp]
    rows.append(dict(fish=t, species=sp, head_label=HEAD_SIDE.get(t, '?'), iou_moment=m_iou, iou_refined=r_iou, iou_ellipse=e_iou,
                     iou_other_fish_same_species=float(np.mean(others)) if others else float("nan"), n_others=len(others)))
    ov = np.zeros((S, S, 3), np.uint8); ov[P & R] = (255, 255, 255); ov[P & ~R] = (60, 60, 230); ov[~P & R] = (230, 160, 40)
    for M_, col in ((P, (0, 255, 0)), (R, (0, 200, 255))):   # head markers: green = photo head end, yellow = mesh head end
        ys_, xs_ = np.nonzero(M_); xh = xs_.min(); cv2.circle(ov, (int(xh), int(ys_[xs_ == xh].mean())), 7, col, 2)
    cv2.putText(ov, f"{t}  IoU {r_iou:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(ov, "red=photo only  orange=mesh only  white=both", (8, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    sc = min(S / imp.shape[1], S / imp.shape[0]); ph = cv2.resize(imp, (int(imp.shape[1] * sc), int(imp.shape[0] * sc))); pm = cv2.resize(mp * 255, ph.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    ph[pm == 0] = (ph[pm == 0] * 0.35).astype(np.uint8); pad = np.zeros((S, S, 3), np.uint8); pad[:ph.shape[0], :ph.shape[1]] = ph
    cv2.imwrite(f"{a.out}/{t}_overlay.png", np.hstack([pad, ov]))
    print(f"{t}: moment {m_iou:.3f} refined {r_iou:.3f} ellipse {e_iou:.3f} other-fish {rows[-1]['iou_other_fish_same_species']:.3f}", flush=True)
with open(f"{a.out}/canonical_iou.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# species summary
sp_rows = []
for sp in sorted({r["species"] for r in rows}):
    rr = [r for r in rows if r["species"] == sp]
    g = lambda k: (np.mean([r[k] for r in rr]), np.std([r[k] for r in rr]))
    sp_rows.append((sp, len(rr), g("iou_refined"), g("iou_moment"), g("iou_other_fish_same_species"), g("iou_ellipse")))
allr = ("all", len(rows), *[(np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])) for k in ("iou_refined", "iou_moment", "iou_other_fish_same_species", "iou_ellipse")])
with open(f"{a.out}/canonical_iou_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Silhouette IoU between the generated mesh (orthographic lateral render) and the canonical input frame, after 2-D similarity alignment. Controls: the same photo against the other fish of the same species (specificity) and against an ellipse with the photo's own moments (shape vs.\\ blob). Mean $\\pm$ std over fish.}\n\\label{tab:canonical-iou}\n\\begin{center}\\small\n\\begin{tabular}{lcccc}\n\\toprule\nSpecies & $n$ & IoU (own mesh) $\\uparrow$ & IoU (other fish, same species) & IoU (ellipse) \\\\\n\\midrule\n")
    for sp, n, ref, mom, oth, ell in sp_rows + [allr]:
        name = sp.replace("_", " ") if sp != "all" else "\\textbf{all}"
        f.write(f"{name} & {n} & {ref[0]:.2f} $\\pm$ {ref[1]:.2f} & {oth[0]:.2f} $\\pm$ {oth[1]:.2f} & {ell[0]:.2f} $\\pm$ {ell[1]:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
print("species summary (refined IoU / other-fish / ellipse):")
for sp, n, ref, mom, oth, ell in sp_rows + [allr]: print(f"  {sp:14s} n={n:2d}  own {ref[0]:.3f}+-{ref[1]:.3f}  moment-only {mom[0]:.3f}  other-fish {oth[0]:.3f}  ellipse {ell[0]:.3f}")
print("wrote", a.out)
