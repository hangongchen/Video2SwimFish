#!/usr/bin/env python
"""Species-POOLED PCA basis as a fallback for fish whose own top-view curvature is unusable
(fish that never moved -> invisible to background subtraction, e.g. bluegill_fish009; three fish
in the tank -> tracker identity swaps, e.g. white_bass_fish019; or < --min_clean clean frames).

Pools the clean frames (max |kappa*BL| < --max_kbl) of every OTHER fish of the species that has at
least --min_clean clean frames, fits one PCA (same math/keys as fit_pca_species.py), and writes it
as fish<id>_pca_basis.npz for each target fish with `source="species_pooled_fallback"` and the
pooled coefficient time series (so the PCA env's a_max / da_max statistics stay species-typical).
Also writes fish<id>_reference.npz as a copy of the best-covered donor's reference (flagged).

Usage: fit_pca_species_pooled.py --species bluegill_fix --view top --targets 009 [014 ...]
       fit_pca_species_pooled.py --species bluegill_fix --view top --auto   # targets = fish with no basis or < min_clean
"""
import argparse, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "species_manifold"
ap = argparse.ArgumentParser()
ap.add_argument("--species", required=True); ap.add_argument("--view", default="top")
ap.add_argument("--targets", nargs="*", default=[]); ap.add_argument("--auto", action="store_true")
ap.add_argument("--max_kbl", type=float, default=4.0); ap.add_argument("--min_clean", type=int, default=100)
ap.add_argument("--resample_fps", type=float, default=60.0)
a = ap.parse_args()
d_dir = ROOT / a.species / a.view

def load(fid):
    d = np.load(d_dir / f"fish{fid}_curvature.npz")
    kap, valid = d["kappa_bl"], d["valid"]
    clean = valid & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < a.max_kbl)
    return d, kap, clean

ids = sorted(p.stem[4:7] for p in d_dir.glob("fish???_curvature.npz"))
info = {fid: load(fid) for fid in ids}
n_clean = {fid: int(info[fid][2].sum()) for fid in ids}
targets = list(a.targets)
if a.auto:
    targets += [fid for fid in ids if fid not in targets and
                (n_clean[fid] < a.min_clean or not (d_dir / f"fish{fid}_pca_basis.npz").exists())]
donors = [fid for fid in ids if fid not in targets and n_clean[fid] >= a.min_clean]
print(f"[pooled:{a.species}/{a.view}] targets={targets} donors={donors} clean={ {f: n_clean[f] for f in ids} }")
if not targets or not donors:
    raise SystemExit("nothing to do")

X = np.concatenate([info[f][1][info[f][2]] for f in donors]); mu = X.mean(0)
U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); evr = S ** 2 / np.sum(S ** 2)
flip = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(1)]); flip[flip == 0] = 1.0; Vt = Vt * flip[:, None]
K = X.shape[1]
# pooled coefficient time series at resample_fps (per donor, runs of consecutive clean frames only), concatenated
chunks = []
for f in donors:
    d, kap, clean = info[f]; t = d["t_sec"].astype(np.float64); coeffs = np.full(kap.shape, np.nan, np.float32)
    coeffs[clean] = ((kap[clean] - mu) @ Vt.T).astype(np.float32)
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / a.resample_fps); out_c = np.full((len(grid), K), np.nan, np.float32)
    vidx = np.where(clean)[0]; breaks = np.where(np.diff(vidx) > 1)[0] + 1
    for run in np.split(vidx, breaks):
        if len(run) < 2: continue
        sel = (grid >= t[run[0]]) & (grid <= t[run[-1]])
        for j in range(K): out_c[sel, j] = np.interp(grid[sel], t[run], coeffs[run, j])
    chunks.append(out_c[np.isfinite(out_c).any(1)])
    chunks.append(np.full((2, K), np.nan, np.float32))   # gap marker between donors
coeffs = np.concatenate(chunks)
cum = np.cumsum(evr); n90, n95, n99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
best_donor = max(donors, key=lambda f: n_clean[f])
for fid in targets:
    out = d_dir / f"fish{fid}_pca_basis.npz"
    np.savez(out, mean=mu.astype(np.float32), components=Vt.astype(np.float32), explained_variance_ratio=evr.astype(np.float32),
             singular_values=S.astype(np.float32), coeffs=coeffs, n90=n90, n95=n95, n99=n99,
             source="species_pooled_fallback", pooled_from=np.array(donors), own_clean_frames=n_clean.get(fid, 0))
    ref_src = d_dir / f"fish{best_donor}_reference.npz"; ref_dst = d_dir / f"fish{fid}_reference.npz"
    if ref_src.exists():
        r = dict(np.load(ref_src, allow_pickle=True)); r["source"] = f"copied_from_fish{best_donor}_species_fallback"
        np.savez(ref_dst, **r)
    print(f"  wrote {out.name} (pooled {X.shape[0]} frames from {len(donors)} fish; 90%={n90} PCs) "
          f"+ reference copied from fish{best_donor}")
