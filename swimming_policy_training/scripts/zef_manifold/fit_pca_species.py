#!/usr/bin/env python
"""Per-fish PCA basis from ONE fish's own video curvature (same math and output keys as
fit_pca.py, which is hardcoded to the pooled ZeF-05 dataset). Output is a drop-in for
`env.pca_basis_path` (salmon_swim_pca_env.py loads mean/components/explained_variance_ratio).

Usage:
    python fit_pca_species.py --species catfish --fish_id 001 --view top
      -> data/species_manifold/catfish/top/fish001_pca_basis.npz
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "species_manifold"

ap = argparse.ArgumentParser()
ap.add_argument("--species", required=True)
ap.add_argument("--fish_id", required=True, help="e.g. 001")
ap.add_argument("--view", default="top", choices=["top", "front"])
ap.add_argument("--resample_fps", type=float, default=None,
                help="resample the coeff time series to this fps before saving (the PCA env derives its "
                     "per-control-step rate cap from coeffs[:-2]-coeffs[2:], i.e. it assumes 60 fps data; "
                     "gaps between valid frames stay NaN)")
ap.add_argument("--max_kbl", type=float, default=None,
                help="drop frames whose max |kappa*BL| exceeds this (end-of-midline noise guard; real fish stay under ~4)")
args = ap.parse_args()

src = ROOT / args.species / args.view / f"fish{args.fish_id}_curvature.npz"
out = ROOT / args.species / args.view / f"fish{args.fish_id}_pca_basis.npz"

d = np.load(src)
kap, valid = d["kappa_bl"], d["valid"]
if args.max_kbl is not None:
    valid = valid & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < args.max_kbl)
X = kap[valid]
mu = X.mean(0)
Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
evr = S ** 2 / np.sum(S ** 2)

flip = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(1)])
flip[flip == 0] = 1.0
Vt = Vt * flip[:, None]

T, K = kap.shape
coeffs = np.full((T, K), np.nan, np.float32)
coeffs[valid] = ((kap[valid] - mu) @ Vt.T).astype(np.float32)

if args.resample_fps is not None:
    # linear interpolation of each PC coefficient onto a uniform grid, only inside runs of
    # consecutive valid frames (so a tracking gap never gets bridged by a fake straight line)
    t = d["t_sec"].astype(np.float64)
    fps_src = float(d["fps"])
    step = 1.0 / args.resample_fps
    grid = np.arange(t[0], t[-1] + 1e-9, step)
    out_c = np.full((len(grid), K), np.nan, np.float32)
    vidx = np.where(valid)[0]
    # split into runs of consecutive valid source frames
    breaks = np.where(np.diff(vidx) > 1)[0] + 1
    for run in np.split(vidx, breaks):
        if len(run) < 2:
            continue
        t0, t1 = t[run[0]], t[run[-1]]
        sel = (grid >= t0) & (grid <= t1)
        for j in range(K):
            out_c[sel, j] = np.interp(grid[sel], t[run], coeffs[run, j])
    print(f"  coeffs resampled {fps_src:.1f} fps -> {args.resample_fps:.0f} fps: {len(coeffs)} -> {len(out_c)} rows, "
          f"{np.isfinite(out_c).all(1).sum()} valid")
    coeffs = out_c

cum = np.cumsum(evr)
n90, n95, n99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
np.savez(out, mean=mu.astype(np.float32), components=Vt.astype(np.float32),
         explained_variance_ratio=evr.astype(np.float32), singular_values=S.astype(np.float32),
         coeffs=coeffs, n90=n90, n95=n95, n99=n99, source=str(src))

print(f"[pca:{args.species}/{args.view}/fish{args.fish_id}] {X.shape[0]} valid frames of {T}, K={K}")
print("  EVR per PC:", " ".join(f"{v:.4f}" for v in evr[:8]), "...")
print("  cumulative:", " ".join(f"{v:.4f}" for v in cum[:8]), "...")
print(f"  components for 90%: {n90}   95%: {n95}   99%: {n99}")
print(f"  wrote {out}")
