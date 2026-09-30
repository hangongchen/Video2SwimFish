#!/usr/bin/env python
"""Degeneracy report for a curvature npz: is kappa(s) a straight line (pure C-bend) or a wave?
Metrics from PCA_INVESTIGATION_white_bass_fish008.md: per-frame linear R^2 of kappa(s), fraction of
frames with R^2>0.95, midline lateral amplitude (BL), 2-mode explained variance (raw frames, |kBL|<4 gate),
PC zero crossings. Usage: pca_degeneracy_report.py <curvature.npz> [<curvature.npz> ...]"""
import sys, numpy as np

def report(path, max_kbl=4.0):
    d = np.load(path); kap, valid, s = d["kappa_bl"], d["valid"], d["s_stations"].astype(np.float64)
    g = valid & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < max_kbl)
    X = kap[g].astype(np.float64); K = len(s)
    A = np.vstack([s, np.ones(K)]).T
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None); fit = (A @ coef).T
    r2 = 1 - ((X - fit) ** 2).sum(1) / np.maximum(((X - X.mean(1, keepdims=True)) ** 2).sum(1), 1e-12)
    # amplitude: kappa -> theta -> midline (unit BL), max |lateral| after removing the chord
    ds = 1.0 / (K - 1); th = np.cumsum(X, 1) * ds; th -= th.mean(1, keepdims=True)
    xy = np.stack([np.cumsum(np.cos(th), 1), np.cumsum(np.sin(th), 1)], -1) * ds
    chord = xy[:, -1] - xy[:, 0]; chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    lat = (xy - xy[:, :1]) @ np.stack([-chord[:, 1], chord[:, 0]], 1)[..., None]
    amp = np.abs(lat[..., 0]).max(1)
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); evr = S ** 2 / (S ** 2).sum()
    zc = [int((np.diff(np.sign(v)) != 0).sum()) for v in Vt[:3]]
    # sign alternation along the body = wave signature: fraction of frames whose kappa(s) has >=2 zero crossings
    nzc = (np.diff(np.sign(X), axis=1) != 0).sum(1)
    print(f"{path}")
    print(f"  valid {int(valid.sum())}/{len(valid)}  gated(|kBL|<{max_kbl}) {int(g.sum())}  max|kBL| on valid {np.nanmax(np.abs(kap[valid])):.1f}  bodylen_px med {np.nanmedian(d['bodylen_px']):.0f}")
    print(f"  kappa(s) linear R2 median {np.median(r2):.3f}   frames R2>0.95 {100*(r2>0.95).mean():.1f}%   frames with >=2 sign changes {100*(nzc>=2).mean():.1f}%")
    print(f"  lateral amplitude median {np.median(amp):.3f} BL  p90 {np.percentile(amp,90):.3f} BL")
    print(f"  EVR PC1..4 {evr[:4].round(3).tolist()}  2-mode {evr[:2].sum():.3f}  PC zero-crossings {zc}")
    return dict(r2_med=float(np.median(r2)), frac_lin=float((r2>0.95).mean()), amp_med=float(np.median(amp)), evr2=float(evr[:2].sum()))

if __name__ == "__main__":
    for p in sys.argv[1:]: report(p)
