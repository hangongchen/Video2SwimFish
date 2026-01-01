#!/usr/bin/env python
"""Validate a curvature extraction: validity, |kBL| range, head/tail sanity (fish should MOVE toward s=0 and bend
more at s=1), travelling-wave phase lag, and a montage of 8 evenly spaced valid frames with the mask outline,
midline and s=0 (green) marker. Usage: validate_extraction.py <species_dir> <fish_id> <video> [--out png]"""
import sys, argparse, numpy as np, cv2
from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; sys.path.insert(0, str(_R / "scripts" / "zef_manifold")); import extract_species_curvature as E
ap = argparse.ArgumentParser(); ap.add_argument("species"); ap.add_argument("fish_id"); ap.add_argument("video"); ap.add_argument("--out", default=None); a = ap.parse_args()
p = f"{_R}/data/species_manifold/{a.species}/top/fish{a.fish_id}_curvature.npz"; d = np.load(p)
kap, valid, mid, L, fps, fr = d["kappa_bl"], d["valid"], d["midline_px"], d["bodylen_px"], float(d["fps"]), d["frame"]
g = valid & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < 4); k = kap[g]
c = np.nanmean(mid, 1); vel = np.diff(c, axis=0); ax = mid[:-1, 0] - mid[:-1, -1]; ok = valid[:-1] & valid[1:] & (np.linalg.norm(vel, axis=1) > 0.02 * np.nanmedian(L))
cos = (vel[ok] * ax[ok]).sum(1) / (np.linalg.norm(vel[ok], axis=1) * np.linalg.norm(ax[ok], axis=1) + 1e-9)
flips = int(d["flipped"][valid].sum()) if "flipped" in d.files else -1
print(f"[{a.species}/fish{a.fish_id}] valid {int(valid.sum())}/{len(valid)}  gated(|kBL|<4) {int(g.sum())}  max|kBL| {np.nanmax(np.abs(kap[valid])):.1f}  bodylen_px {np.nanmedian(L):.0f}  head/tail flips {flips}")
print(f"   HEAD CHECK: moving toward s=0 in {100*(cos>0).mean():.0f}% of {ok.sum()} moving frames (want >>50%);  mean|kappa| head(s<0.25) {np.abs(k[:, :5]).mean():.2f} vs tail(s>0.75) {np.abs(k[:, -5:]).mean():.2f} (want tail > head)")
# travelling wave lag (same as travelling_wave_report)
s = d["s_stations"]; ia, ib = int(np.argmin(abs(s - 0.45))), int(np.argmin(abs(s - 0.9))); idx = np.where(g)[0]; br = np.where(np.diff(idx) > 1)[0] + 1; lags = []
for r in [r for r in np.split(idx, br) if len(r) >= 48]:
    ka, kb = kap[r, ia] - kap[r, ia].mean(), kap[r, ib] - kap[r, ib].mean(); F = np.fft.rfft(kb * np.hanning(len(kb))); f = np.fft.rfftfreq(len(kb), 1 / fps); band = (f > 0.3) & (f < 6)
    if not band.any(): continue
    j = np.argmin(abs(f - f[band][np.argmax(abs(F[band]))])); lags.append((np.degrees(np.angle(np.fft.rfft(ka * np.hanning(len(ka)))[j] * np.conj(F[j]))), len(r)))
if lags: print(f"   WAVE: head->tail phase lag {sum(l*w for l,w in lags)/sum(w for _,w in lags):+.0f} deg over {len(lags)} runs; per-run signs {''.join('+' if l>0 else '-' for l,_ in lags)}")
if a.out:
    cap = cv2.VideoCapture(a.video); n_raw = int(fr[-1]) + 1; bg = E.median_background(cap, n_raw); vi = np.where(valid)[0]; pick = vi[np.linspace(0, len(vi) - 1, 8).astype(int)]; tiles = []
    rt = float(d["ratio_thresh"]) if "ratio_thresh" in d.files and np.isfinite(d["ratio_thresh"]) else None
    for i in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[i])); _, im = cap.read(); gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); cx, cy = np.nanmean(mid[i], 0); x0, y0 = int(cx - 150), int(cy - 150)
        lab, _ = E.segment_frame(gray, bg, thresh=60, open_px=max(1, int(0.019 * np.nanmedian(L))), close_px=9, ratio_thresh=rt)
        crop = im[max(0, y0):y0 + 300, max(0, x0):x0 + 300].copy(); mk = (lab[max(0, y0):y0 + 300, max(0, x0):x0 + 300] > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop, cnts, -1, (255, 200, 0), 1)
        pts = (mid[i] - [max(0, x0), max(0, y0)]).astype(np.int32); cv2.polylines(crop, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 1); cv2.circle(crop, tuple(pts[0]), 5, (0, 255, 0), -1)
        crop = cv2.resize(crop, (360, 360)); cv2.putText(crop, f"{i} t={i/fps:.0f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1); tiles.append(crop)
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])); print("   montage ->", a.out)
