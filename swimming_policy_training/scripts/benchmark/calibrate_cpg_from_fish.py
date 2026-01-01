#!/usr/bin/env python
"""Per-fish CPG calibration (generalizes scripts/zef_pca_rl/calibrate_cpg_from_zef.py): same maths,
input = the fish's OWN top-view curvature (data/species_manifold/<sp>_fix/top/fish<N>_curvature.npz,
clean frames, longest contiguous runs), output = data/fish_assets/<tag>/cpg_params.npz (same schema).
Usage: python calibrate_cpg_from_fish.py --tag catfish_fish002"""
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--max_kbl", type=float, default=4.0); a = ap.parse_args()
sp, num = a.tag.rsplit("_fish", 1); d = np.load(REPO / f"data/species_manifold/{sp}_fix/top/fish{num}_curvature.npz")
kb = d["kappa_bl"].astype(np.float64); fps = float(d["fps"]); S = kb.shape[1]
valid = d["valid"].astype(bool) & (np.nan_to_num(np.abs(kb), nan=np.inf).max(1) < a.max_kbl)
mean_k = np.nanmean(kb[valid], axis=0)
def _runs(mask, min_s):
    idx = np.where(mask)[0]; rr = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if idx.size else []
    return [r for r in rr if len(r) >= int(min_s * fps)]
runs = _runs(valid, 2.0); fallback = False
if not runs:   # fragmented clean data (e.g. lake_sturgeon 001/020): use the tracker-valid runs with |kappa| clipped
    fallback = True; runs = _runs(d["valid"].astype(bool), 2.0); kb = np.clip(np.nan_to_num(kb, nan=0.0), -a.max_kbl, a.max_kbl)
assert runs, "no valid run >= 2 s"
segs = [kb[r] for r in runs]
def per_seg(seg):
    x = seg - seg.mean(0, keepdims=True); post = x[:, S // 2:].mean(1)
    freqs = np.fft.rfftfreq(len(post), 1.0 / fps); P = np.abs(np.fft.rfft(post * np.hanning(len(post)))) ** 2; P[freqs < 0.3] = 0
    return x, float(freqs[int(np.argmax(P))])
xs, f0s = zip(*[per_seg(s) for s in segs]); w = np.array([len(s) for s in segs], float)
f0 = float(np.average(f0s, weights=w)); lo, hi = max(0.3, 0.5 * f0), min(2.0 * f0, fps / 2 - 0.5)
b_, a_ = butter(3, [lo / (fps / 2), hi / (fps / 2)], btype="band")
amps, phs, bodies = [], [], []
for x, seg in zip(xs, segs):
    xb = filtfilt(b_, a_, x, axis=0); an = hilbert(xb, axis=0); amps.append(np.abs(an)); phs.append(np.angle(an)); bodies.append(seg.mean(1))
amp = np.concatenate(amps); E = amp.mean(0); E = E / E.max(); s_ref = int(np.argmax(amp.mean(0)))
phi = np.unwrap(np.angle(np.concatenate([np.exp(1j * (p - p[:, s_ref:s_ref + 1])) for p in phs]).mean(0))); phi = phi - phi[0]
A_max = float(np.percentile(amp[:, int(np.argmax(E))], 99))
inst_f = np.concatenate([np.diff(np.unwrap(p[:, s_ref])) * fps / (2 * np.pi) for p in phs])
f_lo = float(np.clip(np.percentile(inst_f, 5), 0.3, 3.5)); f_hi = float(np.clip(np.percentile(inst_f, 95), 0.3, 3.5))
body = np.concatenate(bodies); b_max = float(np.percentile(np.abs(body - body.mean()), 99))
step = max(1, int(round(fps / 30.0)))                       # frames per 1/30 s control step
st = int(np.argmax(E)); dA_max = float(np.percentile(np.abs(np.diff(amp[::step, st])), 99))
df_max = float(np.clip(np.percentile(np.abs(np.diff(inst_f[::step])), 99), 0.02, 0.5)); db_max = float(np.percentile(np.abs(np.diff(body[::step])), 99))
out = REPO / f"data/fish_assets/{a.tag}/cpg_params.npz"
np.savez(out, envelope=E.astype(np.float32), phase=phi.astype(np.float32), kappa_mean=mean_k.astype(np.float32), f0=f0, f_lo=f_lo, f_hi=f_hi,
         A_max=A_max, b_max=b_max, dA_max=dA_max, df_max=df_max, db_max=db_max, s_ref=s_ref, band=(lo, hi), n_runs=len(runs), n_frames=int(sum(w)), clipped_fallback=fallback)
print(f"[cpg:{a.tag}] runs={len(runs)} frames={int(sum(w))} f0={f0:.2f} Hz f=[{f_lo:.2f},{f_hi:.2f}] A_max={A_max:.2f} b_max={b_max:.2f} rates dA={dA_max:.3f} df={df_max:.3f} db={db_max:.3f} -> {out}")
