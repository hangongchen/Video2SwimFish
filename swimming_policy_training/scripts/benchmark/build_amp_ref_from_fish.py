#!/usr/bin/env python
"""Per-fish AMP reference (same 46-dim feature contract as outputs/zef05_amp_ref/amp_reference_3d_v2.npz,
built from the ZeF-05 recording) from the fish's OWN top-view video:
  bend (2K=40): K=20 midline points from the curvature npz `midline_px` (head first), body-frame lateral
                offset / arc length exactly as bend3d(); vertical offsets = 0 (single top view)
  motion (6):   speed_bl, dir_nose/left/up (velocity in the body basis; up = 0), yaw_body (from psi), pitch_body = 0
Image y points down -> flipped to a right-handed top view so left/yaw signs match the sim's world frame.
Usage: python build_amp_ref_from_fish.py --tag catfish_fish002  -> data/fish_assets/<tag>/amp_reference.npz"""
import argparse
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; K = 20; TRAVEL_MIN_BL = 0.05
NAMES = ("speed_bl", "dir_nose", "dir_left", "dir_up", "yaw_body", "pitch_body")

def smooth(a, w=5):
    if w <= 1 or len(a) < w: return a
    pad = w // 2; ap = np.concatenate([np.repeat(a[:1], pad), a, np.repeat(a[-1:], pad)]); return np.convolve(ap, np.ones(w) / w, mode="valid")

def bend3d(M):
    head = M[0]; rel = M - head; tan = rel[max(1, K // 4)] - rel[0]; tan /= np.linalg.norm(tan) + 1e-9
    up = np.array([0.0, 0.0, 1.0]); up = up - up.dot(tan) * tan; up /= np.linalg.norm(up) + 1e-9; left = np.cross(up, tan)
    bl = np.linalg.norm(np.diff(M, axis=0), axis=1).sum() + 1e-9
    return (rel @ left) / bl, (rel @ up) / bl, bl, np.stack([-tan, left, up])

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--max_kbl", type=float, default=4.0); a = ap.parse_args()
sp, num = a.tag.rsplit("_fish", 1); man = REPO / f"data/species_manifold/{sp}_fix/top"
c = np.load(man / f"fish{num}_curvature.npz"); r = np.load(man / f"fish{num}_reference.npz", allow_pickle=True)
mid = c["midline_px"].astype(np.float64); kap = c["kappa_bl"]; fps = float(c["fps"]); dt = 1.0 / fps
valid = c["valid"].astype(bool) & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < a.max_kbl) & np.isfinite(mid).all((1, 2))
p_bl = r["p_bl"].astype(np.float64); bl_px = float(np.nanmedian(c["bodylen_px"][c["valid"].astype(bool)]))
idx = np.where(valid)[0]; runs = [rr for rr in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(rr) >= 3]
feats, seg, frames = [], [], []
for si, rr in enumerate(runs):
    # midline -> K points (arc-length resample), image y flipped, z = 0
    M = []
    for t in rr:
        P = mid[t][::-1].copy(); P[:, 1] *= -1.0    # midline_px is stored tail->head; bend3d wants head first
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]); s /= s[-1] + 1e-9
        q = np.linspace(0, 1, K); M.append(np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1]), np.zeros(K)], 1))
    M = np.stack(M)                                                             # (T,K,3) px
    out = [bend3d(m) for m in M]; lat = np.stack([o[0] for o in out]); ver = np.stack([o[1] for o in out]); basis = np.stack([o[3] for o in out])
    pos = np.stack([p_bl[rr, 0], -p_bl[rr, 1], np.zeros(len(rr))], 1)           # BL units, y flipped
    vel = np.gradient(pos, dt, axis=0) if len(rr) > 1 else np.zeros_like(pos)
    speed = np.clip(np.linalg.norm(vel, axis=1), 0, 10)
    vb = np.einsum("tij,tj->ti", basis, vel); n = np.linalg.norm(vb, axis=1, keepdims=True)
    vdir = np.where(n > TRAVEL_MIN_BL, vb / np.maximum(n, 1e-9), 0.0).clip(-1, 1)
    nose = basis[:, 0, :]; psi = np.unwrap(np.arctan2(nose[:, 1], nose[:, 0]))
    yaw = np.clip(np.gradient(smooth(psi, 5)) / dt, -10, 10) if len(rr) >= 2 else np.zeros(len(rr))
    f = np.concatenate([lat, ver, speed[:, None], vdir, yaw[:, None], np.zeros((len(rr), 1))], 1).astype(np.float32)
    feats.append(f); seg.append(np.full(len(rr), si)); frames.append(rr)
feat = np.concatenate(feats); out = REPO / f"data/fish_assets/{a.tag}/amp_reference.npz"
np.savez(out, feat=feat, segment_id=np.concatenate(seg), frame=np.concatenate(frames), K=K, n_bend_per_station=2, n_vel=6, vel_names=np.array(NAMES),
         bl_px=bl_px, fps=fps, source=str(man), note="single top view: vertical bend and pitch are zero; y flipped to right-handed")
print(f"[amp_ref:{a.tag}] {feat.shape[0]} frames in {len(runs)} runs -> {out}; speed_bl mean={feat[:,40].mean():.2f} dir_nose mean={feat[:,41].mean():.2f} |lat| p95={np.percentile(np.abs(feat[:,:20]),95):.3f}")
